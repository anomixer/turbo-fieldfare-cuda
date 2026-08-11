#include "tf/core/format/GTurbo.h"

#include <algorithm>
#include <format>
#include <ranges>

#include "tf/core/io/File.h"
#include "tf/core/io/Sha256.h"
#include "tf/core/json/Json.h"

namespace tf::gturbo {
namespace {

json::Value encodeShape(const std::vector<u64>& shape) {
    json::Value array = json::Value::makeArray();
    for (const u64 dim : shape) {
        array.push(json::Value{dim});
    }
    return array;
}

Result<std::vector<u64>> decodeShape(const json::Value& node) {
    TF_TRY(const json::Array* array, node.asArray());
    std::vector<u64> shape;
    shape.reserve(array->size());
    for (const auto& dim : *array) {
        TF_TRY(const u64 value, dim.asUInt());
        shape.push_back(value);
    }
    return shape;
}

Result<u64> requireUInt(const json::Value& node, std::string_view key) {
    TF_TRY(const json::Value* field, node.at(key));
    auto value = field->asUInt();
    if (!value) {
        return std::unexpected(value.error().wrap(key));
    }
    return *value;
}

Result<std::string> requireString(const json::Value& node, std::string_view key) {
    TF_TRY(const json::Value* field, node.at(key));
    auto value = field->asString();
    if (!value) {
        return std::unexpected(value.error().wrap(key));
    }
    return std::string{*value};
}

}  // namespace

// ---------------------------------------------------------------------------
// ResidentIndex
// ---------------------------------------------------------------------------

std::string ResidentTensor::shapeString() const {
    std::string out = "[";
    for (usize i = 0; i < shape.size(); ++i) {
        if (i > 0) {
            out += ',';
        }
        out += std::to_string(shape[i]);
    }
    out += ']';
    return out;
}

void ResidentIndex::add(ResidentTensor tensor) {
    totalBytes_ = std::max(totalBytes_, tensor.range.end());
    tensors_.push_back(std::move(tensor));
}

const ResidentTensor* ResidentIndex::find(std::string_view name) const {
    for (const auto& tensor : tensors_) {
        if (tensor.name == name) {
            return &tensor;
        }
    }
    return nullptr;
}

Result<const ResidentTensor*> ResidentIndex::require(std::string_view name) const {
    if (const ResidentTensor* tensor = find(name)) {
        return tensor;
    }
    return makeError(ErrorCode::NotFound, "resident tensor '{}' is not in the index", name);
}

Status ResidentIndex::validate() const {
    if (tensors_.empty()) {
        return makeError(ErrorCode::MalformedData, "resident index is empty");
    }

    // Sort by offset to check for overlap in one pass rather than pairwise.
    std::vector<const ResidentTensor*> ordered;
    ordered.reserve(tensors_.size());
    for (const auto& tensor : tensors_) {
        ordered.push_back(&tensor);
    }
    std::ranges::sort(ordered, {}, [](const ResidentTensor* t) { return t->range.offset; });

    u64 cursor = 0;
    for (const ResidentTensor* tensor : ordered) {
        if (tensor->range.length == 0) {
            return makeError(ErrorCode::MalformedData, "resident tensor '{}' is empty",
                             tensor->name);
        }
        if (tensor->range.offset < cursor) {
            return makeError(ErrorCode::MalformedData,
                             "resident tensor '{}' at offset {} overlaps the preceding tensor "
                             "ending at {}",
                             tensor->name, tensor->range.offset, cursor);
        }
        // Packed weights are dereferenced as u32 on the GPU, so a misaligned
        // offset is undefined behaviour on the host and a hard fault on the
        // device. Catching it here means a bad install fails at load with a
        // named tensor rather than as an opaque CUDA error mid-decode.
        const u64 required = byteWidth(tensor->dtype);
        if (!isAligned(tensor->range.offset, required)) {
            return makeError(ErrorCode::MalformedData,
                             "resident tensor '{}' of {} starts at offset {}, which is not "
                             "aligned to its {}-byte element size",
                             tensor->name, toString(tensor->dtype), tensor->range.offset,
                             required);
        }
        if (tensor->range.length != tensor->expectedBytesForShape()) {
            return makeError(ErrorCode::MalformedData,
                             "resident tensor '{}' spans {} bytes but its shape {} of {} "
                             "implies {}",
                             tensor->name, tensor->range.length, tensor->shapeString(),
                             toString(tensor->dtype), tensor->expectedBytesForShape());
        }
        cursor = tensor->range.end();
    }

    if (cursor != totalBytes_) {
        return makeError(ErrorCode::MalformedData,
                         "resident index reports {} bytes but its ranges end at {}",
                         totalBytes_, cursor);
    }
    return {};
}

// ---------------------------------------------------------------------------
// ExpertLayout
// ---------------------------------------------------------------------------

std::string ExpertLayout::layerFileName(u64 layer) {
    return std::format("layer_{:02}.bin", layer);
}

Result<ByteRange> ExpertLayout::expertRange(u64 layer, u64 expert) const {
    if (layer >= numLayers) {
        return makeError(ErrorCode::InvalidArgument, "layer {} is out of range (have {})",
                         layer, numLayers);
    }
    if (expert >= expertsPerLayer) {
        return makeError(ErrorCode::InvalidArgument, "expert {} is out of range (have {})",
                         expert, expertsPerLayer);
    }
    // Every blob is identical in size, so position is pure arithmetic.
    return ByteRange{.offset = expert * stride, .length = blobBytes};
}

Result<ByteRange> ExpertLayout::componentRange(u64 layer, u64 expert,
                                               std::string_view role) const {
    TF_TRY(const ByteRange blob, expertRange(layer, expert));
    for (const auto& component : components) {
        if (component.role == role) {
            return ByteRange{.offset = blob.offset + component.offset, .length = component.size};
        }
    }
    return makeError(ErrorCode::NotFound, "expert component '{}' is not in the layout", role);
}

bool ExpertLayout::supportsUnbufferedReads(u64 sectorSize) const {
    if (sectorSize == 0) {
        return false;
    }
    // Unbuffered reads need every blob to start on a sector boundary. The blob
    // length itself may be unaligned, since a read can round up into the pad.
    return isAligned(stride, sectorSize);
}

Status ExpertLayout::validate() const {
    if (numLayers == 0 || expertsPerLayer == 0) {
        return makeError(ErrorCode::MalformedData, "expert layout has a zero dimension");
    }
    if (layerFiles.size() != numLayers) {
        return makeError(ErrorCode::MalformedData,
                         "expert layout names {} files but declares {} layers",
                         layerFiles.size(), numLayers);
    }
    if (blobBytes == 0) {
        return makeError(ErrorCode::MalformedData, "expert blob size is zero");
    }
    if (stride < blobBytes) {
        return makeError(ErrorCode::MalformedData,
                         "expert stride {} is smaller than the {} byte blob", stride,
                         blobBytes);
    }
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
        return makeError(ErrorCode::MalformedData, "alignment {} is not a power of two",
                         alignment);
    }
    if (!isAligned(stride, alignment)) {
        return makeError(ErrorCode::MalformedData,
                         "expert stride {} is not aligned to {}", stride, alignment);
    }
    if (components.empty()) {
        return makeError(ErrorCode::MalformedData, "expert layout has no components");
    }

    // Components must tile the blob exactly: a gap would waste streamed bytes,
    // and an overlap would corrupt a projection.
    std::vector<const ExpertComponent*> ordered;
    ordered.reserve(components.size());
    for (const auto& component : components) {
        ordered.push_back(&component);
    }
    std::ranges::sort(ordered, {}, [](const ExpertComponent* c) { return c->offset; });

    u64 cursor = 0;
    for (const ExpertComponent* component : ordered) {
        if (component->offset != cursor) {
            return makeError(ErrorCode::MalformedData,
                             "expert component '{}' starts at {} but the previous one ends at "
                             "{}",
                             component->role, component->offset, cursor);
        }
        cursor += component->size;
    }
    if (cursor != blobBytes) {
        return makeError(ErrorCode::MalformedData,
                         "expert components total {} bytes but the blob is {}", cursor,
                         blobBytes);
    }

    return {};
}

std::string ExpertLayout::toJson() const {
    json::Value root = json::Value::makeObject();
    root.set("version", kFormatVersion);
    root.set("numLayers", numLayers);
    root.set("expertsPerLayer", expertsPerLayer);
    root.set("blobBytes", blobBytes);
    root.set("stride", stride);
    root.set("alignment", alignment);

    json::Value files = json::Value::makeArray();
    for (const auto& name : layerFiles) {
        files.push(json::Value{name});
    }
    root.set("layerFiles", std::move(files));

    json::Value components_ = json::Value::makeArray();
    for (const auto& component : components) {
        json::Value entry = json::Value::makeObject();
        entry.set("role", component.role);
        entry.set("dtype", std::string{toString(component.dtype)});
        entry.set("shape", encodeShape(component.shape));
        entry.set("offset", component.offset);
        entry.set("size", component.size);
        components_.push(std::move(entry));
    }
    root.set("components", std::move(components_));

    return root.dump(2);
}

Result<ExpertLayout> ExpertLayout::fromJson(std::string_view text) {
    TF_TRY(const json::Value root, json::parse(text));

    TF_TRY(const u64 version, requireUInt(root, "version"));
    if (version != kFormatVersion) {
        return makeError(ErrorCode::Unsupported,
                         "expert layout version {} is not supported (this build reads {})",
                         version, kFormatVersion);
    }

    ExpertLayout layout;
    TF_TRY(layout.numLayers, requireUInt(root, "numLayers"));
    TF_TRY(layout.expertsPerLayer, requireUInt(root, "expertsPerLayer"));
    TF_TRY(layout.blobBytes, requireUInt(root, "blobBytes"));
    TF_TRY(layout.stride, requireUInt(root, "stride"));
    TF_TRY(layout.alignment, requireUInt(root, "alignment"));

    TF_TRY(const json::Value* filesNode, root.at("layerFiles"));
    TF_TRY(const json::Array* files, filesNode->asArray());
    layout.layerFiles.reserve(files->size());
    for (const auto& name : *files) {
        TF_TRY(const std::string_view fileName, name.asString());
        layout.layerFiles.emplace_back(fileName);
    }

    TF_TRY(const json::Value* componentsNode, root.at("components"));
    TF_TRY(const json::Array* componentArray, componentsNode->asArray());
    layout.components.reserve(componentArray->size());
    for (const auto& entry : *componentArray) {
        ExpertComponent component;
        TF_TRY(component.role, requireString(entry, "role"));
        TF_TRY(const std::string dtypeName, requireString(entry, "dtype"));
        TF_TRY(component.dtype, parseDType(dtypeName));
        TF_TRY(const json::Value* shapeNode, entry.at("shape"));
        TF_TRY(component.shape, decodeShape(*shapeNode));
        TF_TRY(component.offset, requireUInt(entry, "offset"));
        TF_TRY(component.size, requireUInt(entry, "size"));
        layout.components.push_back(std::move(component));
    }

    TF_CHECK(layout.validate());
    return layout;
}

// ---------------------------------------------------------------------------
// Manifest
// ---------------------------------------------------------------------------
namespace {

json::Value encodeArch(const ArchInfo& arch) {
    json::Value node = json::Value::makeObject();
    node.set("hiddenSize", arch.hiddenSize);
    node.set("numLayers", arch.numLayers);
    node.set("vocabSize", arch.vocabSize);
    node.set("maxPositionEmbeddings", arch.maxPositionEmbeddings);
    node.set("numHeads", arch.numHeads);
    node.set("numKVHeads", arch.numKVHeads);
    node.set("numGlobalKVHeads", arch.numGlobalKVHeads);
    node.set("headDim", arch.headDim);
    node.set("globalHeadDim", arch.globalHeadDim);
    node.set("slidingWindow", arch.slidingWindow);
    node.set("attentionKEqV", arch.attentionKEqV);
    node.set("intermediateSize", arch.intermediateSize);
    node.set("moeIntermediateSize", arch.moeIntermediateSize);
    node.set("numExperts", arch.numExperts);
    node.set("topKExperts", arch.topKExperts);
    node.set("hiddenActivation", arch.hiddenActivation);
    node.set("rmsNormEps", arch.rmsNormEps);
    node.set("finalLogitSoftcap", arch.finalLogitSoftcap);
    node.set("tieWordEmbeddings", arch.tieWordEmbeddings);
    node.set("slidingRopeTheta", arch.slidingRopeTheta);
    node.set("fullRopeTheta", arch.fullRopeTheta);
    node.set("partialRotaryFactor", arch.partialRotaryFactor);
    node.set("bosTokenId", static_cast<u64>(arch.bosTokenId));
    node.set("padTokenId", static_cast<u64>(arch.padTokenId));

    json::Value eos = json::Value::makeArray();
    for (const u32 id : arch.eosTokenIds) {
        eos.push(json::Value{static_cast<u64>(id)});
    }
    node.set("eosTokenIds", std::move(eos));

    json::Value weightQuant = json::Value::makeObject();
    weightQuant.set("bits", static_cast<u64>(arch.weightQuant.bits));
    weightQuant.set("groupSize", static_cast<u64>(arch.weightQuant.groupSize));
    node.set("weightQuant", std::move(weightQuant));

    json::Value routerQuant = json::Value::makeObject();
    routerQuant.set("bits", static_cast<u64>(arch.routerQuant.bits));
    routerQuant.set("groupSize", static_cast<u64>(arch.routerQuant.groupSize));
    node.set("routerQuant", std::move(routerQuant));

    json::Value types = json::Value::makeArray();
    for (const AttentionKind kind : arch.layerTypes) {
        types.push(json::Value{std::string{toString(kind)}});
    }
    node.set("layerTypes", std::move(types));

    return node;
}

Result<ArchInfo> decodeArch(const json::Value& node) {
    ArchInfo arch;
    TF_TRY(arch.hiddenSize, requireUInt(node, "hiddenSize"));
    TF_TRY(arch.numLayers, requireUInt(node, "numLayers"));
    TF_TRY(arch.vocabSize, requireUInt(node, "vocabSize"));
    TF_TRY(arch.maxPositionEmbeddings, requireUInt(node, "maxPositionEmbeddings"));
    TF_TRY(arch.numHeads, requireUInt(node, "numHeads"));
    TF_TRY(arch.numKVHeads, requireUInt(node, "numKVHeads"));
    TF_TRY(arch.numGlobalKVHeads, requireUInt(node, "numGlobalKVHeads"));
    TF_TRY(arch.headDim, requireUInt(node, "headDim"));
    TF_TRY(arch.globalHeadDim, requireUInt(node, "globalHeadDim"));
    TF_TRY(arch.slidingWindow, requireUInt(node, "slidingWindow"));
    TF_TRY(arch.intermediateSize, requireUInt(node, "intermediateSize"));
    TF_TRY(arch.moeIntermediateSize, requireUInt(node, "moeIntermediateSize"));
    TF_TRY(arch.numExperts, requireUInt(node, "numExperts"));
    TF_TRY(arch.topKExperts, requireUInt(node, "topKExperts"));
    TF_TRY(arch.hiddenActivation, requireString(node, "hiddenActivation"));

    TF_TRY(const json::Value* kEqV, node.at("attentionKEqV"));
    TF_TRY(arch.attentionKEqV, kEqV->asBool());
    TF_TRY(const json::Value* tied, node.at("tieWordEmbeddings"));
    TF_TRY(arch.tieWordEmbeddings, tied->asBool());

    TF_TRY(const json::Value* eps, node.at("rmsNormEps"));
    TF_TRY(arch.rmsNormEps, eps->asDouble());
    TF_TRY(const json::Value* softcap, node.at("finalLogitSoftcap"));
    TF_TRY(arch.finalLogitSoftcap, softcap->asDouble());
    TF_TRY(const json::Value* slidingTheta, node.at("slidingRopeTheta"));
    TF_TRY(arch.slidingRopeTheta, slidingTheta->asDouble());
    TF_TRY(const json::Value* fullTheta, node.at("fullRopeTheta"));
    TF_TRY(arch.fullRopeTheta, fullTheta->asDouble());
    TF_TRY(const json::Value* rotary, node.at("partialRotaryFactor"));
    TF_TRY(arch.partialRotaryFactor, rotary->asDouble());

    TF_TRY(const u64 bos, requireUInt(node, "bosTokenId"));
    arch.bosTokenId = static_cast<u32>(bos);
    TF_TRY(const u64 pad, requireUInt(node, "padTokenId"));
    arch.padTokenId = static_cast<u32>(pad);

    TF_TRY(const json::Value* eosNode, node.at("eosTokenIds"));
    TF_TRY(const json::Array* eos, eosNode->asArray());
    for (const auto& id : *eos) {
        TF_TRY(const u64 value, id.asUInt());
        arch.eosTokenIds.push_back(static_cast<u32>(value));
    }

    TF_TRY(const json::Value* weightQuant, node.at("weightQuant"));
    TF_TRY(const u64 weightBits, requireUInt(*weightQuant, "bits"));
    TF_TRY(const u64 weightGroup, requireUInt(*weightQuant, "groupSize"));
    arch.weightQuant = QuantSpec{.bits = static_cast<u32>(weightBits),
                                 .groupSize = static_cast<u32>(weightGroup)};

    TF_TRY(const json::Value* routerQuant, node.at("routerQuant"));
    TF_TRY(const u64 routerBits, requireUInt(*routerQuant, "bits"));
    TF_TRY(const u64 routerGroup, requireUInt(*routerQuant, "groupSize"));
    arch.routerQuant = QuantSpec{.bits = static_cast<u32>(routerBits),
                                 .groupSize = static_cast<u32>(routerGroup)};

    TF_TRY(const json::Value* typesNode, node.at("layerTypes"));
    TF_TRY(const json::Array* types, typesNode->asArray());
    arch.layerTypes.reserve(types->size());
    for (const auto& kind : *types) {
        TF_TRY(const std::string_view name, kind.asString());
        if (name == "full_attention") {
            arch.layerTypes.push_back(AttentionKind::Full);
        } else if (name == "sliding_attention") {
            arch.layerTypes.push_back(AttentionKind::Sliding);
        } else {
            return makeError(ErrorCode::Unsupported, "unknown layer type '{}'", name);
        }
    }

    TF_CHECK(arch.validate());
    return arch;
}

}  // namespace

std::string Manifest::toJson() const {
    json::Value root = json::Value::makeObject();
    root.set("formatVersion", static_cast<u64>(formatVersion));

    json::Value sourceNode = json::Value::makeObject();
    sourceNode.set("repoId", source.repoId);
    sourceNode.set("revision", source.revision);
    sourceNode.set("toolVersion", source.toolVersion);
    root.set("source", std::move(sourceNode));

    root.set("arch", encodeArch(arch));

    json::Value residentNode = json::Value::makeObject();
    residentNode.set("totalBytes", resident.totalBytes());
    json::Value tensors = json::Value::makeArray();
    for (const auto& tensor : resident.tensors()) {
        json::Value entry = json::Value::makeObject();
        entry.set("name", tensor.name);
        entry.set("dtype", std::string{toString(tensor.dtype)});
        entry.set("shape", encodeShape(tensor.shape));
        entry.set("offset", tensor.range.offset);
        entry.set("size", tensor.range.length);
        tensors.push(std::move(entry));
    }
    residentNode.set("tensors", std::move(tensors));
    root.set("resident", std::move(residentNode));

    json::Value filesNode = json::Value::makeArray();
    for (const auto& file : files) {
        json::Value entry = json::Value::makeObject();
        entry.set("path", file.path);
        entry.set("size", file.size);
        entry.set("sha256", file.sha256);
        filesNode.push(std::move(entry));
    }
    root.set("files", std::move(filesNode));

    return root.dump(2);
}

Result<Manifest> Manifest::fromJson(std::string_view text) {
    TF_TRY(const json::Value root, json::parse(text));

    Manifest manifest;
    TF_TRY(const u64 version, requireUInt(root, "formatVersion"));
    if (version != kFormatVersion) {
        return makeError(ErrorCode::Unsupported,
                         "manifest format version {} is not supported (this build reads {})",
                         version, kFormatVersion);
    }
    manifest.formatVersion = static_cast<u32>(version);

    TF_TRY(const json::Value* sourceNode, root.at("source"));
    TF_TRY(manifest.source.repoId, requireString(*sourceNode, "repoId"));
    TF_TRY(manifest.source.revision, requireString(*sourceNode, "revision"));
    TF_TRY(manifest.source.toolVersion, requireString(*sourceNode, "toolVersion"));

    TF_TRY(const json::Value* archNode, root.at("arch"));
    TF_TRY(manifest.arch, decodeArch(*archNode));

    TF_TRY(const json::Value* residentNode, root.at("resident"));
    TF_TRY(const json::Value* tensorsNode, residentNode->at("tensors"));
    TF_TRY(const json::Array* tensors, tensorsNode->asArray());
    for (const auto& entry : *tensors) {
        ResidentTensor tensor;
        TF_TRY(tensor.name, requireString(entry, "name"));
        TF_TRY(const std::string dtypeName, requireString(entry, "dtype"));
        TF_TRY(tensor.dtype, parseDType(dtypeName));
        TF_TRY(const json::Value* shapeNode, entry.at("shape"));
        TF_TRY(tensor.shape, decodeShape(*shapeNode));
        TF_TRY(tensor.range.offset, requireUInt(entry, "offset"));
        TF_TRY(tensor.range.length, requireUInt(entry, "size"));
        manifest.resident.add(std::move(tensor));
    }

    TF_TRY(const json::Value* filesNode, root.at("files"));
    TF_TRY(const json::Array* filesArray, filesNode->asArray());
    for (const auto& entry : *filesArray) {
        FileDigest digest;
        TF_TRY(digest.path, requireString(entry, "path"));
        TF_TRY(digest.size, requireUInt(entry, "size"));
        TF_TRY(digest.sha256, requireString(entry, "sha256"));
        manifest.files.push_back(std::move(digest));
    }

    return manifest;
}

Status Manifest::validate() const {
    if (formatVersion != kFormatVersion) {
        return makeError(ErrorCode::Unsupported, "unsupported manifest version {}",
                         formatVersion);
    }
    TF_CHECK(arch.validate());
    TF_CHECK(resident.validate());
    TF_CHECK(experts.validate());

    if (experts.numLayers != arch.numLayers) {
        return makeError(ErrorCode::MalformedData,
                         "expert layout covers {} layers but the architecture has {}",
                         experts.numLayers, arch.numLayers);
    }
    if (experts.expertsPerLayer != arch.numExperts) {
        return makeError(ErrorCode::MalformedData,
                         "expert layout has {} experts per layer but the architecture has {}",
                         experts.expertsPerLayer, arch.numExperts);
    }
    if (experts.blobBytes != arch.expertBlobBytes()) {
        return makeError(ErrorCode::MalformedData,
                         "expert blob is {} bytes but the architecture implies {}",
                         experts.blobBytes, arch.expertBlobBytes());
    }
    if (files.empty()) {
        return makeError(ErrorCode::MalformedData, "manifest lists no files");
    }
    return {};
}

// ---------------------------------------------------------------------------
// Install-level entry points
// ---------------------------------------------------------------------------

Result<Manifest> readManifest(const std::filesystem::path& installDir) {
    const std::filesystem::path manifestPath = installDir / kManifestFile;
    if (!std::filesystem::exists(manifestPath)) {
        // Absence specifically means a partial install, which is worth
        // distinguishing from a corrupt one so the caller can offer to resume.
        return makeError(ErrorCode::IncompleteInstall,
                         "{} has no {} - the install is incomplete", installDir.string(),
                         kManifestFile);
    }

    TF_TRY(const std::string text, io::readTextFile(manifestPath));
    auto manifest = Manifest::fromJson(text);
    if (!manifest) {
        return std::unexpected(manifest.error().wrap(manifestPath.string()));
    }

    // The expert layout lives in its own file so the loader can read it without
    // parsing the much larger resident tensor index.
    const std::filesystem::path layoutPath =
            installDir / kExpertsDir / kExpertLayoutFile;
    TF_TRY(const std::string layoutText, io::readTextFile(layoutPath));
    auto layout = ExpertLayout::fromJson(layoutText);
    if (!layout) {
        return std::unexpected(layout.error().wrap(layoutPath.string()));
    }
    manifest->experts = std::move(*layout);

    TF_CHECK(manifest->validate());
    return manifest;
}

Status verifyInstall(const std::filesystem::path& installDir, const Manifest& manifest) {
    for (const auto& entry : manifest.files) {
        const std::filesystem::path path = installDir / entry.path;

        std::error_code ec;
        if (!std::filesystem::exists(path, ec)) {
            return makeError(ErrorCode::VerificationFailed, "{} is missing", entry.path);
        }

        const auto actualSize = std::filesystem::file_size(path, ec);
        if (ec) {
            return makeError(ErrorCode::Io, "{}: {}", entry.path, ec.message());
        }
        if (actualSize != entry.size) {
            return makeError(ErrorCode::VerificationFailed,
                             "{} is {} bytes, expected {}", entry.path, actualSize, entry.size);
        }

        TF_TRY(const std::string digest, io::Sha256::hashFile(path));
        if (digest != entry.sha256) {
            return makeError(ErrorCode::VerificationFailed,
                             "{} hashes to {} but the manifest records {}", entry.path, digest,
                             entry.sha256);
        }
    }
    return {};
}

}  // namespace tf::gturbo
