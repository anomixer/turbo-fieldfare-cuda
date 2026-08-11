#include "tf/repack/RepackPlan.h"

#include <algorithm>
#include <format>
#include <map>
#include <ranges>

namespace tf::repack {
namespace {

/// Splits a copy that exceeds `maxOpBytes` into consecutive bounded pieces, so
/// the executor never needs a scratch buffer larger than one operation.
void emitBounded(std::vector<CopyOp>& ops, u32 shardIndex, ByteRange source,
                 DestKind destKind, u32 destLayer, u64 destOffset, u64 maxOpBytes) {
    u64 copied = 0;
    while (copied < source.length) {
        const u64 chunk = std::min(maxOpBytes, source.length - copied);
        ops.push_back(CopyOp{.shardIndex = shardIndex,
                             .source = ByteRange{.offset = source.offset + copied,
                                                 .length = chunk},
                             .destKind = destKind,
                             .destLayer = destLayer,
                             .destOffset = destOffset + copied});
        copied += chunk;
    }
}

/// Locates a tensor across the parsed shard headers.
struct SourceLocation {
    u32 shardIndex = 0;
    const TensorEntry* entry = nullptr;
    /// Absolute offset of the tensor's data within the shard file.
    u64 fileOffset = 0;
};

class SourceLookup {
public:
    SourceLookup(const SafetensorsIndex& index, const std::vector<SafetensorsHeader>& headers)
        : index_(index), headers_(headers) {
        const auto& shardFiles = index.shardFiles();
        for (u32 i = 0; i < shardFiles.size(); ++i) {
            shardIndexByName_.emplace(shardFiles[i], i);
        }
    }

    [[nodiscard]] Result<SourceLocation> find(std::string_view name) const {
        TF_TRY(const std::string_view shard, index_.requireShardFor(name));

        const auto it = shardIndexByName_.find(std::string{shard});
        if (it == shardIndexByName_.end()) {
            return makeError(ErrorCode::MalformedData,
                             "tensor '{}' names shard '{}', which is not in shardFiles()", name,
                             shard);
        }
        const u32 shardIndex = it->second;
        if (shardIndex >= headers_.size()) {
            return makeError(ErrorCode::InvalidArgument,
                             "no parsed header supplied for shard '{}'", shard);
        }

        const SafetensorsHeader& header = headers_[shardIndex];
        TF_TRY(const TensorEntry* entry, header.require(name));

        return SourceLocation{.shardIndex = shardIndex,
                              .entry = entry,
                              .fileOffset = header.dataOffset() + entry->dataRange.offset};
    }

private:
    const SafetensorsIndex& index_;
    const std::vector<SafetensorsHeader>& headers_;
    std::map<std::string, u32, std::less<>> shardIndexByName_;
};

/// Strips the multimodal wrapper prefix so stored names are independent of it.
std::string canonicalName(std::string_view sourceName) {
    if (sourceName.starts_with(kLanguageModelPrefix)) {
        return std::string{sourceName.substr(kLanguageModelPrefix.size())};
    }
    return std::string{sourceName};
}

bool isVisionTensor(std::string_view name) {
    return name.starts_with("vision_tower.") || name.starts_with("embed_vision.");
}

bool isExpertTensor(std::string_view name) {
    return name.find(kExpertMarker) != std::string_view::npos;
}

/// "gate.weight" -> the source tensor suffix "gate_proj.weight".
std::string sourceSuffixForRole(std::string_view role) {
    const usize dot = role.find('.');
    const std::string_view projection = role.substr(0, dot);
    const std::string_view component = role.substr(dot + 1);
    return std::format("{}_proj.{}", projection, component);
}

}  // namespace

const std::vector<std::string>& expertComponentRoles() {
    // Order defines the on-disk blob layout. Grouping each projection's weight
    // with its scales and biases keeps a single projection contiguous, which is
    // how the dequant kernels consume it.
    static const std::vector<std::string> kRoles = {
            "gate.weight", "gate.scales", "gate.biases",
            "up.weight",   "up.scales",   "up.biases",
            "down.weight", "down.scales", "down.biases",
    };
    return kRoles;
}

u64 RepackPlan::totalCopiedBytes() const {
    u64 total = 0;
    for (const auto& op : ops) {
        total += op.source.length;
    }
    return total;
}

Status RepackPlan::validate() const {
    TF_CHECK(arch.validate());
    TF_CHECK(resident.validate());
    TF_CHECK(experts.validate());

    // Every output byte must be either written by an operation or be stride
    // padding. Checking coverage per destination catches a planning bug here
    // rather than as a garbled projection at inference time.
    std::vector<std::vector<ByteRange>> byDest(1 + experts.numLayers);
    for (const auto& op : ops) {
        if (op.source.length == 0) {
            return makeError(ErrorCode::MalformedData, "plan contains a zero-length copy");
        }
        const usize slot =
                op.destKind == DestKind::Resident ? 0 : static_cast<usize>(op.destLayer) + 1;
        if (slot >= byDest.size()) {
            return makeError(ErrorCode::MalformedData, "copy targets layer {} which is out of "
                                                       "range",
                             op.destLayer);
        }
        byDest[slot].push_back(ByteRange{.offset = op.destOffset, .length = op.source.length});
    }

    // Gaps up to `maxGap` are alignment padding, which preallocation zero-fills
    // rather than any copy writing. Anything larger means a tensor was dropped.
    const auto checkContiguous = [](std::vector<ByteRange>& ranges, u64 expectedBytes,
                                    u64 maxGap, std::string_view what) -> Status {
        std::ranges::sort(ranges, {}, &ByteRange::offset);
        u64 cursor = 0;
        for (const auto& range : ranges) {
            if (range.offset < cursor) {
                return makeError(ErrorCode::MalformedData,
                                 "{}: copy at {} overlaps the previous one ending at {}", what,
                                 range.offset, cursor);
            }
            if (range.offset - cursor > maxGap) {
                return makeError(ErrorCode::MalformedData,
                                 "{}: {} bytes are unwritten before offset {}", what,
                                 range.offset - cursor, range.offset);
            }
            cursor = range.end();
        }
        if (cursor != expectedBytes) {
            return makeError(ErrorCode::MalformedData,
                             "{}: copies cover {} bytes but {} are expected", what, cursor,
                             expectedBytes);
        }
        return {};
    };

    TF_CHECK(checkContiguous(byDest[0], resident.totalBytes(),
                             gturbo::kResidentTensorAlignment - 1, "resident file"));

    for (u64 layer = 0; layer < experts.numLayers; ++layer) {
        // The final expert's trailing pad is never written, so coverage stops
        // at the last blob's end rather than at the file size.
        const u64 covered =
                experts.stride * (experts.expertsPerLayer - 1) + experts.blobBytes;
        TF_CHECK(checkContiguous(byDest[layer + 1], covered, experts.stride - experts.blobBytes,
                                 std::format("expert layer {}", layer)));
    }

    return {};
}

Result<RepackPlan> buildPlan(const ArchInfo& arch, const SafetensorsIndex& index,
                             const std::vector<SafetensorsHeader>& headers,
                             const PlanOptions& options) {
    TF_CHECK(arch.validate());

    if (headers.size() != index.shardFiles().size()) {
        return makeError(ErrorCode::InvalidArgument,
                         "{} shard headers supplied for {} shards", headers.size(),
                         index.shardFiles().size());
    }
    if (options.maxOpBytes == 0) {
        return makeError(ErrorCode::InvalidArgument, "maxOpBytes must be non-zero");
    }

    const SourceLookup lookup{index, headers};

    RepackPlan plan;
    plan.arch = arch;
    plan.shards = index.shardFiles();

    // ---- Resident tensors ------------------------------------------------
    //
    // Walked in index order rather than sorted, so the layout mirrors the
    // source file order. Combined with the operation sort below, that makes
    // both reads and writes run broadly front to back.
    u64 residentCursor = 0;
    for (const auto& [sourceName, shardName] : index.weightMap()) {
        if (isVisionTensor(sourceName) || isExpertTensor(sourceName)) {
            continue;
        }
        if (!sourceName.starts_with(kLanguageModelPrefix)) {
            return makeError(ErrorCode::Unsupported,
                             "unexpected top-level tensor '{}' in the checkpoint", sourceName);
        }

        TF_TRY(const SourceLocation location, lookup.find(sourceName));

        // Align every tensor. Packed weights are read as u32 on the GPU, and a
        // preceding 2-byte tensor would otherwise leave them at 2 mod 4.
        residentCursor = alignUp(residentCursor, gturbo::kResidentTensorAlignment);

        gturbo::ResidentTensor tensor;
        tensor.name = canonicalName(sourceName);
        tensor.dtype = location.entry->dtype;
        tensor.shape = location.entry->shape;
        tensor.range = ByteRange{.offset = residentCursor,
                                 .length = location.entry->dataRange.length};

        emitBounded(plan.ops, location.shardIndex,
                    ByteRange{.offset = location.fileOffset,
                              .length = location.entry->dataRange.length},
                    DestKind::Resident, 0, residentCursor, options.maxOpBytes);

        residentCursor += tensor.range.length;
        plan.resident.add(std::move(tensor));
    }

    if (plan.resident.tensors().empty()) {
        return makeError(ErrorCode::MalformedData,
                         "no resident tensors found; is this a Gemma 4 checkpoint?");
    }

    // ---- Expert layout ---------------------------------------------------
    //
    // Derived from the architecture rather than read from the source, then
    // cross-checked against the actual tensor shapes below.
    const QuantizedLinearLayout gateUp = arch.expertGateUpLayout();
    const QuantizedLinearLayout down = arch.expertDownLayout();

    plan.experts.numLayers = arch.numLayers;
    plan.experts.expertsPerLayer = arch.numExperts;
    plan.experts.blobBytes = arch.expertBlobBytes();
    plan.experts.alignment = options.expertAlignment;
    plan.experts.stride = alignUp(plan.experts.blobBytes, options.expertAlignment);
    for (u64 layer = 0; layer < arch.numLayers; ++layer) {
        plan.experts.layerFiles.push_back(gturbo::ExpertLayout::layerFileName(layer));
    }

    u64 componentOffset = 0;
    for (const auto& role : expertComponentRoles()) {
        const bool isDown = role.starts_with("down.");
        const QuantizedLinearLayout& layout = isDown ? down : gateUp;

        u64 size = 0;
        std::vector<u64> shape;
        DType dtype = DType::U32;
        if (role.ends_with(".weight")) {
            size = layout.weightBytes();
            shape = {layout.outFeatures, layout.packedWordsPerRow()};
            dtype = DType::U32;
        } else {
            size = layout.scaleBytes();
            shape = {layout.outFeatures, layout.groupsPerRow()};
            dtype = DType::BF16;
        }

        plan.experts.components.push_back(gturbo::ExpertComponent{.role = role,
                                                                  .dtype = dtype,
                                                                  .shape = std::move(shape),
                                                                  .offset = componentOffset,
                                                                  .size = size});
        componentOffset += size;
    }

    if (componentOffset != plan.experts.blobBytes) {
        return makeError(ErrorCode::MalformedData,
                         "expert components total {} bytes but the architecture implies {}",
                         componentOffset, plan.experts.blobBytes);
    }

    // ---- Expert slices ---------------------------------------------------
    //
    // Experts are stored stacked: one source tensor per layer holds all 128.
    // Slicing expert `e` is therefore a contiguous sub-range at e * sliceBytes.
    for (u64 layer = 0; layer < arch.numLayers; ++layer) {
        for (const auto& component : plan.experts.components) {
            const std::string sourceName =
                    std::format("{}layers.{}{}{}", kLanguageModelPrefix, layer, kExpertMarker,
                                sourceSuffixForRole(component.role));

            TF_TRY(const SourceLocation location, lookup.find(sourceName));

            // Shape must be [numExperts, ...], and the remainder must match the
            // per-expert component exactly.
            const std::vector<u64>& shape = location.entry->shape;
            if (shape.empty() || shape[0] != arch.numExperts) {
                return makeError(ErrorCode::MalformedData,
                                 "{}: expected a leading dimension of {} experts, found shape {}",
                                 sourceName, arch.numExperts, location.entry->shapeString());
            }

            const u64 sliceBytes = location.entry->dataRange.length / arch.numExperts;
            if (sliceBytes != component.size) {
                return makeError(ErrorCode::MalformedData,
                                 "{}: per-expert slice is {} bytes but the layout expects {}",
                                 sourceName, sliceBytes, component.size);
            }

            for (u64 expert = 0; expert < arch.numExperts; ++expert) {
                emitBounded(plan.ops, location.shardIndex,
                            ByteRange{.offset = location.fileOffset + expert * sliceBytes,
                                      .length = sliceBytes},
                            DestKind::ExpertLayer, static_cast<u32>(layer),
                            expert * plan.experts.stride + component.offset,
                            options.maxOpBytes);
            }
        }
    }

    // Sequential reads per shard: the dominant cost is source I/O, and the
    // destinations are opened for the whole run anyway.
    std::ranges::sort(plan.ops, [](const CopyOp& a, const CopyOp& b) {
        if (a.shardIndex != b.shardIndex) {
            return a.shardIndex < b.shardIndex;
        }
        return a.source.offset < b.source.offset;
    });

    TF_CHECK(plan.validate());
    return plan;
}

}  // namespace tf::repack
