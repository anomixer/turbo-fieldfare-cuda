// End-to-end repack against a synthetic checkpoint.
//
// The point of these tests is the expert slicing arithmetic. Routed experts are
// stored stacked - one source tensor per layer holding all of them - so the
// repacker's core job is cutting expert `e` out of the middle of a large tensor
// and placing it at the right offset in a packed file. Getting that wrong by one
// slice would still produce a plausible-looking 14 GB install that generates
// garbage, so here every output byte is checked against the source it came from.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <windows.h>

#include <filesystem>
#include <format>
#include <map>
#include <string>
#include <vector>

#include "tf/core/format/GTurbo.h"
#include "tf/core/io/File.h"
#include "tf/core/io/Sha256.h"
#include "tf/core/json/Json.h"
#include "tf/repack/Repacker.h"

using namespace tf;
using namespace tf::repack;

namespace {

class TempDir {
public:
    TempDir() {
        static int counter = 0;
        path_ = std::filesystem::temp_directory_path() /
                std::format("tf-repack-{}-{}", ::GetCurrentProcessId(), ++counter);
        std::filesystem::create_directories(path_);
    }
    ~TempDir() {
        std::error_code ec;
        std::filesystem::remove_all(path_, ec);
    }
    TempDir(const TempDir&) = delete;
    TempDir& operator=(const TempDir&) = delete;

    [[nodiscard]] const std::filesystem::path& path() const { return path_; }

private:
    std::filesystem::path path_;
};

// Deliberately tiny, but every dimension still satisfies the MLX affine rules:
// hidden must divide by both 8 (values per u32) and 64 (group size).
constexpr u64 kHidden = 128;
constexpr u64 kLayers = 2;
constexpr u64 kExperts = 4;
constexpr u64 kMoeIntermediate = 64;
constexpr u64 kIntermediate = 64;
constexpr u64 kVocab = 256;
constexpr u64 kHeads = 2;
constexpr u64 kKVHeads = 1;
constexpr u64 kHeadDim = 64;

std::string miniConfig() {
    return std::format(R"({{
      "eos_token_id": [1, 106, 50],
      "quantization": {{
        "group_size": 64, "bits": 4, "mode": "affine",
        "language_model.model.layers.0.router.proj": {{"group_size": 64, "bits": 8}}
      }},
      "text_config": {{
        "hidden_size": {}, "num_hidden_layers": {}, "vocab_size": {},
        "num_attention_heads": {}, "num_key_value_heads": {},
        "num_global_key_value_heads": {}, "head_dim": {}, "global_head_dim": {},
        "intermediate_size": {}, "moe_intermediate_size": {},
        "num_experts": {}, "top_k_experts": 2,
        "sliding_window": 64, "rms_norm_eps": 1e-06,
        "final_logit_softcapping": 30.0, "tie_word_embeddings": true,
        "attention_k_eq_v": true, "hidden_activation": "gelu_pytorch_tanh",
        "max_position_embeddings": 4096, "bos_token_id": 2, "pad_token_id": 0,
        "rope_parameters": {{
          "full_attention": {{"partial_rotary_factor": 0.25, "rope_theta": 1000000.0}},
          "sliding_attention": {{"rope_theta": 10000.0}}
        }},
        "layer_types": ["sliding_attention", "full_attention"]
      }}
    }})",
                       kHidden, kLayers, kVocab, kHeads, kKVHeads, kKVHeads, kHeadDim,
                       kHeadDim, kIntermediate, kMoeIntermediate, kExperts);
}

/// A tensor destined for the synthetic shard.
struct PlannedTensor {
    std::string name;
    DType dtype;
    std::vector<u64> shape;
};

u64 tensorBytes(const PlannedTensor& tensor) {
    u64 count = 1;
    for (const u64 dim : tensor.shape) {
        count *= dim;
    }
    return count * byteWidth(tensor.dtype);
}

/// Fills a tensor with bytes derived from its name and position, so any
/// mis-slicing shows up as a concrete mismatch rather than plausible noise.
u8 patternByte(std::string_view name, u64 offset) {
    u32 hash = 2166136261u;
    for (const char c : name) {
        hash = (hash ^ static_cast<u8>(c)) * 16777619u;
    }
    return static_cast<u8>((hash ^ (offset * 2654435761u)) >> 13);
}

std::vector<PlannedTensor> planTensors() {
    std::vector<PlannedTensor> tensors;

    const auto quantized = [&](const std::string& base, u64 outFeatures, u64 inFeatures,
                               u32 bits) {
        const QuantizedLinearLayout layout{
                .outFeatures = outFeatures,
                .inFeatures = inFeatures,
                .spec = QuantSpec{.bits = bits, .groupSize = 64}};
        tensors.push_back({base + ".weight", DType::U32,
                           {outFeatures, layout.packedWordsPerRow()}});
        tensors.push_back({base + ".scales", DType::BF16,
                           {outFeatures, layout.groupsPerRow()}});
        tensors.push_back({base + ".biases", DType::BF16,
                           {outFeatures, layout.groupsPerRow()}});
    };

    const std::string prefix = "language_model.model.";

    quantized(prefix + "embed_tokens", kVocab, kHidden, 4);
    tensors.push_back({prefix + "norm.weight", DType::BF16, {kHidden}});

    for (u64 layer = 0; layer < kLayers; ++layer) {
        const std::string base = std::format("{}layers.{}", prefix, layer);
        const bool isFull = (layer == 1);

        quantized(base + ".self_attn.q_proj", kHeads * kHeadDim, kHidden, 4);
        quantized(base + ".self_attn.k_proj", kKVHeads * kHeadDim, kHidden, 4);
        // attention_k_eq_v: the full-attention layer has no v_proj.
        if (!isFull) {
            quantized(base + ".self_attn.v_proj", kKVHeads * kHeadDim, kHidden, 4);
        }
        quantized(base + ".self_attn.o_proj", kHidden, kHeads * kHeadDim, 4);
        tensors.push_back({base + ".self_attn.q_norm.weight", DType::BF16, {kHeadDim}});
        tensors.push_back({base + ".self_attn.k_norm.weight", DType::BF16, {kHeadDim}});

        quantized(base + ".mlp.gate_proj", kIntermediate, kHidden, 4);
        quantized(base + ".mlp.up_proj", kIntermediate, kHidden, 4);
        quantized(base + ".mlp.down_proj", kHidden, kIntermediate, 4);

        quantized(base + ".router.proj", kExperts, kHidden, 8);
        tensors.push_back({base + ".router.scale", DType::BF16, {kHidden}});
        tensors.push_back({base + ".router.per_expert_scale", DType::BF16, {kExperts}});

        for (const auto* norm : {".input_layernorm", ".post_attention_layernorm",
                                 ".pre_feedforward_layernorm", ".pre_feedforward_layernorm_2",
                                 ".post_feedforward_layernorm",
                                 ".post_feedforward_layernorm_1",
                                 ".post_feedforward_layernorm_2"}) {
            tensors.push_back({base + norm + ".weight", DType::BF16, {kHidden}});
        }
        tensors.push_back({base + ".layer_scalar", DType::BF16, {1}});

        // Routed experts, stacked: leading dimension is the expert index.
        const QuantizedLinearLayout gateUp{
                .outFeatures = kMoeIntermediate, .inFeatures = kHidden, .spec = kWeightQuant};
        const QuantizedLinearLayout down{
                .outFeatures = kHidden, .inFeatures = kMoeIntermediate, .spec = kWeightQuant};

        for (const auto* projection : {"gate_proj", "up_proj"}) {
            const std::string expertBase =
                    std::format("{}.experts.switch_glu.{}", base, projection);
            tensors.push_back({expertBase + ".weight", DType::U32,
                               {kExperts, kMoeIntermediate, gateUp.packedWordsPerRow()}});
            tensors.push_back({expertBase + ".scales", DType::BF16,
                               {kExperts, kMoeIntermediate, gateUp.groupsPerRow()}});
            tensors.push_back({expertBase + ".biases", DType::BF16,
                               {kExperts, kMoeIntermediate, gateUp.groupsPerRow()}});
        }
        const std::string downBase = std::format("{}.experts.switch_glu.down_proj", base);
        tensors.push_back({downBase + ".weight", DType::U32,
                           {kExperts, kHidden, down.packedWordsPerRow()}});
        tensors.push_back({downBase + ".scales", DType::BF16,
                           {kExperts, kHidden, down.groupsPerRow()}});
        tensors.push_back({downBase + ".biases", DType::BF16,
                           {kExperts, kHidden, down.groupsPerRow()}});
    }

    // A vision tensor, to prove it is excluded from the install.
    tensors.push_back({"vision_tower.std_scale", DType::BF16, {16}});

    return tensors;
}

/// Writes a complete synthetic checkpoint and returns the pattern data for each
/// tensor so tests can compare against it.
std::map<std::string, std::vector<u8>> writeCheckpoint(const std::filesystem::path& dir) {
    std::filesystem::create_directories(dir);

    const auto tensors = planTensors();

    // Build the safetensors header, assigning offsets as we go.
    json::Value header = json::Value::makeObject();
    std::map<std::string, std::vector<u8>> contents;
    u64 cursor = 0;

    for (const auto& tensor : tensors) {
        const u64 bytes = tensorBytes(tensor);

        std::vector<u8> data(static_cast<usize>(bytes));
        for (u64 i = 0; i < bytes; ++i) {
            data[static_cast<usize>(i)] = patternByte(tensor.name, i);
        }
        contents[tensor.name] = std::move(data);

        json::Value entry = json::Value::makeObject();
        entry.set("dtype", std::string{toString(tensor.dtype)});
        json::Value shape = json::Value::makeArray();
        for (const u64 dim : tensor.shape) {
            shape.push(json::Value{dim});
        }
        entry.set("shape", std::move(shape));
        json::Value offsets = json::Value::makeArray();
        offsets.push(json::Value{cursor});
        offsets.push(json::Value{cursor + bytes});
        entry.set("data_offsets", std::move(offsets));
        header.set(tensor.name, std::move(entry));

        cursor += bytes;
    }

    const std::string headerJson = header.dump();

    std::vector<u8> file;
    file.reserve(8 + headerJson.size() + static_cast<usize>(cursor));

    const u64 headerLength = headerJson.size();
    for (int i = 0; i < 8; ++i) {
        file.push_back(static_cast<u8>((headerLength >> (i * 8)) & 0xFF));
    }
    file.insert(file.end(), headerJson.begin(), headerJson.end());
    for (const auto& tensor : tensors) {
        const auto& data = contents.at(tensor.name);
        file.insert(file.end(), data.begin(), data.end());
    }

    REQUIRE(io::writeFileAtomic(dir / "model-00001-of-00001.safetensors", file).has_value());

    // Index sidecar.
    json::Value index = json::Value::makeObject();
    json::Value metadata = json::Value::makeObject();
    metadata.set("total_size", cursor);
    index.set("metadata", std::move(metadata));
    json::Value weightMap = json::Value::makeObject();
    for (const auto& tensor : tensors) {
        weightMap.set(tensor.name, std::string{"model-00001-of-00001.safetensors"});
    }
    index.set("weight_map", std::move(weightMap));
    const std::string indexJson = index.dump(2);
    REQUIRE(io::writeFileAtomic(
                    dir / "model.safetensors.index.json",
                    ByteSpan{reinterpret_cast<const u8*>(indexJson.data()), indexJson.size()})
                    .has_value());

    const std::string config = miniConfig();
    REQUIRE(io::writeFileAtomic(
                    dir / "config.json",
                    ByteSpan{reinterpret_cast<const u8*>(config.data()), config.size()})
                    .has_value());

    const std::string tokenizer = R"({"version":"1.0","model":{"type":"BPE"}})";
    REQUIRE(io::writeFileAtomic(
                    dir / "tokenizer.json",
                    ByteSpan{reinterpret_cast<const u8*>(tokenizer.data()), tokenizer.size()})
                    .has_value());
    const std::string tokenizerConfig = R"({"model_max_length":4096})";
    REQUIRE(io::writeFileAtomic(dir / "tokenizer_config.json",
                                ByteSpan{reinterpret_cast<const u8*>(tokenizerConfig.data()),
                                         tokenizerConfig.size()})
                    .has_value());

    return contents;
}

std::vector<u8> readRange(const std::filesystem::path& path, ByteRange range) {
    auto file = io::File::openRead(path);
    REQUIRE(file.has_value());
    std::vector<u8> data(static_cast<usize>(range.length));
    REQUIRE(file->readExactAt(range.offset, data).has_value());
    return data;
}

}  // namespace

TEST_CASE("repack produces a complete, verifiable install", "[repacker]") {
    const TempDir temp;
    const auto checkpoint = temp.path() / "checkpoint";
    const auto install = temp.path() / "model.gturbo";

    writeCheckpoint(checkpoint);

    const auto result = repackFromCheckpoint(checkpoint, install);
    REQUIRE(result.has_value());

    CHECK(std::filesystem::exists(install / "manifest.json"));
    CHECK(std::filesystem::exists(install / "model_weights.bin"));
    CHECK(std::filesystem::exists(install / "packed_experts" / "layout.json"));
    CHECK(std::filesystem::exists(install / "packed_experts" / "layer_00.bin"));
    CHECK(std::filesystem::exists(install / "packed_experts" / "layer_01.bin"));
    CHECK(std::filesystem::exists(install / "tokenizer" / "tokenizer.json"));

    // The staging directory must not survive a successful run.
    CHECK_FALSE(std::filesystem::exists(install.string() + ".partial"));

    // Independently re-verify every recorded hash.
    CHECK(verifyInstalled(install).has_value());
}

TEST_CASE("expert bytes land exactly where the layout says", "[repacker]") {
    const TempDir temp;
    const auto checkpoint = temp.path() / "checkpoint";
    const auto install = temp.path() / "model.gturbo";

    const auto sourceData = writeCheckpoint(checkpoint);
    REQUIRE(repackFromCheckpoint(checkpoint, install).has_value());

    const auto manifest = gturbo::readManifest(install);
    REQUIRE(manifest.has_value());
    const auto& layout = manifest->experts;

    const QuantizedLinearLayout gateUp{
            .outFeatures = kMoeIntermediate, .inFeatures = kHidden, .spec = kWeightQuant};
    const QuantizedLinearLayout down{
            .outFeatures = kHidden, .inFeatures = kMoeIntermediate, .spec = kWeightQuant};

    // Every expert of every layer, every component: compare the installed bytes
    // against the exact slice of the stacked source tensor they came from.
    for (u64 layer = 0; layer < kLayers; ++layer) {
        const auto layerFile = install / "packed_experts" / layout.layerFiles[layer];

        for (u64 expert = 0; expert < kExperts; ++expert) {
            for (const auto& role : expertComponentRoles()) {
                const bool isDown = role.starts_with("down.");
                const QuantizedLinearLayout& linear = isDown ? down : gateUp;

                const usize dot = role.find('.');
                const std::string projection{role.substr(0, dot)};
                const std::string component{role.substr(dot + 1)};

                const std::string sourceName =
                        std::format("language_model.model.layers.{}.experts.switch_glu.{}_proj.{}",
                                    layer, projection, component);

                const auto& source = sourceData.at(sourceName);
                const u64 sliceBytes = source.size() / kExperts;
                CHECK(sliceBytes == (component == "weight" ? linear.weightBytes()
                                                           : linear.scaleBytes()));

                const auto range = layout.componentRange(layer, expert, role);
                REQUIRE(range.has_value());

                const std::vector<u8> installed = readRange(layerFile, *range);
                const std::vector<u8> expected(
                        source.begin() + static_cast<isize>(expert * sliceBytes),
                        source.begin() + static_cast<isize>((expert + 1) * sliceBytes));

                INFO("layer " << layer << " expert " << expert << " " << role);
                REQUIRE(installed == expected);
            }
        }
    }
}

TEST_CASE("resident tensors round-trip byte-for-byte", "[repacker]") {
    const TempDir temp;
    const auto checkpoint = temp.path() / "checkpoint";
    const auto install = temp.path() / "model.gturbo";

    const auto sourceData = writeCheckpoint(checkpoint);
    REQUIRE(repackFromCheckpoint(checkpoint, install).has_value());

    const auto manifest = gturbo::readManifest(install);
    REQUIRE(manifest.has_value());

    const auto residentPath = install / "model_weights.bin";
    for (const auto& tensor : manifest->resident.tensors()) {
        const std::string sourceName = "language_model.model." + tensor.name;
        const auto it = sourceData.find(sourceName);
        REQUIRE(it != sourceData.end());

        INFO("tensor: " << tensor.name);
        const std::vector<u8> installed = readRange(residentPath, tensor.range);
        REQUIRE(installed == it->second);
    }
}

TEST_CASE("vision tensors are excluded from the install", "[repacker]") {
    const TempDir temp;
    const auto checkpoint = temp.path() / "checkpoint";
    const auto install = temp.path() / "model.gturbo";

    writeCheckpoint(checkpoint);
    REQUIRE(repackFromCheckpoint(checkpoint, install).has_value());

    const auto manifest = gturbo::readManifest(install);
    REQUIRE(manifest.has_value());

    for (const auto& tensor : manifest->resident.tensors()) {
        CHECK(tensor.name.find("vision") == std::string::npos);
    }
    CHECK(manifest->resident.find("std_scale") == nullptr);
}

TEST_CASE("stride padding is zero-filled, not left as stale disk contents",
          "[repacker]") {
    const TempDir temp;
    const auto checkpoint = temp.path() / "checkpoint";
    const auto install = temp.path() / "model.gturbo";

    writeCheckpoint(checkpoint);
    REQUIRE(repackFromCheckpoint(checkpoint, install).has_value());

    const auto manifest = gturbo::readManifest(install);
    REQUIRE(manifest.has_value());
    const auto& layout = manifest->experts;
    REQUIRE(layout.stride > layout.blobBytes);

    const auto layerFile = install / "packed_experts" / layout.layerFiles[0];
    const u64 padBytes = layout.stride - layout.blobBytes;

    for (u64 expert = 0; expert < kExperts; ++expert) {
        const std::vector<u8> pad = readRange(
                layerFile, ByteRange{.offset = expert * layout.stride + layout.blobBytes,
                                     .length = padBytes});
        INFO("expert " << expert << " pad");
        REQUIRE(pad == std::vector<u8>(static_cast<usize>(padBytes), 0));
    }
}

TEST_CASE("manifest round-trips through JSON", "[repacker]") {
    const TempDir temp;
    const auto checkpoint = temp.path() / "checkpoint";
    const auto install = temp.path() / "model.gturbo";

    writeCheckpoint(checkpoint);
    REQUIRE(repackFromCheckpoint(checkpoint, install).has_value());

    const auto manifest = gturbo::readManifest(install);
    REQUIRE(manifest.has_value());

    CHECK(manifest->arch.numLayers == kLayers);
    CHECK(manifest->arch.numExperts == kExperts);
    CHECK(manifest->arch.hiddenSize == kHidden);
    CHECK(manifest->arch.attentionKEqV);
    CHECK(manifest->arch.isFullAttention(1));
    CHECK_FALSE(manifest->arch.isFullAttention(0));
    CHECK(manifest->arch.routerQuant.bits == 8);
    CHECK(manifest->arch.weightQuant.bits == 4);
    CHECK(manifest->source.repoId == "mlx-community/gemma-4-26b-a4b-it-4bit");

    const std::string encoded = manifest->toJson();
    const auto decoded = gturbo::Manifest::fromJson(encoded);
    REQUIRE(decoded.has_value());
    CHECK(decoded->arch.numLayers == manifest->arch.numLayers);
    CHECK(decoded->resident.tensors().size() == manifest->resident.tensors().size());
    CHECK(decoded->files.size() == manifest->files.size());
}

TEST_CASE("corruption is caught by verification", "[repacker]") {
    const TempDir temp;
    const auto checkpoint = temp.path() / "checkpoint";
    const auto install = temp.path() / "model.gturbo";

    writeCheckpoint(checkpoint);
    REQUIRE(repackFromCheckpoint(checkpoint, install).has_value());
    REQUIRE(verifyInstalled(install).has_value());

    SECTION("a flipped byte in an expert file") {
        const auto layerFile = install / "packed_experts" / "layer_00.bin";
        {
            auto file = io::File::openRead(layerFile);
            REQUIRE(file.has_value());
        }
        // Reopen for write and flip one byte.
        std::vector<u8> contents = *io::readBinaryFile(layerFile);
        contents[contents.size() / 2] ^= 0xFF;
        REQUIRE(io::writeFileAtomic(layerFile, contents).has_value());

        const auto status = verifyInstalled(install);
        REQUIRE_FALSE(status.has_value());
        CHECK(status.error().code() == ErrorCode::VerificationFailed);
        CHECK_THAT(status.error().message(), Catch::Matchers::ContainsSubstring("hashes to"));
    }

    SECTION("a truncated file") {
        const auto residentPath = install / "model_weights.bin";
        std::vector<u8> contents = *io::readBinaryFile(residentPath);
        contents.resize(contents.size() / 2);
        REQUIRE(io::writeFileAtomic(residentPath, contents).has_value());

        const auto status = verifyInstalled(install);
        REQUIRE_FALSE(status.has_value());
        CHECK(status.error().code() == ErrorCode::VerificationFailed);
        CHECK_THAT(status.error().message(), Catch::Matchers::ContainsSubstring("bytes, expected"));
    }

    SECTION("a missing file") {
        std::filesystem::remove(install / "packed_experts" / "layer_01.bin");
        const auto status = verifyInstalled(install);
        REQUIRE_FALSE(status.has_value());
        CHECK(status.error().code() == ErrorCode::VerificationFailed);
        CHECK_THAT(status.error().message(), Catch::Matchers::ContainsSubstring("missing"));
    }
}

TEST_CASE("a manifest-less directory reads as incomplete, not corrupt", "[repacker]") {
    const TempDir temp;
    const auto checkpoint = temp.path() / "checkpoint";
    const auto install = temp.path() / "model.gturbo";

    writeCheckpoint(checkpoint);
    REQUIRE(repackFromCheckpoint(checkpoint, install).has_value());

    // The manifest is written last precisely so its absence means "partial".
    std::filesystem::remove(install / "manifest.json");

    const auto manifest = gturbo::readManifest(install);
    REQUIRE_FALSE(manifest.has_value());
    CHECK(manifest.error().code() == ErrorCode::IncompleteInstall);
}

TEST_CASE("existing installs are protected without --overwrite", "[repacker]") {
    const TempDir temp;
    const auto checkpoint = temp.path() / "checkpoint";
    const auto install = temp.path() / "model.gturbo";

    writeCheckpoint(checkpoint);
    REQUIRE(repackFromCheckpoint(checkpoint, install).has_value());

    const auto refused = repackFromCheckpoint(checkpoint, install);
    REQUIRE_FALSE(refused.has_value());
    CHECK(refused.error().code() == ErrorCode::InvalidArgument);
    CHECK_THAT(refused.error().message(), Catch::Matchers::ContainsSubstring("--overwrite"));

    RepackOptions options;
    options.overwrite = true;
    CHECK(repackFromCheckpoint(checkpoint, install, options).has_value());
}

TEST_CASE("cancelling leaves no promoted install", "[repacker]") {
    const TempDir temp;
    const auto checkpoint = temp.path() / "checkpoint";
    const auto install = temp.path() / "model.gturbo";

    writeCheckpoint(checkpoint);

    int calls = 0;
    const auto result = repackFromCheckpoint(
            checkpoint, install, RepackOptions{},
            [&](const Progress&) { return ++calls < 2; });

    REQUIRE_FALSE(result.has_value());
    CHECK(result.error().code() == ErrorCode::Cancelled);
    CHECK_FALSE(std::filesystem::exists(install));
}

TEST_CASE("the scratch bound does not change the output", "[repacker]") {
    const TempDir temp;
    const auto checkpoint = temp.path() / "checkpoint";
    writeCheckpoint(checkpoint);

    // A bound far below one tensor forces heavy splitting; the resulting bytes
    // must be identical to a single-pass copy.
    RepackOptions coarse;
    coarse.plan.maxOpBytes = 8ull * 1024 * 1024;
    RepackOptions fine;
    fine.plan.maxOpBytes = 4096;

    const auto installA = temp.path() / "a.gturbo";
    const auto installB = temp.path() / "b.gturbo";
    REQUIRE(repackFromCheckpoint(checkpoint, installA, coarse).has_value());
    REQUIRE(repackFromCheckpoint(checkpoint, installB, fine).has_value());

    for (const auto* relative : {"model_weights.bin", "packed_experts/layer_00.bin",
                                 "packed_experts/layer_01.bin"}) {
        const auto hashA = io::Sha256::hashFile(installA / relative);
        const auto hashB = io::Sha256::hashFile(installB / relative);
        REQUIRE(hashA.has_value());
        REQUIRE(hashB.has_value());
        INFO("file: " << relative);
        CHECK(*hashA == *hashB);
    }
}
