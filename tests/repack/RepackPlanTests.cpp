#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cstdlib>
#include <filesystem>
#include <set>

#include "tf/core/format/ArchInfo.h"
#include "tf/core/format/Safetensors.h"
#include "tf/repack/RepackPlan.h"

using namespace tf;
using namespace tf::repack;

namespace {

std::filesystem::path checkpointDir() {
    if (const char* override = std::getenv("TF_CHECKPOINT_DIR")) {
        return std::filesystem::path{override};
    }
    if (const char* home = std::getenv("USERPROFILE")) {
        return std::filesystem::path{home} / "model-data" / "gemma-4-26b-a4b-it-4bit";
    }
    return {};
}

bool checkpointAvailable() {
    const auto dir = checkpointDir();
    return !dir.empty() && std::filesystem::exists(dir / "model.safetensors.index.json");
}

/// Loads the real checkpoint's metadata. Reads only JSON sidecars and
/// safetensors headers, so it stays fast despite the 14.3 GB of data behind it.
struct LoadedCheckpoint {
    ArchInfo arch;
    SafetensorsIndex index;
    std::vector<SafetensorsHeader> headers;
};

LoadedCheckpoint loadCheckpoint() {
    const auto dir = checkpointDir();

    auto arch = ArchInfo::readFromCheckpoint(dir);
    REQUIRE(arch.has_value());

    auto index = readSafetensorsIndex(dir);
    REQUIRE(index.has_value());

    std::vector<SafetensorsHeader> headers;
    for (const auto& shard : index->shardFiles()) {
        auto header = readSafetensorsHeader(dir / shard);
        REQUIRE(header.has_value());
        headers.push_back(std::move(*header));
    }

    return LoadedCheckpoint{
            .arch = std::move(*arch), .index = std::move(*index), .headers = std::move(headers)};
}

#define REQUIRE_CHECKPOINT()                                                        \
    if (!checkpointAvailable()) {                                                   \
        SKIP("checkpoint not present at " << checkpointDir().string()               \
                                          << " - run scripts/fetch-checkpoint.ps1"); \
    }

}  // namespace

TEST_CASE("expert component roles lay out a full blob", "[repack]") {
    const auto& roles = expertComponentRoles();
    CHECK(roles.size() == 9);
    // Each projection keeps its weight, scales and biases together so a single
    // dequant reads one contiguous run.
    CHECK(roles[0] == "gate.weight");
    CHECK(roles[2] == "gate.biases");
    CHECK(roles[8] == "down.biases");
}

TEST_CASE("plan covers the whole checkpoint", "[repack][checkpoint]") {
    REQUIRE_CHECKPOINT();
    const auto loaded = loadCheckpoint();

    const auto plan = buildPlan(loaded.arch, loaded.index, loaded.headers);
    REQUIRE(plan.has_value());

    // buildPlan validates coverage internally; assert the headline numbers.
    CHECK(plan->shards.size() == 3);
    CHECK(plan->experts.numLayers == 30);
    CHECK(plan->experts.expertsPerLayer == 128);
    CHECK(plan->experts.blobBytes == 3345408);

    // 4 KiB alignment adds exactly 1,024 bytes of pad per expert.
    CHECK(plan->experts.stride == 3346432);
    CHECK(plan->experts.stride - plan->experts.blobBytes == 1024);
    CHECK(plan->experts.alignment == align::kSector);
}

TEST_CASE("resident tensor count matches the checkpoint minus vision and experts",
          "[repack][checkpoint]") {
    REQUIRE_CHECKPOINT();
    const auto loaded = loadCheckpoint();
    const auto plan = buildPlan(loaded.arch, loaded.index, loaded.headers);
    REQUIRE(plan.has_value());

    u64 vision = 0;
    u64 experts = 0;
    for (const auto& [name, shard] : loaded.index.weightMap()) {
        if (name.starts_with("vision_tower.") || name.starts_with("embed_vision.")) {
            ++vision;
        } else if (name.find(kExpertMarker) != std::string::npos) {
            ++experts;
        }
    }

    // 9 stacked expert tensors per layer.
    CHECK(experts == 9 * 30);
    CHECK(plan->resident.tensors().size() == 1697 - vision - experts);

    // Derived independently: 25 sliding layers at 36 tensors, 5 full-attention
    // layers at 33 (no v_proj), plus 3 embedding tensors and the final norm.
    CHECK(plan->resident.tensors().size() == 25 * 36 + 5 * 33 + 4);
}

TEST_CASE("every resident tensor is aligned to its element size",
          "[repack][checkpoint]") {
    REQUIRE_CHECKPOINT();
    const auto loaded = loadCheckpoint();
    const auto plan = buildPlan(loaded.arch, loaded.index, loaded.headers);
    REQUIRE(plan.has_value());

    // Packed weights are dereferenced as u32 on the GPU. Laying tensors out
    // back to back put 118 of the 236 u32 tensors at 2 mod 4, because a 2-byte
    // tensor such as layer_scalar pushes the cursor off a 4-byte boundary. That
    // is undefined behaviour on the host and a hard fault on the device, and it
    // only surfaced as an opaque cudaErrorMisalignedAddress mid-decode.
    for (const auto& tensor : plan->resident.tensors()) {
        INFO("tensor " << tensor.name << " (" << toString(tensor.dtype) << ") at offset "
                       << tensor.range.offset);
        REQUIRE(isAligned(tensor.range.offset, byteWidth(tensor.dtype)));
        REQUIRE(isAligned(tensor.range.offset, gturbo::kResidentTensorAlignment));
    }

    // Alignment padding must stay negligible against the 1.26 GiB core.
    u64 payload = 0;
    for (const auto& tensor : plan->resident.tensors()) {
        payload += tensor.range.length;
    }
    const u64 padding = plan->residentBytes() - payload;
    INFO("padding " << padding << " bytes across " << plan->resident.tensors().size()
                    << " tensors");
    CHECK(padding < gturbo::kResidentTensorAlignment * plan->resident.tensors().size());
    CHECK(padding < 64 * 1024);
}

TEST_CASE("resident names drop the multimodal prefix and stay unique",
          "[repack][checkpoint]") {
    REQUIRE_CHECKPOINT();
    const auto loaded = loadCheckpoint();
    const auto plan = buildPlan(loaded.arch, loaded.index, loaded.headers);
    REQUIRE(plan.has_value());

    std::set<std::string> seen;
    for (const auto& tensor : plan->resident.tensors()) {
        INFO("tensor: " << tensor.name);
        CHECK_FALSE(tensor.name.starts_with("language_model."));
        CHECK(seen.insert(tensor.name).second);  // no duplicates
    }

    CHECK(plan->resident.find("embed_tokens.weight") != nullptr);
    CHECK(plan->resident.find("norm.weight") != nullptr);
    CHECK(plan->resident.find("layers.0.self_attn.q_proj.weight") != nullptr);
    CHECK(plan->resident.find("layers.0.router.proj.weight") != nullptr);

    // v_proj only on sliding layers; the full-attention layers reuse K.
    CHECK(plan->resident.find("layers.0.self_attn.v_proj.weight") != nullptr);
    CHECK(plan->resident.find("layers.5.self_attn.v_proj.weight") == nullptr);

    // Expert weights belong in the packed files, never the resident index.
    for (const auto& tensor : plan->resident.tensors()) {
        CHECK(tensor.name.find("switch_glu") == std::string::npos);
    }
}

TEST_CASE("output sizes match the expected install", "[repack][checkpoint]") {
    REQUIRE_CHECKPOINT();
    const auto loaded = loadCheckpoint();
    const auto plan = buildPlan(loaded.arch, loaded.index, loaded.headers);
    REQUIRE(plan.has_value());

    // 11.96 GiB of routed experts plus 1,024 bytes of pad per expert.
    CHECK(plan->expertBytes() == 3346432ull * 128 * 30);

    // The resident core is the ~1.35 GB that stays in memory for the session.
    CHECK(plan->residentBytes() > 1'300'000'000ull);
    CHECK(plan->residentBytes() < 1'400'000'000ull);

    // Copied bytes exclude alignment padding, which preallocation zero-fills.
    // Two sources: the expert stride, and the 16-byte resident tensor
    // alignment that keeps packed u32 weights addressable.
    const u64 expertPadding = (3346432ull - 3345408ull) * 128 * 30;
    const u64 totalPadding = plan->totalOutputBytes() - plan->totalCopiedBytes();
    CHECK(totalPadding > expertPadding);

    const u64 residentPadding = totalPadding - expertPadding;
    CHECK(residentPadding < gturbo::kResidentTensorAlignment *
                                    plan->resident.tensors().size());

    // The install is smaller than the 15.3 GB source by the vision tower.
    CHECK(plan->totalOutputBytes() < loaded.index.totalSize());
}

TEST_CASE("operations are bounded and sorted for sequential reads",
          "[repack][checkpoint]") {
    REQUIRE_CHECKPOINT();
    const auto loaded = loadCheckpoint();

    const PlanOptions options{.maxOpBytes = 8ull * 1024 * 1024};
    const auto plan = buildPlan(loaded.arch, loaded.index, loaded.headers, options);
    REQUIRE(plan.has_value());

    u32 lastShard = 0;
    u64 lastOffset = 0;
    for (const auto& op : plan->ops) {
        // No operation exceeds the scratch budget, so the executor runs on a
        // fixed buffer even against the 369 MB embedding table.
        REQUIRE(op.source.length <= options.maxOpBytes);
        REQUIRE(op.source.length > 0);

        // Non-decreasing within and across shards.
        if (op.shardIndex == lastShard) {
            REQUIRE(op.source.offset >= lastOffset);
        } else {
            REQUIRE(op.shardIndex > lastShard);
        }
        lastShard = op.shardIndex;
        lastOffset = op.source.offset;
    }

    // 30 layers x 128 experts x 9 components, plus the resident tensors.
    CHECK(plan->ops.size() > 30 * 128 * 9);
}

TEST_CASE("a smaller scratch budget splits more finely but covers the same bytes",
          "[repack][checkpoint]") {
    REQUIRE_CHECKPOINT();
    const auto loaded = loadCheckpoint();

    // 512 KiB is the bound M1b will use for HTTP range requests.
    const auto coarse = buildPlan(loaded.arch, loaded.index, loaded.headers,
                                  PlanOptions{.maxOpBytes = 8ull * 1024 * 1024});
    const auto fine = buildPlan(loaded.arch, loaded.index, loaded.headers,
                                PlanOptions{.maxOpBytes = 512ull * 1024});
    REQUIRE(coarse.has_value());
    REQUIRE(fine.has_value());

    CHECK(fine->ops.size() > coarse->ops.size());
    CHECK(fine->totalCopiedBytes() == coarse->totalCopiedBytes());
    CHECK(fine->residentBytes() == coarse->residentBytes());
    for (const auto& op : fine->ops) {
        REQUIRE(op.source.length <= 512ull * 1024);
    }
}

TEST_CASE("expert ranges are addressable and aligned", "[repack][checkpoint]") {
    REQUIRE_CHECKPOINT();
    const auto loaded = loadCheckpoint();
    const auto plan = buildPlan(loaded.arch, loaded.index, loaded.headers);
    REQUIRE(plan.has_value());

    const auto& layout = plan->experts;

    const auto first = layout.expertRange(0, 0);
    REQUIRE(first.has_value());
    CHECK(first->offset == 0);
    CHECK(first->length == 3345408);

    const auto second = layout.expertRange(0, 1);
    REQUIRE(second.has_value());
    CHECK(second->offset == 3346432);

    // Every blob starts on a sector boundary, which is what unbuffered DMA
    // reads require.
    for (u64 expert = 0; expert < layout.expertsPerLayer; ++expert) {
        const auto range = layout.expertRange(29, expert);
        REQUIRE(range.has_value());
        REQUIRE(isAligned(range->offset, align::kSector));
    }
    CHECK(layout.supportsUnbufferedReads(4096));
    CHECK(layout.supportsUnbufferedReads(512));

    // Component offsets tile the blob.
    const auto gate = layout.componentRange(3, 7, "gate.weight");
    REQUIRE(gate.has_value());
    CHECK(gate->offset == 7 * layout.stride);
    CHECK(gate->length == 991232);

    const auto downBiases = layout.componentRange(3, 7, "down.biases");
    REQUIRE(downBiases.has_value());
    CHECK(downBiases->offset + downBiases->length == 7 * layout.stride + layout.blobBytes);

    CHECK_FALSE(layout.componentRange(0, 0, "nonexistent").has_value());
    CHECK_FALSE(layout.expertRange(30, 0).has_value());
    CHECK_FALSE(layout.expertRange(0, 128).has_value());
}

TEST_CASE("layout survives a JSON round-trip", "[repack][checkpoint]") {
    REQUIRE_CHECKPOINT();
    const auto loaded = loadCheckpoint();
    const auto plan = buildPlan(loaded.arch, loaded.index, loaded.headers);
    REQUIRE(plan.has_value());

    const std::string json = plan->experts.toJson();
    const auto decoded = gturbo::ExpertLayout::fromJson(json);
    REQUIRE(decoded.has_value());

    CHECK(decoded->numLayers == plan->experts.numLayers);
    CHECK(decoded->stride == plan->experts.stride);
    CHECK(decoded->blobBytes == plan->experts.blobBytes);
    CHECK(decoded->components == plan->experts.components);
    CHECK(decoded->layerFiles == plan->experts.layerFiles);

    // The compact layout stays small: the point of describing experts by stride
    // rather than enumerating all 3,840 of them.
    CHECK(json.size() < 8000);
}

TEST_CASE("mismatched header count is rejected", "[repack][checkpoint]") {
    REQUIRE_CHECKPOINT();
    auto loaded = loadCheckpoint();
    loaded.headers.pop_back();

    const auto plan = buildPlan(loaded.arch, loaded.index, loaded.headers);
    REQUIRE_FALSE(plan.has_value());
    CHECK(plan.error().code() == ErrorCode::InvalidArgument);
    CHECK_THAT(plan.error().message(), Catch::Matchers::ContainsSubstring("shard headers"));
}
