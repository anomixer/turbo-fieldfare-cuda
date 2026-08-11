// Residency planning and KV cache sizing.
//
// These are pure arithmetic, but the arithmetic decides whether the model fits
// and whether anything streams at all, so the expected figures below are
// derived independently from the architecture rather than taken from the
// planner's own output.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cstdlib>
#include <filesystem>

#include "tf/core/format/ArchInfo.h"
#include "tf/runtime/Residency.h"

using namespace tf;
using namespace tf::runtime;

namespace {

/// The real architecture, built directly so these tests do not need the
/// checkpoint present.
ArchInfo gemma4() {
    ArchInfo arch;
    arch.hiddenSize = 2816;
    arch.numLayers = 30;
    arch.vocabSize = 262144;
    arch.maxPositionEmbeddings = 262144;
    arch.numHeads = 16;
    arch.numKVHeads = 8;
    arch.numGlobalKVHeads = 2;
    arch.headDim = 256;
    arch.globalHeadDim = 512;
    arch.slidingWindow = 1024;
    arch.attentionKEqV = true;
    arch.intermediateSize = 2112;
    arch.moeIntermediateSize = 704;
    arch.numExperts = 128;
    arch.topKExperts = 8;
    arch.hiddenActivation = "gelu_pytorch_tanh";
    arch.rmsNormEps = 1e-6;
    arch.finalLogitSoftcap = 30.0;
    arch.tieWordEmbeddings = true;
    arch.slidingRopeTheta = 10000.0;
    arch.fullRopeTheta = 1000000.0;
    arch.partialRotaryFactor = 0.25;
    arch.bosTokenId = 2;
    arch.eosTokenIds = {1, 106, 50};

    arch.layerTypes.reserve(30);
    for (u64 layer = 0; layer < 30; ++layer) {
        arch.layerTypes.push_back(layer % 6 == 5 ? AttentionKind::Full
                                                 : AttentionKind::Sliding);
    }
    return arch;
}

gturbo::ExpertLayout gemma4Experts() {
    gturbo::ExpertLayout layout;
    layout.numLayers = 30;
    layout.expertsPerLayer = 128;
    layout.blobBytes = 3345408;
    layout.alignment = align::kSector;
    layout.stride = alignUp(layout.blobBytes, align::kSector);  // 3346432
    for (u64 layer = 0; layer < 30; ++layer) {
        layout.layerFiles.push_back(gturbo::ExpertLayout::layerFileName(layer));
    }
    // Component table is only validated for tiling, so a single spanning entry
    // is enough for these tests.
    layout.components.push_back(gturbo::ExpertComponent{
            .role = "all", .dtype = DType::U32, .shape = {}, .offset = 0,
            .size = layout.blobBytes});
    return layout;
}

constexpr u64 kResidentCore = 1'353'689'148;  // measured from the real install
constexpr u64 kGiB = 1024ull * 1024 * 1024;

}  // namespace

TEST_CASE("KV rows differ by attention kind", "[residency]") {
    const ArchInfo arch = gemma4();

    // Sliding layers keep the window plus headroom, independent of context.
    CHECK(kvRowsForLayer(arch, 0, 4096) == 1024 + kSlidingWindowHeadroomRows);
    CHECK(kvRowsForLayer(arch, 0, 32768) == 1024 + kSlidingWindowHeadroomRows);
    CHECK(kvRowsForLayer(arch, 0, 4096) == 1152);

    // Full-attention layers must hold the whole context.
    CHECK(kvRowsForLayer(arch, 5, 4096) == 4096);
    CHECK(kvRowsForLayer(arch, 5, 32768) == 32768);
}

TEST_CASE("the sliding ring grows to cover a wider prefill chunk", "[residency]") {
    const ArchInfo arch = gemma4();

    // A chunk writes every key and value before any of its queries read them,
    // so the headroom beyond the window has to be at least the chunk width. Too
    // little and the chunk overwrites rows its own first tokens still need,
    // which corrupts the oldest visible history and reads as plausible output
    // rather than as a failure.
    CHECK(kvRowsForLayer(arch, 0, 4096, 128) == 1024 + 128);
    CHECK(kvRowsForLayer(arch, 0, 4096, 512) == 1024 + 512);

    // Below the floor the headroom stays at the floor.
    CHECK(kvRowsForLayer(arch, 0, 4096, 1) == 1024 + kSlidingWindowHeadroomRows);

    // Full-attention layers append, so the chunk width changes nothing.
    CHECK(kvRowsForLayer(arch, 5, 4096, 512) == 4096);
}

TEST_CASE("KV cache size at 4K matches the expected 305 MiB", "[residency]") {
    const ArchInfo arch = gemma4();

    // Sliding: 8 kv heads x 1152 rows x 256 dim x 2 bytes x 2 tensors.
    CHECK(kvCacheBytesForLayer(arch, 0, 4096) == 8ull * 1152 * 256 * 2 * 2);
    CHECK(kvCacheBytesForLayer(arch, 0, 4096) == 9'437'184);

    // Full: 2 kv heads x 4096 rows x 512 dim x 2 bytes x 2 tensors.
    CHECK(kvCacheBytesForLayer(arch, 5, 4096) == 2ull * 4096 * 512 * 2 * 2);
    CHECK(kvCacheBytesForLayer(arch, 5, 4096) == 16'777'216);

    u64 total = 0;
    for (u64 layer = 0; layer < arch.numLayers; ++layer) {
        total += kvCacheBytesForLayer(arch, layer, 4096);
    }
    // 25 sliding at 9 MiB plus 5 full at 16 MiB is 305 MiB, which is what
    // upstream reports for the same configuration.
    CHECK(total == 25 * 9'437'184ull + 5 * 16'777'216ull);
    CHECK(total / (1024 * 1024) == 305);
}

TEST_CASE("a large budget makes everything resident", "[residency]") {
    const ArchInfo arch = gemma4();
    const auto experts = gemma4Experts();

    const auto plan = planResidency(arch, experts, kResidentCore,
                                    ResidencyBudget{.deviceBytes = 24 * kGiB,
                                                    .contextLength = 4096});
    REQUIRE(plan.has_value());
    INFO(plan->describe());

    CHECK(plan->residentLayerCount() == 30);
    CHECK(plan->streamedLayerCount() == 0);
    CHECK(plan->isFullyResident());
    CHECK(plan->worstCaseStreamedBytesPerToken == 0);
    // All 30 layers of experts, padded to the stride.
    CHECK(plan->expertBytes == experts.stride * 128 * 30);
}

TEST_CASE("a constrained budget streams every layer", "[residency]") {
    const ArchInfo arch = gemma4();
    const auto experts = gemma4Experts();

    // 6 GiB stands in for an 8 GB card once the desktop has taken its share.
    // This is the configuration that keeps the streaming path covered on a
    // machine whose VRAM would otherwise hold the whole model.
    const auto plan = planResidency(arch, experts, kResidentCore,
                                    ResidencyBudget{.deviceBytes = 6 * kGiB,
                                                    .contextLength = 4096,
                                                    .slotsPerStreamedLayer = 16});
    REQUIRE(plan.has_value());
    INFO(plan->describe());

    CHECK(plan->streamedLayerCount() > 0);
    CHECK(plan->totalBytes() <= 6 * kGiB);

    // Worst case is every streamed layer missing all 8 routed experts.
    CHECK(plan->worstCaseStreamedBytesPerToken ==
          plan->streamedLayerCount() * 8 * experts.blobBytes);
}

TEST_CASE("residency degrades continuously with the budget", "[residency]") {
    const ArchInfo arch = gemma4();
    const auto experts = gemma4Experts();

    u64 previousResident = 0;
    for (const u64 budgetGiB : {u64{4}, u64{6}, u64{8}, u64{10}, u64{12}, u64{14}, u64{16}}) {
        const auto plan = planResidency(arch, experts, kResidentCore,
                                        ResidencyBudget{.deviceBytes = budgetGiB * kGiB,
                                                        .contextLength = 4096});
        REQUIRE(plan.has_value());
        INFO("budget " << budgetGiB << " GiB\n" << plan->describe());

        // More budget never means fewer resident layers, and never overruns.
        REQUIRE(plan->residentLayerCount() >= previousResident);
        REQUIRE(plan->totalBytes() <= budgetGiB * kGiB);
        REQUIRE(plan->layers.size() == 30);
        previousResident = plan->residentLayerCount();
    }
    // 16 GiB is enough for the whole model.
    CHECK(previousResident == 30);
}

TEST_CASE("streamed layers get the requested slot count", "[residency]") {
    const ArchInfo arch = gemma4();
    const auto experts = gemma4Experts();

    const auto plan = planResidency(arch, experts, kResidentCore,
                                    ResidencyBudget{.deviceBytes = 5 * kGiB,
                                                    .contextLength = 4096,
                                                    .slotsPerStreamedLayer = 12});
    REQUIRE(plan.has_value());

    for (const auto& layer : plan->layers) {
        if (layer.expertsResident) {
            CHECK(layer.slots == 0);
        } else {
            CHECK(layer.slots == 12);
        }
    }
}

TEST_CASE("resident layers are chosen from the bottom up", "[residency]") {
    const ArchInfo arch = gemma4();
    const auto experts = gemma4Experts();

    const auto plan = planResidency(arch, experts, kResidentCore,
                                    ResidencyBudget{.deviceBytes = 8 * kGiB,
                                                    .contextLength = 4096});
    REQUIRE(plan.has_value());

    // Once a streamed layer appears, everything above it must also stream, so
    // the resident set is a prefix.
    bool seenStreamed = false;
    for (u64 layer = 0; layer < plan->layers.size(); ++layer) {
        if (!plan->layers[static_cast<usize>(layer)].expertsResident) {
            seenStreamed = true;
        } else {
            INFO("layer " << layer << " is resident after a streamed layer");
            REQUIRE_FALSE(seenStreamed);
        }
    }
}

TEST_CASE("a budget too small for the fixed cost is refused", "[residency]") {
    const ArchInfo arch = gemma4();
    const auto experts = gemma4Experts();

    // The resident core alone is 1.26 GiB, so 1 GiB cannot work.
    const auto plan = planResidency(arch, experts, kResidentCore,
                                    ResidencyBudget{.deviceBytes = 1 * kGiB,
                                                    .contextLength = 4096});
    REQUIRE_FALSE(plan.has_value());
    CHECK(plan.error().code() == ErrorCode::OutOfMemory);
    // The message must name the components so the user knows what to lower.
    CHECK_THAT(plan.error().message(), Catch::Matchers::ContainsSubstring("resident core"));
    CHECK_THAT(plan.error().message(), Catch::Matchers::ContainsSubstring("KV cache"));
}

TEST_CASE("a slot count that does not fit is reduced, not refused", "[residency]") {
    const ArchInfo arch = gemma4();
    const auto experts = gemma4Experts();

    // Far more slots than a 6 GiB budget can hold. Slots are a performance
    // knob, and refusing to start because a tuning default was optimistic is
    // the wrong response - but the reduction has to be visible, or a user who
    // asked for 64 would never learn they got fewer.
    const auto plan = planResidency(arch, experts, kResidentCore,
                                    ResidencyBudget{.deviceBytes = 6 * kGiB,
                                                    .contextLength = 4096,
                                                    .slotsPerStreamedLayer = 64});
    REQUIRE(plan.has_value());
    CHECK(plan->slotsPerStreamedLayer < 64);
    CHECK(plan->slotsPerStreamedLayer >= arch.topKExperts);
    CHECK(plan->slotsWereReduced);
    CHECK_THAT(plan->describe(), Catch::Matchers::ContainsSubstring("reduced to fit"));
    CHECK(plan->totalBytes() <= 6 * kGiB);
}

TEST_CASE("a slot count that fits is left alone", "[residency]") {
    const ArchInfo arch = gemma4();
    const auto experts = gemma4Experts();

    const auto plan = planResidency(arch, experts, kResidentCore,
                                    ResidencyBudget{.deviceBytes = 8 * kGiB,
                                                    .contextLength = 4096,
                                                    .slotsPerStreamedLayer = 16});
    REQUIRE(plan.has_value());
    CHECK(plan->slotsPerStreamedLayer == 16);
    CHECK_FALSE(plan->slotsWereReduced);
}

TEST_CASE("a budget too small for top-k slots is still refused", "[residency]") {
    const ArchInfo arch = gemma4();
    const auto experts = gemma4Experts();

    // Below top-k there is nothing sensible left to reduce to: the cache would
    // evict experts the current token still needs.
    const auto plan = planResidency(arch, experts, kResidentCore,
                                    ResidencyBudget{.deviceBytes = 2 * kGiB,
                                                    .contextLength = 4096});
    REQUIRE_FALSE(plan.has_value());
    CHECK(plan.error().code() == ErrorCode::OutOfMemory);
    CHECK_THAT(plan.error().message(), Catch::Matchers::ContainsSubstring("top-8"));
}

TEST_CASE("fewer slots than top-k is refused", "[residency]") {
    const ArchInfo arch = gemma4();
    const auto experts = gemma4Experts();

    // With 4 slots and top-8 routing, every token would evict experts it still
    // needs for the same layer.
    const auto plan = planResidency(arch, experts, kResidentCore,
                                    ResidencyBudget{.deviceBytes = 8 * kGiB,
                                                    .contextLength = 4096,
                                                    .slotsPerStreamedLayer = 4});
    REQUIRE_FALSE(plan.has_value());
    CHECK(plan.error().code() == ErrorCode::InvalidArgument);
    CHECK_THAT(plan.error().message(), Catch::Matchers::ContainsSubstring("top-k"));
}

TEST_CASE("a longer context costs KV and therefore residency", "[residency]") {
    const ArchInfo arch = gemma4();
    const auto experts = gemma4Experts();

    const auto shortContext = planResidency(
            arch, experts, kResidentCore,
            ResidencyBudget{.deviceBytes = 10 * kGiB, .contextLength = 4096});
    const auto longContext = planResidency(
            arch, experts, kResidentCore,
            ResidencyBudget{.deviceBytes = 10 * kGiB, .contextLength = 32768});
    REQUIRE(shortContext.has_value());
    REQUIRE(longContext.has_value());

    // Only the 5 full-attention layers grow with context; the rings do not.
    CHECK(longContext->kvCacheBytes > shortContext->kvCacheBytes);
    CHECK(longContext->residentLayerCount() <= shortContext->residentLayerCount());
}

TEST_CASE("a context beyond the model's maximum is refused", "[residency]") {
    const ArchInfo arch = gemma4();
    const auto experts = gemma4Experts();

    const auto plan = planResidency(
            arch, experts, kResidentCore,
            ResidencyBudget{.deviceBytes = 64 * kGiB, .contextLength = 1'000'000});
    REQUIRE_FALSE(plan.has_value());
    CHECK_THAT(plan.error().message(), Catch::Matchers::ContainsSubstring("maximum"));
}

TEST_CASE("the plan describes itself for the CLI", "[residency]") {
    const ArchInfo arch = gemma4();
    const auto experts = gemma4Experts();

    const auto streamed = planResidency(arch, experts, kResidentCore,
                                        ResidencyBudget{.deviceBytes = 6 * kGiB,
                                                        .contextLength = 4096});
    REQUIRE(streamed.has_value());
    const std::string text = streamed->describe();
    CHECK_THAT(text, Catch::Matchers::ContainsSubstring("resident core"));
    CHECK_THAT(text, Catch::Matchers::ContainsSubstring("streamed"));
    CHECK_THAT(text, Catch::Matchers::ContainsSubstring("per token across PCIe"));

    const auto resident = planResidency(arch, experts, kResidentCore,
                                        ResidencyBudget{.deviceBytes = 24 * kGiB,
                                                        .contextLength = 4096});
    REQUIRE(resident.has_value());
    // A fully resident plan says so rather than reporting a zero worst case.
    CHECK_THAT(resident->describe(),
               Catch::Matchers::ContainsSubstring("nothing streams"));
}

TEST_CASE("the measured machine reaches full residency at its real budget",
          "[residency]") {
    const ArchInfo arch = gemma4();
    const auto experts = gemma4Experts();

    // 14.82 GiB free was measured by tools/cuda-probe on the RTX 5060 Ti.
    // Confirms the M0 finding: this card holds the whole model, so the streamed
    // path needs a deliberately lowered budget to get any coverage.
    const auto plan = planResidency(
            arch, experts, kResidentCore,
            ResidencyBudget{.deviceBytes = static_cast<u64>(14.82 * 1024) * 1024 * 1024,
                            .contextLength = 4096});
    REQUIRE(plan.has_value());
    INFO(plan->describe());
    CHECK(plan->isFullyResident());
}
