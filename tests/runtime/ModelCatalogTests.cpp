// Model catalog and hardware-driven selection.
//
// The point of the recommender is that a streaming mixture of experts runs on
// hardware a dense model of similar footprint cannot. These tests pin that
// property across a range of machines rather than just checking the arithmetic.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>

#include "tf/runtime/ModelCatalog.h"

using namespace tf;
using namespace tf::runtime;

namespace {

constexpr u64 kGiB = 1024ull * 1024 * 1024;

MachineProfile machine(double vramGiB, double ramGiB = 64.0, double diskGiB = 500.0,
                       bool ssd = true) {
    return MachineProfile{.deviceBytes = static_cast<u64>(vramGiB * kGiB),
                          .systemRamBytes = static_cast<u64>(ramGiB * kGiB),
                          .diskFreeBytes = static_cast<u64>(diskGiB * kGiB),
                          .diskIsSolidState = ssd,
                          .hasGpu = true};
}

const ModelFit* fitFor(const Recommendation& recommendation, std::string_view id) {
    const auto it = std::ranges::find_if(recommendation.assessed, [&](const ModelFit& fit) {
        return fit.variant->id == id;
    });
    return it == recommendation.assessed.end() ? nullptr : &*it;
}

}  // namespace

TEST_CASE("the catalog covers the published family", "[catalog]") {
    const auto variants = modelCatalog();
    REQUIRE(variants.size() == 5);

    for (const auto* id : {"e2b", "e4b", "12b", "26b-a4b", "31b"}) {
        INFO("variant " << id);
        REQUIRE(findModel(id).has_value());
    }

    const auto unknown = findModel("70b");
    REQUIRE_FALSE(unknown.has_value());
    // The error lists what is available, so a typo is self-correcting.
    CHECK_THAT(unknown.error().message(), Catch::Matchers::ContainsSubstring("26b-a4b"));
}

TEST_CASE("only the mixture of experts is streamable", "[catalog]") {
    for (const auto& variant : modelCatalog()) {
        INFO("variant " << variant.id);
        if (variant.family == ModelFamily::MixtureOfExperts) {
            REQUIRE(variant.streamableBytes > 0);
            // Its resident core is a small fraction of the whole, which is what
            // makes streaming worthwhile in the first place.
            REQUIRE(variant.residentBytes * 5 < variant.streamableBytes);
        } else {
            // A dense model has nothing to stream: every weight fires per token.
            REQUIRE(variant.streamableBytes == 0);
            REQUIRE(variant.residentBytes == variant.installBytes);
        }
    }
}

TEST_CASE("exactly one variant runs on this build today", "[catalog]") {
    u32 supported = 0;
    for (const auto& variant : modelCatalog()) {
        if (variant.support == RuntimeSupport::Supported) {
            ++supported;
            CHECK(variant.id == "26b-a4b");
            CHECK_FALSE(variant.sizesAreEstimates);
            CHECK_FALSE(variant.revision.empty());
        } else {
            // Anything unsupported must say what it would need.
            INFO("variant " << variant.id);
            REQUIRE_FALSE(variant.supportNote.empty());
        }
    }
    CHECK(supported == 1);
}

TEST_CASE("a mixture of experts runs where a dense model of similar size cannot",
          "[catalog]") {
    const auto moe = findModel("26b-a4b");
    const auto dense = findModel("31b");
    REQUIRE(moe.has_value());
    REQUIRE(dense.has_value());

    // The whole argument for the streaming design: 26B of weights need only
    // the core plus slots resident, while 31B of dense weights need all of it.
    CHECK((*moe)->minimumDeviceBytes() < (*dense)->minimumDeviceBytes() / 4);
    CHECK((*moe)->minimumDeviceBytes() < 4 * kGiB);
    CHECK((*dense)->minimumDeviceBytes() > 17 * kGiB);
}

TEST_CASE("a large card holds the mixture of experts outright", "[catalog]") {
    // The measured free VRAM on the development machine.
    const auto recommendation = recommendModel(machine(14.82));

    REQUIRE(recommendation.best != nullptr);
    CHECK(recommendation.best->id == "26b-a4b");

    const ModelFit* moe = fitFor(recommendation, "26b-a4b");
    REQUIRE(moe != nullptr);
    CHECK(moe->fit == FitQuality::Comfortable);

    // 31B is larger but dense, so it does not fit and must not be recommended.
    const ModelFit* dense = fitFor(recommendation, "31b");
    REQUIRE(dense != nullptr);
    CHECK(dense->fit == FitQuality::DoesNotFit);
    CHECK_THAT(dense->rationale, Catch::Matchers::ContainsSubstring("dense weights"));
}

TEST_CASE("a small card still runs the mixture of experts, streamed", "[catalog]") {
    // 6 GiB stands in for an 8 GB card after the desktop takes its share.
    const auto recommendation = recommendModel(machine(6.0));

    REQUIRE(recommendation.best != nullptr);
    CHECK(recommendation.best->id == "26b-a4b");

    const ModelFit* moe = fitFor(recommendation, "26b-a4b");
    REQUIRE(moe != nullptr);
    CHECK(moe->fit == FitQuality::Streamed);
    CHECK_THAT(moe->rationale, Catch::Matchers::ContainsSubstring("stream"));
}

TEST_CASE("even a 4 GiB card runs the mixture of experts", "[catalog]") {
    // This is the claim the port exists to make good on: a 26B model on a card
    // that cannot hold a 6 GiB dense one.
    const auto recommendation = recommendModel(machine(4.0));

    REQUIRE(recommendation.best != nullptr);
    CHECK(recommendation.best->id == "26b-a4b");
    CHECK(fitFor(recommendation, "26b-a4b")->fit == FitQuality::Streamed);

    // The 12B dense model, less than half the parameters, does not fit at all.
    CHECK(fitFor(recommendation, "12b")->fit == FitQuality::DoesNotFit);
}

TEST_CASE("a card too small for the core recommends nothing", "[catalog]") {
    const auto recommendation = recommendModel(machine(2.0));

    CHECK(recommendation.best == nullptr);
    CHECK(fitFor(recommendation, "26b-a4b")->fit == FitQuality::DoesNotFit);
    CHECK_THAT(recommendation.format(),
               Catch::Matchers::ContainsSubstring("No variant"));
}

TEST_CASE("no GPU means nothing runs", "[catalog]") {
    MachineProfile headless = machine(0.0);
    headless.hasGpu = false;

    const auto recommendation = recommendModel(headless);
    CHECK(recommendation.best == nullptr);
    for (const auto& fit : recommendation.assessed) {
        INFO("variant " << fit.variant->id);
        REQUIRE(fit.fit == FitQuality::DoesNotFit);
        REQUIRE_THAT(fit.rationale, Catch::Matchers::ContainsSubstring("no GPU"));
    }
}

TEST_CASE("insufficient disk rules a variant out entirely", "[catalog]") {
    // Plenty of VRAM, but only 5 GiB of disk: the install cannot land.
    const auto recommendation = recommendModel(machine(24.0, 64.0, 5.0));

    const ModelFit* moe = fitFor(recommendation, "26b-a4b");
    REQUIRE(moe != nullptr);
    CHECK(moe->fit == FitQuality::DoesNotFit);
    CHECK_THAT(moe->rationale, Catch::Matchers::ContainsSubstring("free on disk"));
    CHECK(recommendation.best == nullptr);
}

TEST_CASE("a spinning install disk warns about streaming", "[catalog]") {
    const auto recommendation = recommendModel(machine(6.0, 64.0, 500.0, /*ssd=*/false));

    const ModelFit* moe = fitFor(recommendation, "26b-a4b");
    REQUIRE(moe != nullptr);
    REQUIRE(moe->fit == FitQuality::Streamed);

    // A .gturbo on a spinning disk works but defeats the design.
    const bool warned = std::ranges::any_of(moe->warnings, [](const std::string& warning) {
        return warning.find("spinning disk") != std::string::npos;
    });
    CHECK(warned);
}

TEST_CASE("too little RAM to cache the experts warns", "[catalog]") {
    // 8 GiB of RAM cannot hold the 12 GiB expert set, so reads stay bound by
    // the SSD rather than falling back to the page cache.
    const auto recommendation = recommendModel(machine(6.0, 8.0));

    const ModelFit* moe = fitFor(recommendation, "26b-a4b");
    REQUIRE(moe != nullptr);
    const bool warned = std::ranges::any_of(moe->warnings, [](const std::string& warning) {
        return warning.find("cannot cache") != std::string::npos;
    });
    CHECK(warned);

    // Ample RAM produces no such warning.
    const auto roomy = recommendModel(machine(6.0, 64.0));
    const ModelFit* cached = fitFor(roomy, "26b-a4b");
    CHECK(std::ranges::none_of(cached->warnings, [](const std::string& warning) {
        return warning.find("cannot cache") != std::string::npos;
    }));
}

TEST_CASE("a fitting but unsupported variant is never recommended", "[catalog]") {
    // 24 GiB fits every dense variant, but none of them can run yet.
    const auto recommendation = recommendModel(machine(24.0));

    REQUIRE(recommendation.best != nullptr);
    CHECK(recommendation.best->support == RuntimeSupport::Supported);
    CHECK(recommendation.best->id == "26b-a4b");

    const ModelFit* dense = fitFor(recommendation, "31b");
    REQUIRE(dense != nullptr);
    CHECK(dense->fit == FitQuality::Comfortable);
    CHECK_FALSE(dense->runnable());
    // The report explains why rather than silently skipping it.
    const bool explained = std::ranges::any_of(dense->warnings, [](const std::string& w) {
        return w.find("cannot run it yet") != std::string::npos;
    });
    CHECK(explained);
}

TEST_CASE("larger variants are always assessed first", "[catalog]") {
    const auto recommendation = recommendModel(machine(24.0));

    double previous = 1e9;
    for (const auto& fit : recommendation.assessed) {
        INFO("variant " << fit.variant->id);
        REQUIRE(fit.variant->totalParamsB <= previous);
        previous = fit.variant->totalParamsB;
    }
}

TEST_CASE("the report marks the recommendation", "[catalog]") {
    const auto recommendation = recommendModel(machine(14.82));
    const std::string text = recommendation.format();

    CHECK_THAT(text, Catch::Matchers::ContainsSubstring("Recommended:"));
    CHECK_THAT(text, Catch::Matchers::ContainsSubstring("26B-A4B"));
    CHECK_THAT(text, Catch::Matchers::ContainsSubstring("->"));
}

TEST_CASE("the machine profile reads real disk and memory", "[catalog]") {
    const auto detected = detectMachine(std::filesystem::current_path());
    REQUIRE(detected.has_value());

    CHECK(detected->systemRamBytes > 0);
    CHECK(detected->diskFreeBytes > 0);
    // detectMachine deliberately leaves the GPU fields alone so it works
    // without a backend; the caller fills them in.
    CHECK_FALSE(detected->hasGpu);
}
