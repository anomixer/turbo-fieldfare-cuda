// Validates the dequantizer against genuine MLX-quantized weights.
//
// The round-trip tests in OpsTests.cpp quantize with our own quantizeMatrix and
// read it back with our own dequantizeElement, so a shared misunderstanding of
// the packing order or group layout would cancel out and pass. These tests read
// weights MLX actually produced and check structural properties that only hold
// if the interpretation is right.
//
// Skipped when no install is present; TF_GTURBO_DIR overrides the location.

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <set>
#include <vector>

#include "tf/core/format/GTurbo.h"
#include "tf/core/io/File.h"
#include "tf/core/math/Float.h"
#include "tf/reference/Ops.h"

using namespace tf;
using namespace tf::reference;

namespace {

std::filesystem::path installDir() {
    if (const char* override = std::getenv("TF_GTURBO_DIR")) {
        return std::filesystem::path{override};
    }
    if (const char* home = std::getenv("USERPROFILE")) {
        return std::filesystem::path{home} / "model-data" / "gemma4.gturbo";
    }
    return {};
}

bool installAvailable() {
    const auto dir = installDir();
    return !dir.empty() && std::filesystem::exists(dir / "manifest.json");
}

#define REQUIRE_INSTALL()                                                     \
    if (!installAvailable()) {                                                \
        SKIP("no .gturbo install at " << installDir().string()                \
                                      << " - run tf-repack first");           \
    }

std::vector<u8> readRange(const std::filesystem::path& path, ByteRange range) {
    auto file = io::File::openRead(path);
    REQUIRE(file.has_value());
    std::vector<u8> bytes(static_cast<usize>(range.length));
    REQUIRE(file->readExactAt(range.offset, bytes).has_value());
    return bytes;
}

}  // namespace

TEST_CASE("a real expert dequantizes to exactly 16 levels per group",
          "[real-weights]") {
    REQUIRE_INSTALL();

    const auto manifest = gturbo::readManifest(installDir());
    REQUIRE(manifest.has_value());
    const auto& layout = manifest->experts;
    const auto& arch = manifest->arch;

    const auto layerFile = installDir() / gturbo::kExpertsDir / layout.layerFiles[0];

    const auto weightRange = layout.componentRange(0, 0, "gate.weight");
    const auto scaleRange = layout.componentRange(0, 0, "gate.scales");
    const auto biasRange = layout.componentRange(0, 0, "gate.biases");
    REQUIRE(weightRange.has_value());
    REQUIRE(scaleRange.has_value());
    REQUIRE(biasRange.has_value());

    const auto weightBytes = readRange(layerFile, *weightRange);
    const auto scaleBytes = readRange(layerFile, *scaleRange);
    const auto biasBytes = readRange(layerFile, *biasRange);

    const auto packed = asU32(weightBytes);
    const auto scales = asBf16(scaleBytes);
    const auto biases = asBf16(biasBytes);

    const QuantizedLinearLayout linear = arch.expertGateUpLayout();
    REQUIRE(packed.size() == linear.outFeatures * linear.packedWordsPerRow());
    REQUIRE(scales.size() == linear.outFeatures * linear.groupsPerRow());

    // 4-bit affine quantization means every value in a group of 64 must land on
    // one of 16 levels: q * scale + bias for q in [0, 15]. Any mistake in the
    // packing order, group size or row stride would mix values from different
    // groups and blow this count far past 16.
    for (u64 row : {u64{0}, u64{1}, u64{17}, linear.outFeatures - 1}) {
        for (u64 group : {u64{0}, u64{1}, linear.groupsPerRow() - 1}) {
            std::set<float> levels;
            for (u64 i = 0; i < arch.weightQuant.groupSize; ++i) {
                const u64 column = group * arch.weightQuant.groupSize + i;
                levels.insert(dequantizeElement(packed, scales, biases, row, column,
                                                linear.inFeatures, arch.weightQuant));
            }
            INFO("row " << row << " group " << group << " produced " << levels.size()
                        << " distinct levels");
            REQUIRE(levels.size() <= 16);
            // A group collapsed to a single value would mean a dead scale.
            REQUIRE(levels.size() > 1);
        }
    }
}

TEST_CASE("real dequantized levels are evenly spaced", "[real-weights]") {
    REQUIRE_INSTALL();

    const auto manifest = gturbo::readManifest(installDir());
    REQUIRE(manifest.has_value());
    const auto& layout = manifest->experts;
    const auto& arch = manifest->arch;

    const auto layerFile = installDir() / gturbo::kExpertsDir / layout.layerFiles[3];
    const auto weightBytes =
            readRange(layerFile, *layout.componentRange(3, 11, "down.weight"));
    const auto scaleBytes =
            readRange(layerFile, *layout.componentRange(3, 11, "down.scales"));
    const auto biasBytes =
            readRange(layerFile, *layout.componentRange(3, 11, "down.biases"));

    const auto packed = asU32(weightBytes);
    const auto scales = asBf16(scaleBytes);
    const auto biases = asBf16(biasBytes);
    const QuantizedLinearLayout linear = arch.expertDownLayout();

    // Affine quantization is a uniform lattice, so the gaps between the sorted
    // distinct values of a group must all be multiples of one step. That holds
    // only if the scale and bias are being paired with the right elements.
    std::set<float> levels;
    for (u64 i = 0; i < arch.weightQuant.groupSize; ++i) {
        levels.insert(dequantizeElement(packed, scales, biases, 5, i, linear.inFeatures,
                                        arch.weightQuant));
    }
    REQUIRE(levels.size() >= 3);

    const std::vector<float> sorted(levels.begin(), levels.end());

    // MLX stores signed scales: a negative scale simply means the quantized
    // levels run downward from the bias. Dequantization is q * scale + bias
    // either way, so only the step magnitude matters here.
    const float scale = std::abs(toFloat(scales[5 * linear.groupsPerRow()]));
    REQUIRE(scale > 0.0f);

    for (usize i = 1; i < sorted.size(); ++i) {
        const float gap = sorted[i] - sorted[i - 1];
        const float steps = gap / scale;
        INFO("gap " << gap << " is " << steps << " steps of " << scale);
        REQUIRE(std::abs(steps - std::round(steps)) < 0.01f);
    }
}

TEST_CASE("real weights are finite and sanely scaled", "[real-weights]") {
    REQUIRE_INSTALL();

    const auto manifest = gturbo::readManifest(installDir());
    REQUIRE(manifest.has_value());
    const auto& layout = manifest->experts;
    const auto& arch = manifest->arch;
    const QuantizedLinearLayout linear = arch.expertGateUpLayout();

    const auto layerFile = installDir() / gturbo::kExpertsDir / layout.layerFiles[15];
    const auto weightBytes =
            readRange(layerFile, *layout.componentRange(15, 64, "up.weight"));
    const auto scaleBytes =
            readRange(layerFile, *layout.componentRange(15, 64, "up.scales"));
    const auto biasBytes =
            readRange(layerFile, *layout.componentRange(15, 64, "up.biases"));

    const auto packed = asU32(weightBytes);
    const auto scales = asBf16(scaleBytes);
    const auto biases = asBf16(biasBytes);

    double sum = 0.0;
    double sumSquares = 0.0;
    double worst = 0.0;
    u64 count = 0;

    for (u64 row = 0; row < 64; ++row) {
        for (u64 column = 0; column < linear.inFeatures; ++column) {
            const float value = dequantizeElement(packed, scales, biases, row, column,
                                                  linear.inFeatures, arch.weightQuant);
            REQUIRE(std::isfinite(value));
            sum += value;
            sumSquares += static_cast<double>(value) * value;
            worst = std::max(worst, std::abs(static_cast<double>(value)));
            ++count;
        }
    }

    const double mean = sum / static_cast<double>(count);
    const double rms = std::sqrt(sumSquares / static_cast<double>(count));

    INFO("mean " << mean << " rms " << rms << " max|w| " << worst);
    // Trained projection weights sit near zero mean with a small spread. A
    // misread of the packing would show up as a large bias or an absurd range.
    CHECK(std::abs(mean) < 0.05);
    CHECK(rms > 1e-4);
    CHECK(rms < 1.0);
    CHECK(worst < 10.0);
}

TEST_CASE("resident norm weights confirm the plain-weight convention",
          "[real-weights]") {
    REQUIRE_INSTALL();

    const auto manifest = gturbo::readManifest(installDir());
    REQUIRE(manifest.has_value());

    const auto residentPath = installDir() / gturbo::kResidentFile;

    const auto readNorm = [&](std::string_view name) {
        const auto* tensor = manifest->resident.find(name);
        REQUIRE(tensor != nullptr);
        const auto bytes = readRange(residentPath, tensor->range);
        const auto values = asBf16(bytes);
        std::vector<float> out(values.size());
        for (usize i = 0; i < values.size(); ++i) {
            out[i] = toFloat(values[i]);
        }
        return out;
    };

    const auto inputNorm = readNorm("layers.0.input_layernorm.weight");
    REQUIRE(inputNorm.size() == manifest->arch.hiddenSize);

    double sum = 0.0;
    float minimum = inputNorm[0];
    for (const float value : inputNorm) {
        sum += value;
        minimum = std::min(minimum, value);
    }
    const double mean = sum / static_cast<double>(inputNorm.size());

    INFO("input_layernorm mean " << mean << " min " << minimum);
    // Gemma 4 stores the effective scale directly. Under Gemma 1/2/3's
    // `1 + weight` convention these values would centre near zero; they do not,
    // and applying that convention would inflate every norm by roughly one.
    CHECK(mean > 1.0);
    CHECK(minimum > 0.0);

    // q_norm is near-identity, which is only true without the unit offset.
    const auto qNorm = readNorm("layers.0.self_attn.q_norm.weight");
    REQUIRE(qNorm.size() == manifest->arch.headDim);
    for (const float value : qNorm) {
        CHECK(value > 0.9f);
        CHECK(value < 1.2f);
    }
}

TEST_CASE("a real router projection is 8-bit, giving 256 levels",
          "[real-weights]") {
    REQUIRE_INSTALL();

    const auto manifest = gturbo::readManifest(installDir());
    REQUIRE(manifest.has_value());
    const auto& arch = manifest->arch;
    REQUIRE(arch.routerQuant.bits == 8);

    const auto residentPath = installDir() / gturbo::kResidentFile;
    const auto* weight = manifest->resident.find("layers.0.router.proj.weight");
    const auto* scale = manifest->resident.find("layers.0.router.proj.scales");
    const auto* bias = manifest->resident.find("layers.0.router.proj.biases");
    REQUIRE(weight != nullptr);
    REQUIRE(scale != nullptr);
    REQUIRE(bias != nullptr);

    // Each buffer is named: asU32/asBf16 return borrowed views, so passing a
    // temporary here would dangle.
    const auto weightBytes = readRange(residentPath, weight->range);
    const auto scaleBytes = readRange(residentPath, scale->range);
    const auto biasBytes = readRange(residentPath, bias->range);
    const auto packed = asU32(weightBytes);
    const auto scales = asBf16(scaleBytes);
    const auto biases = asBf16(biasBytes);

    const QuantizedLinearLayout linear = arch.routerLayout();
    REQUIRE(packed.size() == linear.outFeatures * linear.packedWordsPerRow());

    // 8-bit packs 4 values per word rather than 8, and admits 256 levels.
    CHECK(linear.valuesPerWord() == 4);

    std::set<float> levels;
    for (u64 i = 0; i < arch.routerQuant.groupSize; ++i) {
        const float value = dequantizeElement(packed, scales, biases, 0, i,
                                              linear.inFeatures, arch.routerQuant);
        REQUIRE(std::isfinite(value));
        levels.insert(value);
    }
    CHECK(levels.size() <= 256);
    // With 256 levels over 64 samples, near-total distinctness is expected;
    // a 4-bit misread would cap this at 16.
    CHECK(levels.size() > 16);
}
