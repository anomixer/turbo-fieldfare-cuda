// Exercises the GPU backend against real hardware.
//
// Skipped, not failed, when no device is available, so the suite still runs on
// a machine without a GPU. The transfer and synchronization behaviour checked
// here is what the M6 expert streamer and the M7 decode loop are built on.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <numeric>
#include <thread>
#include <vector>

#include "Harness.h"
#include "tf/gpu/Backend.h"
#include "tf/gpu/Kernels.h"
#include "tf/gpu/Preflight.h"

using namespace tf;
using namespace tf::gpu;

namespace {


std::vector<u8> pattern(usize size, u8 seed = 0) {
    std::vector<u8> data(size);
    for (usize i = 0; i < size; ++i) {
        data[i] = static_cast<u8>((i * 31 + seed) & 0xFF);
    }
    return data;
}

}  // namespace

TEST_CASE("device info is plausible", "[gpu]") {
    REQUIRE_GPU_NO_STREAM();

    const DeviceInfo& info = backend.info();
    INFO("device: " << info.name);

    CHECK_FALSE(info.name.empty());
    CHECK(info.totalMemoryBytes > 0);
    // Not asserted: multiprocessorCount is a CUDA concept and D3D12 exposes no
    // equivalent, so the D3D12 backend leaves it zero rather than guessing.
    if (tfBackendKind == BackendKind::Cuda) {
        CHECK(info.multiprocessorCount > 0);
    }
    // Every kernel ported from a Metal simdgroup assumes exactly 32 lanes.
    CHECK(info.warpSize == 32);
    CHECK(info.sharedMemoryPerBlockBytes >= 16 * 1024);
}

TEST_CASE("free memory is reported and bounded by total", "[gpu]") {
    REQUIRE_GPU_NO_STREAM();

    const auto memory = backend.memoryInfo();
    REQUIRE(memory.has_value());
    CHECK(memory->freeBytes > 0);
    CHECK(memory->freeBytes <= memory->totalBytes);
}

TEST_CASE("host to device and back round-trips", "[gpu]") {
    REQUIRE_GPU_NO_STREAM();

    constexpr usize kBytes = 1 << 20;
    const auto source = pattern(kBytes);

    auto buffer = backend.allocate(MemoryKind::Device, kBytes, "roundtrip");
    REQUIRE(buffer.has_value());
    CHECK((*buffer)->size() == kBytes);
    // Device memory is not CPU-addressable.
    CHECK((*buffer)->hostPointer() == nullptr);
    CHECK((*buffer)->hostBytes().empty());

    auto stream = backend.createStream("test");
    REQUIRE(stream.has_value());

    REQUIRE(backend.enqueueUpload(**stream, **buffer, 0, source).has_value());

    std::vector<u8> readback(kBytes);
    REQUIRE(backend.enqueueDownload(**stream, readback, **buffer, 0).has_value());
    REQUIRE((*stream)->synchronize().has_value());

    CHECK(readback == source);
}

TEST_CASE("offsets address sub-ranges of a buffer", "[gpu]") {
    REQUIRE_GPU_NO_STREAM();

    constexpr usize kBytes = 4096;
    auto buffer = backend.allocate(MemoryKind::Device, kBytes, "offsets");
    auto stream = backend.createStream("test");
    REQUIRE(buffer.has_value());
    REQUIRE(stream.has_value());

    REQUIRE(backend.enqueueFill(**stream, **buffer, 0, kBytes, 0).has_value());

    // Write a marker into the middle only.
    const std::vector<u8> marker{0xAA, 0xBB, 0xCC, 0xDD};
    REQUIRE(backend.enqueueUpload(**stream, **buffer, 2048, marker).has_value());

    std::vector<u8> whole(kBytes);
    REQUIRE(backend.enqueueDownload(**stream, whole, **buffer, 0).has_value());
    REQUIRE((*stream)->synchronize().has_value());

    CHECK(whole[2047] == 0);
    CHECK(whole[2048] == 0xAA);
    CHECK(whole[2051] == 0xDD);
    CHECK(whole[2052] == 0);

    // Downloading from an offset reads the same bytes.
    std::vector<u8> slice(4);
    REQUIRE(backend.enqueueDownload(**stream, slice, **buffer, 2048).has_value());
    REQUIRE((*stream)->synchronize().has_value());
    CHECK(slice == marker);
}

TEST_CASE("out-of-range transfers are refused, not truncated", "[gpu]") {
    REQUIRE_GPU_NO_STREAM();

    auto buffer = backend.allocate(MemoryKind::Device, 1024, "bounds");
    auto stream = backend.createStream("test");
    REQUIRE(buffer.has_value());
    REQUIRE(stream.has_value());

    const std::vector<u8> source(512);

    // Past the end.
    auto overrun = backend.enqueueUpload(**stream, **buffer, 768, source);
    REQUIRE_FALSE(overrun.has_value());
    CHECK(overrun.error().code() == ErrorCode::InvalidArgument);
    // The message names the buffer, which is what makes this findable.
    CHECK_THAT(overrun.error().message(), Catch::Matchers::ContainsSubstring("bounds"));

    // Offset beyond the buffer entirely.
    CHECK_FALSE(backend.enqueueUpload(**stream, **buffer, 4096, source).has_value());

    // A copy whose length overruns the source.
    auto other = backend.allocate(MemoryKind::Device, 256, "small");
    REQUIRE(other.has_value());
    CHECK_FALSE(backend.enqueueCopy(**stream, **buffer, 0, **other, 0, 512).has_value());
}

TEST_CASE("pinned host memory is CPU-addressable and transfers", "[gpu]") {
    REQUIRE_GPU_NO_STREAM();

    constexpr usize kBytes = 64 * 1024;
    auto staging = backend.allocate(MemoryKind::PinnedHost, kBytes, "staging");
    REQUIRE(staging.has_value());

    // This is the buffer the expert streamer reads file bytes into.
    auto host = (*staging)->hostBytes();
    REQUIRE(host.size() == kBytes);
    REQUIRE(host.data() != nullptr);

    const auto source = pattern(kBytes, 7);
    std::ranges::copy(source, host.begin());

    auto device = backend.allocate(MemoryKind::Device, kBytes, "device");
    auto stream = backend.createStream("test");
    REQUIRE(device.has_value());
    REQUIRE(stream.has_value());

    REQUIRE(backend.enqueueUpload(**stream, **device, 0, host).has_value());

    std::vector<u8> readback(kBytes);
    REQUIRE(backend.enqueueDownload(**stream, readback, **device, 0).has_value());
    REQUIRE((*stream)->synchronize().has_value());
    CHECK(readback == source);
}

TEST_CASE("device to device copy moves bytes", "[gpu]") {
    REQUIRE_GPU_NO_STREAM();

    constexpr usize kBytes = 8192;
    auto a = backend.allocate(MemoryKind::Device, kBytes, "a");
    auto b = backend.allocate(MemoryKind::Device, kBytes, "b");
    auto stream = backend.createStream("test");
    REQUIRE(a.has_value());
    REQUIRE(b.has_value());
    REQUIRE(stream.has_value());

    const auto source = pattern(kBytes, 3);
    REQUIRE(backend.enqueueUpload(**stream, **a, 0, source).has_value());
    REQUIRE(backend.enqueueFill(**stream, **b, 0, kBytes, 0).has_value());
    REQUIRE(backend.enqueueCopy(**stream, **b, 0, **a, 0, kBytes).has_value());

    std::vector<u8> readback(kBytes);
    REQUIRE(backend.enqueueDownload(**stream, readback, **b, 0).has_value());
    REQUIRE((*stream)->synchronize().has_value());
    CHECK(readback == source);
}

TEST_CASE("fill writes the requested byte", "[gpu]") {
    REQUIRE_GPU_NO_STREAM();

    constexpr usize kBytes = 2048;
    auto buffer = backend.allocate(MemoryKind::Device, kBytes, "fill");
    auto stream = backend.createStream("test");
    REQUIRE(buffer.has_value());
    REQUIRE(stream.has_value());

    REQUIRE(backend.enqueueFill(**stream, **buffer, 0, kBytes, 0x5A).has_value());

    std::vector<u8> readback(kBytes);
    REQUIRE(backend.enqueueDownload(**stream, readback, **buffer, 0).has_value());
    REQUIRE((*stream)->synchronize().has_value());

    CHECK(readback == std::vector<u8>(kBytes, 0x5A));
}

TEST_CASE("events order work across streams", "[gpu]") {
    REQUIRE_GPU_NO_STREAM();

    // The decode loop depends on exactly this: the routed branch waits for the
    // shared-expert kernels and the expert upload without the CPU blocking.
    constexpr usize kBytes = 4 << 20;

    auto producerStream = backend.createStream("producer");
    auto consumerStream = backend.createStream("consumer");
    auto source = backend.allocate(MemoryKind::Device, kBytes, "source");
    auto destination = backend.allocate(MemoryKind::Device, kBytes, "destination");
    auto event = backend.createEvent(/*withTiming=*/false);

    REQUIRE(producerStream.has_value());
    REQUIRE(consumerStream.has_value());
    REQUIRE(source.has_value());
    REQUIRE(destination.has_value());
    REQUIRE(event.has_value());

    const auto payload = pattern(kBytes, 11);

    REQUIRE(backend.enqueueFill(**consumerStream, **destination, 0, kBytes, 0).has_value());
    REQUIRE((*consumerStream)->synchronize().has_value());

    // Producer fills `source`, then signals.
    REQUIRE(backend.enqueueUpload(**producerStream, **source, 0, payload).has_value());
    REQUIRE((*event)->record(**producerStream).has_value());

    // Consumer waits for the signal before copying, without a CPU sync between.
    REQUIRE((*event)->wait(**consumerStream).has_value());
    REQUIRE(backend.enqueueCopy(**consumerStream, **destination, 0, **source, 0, kBytes)
                    .has_value());

    std::vector<u8> readback(kBytes);
    REQUIRE(backend.enqueueDownload(**consumerStream, readback, **destination, 0)
                    .has_value());
    REQUIRE((*consumerStream)->synchronize().has_value());

    CHECK(readback == payload);
}

TEST_CASE("timing events measure elapsed work", "[gpu]") {
    REQUIRE_GPU_NO_STREAM();

    constexpr usize kBytes = 32 << 20;
    auto stream = backend.createStream("timed");
    auto buffer = backend.allocate(MemoryKind::Device, kBytes, "timed");
    auto start = backend.createEvent(/*withTiming=*/true);
    auto stop = backend.createEvent(/*withTiming=*/true);

    REQUIRE(stream.has_value());
    REQUIRE(buffer.has_value());
    REQUIRE(start.has_value());
    REQUIRE(stop.has_value());

    REQUIRE((*start)->record(**stream).has_value());
    for (int i = 0; i < 8; ++i) {
        REQUIRE(backend.enqueueFill(**stream, **buffer, 0, kBytes, static_cast<u8>(i))
                        .has_value());
    }
    REQUIRE((*stop)->record(**stream).has_value());
    REQUIRE((*stop)->synchronize().has_value());

    const auto elapsed = (*stop)->elapsedMillisSince(**start);
    REQUIRE(elapsed.has_value());
    INFO("elapsed " << *elapsed << " ms");
    CHECK(*elapsed > 0.0);
    CHECK(*elapsed < 5000.0);
}

TEST_CASE("timing requires events created with timing enabled", "[gpu]") {
    REQUIRE_GPU_NO_STREAM();

    auto stream = backend.createStream("untimed");
    auto start = backend.createEvent(/*withTiming=*/false);
    auto stop = backend.createEvent(/*withTiming=*/false);
    REQUIRE(stream.has_value());

    REQUIRE((*start)->record(**stream).has_value());
    REQUIRE((*stop)->record(**stream).has_value());
    REQUIRE((*stream)->synchronize().has_value());

    // Returns an error rather than a meaningless number.
    const auto elapsed = (*stop)->elapsedMillisSince(**start);
    REQUIRE_FALSE(elapsed.has_value());
    CHECK(elapsed.error().code() == ErrorCode::InvalidArgument);
}

TEST_CASE("isIdle and isComplete answer without blocking", "[gpu]") {
    REQUIRE_GPU_NO_STREAM();

    auto stream = backend.createStream("idle");
    auto event = backend.createEvent(false);
    REQUIRE(stream.has_value());
    REQUIRE(event.has_value());

    REQUIRE((*stream)->synchronize().has_value());
    const auto idle = (*stream)->isIdle();
    REQUIRE(idle.has_value());
    CHECK(*idle);

    REQUIRE((*event)->record(**stream).has_value());
    REQUIRE((*event)->synchronize().has_value());
    const auto complete = (*event)->isComplete();
    REQUIRE(complete.has_value());
    CHECK(*complete);
}

TEST_CASE("an impossible allocation fails with a diagnosable message", "[gpu]") {
    REQUIRE_GPU_NO_STREAM();

    // Far beyond any card. The message must say how much was asked for and how
    // much was free, since this is the failure the residency planner provokes.
    const auto huge = backend.allocate(MemoryKind::Device, 1ull << 46, "impossible");
    REQUIRE_FALSE(huge.has_value());
    CHECK(huge.error().code() == ErrorCode::OutOfMemory);
    CHECK_THAT(huge.error().message(), Catch::Matchers::ContainsSubstring("impossible"));
    CHECK_THAT(huge.error().message(), Catch::Matchers::ContainsSubstring("free"));

    // The backend stays usable after a failed allocation.
    const auto small = backend.allocate(MemoryKind::Device, 1024, "after-failure");
    REQUIRE(small.has_value());

    // And a kernel launched afterwards must not inherit the failure. CUDA keeps
    // one last-error slot per thread and every launcher reports
    // cudaGetLastError() straight after launching, so an allocation error left
    // unconsumed is picked up by the next kernel and reported against it. That
    // is exactly what happened: a deliberate out-of-memory here surfaced as a
    // spurious rmsNorm failure in an unrelated test file.
    auto stream = backend.createStream("after-failed-allocation");
    REQUIRE(stream.has_value());
    auto input = backend.allocate(MemoryKind::Device, 256 * sizeof(float), "input");
    auto output = backend.allocate(MemoryKind::Device, 256 * sizeof(float), "output");
    REQUIRE(input.has_value());
    REQUIRE(output.has_value());
    REQUIRE(backend.enqueueFill(**stream, **input, 0, 256 * sizeof(float), 0).has_value());

    const auto launched = backend.kernels().scale(
            **stream, ScaleArgs{.input = DeviceView{.buffer = input->get()},
                                .output = DeviceView{.buffer = output->get()},
                                .count = 256,
                                .scalar = 1.0f});
    INFO((launched ? std::string{} : launched.error().toString()));
    CHECK(launched.has_value());
}

TEST_CASE("zero-byte allocations are rejected", "[gpu]") {
    REQUIRE_GPU_NO_STREAM();

    const auto empty = backend.allocate(MemoryKind::Device, 0, "empty");
    REQUIRE_FALSE(empty.has_value());
    CHECK(empty.error().code() == ErrorCode::InvalidArgument);
}

TEST_CASE("every compiled backend can actually be created", "[gpu]") {
    // The list is what callers pick from, so an entry that cannot be
    // constructed is worse than no entry at all.
    for (const BackendKind kind : compiledBackends()) {
        INFO("backend " << toString(kind));
        const auto created = createBackend(kind);
        REQUIRE(created.has_value());
        CHECK_FALSE((*created)->info().name.empty());
    }
}

TEST_CASE("both backends see the same device", "[gpu]") {
    const auto kinds = compiledBackends();
    if (kinds.size() < 2) {
        SKIP("only one backend is compiled in");
    }

    // Not a formality: the kernel tests compare the two against one CPU
    // reference, and that only means anything if they are running on the same
    // silicon.
    auto first = createBackend(kinds[0]);
    auto second = createBackend(kinds[1]);
    REQUIRE(first.has_value());
    REQUIRE(second.has_value());
    CHECK((*first)->info().name == (*second)->info().name);
}

// ---------------------------------------------------------------------------
// Preflight
// ---------------------------------------------------------------------------

TEST_CASE("preflight reports a usable machine as ready", "[gpu][preflight]") {
    REQUIRE_GPU_NO_STREAM();

    PreflightOptions options;
    options.requiredDeviceBytes = 256ull * 1024 * 1024;
    const auto report = runPreflight(options);

    INFO(report.format());
    CHECK(report.canRun());
    CHECK_FALSE(report.checks.empty());
}

TEST_CASE("preflight fails when the memory requirement cannot be met",
          "[gpu][preflight]") {
    REQUIRE_GPU_NO_STREAM();

    PreflightOptions options;
    options.requiredDeviceBytes = 1ull << 45;  // 32 TiB
    const auto report = runPreflight(options);

    INFO(report.format());
    CHECK_FALSE(report.canRun());

    // The failing check must say what to do, not merely that it failed.
    const auto failed = std::ranges::find_if(report.checks, [](const Check& check) {
        return check.status == CheckStatus::Failed;
    });
    REQUIRE(failed != report.checks.end());
    CHECK_FALSE(failed->remediation.empty());
}

TEST_CASE("preflight fails on a missing install and says how to build one",
          "[gpu][preflight]") {
    PreflightOptions options;
    options.installDir = "Z:/definitely/not/here.gturbo";
    const auto report = runPreflight(options);

    INFO(report.format());
    CHECK_FALSE(report.canRun());

    const auto install = std::ranges::find_if(report.checks, [](const Check& check) {
        return check.name == "Model install";
    });
    REQUIRE(install != report.checks.end());
    CHECK(install->status == CheckStatus::Failed);
    CHECK_THAT(install->remediation, Catch::Matchers::ContainsSubstring("tf-repack"));
}

TEST_CASE("the formatted report includes remediation lines", "[gpu][preflight]") {
    PreflightReport report;
    report.checks.push_back(Check{.name = "Something",
                                  .status = CheckStatus::Failed,
                                  .detail = "went wrong",
                                  .remediation = "do this instead"});

    const std::string text = report.format();
    CHECK_THAT(text, Catch::Matchers::ContainsSubstring("FAIL"));
    CHECK_THAT(text, Catch::Matchers::ContainsSubstring("went wrong"));
    CHECK_THAT(text, Catch::Matchers::ContainsSubstring("do this instead"));
    CHECK_FALSE(report.canRun());
}
