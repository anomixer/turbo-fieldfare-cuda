#pragma once

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>

#include <map>
#include <string>

#include "tf/gpu/Backend.h"
#include "tf/gpu/Kernels.h"

/// Shared GPU test fixture, parameterized over every compiled backend.
///
/// This is what makes the D3D12 backend trustworthy: it is checked by exactly
/// the tests that check CUDA, against the same scalar CPU references, rather
/// than by a second suite written to match whatever it happens to do. A kernel
/// that disagrees between backends fails here.
namespace tf::gputest {

/// One backend and a stream on it, created once per backend for the whole run.
/// Creating a device costs hundreds of milliseconds and nothing here mutates
/// global device state.
struct Harness {
    gpu::BackendPtr backend;
    gpu::StreamPtr stream;

    [[nodiscard]] static Harness* instance(gpu::BackendKind kind) {
        static std::map<gpu::BackendKind, Harness> harnesses;

        const auto found = harnesses.find(kind);
        if (found != harnesses.end()) {
            return found->second.backend && found->second.stream ? &found->second : nullptr;
        }

        Harness created;
        auto backend = gpu::createBackend(kind);
        if (backend) {
            created.backend = std::move(*backend);
            auto stream = created.backend->createStream("kernel-tests");
            if (stream) {
                created.stream = std::move(*stream);
            }
        }

        Harness& stored = harnesses.emplace(kind, std::move(created)).first->second;
        return stored.backend && stored.stream ? &stored : nullptr;
    }
};

}  // namespace tf::gputest

/// Runs the enclosing test case once per compiled backend, without a stream.
///
/// The backend name is captured, so a failure says which one produced it -
/// which matters when only one of them is wrong.
#define REQUIRE_GPU_NO_STREAM()                                                    \
    const tf::gpu::BackendKind tfBackendKind = GENERATE(                           \
            Catch::Generators::from_range(tf::gpu::compiledBackends()));           \
    INFO("backend " << tf::gpu::toString(tfBackendKind));                          \
    tf::gputest::Harness* harness = tf::gputest::Harness::instance(tfBackendKind); \
    if (harness == nullptr) {                                                      \
        SKIP("no usable " << tf::gpu::toString(tfBackendKind) << " device");       \
    }                                                                              \
    tf::gpu::IGpuBackend& backend = *harness->backend;                             \
    tf::gpu::IKernels& kernels = backend.kernels();                                \
    static_cast<void>(kernels)

/// The same, plus the shared stream. Most kernel tests want this; the backend
/// tests create their own streams and would collide with it.
#define REQUIRE_GPU()                                    \
    REQUIRE_GPU_NO_STREAM();                             \
    tf::gpu::Stream& stream = *harness->stream;          \
    static_cast<void>(stream)
