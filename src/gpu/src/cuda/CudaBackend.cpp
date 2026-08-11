// CUDA implementation of IGpuBackend.
//
// Host-side only: memory, streams, events and transfers. Kernels arrive in M5
// as .cu translation units. Keeping this a .cpp means it builds with MSVC
// directly and does not pay for nvcc on every edit.
//
// Linked against cudart_static, so there is no CUDA runtime DLL to be missing
// at load time. The only external dependency is the NVIDIA driver itself, and
// its absence surfaces here as an ordinary error rather than a loader failure.

#include <cuda_runtime.h>

#include <memory>
#include <utility>

#include "tf/gpu/Backend.h"
#include "tf/gpu/Kernels.h"

namespace tf::gpu {

/// Defined in CudaKernels.cu.
std::unique_ptr<IKernels> createCudaKernels();

namespace {

/// Maps a CUDA status onto the project's error taxonomy. Most failures are
/// simply GpuFailure; the ones singled out are those a caller can act on.
[[nodiscard]] ErrorCode classify(cudaError_t status) noexcept {
    switch (status) {
        case cudaErrorMemoryAllocation:
            return ErrorCode::OutOfMemory;
        case cudaErrorNoDevice:
        case cudaErrorInsufficientDriver:
        case cudaErrorDevicesUnavailable:
            return ErrorCode::Unsupported;
        case cudaErrorInvalidValue:
        case cudaErrorInvalidDevice:
            return ErrorCode::InvalidArgument;
        default:
            return ErrorCode::GpuFailure;
    }
}

[[nodiscard]] Error cudaError(cudaError_t status, std::string_view what) {
    return Error{classify(status), std::format("{}: {} ({})", what,
                                               cudaGetErrorString(status),
                                               cudaGetErrorName(status))};
}

#define TF_CUDA(expr, what)                                    \
    do {                                                       \
        const cudaError_t tf_status = (expr);                  \
        if (tf_status != cudaSuccess) {                        \
            return std::unexpected(cudaError(tf_status, what)); \
        }                                                      \
    } while (false)

// ---------------------------------------------------------------------------
// Buffer
// ---------------------------------------------------------------------------

class CudaBuffer final : public Buffer {
public:
    CudaBuffer(u64 size, MemoryKind kind, std::string debugName, void* devicePointer,
               void* hostPointer)
        : Buffer(size, kind, std::move(debugName)),
          devicePointer_(devicePointer),
          hostPointer_(hostPointer) {}

    ~CudaBuffer() override {
        // Destructors cannot report, and a failed free during teardown is not
        // actionable, so the status is deliberately dropped.
        switch (kind()) {
            case MemoryKind::Device:
                if (devicePointer_ != nullptr) {
                    static_cast<void>(cudaFree(devicePointer_));
                }
                break;
            case MemoryKind::PinnedHost:
            case MemoryKind::HostVisible:
                if (hostPointer_ != nullptr) {
                    static_cast<void>(cudaFreeHost(hostPointer_));
                }
                break;
        }
    }

    [[nodiscard]] void* hostPointer() const noexcept override { return hostPointer_; }

    /// Address the GPU dereferences. For mapped host allocations this is the
    /// device-side alias of the same physical pages, not the host pointer.
    [[nodiscard]] void* devicePointer() const noexcept { return devicePointer_; }

private:
    void* devicePointer_ = nullptr;
    void* hostPointer_ = nullptr;
};

[[nodiscard]] const CudaBuffer& asCuda(const Buffer& buffer) {
    return static_cast<const CudaBuffer&>(buffer);
}

// ---------------------------------------------------------------------------
// Stream
// ---------------------------------------------------------------------------

class CudaStream final : public Stream {
public:
    CudaStream(cudaStream_t stream, std::string debugName)
        : Stream(std::move(debugName)), stream_(stream) {}

    ~CudaStream() override {
        if (stream_ != nullptr) {
            static_cast<void>(cudaStreamDestroy(stream_));
        }
    }

    [[nodiscard]] Status synchronize() override {
        TF_CUDA(cudaStreamSynchronize(stream_), "synchronizing stream");
        return {};
    }

    [[nodiscard]] Result<bool> isIdle() override {
        const cudaError_t status = cudaStreamQuery(stream_);
        if (status == cudaSuccess) {
            return true;
        }
        if (status == cudaErrorNotReady) {
            return false;
        }
        return std::unexpected(cudaError(status, "querying stream"));
    }

    [[nodiscard]] cudaStream_t handle() const noexcept { return stream_; }

private:
    cudaStream_t stream_ = nullptr;
};

[[nodiscard]] cudaStream_t handleOf(Stream& stream) {
    return static_cast<CudaStream&>(stream).handle();
}

// ---------------------------------------------------------------------------
// Event
// ---------------------------------------------------------------------------

class CudaEvent final : public Event {
public:
    CudaEvent(cudaEvent_t event, bool withTiming) : event_(event), withTiming_(withTiming) {}

    ~CudaEvent() override {
        if (event_ != nullptr) {
            static_cast<void>(cudaEventDestroy(event_));
        }
    }

    [[nodiscard]] Status record(Stream& stream) override {
        TF_CUDA(cudaEventRecord(event_, handleOf(stream)), "recording event");
        return {};
    }

    [[nodiscard]] Status wait(Stream& stream) override {
        // Orders one stream behind another without stalling the CPU, which is
        // what lets the decode loop keep encoding while the GPU catches up.
        TF_CUDA(cudaStreamWaitEvent(handleOf(stream), event_, 0), "waiting on event");
        return {};
    }

    [[nodiscard]] Status synchronize() override {
        TF_CUDA(cudaEventSynchronize(event_), "synchronizing event");
        return {};
    }

    [[nodiscard]] Result<bool> isComplete() override {
        const cudaError_t status = cudaEventQuery(event_);
        if (status == cudaSuccess) {
            return true;
        }
        if (status == cudaErrorNotReady) {
            return false;
        }
        return std::unexpected(cudaError(status, "querying event"));
    }

    [[nodiscard]] Result<double> elapsedMillisSince(const Event& since) const override {
        const auto& other = static_cast<const CudaEvent&>(since);
        if (!withTiming_ || !other.withTiming_) {
            return makeError(ErrorCode::InvalidArgument,
                             "elapsedMillisSince needs both events created with timing");
        }
        float milliseconds = 0.0f;
        TF_CUDA(cudaEventElapsedTime(&milliseconds, other.event_, event_),
                "measuring elapsed time");
        return static_cast<double>(milliseconds);
    }

private:
    cudaEvent_t event_ = nullptr;
    bool withTiming_ = false;
};

// ---------------------------------------------------------------------------
// Backend
// ---------------------------------------------------------------------------

class CudaBackend final : public IGpuBackend {
public:
    explicit CudaBackend(u32 deviceIndex, DeviceInfo info)
        : deviceIndex_(deviceIndex), info_(std::move(info)), kernels_(createCudaKernels()) {}

    ~CudaBackend() override = default;

    [[nodiscard]] const DeviceInfo& info() const noexcept override { return info_; }

    [[nodiscard]] IKernels& kernels() override { return *kernels_; }

    [[nodiscard]] Result<MemoryInfo> memoryInfo() const override {
        TF_CUDA(cudaSetDevice(static_cast<int>(deviceIndex_)), "selecting device");
        usize free = 0;
        usize total = 0;
        TF_CUDA(cudaMemGetInfo(&free, &total), "querying memory");
        return MemoryInfo{.freeBytes = free, .totalBytes = total};
    }

    [[nodiscard]] Result<BufferPtr> allocate(MemoryKind kind, u64 bytes,
                                             std::string debugName) override {
        if (bytes == 0) {
            return makeError(ErrorCode::InvalidArgument, "{}: zero-byte allocation",
                             debugName);
        }
        TF_CUDA(cudaSetDevice(static_cast<int>(deviceIndex_)), "selecting device");

        void* devicePointer = nullptr;
        void* hostPointer = nullptr;

        switch (kind) {
            case MemoryKind::Device: {
                const cudaError_t status = cudaMalloc(&devicePointer, bytes);
                if (status != cudaSuccess) {
                    // Consume the error now that it has been handled. CUDA
                    // keeps one last-error slot per thread, and every kernel
                    // launcher reports cudaGetLastError() straight after
                    // launching - so an allocation failure left sitting there
                    // is picked up by the next kernel and blamed on it. A
                    // recoverable out-of-memory would surface as a spurious
                    // failure in whatever ran next.
                    static_cast<void>(cudaGetLastError());

                    // Report the shortfall: this is the failure the residency
                    // planner is most likely to provoke, and knowing how much
                    // was free is what makes it diagnosable.
                    usize free = 0;
                    usize total = 0;
                    static_cast<void>(cudaMemGetInfo(&free, &total));
                    return makeError(classify(status),
                                     "{}: allocating {:.1f} MiB of device memory failed "
                                     "({:.1f} MiB free of {:.1f} MiB) - {}",
                                     debugName,
                                     static_cast<double>(bytes) / (1024.0 * 1024.0),
                                     static_cast<double>(free) / (1024.0 * 1024.0),
                                     static_cast<double>(total) / (1024.0 * 1024.0),
                                     cudaGetErrorString(status));
                }
                break;
            }
            case MemoryKind::PinnedHost:
                TF_CUDA(cudaHostAlloc(&hostPointer, bytes, cudaHostAllocDefault),
                        std::format("{}: allocating pinned host memory", debugName));
                break;
            case MemoryKind::HostVisible:
                // Mapped: the GPU reads these pages directly across PCIe.
                TF_CUDA(cudaHostAlloc(&hostPointer, bytes, cudaHostAllocMapped),
                        std::format("{}: allocating mapped host memory", debugName));
                TF_CUDA(cudaHostGetDevicePointer(&devicePointer, hostPointer, 0),
                        std::format("{}: mapping host memory to the device", debugName));
                break;
        }

        return std::make_unique<CudaBuffer>(bytes, kind, std::move(debugName), devicePointer,
                                            hostPointer);
    }

    [[nodiscard]] Result<StreamPtr> createStream(std::string debugName) override {
        TF_CUDA(cudaSetDevice(static_cast<int>(deviceIndex_)), "selecting device");
        cudaStream_t stream = nullptr;
        // Non-blocking: work here must not be implicitly serialized against the
        // legacy default stream, or the three-stream decode overlap collapses.
        TF_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                std::format("creating stream '{}'", debugName));
        return std::make_unique<CudaStream>(stream, std::move(debugName));
    }

    [[nodiscard]] Result<EventPtr> createEvent(bool withTiming) override {
        TF_CUDA(cudaSetDevice(static_cast<int>(deviceIndex_)), "selecting device");
        cudaEvent_t event = nullptr;
        TF_CUDA(cudaEventCreateWithFlags(
                        &event, withTiming ? cudaEventDefault : cudaEventDisableTiming),
                "creating event");
        return std::make_unique<CudaEvent>(event, withTiming);
    }

    [[nodiscard]] Status enqueueUpload(Stream& stream, Buffer& destination,
                                       u64 destinationOffset, ByteSpan source) override {
        TF_CHECK(checkRange(destination, destinationOffset, source.size(), "upload"));
        auto* base = static_cast<u8*>(asCuda(destination).devicePointer());
        if (base == nullptr) {
            return makeError(ErrorCode::InvalidArgument,
                             "upload destination '{}' is not device-addressable",
                             destination.debugName());
        }
        TF_CUDA(cudaMemcpyAsync(base + destinationOffset, source.data(), source.size(),
                                cudaMemcpyHostToDevice, handleOf(stream)),
                "enqueuing upload");
        return {};
    }

    [[nodiscard]] Status enqueueDownload(Stream& stream, MutableByteSpan destination,
                                         const Buffer& source, u64 sourceOffset) override {
        TF_CHECK(checkRange(source, sourceOffset, destination.size(), "download"));
        const auto* base = static_cast<const u8*>(asCuda(source).devicePointer());
        if (base == nullptr) {
            return makeError(ErrorCode::InvalidArgument,
                             "download source '{}' is not device-addressable",
                             source.debugName());
        }
        TF_CUDA(cudaMemcpyAsync(destination.data(), base + sourceOffset, destination.size(),
                                cudaMemcpyDeviceToHost, handleOf(stream)),
                "enqueuing download");
        return {};
    }

    [[nodiscard]] Status enqueueCopy(Stream& stream, Buffer& destination,
                                     u64 destinationOffset, const Buffer& source,
                                     u64 sourceOffset, u64 bytes) override {
        TF_CHECK(checkRange(destination, destinationOffset, bytes, "copy destination"));
        TF_CHECK(checkRange(source, sourceOffset, bytes, "copy source"));

        auto* destinationBase = static_cast<u8*>(asCuda(destination).devicePointer());
        const auto* sourceBase = static_cast<const u8*>(asCuda(source).devicePointer());
        if (destinationBase == nullptr || sourceBase == nullptr) {
            return makeError(ErrorCode::InvalidArgument,
                             "device-to-device copy needs both buffers device-addressable");
        }
        TF_CUDA(cudaMemcpyAsync(destinationBase + destinationOffset, sourceBase + sourceOffset,
                                bytes, cudaMemcpyDeviceToDevice, handleOf(stream)),
                "enqueuing copy");
        return {};
    }

    [[nodiscard]] Status enqueueFill(Stream& stream, Buffer& destination,
                                     u64 destinationOffset, u64 bytes, u8 value) override {
        TF_CHECK(checkRange(destination, destinationOffset, bytes, "fill"));
        auto* base = static_cast<u8*>(asCuda(destination).devicePointer());
        if (base == nullptr) {
            return makeError(ErrorCode::InvalidArgument,
                             "fill destination '{}' is not device-addressable",
                             destination.debugName());
        }
        TF_CUDA(cudaMemsetAsync(base + destinationOffset, value, bytes, handleOf(stream)),
                "enqueuing fill");
        return {};
    }

    [[nodiscard]] Status synchronizeDevice() override {
        TF_CUDA(cudaSetDevice(static_cast<int>(deviceIndex_)), "selecting device");
        TF_CUDA(cudaDeviceSynchronize(), "synchronizing device");
        return {};
    }

private:
    /// Bounds-checks before handing an offset to CUDA, which would otherwise
    /// either corrupt neighbouring allocations or fail with a bare
    /// "invalid argument" naming nothing.
    [[nodiscard]] static Status checkRange(const Buffer& buffer, u64 offset, u64 bytes,
                                           std::string_view what) {
        if (offset > buffer.size() || bytes > buffer.size() - offset) {
            return makeError(ErrorCode::InvalidArgument,
                             "{}: range [{}, {}) exceeds buffer '{}' of {} bytes", what,
                             offset, offset + bytes, buffer.debugName(), buffer.size());
        }
        return {};
    }

    u32 deviceIndex_ = 0;
    DeviceInfo info_;
    std::unique_ptr<IKernels> kernels_;
};

}  // namespace

// Accessors the kernel translation unit uses to reach the underlying CUDA
// handles. Defined here so CudaBuffer and CudaStream stay private to this file.
void* cudaDevicePointer(const Buffer& buffer) {
    return asCuda(buffer).devicePointer();
}

cudaStream_t cudaStreamOf(Stream& stream) { return handleOf(stream); }

Result<BackendPtr> createCudaBackend(u32 deviceIndex) {
    int deviceCount = 0;
    const cudaError_t countStatus = cudaGetDeviceCount(&deviceCount);
    if (countStatus != cudaSuccess) {
        return std::unexpected(cudaError(countStatus, "enumerating CUDA devices"));
    }
    if (deviceCount == 0) {
        return makeError(ErrorCode::Unsupported, "no CUDA-capable device is present");
    }
    if (deviceIndex >= static_cast<u32>(deviceCount)) {
        return makeError(ErrorCode::InvalidArgument,
                         "CUDA device {} requested but only {} are present", deviceIndex,
                         deviceCount);
    }

    TF_CUDA(cudaSetDevice(static_cast<int>(deviceIndex)), "selecting device");

    cudaDeviceProp properties{};
    TF_CUDA(cudaGetDeviceProperties(&properties, static_cast<int>(deviceIndex)),
            "querying device properties");

    usize free = 0;
    usize total = 0;
    TF_CUDA(cudaMemGetInfo(&free, &total), "querying device memory");

    DeviceInfo info;
    info.name = properties.name;
    info.backend = BackendKind::Cuda;
    info.architectureMajor = static_cast<u32>(properties.major);
    info.architectureMinor = static_cast<u32>(properties.minor);
    info.totalMemoryBytes = total;
    info.multiprocessorCount = static_cast<u32>(properties.multiProcessorCount);
    info.warpSize = static_cast<u32>(properties.warpSize);
    info.sharedMemoryPerBlockBytes = properties.sharedMemPerBlock;
    info.asyncEngineCount = static_cast<u32>(properties.asyncEngineCount);
    info.unifiedMemory = properties.integrated != 0;

    return std::make_unique<CudaBackend>(deviceIndex, std::move(info));
}

}  // namespace tf::gpu
