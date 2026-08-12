#pragma once

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "tf/core/base/Error.h"
#include "tf/core/base/Types.h"

/// The GPU abstraction the runtime is written against.
///
/// Deliberately narrow. No CUDA type appears in this header, so the D3D12
/// backend planned for M12 is an additional implementation rather than a
/// rewrite of everything above it. Anything that cannot be expressed for both
/// CUDA and D3D12 belongs inside a backend, not here.
///
/// What this layer owns: device discovery, memory, ordered work queues,
/// cross-queue synchronization, transfers and timing. Kernel dispatch is added
/// in M5 as semantic operations (rmsNorm, dequantGemv, ...) rather than a
/// generic "bind these descriptors and launch" call, because the two backends
/// share no useful notion of a pipeline binding.
namespace tf::gpu {

/// Declared in Kernels.h. Forward-declared here so a backend can expose its
/// kernels without this header depending on the operation definitions.
class IKernels;

enum class BackendKind {
    Cuda,
    D3D12,  // M12
};

[[nodiscard]] std::string_view toString(BackendKind kind) noexcept;

/// Where an allocation lives, and who can touch it cheaply.
enum class MemoryKind {
    /// GPU-local. Fastest for the GPU, not addressable by the CPU. Resident
    /// weights, the KV cache and expert slots all live here.
    Device,
    /// Page-locked host memory. The staging ring the expert streamer reads
    /// into: pinned transfers measured 26.49 GiB/s against 17.18 for pageable
    /// on this machine, and only pinned memory can transfer asynchronously.
    PinnedHost,
    /// Host memory the GPU can read directly over PCIe. Useful for small,
    /// write-once data where a staged copy costs more than the slower reads.
    HostVisible,
};

[[nodiscard]] std::string_view toString(MemoryKind kind) noexcept;

struct DeviceInfo {
    std::string name;
    BackendKind backend = BackendKind::Cuda;
    /// Compute capability for CUDA, feature level for D3D12.
    u32 architectureMajor = 0;
    u32 architectureMinor = 0;
    u64 totalMemoryBytes = 0;
    u32 multiprocessorCount = 0;
    /// 32 on every device this project targets. Kernels ported from Metal
    /// simdgroups assume exactly this.
    u32 warpSize = 0;
    u64 sharedMemoryPerBlockBytes = 0;
    /// Transfers can overlap compute when this is non-zero. One engine means
    /// uploads overlap kernels but not downloads.
    u32 asyncEngineCount = 0;
    /// True for integrated GPUs, where device and host memory are the same
    /// physical store and staging copies are pointless.
    bool unifiedMemory = false;
};

struct MemoryInfo {
    u64 freeBytes = 0;
    u64 totalBytes = 0;
};

/// An allocation. Destroying it frees the memory, so lifetime is the caller's.
class Buffer {
public:
    virtual ~Buffer() = default;

    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    [[nodiscard]] u64 size() const noexcept { return size_; }
    [[nodiscard]] MemoryKind kind() const noexcept { return kind_; }

    /// CPU-addressable base pointer, or nullptr for MemoryKind::Device.
    [[nodiscard]] virtual void* hostPointer() const noexcept = 0;

    /// The whole buffer as writable host bytes. Empty for device memory.
    [[nodiscard]] MutableByteSpan hostBytes() const noexcept {
        auto* pointer = static_cast<u8*>(hostPointer());
        return pointer == nullptr ? MutableByteSpan{}
                                  : MutableByteSpan{pointer, static_cast<usize>(size_)};
    }

    [[nodiscard]] std::string_view debugName() const noexcept { return debugName_; }

protected:
    Buffer(u64 size, MemoryKind kind, std::string debugName)
        : size_(size), kind_(kind), debugName_(std::move(debugName)) {}

private:
    u64 size_ = 0;
    MemoryKind kind_ = MemoryKind::Device;
    std::string debugName_;
};

using BufferPtr = std::unique_ptr<Buffer>;

/// An ordered queue. Work submitted to one stream runs in order; work on
/// different streams may overlap and is ordered only through events.
///
/// The decode loop uses three: compute, the shared-expert branch, and expert
/// uploads. That mirrors upstream's cb1/io/cb2 choreography.
class Stream {
public:
    virtual ~Stream() = default;

    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;

    /// Blocks the calling thread until everything queued here has finished.
    [[nodiscard]] virtual Status synchronize() = 0;

    /// True when all queued work has completed, without blocking.
    [[nodiscard]] virtual Result<bool> isIdle() = 0;

    [[nodiscard]] std::string_view debugName() const noexcept { return debugName_; }

protected:
    explicit Stream(std::string debugName) : debugName_(std::move(debugName)) {}

private:
    std::string debugName_;
};

using StreamPtr = std::unique_ptr<Stream>;

/// A marker in a stream, used both to order streams against each other and to
/// time the work between two points.
class Event {
public:
    virtual ~Event() = default;

    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;

    /// Places this event in `stream`. It completes once all prior work in that
    /// stream has completed.
    [[nodiscard]] virtual Status record(Stream& stream) = 0;

    /// Makes `stream` wait for this event before running anything queued after
    /// this call. Does not block the CPU. This is how the routed branch waits
    /// for both the shared-expert kernels and the expert upload.
    [[nodiscard]] virtual Status wait(Stream& stream) = 0;

    /// Blocks the calling thread until this event completes.
    [[nodiscard]] virtual Status synchronize() = 0;

    [[nodiscard]] virtual Result<bool> isComplete() = 0;

    /// Milliseconds between `since` and this event. Both must have been
    /// recorded and completed, and the backend must have been asked for timing
    /// support when the events were created.
    [[nodiscard]] virtual Result<double> elapsedMillisSince(const Event& since) const = 0;

protected:
    // Declaring the copy constructor deleted suppresses the implicit default,
    // which derived backends need.
    Event() = default;
};

using EventPtr = std::unique_ptr<Event>;

class IGpuBackend {
public:
    virtual ~IGpuBackend() = default;

    [[nodiscard]] virtual const DeviceInfo& info() const noexcept = 0;

    /// Free and total device memory right now. The residency planner budgets
    /// against free rather than total: a desktop session already holds around
    /// 1.1 GiB on a 16 GiB card.
    [[nodiscard]] virtual Result<MemoryInfo> memoryInfo() const = 0;

    /// `debugName` surfaces in profilers and in this project's error messages;
    /// it costs nothing at runtime and makes a leak or an overrun findable.
    [[nodiscard]] virtual Result<BufferPtr> allocate(MemoryKind kind, u64 bytes,
                                                     std::string debugName) = 0;

    [[nodiscard]] virtual Result<StreamPtr> createStream(std::string debugName) = 0;

    /// `withTiming` off is cheaper; elapsedMillisSince then fails rather than
    /// returning a meaningless number.
    [[nodiscard]] virtual Result<EventPtr> createEvent(bool withTiming) = 0;

    // ---- Transfers -------------------------------------------------------
    //
    // All enqueue* calls return once the work is queued, not once it has run.
    // Asynchrony against the CPU requires pinned source or destination memory;
    // with pageable memory the driver stages internally and the call blocks.

    [[nodiscard]] virtual Status enqueueUpload(Stream& stream, Buffer& destination,
                                               u64 destinationOffset, ByteSpan source) = 0;

    [[nodiscard]] virtual Status enqueueDownload(Stream& stream, MutableByteSpan destination,
                                                 const Buffer& source, u64 sourceOffset) = 0;

    [[nodiscard]] virtual Status enqueueCopy(Stream& stream, Buffer& destination,
                                             u64 destinationOffset, const Buffer& source,
                                             u64 sourceOffset, u64 bytes) = 0;

    [[nodiscard]] virtual Status enqueueFill(Stream& stream, Buffer& destination,
                                             u64 destinationOffset, u64 bytes, u8 value) = 0;

    /// Blocks until every stream on this device is idle. A debugging and
    /// shutdown tool, not something the decode loop should call.
    [[nodiscard]] virtual Status synchronizeDevice() = 0;

    /// The model operations this backend implements. Owned by the backend and
    /// valid for its lifetime.
    [[nodiscard]] virtual IKernels& kernels() = 0;
};

using BackendPtr = std::unique_ptr<IGpuBackend>;

/// Creates a backend for `deviceIndex`.
///
/// Fails with a message naming what is missing rather than crashing, so a
/// machine with no NVIDIA driver, an outdated one, or no CUDA-capable GPU gets
/// an explanation. See Preflight.h for a check that runs before this.
[[nodiscard]] Result<BackendPtr> createBackend(BackendKind kind, u32 deviceIndex = 0);

/// Backends this build can construct. A build configured without CUDA returns
/// an empty list rather than failing to link.
[[nodiscard]] std::vector<BackendKind> compiledBackends();

}  // namespace tf::gpu
