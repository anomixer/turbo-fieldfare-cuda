// The D3D12 backend: device, memory, streams and transfers.
//
// Kernel dispatch lives in D3D12Kernels.cpp; this file is everything the
// runtime needs that is not a shader.
//
// Two things differ from CUDA in ways the interface has to paper over:
//
//   Command lists, not streams. D3D12 records into a list and executes it as a
//   unit, so a Stream here buffers work and submits it at synchronize(). The
//   runtime already synchronizes at the points that matter - the router
//   readback, the end of a step - so the observable ordering is the same.
//
//   Readbacks are two-step. A device-to-host copy lands in a readback resource,
//   and only after the GPU has finished can it be memcpy'd to the caller. Those
//   deferred copies are queued per stream and flushed on synchronize, which is
//   exactly where the caller was going to wait anyway.

#include <algorithm>
#include <vector>

#include "D3D12Common.h"
#include "tf/gpu/Backend.h"
#include "tf/gpu/Kernels.h"

namespace tf::gpu::d3d12 {

Error hresultError(HRESULT result, std::string_view what) {
    char* buffer = nullptr;
    const DWORD length = ::FormatMessageA(
            FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
                    FORMAT_MESSAGE_IGNORE_INSERTS,
            nullptr, static_cast<DWORD>(result), 0, reinterpret_cast<char*>(&buffer), 0,
            nullptr);

    std::string detail{buffer != nullptr ? buffer : "", length};
    if (buffer != nullptr) {
        ::LocalFree(buffer);
    }
    while (!detail.empty() && (detail.back() == '\n' || detail.back() == '\r')) {
        detail.pop_back();
    }
    if (detail.empty()) {
        detail = std::format("HRESULT {:#010x}", static_cast<u32>(result));
    }

    ErrorCode kind = ErrorCode::GpuFailure;
    if (result == E_OUTOFMEMORY || result == DXGI_ERROR_DEVICE_REMOVED) {
        kind = result == E_OUTOFMEMORY ? ErrorCode::OutOfMemory : ErrorCode::GpuFailure;
    }
    return Error{kind, std::format("{}: {}", what, detail)};
}

namespace {

[[nodiscard]] std::string narrow(std::wstring_view text) {
    if (text.empty()) {
        return {};
    }
    const int needed = ::WideCharToMultiByte(CP_UTF8, 0, text.data(),
                                             static_cast<int>(text.size()), nullptr, 0, nullptr,
                                             nullptr);
    std::string out(static_cast<usize>(needed), '\0');
    ::WideCharToMultiByte(CP_UTF8, 0, text.data(), static_cast<int>(text.size()), out.data(),
                          needed, nullptr, nullptr);
    return out;
}

[[nodiscard]] std::wstring widen(std::string_view text) {
    if (text.empty()) {
        return {};
    }
    const int needed = ::MultiByteToWideChar(CP_UTF8, 0, text.data(),
                                             static_cast<int>(text.size()), nullptr, 0);
    std::wstring out(static_cast<usize>(needed), L'\0');
    ::MultiByteToWideChar(CP_UTF8, 0, text.data(), static_cast<int>(text.size()), out.data(),
                          needed);
    return out;
}

}  // namespace

// ---------------------------------------------------------------------------
// Buffer
// ---------------------------------------------------------------------------

class D3D12Buffer final : public Buffer {
public:
    D3D12Buffer(u64 bytes, MemoryKind kind, std::string debugName)
        : Buffer(bytes, kind, std::move(debugName)) {}

    ~D3D12Buffer() override {
        if (mapped_ != nullptr) {
            resource_->Unmap(0, nullptr);
        }
    }

    [[nodiscard]] void* hostPointer() const noexcept override { return mapped_; }

    [[nodiscard]] ID3D12Resource* resource() const noexcept { return resource_.get(); }
    [[nodiscard]] D3D12_GPU_VIRTUAL_ADDRESS address() const {
        return resource_->GetGPUVirtualAddress();
    }
    [[nodiscard]] D3D12_RESOURCE_STATES state() const noexcept { return state_; }
    void setState(D3D12_RESOURCE_STATES state) noexcept { state_ = state; }

    ComPtr<ID3D12Resource> resource_;
    void* mapped_ = nullptr;
    D3D12_RESOURCE_STATES state_ = D3D12_RESOURCE_STATE_COMMON;
};

// ---------------------------------------------------------------------------
// Stream
// ---------------------------------------------------------------------------

class D3D12Backend;

/// A recorded batch of work.
///
/// Everything enqueued goes into one command list, which is executed when the
/// caller synchronizes or when an event is recorded on it. The queue is shared
/// with every other stream on the device: D3D12 compute queues do not overlap
/// the way CUDA streams do, and pretending otherwise would only move the
/// serialization somewhere less visible.
class D3D12Stream final : public Stream {
public:
    explicit D3D12Stream(std::string debugName) : Stream(std::move(debugName)) {}

    [[nodiscard]] Status synchronize() override;
    [[nodiscard]] Result<bool> isIdle() override;

    /// Copies queued to run after the GPU finishes, because their source is a
    /// readback resource the GPU is still writing.
    struct PendingReadback {
        MutableByteSpan destination;
        ComPtr<ID3D12Resource> staging;
        u64 bytes = 0;
    };

    [[nodiscard]] Status begin();
    [[nodiscard]] Status flush();

    D3D12Backend* backend = nullptr;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
    std::vector<PendingReadback> readbacks;
    /// Resources kept alive until the batch completes.
    std::vector<ComPtr<ID3D12Resource>> retained;
    bool recording = false;
    u64 lastSubmitted = 0;
};

// ---------------------------------------------------------------------------
// Event
// ---------------------------------------------------------------------------

class D3D12Event final : public Event {
public:
    D3D12Event() = default;

    [[nodiscard]] Status record(Stream& stream) override;
    [[nodiscard]] Status wait(Stream& stream) override;
    [[nodiscard]] Status synchronize() override;
    [[nodiscard]] Result<bool> isComplete() override;
    [[nodiscard]] Result<double> elapsedMillisSince(const Event& since) const override;

    D3D12Backend* backend = nullptr;
    u64 value = 0;
    /// Slot in the backend's timestamp heap, or -1 when this event was created
    /// without timing and has no place to write one.
    i32 timestampSlot = -1;
};

/// Timestamp query slots.
///
/// D3D12 has no equivalent of a CUDA event that carries its own time, so timed
/// events write into a shared query heap and read back the resolved ticks. A
/// fixed pool because the runtime creates a handful of events for the lifetime
/// of a process, not one per step.
constexpr u32 kTimestampSlots = 256;

// ---------------------------------------------------------------------------
// Backend
// ---------------------------------------------------------------------------

class D3D12Backend final : public IGpuBackend {
public:
    [[nodiscard]] static Result<std::unique_ptr<D3D12Backend>> create(u32 deviceIndex);

    ~D3D12Backend() override {
        if (fenceEvent_ != nullptr) {
            ::CloseHandle(fenceEvent_);
        }
    }

    [[nodiscard]] const DeviceInfo& info() const noexcept override { return info_; }
    [[nodiscard]] Result<MemoryInfo> memoryInfo() const override;

    [[nodiscard]] Result<BufferPtr> allocate(MemoryKind kind, u64 bytes,
                                             std::string debugName) override;
    [[nodiscard]] Result<StreamPtr> createStream(std::string debugName) override;
    [[nodiscard]] Result<EventPtr> createEvent(bool withTiming) override;

    [[nodiscard]] Status enqueueUpload(Stream& stream, Buffer& destination, u64 offset,
                                       ByteSpan source) override;
    [[nodiscard]] Status enqueueDownload(Stream& stream, MutableByteSpan destination,
                                         const Buffer& source, u64 offset) override;
    [[nodiscard]] Status enqueueCopy(Stream& stream, Buffer& destination, u64 destinationOffset,
                                     const Buffer& source, u64 sourceOffset, u64 bytes) override;
    [[nodiscard]] Status enqueueFill(Stream& stream, Buffer& destination, u64 offset, u64 bytes,
                                     u8 value) override;
    [[nodiscard]] Status synchronizeDevice() override;
    [[nodiscard]] IKernels& kernels() override { return *kernels_; }

    /// Submits a stream's list and returns the fence value that marks it done.
    [[nodiscard]] Result<u64> submit(D3D12Stream& stream);
    [[nodiscard]] Status waitForFence(u64 value);
    [[nodiscard]] Result<bool> fenceReached(u64 value) const;

    /// Barriers a buffer into the state an operation needs. D3D12 requires
    /// this where CUDA has no equivalent concept at all.
    void transition(D3D12Stream& stream, D3D12Buffer& buffer, D3D12_RESOURCE_STATES state);

    [[nodiscard]] ID3D12Device* device() const noexcept { return device_.get(); }
    [[nodiscard]] ID3D12RootSignature* rootSignature() const noexcept {
        return rootSignature_.get();
    }
    [[nodiscard]] ID3D12PipelineState* pipeline(ShaderId id) const noexcept {
        return pipelines_[static_cast<usize>(id)].get();
    }

    /// Staging buffers for uploads, kept alive until the batch completes.
    [[nodiscard]] Result<ComPtr<ID3D12Resource>> createStaging(u64 bytes, bool readback);

    /// Claims a slot in the timestamp heap. Wraps once exhausted, which is
    /// harmless: an event that outlives 256 later events was not going to give
    /// a meaningful interval anyway.
    [[nodiscard]] i32 claimTimestampSlot() {
        if (!timestampHeap_) {
            return -1;
        }
        return static_cast<i32>(nextTimestampSlot_++ % kTimestampSlots);
    }

    /// Records a timestamp into `slot` and resolves it so the value is
    /// readable once the queue reaches this point.
    void writeTimestamp(D3D12Stream& stream, i32 slot);

    [[nodiscard]] Result<double> timestampMillisBetween(i32 first, i32 second) const;

private:
    [[nodiscard]] Status buildRootSignature();
    [[nodiscard]] Status buildPipelines();

    ComPtr<IDXGIAdapter3> adapter_;
    ComPtr<ID3D12Device> device_;
    ComPtr<ID3D12CommandQueue> queue_;
    ComPtr<ID3D12Fence> fence_;
    ComPtr<ID3D12RootSignature> rootSignature_;
    ComPtr<ID3D12PipelineState> pipelines_[static_cast<usize>(ShaderId::Count)];
    HANDLE fenceEvent_ = nullptr;
    u64 nextFenceValue_ = 1;

    ComPtr<ID3D12QueryHeap> timestampHeap_;
    ComPtr<ID3D12Resource> timestampReadback_;
    u64 timestampFrequency_ = 0;
    u32 nextTimestampSlot_ = 0;

    DeviceInfo info_;
    std::unique_ptr<IKernels> kernels_;

    friend Result<std::unique_ptr<IKernels>> createD3D12Kernels(D3D12Backend& backend);
};

/// Defined in D3D12Kernels.cpp.
[[nodiscard]] Result<std::unique_ptr<IKernels>> createD3D12Kernels(D3D12Backend& backend);

Result<MemoryInfo> D3D12Backend::memoryInfo() const {
    DXGI_QUERY_VIDEO_MEMORY_INFO local{};
    TF_D3D(adapter_->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &local),
           "querying video memory");

    // Budget is what the OS is willing to give this process, which is the
    // figure the residency planner should respect - not the card's total.
    const u64 used = local.CurrentUsage;
    const u64 budget = local.Budget;
    return MemoryInfo{.freeBytes = budget > used ? budget - used : 0,
                      .totalBytes = info_.totalMemoryBytes};
}

Result<ComPtr<ID3D12Resource>> D3D12Backend::createStaging(u64 bytes, bool readback) {
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = readback ? D3D12_HEAP_TYPE_READBACK : D3D12_HEAP_TYPE_UPLOAD;

    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = bytes;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = DXGI_FORMAT_UNKNOWN;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

    ComPtr<ID3D12Resource> resource;
    TF_D3D(device_->CreateCommittedResource(
                   &heap, D3D12_HEAP_FLAG_NONE, &desc,
                   readback ? D3D12_RESOURCE_STATE_COPY_DEST
                            : D3D12_RESOURCE_STATE_GENERIC_READ,
                   nullptr, IID_PPV_ARGS(resource.put())),
           "creating a staging buffer");
    return resource;
}

Result<BufferPtr> D3D12Backend::allocate(MemoryKind kind, u64 bytes, std::string debugName) {
    if (bytes == 0) {
        return makeError(ErrorCode::InvalidArgument, "{}: a zero-byte allocation", debugName);
    }

    // A request larger than the card has cannot be satisfied, and D3D12
    // reports it as E_INVALIDARG rather than E_OUTOFMEMORY - which would
    // surface as "the driver rejected this" instead of "it does not fit".
    if (kind == MemoryKind::Device && bytes > info_.totalMemoryBytes) {
        DXGI_QUERY_VIDEO_MEMORY_INFO local{};
        static_cast<void>(
                adapter_->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &local));
        return makeError(ErrorCode::OutOfMemory,
                         "{}: allocating {:.1f} MiB of device memory failed ({:.1f} MiB free "
                         "of {:.1f} MiB total)",
                         debugName, static_cast<double>(bytes) / (1024.0 * 1024.0),
                         static_cast<double>(local.Budget > local.CurrentUsage
                                                     ? local.Budget - local.CurrentUsage
                                                     : 0) /
                                 (1024.0 * 1024.0),
                         static_cast<double>(info_.totalMemoryBytes) / (1024.0 * 1024.0));
    }

    auto buffer = std::make_unique<D3D12Buffer>(bytes, kind, std::move(debugName));

    D3D12_HEAP_PROPERTIES heap{};
    D3D12_RESOURCE_STATES initial = D3D12_RESOURCE_STATE_COMMON;
    D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE;

    switch (kind) {
        case MemoryKind::Device:
            heap.Type = D3D12_HEAP_TYPE_DEFAULT;
            initial = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
            break;
        case MemoryKind::PinnedHost:
        case MemoryKind::HostVisible:
            // Both map to an upload heap: write-combined system memory the GPU
            // reads over PCIe, which is what pinned host memory is for here.
            heap.Type = D3D12_HEAP_TYPE_UPLOAD;
            initial = D3D12_RESOURCE_STATE_GENERIC_READ;
            break;
    }

    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = bytes;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = DXGI_FORMAT_UNKNOWN;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.Flags = flags;

    const HRESULT created = device_->CreateCommittedResource(
            &heap, D3D12_HEAP_FLAG_NONE, &desc, initial, nullptr,
            IID_PPV_ARGS(buffer->resource_.put()));
    if (FAILED(created)) {
        DXGI_QUERY_VIDEO_MEMORY_INFO local{};
        static_cast<void>(
                adapter_->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &local));
        return makeError(created == E_OUTOFMEMORY ? ErrorCode::OutOfMemory
                                                  : ErrorCode::GpuFailure,
                         "{}: allocating {:.1f} MiB of device memory failed ({:.1f} MiB of "
                         "budget free)",
                         buffer->debugName(),
                         static_cast<double>(bytes) / (1024.0 * 1024.0),
                         static_cast<double>(local.Budget > local.CurrentUsage
                                                     ? local.Budget - local.CurrentUsage
                                                     : 0) /
                                 (1024.0 * 1024.0));
    }

    buffer->setState(initial);
    if (!buffer->debugName().empty()) {
        buffer->resource_->SetName(widen(buffer->debugName()).c_str());
    }

    if (kind != MemoryKind::Device) {
        const D3D12_RANGE nothingRead{0, 0};
        TF_D3D(buffer->resource_->Map(0, &nothingRead, &buffer->mapped_),
               "mapping host-visible memory");
    }

    return BufferPtr{buffer.release()};
}

Result<StreamPtr> D3D12Backend::createStream(std::string debugName) {
    auto stream = std::make_unique<D3D12Stream>(std::move(debugName));
    stream->backend = this;

    TF_D3D(device_->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                           IID_PPV_ARGS(stream->allocator.put())),
           "creating a command allocator");
    TF_D3D(device_->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
                                      stream->allocator.get(), nullptr,
                                      IID_PPV_ARGS(stream->list.put())),
           "creating a command list");
    stream->recording = true;
    return StreamPtr{stream.release()};
}

Result<EventPtr> D3D12Backend::createEvent(bool withTiming) {
    auto event = std::make_unique<D3D12Event>();
    event->backend = this;
    if (withTiming) {
        event->timestampSlot = claimTimestampSlot();
    }
    return EventPtr{event.release()};
}

void D3D12Backend::transition(D3D12Stream& stream, D3D12Buffer& buffer,
                              D3D12_RESOURCE_STATES state) {
    if (buffer.state() == state || buffer.kind() != MemoryKind::Device) {
        // Upload-heap resources live permanently in GENERIC_READ and cannot be
        // transitioned.
        return;
    }
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = buffer.resource();
    barrier.Transition.StateBefore = buffer.state();
    barrier.Transition.StateAfter = state;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    stream.list->ResourceBarrier(1, &barrier);
    buffer.setState(state);
}

Result<u64> D3D12Backend::submit(D3D12Stream& stream) {
    if (!stream.recording) {
        return stream.lastSubmitted;
    }
    TF_D3D(stream.list->Close(), "closing the command list");

    ID3D12CommandList* lists[] = {stream.list.get()};
    queue_->ExecuteCommandLists(1, lists);

    const u64 value = nextFenceValue_++;
    TF_D3D(queue_->Signal(fence_.get(), value), "signalling the fence");
    stream.lastSubmitted = value;
    stream.recording = false;
    return value;
}

Status D3D12Backend::waitForFence(u64 value) {
    if (fence_->GetCompletedValue() >= value) {
        return {};
    }
    TF_D3D(fence_->SetEventOnCompletion(value, fenceEvent_), "arming the fence event");
    ::WaitForSingleObject(fenceEvent_, INFINITE);
    return {};
}

Result<bool> D3D12Backend::fenceReached(u64 value) const {
    return fence_->GetCompletedValue() >= value;
}

void D3D12Backend::writeTimestamp(D3D12Stream& stream, i32 slot) {
    if (slot < 0 || !timestampHeap_) {
        return;
    }
    if (!stream.begin()) {
        return;
    }
    const auto index = static_cast<UINT>(slot);
    stream.list->EndQuery(timestampHeap_.get(), D3D12_QUERY_TYPE_TIMESTAMP, index);
    // Resolved straight away rather than in a batch: the slots are read
    // individually and by the time anyone asks, the copy has to have happened.
    stream.list->ResolveQueryData(timestampHeap_.get(), D3D12_QUERY_TYPE_TIMESTAMP, index, 1,
                                  timestampReadback_.get(), index * sizeof(u64));
}

Result<double> D3D12Backend::timestampMillisBetween(i32 first, i32 second) const {
    if (!timestampReadback_ || timestampFrequency_ == 0) {
        return makeError(ErrorCode::Unsupported, "this device does not report timestamps");
    }

    void* mapped = nullptr;
    const D3D12_RANGE whole{0, kTimestampSlots * sizeof(u64)};
    TF_D3D(timestampReadback_->Map(0, &whole, &mapped), "mapping the timestamp readback");

    const auto* ticks = static_cast<const u64*>(mapped);
    const u64 start = ticks[static_cast<usize>(first)];
    const u64 stop = ticks[static_cast<usize>(second)];

    const D3D12_RANGE nothingWritten{0, 0};
    timestampReadback_->Unmap(0, &nothingWritten);

    if (stop < start) {
        return makeError(ErrorCode::InvalidArgument,
                         "the later event was recorded before the earlier one");
    }
    return static_cast<double>(stop - start) * 1000.0 /
           static_cast<double>(timestampFrequency_);
}

Status D3D12Backend::synchronizeDevice() {
    const u64 value = nextFenceValue_++;
    TF_D3D(queue_->Signal(fence_.get(), value), "signalling the fence");
    return waitForFence(value);
}

Status D3D12Stream::begin() {
    if (recording) {
        return {};
    }
    TF_D3D(allocator->Reset(), "resetting the command allocator");
    TF_D3D(list->Reset(allocator.get(), nullptr), "resetting the command list");
    recording = true;
    return {};
}

Status D3D12Stream::flush() {
    TF_TRY(const u64 value, backend->submit(*this));
    TF_CHECK(backend->waitForFence(value));

    // Only now are the readback resources safe to read: the copies that filled
    // them have completed.
    for (PendingReadback& pending : readbacks) {
        void* mapped = nullptr;
        const D3D12_RANGE whole{0, static_cast<SIZE_T>(pending.bytes)};
        TF_D3D(pending.staging->Map(0, &whole, &mapped), "mapping a readback buffer");
        std::memcpy(pending.destination.data(), mapped, static_cast<usize>(pending.bytes));
        const D3D12_RANGE nothingWritten{0, 0};
        pending.staging->Unmap(0, &nothingWritten);
    }
    readbacks.clear();
    retained.clear();
    return begin();
}

Status D3D12Stream::synchronize() { return flush(); }

Result<bool> D3D12Stream::isIdle() {
    if (recording && lastSubmitted == 0) {
        return true;
    }
    return backend->fenceReached(lastSubmitted);
}

Status D3D12Event::record(Stream& stream) {
    auto& typed = static_cast<D3D12Stream&>(stream);
    // The timestamp is written before the submit, so it lands in the same batch
    // and marks this point in the queue rather than the next one.
    backend->writeTimestamp(typed, timestampSlot);
    // Submitting here rather than only inserting a marker: the value has to
    // correspond to work the queue has actually been given, or a wait on it
    // would return before the work it names was even recorded.
    TF_TRY(value, backend->submit(typed));
    return typed.begin();
}

Status D3D12Event::wait(Stream& stream) {
    auto& typed = static_cast<D3D12Stream&>(stream);
    // One queue serves every stream, so work recorded after this point already
    // follows the signalled work. Flushing keeps the ordering explicit rather
    // than relying on that.
    TF_CHECK(backend->waitForFence(value));
    static_cast<void>(typed);
    return {};
}

Status D3D12Event::synchronize() { return backend->waitForFence(value); }

Result<bool> D3D12Event::isComplete() { return backend->fenceReached(value); }

Result<double> D3D12Event::elapsedMillisSince(const Event& since) const {
    const auto& other = static_cast<const D3D12Event&>(since);
    if (timestampSlot < 0 || other.timestampSlot < 0) {
        return makeError(ErrorCode::InvalidArgument,
                         "both events must have been created with timing enabled");
    }
    return backend->timestampMillisBetween(other.timestampSlot, timestampSlot);
}

// ---------------------------------------------------------------------------
// Transfers
// ---------------------------------------------------------------------------

namespace {

[[nodiscard]] Status checkRange(const Buffer& buffer, u64 offset, u64 bytes,
                                std::string_view what) {
    if (offset > buffer.size() || bytes > buffer.size() - offset) {
        return makeError(ErrorCode::InvalidArgument,
                         "{}: [{}, {}) runs past buffer '{}' of {} bytes", what, offset,
                         offset + bytes, buffer.debugName(), buffer.size());
    }
    return {};
}

}  // namespace

Status D3D12Backend::enqueueUpload(Stream& stream, Buffer& destination, u64 offset,
                                   ByteSpan source) {
    TF_CHECK(checkRange(destination, offset, source.size(), "upload"));
    if (source.empty()) {
        return {};
    }

    auto& typed = static_cast<D3D12Stream&>(stream);
    auto& target = static_cast<D3D12Buffer&>(destination);
    TF_CHECK(typed.begin());

    if (target.kind() != MemoryKind::Device) {
        std::memcpy(static_cast<u8*>(target.hostPointer()) + offset, source.data(),
                    source.size());
        return {};
    }

    TF_TRY(ComPtr<ID3D12Resource> staging, createStaging(source.size(), /*readback=*/false));
    void* mapped = nullptr;
    const D3D12_RANGE nothingRead{0, 0};
    TF_D3D(staging->Map(0, &nothingRead, &mapped), "mapping an upload buffer");
    std::memcpy(mapped, source.data(), source.size());
    staging->Unmap(0, nullptr);

    transition(typed, target, D3D12_RESOURCE_STATE_COPY_DEST);
    typed.list->CopyBufferRegion(target.resource(), offset, staging.get(), 0, source.size());
    transition(typed, target, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    typed.retained.push_back(std::move(staging));
    return {};
}

Status D3D12Backend::enqueueDownload(Stream& stream, MutableByteSpan destination,
                                     const Buffer& source, u64 offset) {
    TF_CHECK(checkRange(source, offset, destination.size(), "download"));
    if (destination.empty()) {
        return {};
    }

    auto& typed = static_cast<D3D12Stream&>(stream);
    auto& origin = const_cast<D3D12Buffer&>(static_cast<const D3D12Buffer&>(source));
    TF_CHECK(typed.begin());

    if (origin.kind() != MemoryKind::Device) {
        std::memcpy(destination.data(), static_cast<const u8*>(origin.hostPointer()) + offset,
                    destination.size());
        return {};
    }

    TF_TRY(ComPtr<ID3D12Resource> staging, createStaging(destination.size(), /*readback=*/true));

    transition(typed, origin, D3D12_RESOURCE_STATE_COPY_SOURCE);
    typed.list->CopyBufferRegion(staging.get(), 0, origin.resource(), offset,
                                 destination.size());
    transition(typed, origin, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    // The memcpy out of the staging buffer waits for the GPU; see flush().
    typed.readbacks.push_back(D3D12Stream::PendingReadback{
            .destination = destination, .staging = staging, .bytes = destination.size()});
    return {};
}

Status D3D12Backend::enqueueCopy(Stream& stream, Buffer& destination, u64 destinationOffset,
                                 const Buffer& source, u64 sourceOffset, u64 bytes) {
    TF_CHECK(checkRange(destination, destinationOffset, bytes, "copy destination"));
    TF_CHECK(checkRange(source, sourceOffset, bytes, "copy source"));
    if (bytes == 0) {
        return {};
    }

    auto& typed = static_cast<D3D12Stream&>(stream);
    auto& target = static_cast<D3D12Buffer&>(destination);
    auto& origin = const_cast<D3D12Buffer&>(static_cast<const D3D12Buffer&>(source));
    TF_CHECK(typed.begin());

    transition(typed, origin, D3D12_RESOURCE_STATE_COPY_SOURCE);
    transition(typed, target, D3D12_RESOURCE_STATE_COPY_DEST);
    typed.list->CopyBufferRegion(target.resource(), destinationOffset, origin.resource(),
                                 sourceOffset, bytes);
    transition(typed, origin, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    transition(typed, target, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    return {};
}

Status D3D12Backend::enqueueFill(Stream& stream, Buffer& destination, u64 offset, u64 bytes,
                                 u8 value) {
    TF_CHECK(checkRange(destination, offset, bytes, "fill"));
    if (bytes == 0) {
        return {};
    }

    // ClearUnorderedAccessViewUint needs a descriptor heap and a shader-visible
    // view. Filling from an upload buffer is fewer moving parts, and a fill is
    // not on any hot path - it happens at setup.
    auto& typed = static_cast<D3D12Stream&>(stream);
    auto& target = static_cast<D3D12Buffer&>(destination);
    TF_CHECK(typed.begin());

    if (target.kind() != MemoryKind::Device) {
        std::memset(static_cast<u8*>(target.hostPointer()) + offset, value,
                    static_cast<usize>(bytes));
        return {};
    }

    TF_TRY(ComPtr<ID3D12Resource> staging, createStaging(bytes, /*readback=*/false));
    void* mapped = nullptr;
    const D3D12_RANGE nothingRead{0, 0};
    TF_D3D(staging->Map(0, &nothingRead, &mapped), "mapping a fill buffer");
    std::memset(mapped, value, static_cast<usize>(bytes));
    staging->Unmap(0, nullptr);

    transition(typed, target, D3D12_RESOURCE_STATE_COPY_DEST);
    typed.list->CopyBufferRegion(target.resource(), offset, staging.get(), 0, bytes);
    transition(typed, target, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    typed.retained.push_back(std::move(staging));
    return {};
}

// ---------------------------------------------------------------------------
// Creation
// ---------------------------------------------------------------------------

Status D3D12Backend::buildRootSignature() {
    // Root descriptors only: no descriptor heaps at all. Every kernel argument
    // is a raw buffer address plus a handful of constants, and a root UAV costs
    // two DWORDs against the 64 available. Descriptor tables would mean a heap,
    // a ring allocator and lifetime tracking for no benefit here.
    D3D12_ROOT_PARAMETER parameters[9]{};
    for (u32 i = 0; i < 8; ++i) {
        parameters[i].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
        parameters[i].Descriptor.ShaderRegister = i;
        parameters[i].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    }
    parameters[8].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    parameters[8].Constants.ShaderRegister = 0;
    parameters[8].Constants.Num32BitValues = 16;
    parameters[8].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    D3D12_ROOT_SIGNATURE_DESC desc{};
    desc.NumParameters = 9;
    desc.pParameters = parameters;

    ComPtr<ID3DBlob> serialized;
    ComPtr<ID3DBlob> errors;
    const HRESULT result = ::D3D12SerializeRootSignature(
            &desc, D3D_ROOT_SIGNATURE_VERSION_1_0, serialized.put(), errors.put());
    if (FAILED(result)) {
        const std::string detail =
                errors ? std::string{static_cast<const char*>(errors->GetBufferPointer()),
                                     errors->GetBufferSize()}
                       : std::string{};
        return makeError(ErrorCode::GpuFailure, "serializing the root signature: {}", detail);
    }

    TF_D3D(device_->CreateRootSignature(0, serialized->GetBufferPointer(),
                                        serialized->GetBufferSize(),
                                        IID_PPV_ARGS(rootSignature_.put())),
           "creating the root signature");
    return {};
}

Status D3D12Backend::buildPipelines() {
    for (u32 i = 0; i < static_cast<u32>(ShaderId::Count); ++i) {
        const auto id = static_cast<ShaderId>(i);
        const ShaderBlob blob = shaderBlob(id);
        if (blob.bytes == nullptr || blob.size == 0) {
            return makeError(ErrorCode::GpuFailure, "shader '{}' was not compiled into this "
                                                    "build",
                             shaderName(id));
        }

        D3D12_COMPUTE_PIPELINE_STATE_DESC desc{};
        desc.pRootSignature = rootSignature_.get();
        desc.CS.pShaderBytecode = blob.bytes;
        desc.CS.BytecodeLength = blob.size;

        const HRESULT created = device_->CreateComputePipelineState(
                &desc, IID_PPV_ARGS(pipelines_[i].put()));
        if (FAILED(created)) {
            return std::unexpected(hresultError(
                    created, std::format("creating the pipeline for '{}'", shaderName(id))));
        }
    }
    return {};
}

Result<std::unique_ptr<D3D12Backend>> D3D12Backend::create(u32 deviceIndex) {
    auto backend = std::make_unique<D3D12Backend>();

    ComPtr<IDXGIFactory4> factory;
    TF_D3D(::CreateDXGIFactory2(0, IID_PPV_ARGS(factory.put())), "creating a DXGI factory");

    ComPtr<IDXGIAdapter1> candidate;
    u32 seen = 0;
    for (u32 i = 0; factory->EnumAdapters1(i, candidate.put()) != DXGI_ERROR_NOT_FOUND; ++i) {
        DXGI_ADAPTER_DESC1 desc{};
        if (FAILED(candidate->GetDesc1(&desc))) {
            continue;
        }
        if ((desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0) {
            continue;
        }
        if (seen++ != deviceIndex) {
            continue;
        }

        if (FAILED(::D3D12CreateDevice(candidate.get(), D3D_FEATURE_LEVEL_12_0,
                                       IID_PPV_ARGS(backend->device_.put())))) {
            continue;
        }
        TF_D3D(candidate->QueryInterface(IID_PPV_ARGS(backend->adapter_.put())),
               "querying the adapter");

        backend->info_.name = narrow(desc.Description);
        backend->info_.totalMemoryBytes = desc.DedicatedVideoMemory;
        break;
    }

    if (!backend->device_) {
        return makeError(ErrorCode::Unsupported,
                         "no Direct3D 12 device at index {}; this build needs feature level "
                         "12_0 hardware",
                         deviceIndex);
    }

    // Wave operations are how every reduction here is written; without them the
    // kernels would need a shared-memory fallback that is slower everywhere.
    D3D12_FEATURE_DATA_D3D12_OPTIONS1 options{};
    if (SUCCEEDED(backend->device_->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS1, &options,
                                                        sizeof(options)))) {
        if (options.WaveOps == FALSE) {
            return makeError(ErrorCode::Unsupported,
                             "{} does not support wave operations, which every reduction in "
                             "these shaders uses",
                             backend->info_.name);
        }
        backend->info_.warpSize = options.WaveLaneCountMin;
    }

    D3D12_FEATURE_DATA_SHADER_MODEL model{D3D_SHADER_MODEL_6_6};
    if (FAILED(backend->device_->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &model,
                                                     sizeof(model))) ||
        model.HighestShaderModel < D3D_SHADER_MODEL_6_6) {
        return makeError(ErrorCode::Unsupported,
                         "{} does not support shader model 6.6", backend->info_.name);
    }

    D3D12_FEATURE_DATA_ARCHITECTURE architecture{};
    if (SUCCEEDED(backend->device_->CheckFeatureSupport(D3D12_FEATURE_ARCHITECTURE,
                                                        &architecture,
                                                        sizeof(architecture)))) {
        backend->info_.unifiedMemory = architecture.UMA != FALSE;
    }
    backend->info_.sharedMemoryPerBlockBytes = 32 * 1024;
    backend->info_.multiprocessorCount = 0;  // D3D12 does not report this

    D3D12_COMMAND_QUEUE_DESC queueDesc{};
    queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    TF_D3D(backend->device_->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(backend->queue_.put())),
           "creating the command queue");
    TF_D3D(backend->device_->CreateFence(0, D3D12_FENCE_FLAG_NONE,
                                         IID_PPV_ARGS(backend->fence_.put())),
           "creating the fence");

    backend->fenceEvent_ = ::CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (backend->fenceEvent_ == nullptr) {
        return makeError(ErrorCode::GpuFailure, "creating the fence event");
    }

    // Timestamps: a query heap plus somewhere to resolve them to. A device
    // that refuses either simply reports no timing, which elapsedMillisSince
    // then says rather than inventing a number.
    if (SUCCEEDED(backend->queue_->GetTimestampFrequency(&backend->timestampFrequency_))) {
        D3D12_QUERY_HEAP_DESC heapDesc{};
        heapDesc.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
        heapDesc.Count = kTimestampSlots;
        if (SUCCEEDED(backend->device_->CreateQueryHeap(
                    &heapDesc, IID_PPV_ARGS(backend->timestampHeap_.put())))) {
            auto readback = backend->createStaging(kTimestampSlots * sizeof(u64),
                                                   /*readback=*/true);
            if (readback) {
                backend->timestampReadback_ = std::move(*readback);
            } else {
                backend->timestampHeap_.reset();
            }
        }
    }

    TF_CHECK(backend->buildRootSignature());
    TF_CHECK(backend->buildPipelines());
    TF_TRY(backend->kernels_, createD3D12Kernels(*backend));
    return backend;
}

// ---------------------------------------------------------------------------
// Accessors for the dispatch layer
// ---------------------------------------------------------------------------
//
// D3D12Kernels.cpp needs a handful of things from these classes and nothing
// else. Free functions rather than a shared private header: the surface is
// small enough to name, and naming it keeps the kernels from reaching into the
// backend's internals.

ID3D12Device* backendDevice(D3D12Backend& backend) { return backend.device(); }

ID3D12RootSignature* backendRootSignature(D3D12Backend& backend) {
    return backend.rootSignature();
}

ID3D12PipelineState* backendPipeline(D3D12Backend& backend, ShaderId id) {
    return backend.pipeline(id);
}

Status streamBegin(Stream& stream) { return static_cast<D3D12Stream&>(stream).begin(); }

ID3D12GraphicsCommandList* streamList(Stream& stream) {
    return static_cast<D3D12Stream&>(stream).list.get();
}

D3D12_GPU_VIRTUAL_ADDRESS viewAddress(const DeviceView& view) {
    if (!view.valid()) {
        return 0;
    }
    const auto& buffer = static_cast<const D3D12Buffer&>(*view.buffer);
    // A root descriptor carries a byte address, so the view offset folds in
    // here and the shader indexes from zero.
    return buffer.address() + view.offset;
}

bool viewIsDevice(const DeviceView& view) {
    return view.valid() && view.buffer->kind() == MemoryKind::Device;
}

/// A full UAV barrier between dispatches.
///
/// Every kernel reads what the previous one wrote. D3D12 will otherwise overlap
/// them, where CUDA gets the ordering from the stream for free. One barrier per
/// dispatch is heavier than tracking which buffers actually alias, but a missed
/// dependency here produces numbers that are wrong only sometimes.
void streamBarrierAll(Stream& stream) {
    auto& typed = static_cast<D3D12Stream&>(stream);
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    barrier.UAV.pResource = nullptr;  // every UAV
    typed.list->ResourceBarrier(1, &barrier);
}

Result<BackendPtr> createD3D12Backend(u32 deviceIndex) {
    TF_TRY(auto backend, D3D12Backend::create(deviceIndex));
    return BackendPtr{backend.release()};
}

}  // namespace tf::gpu::d3d12
