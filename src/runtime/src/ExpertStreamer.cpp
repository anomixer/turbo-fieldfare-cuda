#include "tf/runtime/ExpertStreamer.h"

#include <algorithm>
#include <format>
#include <map>
#include <utility>

#include "ReadPool.h"

namespace tf::runtime {
namespace {

struct SlotState {
    u32 expert = kEmptySlot;
    /// Cumulative selections of the expert currently held. Drives LFU.
    u64 useCount = 0;
    /// Logical clock value at the last touch. Drives LRU and breaks LFU ties.
    u64 lastUse = 0;
    /// Claimed by a plan whose fetch has not completed. Such a slot must not be
    /// evicted, or the fetch would land on top of a different expert.
    bool reserved = false;
};

struct LayerState {
    io::File file;
    gpu::BufferPtr slots;
    std::vector<SlotState> slotStates;
    /// Cumulative selections per expert, retained across evictions so an expert
    /// that comes back is not treated as new.
    std::vector<u64> expertUseCount;
    u64 clock = 0;
};

}  // namespace

struct ExpertStreamer::Impl {
    gpu::IGpuBackend* backend = nullptr;
    gturbo::ExpertLayout layout;
    StreamerOptions options;

    /// Sparse: only the layers the residency planner marked streamed.
    std::map<u64, LayerState> layers;

    /// One pinned buffer per slot, so every miss in a fetch reads into its own
    /// staging area and the parallel reads never overlap.
    std::vector<gpu::BufferPtr> staging;

    /// Recorded after each fetch's uploads. The next fetch waits on it before
    /// overwriting staging, since an upload still in flight would otherwise
    /// read bytes being replaced. One sync per fetch, on a thread that has just
    /// blocked for the router readback anyway. M13 could double-buffer this.
    gpu::EventPtr uploadsDone;
    bool uploadsPending = false;

    std::unique_ptr<ReadPool> pool;
    std::mutex mutex;
};

ExpertStreamer::ExpertStreamer() = default;
ExpertStreamer::~ExpertStreamer() = default;
ExpertStreamer::ExpertStreamer(ExpertStreamer&&) noexcept = default;
ExpertStreamer& ExpertStreamer::operator=(ExpertStreamer&&) noexcept = default;

Result<ExpertStreamer> ExpertStreamer::create(gpu::IGpuBackend& backend,
                                              const std::filesystem::path& installDir,
                                              const gturbo::ExpertLayout& layout,
                                              std::span<const u64> streamedLayers,
                                              const StreamerOptions& options) {
    TF_CHECK(layout.validate());

    if (options.slotsPerLayer == 0) {
        return makeError(ErrorCode::InvalidArgument, "a streamed layer needs at least one slot");
    }
    if (options.slotsPerLayer > layout.expertsPerLayer) {
        return makeError(ErrorCode::InvalidArgument,
                         "{} slots requested but a layer only has {} experts",
                         options.slotsPerLayer, layout.expertsPerLayer);
    }

    io::ReadMode mode =
            options.unbufferedReads ? io::ReadMode::Unbuffered : io::ReadMode::RandomBuffered;

    if (options.unbufferedReads) {
        // Unbuffered reads DMA straight into pinned memory, which only works
        // when every blob starts on a sector boundary.
        TF_TRY(const u64 sectorSize, io::File::sectorSize(installDir));
        if (!layout.supportsUnbufferedReads(sectorSize)) {
            return makeError(ErrorCode::InvalidArgument,
                             "unbuffered reads need the {} byte expert stride aligned to the "
                             "{} byte sector",
                             layout.stride, sectorSize);
        }
    }

    ExpertStreamer streamer;
    streamer.impl_ = std::make_unique<Impl>();
    streamer.impl_->backend = &backend;
    streamer.impl_->layout = layout;
    streamer.impl_->options = options;
    streamer.blobBytes_ = layout.blobBytes;
    streamer.slotsPerLayer_ = options.slotsPerLayer;

    const std::filesystem::path expertsDir = installDir / gturbo::kExpertsDir;

    for (const u64 layer : streamedLayers) {
        if (layer >= layout.numLayers) {
            return makeError(ErrorCode::InvalidArgument,
                             "streamed layer {} is outside the {} layers in the layout", layer,
                             layout.numLayers);
        }

        LayerState state;
        TF_TRY(state.file,
               io::File::openRead(expertsDir / layout.layerFiles[static_cast<usize>(layer)],
                                  mode));
        TF_TRY(state.slots,
               backend.allocate(gpu::MemoryKind::Device,
                                u64{options.slotsPerLayer} * layout.stride,
                                std::format("expert-slots-{:02}", layer)));

        state.slotStates.assign(options.slotsPerLayer, SlotState{});
        state.expertUseCount.assign(static_cast<usize>(layout.expertsPerLayer), 0);

        streamer.impl_->layers.emplace(layer, std::move(state));
    }

    // Reads are unbuffered-capable, so staging must be sector-aligned in size
    // as well as pinned. Rounding up to the stride covers both.
    streamer.impl_->staging.reserve(options.slotsPerLayer);
    for (u32 i = 0; i < options.slotsPerLayer; ++i) {
        TF_TRY(gpu::BufferPtr buffer,
               backend.allocate(gpu::MemoryKind::PinnedHost, layout.stride,
                                std::format("expert-staging-{}", i)));
        streamer.impl_->staging.push_back(std::move(buffer));
    }

    TF_TRY(streamer.impl_->uploadsDone, backend.createEvent(/*withTiming=*/false));
    streamer.impl_->pool = std::make_unique<ReadPool>(options.readThreads);

    return streamer;
}

bool ExpertStreamer::isStreamed(u64 layer) const {
    return impl_ != nullptr && impl_->layers.contains(layer);
}

gpu::DeviceView ExpertStreamer::slotView(u64 layer, u32 slot) const {
    const auto it = impl_->layers.find(layer);
    if (it == impl_->layers.end()) {
        return {};
    }
    return gpu::DeviceView{.buffer = it->second.slots.get(),
                           .offset = u64{slot} * impl_->layout.stride};
}

void ExpertStreamer::resetStats() { stats_ = StreamerStats{}; }

void ExpertStreamer::clearCache() {
    if (impl_ == nullptr) {
        return;
    }
    const std::lock_guard lock{impl_->mutex};
    for (auto& [layer, state] : impl_->layers) {
        state.slotStates.assign(impl_->options.slotsPerLayer, SlotState{});
        std::ranges::fill(state.expertUseCount, 0);
        state.clock = 0;
    }
}

Result<ExpertCachePlan> ExpertStreamer::plan(u64 layer, std::span<const u32> experts) {
    if (impl_ == nullptr) {
        return makeError(ErrorCode::InvalidArgument, "streamer is not initialized");
    }
    const auto it = impl_->layers.find(layer);
    if (it == impl_->layers.end()) {
        return makeError(ErrorCode::InvalidArgument,
                         "layer {} is resident, not streamed; read its experts directly",
                         layer);
    }
    if (experts.size() > impl_->options.slotsPerLayer) {
        return makeError(ErrorCode::InvalidArgument,
                         "{} experts requested but only {} slots exist per layer",
                         experts.size(), impl_->options.slotsPerLayer);
    }

    LayerState& state = it->second;
    const std::lock_guard lock{impl_->mutex};

    ExpertCachePlan plan;
    plan.layer = layer;
    plan.experts.assign(experts.begin(), experts.end());
    plan.slots.assign(experts.size(), kEmptySlot);

    ++state.clock;

    // A duplicate expert within one request must map to the same slot rather
    // than being fetched twice.
    for (u32 i = 0; i < experts.size(); ++i) {
        const u32 expert = experts[i];
        if (expert >= impl_->layout.expertsPerLayer) {
            return makeError(ErrorCode::InvalidArgument,
                             "expert {} is outside the {} experts in layer {}", expert,
                             impl_->layout.expertsPerLayer, layer);
        }
        for (u32 j = 0; j < i; ++j) {
            if (experts[j] == expert) {
                plan.slots[i] = plan.slots[j];
                plan.hits.push_back(i);
                break;
            }
        }
        if (plan.slots[i] != kEmptySlot) {
            continue;
        }

        state.expertUseCount[expert] += 1;

        // Already resident?
        bool found = false;
        for (u32 slot = 0; slot < state.slotStates.size(); ++slot) {
            if (state.slotStates[slot].expert == expert) {
                plan.slots[i] = slot;
                state.slotStates[slot].useCount = state.expertUseCount[expert];
                state.slotStates[slot].lastUse = state.clock;
                state.slotStates[slot].reserved = true;
                plan.hits.push_back(i);
                found = true;
                break;
            }
        }
        if (found) {
            continue;
        }

        // Miss: take a free slot, else evict.
        u32 chosen = kEmptySlot;
        for (u32 slot = 0; slot < state.slotStates.size(); ++slot) {
            if (state.slotStates[slot].expert == kEmptySlot &&
                !state.slotStates[slot].reserved) {
                chosen = slot;
                break;
            }
        }

        if (chosen == kEmptySlot) {
            // Evict, never touching a slot this same plan already claimed.
            for (u32 slot = 0; slot < state.slotStates.size(); ++slot) {
                if (state.slotStates[slot].reserved) {
                    continue;
                }
                if (chosen == kEmptySlot) {
                    chosen = slot;
                    continue;
                }
                const SlotState& candidate = state.slotStates[slot];
                const SlotState& best = state.slotStates[chosen];

                const bool better =
                        impl_->options.policy == EvictionPolicy::Lru
                                ? candidate.lastUse < best.lastUse
                                : (candidate.useCount < best.useCount ||
                                   (candidate.useCount == best.useCount &&
                                    candidate.lastUse < best.lastUse));
                if (better) {
                    chosen = slot;
                }
            }
        }

        if (chosen == kEmptySlot) {
            // Only reachable if every slot were reserved, which the size check
            // above rules out.
            return makeError(ErrorCode::Unknown,
                             "layer {}: no slot available for expert {}", layer, expert);
        }

        state.slotStates[chosen] = SlotState{.expert = expert,
                                             .useCount = state.expertUseCount[expert],
                                             .lastUse = state.clock,
                                             .reserved = true};
        plan.slots[i] = chosen;
        plan.misses.push_back(i);
    }

    stats_.requests += experts.size();
    stats_.hits += plan.hits.size();
    stats_.misses += plan.misses.size();

    return plan;
}

Status ExpertStreamer::fetch(const ExpertCachePlan& plan, gpu::Stream& stream) {
    if (impl_ == nullptr) {
        return makeError(ErrorCode::InvalidArgument, "streamer is not initialized");
    }
    const auto it = impl_->layers.find(plan.layer);
    if (it == impl_->layers.end()) {
        return makeError(ErrorCode::InvalidArgument, "layer {} is not streamed", plan.layer);
    }
    LayerState& state = it->second;

    // Release the reservations regardless of how this returns, so a failed
    // fetch cannot leave slots permanently pinned.
    struct ReleaseGuard {
        Impl* impl;
        LayerState* state;
        const ExpertCachePlan* plan;
        ~ReleaseGuard() {
            const std::lock_guard lock{impl->mutex};
            for (const u32 slot : plan->slots) {
                if (slot != kEmptySlot && slot < state->slotStates.size()) {
                    state->slotStates[slot].reserved = false;
                }
            }
        }
    } guard{impl_.get(), &state, &plan};

    if (plan.misses.empty()) {
        return {};
    }
    if (plan.misses.size() > impl_->staging.size()) {
        return makeError(ErrorCode::InvalidArgument,
                         "{} misses exceed the {} staging buffers", plan.misses.size(),
                         impl_->staging.size());
    }

    // Wait for the previous fetch's uploads before overwriting staging.
    if (impl_->uploadsPending) {
        TF_CHECK(impl_->uploadsDone->synchronize());
        impl_->uploadsPending = false;
    }

    // Reads are independent, so they run across the pool. Errors are collected
    // per index rather than thrown, since the pool has no exception channel.
    std::vector<Status> results(plan.misses.size());
    const u64 blobBytes = impl_->layout.blobBytes;

    impl_->pool->parallelFor(static_cast<u32>(plan.misses.size()), [&](u32 index) {
        const u32 requestIndex = plan.misses[index];
        const u32 expert = plan.experts[requestIndex];

        const auto range = impl_->layout.expertRange(plan.layer, expert);
        if (!range) {
            results[index] = std::unexpected(range.error());
            return;
        }

        auto destination = impl_->staging[index]->hostBytes();
        results[index] = state.file.readExactAt(
                range->offset, MutableByteSpan{destination.data(),
                                               static_cast<usize>(blobBytes)});
    });

    for (usize i = 0; i < results.size(); ++i) {
        if (!results[i]) {
            return std::unexpected(results[i].error().wrap(
                    std::format("layer {} expert {}", plan.layer,
                                plan.experts[plan.misses[i]])));
        }
    }

    for (u32 index = 0; index < plan.misses.size(); ++index) {
        const u32 requestIndex = plan.misses[index];
        const u32 slot = plan.slots[requestIndex];

        TF_CHECK(impl_->backend->enqueueUpload(
                stream, *state.slots, u64{slot} * impl_->layout.stride,
                ByteSpan{impl_->staging[index]->hostBytes().data(),
                         static_cast<usize>(blobBytes)}));
    }

    TF_CHECK(impl_->uploadsDone->record(stream));
    impl_->uploadsPending = true;

    stats_.bytesRead += blobBytes * plan.misses.size();
    stats_.readCount += plan.misses.size();

    return {};
}

}  // namespace tf::runtime
