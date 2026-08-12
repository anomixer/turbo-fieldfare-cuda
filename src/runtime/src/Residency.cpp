#include "tf/runtime/Residency.h"

#include "tf/runtime/ForwardRunner.h"

#include <algorithm>
#include <format>

namespace tf::runtime {
namespace {

constexpr u64 kFp16Bytes = 2;

std::string gib(u64 bytes) {
    return std::format("{:.2f} GiB", static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0));
}

std::string mib(u64 bytes) {
    return std::format("{:.1f} MiB", static_cast<double>(bytes) / (1024.0 * 1024.0));
}

}  // namespace

u64 kvRowsForLayer(const ArchInfo& arch, u64 layer, u64 contextLength,
                   u64 maxPrefillTokens) {
    if (arch.isFullAttention(layer)) {
        // Append-only: the whole prompt has to fit.
        return contextLength;
    }
    // A ring only ever needs the window, plus headroom so a prefill chunk can
    // be written without overwriting rows that same chunk still reads. See
    // kSlidingWindowHeadroomRows for why the headroom must cover the chunk.
    const u64 window = std::min(arch.slidingWindow, contextLength);
    return window + std::max(kSlidingWindowHeadroomRows, maxPrefillTokens);
}

u64 kvCacheBytesForLayer(const ArchInfo& arch, u64 layer, u64 contextLength,
                         u64 maxPrefillTokens) {
    const u64 rows = kvRowsForLayer(arch, layer, contextLength, maxPrefillTokens);
    const u64 perTensor = arch.kvHeadsFor(layer) * rows * arch.headDimFor(layer) * kFp16Bytes;
    return perTensor * 2;  // keys and values
}

u64 ResidencyPlan::residentLayerCount() const {
    return static_cast<u64>(
            std::ranges::count_if(layers, [](const LayerResidency& layer) {
                return layer.expertsResident;
            }));
}

u64 ResidencyPlan::streamedLayerCount() const {
    return layers.size() - residentLayerCount();
}

std::string ResidencyPlan::describe() const {
    std::string out;
    out += std::format("  resident core   {:>12}\n", gib(coreBytes));
    out += std::format("  KV cache        {:>12}\n", gib(kvCacheBytes));
    out += std::format("  experts         {:>12}\n", gib(expertBytes));
    out += std::format("  scratch         {:>12}\n", gib(scratchBytes));
    out += std::format("  total           {:>12}\n", gib(totalBytes()));
    out += std::format("  layers          {} resident, {} streamed\n", residentLayerCount(),
                       streamedLayerCount());
    if (streamedLayerCount() > 0) {
        out += std::format("  slots           {} per streamed layer{}\n", slotsPerStreamedLayer,
                           slotsWereReduced ? " (reduced to fit the budget)" : "");
    }
    if (streamedLayerCount() > 0) {
        out += std::format("  worst case      {} per token across PCIe\n",
                           mib(worstCaseStreamedBytesPerToken));
    } else {
        out += "  worst case      nothing streams; all experts are resident\n";
    }
    return out;
}

Result<ResidencyPlan> planResidency(const ArchInfo& arch, const gturbo::ExpertLayout& experts,
                                    u64 residentCoreBytes, const ResidencyBudget& budget) {
    TF_CHECK(arch.validate());
    TF_CHECK(experts.validate());

    if (experts.numLayers != arch.numLayers) {
        return makeError(ErrorCode::InvalidArgument,
                         "expert layout covers {} layers but the architecture has {}",
                         experts.numLayers, arch.numLayers);
    }
    if (budget.contextLength == 0) {
        return makeError(ErrorCode::InvalidArgument, "context length must be non-zero");
    }
    if (budget.contextLength > arch.maxPositionEmbeddings) {
        return makeError(ErrorCode::InvalidArgument,
                         "context length {} exceeds the model's maximum of {}",
                         budget.contextLength, arch.maxPositionEmbeddings);
    }
    if (budget.slotsPerStreamedLayer == 0) {
        return makeError(ErrorCode::InvalidArgument,
                         "a streamed layer needs at least one expert slot");
    }
    // Fewer slots than the router selects would thrash: every token would evict
    // an expert it is about to need again within the same layer.
    if (budget.slotsPerStreamedLayer < arch.topKExperts) {
        return makeError(ErrorCode::InvalidArgument,
                         "{} slots per layer is below top-k of {}; the cache would evict "
                         "experts still needed for the current token",
                         budget.slotsPerStreamedLayer, arch.topKExperts);
    }

    ResidencyPlan plan;
    plan.coreBytes = residentCoreBytes;
    // Left at zero, the scratch reservation is the larger of a fixed floor and
    // what the prefill chunk actually needs. Both halves matter: batching
    // prompt tokens scales every per-token intermediate, so a fixed figure goes
    // silently too small at a large chunk - but the runner's buffers are not
    // the only thing living in this reservation, so the derived figure must
    // never be allowed to shrink it either.
    plan.scratchBytes =
            budget.scratchBytes > 0
                    ? budget.scratchBytes
                    : std::max(kMinimumScratchBytes,
                               ForwardRunner::estimateScratchBytes(arch,
                                                                   budget.maxPrefillTokens));
    plan.layers.assign(static_cast<usize>(arch.numLayers), LayerResidency{});

    for (u64 layer = 0; layer < arch.numLayers; ++layer) {
        plan.kvCacheBytes += kvCacheBytesForLayer(arch, layer, budget.contextLength,
                                                  budget.maxPrefillTokens);
    }

    const u64 fixed = plan.coreBytes + plan.kvCacheBytes + plan.scratchBytes;
    if (fixed > budget.deviceBytes) {
        return makeError(ErrorCode::OutOfMemory,
                         "the resident core ({}), KV cache at {} tokens ({}) and scratch ({}) "
                         "need {} but the budget is {}",
                         gib(plan.coreBytes), budget.contextLength, gib(plan.kvCacheBytes),
                         gib(plan.scratchBytes), gib(fixed), gib(budget.deviceBytes));
    }

    // Slots are the floor every layer gets. A request that does not fit is
    // reduced rather than refused: the number is a performance knob, and
    // failing to start because a tuning default was optimistic is the wrong
    // response. The plan reports what was actually used, so the reduction is
    // visible rather than silent.
    const u64 affordable = (budget.deviceBytes - fixed) / (experts.stride * arch.numLayers);
    u32 slots = budget.slotsPerStreamedLayer;
    if (affordable < slots) {
        slots = static_cast<u32>(affordable);
    }

    // Below top-k the cache would evict experts the current token still needs,
    // so that is the point where there is nothing sensible left to do.
    if (slots < arch.topKExperts) {
        return makeError(
                ErrorCode::OutOfMemory,
                "{} of budget leaves room for only {} expert slots per layer after {} of "
                "fixed allocations, below the top-{} the router selects; reduce --context "
                "or raise --vram-budget",
                gib(budget.deviceBytes), slots, gib(fixed), arch.topKExperts);
    }

    plan.slotsPerStreamedLayer = slots;
    plan.slotsWereReduced = slots < budget.slotsPerStreamedLayer;

    const u64 slotBytesPerLayer = u64{slots} * experts.stride;
    const u64 minimumExpertBytes = slotBytesPerLayer * arch.numLayers;

    // Start from all-streamed, then promote layers to resident while the
    // upgrade still fits. Promoting frees that layer's slots, so the cost of a
    // promotion is the difference, not the full expert set.
    u64 used = fixed + minimumExpertBytes;
    const u64 residentLayerBytes = experts.stride * experts.expertsPerLayer;
    const u64 promotionCost = residentLayerBytes - slotBytesPerLayer;

    for (auto& layer : plan.layers) {
        layer.expertsResident = false;
        layer.slots = slots;
    }

    for (u64 layer = 0; layer < arch.numLayers; ++layer) {
        if (used + promotionCost > budget.deviceBytes) {
            break;
        }
        plan.layers[static_cast<usize>(layer)] = LayerResidency{.expertsResident = true,
                                                                .slots = 0};
        used += promotionCost;
    }

    for (const auto& layer : plan.layers) {
        plan.expertBytes += layer.expertsResident ? residentLayerBytes
                                                  : u64{layer.slots} * experts.stride;
    }

    // Worst case is every routed expert on every streamed layer missing.
    plan.worstCaseStreamedBytesPerToken =
            plan.streamedLayerCount() * arch.topKExperts * experts.blobBytes;

    return plan;
}

}  // namespace tf::runtime
