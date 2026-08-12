#include "tf/runtime/Generator.h"

#include <algorithm>
#include <chrono>
#include <format>

namespace tf::runtime {

std::string_view describe(StopReason reason) {
    switch (reason) {
        case StopReason::StopToken:
            return "stop token";
        case StopReason::StopString:
            return "stop string";
        case StopReason::Length:
            return "token limit";
        case StopReason::ContextFull:
            return "context full";
        case StopReason::Cancelled:
            return "cancelled";
    }
    return "unknown";
}

std::string GenerationStats::describe() const {
    return std::format(
            "{} prompt tokens in {:.2f}s ({:.1f} tok/s), {} generated in {:.2f}s ({:.1f} "
            "tok/s), stopped on {}, seed {}",
            promptTokens, prefillSeconds, prefillTokensPerSecond(), generatedTokens,
            decodeSeconds, decodeTokensPerSecond(), tf::runtime::describe(reason), seed);
}

Generator::Generator() = default;
Generator::~Generator() = default;
Generator::Generator(Generator&&) noexcept = default;
Generator& Generator::operator=(Generator&&) noexcept = default;

Result<Generator> Generator::create(gpu::IGpuBackend& backend, Model& model,
                                    const Tokenizer& tokenizer, u64 contextLength,
                                    u32 maxPrefillChunk) {
    if (maxPrefillChunk == 0) {
        return makeError(ErrorCode::InvalidArgument, "a prefill chunk of zero tokens");
    }

    Generator generator;
    generator.model_ = &model;
    generator.tokenizer_ = &tokenizer;
    generator.maxPrefillChunk_ = maxPrefillChunk;

    TF_TRY(KVCacheManager cache,
           KVCacheManager::create(backend, model.arch(), contextLength, maxPrefillChunk));
    generator.cache_ = std::make_unique<KVCacheManager>(std::move(cache));

    TF_TRY(generator.runner_,
           ForwardRunner::create(backend, model, *generator.cache_, maxPrefillChunk));
    return generator;
}

u64 Generator::position() const noexcept { return cache_->position(); }
u64 Generator::contextLength() const noexcept { return cache_->contextLength(); }

void Generator::reset() {
    cache_->reset();
    history_.clear();
    cached_.clear();
}

u64 Generator::reusablePrefix(std::span<const u32> promptIds) const {
    // The cache and the new prompt must agree token for token; the first
    // difference ends the reusable region.
    u64 shared = 0;
    const u64 limit = std::min<u64>(cached_.size(), promptIds.size());
    while (shared < limit && cached_[static_cast<usize>(shared)] == promptIds[shared]) {
        ++shared;
    }

    // At least one token has to be left to prefill and decode from: the decode
    // loop needs a token to step on, and it is the only place logits appear.
    if (shared >= promptIds.size()) {
        shared = promptIds.size() - 1;
    }

    // Rewinding further than the sliding-window rings can support would leave
    // the re-prefilled tokens attending over rows from a different part of the
    // sequence. Nothing about that fails loudly, so it is checked here.
    const u64 earliest = cache_->earliestSafeRewind();
    if (shared < earliest) {
        return 0;
    }
    return shared;
}

Status Generator::prefill(std::span<const u32> promptIds, u32 chunk) {
    for (usize offset = 0; offset < promptIds.size(); offset += chunk) {
        const usize count = std::min(static_cast<usize>(chunk), promptIds.size() - offset);
        TF_CHECK(runner_.prefillChunk(promptIds.subspan(offset, count), cache_->position()));
        cache_->advance(count);
    }
    return {};
}

Result<GenerationStats> Generator::generate(std::span<const u32> promptIds,
                                            const GenerationOptions& options,
                                            const TextCallback& onText) {
    TF_CHECK(options.sampling.validate());
    if (promptIds.empty()) {
        return makeError(ErrorCode::InvalidArgument, "an empty prompt");
    }
    if (options.prefillChunk == 0 || options.prefillChunk > maxPrefillChunk_) {
        return makeError(ErrorCode::InvalidArgument,
                         "a prefill chunk of {} is outside the 1..{} this generator was built "
                         "for",
                         options.prefillChunk, maxPrefillChunk_);
    }
    if (promptIds.size() >= cache_->contextLength()) {
        return makeError(ErrorCode::InvalidArgument,
                         "a {} token prompt at position {} does not fit the {} token context",
                         promptIds.size(), cache_->position(), cache_->contextLength());
    }

    Sampler sampler{options.sampling};
    StopMatcher matcher{options.stopStrings};
    AssistantDecoder decoder{*tokenizer_};

    GenerationStats stats;
    stats.promptTokens = promptIds.size();
    stats.seed = sampler.seed();

    // ---- Prefill ---------------------------------------------------------
    //
    // Whatever prefix the cache already holds is reused; a chat client resends
    // the whole conversation every turn, so this is usually all of it but the
    // newest message. Every remaining prompt token except the last goes through
    // the batched path - the last is left for the decode loop, which needs a
    // token to step on and is the only place logits are wanted.
    const u64 reused = reusablePrefix(promptIds);
    stats.cachedPromptTokens = reused;

    cache_->rewindTo(reused);
    cached_.assign(promptIds.begin(), promptIds.begin() + static_cast<isize>(reused));

    const auto prefillStart = std::chrono::steady_clock::now();
    const auto prefilled = prefill(promptIds.subspan(reused, promptIds.size() - reused - 1),
                                   options.prefillChunk);
    if (!prefilled) {
        // The cache advanced for whatever chunks did land, so the record of
        // what it holds is no longer trustworthy. Dropping it costs the next
        // request a full prefill; keeping a wrong one would have it reuse
        // tokens that were never written.
        cached_.clear();
        cache_->reset();
        return std::unexpected(prefilled.error());
    }
    stats.prefillSeconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - prefillStart)
                    .count();

    cached_.insert(cached_.end(), promptIds.begin() + static_cast<isize>(reused),
                   promptIds.end() - 1);

    // ---- Decode ----------------------------------------------------------
    const auto isStopToken = [&](u32 token) {
        return std::ranges::find(options.stopTokens, token) != options.stopTokens.end();
    };

    u32 current = promptIds.back();
    stats.reason = StopReason::Length;
    const auto decodeStart = std::chrono::steady_clock::now();

    for (u64 produced = 0; produced < options.maxTokens; ++produced) {
        if (!cache_->canAccept(1)) {
            stats.reason = StopReason::ContextFull;
            break;
        }

        TF_CHECK(runner_.decodeStep(current, cache_->position()));
        cache_->advance(1);
        // `current` is now in the cache, so it belongs to the prefix the next
        // request can reuse.
        cached_.push_back(current);

        u32 next = 0;
        if (options.sampling.isGreedy() && options.sampling.repetitionPenalty == 1.0f) {
            // Nothing needs the logits on the host, so leave them on the device
            // and let the argmax kernel decide. Saves a 1 MiB download per
            // token.
            TF_TRY(next, runner_.greedyToken());
        } else {
            TF_TRY(logits_, runner_.readLogits());
            next = sampler.sample(logits_, history_);
        }

        // A stop token ends the turn and is not part of the answer, so it is
        // never decoded into the output.
        if (isStopToken(next)) {
            stats.reason = StopReason::StopToken;
            break;
        }

        history_.push_back(next);
        ++stats.generatedTokens;
        current = next;

        // The answer and the thinking are separated before either reaches the
        // caller: the raw stream is channel-structured markup, not text.
        const AssistantDecoder::Update decoded = decoder.push(next);
        if (!decoded.thinking.empty() && onThinking_) {
            onThinking_(decoded.thinking);
        }
        if (decoded.content.empty()) {
            continue;
        }

        const StopMatcher::Update update = matcher.push(decoded.content);
        if (!update.emit.empty() && !onText(update.emit)) {
            stats.reason = StopReason::Cancelled;
            break;
        }
        if (update.stopped) {
            stats.reason = StopReason::StopString;
            break;
        }
    }

    // Text held back against a stop string that never completed still belongs
    // to the answer.
    if (stats.reason != StopReason::StopString) {
        const std::string remaining = matcher.flush();
        if (!remaining.empty()) {
            static_cast<void>(onText(remaining));
        }
    }

    stats.decodeSeconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - decodeStart)
                    .count();
    return stats;
}

}  // namespace tf::runtime
