#pragma once

#include <functional>
#include <memory>
#include <mutex>
#include <string>

#include "Api.h"
#include "tf/core/base/Error.h"
#include "tf/core/tokenizer/ChatTemplate.h"
#include "tf/core/tokenizer/Tokenizer.h"
#include "tf/gpu/Backend.h"
#include "tf/runtime/Generator.h"
#include "tf/runtime/Model.h"

namespace tf::server {

struct EngineOptions {
    std::filesystem::path modelDir;
    u64 contextLength = 4096;
    u32 prefillChunk = 128;
    u64 vramBudget = 0;
    /// Zero leaves the residency planner's default.
    u32 expertSlots = 0;
    /// Name reported by /v1/models and echoed in responses.
    std::string modelName;
};

/// Owns the model and answers one request at a time.
///
/// Inference is serialized behind a mutex. That is not a simplification to
/// revisit later: there is one GPU, one KV cache and one prompt cache, and a
/// second concurrent generation would either need its own 0.3 GiB cache and
/// interleave badly on the same device, or corrupt the first. Connections are
/// still handled concurrently, so a second client waits rather than being
/// refused.
class Engine {
public:
    Engine();
    ~Engine();

    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    [[nodiscard]] static Result<std::unique_ptr<Engine>> create(const EngineOptions& options);

    /// Called with each piece of text as it is produced. Returning false stops
    /// generation, which is how a disconnected client stops costing GPU time.
    using TextCallback = std::function<bool(std::string_view)>;

    /// Runs one request to completion. Blocks until any earlier request
    /// finishes.
    [[nodiscard]] Result<runtime::GenerationStats> run(const CompletionRequest& request,
                                                       const TextCallback& onText);

    [[nodiscard]] const std::string& modelName() const noexcept { return modelName_; }
    [[nodiscard]] const runtime::Model& model() const noexcept { return *model_; }

    /// Longest prompt the context can hold, for the diagnostics endpoint.
    [[nodiscard]] u64 contextLength() const noexcept { return contextLength_; }

    /// Requests answered since startup, and how many prompt tokens the cache
    /// saved. Reported by /health; a hit rate near zero on a chat workload
    /// means something is defeating the prompt cache.
    struct Stats {
        u64 requests = 0;
        u64 promptTokens = 0;
        u64 cachedPromptTokens = 0;
        u64 generatedTokens = 0;
    };
    [[nodiscard]] Stats stats() const;

private:
    [[nodiscard]] Result<std::vector<u32>> renderPrompt(const CompletionRequest& request,
                                                        std::vector<u32>& stopTokens) const;

    gpu::BackendPtr backend_;
    std::unique_ptr<runtime::Model> model_;
    std::unique_ptr<Tokenizer> tokenizer_;
    std::unique_ptr<ChatTemplate> chatTemplate_;
    std::unique_ptr<runtime::Generator> generator_;

    std::string modelName_;
    u64 contextLength_ = 0;
    u32 prefillChunk_ = 128;

    /// Guards the generator, which owns the KV cache and the prompt cache.
    mutable std::mutex mutex_;
    Stats stats_;
};

/// Routes one HTTP request to the right endpoint.
[[nodiscard]] void handleRequest(Engine& engine, const Request& request,
                                 Connection& connection);

}  // namespace tf::server
