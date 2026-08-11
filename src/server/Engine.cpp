#include "Engine.h"

#include <format>

#include "tf/core/json/Json.h"

namespace tf::server {

Engine::Engine() = default;
Engine::~Engine() = default;

Result<std::unique_ptr<Engine>> Engine::create(const EngineOptions& options) {
    const auto backends = gpu::compiledBackends();
    if (backends.empty()) {
        return makeError(ErrorCode::Unsupported, "this build has no GPU backend");
    }

    auto engine = std::make_unique<Engine>();
    TF_TRY(engine->backend_, gpu::createBackend(backends.front()));

    runtime::Model::LoadOptions load;
    load.budget.deviceBytes = options.vramBudget;
    load.budget.contextLength = options.contextLength;
    if (options.expertSlots > 0) {
        load.budget.slotsPerStreamedLayer = options.expertSlots;
    }
    load.budget.maxPrefillTokens = options.prefillChunk;

    TF_TRY(runtime::Model model,
           runtime::Model::load(*engine->backend_, options.modelDir, load));
    engine->model_ = std::make_unique<runtime::Model>(std::move(model));

    TF_TRY(Tokenizer tokenizer,
           Tokenizer::loadFromFile(options.modelDir / "tokenizer" / "tokenizer.json"));
    engine->tokenizer_ = std::make_unique<Tokenizer>(std::move(tokenizer));

    TF_TRY(ChatTemplate chat, ChatTemplate::create(*engine->tokenizer_));
    engine->chatTemplate_ = std::make_unique<ChatTemplate>(std::move(chat));

    TF_TRY(runtime::Generator generator,
           runtime::Generator::create(*engine->backend_, *engine->model_, *engine->tokenizer_,
                                      options.contextLength, options.prefillChunk));
    engine->generator_ = std::make_unique<runtime::Generator>(std::move(generator));

    engine->contextLength_ = options.contextLength;
    engine->prefillChunk_ = options.prefillChunk;
    engine->modelName_ = options.modelName.empty()
                                 ? options.modelDir.stem().string()
                                 : options.modelName;
    return engine;
}

Engine::Stats Engine::stats() const {
    const std::lock_guard lock{mutex_};
    return stats_;
}

Result<std::vector<u32>> Engine::renderPrompt(const CompletionRequest& request,
                                              std::vector<u32>& stopTokens) const {
    if (request.chat) {
        TF_TRY(std::vector<u32> ids, chatTemplate_->render(request.messages));
        stopTokens = chatTemplate_->stopTokens();
        return ids;
    }

    std::vector<u32> ids = tokenizer_->encode(request.prompt);
    ids.insert(ids.begin(), model_->arch().bosTokenId);
    stopTokens.assign(model_->arch().eosTokenIds.begin(), model_->arch().eosTokenIds.end());
    return ids;
}

Result<runtime::GenerationStats> Engine::run(const CompletionRequest& request,
                                             const TextCallback& onText) {
    // Held for the whole generation: the generator owns the KV cache and the
    // prompt cache, and a second request touching either mid-run would corrupt
    // both.
    const std::lock_guard lock{mutex_};

    std::vector<u32> stopTokens;
    TF_TRY(const std::vector<u32> promptIds, renderPrompt(request, stopTokens));

    if (promptIds.size() >= contextLength_) {
        return makeError(ErrorCode::InvalidArgument,
                         "a {} token prompt does not fit the {} token context; start the "
                         "server with a larger --context",
                         promptIds.size(), contextLength_);
    }

    runtime::GenerationOptions generation;
    generation.sampling = request.sampling;
    generation.maxTokens = request.maxTokens;
    generation.stopStrings = request.stopStrings;
    generation.stopTokens = std::move(stopTokens);
    generation.prefillChunk = prefillChunk_;

    TF_TRY(const runtime::GenerationStats stats,
           generator_->generate(promptIds, generation, onText));

    stats_.requests += 1;
    stats_.promptTokens += stats.promptTokens;
    stats_.cachedPromptTokens += stats.cachedPromptTokens;
    stats_.generatedTokens += stats.generatedTokens;
    return stats;
}

// ---------------------------------------------------------------------------
// Routing
// ---------------------------------------------------------------------------

namespace {

/// Everything wrong with a request body is the caller's problem, whatever
/// internal code it surfaced as: malformed JSON, a missing field and an
/// unsupported option all mean "fix the request". Mapping them by internal code
/// would report a missing `messages` key as a 500 - blaming the server for the
/// client's mistake - or a missing JSON key as a 404, which is worse.
[[nodiscard]] u32 requestStatus(const Error&) { return 400; }

/// A failure during generation, where the internal code does carry meaning.
[[nodiscard]] u32 engineStatus(const Error& error) {
    switch (error.code()) {
        case ErrorCode::InvalidArgument:
            // A prompt that does not fit the context, for instance.
            return 400;
        case ErrorCode::Unsupported:
            return 501;
        case ErrorCode::OutOfMemory:
            // Transient: another request may well fit once this one is gone.
            return 503;
        default:
            return 500;
    }
}

void handleCompletion(Engine& engine, const Request& request, Connection& connection,
                      bool chat) {
    auto parsed = CompletionRequest::parse(request.body, chat);
    if (!parsed) {
        static_cast<void>(connection.sendResponse(
                Response::error(requestStatus(parsed.error()), parsed.error().message())));
        return;
    }

    const std::string id = makeCompletionId(chat);
    const std::string& model =
            parsed->model.empty() ? engine.modelName() : parsed->model;

    if (!parsed->stream) {
        std::string content;
        auto stats = engine.run(*parsed, [&](std::string_view piece) {
            content += piece;
            return true;
        });
        if (!stats) {
            static_cast<void>(connection.sendResponse(
                    Response::error(engineStatus(stats.error()), stats.error().message())));
            return;
        }

        const Usage usage{.promptTokens = stats->promptTokens,
                          .completionTokens = stats->generatedTokens};
        static_cast<void>(connection.sendResponse(Response::json(
                200, completionBody(id, model, content, stats->reason, usage, chat))));
        return;
    }

    // ---- Streaming -------------------------------------------------------
    //
    // Once the 200 is on the wire there is no way to report a failure as a
    // status, so everything that can be validated has been by now. A later
    // error can only end the stream.
    if (!connection.beginEventStream()) {
        return;
    }

    bool first = true;
    bool clientGone = false;
    auto stats = engine.run(*parsed, [&](std::string_view piece) {
        const std::string body = chunkBody(id, model, piece, first, nullptr, chat);
        first = false;
        if (!connection.sendEvent(body)) {
            // The client hung up. Returning false stops generation, which is
            // the point: an abandoned request should stop costing GPU time
            // rather than run to max_tokens.
            clientGone = true;
            return false;
        }
        return true;
    });

    if (clientGone) {
        return;
    }

    const runtime::StopReason reason =
            stats ? stats->reason : runtime::StopReason::Cancelled;
    if (!stats) {
        // Mid-stream failures are surfaced as an error frame. Clients do not
        // all parse it, but it is better than a stream that simply stops.
        static_cast<void>(connection.sendEvent(
                Response::error(engineStatus(stats.error()), stats.error().message()).body));
    } else if (first) {
        // Nothing was generated at all, so no chunk has carried the role yet.
        static_cast<void>(
                connection.sendEvent(chunkBody(id, model, "", true, nullptr, chat)));
    }

    static_cast<void>(connection.sendEvent(chunkBody(id, model, "", false, &reason, chat)));
    static_cast<void>(connection.endEventStream());
}

void handleHealth(Engine& engine, Connection& connection) {
    const Engine::Stats stats = engine.stats();

    json::Value root = json::Value::makeObject();
    root.set("status", "ok");
    root.set("model", engine.modelName());
    root.set("context_length", engine.contextLength());
    root.set("requests", stats.requests);
    root.set("prompt_tokens", stats.promptTokens);
    root.set("cached_prompt_tokens", stats.cachedPromptTokens);
    root.set("generated_tokens", stats.generatedTokens);
    static_cast<void>(connection.sendResponse(Response::json(200, root.dump())));
}

}  // namespace

void handleRequest(Engine& engine, const Request& request, Connection& connection) {
    const std::string_view path = request.path();

    if (request.method == "OPTIONS") {
        // Browsers preflight cross-origin requests. Answering keeps a local web
        // UI working without a proxy.
        static_cast<void>(connection.sendResponse(Response{.status = 204}));
        return;
    }

    if (path == "/health" || path == "/v1/health") {
        handleHealth(engine, connection);
        return;
    }

    if (path == "/v1/models") {
        if (request.method != "GET") {
            static_cast<void>(connection.sendResponse(
                    Response::error(405, "/v1/models accepts GET")));
            return;
        }
        static_cast<void>(
                connection.sendResponse(Response::json(200, modelsBody(engine.modelName()))));
        return;
    }

    if (path == "/v1/chat/completions" || path == "/v1/completions") {
        if (request.method != "POST") {
            static_cast<void>(connection.sendResponse(
                    Response::error(405, std::format("{} accepts POST", path))));
            return;
        }
        handleCompletion(engine, request, connection, path == "/v1/chat/completions");
        return;
    }

    static_cast<void>(connection.sendResponse(Response::error(
            404, std::format("no endpoint {}; this server serves /v1/chat/completions, "
                             "/v1/completions, /v1/models and /health",
                             path),
            "not_found")));
}

}  // namespace tf::server
