#include "AppModel.h"

#include <windows.h>

#include <format>
#include <mutex>

namespace tf::gui {
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

/// Same TF_GUI_LOG trace as main.cpp, for the pipe thread.
///
/// The lock is the point: the UI thread and the pipe thread both trace, and
/// unsynchronized appends interleave mid-line, which makes the log lie exactly
/// when it is being used to work out what two threads did to each other.
void trace(std::string_view what) {
    static std::mutex mutex;
    const std::lock_guard lock{mutex};

    wchar_t path[1024] = {};
    if (::GetEnvironmentVariableW(L"TF_GUI_LOG", path, 1024) == 0) {
        return;
    }
    if (FILE* handle = ::_wfopen(path, L"a"); handle != nullptr) {
        std::fwrite(what.data(), 1, what.size(), handle);
        std::fputc('\n', handle);
        std::fclose(handle);
    }
}

}  // namespace

AppModel::AppModel() {
    settings.sampling.temperature = 0.2f;
    settings.maxTokens = 512;
}

AppModel::~AppModel() { disconnect(); }

Status AppModel::startService(const std::wstring& servicePath, const std::wstring& modelDir) {
    // --exit-when-idle: a service this window started should not outlive it.
    std::wstring command = std::format(LR"("{}" --model "{}" --exit-when-idle)", servicePath,
                                       modelDir);

    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    PROCESS_INFORMATION process{};

    if (::CreateProcessW(nullptr, command.data(), nullptr, nullptr, FALSE,
                         CREATE_NO_WINDOW, nullptr, nullptr, &startup, &process) == 0) {
        return makeError(ErrorCode::Io, "could not start {}", narrow(servicePath));
    }
    ::CloseHandle(process.hThread);
    serviceProcess_ = process.hProcess;
    return {};
}

Status AppModel::connectToService(const std::wstring& servicePath,
                                  const std::wstring& modelDir) {
    const std::string pipeName = svc::defaultPipeName();

    if (!svc::PipeClient::isRunning(pipeName) && !servicePath.empty()) {
        TF_CHECK(startService(servicePath, modelDir));
    }

    // Generous: the service has to load 13 GiB before it accepts, which takes
    // several seconds from a warm page cache and longer from cold disk.
    TF_TRY(svc::PipeChannel channel, svc::PipeClient::connect(pipeName, 120000));
    channel_ = std::make_unique<svc::PipeChannel>(std::move(channel));

    connected_.store(true);
    stopping_.store(false);
    reader_ = std::thread{[this] { readLoop(); }};
    return {};
}

void AppModel::disconnect() {
    stopping_.store(true);
    if (channel_) {
        channel_->close();
    }
    if (reader_.joinable()) {
        reader_.join();
    }
    channel_.reset();
    connected_.store(false);

    if (serviceProcess_ != nullptr) {
        // The service exits on its own once the pipe closes, because it was
        // started with --exit-when-idle. Waiting briefly keeps a visible
        // process from lingering after the window is gone; killing it after
        // that is better than leaking 13 GiB.
        const auto handle = static_cast<HANDLE>(serviceProcess_);
        if (::WaitForSingleObject(handle, 3000) == WAIT_TIMEOUT) {
            ::TerminateProcess(handle, 0);
        }
        ::CloseHandle(handle);
        serviceProcess_ = nullptr;
    }
}

void AppModel::readLoop() {
    trace("readLoop start");
    while (!stopping_.load()) {
        auto incoming = channel_->read();
        if (!incoming) {
            trace(std::format("read failed: {}", incoming.error().message()));
            break;
        }
        trace(std::format("message kind {}", static_cast<int>(incoming->kind)));

        switch (incoming->kind) {
            case svc::MessageKind::Ready:
                trace(callbacks_.onReady ? "ready, dispatching" : "ready, NO CALLBACK");
                if (callbacks_.onReady) {
                    callbacks_.onReady(incoming->ready);
                }
                break;

            case svc::MessageKind::Token:
                if (incoming->id == currentRequest_ && callbacks_.onToken) {
                    callbacks_.onToken(std::move(incoming->text));
                }
                break;

            case svc::MessageKind::Done:
                busy_.store(false);
                if (callbacks_.onDone) {
                    callbacks_.onDone(incoming->done);
                }
                break;

            case svc::MessageKind::Error:
                busy_.store(false);
                if (callbacks_.onError) {
                    callbacks_.onError(std::move(incoming->text));
                }
                break;

            default:
                break;
        }
    }

    busy_.store(false);
    connected_.store(false);
    if (!stopping_.load() && callbacks_.onDisconnected) {
        callbacks_.onDisconnected();
    }
}

Status AppModel::send(const std::string& prompt) {
    if (!connected_.load()) {
        return makeError(ErrorCode::Cancelled, "not connected to a decode service");
    }
    if (busy_.load()) {
        return makeError(ErrorCode::InvalidArgument, "a generation is already running");
    }

    turns_.push_back(Turn{.fromUser = true, .text = prompt});
    turns_.push_back(Turn{.fromUser = false, .text = {}});

    svc::Message message{.kind = svc::MessageKind::Generate};
    message.generate = settings;
    message.generate.id = nextRequestId_.fetch_add(1);
    message.generate.chat = true;
    message.generate.prompt.clear();

    // The whole conversation goes every time, which is what lets the service's
    // prompt cache reuse all of it but the newest turn. Sending only the new
    // message would be smaller on the wire and far slower to answer.
    message.generate.messages.clear();
    for (const Turn& turn : turns_) {
        if (turn.text.empty() && !turn.fromUser) {
            continue;  // the answer being generated now
        }
        message.generate.messages.push_back(
                ChatMessage{turn.fromUser ? ChatRole::User : ChatRole::Model, turn.text});
    }

    currentRequest_ = message.generate.id;
    busy_.store(true);

    const std::lock_guard lock{writeMutex_};
    const auto sent = channel_->write(message);
    if (!sent) {
        busy_.store(false);
        return std::unexpected(sent.error());
    }
    return {};
}

void AppModel::cancel() {
    if (!connected_.load() || !busy_.load()) {
        return;
    }
    svc::Message message{.kind = svc::MessageKind::Cancel};
    message.id = currentRequest_;

    const std::lock_guard lock{writeMutex_};
    static_cast<void>(channel_->write(message));
}

Status AppModel::reset() {
    turns_.clear();
    if (!connected_.load()) {
        return {};
    }
    const std::lock_guard lock{writeMutex_};
    return channel_->write(svc::Message{.kind = svc::MessageKind::Reset});
}

void AppModel::appendToAnswer(std::string_view piece) {
    if (turns_.empty() || turns_.back().fromUser) {
        turns_.push_back(Turn{.fromUser = false, .text = {}});
    }
    turns_.back().text += piece;
}

}  // namespace tf::gui
