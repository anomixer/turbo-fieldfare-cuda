#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "Pipe.h"
#include "Protocol.h"
#include "tf/core/base/Error.h"

namespace tf::gui {

/// One turn in the conversation, as the UI shows it.
struct Turn {
    bool fromUser = false;
    std::string text;
};

/// Everything the window needs to know that is not a control.
///
/// Deliberately free of any XAML type. The UI observes it through callbacks, so
/// the conversation logic and the service connection can be reasoned about -
/// and changed - without touching the visual tree.
class AppModel {
public:
    AppModel();
    ~AppModel();

    AppModel(const AppModel&) = delete;
    AppModel& operator=(const AppModel&) = delete;

    /// Called on the reader thread. The window wraps these to hop onto the UI
    /// thread, which is where every XAML property must be touched.
    struct Callbacks {
        std::function<void(const svc::ReadyInfo&)> onReady;
        /// A piece of the answer to the current turn.
        std::function<void(std::string)> onToken;
        std::function<void(const svc::DoneInfo&)> onDone;
        std::function<void(std::string)> onError;
        /// The connection dropped, usually because the service exited.
        std::function<void()> onDisconnected;
    };

    void setCallbacks(Callbacks callbacks) { callbacks_ = std::move(callbacks); }

    /// Connects to a running service, starting one if `servicePath` is given
    /// and nothing is listening.
    ///
    /// Starting it here rather than requiring the user to is the difference
    /// between an application and a pair of programs, but the started service
    /// exits with the window: a stray 13 GiB process nobody can see is worse
    /// than a slow start next time.
    [[nodiscard]] Status connectToService(const std::wstring& servicePath,
                                          const std::wstring& modelDir);

    /// Sends the whole conversation plus `prompt`, so the service's prompt
    /// cache can reuse everything but the new turn.
    [[nodiscard]] Status send(const std::string& prompt);

    /// Asks the service to stop the current generation. Takes effect within a
    /// token, because the service reads on its own thread.
    void cancel();

    /// Clears the conversation and the service's KV cache.
    [[nodiscard]] Status reset();

    void disconnect();

    [[nodiscard]] bool connected() const noexcept { return connected_.load(); }
    [[nodiscard]] bool busy() const noexcept { return busy_.load(); }

    /// The conversation so far. Only touched from the UI thread.
    [[nodiscard]] const std::vector<Turn>& turns() const noexcept { return turns_; }

    /// Appends to the last assistant turn, or starts one.
    void appendToAnswer(std::string_view piece);

    svc::GenerateRequest settings;

private:
    void readLoop();
    [[nodiscard]] Status startService(const std::wstring& servicePath,
                                      const std::wstring& modelDir);

    std::unique_ptr<svc::PipeChannel> channel_;
    std::thread reader_;
    std::atomic<bool> connected_{false};
    std::atomic<bool> busy_{false};
    std::atomic<bool> stopping_{false};
    std::atomic<u64> nextRequestId_{1};
    u64 currentRequest_ = 0;

    /// Only the UI thread writes messages, so the write side stays
    /// single-threaded as PipeChannel requires.
    std::mutex writeMutex_;

    Callbacks callbacks_;
    std::vector<Turn> turns_;

    /// Handle to a service this process started, so it can be ended with the
    /// window.
    void* serviceProcess_ = nullptr;
};

}  // namespace tf::gui
