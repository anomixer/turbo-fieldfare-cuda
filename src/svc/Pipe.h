#pragma once

#include <memory>
#include <string>

#include "Protocol.h"
#include "tf/core/base/Error.h"
#include "tf/core/base/Types.h"

namespace tf::svc {

/// One end of a connected named pipe, speaking framed messages.
///
/// Reads and writes may be used from different threads - the service writes
/// tokens from its decode loop while its reader thread waits for a cancel - but
/// each direction is single-threaded. That is enough: the alternative, locking
/// the write side, would make a cancel wait for the token it is trying to
/// interrupt.
class PipeChannel {
public:
    PipeChannel();
    ~PipeChannel();

    PipeChannel(const PipeChannel&) = delete;
    PipeChannel& operator=(const PipeChannel&) = delete;
    PipeChannel(PipeChannel&&) noexcept;
    PipeChannel& operator=(PipeChannel&&) noexcept;

    /// Reads one message. A Cancelled error means the peer closed cleanly,
    /// which is how a front end exiting is distinguished from a fault.
    [[nodiscard]] Result<Message> read();

    [[nodiscard]] Status write(const Message& message);

    [[nodiscard]] bool isOpen() const noexcept;
    void close();

private:
    friend class PipeServer;
    friend class PipeClient;

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/// Listens on a named pipe.
///
/// One connection is served at a time. The service holds a single model with a
/// single KV cache, so a second front end would either interleave into the
/// first one's conversation or need its own cache; making it wait is the honest
/// behaviour.
class PipeServer {
public:
    PipeServer();
    ~PipeServer();

    PipeServer(const PipeServer&) = delete;
    PipeServer& operator=(const PipeServer&) = delete;
    PipeServer(PipeServer&&) noexcept;
    PipeServer& operator=(PipeServer&&) noexcept;

    /// Fails when another service already owns the name, which is how a second
    /// launch reports "already running" rather than silently doing nothing.
    [[nodiscard]] static Result<PipeServer> create(const std::string& pipeName);

    /// Waits for a front end. Returns a Cancelled error after stop().
    [[nodiscard]] Result<PipeChannel> accept();

    /// Wakes a blocked accept(). Safe from another thread or a console handler.
    void stop();

    [[nodiscard]] const std::string& name() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/// Connects to a running service.
class PipeClient {
public:
    /// `timeoutMillis` covers the case where the service is starting up and has
    /// created the pipe but not yet accepted, as well as where it is loading a
    /// model and has not created it at all.
    [[nodiscard]] static Result<PipeChannel> connect(const std::string& pipeName,
                                                     u32 timeoutMillis = 5000);

    /// True when a service is listening. Used to decide whether to start one.
    [[nodiscard]] static bool isRunning(const std::string& pipeName);
};

}  // namespace tf::svc
