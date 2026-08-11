#include "Pipe.h"

#include <windows.h>

#include <atomic>
#include <format>
#include <utility>

namespace tf::svc {
namespace {

[[nodiscard]] Error lastError(std::string_view what) {
    const DWORD code = ::GetLastError();
    char* buffer = nullptr;
    const DWORD length = ::FormatMessageA(
            FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
                    FORMAT_MESSAGE_IGNORE_INSERTS,
            nullptr, code, 0, reinterpret_cast<char*>(&buffer), 0, nullptr);

    std::string detail{buffer != nullptr ? buffer : "", length};
    if (buffer != nullptr) {
        ::LocalFree(buffer);
    }
    while (!detail.empty() && (detail.back() == '\n' || detail.back() == '\r')) {
        detail.pop_back();
    }
    if (detail.empty()) {
        detail = std::format("error {}", code);
    }

    // A peer that closed is not a fault; the caller has to be able to tell.
    const ErrorCode kind = (code == ERROR_BROKEN_PIPE || code == ERROR_PIPE_NOT_CONNECTED ||
                            code == ERROR_NO_DATA)
                                   ? ErrorCode::Cancelled
                                   : ErrorCode::Io;
    return Error{kind, std::format("{}: {}", what, detail)};
}

[[nodiscard]] std::wstring widen(std::string_view text) {
    if (text.empty()) {
        return {};
    }
    const int needed = ::MultiByteToWideChar(CP_UTF8, 0, text.data(),
                                             static_cast<int>(text.size()), nullptr, 0);
    std::wstring wide(static_cast<usize>(needed), L'\0');
    ::MultiByteToWideChar(CP_UTF8, 0, text.data(), static_cast<int>(text.size()), wide.data(),
                          needed);
    return wide;
}

/// Pipe buffer. Large enough that a typical token message never blocks, small
/// enough that a stalled reader applies backpressure rather than letting the
/// service queue megabytes of tokens nobody is reading.
constexpr DWORD kPipeBufferBytes = 64 * 1024;

}  // namespace

// ---------------------------------------------------------------------------
// PipeChannel
// ---------------------------------------------------------------------------

/// Both ends of the channel do overlapped I/O, and the reason is a deadlock
/// rather than throughput.
///
/// Windows serializes operations on a *synchronous* handle. A duplex client
/// that reads on one thread and writes on another therefore deadlocks the
/// moment it tries to write: the reader is parked in ReadFile waiting for a
/// message the peer will only send once it has the request, and the write
/// queues behind that read forever. The GUI is exactly that shape - a reader
/// thread plus sends from the UI thread - and it froze on the first Send, with
/// the UI thread stuck in WriteFile.
///
/// A request/response client that writes then reads never overlaps the two and
/// so never sees it, which is why tf-decode-client was unaffected and the bug
/// looked like a GUI problem.
///
/// One event per direction is enough: reads and writes each happen on one
/// thread at a time, and they must not share an event or they would consume
/// each other's completion.
struct PipeChannel::Impl {
    HANDLE handle = INVALID_HANDLE_VALUE;
    HANDLE readEvent = nullptr;
    HANDLE writeEvent = nullptr;
    /// True on the service side, where disconnecting is part of closing.
    bool server = false;

    ~Impl() { close(); }

    [[nodiscard]] bool createEvents() {
        readEvent = ::CreateEventW(nullptr, TRUE, FALSE, nullptr);
        writeEvent = ::CreateEventW(nullptr, TRUE, FALSE, nullptr);
        return readEvent != nullptr && writeEvent != nullptr;
    }

    void close() {
        if (handle != INVALID_HANDLE_VALUE) {
            if (server) {
                ::FlushFileBuffers(handle);
                ::DisconnectNamedPipe(handle);
            }
            // Any operation still pending is cancelled by the close, and the
            // waiter is released with an error rather than left on a handle
            // that is about to disappear.
            ::CancelIoEx(handle, nullptr);
            ::CloseHandle(handle);
            handle = INVALID_HANDLE_VALUE;
        }
        for (HANDLE* event : {&readEvent, &writeEvent}) {
            if (*event != nullptr) {
                ::CloseHandle(*event);
                *event = nullptr;
            }
        }
    }

    /// Issues one overlapped operation and waits for it to finish.
    [[nodiscard]] Status transfer(bool writing, void* data, DWORD bytes, HANDLE event,
                                  DWORD& transferred) {
        ::ResetEvent(event);
        OVERLAPPED overlapped{};
        overlapped.hEvent = event;

        const BOOL started =
                writing ? ::WriteFile(handle, data, bytes, nullptr, &overlapped)
                        : ::ReadFile(handle, data, bytes, nullptr, &overlapped);
        if (started == 0 && ::GetLastError() != ERROR_IO_PENDING) {
            return std::unexpected(
                    lastError(writing ? "writing to the pipe" : "reading from the pipe"));
        }
        if (::GetOverlappedResult(handle, &overlapped, &transferred, TRUE) == 0) {
            return std::unexpected(
                    lastError(writing ? "writing to the pipe" : "reading from the pipe"));
        }
        return {};
    }

    [[nodiscard]] Status writeAll(const void* data, usize bytes) {
        auto* cursor = const_cast<u8*>(static_cast<const u8*>(data));
        usize written = 0;
        while (written < bytes) {
            DWORD chunk = 0;
            TF_CHECK(transfer(true, cursor + written, static_cast<DWORD>(bytes - written),
                              writeEvent, chunk));
            written += chunk;
        }
        return {};
    }

    [[nodiscard]] Status readAll(void* data, usize bytes) {
        auto* cursor = static_cast<u8*>(data);
        usize read = 0;
        while (read < bytes) {
            DWORD chunk = 0;
            TF_CHECK(transfer(false, cursor + read, static_cast<DWORD>(bytes - read),
                              readEvent, chunk));
            if (chunk == 0) {
                return makeError(ErrorCode::Cancelled, "the peer closed the pipe");
            }
            read += chunk;
        }
        return {};
    }
};

PipeChannel::PipeChannel() : impl_(std::make_unique<Impl>()) {}
PipeChannel::~PipeChannel() = default;
PipeChannel::PipeChannel(PipeChannel&&) noexcept = default;
PipeChannel& PipeChannel::operator=(PipeChannel&&) noexcept = default;

bool PipeChannel::isOpen() const noexcept {
    return impl_ != nullptr && impl_->handle != INVALID_HANDLE_VALUE;
}

void PipeChannel::close() {
    if (impl_) {
        impl_->close();
    }
}

Status PipeChannel::write(const Message& message) {
    if (!isOpen()) {
        return makeError(ErrorCode::Cancelled, "writing to a closed pipe");
    }
    const std::string framed = frame(message.encode());
    return impl_->writeAll(framed.data(), framed.size());
}

Result<Message> PipeChannel::read() {
    if (!isOpen()) {
        return makeError(ErrorCode::Cancelled, "reading from a closed pipe");
    }

    u8 header[4] = {};
    TF_CHECK(impl_->readAll(header, sizeof(header)));

    const u32 length = static_cast<u32>(header[0]) | (static_cast<u32>(header[1]) << 8) |
                       (static_cast<u32>(header[2]) << 16) |
                       (static_cast<u32>(header[3]) << 24);
    if (length == 0) {
        return makeError(ErrorCode::MalformedData, "a zero-length message");
    }
    if (length > kMaxMessageBytes) {
        // A peer that declares a huge message must not be able to make this
        // process allocate it.
        return makeError(ErrorCode::MalformedData,
                         "a {} byte message exceeds the {} byte limit", length,
                         kMaxMessageBytes);
    }

    std::string payload(static_cast<usize>(length), '\0');
    TF_CHECK(impl_->readAll(payload.data(), payload.size()));
    return Message::decode(payload);
}

// ---------------------------------------------------------------------------
// PipeServer
// ---------------------------------------------------------------------------

struct PipeServer::Impl {
    std::string name;
    std::wstring wideName;
    HANDLE pending = INVALID_HANDLE_VALUE;
    HANDLE stopEvent = nullptr;
    HANDLE connectEvent = nullptr;
    std::atomic<bool> stopping{false};

    ~Impl() {
        if (pending != INVALID_HANDLE_VALUE) {
            ::CloseHandle(pending);
        }
        if (stopEvent != nullptr) {
            ::CloseHandle(stopEvent);
        }
        if (connectEvent != nullptr) {
            ::CloseHandle(connectEvent);
        }
    }
};

PipeServer::PipeServer() : impl_(std::make_unique<Impl>()) {}
PipeServer::~PipeServer() = default;
PipeServer::PipeServer(PipeServer&&) noexcept = default;
PipeServer& PipeServer::operator=(PipeServer&&) noexcept = default;

const std::string& PipeServer::name() const noexcept { return impl_->name; }

Result<PipeServer> PipeServer::create(const std::string& pipeName) {
    PipeServer server;
    server.impl_->name = pipeName;
    server.impl_->wideName = widen(pipeName);

    server.impl_->stopEvent = ::CreateEventW(nullptr, TRUE, FALSE, nullptr);
    server.impl_->connectEvent = ::CreateEventW(nullptr, TRUE, FALSE, nullptr);
    if (server.impl_->stopEvent == nullptr || server.impl_->connectEvent == nullptr) {
        return std::unexpected(lastError("creating the pipe events"));
    }

    // FILE_FLAG_FIRST_PIPE_INSTANCE is what makes a second service fail here
    // rather than quietly creating a second instance that some clients would
    // connect to instead.
    server.impl_->pending = ::CreateNamedPipeW(
            server.impl_->wideName.c_str(),
            PIPE_ACCESS_DUPLEX | FILE_FLAG_OVERLAPPED | FILE_FLAG_FIRST_PIPE_INSTANCE,
            PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT, 1, kPipeBufferBytes,
            kPipeBufferBytes, 0, nullptr);
    if (server.impl_->pending == INVALID_HANDLE_VALUE) {
        const DWORD code = ::GetLastError();
        if (code == ERROR_ACCESS_DENIED || code == ERROR_PIPE_BUSY) {
            return makeError(ErrorCode::Io,
                             "a decode service is already listening on {}", pipeName);
        }
        return std::unexpected(lastError(std::format("creating the pipe {}", pipeName)));
    }
    return server;
}

Result<PipeChannel> PipeServer::accept() {
    if (impl_->stopping.load(std::memory_order_relaxed)) {
        return makeError(ErrorCode::Cancelled, "the pipe server is stopping");
    }
    if (impl_->pending == INVALID_HANDLE_VALUE) {
        // A previous accept handed the instance away; make another.
        impl_->pending = ::CreateNamedPipeW(
                impl_->wideName.c_str(), PIPE_ACCESS_DUPLEX | FILE_FLAG_OVERLAPPED,
                PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT, 1, kPipeBufferBytes,
                kPipeBufferBytes, 0, nullptr);
        if (impl_->pending == INVALID_HANDLE_VALUE) {
            return std::unexpected(lastError("recreating the pipe"));
        }
    }

    // Overlapped so the wait can be interrupted by stop(); a blocking
    // ConnectNamedPipe could only be broken by connecting to ourselves.
    ::ResetEvent(impl_->connectEvent);
    OVERLAPPED overlapped{};
    overlapped.hEvent = impl_->connectEvent;

    bool connected = ::ConnectNamedPipe(impl_->pending, &overlapped) != 0;
    if (!connected) {
        const DWORD code = ::GetLastError();
        if (code == ERROR_PIPE_CONNECTED) {
            // The client beat us to it, which is a success.
            connected = true;
        } else if (code == ERROR_IO_PENDING) {
            HANDLE waits[2] = {impl_->connectEvent, impl_->stopEvent};
            const DWORD signalled = ::WaitForMultipleObjects(2, waits, FALSE, INFINITE);
            if (signalled == WAIT_OBJECT_0 + 1) {
                ::CancelIo(impl_->pending);
                return makeError(ErrorCode::Cancelled, "the pipe server was stopped");
            }
            DWORD transferred = 0;
            connected = ::GetOverlappedResult(impl_->pending, &overlapped, &transferred,
                                              FALSE) != 0;
        } else {
            return std::unexpected(lastError("waiting for a connection"));
        }
    }
    if (!connected) {
        return std::unexpected(lastError("accepting a connection"));
    }

    PipeChannel channel;
    channel.impl_->handle = std::exchange(impl_->pending, INVALID_HANDLE_VALUE);
    channel.impl_->server = true;
    // The listening handle was already created FILE_FLAG_OVERLAPPED so the
    // accept could be interrupted, which means every transfer on it has to
    // carry an OVERLAPPED too. Passing nullptr on an asynchronous handle
    // returns before the operation finishes and reports a byte count that has
    // not been written yet.
    if (!channel.impl_->createEvents()) {
        return std::unexpected(lastError("creating the pipe I/O events"));
    }
    return channel;
}

void PipeServer::stop() {
    if (impl_ == nullptr) {
        return;
    }
    impl_->stopping.store(true, std::memory_order_relaxed);
    if (impl_->stopEvent != nullptr) {
        ::SetEvent(impl_->stopEvent);
    }
}

// ---------------------------------------------------------------------------
// PipeClient
// ---------------------------------------------------------------------------

Result<PipeChannel> PipeClient::connect(const std::string& pipeName, u32 timeoutMillis) {
    const std::wstring wide = widen(pipeName);
    const ULONGLONG deadline = ::GetTickCount64() + timeoutMillis;

    for (;;) {
        // FILE_FLAG_OVERLAPPED, without which a client that reads on one thread
        // and writes on another deadlocks on its first write. See PipeChannel::Impl.
        const HANDLE handle =
                ::CreateFileW(wide.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr,
                              OPEN_EXISTING, FILE_FLAG_OVERLAPPED, nullptr);
        if (handle != INVALID_HANDLE_VALUE) {
            PipeChannel channel;
            channel.impl_->handle = handle;
            channel.impl_->server = false;
            if (!channel.impl_->createEvents()) {
                return std::unexpected(lastError("creating the pipe I/O events"));
            }
            return channel;
        }

        const DWORD code = ::GetLastError();
        // ERROR_FILE_NOT_FOUND means the service has not created the pipe yet,
        // which is normal while it loads a 13 GiB model. ERROR_PIPE_BUSY means
        // it is serving someone else.
        if (code != ERROR_FILE_NOT_FOUND && code != ERROR_PIPE_BUSY) {
            return std::unexpected(lastError(std::format("connecting to {}", pipeName)));
        }
        if (::GetTickCount64() >= deadline) {
            return makeError(ErrorCode::Cancelled,
                             "no decode service answered on {} within {} ms", pipeName,
                             timeoutMillis);
        }
        ::Sleep(50);
    }
}

bool PipeClient::isRunning(const std::string& pipeName) {
    // WaitNamedPipe with a zero timeout answers "does this exist" without
    // taking the instance, so it does not race with a real client connecting.
    return ::WaitNamedPipeW(widen(pipeName).c_str(), 0) != 0 ||
           ::GetLastError() == ERROR_SEM_TIMEOUT;
}

}  // namespace tf::svc
