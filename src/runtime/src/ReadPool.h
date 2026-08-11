#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

#include "tf/core/base/Types.h"

namespace tf::runtime {

/// A fixed pool that runs an indexed job across its workers and blocks until
/// every index has been handled.
///
/// Exists because expert fetches happen ~30 times per token: creating threads
/// per fetch would mean thousands of thread creations per second. The workers
/// are parked on a condition variable between jobs, so an idle pool costs
/// nothing.
///
/// One job at a time. The caller is the only submitter, which is true of the
/// decode loop, so there is no queue to manage.
class ReadPool {
public:
    explicit ReadPool(u32 threadCount) {
        const u32 threads = threadCount == 0 ? 1 : threadCount;
        workers_.reserve(threads);
        for (u32 i = 0; i < threads; ++i) {
            workers_.emplace_back([this] { workerLoop(); });
        }
    }

    ~ReadPool() {
        {
            const std::lock_guard lock{mutex_};
            stopping_ = true;
        }
        wake_.notify_all();
        for (auto& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

    ReadPool(const ReadPool&) = delete;
    ReadPool& operator=(const ReadPool&) = delete;

    /// Runs `body(i)` for every i in [0, count) and returns once all have
    /// finished. The calling thread participates, so a count of one costs no
    /// hand-off at all.
    void parallelFor(u32 count, const std::function<void(u32)>& body) {
        if (count == 0) {
            return;
        }
        if (count == 1 || workers_.empty()) {
            body(0);
            return;
        }

        {
            const std::lock_guard lock{mutex_};
            body_ = &body;
            count_ = count;
            next_.store(0, std::memory_order_relaxed);
            remaining_.store(count, std::memory_order_relaxed);
            ++generation_;
        }
        wake_.notify_all();

        // The submitting thread pulls work too rather than idling.
        runIndices();

        std::unique_lock lock{mutex_};
        done_.wait(lock, [this] { return remaining_.load(std::memory_order_acquire) == 0; });
        body_ = nullptr;
    }

private:
    void runIndices() {
        while (true) {
            const u32 index = next_.fetch_add(1, std::memory_order_relaxed);
            if (index >= count_) {
                return;
            }
            (*body_)(index);
            if (remaining_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                const std::lock_guard lock{mutex_};
                done_.notify_all();
            }
        }
    }

    void workerLoop() {
        u64 seen = 0;
        while (true) {
            {
                std::unique_lock lock{mutex_};
                wake_.wait(lock, [this, seen] { return stopping_ || generation_ != seen; });
                if (stopping_) {
                    return;
                }
                seen = generation_;
            }
            runIndices();
        }
    }

    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable wake_;
    std::condition_variable done_;

    const std::function<void(u32)>* body_ = nullptr;
    u32 count_ = 0;
    std::atomic<u32> next_{0};
    std::atomic<u32> remaining_{0};
    u64 generation_ = 0;
    bool stopping_ = false;
};

}  // namespace tf::runtime
