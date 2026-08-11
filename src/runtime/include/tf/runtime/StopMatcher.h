#pragma once

#include <string>
#include <string_view>
#include <vector>

#include "tf/core/base/Types.h"

namespace tf::runtime {

/// Watches streamed text for stop sequences that may straddle token boundaries.
///
/// A stop string is not a token. "\n\nUser:" can arrive as "\n\n", "User" and
/// ":" across three steps, so matching each decoded piece in isolation misses
/// it. Equally, text that *might* still become a stop string must not be
/// emitted yet - printing it and retracting it afterwards is not possible once
/// it has gone to a terminal or an HTTP stream.
///
/// So the matcher holds back the longest suffix of what it has seen that is a
/// proper prefix of some stop string, and releases it as soon as the next piece
/// proves it was not the start of one.
class StopMatcher {
public:
    StopMatcher() = default;

    /// Empty stop strings are dropped: they would match immediately and stop
    /// generation before it began.
    explicit StopMatcher(std::vector<std::string> stopStrings);

    struct Update {
        /// Text that is now safe to show.
        std::string emit;
        /// True once a stop string completed. `emit` holds everything before
        /// it; the stop string itself is never emitted.
        bool stopped = false;
    };

    [[nodiscard]] Update push(std::string_view text);

    /// Releases whatever is still held back, for when generation ends without
    /// hitting a stop string - a token limit, or an end-of-turn token.
    [[nodiscard]] std::string flush();

    /// True when there are no stop strings at all, in which case push() always
    /// emits its whole input and holds nothing.
    [[nodiscard]] bool empty() const noexcept { return stopStrings_.empty(); }

private:
    /// Length of the longest suffix of `pending_` that is a proper prefix of
    /// some stop string.
    [[nodiscard]] usize heldSuffixLength() const;

    std::vector<std::string> stopStrings_;
    /// Longest stop string, which bounds how much has to be kept.
    usize longest_ = 0;
    /// Text seen but not yet emitted.
    std::string pending_;
};

}  // namespace tf::runtime
