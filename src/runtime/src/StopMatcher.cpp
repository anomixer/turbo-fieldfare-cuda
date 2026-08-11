#include "tf/runtime/StopMatcher.h"

#include <algorithm>

namespace tf::runtime {

StopMatcher::StopMatcher(std::vector<std::string> stopStrings)
    : stopStrings_(std::move(stopStrings)) {
    // An empty stop string is a prefix of everything and would end generation
    // before the first token.
    std::erase_if(stopStrings_, [](const std::string& stop) { return stop.empty(); });
    for (const std::string& stop : stopStrings_) {
        longest_ = std::max(longest_, stop.size());
    }
}

usize StopMatcher::heldSuffixLength() const {
    // A suffix longer than the longest stop string cannot be a proper prefix of
    // one, so only that much of the tail is worth testing.
    const usize limit = std::min(pending_.size(), longest_ > 0 ? longest_ - 1 : 0);
    for (usize length = limit; length > 0; --length) {
        const std::string_view suffix{pending_.data() + (pending_.size() - length), length};
        for (const std::string& stop : stopStrings_) {
            if (stop.size() > length && std::string_view{stop}.substr(0, length) == suffix) {
                return length;
            }
        }
    }
    return 0;
}

StopMatcher::Update StopMatcher::push(std::string_view text) {
    if (stopStrings_.empty()) {
        return Update{.emit = std::string{text}, .stopped = false};
    }

    // Where the search has to start: a stop string can begin inside what was
    // already held back and finish inside the new text.
    const usize searchFrom = pending_.size() >= longest_ ? pending_.size() - longest_ + 1 : 0;
    pending_ += text;

    usize earliest = std::string::npos;
    for (const std::string& stop : stopStrings_) {
        const usize found = pending_.find(stop, searchFrom);
        if (found != std::string::npos) {
            earliest = std::min(earliest, found);
        }
    }

    if (earliest != std::string::npos) {
        Update update{.emit = pending_.substr(0, earliest), .stopped = true};
        pending_.clear();
        return update;
    }

    const usize held = heldSuffixLength();
    Update update{.emit = pending_.substr(0, pending_.size() - held), .stopped = false};
    pending_.erase(0, pending_.size() - held);
    return update;
}

std::string StopMatcher::flush() {
    std::string remaining = std::move(pending_);
    pending_.clear();
    return remaining;
}

}  // namespace tf::runtime
