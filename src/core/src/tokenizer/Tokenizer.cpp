#include "tf/core/tokenizer/Tokenizer.h"

#include <algorithm>
#include <array>
#include <queue>

#include "tf/core/io/File.h"
#include "tf/core/json/Json.h"

namespace tf {
namespace {

/// SentencePiece writes a space as U+2581 LOWER ONE EIGHTH BLOCK.
constexpr std::string_view kSpaceMarker = "\xE2\x96\x81";

[[nodiscard]] u64 mergeKey(u32 left, u32 right) {
    return (static_cast<u64>(left) << 32) | right;
}

/// Length in bytes of the UTF-8 sequence starting with `lead`. Returns 1 for a
/// malformed lead byte so that decomposition always makes progress.
[[nodiscard]] usize utf8Length(u8 lead) {
    if ((lead & 0x80u) == 0) {
        return 1;
    }
    if ((lead & 0xE0u) == 0xC0u) {
        return 2;
    }
    if ((lead & 0xF0u) == 0xE0u) {
        return 3;
    }
    if ((lead & 0xF8u) == 0xF0u) {
        return 4;
    }
    return 1;
}

[[nodiscard]] std::string byteTokenName(u8 value) {
    static constexpr char kHex[] = "0123456789ABCDEF";
    std::string name = "<0x";
    name.push_back(kHex[value >> 4]);
    name.push_back(kHex[value & 0x0F]);
    name.push_back('>');
    return name;
}

/// Recognises `<0xXX>` and returns the byte it stands for.
[[nodiscard]] bool parseByteToken(std::string_view piece, u8& value) {
    if (piece.size() != 6 || !piece.starts_with("<0x") || piece.back() != '>') {
        return false;
    }
    const auto hex = [](char c) -> int {
        if (c >= '0' && c <= '9') {
            return c - '0';
        }
        if (c >= 'A' && c <= 'F') {
            return c - 'A' + 10;
        }
        if (c >= 'a' && c <= 'f') {
            return c - 'a' + 10;
        }
        return -1;
    };
    const int high = hex(piece[3]);
    const int low = hex(piece[4]);
    if (high < 0 || low < 0) {
        return false;
    }
    value = static_cast<u8>((high << 4) | low);
    return true;
}

/// One candidate merge, ordered by rank so the lowest is applied first.
struct Candidate {
    u32 rank = 0;
    u32 left = 0;   ///< index into the symbol list
    u32 right = 0;
    u32 mergedId = 0;
    /// Versions of the endpoints when queued, so a stale entry is detectable.
    u32 leftVersion = 0;
    u32 rightVersion = 0;

    bool operator>(const Candidate& other) const { return rank > other.rank; }
};

}  // namespace

Result<Tokenizer> Tokenizer::loadFromFile(const std::filesystem::path& path) {
    // tokenizer.json is 30 MB for this checkpoint, dominated by the merge list.
    TF_TRY(const std::string text, io::readTextFile(path, 512ull * 1024 * 1024));
    auto tokenizer = parse(text);
    if (!tokenizer) {
        return std::unexpected(tokenizer.error().wrap(path.string()));
    }
    return tokenizer;
}

Result<Tokenizer> Tokenizer::parse(std::string_view json) {
    TF_TRY(const json::Value root, json::parse(json));

    TF_TRY(const json::Value* model, root.at("model"));
    TF_TRY(const std::string_view type, (*model->at("type"))->asString());
    if (type != "BPE") {
        return makeError(ErrorCode::Unsupported,
                         "tokenizer model type '{}' is not supported (expected BPE)", type);
    }

    Tokenizer tokenizer;

    // ---- Vocabulary ------------------------------------------------------
    TF_TRY(const json::Value* vocabValue, model->at("vocab"));
    TF_TRY(const json::Object* vocab, vocabValue->asObject());

    tokenizer.pieceToId_.reserve(vocab->size() * 2);
    for (const auto& [piece, idValue] : *vocab) {
        TF_TRY(const u64 id, idValue.asUInt());
        if (id >= tokenizer.idToPiece_.size()) {
            tokenizer.idToPiece_.resize(static_cast<usize>(id) + 1);
        }
        tokenizer.idToPiece_[static_cast<usize>(id)] = piece;
        tokenizer.pieceToId_.emplace(piece, static_cast<u32>(id));
    }
    if (tokenizer.idToPiece_.empty()) {
        return makeError(ErrorCode::MalformedData, "tokenizer vocabulary is empty");
    }
    tokenizer.special_.assign(tokenizer.idToPiece_.size(), false);

    // ---- Byte fallback ---------------------------------------------------
    if (const json::Value* fallback = model->find("byte_fallback")) {
        tokenizer.hasByteFallback_ = fallback->asBool().value_or(false);
    }
    if (tokenizer.hasByteFallback_) {
        tokenizer.byteFallbackIds_.assign(256, 0);
        for (u32 value = 0; value < 256; ++value) {
            const auto it = tokenizer.pieceToId_.find(byteTokenName(static_cast<u8>(value)));
            if (it == tokenizer.pieceToId_.end()) {
                return makeError(ErrorCode::MalformedData,
                                 "byte_fallback is set but the vocabulary has no {} token",
                                 byteTokenName(static_cast<u8>(value)));
            }
            tokenizer.byteFallbackIds_[value] = it->second;
        }
    }

    if (const json::Value* unk = model->find("unk_token")) {
        if (const auto name = unk->asString(); name.has_value()) {
            if (const auto it = tokenizer.pieceToId_.find(std::string{*name});
                it != tokenizer.pieceToId_.end()) {
                tokenizer.unknownId_ = it->second;
            }
        }
    }

    // ---- Merges ----------------------------------------------------------
    //
    // Rank is position in the list, so the earliest applicable merge wins. Both
    // halves must already be vocabulary entries; a pair naming something
    // unknown can never fire and is skipped rather than treated as corruption.
    TF_TRY(const json::Value* mergesValue, model->at("merges"));
    TF_TRY(const json::Array* merges, mergesValue->asArray());

    tokenizer.mergeRanks_.reserve(merges->size() * 2);
    for (u32 rank = 0; rank < merges->size(); ++rank) {
        const json::Value& entry = (*merges)[rank];

        std::string_view left;
        std::string_view right;
        if (const auto pair = entry.asArray(); pair.has_value()) {
            if ((*pair)->size() != 2) {
                return makeError(ErrorCode::MalformedData,
                                 "merge {} has {} elements, expected 2", rank,
                                 (*pair)->size());
            }
            TF_TRY(left, (**pair)[0].asString());
            TF_TRY(right, (**pair)[1].asString());
        } else {
            // Older exports store "left right" as a single space-separated
            // string.
            TF_TRY(const std::string_view text, entry.asString());
            const auto gap = text.find(' ');
            if (gap == std::string_view::npos) {
                return makeError(ErrorCode::MalformedData, "merge {} has no separator", rank);
            }
            left = text.substr(0, gap);
            right = text.substr(gap + 1);
        }

        const auto leftIt = tokenizer.pieceToId_.find(std::string{left});
        const auto rightIt = tokenizer.pieceToId_.find(std::string{right});
        if (leftIt == tokenizer.pieceToId_.end() || rightIt == tokenizer.pieceToId_.end()) {
            continue;
        }

        std::string merged;
        merged.reserve(left.size() + right.size());
        merged.append(left);
        merged.append(right);

        const auto mergedIt = tokenizer.pieceToId_.find(merged);
        if (mergedIt == tokenizer.pieceToId_.end()) {
            continue;
        }

        tokenizer.mergeRanks_.emplace(mergeKey(leftIt->second, rightIt->second), rank);
    }

    // ---- Added tokens ----------------------------------------------------
    if (const json::Value* added = root.find("added_tokens")) {
        if (const auto entries = added->asArray(); entries.has_value()) {
            for (const auto& entry : **entries) {
                AddedToken token;
                TF_TRY(const u64 id, (*entry.at("id"))->asUInt());
                token.id = static_cast<u32>(id);
                TF_TRY(const std::string_view content, (*entry.at("content"))->asString());
                token.content = content;
                if (const json::Value* special = entry.find("special")) {
                    token.special = special->asBool().value_or(false);
                }

                if (token.id < tokenizer.special_.size()) {
                    tokenizer.special_[token.id] = token.special;
                }
                tokenizer.addedTokens_.push_back(std::move(token));
            }
        }
    }

    // Longest first, so a literal match never stops short of a longer token
    // that also matches at the same position.
    std::ranges::sort(tokenizer.addedTokens_,
                      [](const AddedToken& a, const AddedToken& b) {
                          return a.content.size() > b.content.size();
                      });

    return tokenizer;
}

std::string_view Tokenizer::piece(u32 id) const {
    if (id >= idToPiece_.size()) {
        return {};
    }
    return idToPiece_[id];
}

bool Tokenizer::isSpecial(u32 id) const {
    return id < special_.size() && special_[id];
}

Result<u32> Tokenizer::idFor(std::string_view token) const {
    const auto it = pieceToId_.find(std::string{token});
    if (it == pieceToId_.end()) {
        return makeError(ErrorCode::NotFound, "'{}' is not in the vocabulary", token);
    }
    return it->second;
}

std::vector<u32> Tokenizer::initialSymbols(std::string_view normalized) const {
    std::vector<u32> symbols;
    symbols.reserve(normalized.size());

    usize position = 0;
    while (position < normalized.size()) {
        const usize length =
                std::min(utf8Length(static_cast<u8>(normalized[position])),
                         normalized.size() - position);
        const std::string_view character = normalized.substr(position, length);

        if (const auto it = pieceToId_.find(std::string{character});
            it != pieceToId_.end()) {
            symbols.push_back(it->second);
        } else if (hasByteFallback_) {
            // byte_fallback: a character with no vocabulary entry becomes its
            // individual UTF-8 bytes, which are always present.
            for (const char byte : character) {
                symbols.push_back(byteFallbackIds_[static_cast<u8>(byte)]);
            }
        } else {
            symbols.push_back(unknownId_);
        }
        position += length;
    }
    return symbols;
}

void Tokenizer::encodeSegment(std::string_view text, std::vector<u32>& out) const {
    if (text.empty()) {
        return;
    }

    // Normalizer: replace each space with U+2581. There is no Prepend in this
    // configuration, so no leading marker is synthesized.
    std::string normalized;
    normalized.reserve(text.size());
    for (const char c : text) {
        if (c == ' ') {
            normalized.append(kSpaceMarker);
        } else {
            normalized.push_back(c);
        }
    }

    std::vector<u32> symbols = initialSymbols(normalized);
    if (symbols.empty()) {
        return;
    }

    // Doubly linked list over the symbol array, so a merge is O(1) and indices
    // stay stable as neighbours disappear.
    const auto count = symbols.size();
    std::vector<isize> previous(count);
    std::vector<isize> next(count);
    std::vector<u32> version(count, 0);
    std::vector<bool> alive(count, true);

    for (usize i = 0; i < count; ++i) {
        previous[i] = static_cast<isize>(i) - 1;
        next[i] = static_cast<isize>(i) + 1 < static_cast<isize>(count)
                          ? static_cast<isize>(i) + 1
                          : -1;
    }

    std::priority_queue<Candidate, std::vector<Candidate>, std::greater<>> queue;

    const auto offer = [&](isize left, isize right) {
        if (left < 0 || right < 0) {
            return;
        }
        const auto it = mergeRanks_.find(mergeKey(symbols[static_cast<usize>(left)],
                                                   symbols[static_cast<usize>(right)]));
        if (it == mergeRanks_.end()) {
            return;
        }
        std::string merged;
        merged.append(idToPiece_[symbols[static_cast<usize>(left)]]);
        merged.append(idToPiece_[symbols[static_cast<usize>(right)]]);

        const auto mergedIt = pieceToId_.find(merged);
        if (mergedIt == pieceToId_.end()) {
            return;
        }
        queue.push(Candidate{.rank = it->second,
                             .left = static_cast<u32>(left),
                             .right = static_cast<u32>(right),
                             .mergedId = mergedIt->second,
                             .leftVersion = version[static_cast<usize>(left)],
                             .rightVersion = version[static_cast<usize>(right)]});
    };

    for (usize i = 0; i + 1 < count; ++i) {
        offer(static_cast<isize>(i), static_cast<isize>(i) + 1);
    }

    while (!queue.empty()) {
        const Candidate candidate = queue.top();
        queue.pop();

        // A queued merge is stale once either endpoint has been merged into
        // something else, or its neighbour has changed.
        if (!alive[candidate.left] || !alive[candidate.right]) {
            continue;
        }
        if (version[candidate.left] != candidate.leftVersion ||
            version[candidate.right] != candidate.rightVersion) {
            continue;
        }
        if (next[candidate.left] != static_cast<isize>(candidate.right)) {
            continue;
        }

        symbols[candidate.left] = candidate.mergedId;
        ++version[candidate.left];
        alive[candidate.right] = false;

        const isize after = next[candidate.right];
        next[candidate.left] = after;
        if (after >= 0) {
            previous[static_cast<usize>(after)] = static_cast<isize>(candidate.left);
        }

        offer(previous[candidate.left], static_cast<isize>(candidate.left));
        offer(static_cast<isize>(candidate.left), after);
    }

    for (usize i = 0; i < count; ++i) {
        if (alive[i]) {
            out.push_back(symbols[i]);
        }
    }
}

std::vector<u32> Tokenizer::encode(std::string_view text) const {
    std::vector<u32> ids;

    // Added tokens are matched literally and never merged, so the input splits
    // into ordinary runs around them.
    usize position = 0;
    usize segmentStart = 0;

    while (position < text.size()) {
        const AddedToken* matched = nullptr;
        for (const auto& token : addedTokens_) {
            if (!token.content.empty() &&
                text.compare(position, token.content.size(), token.content) == 0) {
                matched = &token;
                break;  // sorted longest first
            }
        }

        if (matched == nullptr) {
            ++position;
            continue;
        }

        encodeSegment(text.substr(segmentStart, position - segmentStart), ids);
        ids.push_back(matched->id);
        position += matched->content.size();
        segmentStart = position;
    }

    encodeSegment(text.substr(segmentStart), ids);
    return ids;
}

std::string Tokenizer::decode(std::span<const u32> ids, bool skipSpecialTokens) const {
    std::string out;

    // Byte-fallback tokens have to be gathered before conversion: a multi-byte
    // character arrives as several `<0xXX>` tokens that only mean something
    // once concatenated.
    std::string pendingBytes;
    const auto flush = [&] {
        out.append(pendingBytes);
        pendingBytes.clear();
    };

    for (const u32 id : ids) {
        if (id >= idToPiece_.size()) {
            continue;
        }
        if (skipSpecialTokens && isSpecial(id)) {
            flush();
            continue;
        }

        const std::string& stored = idToPiece_[id];
        u8 byteValue = 0;
        if (parseByteToken(stored, byteValue)) {
            pendingBytes.push_back(static_cast<char>(byteValue));
            continue;
        }
        flush();

        // Decoder: replace U+2581 with a space.
        usize i = 0;
        while (i < stored.size()) {
            if (stored.compare(i, kSpaceMarker.size(), kSpaceMarker) == 0) {
                out.push_back(' ');
                i += kSpaceMarker.size();
            } else {
                out.push_back(stored[i]);
                ++i;
            }
        }
    }
    flush();
    return out;
}

std::string Tokenizer::decodeOne(u32 id, bool skipSpecialTokens) const {
    const std::array<u32, 1> single{id};
    return decode(single, skipSpecialTokens);
}

}  // namespace tf
