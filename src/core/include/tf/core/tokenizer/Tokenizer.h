#pragma once

#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "tf/core/base/Error.h"
#include "tf/core/base/Types.h"

namespace tf {

/// A token that is matched literally in the input and never passed through BPE.
struct AddedToken {
    u32 id = 0;
    std::string content;
    bool special = false;
};

/// Byte-pair encoder for Gemma's `tokenizer.json`.
///
/// The configuration this implements, read from the installed file rather than
/// assumed:
///
///   normalizer     Replace " " with U+2581. Note there is no Prepend, so a
///                  leading space is not synthesized - adding one changes the
///                  tokenization of the first word.
///   pre_tokenizer  Split on " ", which is a no-op because normalization has
///                  already removed every space.
///   model          BPE, byte_fallback, fuse_unk, ignore_merges false,
///                  no continuing_subword_prefix or end_of_word_suffix.
///   decoder        Replace U+2581 with " ", then ByteFallback, then Fuse.
///
/// Merges are ranked by position in the file; the lowest rank wins.
class Tokenizer {
public:
    [[nodiscard]] static Result<Tokenizer> loadFromFile(const std::filesystem::path& path);
    [[nodiscard]] static Result<Tokenizer> parse(std::string_view json);

    /// Encodes text to ids. Added tokens present in the text are matched
    /// literally; everything else is normalized and merged.
    [[nodiscard]] std::vector<u32> encode(std::string_view text) const;

    /// Decodes ids back to text, undoing the U+2581 substitution and
    /// reassembling byte-fallback runs into their original bytes.
    [[nodiscard]] std::string decode(std::span<const u32> ids,
                                     bool skipSpecialTokens = true) const;

    /// The stored piece for an id, or empty when out of range. This is the raw
    /// vocabulary entry, still carrying U+2581 for spaces.
    [[nodiscard]] std::string_view piece(u32 id) const;

    /// Renders a single id as display text, which is what a streaming decoder
    /// needs per token.
    [[nodiscard]] std::string decodeOne(u32 id, bool skipSpecialTokens = true) const;

    [[nodiscard]] Result<u32> idFor(std::string_view token) const;

    [[nodiscard]] u64 vocabularySize() const noexcept { return idToPiece_.size(); }
    [[nodiscard]] u64 mergeCount() const noexcept { return mergeRanks_.size(); }
    [[nodiscard]] const std::vector<AddedToken>& addedTokens() const noexcept {
        return addedTokens_;
    }

    [[nodiscard]] bool isSpecial(u32 id) const;

private:
    /// Encodes one run of ordinary text: normalize, split to symbols, merge.
    void encodeSegment(std::string_view text, std::vector<u32>& out) const;

    /// Initial symbols for a normalized string. A character absent from the
    /// vocabulary decomposes into its UTF-8 bytes as `<0xXX>` tokens, which is
    /// what byte_fallback means.
    [[nodiscard]] std::vector<u32> initialSymbols(std::string_view normalized) const;

    std::unordered_map<std::string, u32> pieceToId_;
    std::vector<std::string> idToPiece_;
    /// Key is (left id << 32) | right id.
    std::unordered_map<u64, u32> mergeRanks_;
    std::vector<AddedToken> addedTokens_;
    std::vector<bool> special_;
    /// Ids of the 256 `<0xXX>` byte tokens, indexed by byte value.
    std::vector<u32> byteFallbackIds_;
    bool hasByteFallback_ = false;
    u32 unknownId_ = 0;
};

}  // namespace tf
