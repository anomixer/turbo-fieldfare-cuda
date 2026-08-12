#pragma once

#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "tf/core/base/Error.h"
#include "tf/core/base/Types.h"

namespace tf::json {

class Value;

/// Objects keep insertion order rather than sorting keys. Two reasons: manifest
/// output must be byte-reproducible so its SHA-256 is stable across runs, and
/// round-tripping a source config should not silently reorder it.
using Object = std::vector<std::pair<std::string, Value>>;
using Array = std::vector<Value>;

class Value {
public:
    enum class Type { Null, Bool, Int, Double, String, Array, Object };

    Value() : storage_(std::monostate{}) {}
    Value(std::nullptr_t) : storage_(std::monostate{}) {}
    Value(bool v) : storage_(v) {}
    // All integral widths funnel to i64 so callers need not cast; without the
    // narrower overloads a u32 argument is ambiguous between int and u64.
    Value(i64 v) : storage_(v) {}
    Value(int v) : storage_(static_cast<i64>(v)) {}
    Value(u32 v) : storage_(static_cast<i64>(v)) {}
    Value(u64 v) : storage_(static_cast<i64>(v)) {}
    Value(double v) : storage_(v) {}
    Value(std::string v) : storage_(std::move(v)) {}
    Value(std::string_view v) : storage_(std::string{v}) {}
    Value(const char* v) : storage_(std::string{v}) {}
    Value(Array v) : storage_(std::move(v)) {}
    Value(Object v) : storage_(std::move(v)) {}

    [[nodiscard]] Type type() const noexcept;
    [[nodiscard]] bool isNull() const noexcept { return type() == Type::Null; }
    [[nodiscard]] bool isObject() const noexcept { return type() == Type::Object; }
    [[nodiscard]] bool isArray() const noexcept { return type() == Type::Array; }
    /// True for Int as well as Double - JSON does not distinguish, so 1 and 1.0
    /// must both satisfy a "number" query.
    [[nodiscard]] bool isNumber() const noexcept;

    // ---- Typed accessors -------------------------------------------------
    // Each returns an error naming the requested type and what was found, so a
    // malformed config.json reports the offending field rather than a bad_variant.

    [[nodiscard]] Result<bool> asBool() const;
    [[nodiscard]] Result<i64> asInt() const;
    [[nodiscard]] Result<u64> asUInt() const;
    [[nodiscard]] Result<double> asDouble() const;
    [[nodiscard]] Result<std::string_view> asString() const;
    [[nodiscard]] Result<const Array*> asArray() const;
    [[nodiscard]] Result<const Object*> asObject() const;

    // ---- Object navigation -----------------------------------------------

    /// Returns nullptr when absent or when this is not an object.
    [[nodiscard]] const Value* find(std::string_view key) const;

    /// Like find() but reports a missing key as an error, for required fields.
    [[nodiscard]] Result<const Value*> at(std::string_view key) const;

    /// Resolves a dotted path such as "text_config.num_experts". Intended for
    /// reading nested model configs without a chain of at() calls.
    [[nodiscard]] Result<const Value*> path(std::string_view dotted) const;

    void set(std::string_view key, Value value);
    void push(Value value);

    [[nodiscard]] static Value makeObject() { return Value{Object{}}; }
    [[nodiscard]] static Value makeArray() { return Value{Array{}}; }

    /// Serializes to UTF-8. `indent` of 0 emits compact output; anything larger
    /// pretty-prints with that many spaces per level.
    [[nodiscard]] std::string dump(int indent = 0) const;

private:
    std::variant<std::monostate, bool, i64, double, std::string, Array, Object> storage_;

    void dumpTo(std::string& out, int indent, int depth) const;
};

/// Parses UTF-8 JSON. Strict: no comments, no trailing commas, no NaN/Infinity.
/// Errors carry the byte offset of the offending character.
[[nodiscard]] Result<Value> parse(std::string_view text);

}  // namespace tf::json
