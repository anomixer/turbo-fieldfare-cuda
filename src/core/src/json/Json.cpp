#include "tf/core/json/Json.h"

#include <charconv>
#include <cmath>
#include <cstdio>

namespace tf::json {
namespace {

std::string_view typeName(Value::Type type) {
    switch (type) {
        case Value::Type::Null:   return "null";
        case Value::Type::Bool:   return "bool";
        case Value::Type::Int:    return "int";
        case Value::Type::Double: return "double";
        case Value::Type::String: return "string";
        case Value::Type::Array:  return "array";
        case Value::Type::Object: return "object";
    }
    return "?";
}

}  // namespace

Value::Type Value::type() const noexcept {
    switch (storage_.index()) {
        case 0: return Type::Null;
        case 1: return Type::Bool;
        case 2: return Type::Int;
        case 3: return Type::Double;
        case 4: return Type::String;
        case 5: return Type::Array;
        default: return Type::Object;
    }
}

bool Value::isNumber() const noexcept {
    const Type t = type();
    return t == Type::Int || t == Type::Double;
}

Result<bool> Value::asBool() const {
    if (const auto* v = std::get_if<bool>(&storage_)) {
        return *v;
    }
    return makeError(ErrorCode::MalformedData, "expected bool, found {}", typeName(type()));
}

Result<i64> Value::asInt() const {
    if (const auto* v = std::get_if<i64>(&storage_)) {
        return *v;
    }
    // Accept a double that is exactly integral: JSON writers routinely emit
    // 30.0 where a count is meant.
    if (const auto* d = std::get_if<double>(&storage_)) {
        if (std::trunc(*d) == *d && std::abs(*d) <= 9007199254740992.0) {
            return static_cast<i64>(*d);
        }
    }
    return makeError(ErrorCode::MalformedData, "expected int, found {}", typeName(type()));
}

Result<u64> Value::asUInt() const {
    TF_TRY(const i64 value, asInt());
    if (value < 0) {
        return makeError(ErrorCode::MalformedData, "expected non-negative int, found {}", value);
    }
    return static_cast<u64>(value);
}

Result<double> Value::asDouble() const {
    if (const auto* d = std::get_if<double>(&storage_)) {
        return *d;
    }
    if (const auto* i = std::get_if<i64>(&storage_)) {
        return static_cast<double>(*i);
    }
    return makeError(ErrorCode::MalformedData, "expected number, found {}", typeName(type()));
}

Result<std::string_view> Value::asString() const {
    if (const auto* s = std::get_if<std::string>(&storage_)) {
        return std::string_view{*s};
    }
    return makeError(ErrorCode::MalformedData, "expected string, found {}", typeName(type()));
}

Result<const Array*> Value::asArray() const {
    if (const auto* a = std::get_if<Array>(&storage_)) {
        return a;
    }
    return makeError(ErrorCode::MalformedData, "expected array, found {}", typeName(type()));
}

Result<const Object*> Value::asObject() const {
    if (const auto* o = std::get_if<Object>(&storage_)) {
        return o;
    }
    return makeError(ErrorCode::MalformedData, "expected object, found {}", typeName(type()));
}

const Value* Value::find(std::string_view key) const {
    const auto* object = std::get_if<Object>(&storage_);
    if (object == nullptr) {
        return nullptr;
    }
    for (const auto& [name, value] : *object) {
        if (name == key) {
            return &value;
        }
    }
    return nullptr;
}

Result<const Value*> Value::at(std::string_view key) const {
    if (!isObject()) {
        return makeError(ErrorCode::MalformedData,
                         "cannot read key '{}' from {}", key, typeName(type()));
    }
    if (const Value* found = find(key)) {
        return found;
    }
    return makeError(ErrorCode::MalformedData, "missing required key '{}'", key);
}

Result<const Value*> Value::path(std::string_view dotted) const {
    const Value* current = this;
    usize start = 0;
    while (start <= dotted.size()) {
        const usize dot = dotted.find('.', start);
        const std::string_view segment =
                dotted.substr(start, dot == std::string_view::npos ? dot : dot - start);

        auto next = current->at(segment);
        if (!next) {
            return std::unexpected(next.error().wrap(std::string{dotted}));
        }
        current = *next;

        if (dot == std::string_view::npos) {
            break;
        }
        start = dot + 1;
    }
    return current;
}

void Value::set(std::string_view key, Value value) {
    if (!isObject()) {
        storage_ = Object{};
    }
    auto& object = std::get<Object>(storage_);
    for (auto& [name, existing] : object) {
        if (name == key) {
            existing = std::move(value);
            return;
        }
    }
    object.emplace_back(std::string{key}, std::move(value));
}

void Value::push(Value value) {
    if (!isArray()) {
        storage_ = Array{};
    }
    std::get<Array>(storage_).push_back(std::move(value));
}

// ---------------------------------------------------------------------------
// Serialization
// ---------------------------------------------------------------------------
namespace {

void escapeInto(std::string& out, std::string_view text) {
    out.push_back('"');
    for (const char c : text) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            case '\b': out += "\\b";  break;
            case '\f': out += "\\f";  break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buffer[7];
                    std::snprintf(buffer, sizeof(buffer), "\\u%04x",
                                  static_cast<unsigned>(static_cast<unsigned char>(c)));
                    out += buffer;
                } else {
                    // UTF-8 continuation bytes pass through unmodified.
                    out.push_back(c);
                }
        }
    }
    out.push_back('"');
}

void writeDouble(std::string& out, double value) {
    // shortest round-trippable form
    char buffer[32];
    const auto [ptr, ec] = std::to_chars(buffer, buffer + sizeof(buffer), value);
    if (ec == std::errc{}) {
        out.append(buffer, ptr);
    } else {
        out += "0";
    }
}

}  // namespace

void Value::dumpTo(std::string& out, int indent, int depth) const {
    const bool pretty = indent > 0;
    const auto newlineIndent = [&](int level) {
        if (pretty) {
            out.push_back('\n');
            out.append(static_cast<usize>(indent * level), ' ');
        }
    };

    switch (type()) {
        case Type::Null:
            out += "null";
            break;
        case Type::Bool:
            out += std::get<bool>(storage_) ? "true" : "false";
            break;
        case Type::Int:
            out += std::to_string(std::get<i64>(storage_));
            break;
        case Type::Double:
            writeDouble(out, std::get<double>(storage_));
            break;
        case Type::String:
            escapeInto(out, std::get<std::string>(storage_));
            break;
        case Type::Array: {
            const auto& array = std::get<Array>(storage_);
            if (array.empty()) {
                out += "[]";
                break;
            }
            out.push_back('[');
            for (usize i = 0; i < array.size(); ++i) {
                if (i > 0) {
                    out.push_back(',');
                }
                newlineIndent(depth + 1);
                array[i].dumpTo(out, indent, depth + 1);
            }
            newlineIndent(depth);
            out.push_back(']');
            break;
        }
        case Type::Object: {
            const auto& object = std::get<Object>(storage_);
            if (object.empty()) {
                out += "{}";
                break;
            }
            out.push_back('{');
            for (usize i = 0; i < object.size(); ++i) {
                if (i > 0) {
                    out.push_back(',');
                }
                newlineIndent(depth + 1);
                escapeInto(out, object[i].first);
                out.push_back(':');
                if (pretty) {
                    out.push_back(' ');
                }
                object[i].second.dumpTo(out, indent, depth + 1);
            }
            newlineIndent(depth);
            out.push_back('}');
            break;
        }
    }
}

std::string Value::dump(int indent) const {
    std::string out;
    dumpTo(out, indent, 0);
    return out;
}

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------
namespace {

class Parser {
public:
    explicit Parser(std::string_view text) : text_(text) {}

    Result<Value> run() {
        skipWhitespace();
        TF_TRY(Value root, parseValue(0));
        skipWhitespace();
        if (pos_ != text_.size()) {
            return fail("trailing content after top-level value");
        }
        return root;
    }

private:
    static constexpr int kMaxDepth = 128;

    std::string_view text_;
    usize pos_ = 0;

    [[nodiscard]] std::unexpected<Error> fail(std::string_view what) const {
        return makeError(ErrorCode::MalformedData, "json at byte {}: {}", pos_, what);
    }

    [[nodiscard]] bool atEnd() const { return pos_ >= text_.size(); }
    [[nodiscard]] char peek() const { return text_[pos_]; }

    void skipWhitespace() {
        while (!atEnd()) {
            const char c = text_[pos_];
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                ++pos_;
            } else {
                break;
            }
        }
    }

    bool consume(char expected) {
        if (!atEnd() && text_[pos_] == expected) {
            ++pos_;
            return true;
        }
        return false;
    }

    bool consumeLiteral(std::string_view literal) {
        if (text_.compare(pos_, literal.size(), literal) == 0) {
            pos_ += literal.size();
            return true;
        }
        return false;
    }

    Result<Value> parseValue(int depth) {
        if (depth > kMaxDepth) {
            return fail("nesting too deep");
        }
        if (atEnd()) {
            return fail("unexpected end of input");
        }

        switch (peek()) {
            case '{': return parseObject(depth);
            case '[': return parseArray(depth);
            case '"': {
                TF_TRY(std::string s, parseString());
                return Value{std::move(s)};
            }
            case 't':
                if (consumeLiteral("true")) { return Value{true}; }
                return fail("invalid literal");
            case 'f':
                if (consumeLiteral("false")) { return Value{false}; }
                return fail("invalid literal");
            case 'n':
                if (consumeLiteral("null")) { return Value{nullptr}; }
                return fail("invalid literal");
            default:
                return parseNumber();
        }
    }

    Result<Value> parseObject(int depth) {
        ++pos_;  // '{'
        Object object;
        skipWhitespace();
        if (consume('}')) {
            return Value{std::move(object)};
        }

        while (true) {
            skipWhitespace();
            if (atEnd() || peek() != '"') {
                return fail("expected object key");
            }
            TF_TRY(std::string key, parseString());

            skipWhitespace();
            if (!consume(':')) {
                return fail("expected ':' after object key");
            }

            skipWhitespace();
            TF_TRY(Value value, parseValue(depth + 1));
            object.emplace_back(std::move(key), std::move(value));

            skipWhitespace();
            if (consume(',')) {
                continue;
            }
            if (consume('}')) {
                return Value{std::move(object)};
            }
            return fail("expected ',' or '}' in object");
        }
    }

    Result<Value> parseArray(int depth) {
        ++pos_;  // '['
        Array array;
        skipWhitespace();
        if (consume(']')) {
            return Value{std::move(array)};
        }

        while (true) {
            skipWhitespace();
            TF_TRY(Value value, parseValue(depth + 1));
            array.push_back(std::move(value));

            skipWhitespace();
            if (consume(',')) {
                continue;
            }
            if (consume(']')) {
                return Value{std::move(array)};
            }
            return fail("expected ',' or ']' in array");
        }
    }

    Result<std::string> parseString() {
        ++pos_;  // opening quote
        std::string out;
        while (true) {
            if (atEnd()) {
                return fail("unterminated string");
            }
            const char c = text_[pos_++];
            if (c == '"') {
                return out;
            }
            if (c != '\\') {
                out.push_back(c);
                continue;
            }
            if (atEnd()) {
                return fail("unterminated escape");
            }
            const char esc = text_[pos_++];
            switch (esc) {
                case '"':  out.push_back('"');  break;
                case '\\': out.push_back('\\'); break;
                case '/':  out.push_back('/');  break;
                case 'b':  out.push_back('\b'); break;
                case 'f':  out.push_back('\f'); break;
                case 'n':  out.push_back('\n'); break;
                case 'r':  out.push_back('\r'); break;
                case 't':  out.push_back('\t'); break;
                case 'u': {
                    TF_TRY(const u32 code, parseUnicodeEscape());
                    appendUtf8(out, code);
                    break;
                }
                default:
                    return fail("invalid escape character");
            }
        }
    }

    Result<u32> parseHex4() {
        if (pos_ + 4 > text_.size()) {
            return fail("truncated \\u escape");
        }
        u32 value = 0;
        for (int i = 0; i < 4; ++i) {
            const char c = text_[pos_++];
            value <<= 4;
            if (c >= '0' && c <= '9') {
                value |= static_cast<u32>(c - '0');
            } else if (c >= 'a' && c <= 'f') {
                value |= static_cast<u32>(c - 'a' + 10);
            } else if (c >= 'A' && c <= 'F') {
                value |= static_cast<u32>(c - 'A' + 10);
            } else {
                return fail("invalid hex digit in \\u escape");
            }
        }
        return value;
    }

    Result<u32> parseUnicodeEscape() {
        TF_TRY(u32 code, parseHex4());

        // Surrogate pair: a high surrogate must be followed by \uDC00-\uDFFF.
        if (code >= 0xD800 && code <= 0xDBFF) {
            if (pos_ + 1 < text_.size() && text_[pos_] == '\\' && text_[pos_ + 1] == 'u') {
                pos_ += 2;
                TF_TRY(const u32 low, parseHex4());
                if (low < 0xDC00 || low > 0xDFFF) {
                    return fail("invalid low surrogate");
                }
                code = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00);
            } else {
                return fail("unpaired high surrogate");
            }
        } else if (code >= 0xDC00 && code <= 0xDFFF) {
            return fail("unpaired low surrogate");
        }
        return code;
    }

    static void appendUtf8(std::string& out, u32 code) {
        if (code < 0x80) {
            out.push_back(static_cast<char>(code));
        } else if (code < 0x800) {
            out.push_back(static_cast<char>(0xC0 | (code >> 6)));
            out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
        } else if (code < 0x10000) {
            out.push_back(static_cast<char>(0xE0 | (code >> 12)));
            out.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
        } else {
            out.push_back(static_cast<char>(0xF0 | (code >> 18)));
            out.push_back(static_cast<char>(0x80 | ((code >> 12) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
        }
    }

    Result<Value> parseNumber() {
        const usize start = pos_;
        if (!atEnd() && (peek() == '-' || peek() == '+')) {
            ++pos_;
        }

        bool isDouble = false;
        while (!atEnd()) {
            const char c = peek();
            if (c >= '0' && c <= '9') {
                ++pos_;
            } else if (c == '.' || c == 'e' || c == 'E') {
                isDouble = true;
                ++pos_;
            } else if ((c == '-' || c == '+') && (text_[pos_ - 1] == 'e' || text_[pos_ - 1] == 'E')) {
                ++pos_;
            } else {
                break;
            }
        }

        const std::string_view token = text_.substr(start, pos_ - start);
        if (token.empty()) {
            return fail("expected a value");
        }

        if (!isDouble) {
            i64 value = 0;
            const auto [ptr, ec] =
                    std::from_chars(token.data(), token.data() + token.size(), value);
            if (ec == std::errc{} && ptr == token.data() + token.size()) {
                return Value{value};
            }
            // Fall through: an integer too large for i64 is kept as a double
            // rather than rejected, since it cannot be an offset we care about.
        }

        double value = 0.0;
        const auto [ptr, ec] = std::from_chars(token.data(), token.data() + token.size(), value);
        if (ec != std::errc{} || ptr != token.data() + token.size()) {
            pos_ = start;
            return fail("invalid number");
        }
        return Value{value};
    }
};

}  // namespace

Result<Value> parse(std::string_view text) {
    // Tolerate a UTF-8 BOM; some Windows tooling writes config files with one.
    if (text.size() >= 3 && static_cast<unsigned char>(text[0]) == 0xEF &&
        static_cast<unsigned char>(text[1]) == 0xBB &&
        static_cast<unsigned char>(text[2]) == 0xBF) {
        text.remove_prefix(3);
    }
    Parser parser{text};
    return parser.run();
}

}  // namespace tf::json
