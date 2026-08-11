#include <catch2/catch_test_macros.hpp>

#include <windows.h>

#include "tf/core/base/Error.h"
#include "tf/core/base/Types.h"

using namespace tf;

TEST_CASE("alignUp rounds to power-of-two boundaries", "[base]") {
    CHECK(alignUp(0, align::kSector) == 0);
    CHECK(alignUp(1, align::kSector) == 4096);
    CHECK(alignUp(4096, align::kSector) == 4096);
    CHECK(alignUp(4097, align::kSector) == 8192);

    // A 3.36 MB expert blob padded to the preferred 2 MiB stride.
    constexpr u64 expertBytes = 3'523'215;
    CHECK(alignUp(expertBytes, align::kLargePage) == 4ull * 1024 * 1024);
    CHECK(isAligned(alignUp(expertBytes, align::kSector), align::kSector));
}

TEST_CASE("ByteRange adjacency drives read coalescing", "[base]") {
    const ByteRange a{.offset = 0, .length = 4096};
    const ByteRange b{.offset = 4096, .length = 4096};
    const ByteRange gap{.offset = 8193, .length = 4096};

    CHECK(a.end() == 4096);
    CHECK(a.adjacentTo(b));
    CHECK_FALSE(b.adjacentTo(gap));
    CHECK_FALSE(a.adjacentTo(gap));
    CHECK(ByteRange{}.empty());
}

namespace {

Result<int> succeeds() { return 7; }

Result<int> fails() {
    return makeError(ErrorCode::MalformedData, "expert {} out of range", 42);
}

// Two TF_TRY uses in one function prove the __LINE__ pasting actually expands.
Result<int> chained() {
    TF_TRY(auto first, succeeds());
    TF_TRY(const auto second, succeeds());
    return first + second;
}

Result<int> propagates() {
    TF_TRY(auto value, fails());
    return value;
}

}  // namespace

TEST_CASE("TF_TRY binds values and propagates errors", "[base]") {
    const auto ok = chained();
    REQUIRE(ok.has_value());
    CHECK(*ok == 14);

    const auto bad = propagates();
    REQUIRE_FALSE(bad.has_value());
    CHECK(bad.error().code() == ErrorCode::MalformedData);
    CHECK(bad.error().message() == "expert 42 out of range");
}

TEST_CASE("Error::wrap adds context but keeps the code", "[base]") {
    const Error inner{ErrorCode::NotFound, "no such file"};
    const Error wrapped = inner.wrap("loading layer_07.bin");

    CHECK(wrapped.code() == ErrorCode::NotFound);
    CHECK(wrapped.message() == "loading layer_07.bin: no such file");
    CHECK(wrapped.toString() == "[not-found] loading layer_07.bin: no such file");
}

TEST_CASE("Win32 errors are classified and formatted", "[base]") {
    const Error notFound = win32Error(ERROR_FILE_NOT_FOUND, "opening manifest.json");
    CHECK(notFound.code() == ErrorCode::NotFound);
    CHECK(notFound.message().starts_with("opening manifest.json: "));
    // The system message text is locale-dependent, so assert only that
    // FormatMessage produced something beyond our own prefix.
    CHECK(notFound.message().size() > std::string("opening manifest.json: ").size());

    CHECK(win32Error(ERROR_DISK_FULL, "writing").code() == ErrorCode::OutOfDiskSpace);
    CHECK(win32Error(ERROR_OPERATION_ABORTED, "reading").code() == ErrorCode::Cancelled);
    CHECK(win32Error(ERROR_ACCESS_DENIED, "opening").code() == ErrorCode::Io);
}
