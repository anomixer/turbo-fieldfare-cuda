#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <windows.h>

#include <filesystem>
#include <format>
#include <numeric>
#include <string>
#include <system_error>
#include <vector>

#include "tf/core/io/File.h"

using namespace tf;

namespace {

/// Per-test scratch directory, removed on scope exit.
class TempDir {
public:
    TempDir() {
        static int counter = 0;
        path_ = std::filesystem::temp_directory_path() /
                std::format("tf-test-{}-{}", ::GetCurrentProcessId(), ++counter);
        std::filesystem::create_directories(path_);
    }

    ~TempDir() {
        std::error_code ec;
        std::filesystem::remove_all(path_, ec);
    }

    TempDir(const TempDir&) = delete;
    TempDir& operator=(const TempDir&) = delete;

    [[nodiscard]] std::filesystem::path file(std::string_view name) const {
        return path_ / name;
    }

    [[nodiscard]] const std::filesystem::path& path() const { return path_; }

private:
    std::filesystem::path path_;
};

std::vector<u8> makePattern(usize size) {
    std::vector<u8> data(size);
    std::iota(data.begin(), data.end(), u8{0});
    return data;
}

}  // namespace

TEST_CASE("write then read back a file", "[io]") {
    const TempDir dir;
    const auto path = dir.file("roundtrip.bin");
    const std::vector<u8> written = makePattern(5000);

    {
        auto file = io::File::create(path);
        REQUIRE(file.has_value());
        REQUIRE(file->write(written).has_value());
    }

    auto file = io::File::openRead(path);
    REQUIRE(file.has_value());
    CHECK(*file->size() == written.size());

    std::vector<u8> read(written.size());
    REQUIRE(file->readExactAt(0, read).has_value());
    CHECK(read == written);
}

TEST_CASE("readAt is positional and does not disturb a shared handle", "[io]") {
    const TempDir dir;
    const auto path = dir.file("positional.bin");
    const std::vector<u8> written = makePattern(1024);

    {
        auto file = io::File::create(path);
        REQUIRE(file->write(written).has_value());
    }

    auto file = io::File::openRead(path, io::ReadMode::RandomBuffered);
    REQUIRE(file.has_value());

    // Read out of order; each call carries its own offset, so results must not
    // depend on the sequence. This is what makes concurrent expert fetches on
    // one handle safe.
    std::vector<u8> tail(16);
    std::vector<u8> head(16);
    REQUIRE(file->readExactAt(1000, tail).has_value());
    REQUIRE(file->readExactAt(0, head).has_value());

    CHECK(head[0] == written[0]);
    CHECK(head[15] == written[15]);
    CHECK(tail[0] == written[1000]);
    CHECK(tail[15] == written[1015]);
}

TEST_CASE("reading past the end is short, and readExactAt calls it an error", "[io]") {
    const TempDir dir;
    const auto path = dir.file("short.bin");
    {
        auto file = io::File::create(path);
        REQUIRE(file->write(makePattern(100)).has_value());
    }

    auto file = io::File::openRead(path);
    std::vector<u8> buffer(50);

    // Straddling the end returns what exists.
    const auto partial = file->readAt(80, buffer);
    REQUIRE(partial.has_value());
    CHECK(*partial == 20);

    // Starting past the end returns zero rather than failing.
    const auto beyond = file->readAt(500, buffer);
    REQUIRE(beyond.has_value());
    CHECK(*beyond == 0);

    // readExactAt treats truncation as corruption, which is what the repacker
    // and expert streamer need.
    const auto exact = file->readExactAt(80, buffer);
    REQUIRE_FALSE(exact.has_value());
    CHECK(exact.error().code() == ErrorCode::MalformedData);
    CHECK_THAT(exact.error().message(), Catch::Matchers::ContainsSubstring("short read"));
}

TEST_CASE("writeAt places bytes at an explicit offset", "[io]") {
    const TempDir dir;
    const auto path = dir.file("sparse.bin");

    {
        auto file = io::File::create(path);
        REQUIRE(file->preallocate(4096).has_value());
        const std::vector<u8> marker{0xDE, 0xAD, 0xBE, 0xEF};
        REQUIRE(file->writeAt(2048, marker).has_value());
    }

    auto file = io::File::openRead(path);
    CHECK(*file->size() == 4096);

    std::vector<u8> read(4);
    REQUIRE(file->readExactAt(2048, read).has_value());
    CHECK(read == std::vector<u8>{0xDE, 0xAD, 0xBE, 0xEF});

    // Preallocated space reads back as zeros, never as recycled disk contents.
    std::vector<u8> gap(16);
    REQUIRE(file->readExactAt(1024, gap).has_value());
    CHECK(gap == std::vector<u8>(16, 0));
}

TEST_CASE("opening a missing file reports NotFound", "[io]") {
    const TempDir dir;
    const auto result = io::File::openRead(dir.file("nope.bin"));
    REQUIRE_FALSE(result.has_value());
    CHECK(result.error().code() == ErrorCode::NotFound);
    CHECK_THAT(result.error().message(), Catch::Matchers::ContainsSubstring("nope.bin"));
}

TEST_CASE("memory mapping exposes file bytes", "[io]") {
    const TempDir dir;
    const auto path = dir.file("mapped.bin");
    const std::vector<u8> written = makePattern(8192);
    {
        auto file = io::File::create(path);
        REQUIRE(file->write(written).has_value());
    }

    auto mapped = io::MappedFile::open(path);
    REQUIRE(mapped.has_value());
    CHECK(mapped->size() == 8192);
    CHECK(mapped->isOpen());
    CHECK(mapped->data()[0] == written[0]);
    CHECK(mapped->data()[8191] == written[8191]);

    const auto slice = mapped->range(ByteRange{.offset = 4096, .length = 256});
    REQUIRE(slice.has_value());
    CHECK(slice->size() == 256);
    CHECK((*slice)[0] == written[4096]);

    // Out-of-bounds ranges are refused rather than reading past the mapping.
    CHECK_FALSE(mapped->range(ByteRange{.offset = 8000, .length = 1000}).has_value());
    CHECK_FALSE(mapped->range(ByteRange{.offset = 99999, .length = 1}).has_value());
}

TEST_CASE("mapping an empty file is an error, not a crash", "[io]") {
    const TempDir dir;
    const auto path = dir.file("empty.bin");
    { auto file = io::File::create(path); }

    const auto mapped = io::MappedFile::open(path);
    REQUIRE_FALSE(mapped.has_value());
    CHECK(mapped.error().code() == ErrorCode::MalformedData);
}

TEST_CASE("atomic write replaces the target and leaves no temp behind", "[io]") {
    const TempDir dir;
    const auto path = dir.file("manifest.json");

    const std::string first = R"({"version":1})";
    REQUIRE(io::writeFileAtomic(path, ByteSpan{reinterpret_cast<const u8*>(first.data()),
                                               first.size()})
                    .has_value());
    CHECK(*io::readTextFile(path) == first);

    // Overwriting an existing file must succeed, since reinstalls rewrite the
    // manifest in place.
    const std::string second = R"({"version":2,"arch":"gemma4"})";
    REQUIRE(io::writeFileAtomic(path, ByteSpan{reinterpret_cast<const u8*>(second.data()),
                                               second.size()})
                    .has_value());
    CHECK(*io::readTextFile(path) == second);

    CHECK_FALSE(std::filesystem::exists(dir.file("manifest.json.tmp")));
}

TEST_CASE("whole-file reads refuse oversized files", "[io]") {
    const TempDir dir;
    const auto path = dir.file("big.bin");
    {
        auto file = io::File::create(path);
        REQUIRE(file->write(makePattern(4096)).has_value());
    }

    const auto refused = io::readBinaryFile(path, /*maxBytes=*/1024);
    REQUIRE_FALSE(refused.has_value());
    CHECK(refused.error().code() == ErrorCode::InvalidArgument);

    CHECK(io::readBinaryFile(path, 8192)->size() == 4096);
}

TEST_CASE("volume queries answer for paths that do not exist yet", "[io]") {
    const TempDir dir;

    // The installer checks free space before creating the target directory.
    const auto space = io::availableSpace(dir.path() / "not" / "created" / "yet");
    REQUIRE(space.has_value());
    CHECK(*space > 0);

    const auto sector = io::File::sectorSize(dir.path());
    REQUIRE(sector.has_value());
    // Unbuffered I/O alignment depends on this being a sane power of two.
    CHECK(*sector >= 512);
    CHECK(isAligned(align::kSector, *sector));
}
