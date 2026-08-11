// Resume state and the install lock.
//
// These are the parts of the streaming installer that decide whether a second
// run continues safely or corrupts what the first one wrote. Both are testable
// without touching the network, which matters: the failure they guard against
// only appears after a 14 GB download is interrupted.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <filesystem>
#include <fstream>
#include <string>

#include "tf/repack/Installer.h"

using namespace tf;
using namespace tf::repack;

namespace {

/// A scratch directory that cleans up after itself.
class TempDir {
public:
    TempDir() {
        path_ = std::filesystem::temp_directory_path() /
                std::format("tf-installer-{}", ++counter());
        std::filesystem::create_directories(path_);
    }
    ~TempDir() {
        std::error_code ec;
        std::filesystem::remove_all(path_, ec);
    }

    TempDir(const TempDir&) = delete;
    TempDir& operator=(const TempDir&) = delete;

    [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }

private:
    static int& counter() {
        static int value = 0;
        return value;
    }
    std::filesystem::path path_;
};

}  // namespace

// ---------------------------------------------------------------------------
// Resume state
// ---------------------------------------------------------------------------

TEST_CASE("install state round-trips through JSON", "[installer]") {
    const InstallState written{.repoId = "mlx-community/gemma-4-26b-a4b-it-4bit",
                               .revision = "0d77464eeb233a2da68ebf9d7dc4edaac7db956d",
                               .planFingerprint = "shards=[a,b] ops=1234",
                               .opsCompleted = 900,
                               .bytesWritten = 5ull * 1024 * 1024 * 1024};

    auto read = InstallState::parse(written.toJson());
    REQUIRE(read.has_value());
    CHECK(read->repoId == written.repoId);
    CHECK(read->revision == written.revision);
    CHECK(read->planFingerprint == written.planFingerprint);
    CHECK(read->opsCompleted == written.opsCompleted);
    // A byte count past 4 GiB has to survive, which it will not if anything on
    // the path narrows to 32 bits.
    CHECK(read->bytesWritten == written.bytesWritten);
}

TEST_CASE("a truncated or malformed state file is rejected, not guessed at",
          "[installer]") {
    // A resume that reads a half-written state file and proceeds anyway would
    // write the rest of the model at the wrong offsets.
    CHECK_FALSE(InstallState::parse("").has_value());
    CHECK_FALSE(InstallState::parse("{").has_value());
    CHECK_FALSE(InstallState::parse("{\"repoId\": \"a\"}").has_value());
    CHECK_FALSE(InstallState::parse("[]").has_value());
}

TEST_CASE("the fingerprint changes with anything that moves a byte",
          "[installer]") {
    RepackPlan plan;
    plan.shards = {"model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"};
    plan.experts.numLayers = 30;
    plan.experts.stride = 3345408;
    plan.ops.resize(1000);

    const RemoteSource source;
    const std::string base = planFingerprint(plan, source);
    CHECK_FALSE(base.empty());
    // Same inputs, same answer - otherwise no resume would ever be accepted.
    CHECK(planFingerprint(plan, source) == base);

    SECTION("a different revision") {
        RemoteSource other = source;
        other.revision = "1111111111111111111111111111111111111111";
        CHECK(planFingerprint(plan, other) != base);
    }
    SECTION("a different repository") {
        RemoteSource other = source;
        other.repoId = "someone/else";
        CHECK(planFingerprint(plan, other) != base);
    }
    SECTION("a different shard set") {
        RepackPlan other = plan;
        other.shards.push_back("model-00003-of-00003.safetensors");
        CHECK(planFingerprint(other, source) != base);
    }
    SECTION("a different operation split") {
        // --max-op-bytes changes how many operations the same bytes become, and
        // resuming across that would skip the wrong amount of work.
        RepackPlan other = plan;
        other.ops.resize(2000);
        CHECK(planFingerprint(other, source) != base);
    }
    SECTION("a different expert stride") {
        RepackPlan other = plan;
        other.experts.stride = 4194304;
        CHECK(planFingerprint(other, source) != base);
    }
}

// ---------------------------------------------------------------------------
// Lock
// ---------------------------------------------------------------------------

TEST_CASE("the install lock excludes a second installer", "[installer]") {
    TempDir temp;

    auto first = InstallLock::acquire(temp.path());
    REQUIRE(first.has_value());

    // Two installers writing one partial directory would interleave their
    // operations and produce a directory that passes its own hash check while
    // being internally inconsistent.
    auto second = InstallLock::acquire(temp.path());
    REQUIRE_FALSE(second.has_value());
    CHECK_THAT(second.error().message(),
               Catch::Matchers::ContainsSubstring("another installer"));
}

TEST_CASE("releasing the lock lets the next installer in", "[installer]") {
    TempDir temp;

    {
        auto held = InstallLock::acquire(temp.path());
        REQUIRE(held.has_value());
    }

    // The lock file is opened FILE_FLAG_DELETE_ON_CLOSE, so a crashed installer
    // leaves nothing behind that a later run cannot clear.
    auto again = InstallLock::acquire(temp.path());
    CHECK(again.has_value());
}

TEST_CASE("a moved lock still holds, and is released once", "[installer]") {
    TempDir temp;

    auto original = InstallLock::acquire(temp.path());
    REQUIRE(original.has_value());

    InstallLock moved = std::move(*original);
    // Ownership transferred, so the directory is still locked.
    CHECK_FALSE(InstallLock::acquire(temp.path()).has_value());

    moved.release();
    CHECK(InstallLock::acquire(temp.path()).has_value());
}

TEST_CASE("the lock creates the directory it guards", "[installer]") {
    TempDir temp;
    const std::filesystem::path nested = temp.path() / "output.gturbo.partial";

    // The installer acquires the lock before it has written anything, so the
    // partial directory may not exist yet.
    auto lock = InstallLock::acquire(nested);
    REQUIRE(lock.has_value());
    CHECK(std::filesystem::exists(nested));
}
