// URL handling, retry policy and the Hugging Face endpoints.
//
// The pure parts run everywhere. The parts that touch the network are tagged
// [.][network] and hidden by default: a test suite that fails because a CDN is
// slow teaches people to ignore failures. Run them deliberately with
// `tf_tests "[network]"`.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <string>
#include <vector>

#include "tf/core/net/Http.h"
#include "tf/repack/Remote.h"

using namespace tf;
using namespace tf::repack;

// ---------------------------------------------------------------------------
// URLs
// ---------------------------------------------------------------------------

TEST_CASE("URLs split into the pieces WinHTTP needs", "[remote]") {
    auto parsed = net::parseUrl("https://huggingface.co/api/models/a/b/tree/main?recursive=1");
    REQUIRE(parsed.has_value());
    CHECK(parsed->host == "huggingface.co");
    CHECK(parsed->port == 443);
    CHECK(parsed->secure);
    // The query is part of the path as far as WinHttpOpenRequest is concerned.
    CHECK(parsed->path == "/api/models/a/b/tree/main?recursive=1");
}

TEST_CASE("a URL with an explicit port and no path still parses", "[remote]") {
    auto parsed = net::parseUrl("http://127.0.0.1:8080");
    REQUIRE(parsed.has_value());
    CHECK(parsed->host == "127.0.0.1");
    CHECK(parsed->port == 8080);
    CHECK_FALSE(parsed->secure);
    // An empty path would make WinHttpOpenRequest fetch nothing.
    CHECK(parsed->path == "/");
}

TEST_CASE("nonsense is refused rather than half-parsed", "[remote]") {
    CHECK_FALSE(net::parseUrl("not a url").has_value());
    CHECK_FALSE(net::parseUrl("").has_value());
}

TEST_CASE("file URLs point at the resolve endpoint for a pinned revision",
          "[remote]") {
    const RemoteSource source{.repoId = "mlx-community/gemma-4-26b-a4b-it-4bit",
                              .revision = "0d77464eeb233a2da68ebf9d7dc4edaac7db956d"};

    const std::string url = source.fileUrl("model-00001-of-00004.safetensors");
    CHECK(url ==
          "https://huggingface.co/mlx-community/gemma-4-26b-a4b-it-4bit/resolve/"
          "0d77464eeb233a2da68ebf9d7dc4edaac7db956d/model-00001-of-00004.safetensors");

    // And it must be a URL the client can actually use.
    CHECK(net::parseUrl(url).has_value());
}

TEST_CASE("characters that would change the path are encoded", "[remote]") {
    RemoteSource source;
    source.repoId = "owner/name with space";

    const std::string url = source.fileUrl("dir/file?.bin");
    // The slash is structural and stays; the space and the question mark would
    // otherwise end the path or start a query.
    CHECK(url.find("name%20with%20space") != std::string::npos);
    CHECK(url.find("file%3F.bin") != std::string::npos);
    CHECK(url.find("dir/file") != std::string::npos);
}

TEST_CASE("an endpoint override is honoured, for a mirror", "[remote]") {
    RemoteSource source;
    source.endpoint = "https://hf-mirror.example";
    CHECK(source.fileUrl("config.json").starts_with("https://hf-mirror.example/"));
}

TEST_CASE("a source is validated before any request is made", "[remote]") {
    CHECK(RemoteSource{}.validate().has_value());

    RemoteSource noOwner;
    noOwner.repoId = "justaname";
    CHECK_FALSE(noOwner.validate().has_value());

    RemoteSource noRevision;
    noRevision.revision.clear();
    CHECK_FALSE(noRevision.validate().has_value());
}

TEST_CASE("a token becomes a bearer header, and no token becomes none",
          "[remote]") {
    RemoteSource anonymous;
    CHECK(anonymous.authHeaders().empty());

    RemoteSource gated;
    gated.token = "hf_secret";
    const auto headers = gated.authHeaders();
    REQUIRE(headers.size() == 1);
    CHECK(headers[0].first == "Authorization");
    CHECK(headers[0].second == "Bearer hf_secret");
}

// ---------------------------------------------------------------------------
// Retry
// ---------------------------------------------------------------------------

TEST_CASE("only transient failures are retried", "[remote]") {
    // These will not become successes however many times they are asked.
    CHECK_FALSE(net::isRetryable(200));
    CHECK_FALSE(net::isRetryable(206));
    CHECK_FALSE(net::isRetryable(401));
    CHECK_FALSE(net::isRetryable(403));
    CHECK_FALSE(net::isRetryable(404));

    // These usually will.
    CHECK(net::isRetryable(408));
    CHECK(net::isRetryable(429));
    CHECK(net::isRetryable(500));
    CHECK(net::isRetryable(502));
    CHECK(net::isRetryable(503));
}

TEST_CASE("retry backoff doubles and then stops growing", "[remote]") {
    const RetryPolicy policy{.maxAttempts = 10,
                             .initialDelayMillis = 500,
                             .maxDelayMillis = 4000};

    CHECK(policy.delayForAttempt(0) == 0);
    CHECK(policy.delayForAttempt(1) == 500);
    CHECK(policy.delayForAttempt(2) == 1000);
    CHECK(policy.delayForAttempt(3) == 2000);
    CHECK(policy.delayForAttempt(4) == 4000);
    // Capped, not unbounded: an install should keep trying at a steady pace
    // rather than eventually sleeping for hours.
    CHECK(policy.delayForAttempt(9) == 4000);
}

// ---------------------------------------------------------------------------
// Live network
// ---------------------------------------------------------------------------

TEST_CASE("the client fetches a small file over HTTPS", "[.][network]") {
    auto client = net::HttpClient::create();
    REQUIRE(client.has_value());

    const RemoteSource source;
    auto config = fetchSmallFile(*client, source, "config.json");
    REQUIRE(config.has_value());
    REQUIRE_FALSE(config->empty());

    const std::string text{reinterpret_cast<const char*>(config->data()), config->size()};
    CHECK(text.find("num_hidden_layers") != std::string::npos);
}

TEST_CASE("the repository listing reports files and sizes", "[.][network]") {
    auto client = net::HttpClient::create();
    REQUIRE(client.has_value());

    auto files = listRemoteFiles(*client, RemoteSource{});
    REQUIRE(files.has_value());
    REQUIRE_FALSE(files->empty());

    bool sawWeights = false;
    for (const RemoteFile& file : *files) {
        if (file.name.ends_with(".safetensors")) {
            sawWeights = true;
            // A size of zero would make the streaming reader's bounds check
            // useless.
            CHECK(file.size > 0);
        }
    }
    CHECK(sawWeights);
}

TEST_CASE("ranged reads land on the bytes they asked for", "[.][network]") {
    auto client = net::HttpClient::create();
    REQUIRE(client.has_value());

    const RemoteSource source;

    // config.json is small, so its whole content can be compared against what
    // ranged reads produce - which is the property the installer depends on.
    auto whole = fetchSmallFile(*client, source, "config.json");
    REQUIRE(whole.has_value());
    REQUIRE(whole->size() > 64);

    auto stream = RemoteFileStream::open(*client, source, "config.json", whole->size());
    REQUIRE(stream.has_value());

    // Sequential, then a forward skip, in the pattern the repacker produces.
    std::vector<u8> head(32);
    REQUIRE(stream->readExactAt(0, head).has_value());
    CHECK(std::equal(head.begin(), head.end(), whole->begin()));

    std::vector<u8> later(16);
    const u64 offset = whole->size() - 16;
    REQUIRE(stream->readExactAt(offset, later).has_value());
    CHECK(std::equal(later.begin(), later.end(), whole->begin() + offset));
}

TEST_CASE("a backward seek reopens rather than returning the wrong bytes",
          "[.][network]") {
    auto client = net::HttpClient::create();
    REQUIRE(client.has_value());

    const RemoteSource source;
    auto whole = fetchSmallFile(*client, source, "config.json");
    REQUIRE(whole.has_value());

    auto stream = RemoteFileStream::open(*client, source, "config.json", whole->size());
    REQUIRE(stream.has_value());

    std::vector<u8> tail(16);
    REQUIRE(stream->readExactAt(whole->size() - 16, tail).has_value());

    std::vector<u8> head(16);
    REQUIRE(stream->readExactAt(0, head).has_value());
    CHECK(std::equal(head.begin(), head.end(), whole->begin()));
    // The seek backwards had to cost a reconnect; if it did not, the stream
    // handed back whatever came next instead.
    CHECK(stream->reconnects() > 0);
}

TEST_CASE("reading past the end of a file is refused", "[.][network]") {
    auto client = net::HttpClient::create();
    REQUIRE(client.has_value());

    const RemoteSource source;
    auto stream = RemoteFileStream::open(*client, source, "config.json", 100);
    REQUIRE(stream.has_value());

    std::vector<u8> buffer(64);
    auto status = stream->readExactAt(80, buffer);
    REQUIRE_FALSE(status.has_value());
    CHECK(status.error().code() == ErrorCode::InvalidArgument);
}

TEST_CASE("a missing file reports not-found rather than retrying forever",
          "[.][network]") {
    auto client = net::HttpClient::create();
    REQUIRE(client.has_value());

    auto missing = fetchSmallFile(*client, RemoteSource{}, "no-such-file.json");
    REQUIRE_FALSE(missing.has_value());
    CHECK(missing.error().code() == ErrorCode::NotFound);
}
