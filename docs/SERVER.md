# tf-server

```
tf-server --model C:\models\gemma4.gturbo
```

Then point any OpenAI client at `http://127.0.0.1:8080/v1` with any API key.

| Endpoint | |
|---|---|
| `POST /v1/chat/completions` | streaming and non-streaming |
| `POST /v1/completions` | raw completion, no instruction template |
| `GET /v1/models` | |
| `GET /health` | counters, including prompt cache effectiveness |

## Browser clients and CORS

`tf-server` accepts browser-based clients directly. Every JSON and streaming
response includes these headers:

- `Access-Control-Allow-Origin: *`
- `Access-Control-Allow-Methods: GET, POST, OPTIONS`
- `Access-Control-Allow-Headers: Content-Type, Authorization`

An `OPTIONS` preflight request to any path returns `204 No Content` with the
same headers. No proxy is needed for a local browser extension or web UI.

The wire format *is* the compatibility surface, which is why it has its own
tests: a field spelled wrong here does not fail, it makes a client silently
ignore a setting or render an empty message. `Api.cpp` needs neither a socket
nor a GPU, so those shapes are pinned directly.

## Prompt caching is the reason this is fast

A chat client resends the entire conversation every turn. Without caching, turn
ten reprocesses everything from turn one - the cost per turn grows without
bound even though only one message was added.

The generator keeps a record of every token the KV cache represents. A new
prompt is matched against it, the cache is rewound to the first divergence, and
only the remainder is prefilled. In practice the divergence is at the end, so a
follow-up turn reprocesses only the new message. Measured over HTTP on a
six-fact conversation: turn two reused 43 of its 61 prompt tokens.

`/health` reports the hit rate. On a chat workload a figure near zero means
something is defeating it - a client that reorders messages, or a system prompt
being rewritten each turn.

### The rewind is bounded, and the bound is not obvious

The 25 sliding-window layers hold only the most recent `capacity` positions.
Rewinding to P and re-prefilling from there means the token at P attends back to
`P - window + 1`, and those rows have to still be in the ring. Once the sequence
is longer than a ring that bounds the rewind to roughly the headroom beyond the
window - a hundred-odd tokens, not thousands.

Rewinding past that point does not fail loudly. The model attends over rows
belonging to a different part of the sequence and produces confident nonsense.
So `KVCacheManager::earliestSafeRewind()` computes the bound and the generator
asks rather than assuming; when the answer is "too far", the prompt is
reprocessed from scratch.

This costs nothing in the common case, because the common case is a *forward*
extension where the shared prefix is the entire cache and there is no rewind at
all.

Correctness is pinned by generating the same prompt cold and warm and requiring
byte-identical output. Greedy decoding makes that exact rather than approximate:
the reused rows are the rows a cold run would have written, so any difference
means the cache changed the computation.

## Concurrency

Connections are handled concurrently, one thread each. **Inference is
serialized behind a mutex.**

That is not a simplification to revisit. There is one GPU, one KV cache and one
prompt cache; a second concurrent generation would need its own cache, interleave
badly on the same device, and destroy the first request's prompt cache. A second
client waits rather than being refused, which is the right trade for a
single-user local server.

Thread-per-connection rather than an IOCP reactor for the same reason: the work
is serialized on one GPU anyway, so a reactor would be complexity bought for
nothing.

## Streaming

Server-sent events: `data: {json}\n\n` per chunk, terminated by `data: [DONE]`.
The first chunk carries `delta.role`, later ones carry only `delta.content`, and
the final one carries `finish_reason` with an empty delta - which is what every
SDK reads.

`finish_reason` is `stop` for a stop token or stop string, and `length` for
either the token limit or a full context. Both of the latter mean the answer was
cut off, which is what a client acts on; it does not care which limit was hit.

Once the 200 is on the wire there is no way to report a failure as a status, so
everything that can be validated is validated before streaming starts. A later
failure can only end the stream, and is sent as an error frame first.

**A disconnected client stops generation.** When a write fails mid-stream the
text callback returns false, which unwinds the decode loop. An abandoned request
should stop costing GPU time rather than run to `max_tokens`.

## Status codes

Anything wrong with a request body is 400, whatever internal code it surfaced
as. Mapping by internal code reported a missing `messages` key as a 500 -
blaming the server for the client's mistake - and would have turned a missing
JSON key into a 404. Failures *during* generation keep their meaning: a prompt
too long for the context is 400, an allocation failure is 503.

Errors use OpenAI's envelope, `{"error":{"message":...}}`. A bare string body
renders as an empty message in every SDK, which turns a clear server-side error
into a mystery on the client.

## Binding

Loopback by default. This process holds a 13 GiB model and does not
authenticate, so `--host` on anything else should be a deliberate act.

The listening socket uses `SO_EXCLUSIVEADDRUSE` rather than `SO_REUSEADDR`: on
Windows the latter lets another process bind the same port and steal
connections. A port already in use is reported rather than silently shared.

`--port 0` picks a free port and prints it.

## What it does not do

- `n` above 1. It would need independent sampling streams over one KV cache.
- Batched prompts on `/v1/completions`.
- Image content parts. They are refused rather than dropped, because silently
  ignoring an image answers a question the user did not ask.
- Embeddings, fine-tuning, function calling.
- TLS. It is loopback; put a reverse proxy in front if it needs to leave the
  machine.
