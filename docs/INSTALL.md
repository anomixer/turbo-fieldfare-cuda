# Installing

```
tf-repack --install --output C:\models\gemma4.gturbo
```

Streams the checkpoint from Hugging Face and writes the `.gturbo` install
directly. The source is never written to disk.

That last point is the reason this exists rather than "download, then repack".
The conventional route needs room for a 15 GB download *and* a 14.3 GiB install
at the same time, then deletes the download. Streaming needs room for the
install alone. A machine with 20 GB free can install a model it could not
otherwise obtain.

## Why streaming works here

The repacker's copy operations are sorted by `(shard, source offset)`, which M1a
did for local sequential reads. That ordering is what makes a network source
viable: each shard is consumed strictly forward, so one HTTP range request
covers all of it. The alternative - a request per operation - would be 35,686
round trips.

`RemoteFileStream` therefore keeps one connection per shard and reads forward.
A gap under 256 KiB is absorbed by reading and discarding, which beats a new
HTTPS request plus TLS handshake. A larger gap, or a backward seek, reopens at
the new offset. With a sorted plan the backward case should never happen; it is
implemented anyway because returning the wrong bytes would be far worse than
being slow.

Planning needs the safetensors headers, not the shards. Those come from two
ranged reads per shard - eight bytes for the length, then about 90 KB of JSON -
so choosing the layout costs a fraction of a percent of the transfer.

## Resume

A 14 GB download that restarts from zero after a dropped connection is not
usable, so progress is checkpointed every 256 MiB into `install-state.json` in
the partial directory:

```json
{
  "repoId": "mlx-community/gemma-4-26b-a4b-it-4bit",
  "revision": "0d77464eeb233a2da68ebf9d7dc4edaac7db956d",
  "planFingerprint": "...@... shards=[...] ops=35686 resident=1353689568 ...",
  "opsCompleted": 27893,
  "bytesWritten": 11038036778
}
```

Operations are deterministic given the plan, and the output files persist in the
partial directory, so an operation index is a complete resume point.

**The fingerprint is the part that matters.** It records everything that decides
where a byte lands: the repository, the revision, the shard set, the operation
count, the expert stride. Resuming against a plan that differs in any of those
would interleave two layouts into one file and produce an install that passes
its own hash check while being internally wrong - the worst possible failure,
because it looks fine. A mismatch starts over instead.

Two ordering rules that this depends on, both learned the hard way:

- The state file is written **after** flushing the outputs, never before. It is
  trusted completely on resume, so anything it over-reports is a hole in the
  install.
- The final checkpoint is written **before** the files are closed, because
  writing it flushes them. Getting this backwards fails only at the very end of
  a 14 GB run, which is an expensive place to find out.

Resumed files are reopened with `File::openWrite`, which keeps their contents.
`File::create` would truncate - throwing away precisely the work the resume
exists to preserve.

## Concurrency and atomicity

The partial directory is guarded by `install.lock`, opened with no sharing at
all, so a second installer fails immediately with a clear message rather than
the two interleaving writes. It is `FILE_FLAG_DELETE_ON_CLOSE`, so a crashed
installer leaves nothing a later run cannot clear. The lock is taken before
anything is read or removed, so two installers cannot both decide to start fresh
and then race.

The install is built in `<output>.partial` and renamed on success. The manifest
is written last: its presence is the completion marker, so an interrupted run
can never leave a directory that looks complete.

## Network behaviour

WinHTTP rather than libcurl or WinINet: in-box on every supported Windows, works
from a service, and follows the 302 to Hugging Face's CDN with no redirect
handling of our own. It picks up the system proxy automatically.

Retries use exponential backoff capped at 30 seconds, and only for failures that
might resolve - 408, 429 and 5xx. A 404 or a 403 on a gated repository will not
become a 200 however many times it is asked, so those fail immediately with an
explanation.

A server that ignores `Range` answers 200 with the whole body. That is only
correct from offset zero; anywhere else the bytes would be silently misaligned,
so it is treated as an error.

## Gated repositories

`--hf-token`, or `HF_TOKEN` in the environment. The flag wins so a one-off can
override the environment. The default checkpoint is public and needs neither.

## Verification

`--verify-install <dir>` re-hashes every file against the manifest. The install
does this automatically before promoting unless `--no-verify` is passed; it
costs a full re-read of 14.3 GiB, and it is on by default because a silently
corrupt install surfaces much later as unexplainable output.

The strongest check available is that a streamed install and a local repack of
the same revision produce **byte-identical** files. Both drive the same planner
over the same bytes, so the digests in the two manifests match exactly.

## Repacking a local checkpoint

If the checkpoint is already downloaded:

```
tf-repack --checkpoint C:\hf\gemma-4-26b-a4b-it-4bit --output C:\models\gemma4.gturbo
```

Same planner, same output; only the byte source differs.
