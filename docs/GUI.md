# The desktop app

```
scripts\build-gui.ps1
build\gui\Release\TurboFieldfare.exe
```

The normal `build.bat` workflow also invokes `scripts\build-gui.ps1` after the
command-line/server binaries finish. Run the GUI script directly when only the
desktop front end needs to be rebuilt.

Two processes. `TurboFieldfare.exe` is the window; `tf-decode.exe` holds the
model. The window starts the service if one is not already listening, and the
service exits with the window.

## Why two processes

The GUI never loads a model. A XAML crash, a hung UI thread or a shader-compile
stall would otherwise take a CUDA context and 13 GiB of VRAM with it, and
reloading costs about eight seconds - long enough that users learn not to close
the window.

The dependency check is mechanical rather than aspirational: `TurboFieldfare.exe`
links `tfsvc`, `tfcore` and `tfruntime`, and its import table is
`KERNEL32`, `USER32`, `ADVAPI32`, the CRT, `ole32`, the WinRT stubs and the
Windows App SDK bootstrapper. No CUDA, no `winhttp`. `tfruntime` is linked only
for `SamplingParams::validate()`; a static link pulls that object and nothing
else.

## Almost no XAML markup

The visual tree is built in code. There is exactly one markup file, `App.xaml`,
and it exists because the alternative does not work.

The original design had no markup at all: WinUI's XAML compiler is MSBuild-only
and drags `midl`, generated headers and a second project system into a
repository that is otherwise CMake, so building the tree imperatively meant
neither tool ran. That part was right, and the tree is still built in code.

What was wrong is that an app with no markup has no `Application` definition
either. Nothing populates `Application.Resources`, so no control has a template,
and the app throws a bare `E_FAIL` in its first layout pass - after the window
has activated, which is why it appeared to start and then vanish. Loading
`XamlControlsResources` by hand does not substitute for it: that needs a
`resources.pri`, and only the XAML build step produces one.

So `App.xaml` carries the control styles and nothing else.

### Wiring the markup compiler by hand

The Windows App SDK hooks `MarkupCompilePass1` into `PrepareResources`, a
managed-project target a `vcxproj` never runs. Its own comment points native
builds at `Microsoft.Windows.UI.Xaml.Cpp.targets`, which ships with a Visual
Studio component that is not installed here and is not in the NuGet package. The
compiler is imported and its condition is satisfied; only the sequencing is
missing. `TurboFieldfare.vcxproj` supplies it:

| Piece | Why |
|---|---|
| `TfCollectXamlReferences` | Populates `@(ReferencePath)` with the SDK winmds and the Windows platform *contracts*. Empty, the compiler fails `WMC1007`; with `UnionMetadata\Windows.winmd` instead of the contracts it fails `WMC9999` on `Windows.Foundation.FoundationContract`, because it resolves by assembly identity. |
| `TfMarkupCompile` | Runs both passes before `ClCompile`, and adds `XamlTypeInfo.Impl.g.cpp` (pass 1) and `XamlTypeInfo.g.cpp` (pass 2) to the compile. Pass 1's output calls into pass 2's table, so pass 1 alone fails to link. |
| `$(IntDir)` on the include path | Where `App.xaml.g.h` lands. Without it `App.g.h`'s `__has_include` quietly misses and the only symptom is `InitializeComponent` not being declared. |
| `ForcedIncludeFiles=unknwn.h` | The generated type-info file implements classic COM interfaces and needs it ahead of the C++/WinRT headers. A project with a PCH gets this for free; this one has none. |
| `TfCopyXbf` | Puts `App.xbf` beside the executable, where it is loaded from at runtime. |

### Other things the project file needs, none of them obvious

| Setting | Why |
|---|---|
| `ResolveNuGetPackages=false` | Otherwise the legacy project.json resolver runs instead of the PackageReference path and fails with "Sequence contains no elements". |
| A no-op `AddProcessedXamlFilesToCopyLocal` target | The common targets reference it once XAML processing is on; its real definition is behind the packaging path this project does not take. |
| `LanguageStandard` in `ClCompile`, not a `PropertyGroup` | In a `PropertyGroup` it is silently ignored, and `tfcore`'s headers then fail on `std::expected`. |

## Windows App Runtime

Unpackaged, so the runtime is loaded by the bootstrapper at startup. It asks for
**1.7**. If that is not installed the app says so and points at `tf-cli`, which
needs nothing extra - rather than showing the SDK's own modal dialog, which
blocks forever with no window behind it.

Getting this wrong is quiet: asking for a version that is not installed makes
the process hang in the bootstrapper before `main`, with no window and nothing
in the log.

## Threading

Every callback from the service arrives on the pipe reader thread, and every
XAML property must be touched on the UI thread. The bridge is one
`DispatcherQueue.TryEnqueue` per callback, wrapped once in `OnLaunched`.
Touching a control from the pipe thread is a crash, and an intermittent one.

Connecting also runs off the UI thread: it blocks until the service has loaded
13 GiB, and doing it inline would leave the window unpainted until the model was
ready - which looks exactly like a hang.

## Markdown

The model answers in markdown, so the transcript is a `RichTextBlock` and a
small parser turns the answer into styled runs. A `TextBlock` can only show the
asterisks and backticks as literal text, which is what it did.

It covers what a chat model actually emits: headings, fenced and inline code,
bold, italic, and bullet or numbered lists. It is not CommonMark and does not
try to be - anything unrecognised falls through as its own literal text, which
is the right failure here. Numbered lists lose their numbering, because a
`RichTextBlock` has no list primitive and a hanging indent per item is more
machinery than this earns.

The user's own prompts are rendered as plain text, not markdown. Formatting
what someone typed would eat their asterisks.

Hand-written rather than the Community Toolkit's `MarkdownTextBlock`, which the
plan listed first: that is another package and more of the XAML build machinery
this project already spends enough on, for a few hundred lines of parsing.

## Conversation and the prompt cache

The window sends the *whole* conversation on every turn, not just the new
message. That is what lets the service's prompt cache reuse everything but the
newest turn. Sending only the new message would be smaller on the wire and far
slower to answer.

The inspector line under the transcript reports what the cache actually saved,
so a configuration that defeats it is visible rather than merely slow.

## Diagnostics

Set `TF_GUI_LOG` to a file path to trace startup. A GUI-subsystem process that
dies before it paints has nowhere else to say why; this is the log to ask for
when the window does not appear.

## What is verified, and what is not

Verified by running it: the app builds from a clean tree, bootstraps the Windows
App Runtime, constructs the visual tree, activates the window, starts
`tf-decode`, and connects over the named pipe - the trace ends at `connected`.
The window is real: visible, 1920x1023, titled, and it survives instead of
dying in layout.

Sending is verified too, by driving the window through UI Automation rather
than by hand: set the prompt box, invoke Send, and watch the window stay
responsive for the whole generation. A 200-word answer came back at 19.1 tok/s
with 14 prompt tokens served from the cache, and `IsHungAppWindow` stayed false
throughout. That is worth keeping as the way to test this - it exercises the
real event wiring, which no unit test reaches.

## The freeze on Send

Worth recording, because the cause was not in the GUI.

Pressing Send wedged the UI thread permanently, stuck in `WriteFile`. The pipe
client opened its handle without `FILE_FLAG_OVERLAPPED`, and Windows serializes
I/O on a synchronous handle: the reader thread sat in `ReadFile` waiting for a
message that could only arrive once the request was sent, and the UI thread's
write queued behind that read forever.

`tf-decode-client` never saw it because a request/response client writes and
then reads, never both at once. Any duplex client of that channel would have
hit it. Both ends now do real overlapped transfers with one event per
direction - the server needed it as well, since its handle was already
overlapped for the interruptible accept while its transfers passed `nullptr`
for the `OVERLAPPED`.

Separately, the transcript used to re-render once per token. A render rebuilds
the whole transcript and re-lays out the block, so at 30-40 tokens a second
that is quadratic in the conversation, and the pipe thread enqueued renders
whether or not the UI thread could keep up. Renders now coalesce to one pending
at a time at low priority. That was not the freeze, but it would have become
one.

`tf-decode-client --prompt "..."` is the way to tell a GUI problem from a
service problem.

## Not built

The model install view. `tf-repack --install` does the work and reports
progress, but it is not wired into the window: an install UI needs progress,
cancellation, resume and error recovery surfaces, and half of that is worse than
a documented command line.
