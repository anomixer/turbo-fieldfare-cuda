// TurboFieldfare - the WinUI 3 front end.
//
// The visual tree is built in code rather than in XAML markup. That is a build
// decision, not a stylistic one: the XAML compiler is MSBuild-only and drags
// midl, generated headers and a second project system into a repository that is
// otherwise CMake. Building the tree imperatively keeps this to one vcxproj that
// links the CMake-built static libraries, which is what the plan called for.
//
// This process never loads a model. It talks to tf-decode over a named pipe, so
// a crash here costs a window rather than 13 GiB of VRAM and an eight second
// reload.

#include <windows.h>

#include <MddBootstrap.h>

#include <cstdio>
#include <cwchar>
#include <format>
#include <functional>
#include <algorithm>
#include <memory>
#include <mutex>
#include <filesystem>
#include <vector>
#include <thread>
#include <string>

#include <winrt/Microsoft.UI.Dispatching.h>
#include <winrt/Microsoft.UI.Input.h>
#include <winrt/Microsoft.UI.Text.h>
#include <winrt/Microsoft.UI.Xaml.Controls.Primitives.h>
#include <winrt/Microsoft.UI.Xaml.Controls.h>
#include <winrt/Microsoft.UI.Xaml.Documents.h>
#include <winrt/Microsoft.UI.Xaml.Input.h>
#include <winrt/Microsoft.UI.Xaml.Media.h>
#include <winrt/Microsoft.UI.Xaml.h>
#include <winrt/Windows.Foundation.Collections.h>
#include <winrt/Windows.Foundation.h>
#include <winrt/Windows.System.h>
#include <winrt/Windows.UI.Core.h>
#include <winrt/Windows.UI.h>

// Generated from App.idl, and it pulls in App.xaml.g.h - the XAML build step's
// output - when that exists. AppT and InitializeComponent come from there, and
// InitializeComponent is what loads the control styles.
#include "App.g.h"

#include "AppModel.h"

using namespace winrt;
using namespace winrt::Microsoft::UI::Xaml;
using namespace winrt::Microsoft::UI::Xaml::Controls;
using namespace winrt::Microsoft::UI::Xaml::Documents;

namespace {

/// Startup diagnostics for a windowless subsystem.
///
/// A GUI process that dies before it paints has nowhere to say why. Set
/// TF_GUI_LOG to a file path to find out where it got to - off by default,
/// since an installed application has no business writing beside its
/// executable.
void trace(std::string_view what) {
    // The UI thread and the pipe thread both trace. Unsynchronized appends
    // interleave mid-line, which corrupts the log precisely when it is being
    // read to find out what those two threads did to each other.
    static std::mutex mutex;
    const std::lock_guard lock{mutex};

    wchar_t path[1024] = {};
    if (::GetEnvironmentVariableW(L"TF_GUI_LOG", path, 1024) == 0) {
        return;
    }
    if (FILE* handle = ::_wfopen(path, L"a"); handle != nullptr) {
        std::fwrite(what.data(), 1, what.size(), handle);
        std::fputc('\n', handle);
        std::fclose(handle);
    }
}

[[nodiscard]] hstring toHstring(std::string_view text) {
    if (text.empty()) {
        return {};
    }
    const int needed = ::MultiByteToWideChar(CP_UTF8, 0, text.data(),
                                             static_cast<int>(text.size()), nullptr, 0);
    std::wstring wide(static_cast<size_t>(needed), L'\0');
    ::MultiByteToWideChar(CP_UTF8, 0, text.data(), static_cast<int>(text.size()), wide.data(),
                          needed);
    return hstring{wide};
}

[[nodiscard]] std::string fromHstring(hstring const& text) {
    if (text.empty()) {
        return {};
    }
    const int needed = ::WideCharToMultiByte(CP_UTF8, 0, text.c_str(),
                                             static_cast<int>(text.size()), nullptr, 0, nullptr,
                                             nullptr);
    std::string out(static_cast<size_t>(needed), '\0');
    ::WideCharToMultiByte(CP_UTF8, 0, text.c_str(), static_cast<int>(text.size()), out.data(),
                          needed, nullptr, nullptr);
    return out;
}

/// Where tf-decode.exe lives: next to this executable, since they ship
/// together.
[[nodiscard]] std::wstring serviceBesideMe() {
    wchar_t path[MAX_PATH] = {};
    const DWORD length = ::GetModuleFileNameW(nullptr, path, MAX_PATH);
    std::wstring directory{path, length};
    const size_t slash = directory.find_last_of(L'\\');
    if (slash == std::wstring::npos) {
        return L"tf-decode.exe";
    }
    return directory.substr(0, slash + 1) + L"tf-decode.exe";
}

/// The model directory, from the environment or the usual location.
[[nodiscard]] std::wstring defaultModelDir() {
    std::vector<std::filesystem::path> candidates;
    wchar_t buffer[1024] = {};
    if (::GetEnvironmentVariableW(L"TF_GTURBO_DIR", buffer, 1024) > 0) {
        candidates.emplace_back(buffer);
    }
    // Release layout: <repo>\build\gui\Release\TurboFieldfare.exe.
    wchar_t modulePath[MAX_PATH] = {};
    const DWORD moduleLength = ::GetModuleFileNameW(nullptr, modulePath, MAX_PATH);
    if (moduleLength > 0) {
        const std::filesystem::path module{modulePath, modulePath + moduleLength};
        candidates.push_back(module.parent_path().parent_path().parent_path().parent_path() /
                             L"models" / L"gemma4.gturbo");
    }
    if (::GetEnvironmentVariableW(L"USERPROFILE", buffer, 1024) > 0) {
        candidates.emplace_back(std::wstring{buffer} + L"\\model-data\\gemma4.gturbo");
    }
    candidates.emplace_back(L"C:\\models\\gemma4.gturbo");
    for (const auto& candidate : candidates) {
        if (std::filesystem::exists(candidate / L"manifest.json")) {
            return candidate.wstring();
        }
    }
    return {};
}

/// The window and everything in it.
///
/// Named after the views the plan listed - composer, output, status, inspector -
/// which here are regions of one grid rather than separate types. At this size
/// splitting them into classes would add indirection without removing anything.
class MainView {
public:
    explicit MainView(tf::gui::AppModel& model) : model_(model) {}

    [[nodiscard]] Window build() {
        window_ = Window{};
        window_.Title(L"TurboFieldfare");

        Grid root;
        root.RowDefinitions().Append(rowAuto());   // status
        root.RowDefinitions().Append(rowStar());   // output
        root.RowDefinitions().Append(rowAuto());   // inspector
        root.RowDefinitions().Append(rowAuto());   // composer

        root.Children().Append(buildStatusBar());
        root.Children().Append(buildOutputPane());
        root.Children().Append(buildInspector());
        root.Children().Append(buildComposer());

        Grid::SetRow(statusBar_, 0);
        Grid::SetRow(outputScroll_, 1);
        Grid::SetRow(inspector_, 2);
        Grid::SetRow(composer_, 3);

        window_.Content(root);
        return window_;
    }

    /// Every one of these runs on the UI thread, because the reader thread
    /// hands them over through the dispatcher. Touching a XAML property from
    /// the pipe thread would be a crash, and an intermittent one.
    void onReady(const tf::svc::ReadyInfo& info) {
        trace("onReady running on UI thread");
        status_.Text(toHstring(std::format("{} on {} - {} token context{}", info.model,
                                           info.device, info.contextLength,
                                           info.fullyResident ? "" : ", streaming experts")));
        sendButton_.IsEnabled(true);
        promptBox_.IsEnabled(true);
        promptBox_.Focus(FocusState::Programmatic);
    }

    void onToken(std::string piece) {
        model_.appendToAnswer(piece);
        scheduleTranscriptRefresh();
    }

    /// Coalesces transcript renders, and the coalescing is the whole point.
    ///
    /// Rendering per token wedged the window. A render rebuilds the entire
    /// transcript and re-sets the TextBlock, so XAML re-measures everything, and
    /// ChangeView adds a scroll pass on top; that is O(n) in the conversation,
    /// run once per token at 30-40 tokens a second, which is O(n squared)
    /// overall. Past a few hundred tokens the UI thread cannot keep up, and
    /// since the pipe thread enqueues regardless there is no backpressure - the
    /// queue grows without bound and the thread never returns to idle. It stops
    /// repainting and stops accepting input, which reads as a freeze right
    /// after Send.
    ///
    /// One pending render at a time collapses a burst into a single pass and
    /// makes the render rate follow what the thread can actually do. Low
    /// priority puts it behind input and layout, so typing stays live while
    /// tokens stream.
    void scheduleTranscriptRefresh() {
        if (transcriptRefreshPending_) {
            return;
        }
        transcriptRefreshPending_ = true;
        dispatcher().TryEnqueue(Microsoft::UI::Dispatching::DispatcherQueuePriority::Low,
                                [this] {
                                    transcriptRefreshPending_ = false;
                                    refreshTranscript();
                                });
    }

    void onDone(const tf::svc::DoneInfo& info) {
        const double tokensPerSecond =
                info.decodeSeconds > 0.0
                        ? static_cast<double>(info.generatedTokens) / info.decodeSeconds
                        : 0.0;
        inspector_.Text(toHstring(std::format(
                "{} prompt tokens ({} reused) in {:.2f}s - {} generated in {:.2f}s at "
                "{:.1f} tok/s - {}",
                info.promptTokens, info.cachedPromptTokens, info.prefillSeconds,
                info.generatedTokens, info.decodeSeconds, tokensPerSecond, info.reason)));
        setBusy(false);
        // The last tokens may have arrived after the final coalesced render was
        // already queued, so the answer needs one guaranteed pass at the end.
        refreshTranscript();
    }

    void onError(std::string message) {
        model_.appendToAnswer("\n[error] " + message);
        refreshTranscript();
        setBusy(false);
    }

    void onDisconnected() {
        status_.Text(L"The decode service stopped.");
        sendButton_.IsEnabled(false);
        promptBox_.IsEnabled(false);
        setBusy(false);
    }

    [[nodiscard]] Microsoft::UI::Dispatching::DispatcherQueue dispatcher() const {
        return window_.DispatcherQueue();
    }

private:
    static RowDefinition rowAuto() {
        RowDefinition row;
        row.Height(GridLengthHelper::Auto());
        return row;
    }
    static RowDefinition rowStar() {
        RowDefinition row;
        row.Height(GridLengthHelper::FromValueAndType(1, GridUnitType::Star));
        return row;
    }

    FrameworkElement buildStatusBar() {
        status_ = TextBlock{};
        status_.Text(L"Starting the decode service...");
        status_.Margin(ThicknessHelper::FromLengths(12, 10, 12, 6));
        status_.TextWrapping(TextWrapping::Wrap);
        statusBar_ = status_;
        return statusBar_;
    }

    FrameworkElement buildOutputPane() {
        // RichTextBlock rather than TextBlock: the model answers in markdown,
        // and a TextBlock can only show the asterisks and backticks raw.
        transcript_ = RichTextBlock{};
        transcript_.TextWrapping(TextWrapping::Wrap);
        transcript_.IsTextSelectionEnabled(true);
        transcript_.Margin(ThicknessHelper::FromLengths(12, 4, 12, 4));

        outputScroll_ = ScrollViewer{};
        outputScroll_.Content(transcript_);
        outputScroll_.VerticalScrollBarVisibility(ScrollBarVisibility::Auto);
        return outputScroll_;
    }

    FrameworkElement buildInspector() {
        inspector_ = TextBlock{};
        inspector_.Margin(ThicknessHelper::FromLengths(12, 4, 12, 4));
        inspector_.Opacity(0.7);
        inspector_.TextWrapping(TextWrapping::Wrap);
        return inspector_;
    }

    FrameworkElement buildComposer() {
        promptBox_ = TextBox{};
        promptBox_.PlaceholderText(L"Ask something. Enter to send, Shift+Enter for a new line.");
        promptBox_.AcceptsReturn(true);
        promptBox_.TextWrapping(TextWrapping::Wrap);
        promptBox_.MaxHeight(140);
        promptBox_.IsEnabled(false);

        // Enter sends, Shift+Enter inserts a newline. Doing this on KeyDown
        // rather than KeyUp is what stops the newline reaching the box.
        // PreviewKeyDown, not KeyDown. A TextBox with AcceptsReturn consumes
        // Enter itself to insert the newline, and the bubbling KeyDown never
        // reaches a handler out here - the key appeared to do nothing but add a
        // line. PreviewKeyDown tunnels down before the control sees it, so
        // marking it handled is what stops the newline.
        //
        // Shift comes from GetKeyState rather than
        // InputKeyboardSource::GetKeyStateForCurrentThread: it is the state as
        // of the message being processed, which is exactly the question being
        // asked, and it cannot fail on a thread with no input source attached.
        promptBox_.PreviewKeyDown([this](auto&&, Input::KeyRoutedEventArgs const& args) {
            if (args.Key() != Windows::System::VirtualKey::Enter) {
                return;
            }
            const bool shiftDown = (::GetKeyState(VK_SHIFT) & 0x8000) != 0;
            trace(shiftDown ? "enter with shift, newline" : "enter, submitting");
            if (shiftDown) {
                return;
            }
            args.Handled(true);
            submit();
        });

        sendButton_ = Button{};
        sendButton_.Content(box_value(L"Send"));
        sendButton_.IsEnabled(false);
        sendButton_.Margin(ThicknessHelper::FromLengths(8, 0, 0, 0));
        sendButton_.Click([this](auto&&, auto&&) { submit(); });

        stopButton_ = Button{};
        stopButton_.Content(box_value(L"Stop"));
        stopButton_.IsEnabled(false);
        stopButton_.Margin(ThicknessHelper::FromLengths(8, 0, 0, 0));
        stopButton_.Click([this](auto&&, auto&&) { model_.cancel(); });

        Button clearButton;
        clearButton.Content(box_value(L"Clear"));
        clearButton.Margin(ThicknessHelper::FromLengths(8, 0, 0, 0));
        clearButton.Click([this](auto&&, auto&&) {
            static_cast<void>(model_.reset());
            transcript_.Blocks().Clear();
            inspector_.Text(L"");
        });

        Grid row;
        row.Margin(ThicknessHelper::FromLengths(12, 6, 12, 12));
        ColumnDefinition stretch;
        stretch.Width(GridLengthHelper::FromValueAndType(1, GridUnitType::Star));
        row.ColumnDefinitions().Append(stretch);
        for (int i = 0; i < 3; ++i) {
            ColumnDefinition fixedWidth;
            fixedWidth.Width(GridLengthHelper::Auto());
            row.ColumnDefinitions().Append(fixedWidth);
        }

        row.Children().Append(promptBox_);
        row.Children().Append(sendButton_);
        row.Children().Append(stopButton_);
        row.Children().Append(clearButton);
        Grid::SetColumn(promptBox_, 0);
        Grid::SetColumn(sendButton_, 1);
        Grid::SetColumn(stopButton_, 2);
        Grid::SetColumn(clearButton, 3);

        composer_ = row;
        return composer_;
    }

    void submit() {
        const std::string prompt = fromHstring(promptBox_.Text());
        if (prompt.empty() || model_.busy()) {
            return;
        }
        trace("submit: clearing prompt box");
        promptBox_.Text(L"");
        trace("submit: setBusy");
        setBusy(true);
        trace("submit: inspector text");
        inspector_.Text(L"Thinking...");

        trace("submit: writing to pipe");
        if (const auto sent = model_.send(prompt); !sent) {
            model_.appendToAnswer("\n[error] " + sent.error().message());
            setBusy(false);
        }
        trace("submit: written, refreshing");
        refreshTranscript();
        trace("submit: done");
    }

    void setBusy(bool busy) {
        sendButton_.IsEnabled(!busy && model_.connected());
        stopButton_.IsEnabled(busy);
        promptBox_.IsEnabled(!busy && model_.connected());
    }

    /// Rebuilds the whole transcript. Fine at this scale - a conversation is a
    /// few thousand characters - and it keeps the streaming update from needing
    /// to track where the last token went.
    void refreshTranscript() {
        transcript_.Blocks().Clear();
        for (const tf::gui::Turn& turn : model_.turns()) {
            if (turn.text.empty()) {
                continue;
            }
            transcript_.Blocks().Append(speakerLine(turn.fromUser ? L"You" : L"Model"));
            if (turn.fromUser) {
                // A prompt is what the user typed. Rendering it as markdown
                // would eat their asterisks and backticks.
                transcript_.Blocks().Append(plainParagraph(toHstring(turn.text)));
            } else {
                renderMarkdown(turn.text);
            }
        }
        // A boxed double is an IReference<double>, which is what ChangeView
        // takes; its constructor is private, so it cannot be made directly.
        const auto bottom = box_value(outputScroll_.ScrollableHeight())
                                    .as<Windows::Foundation::IReference<double>>();
        outputScroll_.ChangeView(nullptr, bottom, nullptr);
    }

    tf::gui::AppModel& model_;
    Window window_{nullptr};

    FrameworkElement statusBar_{nullptr};
    FrameworkElement composer_{nullptr};
    ScrollViewer outputScroll_{nullptr};
    TextBlock status_{nullptr};
    RichTextBlock transcript_{nullptr};
    TextBlock inspector_{nullptr};
    TextBox promptBox_{nullptr};
    Button sendButton_{nullptr};
    Button stopButton_{nullptr};

    // -----------------------------------------------------------------------
    // Markdown
    //
    // Enough of it for what a chat model actually emits: headings, fenced and
    // inline code, bold, italic, and bullet or numbered lists. Not a CommonMark
    // implementation and not trying to be - anything unrecognised falls through
    // as its own literal text, which is the right failure for a chat window.
    //
    // Hand-written for the same reason the tokenizer and the JSON reader are:
    // the alternative is a markdown control from the Community Toolkit, which
    // means another package and more of the XAML build machinery this project
    // just finished fighting.
    // -----------------------------------------------------------------------

    [[nodiscard]] static Paragraph speakerLine(std::wstring_view who) {
        Run run;
        run.Text(who);
        run.FontWeight(Microsoft::UI::Text::FontWeights::SemiBold());
        Paragraph paragraph;
        paragraph.Inlines().Append(run);
        paragraph.Margin(ThicknessHelper::FromLengths(0, 10, 0, 2));
        return paragraph;
    }

    [[nodiscard]] static Paragraph plainParagraph(hstring const& text) {
        Run run;
        run.Text(text);
        Paragraph paragraph;
        paragraph.Inlines().Append(run);
        return paragraph;
    }

    void renderMarkdown(const std::string& source) {
        std::vector<std::string> lines;
        for (size_t start = 0; start <= source.size();) {
            const size_t end = source.find('\n', start);
            lines.push_back(source.substr(start, end == std::string::npos ? end : end - start));
            if (end == std::string::npos) {
                break;
            }
            start = end + 1;
        }

        for (size_t i = 0; i < lines.size(); ++i) {
            const std::string& line = lines[i];

            if (line.starts_with("```")) {
                std::string code;
                for (++i; i < lines.size() && !lines[i].starts_with("```"); ++i) {
                    if (!code.empty()) {
                        code += "\n";
                    }
                    code += lines[i];
                }
                transcript_.Blocks().Append(codeBlock(code));
                continue;
            }

            const std::string_view trimmed = trimLeft(line);
            if (trimmed.empty()) {
                continue;
            }

            if (trimmed.starts_with("#")) {
                size_t level = 0;
                while (level < trimmed.size() && trimmed[level] == '#') {
                    ++level;
                }
                if (level <= 6 && level < trimmed.size() && trimmed[level] == ' ') {
                    Paragraph heading;
                    appendInlines(heading, std::string{trimmed.substr(level + 1)});
                    for (const auto& run : heading.Inlines()) {
                        if (auto text = run.try_as<Run>()) {
                            text.FontWeight(Microsoft::UI::Text::FontWeights::Bold());
                            text.FontSize(level <= 2 ? 20.0 : 16.0);
                        }
                    }
                    heading.Margin(ThicknessHelper::FromLengths(0, 8, 0, 2));
                    transcript_.Blocks().Append(heading);
                    continue;
                }
            }

            // "- ", "* " and "1. " all become a bulleted line. Numbering is not
            // preserved, because a RichTextBlock has no list primitive and a
            // hanging indent per item is more machinery than this earns.
            std::string body;
            bool isListItem = false;
            if (trimmed.starts_with("- ") || trimmed.starts_with("* ")) {
                body = std::string{trimmed.substr(2)};
                isListItem = true;
            } else if (const size_t dot = trimmed.find(". ");
                       dot != std::string_view::npos && dot > 0 && dot <= 3 &&
                       std::ranges::all_of(trimmed.substr(0, dot),
                                           [](char c) { return c >= '0' && c <= '9'; })) {
                body = std::string{trimmed.substr(dot + 2)};
                isListItem = true;
            }

            Paragraph paragraph;
            if (isListItem) {
                Run bullet;
                bullet.Text(L"\x2022  ");
                paragraph.Inlines().Append(bullet);
                paragraph.Margin(ThicknessHelper::FromLengths(16, 1, 0, 1));
                appendInlines(paragraph, body);
            } else {
                paragraph.Margin(ThicknessHelper::FromLengths(0, 2, 0, 2));
                appendInlines(paragraph, std::string{trimmed});
            }
            transcript_.Blocks().Append(paragraph);
        }
    }

    [[nodiscard]] static Paragraph codeBlock(const std::string& code) {
        Run run;
        run.Text(toHstring(code));
        run.FontFamily(Media::FontFamily{L"Cascadia Mono, Consolas"});
        Paragraph paragraph;
        paragraph.Inlines().Append(run);
        paragraph.Margin(ThicknessHelper::FromLengths(16, 6, 0, 6));
        return paragraph;
    }

    /// Splits one line into styled runs. Single pass, because the markers are
    /// all paired and none of them nest in what a chat model produces.
    static void appendInlines(Paragraph const& paragraph, const std::string& line) {
        std::string pending;
        const auto flush = [&](Run const& styled) {
            if (!pending.empty()) {
                Run plain;
                plain.Text(toHstring(pending));
                paragraph.Inlines().Append(plain);
                pending.clear();
            }
            if (styled != nullptr) {
                paragraph.Inlines().Append(styled);
            }
        };

        for (size_t i = 0; i < line.size();) {
            const auto marked = [&](std::string_view mark) -> size_t {
                if (!std::string_view{line}.substr(i).starts_with(mark)) {
                    return std::string::npos;
                }
                return line.find(mark, i + mark.size());
            };

            size_t close = std::string::npos;
            std::string_view mark;
            for (std::string_view candidate : {"**", "`", "*", "_"}) {
                close = marked(candidate);
                if (close != std::string::npos) {
                    mark = candidate;
                    break;
                }
            }

            if (close == std::string::npos) {
                pending += line[i];
                ++i;
                continue;
            }

            Run styled;
            styled.Text(toHstring(line.substr(i + mark.size(), close - i - mark.size())));
            if (mark == "**") {
                styled.FontWeight(Microsoft::UI::Text::FontWeights::Bold());
            } else if (mark == "`") {
                styled.FontFamily(Media::FontFamily{L"Cascadia Mono, Consolas"});
            } else {
                styled.FontStyle(winrt::Windows::UI::Text::FontStyle::Italic);
            }
            flush(styled);
            i = close + mark.size();
        }
        flush(nullptr);
    }

    [[nodiscard]] static std::string_view trimLeft(std::string_view text) {
        while (!text.empty() && (text.front() == ' ' || text.front() == '\t' ||
                                 text.front() == '\r')) {
            text.remove_prefix(1);
        }
        while (!text.empty() && text.back() == '\r') {
            text.remove_suffix(1);
        }
        return text;
    }

    /// UI thread only, so it needs no synchronization: every path that touches
    /// it arrives through the dispatcher.
    bool transcriptRefreshPending_ = false;
};

}  // namespace

namespace winrt::TurboFieldfare::implementation {

/// The control styles come from App.xaml, which InitializeComponent loads.
///
/// This used to be a plain ApplicationT<App> with no markup anywhere. That is
/// what made the window die in its first layout pass: nothing populated
/// Application.Resources, so no control had a template.
struct App : AppT<App> {
    App() { InitializeComponent(); }

    void OnLaunched(LaunchActivatedEventArgs const&) try {
        trace("launched");

        // OnLaunched's own catch covers only OnLaunched. Anything thrown later
        // on the UI thread - during layout, or from a dispatched callback -
        // escapes into the message loop, where an unhandled HRESULT is a silent
        // process kill with nothing but a stowed-exception code to show for it.
        UnhandledException([](auto&&, UnhandledExceptionEventArgs const& args) {
            trace(std::format("unhandled 0x{:08X}: {}",
                              static_cast<uint32_t>(args.Exception()),
                              winrt::to_string(args.Message())));
        });

        view_ = std::make_unique<MainView>(model_);
        window_ = view_->build();
        trace("tree built");
        window_.Activate();
        trace("activated");

        // Every callback arrives on the pipe reader thread and has to hop to
        // the UI thread before touching a control. Capturing the dispatcher
        // once is what makes that a one-line wrap at each site.
        const auto ui = view_->dispatcher();
        const auto onUi = [ui](std::function<void()> work) {
            const bool queued = ui.TryEnqueue([work = std::move(work)] { work(); });
            trace(queued ? "enqueued" : "TryEnqueue REFUSED");
        };

        tf::gui::AppModel::Callbacks callbacks;
        callbacks.onReady = [this, onUi](const tf::svc::ReadyInfo& info) {
            onUi([this, info] { view_->onReady(info); });
        };
        callbacks.onToken = [this, onUi](std::string piece) {
            onUi([this, piece = std::move(piece)]() mutable {
                view_->onToken(std::move(piece));
            });
        };
        callbacks.onDone = [this, onUi](const tf::svc::DoneInfo& info) {
            onUi([this, info] { view_->onDone(info); });
        };
        callbacks.onError = [this, onUi](std::string message) {
            onUi([this, message = std::move(message)]() mutable {
                view_->onError(std::move(message));
            });
        };
        callbacks.onDisconnected = [this, onUi] {
            onUi([this] { view_->onDisconnected(); });
        };
        model_.setCallbacks(std::move(callbacks));

        window_.Closed([this](auto&&, auto&&) { model_.disconnect(); });

        // Connecting blocks until the service has loaded 13 GiB, so it runs off
        // the UI thread - otherwise the window would not paint until the model
        // was ready, which looks exactly like a hang.
        std::thread{[this, onUi] {
            try {
                trace("connecting");
                const auto connected =
                        model_.connectToService(serviceBesideMe(), defaultModelDir());
                trace(connected ? "connected" : "connect failed");
                if (!connected) {
                    const std::string message = connected.error().message();
                    onUi([this, message] { view_->onError(message); });
                }
            } catch (const std::exception& error) {
                // A detached thread that throws takes the process with it, and
                // does so without running the handler above, which sees only
                // the UI thread.
                trace(std::format("connect thread threw: {}", error.what()));
            } catch (...) {
                trace("connect thread threw a non-std exception");
            }
        }}.detach();
        trace("launch complete");
    } catch (winrt::hresult_error const& error) {
        // An escaping HRESULT would otherwise terminate the process with no
        // window and no message.
        trace(std::format("hresult {:#x}: {}", static_cast<uint32_t>(error.code()),
                          fromHstring(error.message())));
        throw;
    } catch (std::exception const& error) {
        trace(std::string{"exception: "} + error.what());
        throw;
    }

    tf::gui::AppModel model_;
    std::unique_ptr<MainView> view_;
    Window window_{nullptr};
};

}  // namespace winrt::TurboFieldfare::implementation

namespace winrt::TurboFieldfare::factory_implementation {
struct App : AppT<App, implementation::App> {};
}  // namespace winrt::TurboFieldfare::factory_implementation

int APIENTRY wWinMain(HINSTANCE, HINSTANCE, PWSTR, int) {
    // Unpackaged, so the Windows App SDK runtime has to be loaded by hand.
    trace("--- start ---");
    // Match the runtime floor encoded by the 1.7 SDK package (7000.522.1444).
    const PACKAGE_VERSION minimum{ 0x1B58020A05A40000u };
    // Do not use MddBootstrapInitialize here: its default no-match behavior
    // displays a generic Windows dialog before control returns to this app.
    // Initialize2 with None lets us report the actual HRESULT and recovery
    // instructions ourselves.
    const HRESULT bootstrap = ::MddBootstrapInitialize2(
            0x00010007,
            L"",
            minimum,
            MddBootstrapInitializeOptions_None);
    if (FAILED(bootstrap)) {
        trace(std::format("bootstrap failed: HRESULT 0x{:08X}",
                          static_cast<unsigned long>(bootstrap)));
        wchar_t detail[256] = {};
        std::swprintf(detail, std::size(detail),
                      L"The Windows App Runtime 1.7 is not available (HRESULT 0x%08lX).\n\n"
                      L"Install the x64 Windows App Runtime 1.7, then try again.",
                      static_cast<unsigned long>(bootstrap));
        ::MessageBoxW(nullptr,
                      detail, L"TurboFieldfare", MB_OK | MB_ICONERROR);
        return 1;
    }

    trace("bootstrapped");
    init_apartment(apartment_type::single_threaded);
    Application::Start(
            [](auto&&) { make<winrt::TurboFieldfare::implementation::App>(); });
    trace("message loop ended");

    ::MddBootstrapShutdown();
    return 0;
}
