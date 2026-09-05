//! Pure view state and action enum for the interactive mode.
//!
//! [`ViewState`] is a **data snapshot** of everything the presentation layer
//! needs to render one frame. It deliberately holds no live
//! [`crate::core::agent_session`] handle and performs no I/O. The runtime loop
//! (a later phase) owns one of these, mutates it in response to agent events,
//! and feeds [`ViewAction`]s emitted by user input back into its own logic.
//!
//! Field names mirror the reference `interactive-mode.ts` child containers and
//! the `FooterDataProvider`/`AgentSession.state` shapes so the port is faithful.

use std::collections::BTreeMap;
use std::sync::Arc;

use pi_ai::auth::AuthType;
use pi_ai::{AssistantMessage, ModelThinkingLevel};
use pi_ext::sanitize::SanitizedSlot;
use pi_tui::component::EventResult;

use crate::core::settings::ThemeMode;

use super::messages::MessageView;
use super::theme::ResolvedTheme;

/// Top-level interactive view-model state (the data behind one paint).
///
/// Named `ViewState` at the public boundary (re-exported from
/// [`super`]); internally aliased `InteractiveMode` would collide with the
/// reference class name, so this struct is `ViewState`.
#[expect(
    clippy::struct_excessive_bools,
    reason = "view-model mirrors the reference UI state; splitting would obscure the 1:1 mapping"
)]
pub struct ViewState {
    /// Resolved theme used for this frame.
    pub theme: Arc<ResolvedTheme>,
    /// Terminal width the view was last composed for.
    pub width: u16,
    /// Terminal height the view was last composed for.
    pub height: u16,
    /// Quiet mode suppresses the logo/key-hint header.
    pub quiet: bool,
    /// Header content (logo, version, hints).
    pub header: HeaderData,
    /// Loaded skills/prompts/themes/context summary lines.
    pub resources: Vec<LoadedResource>,
    /// Settled + streaming chat messages in display order.
    pub messages: Vec<MessageView>,
    /// Steering/follow-up queue and pending bash.
    pub pending: PendingQueue,
    /// Active status indicator (working/retry/compaction/branch).
    pub status: Option<SessionStatus>,
    /// Extension widgets rendered above the editor.
    pub widgets_above: WidgetStack,
    /// Editor line(s) — the active input area view-model.
    pub editor: EditorView,
    /// Extension widgets rendered below the editor.
    pub widgets_below: WidgetStack,
    /// Footer status-bar data.
    pub footer: FooterData,
    /// Extension shortcut rows shown in the hotkeys overlay.
    pub extension_shortcuts: Vec<ShortcutHint>,
    /// Complete sanitized slot backing the current extension overlay.
    pub extension_overlay_slot: Option<SanitizedSlot>,
    /// Optional overlay currently shown (shortcut help, selectors, etc.).
    pub overlay: Option<Overlay>,
    /// Which area currently holds focus.
    pub focus: FocusArea,
    /// Whether the view is mid-stream (mutates only the active tail).
    pub streaming: bool,
    /// Startup diagnostics (errors/warnings from resource load).
    pub diagnostics: StartupDiagnostics,
    /// Active first-run wizard step (`None` when the wizard is not running).
    pub first_run_step: Option<usize>,
    /// Highlighted option index within the current first-run step.
    pub first_run_selected: usize,
    /// Family chosen on the first-run family step (label form).
    pub first_run_family: Option<String>,
    /// Mode chosen on the first-run mode step.
    pub first_run_mode: Option<ThemeMode>,
    /// Extension-set working-indicator message override. `None` means the
    /// default text (`Working…`) is used while streaming.
    pub working_message: Option<String>,
    /// Whether the working indicator is shown at agent start. Persisted across
    /// turns so `ui.setWorkingVisible(false)` is honored at `AgentStart`.
    pub working_visible: bool,
    /// Whether the terminal advertises OSC 8 hyperlink support. Read by
    /// view composition via [`super::theme::with_hyperlinks`] so every
    /// markdown surface honors the same capability.
    pub hyperlinks: bool,
    /// Override spinner indicator frames. `None` uses the default 10-frame
    /// braille animation; `Some` with a single frame renders a static
    /// indicator (TUI-T11 reduced-motion mechanism per TUI-G1 decision).
    pub indicator_frames: Option<Vec<String>>,
    /// Active auth/login progress (drives the Login overlay content).
    pub auth_progress: Option<AuthProgress>,
    /// True while an extension `HostUi` Input/Editor/Select/Confirm dialog is open.
    pub extension_dialog: bool,
}

/// Header view-model data.
#[derive(Clone, Debug, Default)]
pub struct HeaderData {
    /// Product display name (e.g. `pi`).
    pub app_name: String,
    /// Version string.
    pub version: String,
    /// Whether the hint block is expanded.
    pub expanded: bool,
    /// One-line onboarding hint shown when collapsed.
    pub onboarding: Option<String>,
}

/// One loaded resource line shown above the chat.
#[derive(Clone, Debug)]
pub struct LoadedResource {
    /// Kind label (`skill`, `prompt`, `theme`, `context`, …).
    pub kind: String,
    /// Display name.
    pub label: String,
}

/// Editor area view-model (single-line + pending input).
#[derive(Clone, Debug, Default)]
pub struct EditorView {
    /// Current raw text (pre-wrap, single logical line for the snapshot).
    pub text: String,
    /// Visible cursor column.
    pub cursor: usize,
    /// Placeholder when empty.
    pub placeholder: String,
    /// Border color slot tracks thinking level or bash mode.
    pub border: EditorBorder,
    /// Whether a large-paste marker is present.
    pub paste_marker: Option<String>,
}

/// Editor border style selector.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum EditorBorder {
    /// Default muted border.
    #[default]
    Muted,
    /// Bash-mode border (`bashMode` color).
    Bash,
    /// Thinking-level border.
    Thinking(ModelThinkingLevel),
}

/// Billing source displayed by the footer.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BillingMode {
    /// Usage is metered and shown only when it has a non-zero cost.
    #[default]
    Metered,
    /// Usage is covered by an OAuth subscription.
    Subscription,
}

/// Footer feature flags (billing, compaction, model capability, experimental).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FooterFlags {
    /// Billing source for the active model.
    pub billing: BillingMode,
    /// Auto-compact enabled.
    pub auto_compact: bool,
    /// Whether the active model supports reasoning.
    pub reasoning: bool,
    /// Experimental features enabled (shows `xp`).
    pub experimental: bool,
}

impl Default for FooterFlags {
    fn default() -> Self {
        Self {
            billing: BillingMode::default(),
            auto_compact: true,
            reasoning: false,
            experimental: false,
        }
    }
}

/// Footer data ported from `FooterComponent` + `FooterDataProvider`.
#[derive(Clone, Debug)]
pub struct FooterData {
    /// Working directory (raw, before `~` collapse).
    pub cwd: String,
    /// Home directory for `~` collapsing; empty disables it.
    pub home: String,
    /// Git branch when available.
    pub git_branch: Option<String>,
    /// Session display name when set.
    pub session_name: Option<String>,
    /// Cumulative input tokens across the session.
    pub total_input: u64,
    /// Cumulative output tokens across the session.
    pub total_output: u64,
    /// Cumulative cache-read tokens.
    pub total_cache_read: u64,
    /// Cumulative cache-write tokens.
    pub total_cache_write: u64,
    /// Latest cache-hit rate percent, when known.
    pub cache_hit_rate: Option<f64>,
    /// Cumulative cost in USD.
    pub total_cost: f64,
    /// Context-window size in tokens.
    pub context_window: u64,
    /// Context usage percent (`None` ⇒ unknown/`?`).
    pub context_percent: Option<f64>,
    /// Active model id (or `"no-model"`).
    pub model_id: String,
    /// Provider id when multi-provider.
    pub provider: Option<String>,
    /// Available provider count (provider prefix shown when > 1).
    pub provider_count: usize,
    /// Active thinking level.
    pub thinking_level: ModelThinkingLevel,
    /// Feature flags (billing / auto-compact / reasoning / experimental).
    pub flags: FooterFlags,
    /// Sorted extension status texts.
    pub extension_statuses: BTreeMap<String, String>,
}

impl Default for FooterData {
    fn default() -> Self {
        Self {
            cwd: String::new(),
            home: String::new(),
            git_branch: None,
            session_name: None,
            total_input: 0,
            total_output: 0,
            total_cache_read: 0,
            total_cache_write: 0,
            cache_hit_rate: None,
            total_cost: 0.0,
            context_window: 0,
            context_percent: None,
            model_id: String::new(),
            provider: None,
            provider_count: 0,
            thinking_level: ModelThinkingLevel::Off,
            flags: FooterFlags::default(),
            extension_statuses: BTreeMap::new(),
        }
    }
}

/// Pending message queue (steering/follow-up + pending bash).
#[derive(Clone, Debug, Default)]
pub struct PendingQueue {
    /// Steering messages queued for the current turn.
    pub steering: Vec<PendingMessage>,
    /// Follow-up messages queued for the next turn.
    pub follow_up: Vec<PendingMessage>,
    /// Queue mode for follow-ups.
    pub follow_up_mode: QueueMode,
}

/// Queue delivery mode.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum QueueMode {
    /// Deliver all queued messages.
    #[default]
    All,
    /// Deliver one at a time.
    OneAtATime,
}

/// One pending (queued) message.
#[derive(Clone, Debug)]
pub struct PendingMessage {
    /// Queue kind.
    pub kind: PendingKind,
    /// Message text.
    pub text: String,
}

/// Pending message kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PendingKind {
    /// Steering injected mid-turn.
    Steering,
    /// Follow-up queued for next turn.
    FollowUp,
}

/// Status indicator shown above the editor while the session is busy.
#[derive(Clone, Debug)]
pub struct SessionStatus {
    /// Indicator kind.
    pub kind: StatusKind,
    /// Spinner frame index into the braille set.
    pub frame: usize,
    /// Whole seconds since this status phase began; 0 suppresses the counter.
    pub elapsed_secs: u64,
    /// Status message text.
    pub message: String,
}

/// Status indicator kind (ports `StatusIndicatorKind`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StatusKind {
    /// Generic working spinner.
    Working,
    /// Retry countdown.
    Retry,
    /// Compaction in progress.
    Compaction,
    /// Branch summarization in progress.
    BranchSummary,
}

/// Extension widget slot (above or below the editor).
#[derive(Clone, Debug)]
pub struct WidgetSlot {
    /// Complete sanitized slot, including structured runs and link metadata.
    pub slot: SanitizedSlot,
    /// Whether this slot currently owns input focus.
    pub focused: bool,
}

/// Ordered stack of widget slots.
pub type WidgetStack = Vec<WidgetSlot>;

/// Overlay shown on top of the inline viewport.
#[derive(Clone, Debug)]
pub struct Overlay {
    /// Overlay kind.
    pub kind: OverlayKind,
    /// Rendered lines (already styled, width-truncated by caller).
    pub lines: Vec<String>,
    /// Resolved overlay height.
    pub height: u16,
}

/// Overlay kind discriminator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OverlayKind {
    /// Shortcut/help keybinding table.
    ShortcutHelp,
    /// Release notes / changelog.
    Changelog,
    /// First-time setup wizard.
    FirstTimeSetup,
    /// Login dialog.
    Login,
    /// Extension custom overlay.
    Extension,
}

/// Focusable area of the inline viewport.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FocusArea {
    /// Editor (default).
    #[default]
    Editor,
    /// An active selector replacing the editor.
    Selector,
    /// A focused extension widget.
    Widget,
    /// A focused overlay.
    Overlay,
}

/// Startup diagnostics collected during resource load.
#[derive(Clone, Debug, Default)]
pub struct StartupDiagnostics {
    /// Diagnostic entries.
    pub entries: Vec<StartupDiagnostic>,
}

/// One startup diagnostic.
#[derive(Clone, Debug)]
pub struct StartupDiagnostic {
    /// Severity.
    pub severity: DiagnosticSeverity,
    /// Source label (e.g. `skills`, `themes`).
    pub source: String,
    /// Message.
    pub message: String,
}

/// Diagnostic severity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DiagnosticSeverity {
    /// Warning (rendered with `warning` color).
    Warning,
    /// Error (rendered with `error` color).
    Error,
}

// ---------------------------------------------------------------------------
// Selector view-models
// ---------------------------------------------------------------------------

/// Model selector entry.
#[derive(Clone, Debug)]
pub struct ModelSelectorEntry {
    /// `provider/model` value.
    pub value: String,
    /// Display label.
    pub label: String,
    /// Optional description.
    pub description: Option<String>,
}

/// Session picker entry.
#[derive(Clone, Debug)]
pub struct SessionPickerEntry {
    /// Session id.
    pub value: String,
    /// Display label.
    pub label: String,
    /// Optional timestamp/summary.
    pub description: Option<String>,
}

/// Tree (branch) entry.
#[derive(Clone, Debug)]
pub struct TreeEntry {
    /// Entry id.
    pub value: String,
    /// Indented display label.
    pub label: String,
    /// Depth for indentation.
    pub depth: usize,
}

/// Settings/config row (ports `SettingItem`).
#[derive(Clone, Debug)]
pub struct SettingsRow {
    /// Row id.
    pub id: String,
    /// Label.
    pub label: String,
    /// Description when selected.
    pub description: Option<String>,
    /// Current value text.
    pub current_value: String,
    /// Cycle values.
    pub values: Option<Vec<String>>,
}

/// Config selector reuses [`SettingsRow`].
pub type ConfigSelectorEntry = SettingsRow;

/// Auth/login selector entry.
#[derive(Clone, Debug)]
pub struct AuthSelectorEntry {
    /// Provider value.
    pub value: String,
    /// Display label.
    pub label: String,
    /// Optional description.
    pub description: Option<String>,
}

/// One login option for a provider + auth-type combination.
///
/// Ports the `AuthSelectorProvider` shape from the reference
/// `getLoginProviderOptions`: each entry pairs a provider id/name with the
/// auth mechanism it supports, plus whether interactive login is available
/// (ambient-only API-key providers have `has_login = false`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoginProviderOption {
    /// Provider id.
    pub id: String,
    /// Display name (falls back to the id).
    pub name: String,
    /// Auth mechanism.
    pub auth_type: AuthType,
    /// Whether interactive login is available (`false` for ambient-only keys).
    pub has_login: bool,
    /// OAuth login-label override (e.g. "Sign in with Kimi Code").
    pub login_label: Option<String>,
}

/// One stored credential offered by the `/logout` selector.
///
/// Ports the `AuthSelectorProvider` shape upstream builds in
/// `getLogoutProviderOptions`: the runtime keeps the fetched list so the
/// confirm handler can format the removal notice by credential kind.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LogoutOption {
    /// Provider id the credential belongs to.
    pub id: String,
    /// Provider display name (falls back to the id).
    pub name: String,
    /// Whether the credential is an OAuth credential (vs a stored API key).
    pub is_oauth: bool,
}

/// OAuth progress state.
#[derive(Clone, Debug)]
pub struct AuthProgress {
    /// Current stage.
    pub stage: OAuthStage,
    /// Provider being authenticated.
    pub provider: String,
    /// Auth URL or device code, when applicable.
    pub detail: Option<String>,
}

/// OAuth stage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OAuthStage {
    /// Awaiting browser callback.
    BrowserCallback,
    /// Device-code flow awaiting user input.
    DeviceCode,
    /// Manual API-key entry.
    ManualKey,
    /// Exchanging token.
    Exchanging,
    /// Completed successfully.
    Done,
    /// Failed.
    Failed,
}

/// Compaction progress indicator data.
#[derive(Clone, Debug)]
pub struct CompactionProgress {
    /// Trigger reason.
    pub reason: CompactionReason,
}

/// Compaction trigger reason.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompactionReason {
    /// User-initiated `/compact`.
    Manual,
    /// Threshold-based auto-compact.
    Threshold,
    /// Overflow auto-compact.
    Overflow,
}

/// Retry progress with countdown.
#[derive(Clone, Debug)]
pub struct RetryProgress {
    /// Current attempt (1-based).
    pub attempt: u32,
    /// Max attempts.
    pub max_attempts: u32,
    /// Seconds remaining.
    pub seconds: u32,
}

/// Bash execution progress.
#[derive(Clone, Debug)]
pub struct BashProgress {
    /// Command being run.
    pub command: String,
    /// Captured output so far.
    pub output: String,
    /// Whether collapsed (preview) or expanded.
    pub expanded: bool,
    /// Exit code when finished.
    pub exit_code: Option<i32>,
    /// Whether cancelled.
    pub cancelled: bool,
}

/// Shortcut hint row.
#[derive(Clone, Debug)]
pub struct ShortcutHint {
    /// Key text.
    pub key: String,
    /// Action description.
    pub action: String,
}

impl ViewState {
    /// Build an empty default state using the dark theme.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            theme: super::theme::dark(),
            width: 80,
            height: 24,
            quiet: false,
            header: HeaderData {
                app_name: "pi".to_owned(),
                version: super::super::super::VERSION.to_owned(),
                expanded: false,
                onboarding: None,
            },
            resources: Vec::new(),
            messages: Vec::new(),
            pending: PendingQueue::default(),
            status: None,
            widgets_above: Vec::new(),
            editor: EditorView {
                text: String::new(),
                cursor: 0,
                placeholder: "Type a message…".to_owned(),
                border: EditorBorder::Muted,
                paste_marker: None,
            },
            widgets_below: Vec::new(),
            footer: FooterData::default(),
            extension_shortcuts: Vec::new(),
            extension_overlay_slot: None,
            overlay: None,
            focus: FocusArea::Editor,
            streaming: false,
            diagnostics: StartupDiagnostics::default(),
            first_run_step: None,
            first_run_selected: 0,
            first_run_family: None,
            first_run_mode: None,
            working_message: None,
            working_visible: true,
            hyperlinks: false,
            indicator_frames: None,
            auth_progress: None,
            extension_dialog: false,
        }
    }

    /// Build a streaming state with a single in-flight assistant tail.
    #[must_use]
    pub fn streaming(partial: AssistantMessage) -> Self {
        let mut s = Self::empty();
        s.streaming = true;
        s.status = Some(SessionStatus {
            kind: StatusKind::Working,
            frame: 0,
            elapsed_secs: 0,
            message: "Working…".to_owned(),
        });
        s.messages.push(MessageView::streaming_assistant(partial));
        s
    }

    /// Apply a resize, recording the new size and invalidating width caches.
    pub fn resize(&mut self, width: u16, height: u16) {
        self.width = width;
        self.height = height;
    }

    /// Whether the editor input is currently empty.
    #[must_use]
    pub fn editor_empty(&self) -> bool {
        self.editor.text.is_empty()
    }
}

impl Default for ViewState {
    fn default() -> Self {
        Self::empty()
    }
}

// ---------------------------------------------------------------------------
// Pure action enum
// ---------------------------------------------------------------------------

/// Pure, allocation-light actions emitted by the presentation layer for the
/// runtime event loop to consume.
///
/// The view-model translates raw [`pi_tui::component::UiEvent`]s and selector
/// confirmations into these semantic actions. The runtime owns the session and
/// decides how (or whether) to apply each one. No action performs I/O.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ViewAction {
    /// Do nothing (event ignored).
    None,
    /// The frame needs a repaint.
    Render,
    /// Event consumed, no repaint required.
    Consumed,
    /// Submit the current editor text as a prompt.
    Submit {
        /// Submitted text (paste markers already expanded by the runtime).
        text: String,
    },
    /// Submit a bash command (`!`/`!!` prefix).
    SubmitBash {
        /// Command text.
        command: String,
        /// Whether to exclude from context (`!!`).
        exclude_from_context: bool,
    },
    /// Abort the in-flight stream / bash run.
    Interrupt,
    /// Clear the editor.
    ClearEditor,
    /// Unconditional application exit (double Ctrl+C / `/quit`).
    Exit,
    /// Named `app.exit` (default Ctrl+D on empty editor). Distinct from
    /// [`Exit`] so extension Input dialogs can suppress only this chord
    /// without blocking the unconditional double-Ctrl+C shutdown.
    AppExit,
    /// Suspend the process (Ctrl+Z).
    Suspend,
    /// Cycle the thinking level forward/backward.
    CycleThinking {
        /// Direction.
        forward: bool,
    },
    /// Toggle thinking visibility.
    ToggleThinking,
    /// Cycle models.
    CycleModel {
        /// Direction.
        forward: bool,
    },
    /// Expand/collapse tool output.
    ToggleToolExpand,
    /// Open the model selector.
    OpenModelSelector,
    /// Open the settings menu.
    OpenSettings,
    /// Open a nested settings submenu.
    OpenSettingsSubmenu {
        /// Submenu id.
        id: String,
    },
    /// Open the session picker (`/resume`).
    OpenSessionPicker,
    /// Open the tree selector (`/tree`).
    OpenTreeSelector,
    /// Open the user-message fork selector (`/fork`).
    OpenForkSelector,
    /// Open the trust selector (`/trust`).
    OpenTrustSelector,
    /// Open the auth/login selector (`/login`).
    OpenLogin {
        /// Provider hint, when provided.
        provider: Option<String>,
    },
    /// Log out of a provider (`/logout`).
    Logout,
    /// Open the scoped-models selector.
    OpenScopedModels,
    /// Open the config selector.
    OpenConfigSelector,
    /// Toggle the shortcut/help overlay.
    ToggleShortcutHelp,
    /// Show release notes / changelog.
    ShowChangelog,
    /// Paste the given text into the editor.
    Paste {
        /// Pasted text.
        text: String,
    },
    /// Queue a follow-up message (Alt+Enter while streaming).
    QueueFollowUp {
        /// Follow-up text.
        text: String,
    },
    /// Restore the last queued follow-up to the editor (Alt+Up).
    DequeueFollowUp,
    /// Copy the last assistant message to the clipboard.
    CopyLastAssistant,
    /// Open the external editor (Ctrl+G).
    ExternalEditor,
    /// Reload keybindings/extensions/skills/prompts/themes/context.
    Reload,
    /// A slash command was submitted.
    SlashCommand {
        /// Command name without the leading `/`.
        name: String,
        /// Raw argument string.
        args: String,
    },
    /// Selector confirmed a value.
    SelectConfirmed {
        /// Selector kind.
        selector: SelectorKind,
        /// Selected value.
        value: String,
    },
    /// Selector cancelled.
    SelectCancelled,
    /// Focus moved to a new area.
    FocusChanged {
        /// New focus.
        area: FocusArea,
    },
    /// Overlay requested.
    ShowOverlay {
        /// Overlay kind.
        kind: OverlayKind,
    },
    /// Overlay dismissed.
    DismissOverlay,
    /// New session.
    NewSession,
    /// Fork the session.
    Fork,
    /// Clone the session.
    Clone,
    /// Manual compaction.
    Compact {
        /// Optional custom instructions.
        instructions: Option<String>,
    },
    /// Resize recorded.
    Resize {
        /// New width.
        width: u16,
        /// New height.
        height: u16,
    },
}

/// Selector kind discriminator for [`ViewAction::SelectConfirmed`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SelectorKind {
    /// Model selector.
    Model,
    /// Session picker.
    Session,
    /// Tree selector.
    Tree,
    /// User-message fork selector.
    Fork,
    /// Trust selector.
    Trust,
    /// Auth/login selector.
    Auth,
    /// Auth-type selector (oauth / api-key labels).
    AuthType,
    /// Settings menu.
    Settings,
    /// Config selector.
    Config,
    /// Scoped-models selector.
    ScopedModels,
    /// Theme selector (`/theme`).
    Theme,
    /// `/import` replace-session confirmation (Yes/No).
    ImportConfirm,
    /// `/import` continue-in-fallback-cwd confirmation (Yes/No).
    ImportCwdConfirm,
    /// `/logout` stored-credential selector.
    Logout,
}

impl From<EventResult> for ViewAction {
    fn from(result: EventResult) -> Self {
        match result {
            EventResult::Ignored => Self::None,
            EventResult::Consumed => Self::Consumed,
            EventResult::Render => Self::Render,
        }
    }
}
