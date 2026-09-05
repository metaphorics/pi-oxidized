//! Product keyboard/focus gauntlet (TUI-V2) driven through
//! `pi_tui::testkit::RecordingSession`.
//!
//! Every scenario completes a keyboard-only flow and proves the current
//! dispatch semantics from canonical transcript content — ctrl+d resolution
//! order per selector context, focus restore after overlays and cancelled
//! dialogs, streaming interrupt, and rebind reflection in rendered hints.
//! Zero dispatch-semantics changes land here: the measured key-routing
//! divergence while a focusable extension overlay stacks under a pending
//! dialog is recorded in the canonical transcripts (digest-stable frames),
//! never asserted as a contract — semantics changes belong to TUI-G7.
//!
//! Checkpoints use content predicates then quiescence — never timer-only
//! waits. Artifacts land under
//! `target/verification/tui-transcripts/<row>/keyboard-*/run-{1,2,3}/` with a
//! `verdict.json` per row.

use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use pi_tui::testkit::driver::{
    Geometry as DriverGeometry, LaunchSpec, SettlePolicy, TerminalDriver,
};
use pi_tui::testkit::repeat::{RepeatError, run_k};
use pi_tui::testkit::transcript::{
    CapabilityProfile, ClaimClass, DriverKind, Geometry, NormalizationContext, OutputCanon, RowId,
    RowTier, RunnerRow, Scenario, TimingEnvelope, TranscriptArtifact, TranscriptMode,
    TranscriptRecorder, TranscriptSpec,
};
use pi_tui::testkit::{RecordingError, RecordingSession};
use tempfile::TempDir;

#[cfg(windows)]
use pi_tui::testkit::conpty::ConPtyDriver;
#[cfg(unix)]
use pi_tui::testkit::posix::PosixPtyDriver;

const INITIAL_COLS: u16 = 80;
const INITIAL_ROWS: u16 = 24;
const K: usize = 3;
const FINAL_MARKER: &str = "PI_VERIFICATION_FINAL_TUI";
const VERIFICATION_PROVIDER: &str = "verification";
const VERIFICATION_MODEL: &str = "model";
const VERIFICATION_PROFILE_FLAG: &str = "verification-profile";
const VERIFICATION_PROFILE: &str = "tui-transcript-profile";
const READY_MARKERS: &[&[u8]] = &[
    b"type a message",
    b"type a message to begin",
    b"No messages",
];
const KEY_ENTER: &[u8] = b"\r";
const KEY_ESCAPE: &[u8] = b"\x1b";
const KEY_DOWN: &[u8] = b"\x1b[B";
const KEY_LEFT: &[u8] = b"\x1b[D";
const KEY_CTRL_D: &[u8] = b"\x04";
const BACKSPACE: &[u8] = b"\x7f";
const STACKED_DIALOG_TITLE: &str = "Verification stacked select";
const STACKED_OVERLAY_LINE: &str = "Verification overlay-stack state=pending";

#[derive(Debug)]
enum CorpusError {
    Prerequisite(String),
    Driver(String),
    Transcript(String),
    Io(String),
    Assert(String),
}

impl std::fmt::Display for CorpusError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Prerequisite(message)
            | Self::Driver(message)
            | Self::Transcript(message)
            | Self::Io(message)
            | Self::Assert(message) => write!(formatter, "{message}"),
        }
    }
}

impl From<RecordingError> for CorpusError {
    fn from(error: RecordingError) -> Self {
        match error {
            RecordingError::Driver(error) => Self::Driver(error.to_string()),
            RecordingError::Transcript(error) => Self::Transcript(error.to_string()),
            RecordingError::FinishBeforeClose => Self::Driver(error.to_string()),
        }
    }
}

impl From<pi_tui::testkit::driver::DriverError> for CorpusError {
    fn from(error: pi_tui::testkit::driver::DriverError) -> Self {
        Self::Driver(error.to_string())
    }
}

impl From<std::io::Error> for CorpusError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error.to_string())
    }
}

struct Sandbox {
    _root: TempDir,
    home_dir: PathBuf,
    agent_dir: PathBuf,
    session_dir: PathBuf,
    work_dir: PathBuf,
}

/// Per-scenario launch knobs beyond the shared verification environment.
struct LaunchOpts {
    include_extension: bool,
    wizard: bool,
    /// Deterministic streaming cadence for the interrupt scenarios.
    chunk_count: u32,
    chunk_delay_ms: u32,
    /// Seed `$agent/keybindings.json` with rebinds before boot.
    rebind_keybindings: bool,
}

impl Default for LaunchOpts {
    fn default() -> Self {
        Self {
            include_extension: true,
            wizard: false,
            chunk_count: 3,
            chunk_delay_ms: 0,
            rebind_keybindings: false,
        }
    }
}

struct LaunchEnv {
    argv: Vec<String>,
    env: BTreeMap<String, String>,
    cwd: PathBuf,
    context: NormalizationContext,
}

#[cfg(unix)]
type HostSession = <PosixPtyDriver as TerminalDriver>::Session;
#[cfg(windows)]
type HostSession = <ConPtyDriver as TerminalDriver>::Session;

struct ProductRun {
    recording: RecordingSession<HostSession>,
    context: NormalizationContext,
    policy: SettlePolicy,
    raw_acc: Vec<u8>,
    wall_started: Instant,
    settle_windows_ms: Vec<u64>,
    saw_snapshot: bool,
}

fn workspace_root() -> Result<PathBuf, CorpusError> {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest
        .parent()
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .ok_or_else(|| {
            CorpusError::Prerequisite(format!(
                "workspace root not found above {}",
                manifest.display()
            ))
        })
}

fn extension_path() -> Result<PathBuf, CorpusError> {
    Ok(workspace_root()?.join("scripts/verification/extension.ts"))
}

fn extension_host_path() -> Result<PathBuf, CorpusError> {
    let root = workspace_root()?;
    let mut path = root
        .join("packages")
        .join("extension-host")
        .join("dist")
        .join("pi-extension-host");
    if cfg!(windows) {
        path.set_extension("exe");
    }
    Ok(path)
}

fn pi_binary() -> Result<PathBuf, CorpusError> {
    let compiled = PathBuf::from(env!("CARGO_BIN_EXE_pi"));
    if compiled.is_file() {
        return Ok(compiled);
    }
    let fallback = target_root().join("debug").join("pi");
    if fallback.is_file() {
        return Ok(fallback);
    }
    Err(CorpusError::Prerequisite(format!(
        "product prerequisite missing: CARGO_BIN_EXE_pi points at missing binary {} (fallback {})",
        compiled.display(),
        fallback.display()
    )))
}

fn require_prerequisites() -> Result<(), CorpusError> {
    let binary = pi_binary()?;
    let host = extension_host_path()?;
    let extension = extension_path()?;
    if !host.is_file() {
        return Err(CorpusError::Prerequisite(format!(
            "product prerequisite missing: extension host binary at {} (build with `bun run --cwd packages/extension-host build`; Unix expects pi-extension-host, Windows pi-extension-host.exe)",
            host.display()
        )));
    }
    if !extension.is_file() {
        return Err(CorpusError::Prerequisite(format!(
            "product prerequisite missing: verification extension at {}",
            extension.display()
        )));
    }
    let _ = DriverGeometry::new(1, 1).map_err(|error| {
        CorpusError::Prerequisite(format!("geometry prerequisite failed: {error}"))
    })?;
    let _ = binary;
    Ok(())
}

fn create_sandbox() -> Result<Sandbox, CorpusError> {
    let root = TempDir::new()?;
    let home_dir = root.path().join("home");
    let agent_dir = root.path().join("agent");
    let session_dir = root.path().join("sessions");
    let work_dir = root.path().join("work");
    for directory in [&home_dir, &agent_dir, &session_dir, &work_dir] {
        fs::create_dir_all(directory)?;
    }
    Ok(Sandbox {
        _root: root,
        home_dir,
        agent_dir,
        session_dir,
        work_dir,
    })
}

fn common_argv(include_extension: bool) -> Result<Vec<String>, CorpusError> {
    let mut argv = vec![pi_binary()?.to_string_lossy().into_owned()];
    if include_extension {
        argv.extend([
            "--provider".to_owned(),
            VERIFICATION_PROVIDER.to_owned(),
            "--model".to_owned(),
            VERIFICATION_MODEL.to_owned(),
            "--api-key".to_owned(),
            "verification-key".to_owned(),
            "--extension".to_owned(),
            extension_path()?.to_string_lossy().into_owned(),
            format!("--{VERIFICATION_PROFILE_FLAG}"),
            VERIFICATION_PROFILE.to_owned(),
        ]);
    }
    argv.extend([
        "--offline".to_owned(),
        "--no-context-files".to_owned(),
        "--no-skills".to_owned(),
        "--no-prompt-templates".to_owned(),
        "--no-themes".to_owned(),
        "--approve".to_owned(),
    ]);
    Ok(argv)
}

fn launch_env(sandbox: &Sandbox, opts: &LaunchOpts) -> Result<LaunchEnv, CorpusError> {
    require_prerequisites()?;
    let mut env = BTreeMap::new();
    env.insert(
        "HOME".to_owned(),
        sandbox.home_dir.to_string_lossy().into_owned(),
    );
    env.insert("PI_OFFLINE".to_owned(), "1".to_owned());
    env.insert(
        "PI_EXTENSION_HOST".to_owned(),
        extension_host_path()?.to_string_lossy().into_owned(),
    );
    env.insert("PI_VERIFICATION_MODE".to_owned(), "text".to_owned());
    env.insert(
        "PI_VERIFICATION_CHUNK_COUNT".to_owned(),
        opts.chunk_count.to_string(),
    );
    env.insert(
        "PI_VERIFICATION_CHUNK_DELAY_MS".to_owned(),
        opts.chunk_delay_ms.to_string(),
    );
    env.insert(
        "PI_VERIFICATION_FINAL_MARKER".to_owned(),
        FINAL_MARKER.to_owned(),
    );

    if opts.wizard {
        // First-run gate requires PI_EXPERIMENTAL=1 and no PI_CODING_AGENT_DIR override.
        env.insert("PI_EXPERIMENTAL".to_owned(), "1".to_owned());
    } else {
        env.insert(
            "PI_CODING_AGENT_DIR".to_owned(),
            sandbox.agent_dir.to_string_lossy().into_owned(),
        );
        env.insert(
            "PI_CODING_AGENT_SESSION_DIR".to_owned(),
            sandbox.session_dir.to_string_lossy().into_owned(),
        );
        if opts.rebind_keybindings {
            fs::write(
                sandbox.agent_dir.join("keybindings.json"),
                "{\n  \"app.tools.expand\": \"ctrl+m\",\n  \"app.thinking.cycle\": \"f9\"\n}\n",
            )?;
        }
    }

    let argv = if opts.wizard {
        // Wizard coverage still hard-requires extension host/extension
        // existence, but avoids loading the extension so setup UI stays
        // deterministic.
        let mut argv = vec![pi_binary()?.to_string_lossy().into_owned()];
        argv.extend([
            "--offline".to_owned(),
            "--no-context-files".to_owned(),
            "--no-skills".to_owned(),
            "--no-prompt-templates".to_owned(),
            "--no-themes".to_owned(),
            "--approve".to_owned(),
        ]);
        argv
    } else {
        common_argv(opts.include_extension)?
    };

    let context = NormalizationContext {
        home: Some(sandbox.home_dir.as_os_str().as_encoded_bytes().to_vec()),
        cwd: Some(sandbox.work_dir.as_os_str().as_encoded_bytes().to_vec()),
    };
    Ok(LaunchEnv {
        argv,
        env,
        cwd: sandbox.work_dir.clone(),
        context,
    })
}

fn host_row_id() -> RowId {
    match (std::env::consts::OS, std::env::consts::ARCH) {
        ("linux", "aarch64") => RowId::GnuArm64,
        ("macos", "x86_64") => RowId::DarwinX64,
        ("macos", "aarch64") => RowId::DarwinArm64,
        ("windows", _) => RowId::WindowsX64,
        _ => RowId::GnuX64,
    }
}

fn host_driver_kind() -> DriverKind {
    if cfg!(windows) {
        DriverKind::ConPty
    } else {
        DriverKind::PosixPty
    }
}

/// Row directory label + `RunnerRow`. Absent `PI_TUI_TIER_ROW` ⇒ `local`.
fn resolve_row() -> Result<(String, RunnerRow), CorpusError> {
    match std::env::var("PI_TUI_TIER_ROW") {
        Ok(value) if !value.trim().is_empty() => {
            let value = value.trim().to_owned();
            let (tier, id, runner_image) = parse_tier_row(&value)?;
            Ok((
                value,
                RunnerRow {
                    tier,
                    id,
                    runner_image,
                },
            ))
        }
        _ => Ok((
            "local".to_owned(),
            RunnerRow {
                tier: RowTier::Local,
                id: host_row_id(),
                runner_image: None,
            },
        )),
    }
}

fn parse_tier_row(raw: &str) -> Result<(RowTier, RowId, Option<String>), CorpusError> {
    if raw == "local" {
        return Ok((RowTier::Local, host_row_id(), None));
    }
    let (tier_prefix, rest) = if let Some(rest) = raw.strip_prefix("tier-n/") {
        (RowTier::TierN, rest)
    } else if let Some(rest) = raw.strip_prefix("tier-n:") {
        (RowTier::TierN, rest)
    } else {
        (RowTier::Local, raw)
    };
    let (id_raw, image) = match rest.split_once('@') {
        Some((id, image)) => (id, Some(image.to_owned())),
        None => (rest, None),
    };
    let id = match id_raw {
        "gnu-x64" | "GnuX64" => RowId::GnuX64,
        "gnu-arm64" | "GnuArm64" => RowId::GnuArm64,
        "darwin-x64" | "DarwinX64" => RowId::DarwinX64,
        "darwin-arm64" | "DarwinArm64" => RowId::DarwinArm64,
        "windows-x64" | "WindowsX64" => RowId::WindowsX64,
        other => {
            return Err(CorpusError::Prerequisite(format!(
                "unknown PI_TUI_TIER_ROW id {other}"
            )));
        }
    };
    if tier_prefix == RowTier::TierN && image.as_ref().is_none_or(String::is_empty) {
        return Err(CorpusError::Prerequisite(
            "Tier-n PI_TUI_TIER_ROW requires @runner-image".to_owned(),
        ));
    }
    Ok((tier_prefix, id, image))
}

fn target_root() -> PathBuf {
    if let Ok(target) = std::env::var("CARGO_TARGET_DIR") {
        return PathBuf::from(target);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target")
}

fn artifact_path(row_label: &str, scenario_dir: &str, iteration: usize) -> PathBuf {
    target_root()
        .join("verification/tui-transcripts")
        .join(row_label)
        .join(scenario_dir)
        .join(format!("run-{}", iteration + 1))
        .join("transcript.artifact.json")
}

fn write_artifact(
    row_label: &str,
    scenario_dir: &str,
    iteration: usize,
    artifact: &TranscriptArtifact,
) -> Result<PathBuf, CorpusError> {
    if artifact.row.tier == RowTier::TierN && row_label == "local" {
        return Err(CorpusError::Assert(
            "local runs must never claim Tier N".to_owned(),
        ));
    }
    let path = artifact_path(row_label, scenario_dir, iteration);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let body = serde_json::to_vec_pretty(artifact)
        .map_err(|error| CorpusError::Io(format!("serialize artifact: {error}")))?;
    let mut file = fs::File::create(&path)?;
    file.write_all(&body)?;
    file.write_all(b"\n")?;
    Ok(path)
}

fn contains_bytes(haystack: &[u8], needle: &[u8]) -> bool {
    haystack
        .windows(needle.len())
        .any(|window| window == needle)
}

fn ready_predicate(bytes: &[u8]) -> bool {
    let screen = String::from_utf8_lossy(bytes);
    if screen.contains("esc to cancel") {
        return false;
    }
    if screen.contains("Choose a theme family") {
        return true;
    }
    READY_MARKERS
        .iter()
        .any(|marker| screen.contains(String::from_utf8_lossy(marker).as_ref()))
        || screen.contains(FINAL_MARKER)
}

fn observed_claims(raw: &[u8], saw_snapshot: bool) -> Vec<ClaimClass> {
    let mut claims = vec![
        ClaimClass::Execution,
        ClaimClass::Protocol,
        ClaimClass::Pty,
        ClaimClass::Render,
    ];
    let audit = pi_tui::terminal::audit_bytes(raw);
    if audit.sync_begin > 0 && audit.sync_begin == audit.sync_end {
        claims.push(ClaimClass::SynchronizedOutput);
    }
    if audit.clear_2j == 0 && audit.clear_3j == 0 {
        claims.push(ClaimClass::NoClear);
    }
    if saw_snapshot {
        claims.push(ClaimClass::Snapshot);
    }
    claims.sort();
    claims.dedup();
    claims
}

fn open_driver(spec: &LaunchSpec) -> Result<HostSession, CorpusError> {
    #[cfg(unix)]
    {
        Ok(PosixPtyDriver.open(spec)?)
    }
    #[cfg(windows)]
    {
        Ok(ConPtyDriver.open(spec)?)
    }
}

impl ProductRun {
    fn open(
        launch: LaunchEnv,
        row: RunnerRow,
        initial_claims: Vec<ClaimClass>,
    ) -> Result<Self, CorpusError> {
        let geometry = Geometry {
            cols: INITIAL_COLS,
            rows: INITIAL_ROWS,
        };
        let profile = CapabilityProfile::Xterm256ColorTruecolor;
        let spec = LaunchSpec {
            argv: launch.argv.clone(),
            cwd: launch.cwd,
            env: launch.env,
            geometry,
            profile,
        };
        let session = open_driver(&spec)?;
        let recorder = TranscriptRecorder::new(TranscriptSpec {
            scenario: Scenario::KeyboardGauntlet,
            row,
            geometry,
            capability_profile: profile,
            driver_kind: host_driver_kind(),
            mode: TranscriptMode::Standard,
            claims: initial_claims,
            timing: TimingEnvelope::default(),
            output_canon: OutputCanon::SettledFrame,
        });
        let recording = RecordingSession::new(session, recorder, launch.argv, &launch.context)?;
        Ok(Self {
            recording,
            context: launch.context,
            policy: SettlePolicy::new(Duration::from_millis(150), Duration::from_secs(45))?,
            raw_acc: Vec::new(),
            wall_started: Instant::now(),
            settle_windows_ms: Vec::new(),
            saw_snapshot: false,
        })
    }

    fn write_input(&mut self, bytes: &[u8]) -> Result<(), CorpusError> {
        self.recording.write(bytes)?;
        Ok(())
    }

    fn send_line(&mut self, line: &str) -> Result<(), CorpusError> {
        let mut bytes = line.as_bytes().to_vec();
        bytes.extend_from_slice(KEY_ENTER);
        self.write_input(&bytes)
    }

    fn settle_output<F>(&mut self, predicate: F) -> Result<Vec<u8>, CorpusError>
    where
        F: FnMut(&[u8]) -> bool,
    {
        Ok(self.settle_frame(predicate)?.batch.bytes)
    }

    fn settle_frame<F>(
        &mut self,
        mut predicate: F,
    ) -> Result<pi_tui::testkit::driver::SettledFrame, CorpusError>
    where
        F: FnMut(&[u8]) -> bool,
    {
        let started = Instant::now();
        let mut last_screen = String::new();
        let result = self.recording.read_settled_frame_where(
            &self.policy,
            |snapshot| {
                last_screen = snapshot.lines.join("\n");
                predicate(last_screen.as_bytes())
            },
            &self.context,
        );
        let frame = result.map_err(|error| {
            CorpusError::Assert(format!(
                "{error}; pred_ready={} pred_screen:\n{last_screen}",
                ready_predicate(last_screen.as_bytes())
            ))
        })?;
        self.settle_windows_ms
            .push(u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX));
        self.raw_acc.extend_from_slice(&frame.batch.bytes);
        self.saw_snapshot = true;
        Ok(frame)
    }

    fn finish(mut self) -> Result<(TranscriptArtifact, Option<u32>), CorpusError> {
        let status = self.recording.close()?;
        let mut artifact = self.recording.finish()?;
        artifact.timing.wall_ms =
            u64::try_from(self.wall_started.elapsed().as_millis()).unwrap_or(u64::MAX);
        artifact.timing.settle_windows_ms = self.settle_windows_ms;
        artifact.claims = observed_claims(&self.raw_acc, self.saw_snapshot);
        artifact.digest = pi_tui::testkit::transcript::digest_canonical(&artifact)
            .map_err(|error| CorpusError::Transcript(error.to_string()))?;
        Ok((artifact, Some(status.code)))
    }
}

/// Run one scenario k times, returning the final digest for the verdict.
fn run_scenario_k(
    scenario_dir: &str,
    producer: impl FnMut(usize) -> Result<TranscriptArtifact, CorpusError>,
) -> Result<String, CorpusError> {
    match run_k(K, producer) {
        Ok(report) => {
            assert_eq!(report.k, K);
            assert!(
                report.digest.starts_with("sha256:"),
                "{scenario_dir}: digest must be sha256-prefixed"
            );
            assert!(
                !report.canonical_bytes.is_empty(),
                "{scenario_dir}: canonical bytes must be non-empty"
            );
            Ok(report.digest)
        }
        Err(RepeatError::Divergence {
            first_divergent_seq,
            left_digest,
            right_digest,
        }) => Err(CorpusError::Assert(format!(
            "{scenario_dir}: run-to-run divergence at seq {first_divergent_seq}: {left_digest} != {right_digest}"
        ))),
        Err(error) => Err(CorpusError::Assert(format!("{scenario_dir}: {error}"))),
    }
}

fn quit_cleanly(run: &mut ProductRun) -> Result<(), CorpusError> {
    run.send_line("/quit")?;
    Ok(())
}

/// Assert the settled-frame screen model shows the needle on one contiguous
/// line (cell-diff writes fragment raw text; the snapshot is the truth).
fn frame_lines_contain(
    scenario: &str,
    frame: &pi_tui::testkit::driver::SettledFrame,
    needle: &str,
) -> Result<(), CorpusError> {
    if frame
        .snapshot
        .lines
        .iter()
        .any(|line| line.contains(needle))
    {
        return Ok(());
    }
    Err(CorpusError::Assert(format!(
        "{scenario}: settled frame missing {needle:?} on any screen line; screen:\n{}",
        frame.snapshot.lines.join("\n")
    )))
}

fn dismiss_until_gone(
    run: &mut ProductRun,
    scenario: &str,
    title: &str,
) -> Result<(), CorpusError> {
    run.write_input(KEY_ESCAPE)?;
    match run.settle_frame(|bytes| !contains_bytes(bytes, title.as_bytes())) {
        Ok(_) => Ok(()),
        Err(error) => Err(CorpusError::Assert(format!(
            "{scenario}: {title:?} still on screen after Esc: {error}"
        ))),
    }
}

/// Type a sentinel into the editor and prove focus restored.
fn prove_editor_focus(
    run: &mut ProductRun,
    scenario: &str,
    sentinel: &str,
) -> Result<(), CorpusError> {
    for byte in sentinel.bytes() {
        run.write_input(&[byte])?;
    }
    let frame = run.settle_frame(|bytes| contains_bytes(bytes, sentinel.as_bytes()))?;
    let on_prompt = frame.snapshot.lines.iter().any(|line| {
        let trimmed = line.trim();
        trimmed.contains(sentinel) && trimmed.contains('❯')
    });
    if !on_prompt {
        return Err(CorpusError::Assert(format!(
            "{scenario}: sentinel {sentinel:?} did not land on the composer; screen:\n{}",
            frame.snapshot.lines.join("\n")
        )));
    }
    for _ in 0..sentinel.len() {
        run.write_input(BACKSPACE)?;
    }
    let _ = run.settle_output(ready_predicate)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Scenario 1: keyboard-only first-run wizard
// ---------------------------------------------------------------------------

fn run_keyboard_wizard(
    iteration: usize,
    row_label: &str,
    row: &RunnerRow,
) -> Result<TranscriptArtifact, CorpusError> {
    let sandbox = create_sandbox()?;
    let launch = launch_env(
        &sandbox,
        &LaunchOpts {
            wizard: true,
            include_extension: false,
            ..LaunchOpts::default()
        },
    )?;
    let mut run = ProductRun::open(launch, row.clone(), vec![ClaimClass::Execution])?;

    let frame = run.settle_frame(|bytes| contains_bytes(bytes, b"Choose a theme family"))?;
    frame_lines_contain("keyboard-wizard", &frame, "Choose a theme family")?;

    // ctrl+d during the wizard is ignored (first_run_step guards app.exit):
    // the family step must still be on screen afterwards.
    run.write_input(KEY_CTRL_D)?;
    let frame = run.settle_frame(|bytes| contains_bytes(bytes, b"Choose a theme family"))?;
    frame_lines_contain("keyboard-wizard", &frame, "Choose a theme family")?;

    // Keyboard-only completion: Down+Enter per step (family → mode → analytics).
    run.write_input(KEY_DOWN)?;
    run.write_input(KEY_ENTER)?;
    let frame = run.settle_frame(|bytes| {
        contains_bytes(bytes, b"Choose a theme mode") || ready_predicate(bytes)
    })?;
    frame_lines_contain("keyboard-wizard", &frame, "Choose a theme mode")?;

    run.write_input(KEY_DOWN)?;
    run.write_input(KEY_ENTER)?;
    let frame = run.settle_frame(|bytes| {
        contains_bytes(bytes, b"anonymous usage") || ready_predicate(bytes)
    })?;
    frame_lines_contain("keyboard-wizard", &frame, "anonymous usage")?;

    // Down selects "Don't share" → analytics must persist false.
    run.write_input(KEY_DOWN)?;
    run.write_input(KEY_ENTER)?;
    // The dismissal reanchor repaints every row; settle directly on the
    // composer-ready marker being visible in the reconstructed viewport.
    let frame = run.settle_frame(|bytes| {
        contains_bytes(bytes, b"type a message") || contains_bytes(bytes, b"No messages")
    })?;
    frame_lines_contain("keyboard-wizard", &frame, "type a message")
        .or_else(|_| frame_lines_contain("keyboard-wizard", &frame, "No messages"))?;

    quit_cleanly(&mut run)?;
    let (artifact, exit_code) = run.finish()?;
    if exit_code != Some(0) {
        return Err(CorpusError::Assert(format!(
            "keyboard-wizard: expected clean exit 0, got {exit_code:?}"
        )));
    }

    // Persisted settings prove the keyboard-only flow completed on disk.
    let settings_path = sandbox
        .home_dir
        .join(".pi")
        .join("agent")
        .join("settings.json");
    let raw = fs::read_to_string(&settings_path).map_err(|error| {
        CorpusError::Assert(format!(
            "keyboard-wizard: settings not persisted at {}: {error}",
            settings_path.display()
        ))
    })?;
    let value: serde_json::Value = serde_json::from_str(&raw).map_err(|error| {
        CorpusError::Assert(format!("keyboard-wizard: settings parse: {error}"))
    })?;
    let theme = value
        .get("theme")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default();
    let theme_mode = value
        .get("themeMode")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default();
    let analytics = value
        .get("enableAnalytics")
        .and_then(serde_json::Value::as_bool);
    if theme.is_empty() || theme_mode != "dark" || analytics != Some(false) {
        return Err(CorpusError::Assert(format!(
            "keyboard-wizard: persisted settings wrong (theme={theme:?}, themeMode={theme_mode:?}, enableAnalytics={analytics:?}) from {raw}"
        )));
    }

    write_artifact(row_label, "keyboard-wizard", iteration, &artifact)?;
    Ok(artifact)
}

// ---------------------------------------------------------------------------
// Scenario 2: slash-command selector flows
// ---------------------------------------------------------------------------

fn run_keyboard_slash_flows(
    iteration: usize,
    row_label: &str,
    row: &RunnerRow,
) -> Result<TranscriptArtifact, CorpusError> {
    let sandbox = create_sandbox()?;
    let launch = launch_env(&sandbox, &LaunchOpts::default())?;
    let mut run = ProductRun::open(launch, row.clone(), vec![ClaimClass::Execution])?;
    let _ = run.settle_frame(ready_predicate)?;
    let scenario = "keyboard-slash-flows";

    // /login: auth-type rows are unique to this selector (boot chrome never
    // mentions signing in).
    run.send_line("/login")?;
    let frame = run.settle_frame(|bytes| contains_bytes(bytes, b"Sign in with an account"))?;
    frame_lines_contain(scenario, &frame, "Sign in with an account")?;
    run.write_input(KEY_ESCAPE)?;
    let _ = run.settle_output(ready_predicate)?;

    // /tree: empty conversation tree → the selector's empty copy renders.
    run.send_line("/tree")?;
    let frame = run.settle_frame(|bytes| contains_bytes(bytes, b"No entries found"))?;
    frame_lines_contain(scenario, &frame, "No entries found")?;
    run.write_input(KEY_ESCAPE)?;
    let _ = run.settle_output(ready_predicate)?;
    prove_editor_focus(&mut run, scenario, "treefocus")?;

    // /settings: settings selector rows.
    run.send_line("/settings")?;
    let frame = run.settle_frame(|bytes| contains_bytes(bytes, b"Auto-compact"))?;
    frame_lines_contain(scenario, &frame, "Auto-compact")?;
    run.write_input(KEY_ESCAPE)?;
    let _ = run.settle_output(ready_predicate)?;
    prove_editor_focus(&mut run, scenario, "settingsfocus")?;

    // /import with no argument: usage notice lands in the transcript.
    run.send_line("/import")?;
    let frame = run.settle_frame(|bytes| contains_bytes(bytes, b"Usage: /import"))?;
    frame_lines_contain(scenario, &frame, "Usage: /import")?;
    prove_editor_focus(&mut run, scenario, "importusagefocus")?;

    // /import <path>: replace-session confirm selector, Esc → cancelled.
    let import_path = sandbox.work_dir.join("verification-export.jsonl");
    fs::write(&import_path, "{}\n")?;
    run.send_line(&format!("/import {}", import_path.to_string_lossy()))?;
    let frame = run.settle_frame(|bytes| contains_bytes(bytes, b"Yes, replace current session"))?;
    frame_lines_contain(scenario, &frame, "Yes, replace current session")?;
    run.write_input(KEY_ESCAPE)?;
    let frame = run.settle_frame(|bytes| contains_bytes(bytes, b"Import cancelled"))?;
    frame_lines_contain(scenario, &frame, "Import cancelled")?;
    prove_editor_focus(&mut run, scenario, "importfocus")?;

    // /logout with the verification provider loaded opens the credential
    // selector (Cancel + stored provider). The prompt is a placeholder, so
    // the unique painted row is the Cancel item.
    run.send_line("/logout")?;
    let frame = run.settle_frame(|bytes| contains_bytes(bytes, "→ Cancel".as_bytes()))?;
    frame_lines_contain(scenario, &frame, "→ Cancel")?;
    run.write_input(KEY_ESCAPE)?;
    let _ = run.settle_output(ready_predicate)?;
    prove_editor_focus(&mut run, scenario, "logoutfocus")?;

    quit_cleanly(&mut run)?;
    let (artifact, exit_code) = run.finish()?;
    if exit_code != Some(0) {
        return Err(CorpusError::Assert(format!(
            "{scenario}: expected clean exit 0, got {exit_code:?}"
        )));
    }
    write_artifact(row_label, scenario, iteration, &artifact)?;
    Ok(artifact)
}

// ---------------------------------------------------------------------------
// Scenario 3: ctrl+d resolution order per context
// ---------------------------------------------------------------------------

fn run_keyboard_ctrl_d_order(
    iteration: usize,
    row_label: &str,
    row: &RunnerRow,
) -> Result<TranscriptArtifact, CorpusError> {
    let sandbox = create_sandbox()?;
    let launch = launch_env(&sandbox, &LaunchOpts::default())?;
    let mut run = ProductRun::open(launch, row.clone(), vec![ClaimClass::Execution])?;
    let _ = run.settle_frame(ready_predicate)?;
    let scenario = "keyboard-ctrl-d-order";

    // While a selector holds focus, ctrl+d must resolve inside the selector
    // context (filter chord / delete guard / swallow) — never app exit. Each
    // needle is unique to the opened selector (absent from boot chrome and
    // from the transcript so far) so the settle proves the open, and the
    // post-Esc sentinel proves liveness would have caught an exit regression.
    // /tree runs before the seeded turn: the empty tree's copy ("No entries
    // found") is unique, while a seeded tree repeats chat text.
    let selector_probes: [(&str, &str, &[u8]); 5] = [
        ("/tree", "No entries found", b"No entries found"),
        ("/model", "Nova 2 Lite", b"Nova 2 Lite"),
        ("/settings", "Auto-compact", b"Auto-compact"),
        (
            "/login",
            "Sign in with an account",
            b"Sign in with an account",
        ),
        ("/resume", "msgs", b"msgs"),
    ];
    for (probe, (command, needle, raw_needle)) in selector_probes.iter().enumerate() {
        run.send_line(command)?;
        let frame = run.settle_frame(|bytes| contains_bytes(bytes, raw_needle))?;
        frame_lines_contain(scenario, &frame, needle)?;
        run.write_input(KEY_CTRL_D)?;
        // The selector context must still be on screen after ctrl+d.
        let frame = run.settle_frame(|bytes| {
            contains_bytes(bytes, raw_needle)
                || contains_bytes(bytes, b"Cannot delete the currently active session")
        })?;
        if *command == "/resume" {
            // Active-row delete guard: error surfaces, selector stays, app lives.
            frame_lines_contain(
                scenario,
                &frame,
                "Cannot delete the currently active session",
            )?;
            frame_lines_contain(scenario, &frame, "esc to cancel")?;
        } else {
            frame_lines_contain(scenario, &frame, needle)?;
        }
        run.write_input(KEY_ESCAPE)?;
        let _ = run.settle_output(ready_predicate)?;
        let sentinel = format!("cd{probe}");
        prove_editor_focus(&mut run, scenario, &sentinel)?;
        if *command == "/tree" {
            // Deterministic seeded turn so /resume has the active session.
            run.send_line("verification seeded turn")?;
            let _ = run.settle_frame(|bytes| contains_bytes(bytes, FINAL_MARKER.as_bytes()))?;
        }
    }

    // Non-empty editor: ctrl+d is forward-delete, not exit.
    run.write_input(b"\x05")?;
    for _ in 0..32 {
        run.write_input(BACKSPACE)?;
    }
    run.write_input(b"xyz")?;
    run.write_input(KEY_LEFT)?;
    run.write_input(KEY_CTRL_D)?;
    let frame =
        run.settle_frame(|bytes| contains_bytes(bytes, b"xy") && !contains_bytes(bytes, b"xyz"))?;
    if !frame
        .snapshot
        .lines
        .iter()
        .any(|line| line.contains('❯') && line.contains("xy") && !line.contains("xyz"))
    {
        return Err(CorpusError::Assert(format!(
            "{scenario}: forward-delete did not leave xy on the composer; screen:\n{}",
            frame.snapshot.lines.join("\n")
        )));
    }
    run.write_input(b"\x05")?;
    for _ in 0..8 {
        run.write_input(BACKSPACE)?;
    }
    run.write_input(KEY_CTRL_D)?;
    let (artifact, exit_code) = run.finish()?;
    if exit_code != Some(0) {
        return Err(CorpusError::Assert(format!(
            "{scenario}: empty-editor ctrl+d must exit cleanly (0), got {exit_code:?}"
        )));
    }
    write_artifact(row_label, scenario, iteration, &artifact)?;
    Ok(artifact)
}

// ---------------------------------------------------------------------------
// Scenario 4: streaming interrupt
// ---------------------------------------------------------------------------

fn run_keyboard_streaming_interrupt(
    iteration: usize,
    row_label: &str,
    row: &RunnerRow,
) -> Result<TranscriptArtifact, CorpusError> {
    let sandbox = create_sandbox()?;
    let launch = launch_env(
        &sandbox,
        &LaunchOpts {
            chunk_count: 12,
            chunk_delay_ms: 300,
            ..LaunchOpts::default()
        },
    )?;
    let mut run = ProductRun::open(launch, row.clone(), vec![ClaimClass::Execution])?;
    let _ = run.settle_frame(ready_predicate)?;
    let scenario = "keyboard-streaming-interrupt";

    // Esc during a stream must preserve liveness. The deterministic provider
    // does not cancel its scripted reply, so wait for its terminal frame.
    run.send_line("verification interruptible stream one")?;
    let _ = run.settle_frame(|bytes| contains_bytes(bytes, b"verification-chunk-0001"))?;
    run.write_input(KEY_ESCAPE)?;
    let frame = run.settle_frame(|bytes| {
        contains_bytes(bytes, b"Aborting")
            || contains_bytes(bytes, b"(cancelled)")
            || contains_bytes(bytes, FINAL_MARKER.as_bytes())
    })?;
    frame_lines_contain(scenario, &frame, FINAL_MARKER)?;

    run.send_line("verification interruptible stream two")?;
    let _ = run.settle_frame(|bytes| {
        contains_bytes(bytes, b"verification interruptible stream two")
            && contains_bytes(bytes, b"verification-chunk-0001")
    })?;
    let frame = run.settle_frame(|bytes| contains_bytes(bytes, FINAL_MARKER.as_bytes()))?;
    frame_lines_contain(scenario, &frame, FINAL_MARKER)?;
    prove_editor_focus(&mut run, scenario, "streamfocus")?;
    quit_cleanly(&mut run)?;
    let (artifact, exit_code) = run.finish()?;
    if exit_code != Some(0) {
        return Err(CorpusError::Assert(format!(
            "{scenario}: expected clean exit 0, got {exit_code:?}"
        )));
    }
    write_artifact(row_label, scenario, iteration, &artifact)?;
    Ok(artifact)
}

// ---------------------------------------------------------------------------
// Scenario 5: overlay focus restore and overlay-over-overlay stacking
// ---------------------------------------------------------------------------

fn run_keyboard_overlay_focus(
    iteration: usize,
    row_label: &str,
    row: &RunnerRow,
) -> Result<TranscriptArtifact, CorpusError> {
    let sandbox = create_sandbox()?;
    let launch = launch_env(&sandbox, &LaunchOpts::default())?;
    let mut run = ProductRun::open(launch, row.clone(), vec![ClaimClass::Execution])?;
    let _ = run.settle_frame(ready_predicate)?;
    let scenario = "keyboard-overlay-focus";

    run.send_line("/hotkeys")?;
    let frame = run.settle_frame(|bytes| contains_bytes(bytes, b"Keyboard shortcuts"))?;
    frame_lines_contain(scenario, &frame, "Keyboard shortcuts")?;
    dismiss_until_gone(&mut run, scenario, "Keyboard shortcuts")?;
    prove_editor_focus(&mut run, scenario, "hotkeysfocus")?;

    run.send_line("/verification-dialogs")?;
    for title in [
        "Verification select prompt",
        "Verification confirm prompt",
        "Verification input prompt",
        "Verification editor prompt",
    ] {
        let frame = run.settle_frame(|bytes| contains_bytes(bytes, title.as_bytes()))?;
        frame_lines_contain(scenario, &frame, title)?;
        dismiss_until_gone(&mut run, scenario, title)?;
    }
    let _ = run.settle_frame(ready_predicate)?;
    prove_editor_focus(&mut run, scenario, "dialogsfocus")?;

    // Overlay-over-overlay: focusable extension overlay + pending select
    // dialog stack (both visible). The Esc-while-stacked key-routing behavior
    // is RECORDED in the canonical frames (current semantics route keys to
    // the overlay slot — TUI-G7 owns any change), not asserted as contract.
    // Hard asserts cover stacking, timeout auto-cancel, overlay completion,
    // and editor focus restore.
    run.send_line("/verification-overlay-stack")?;
    let frame = run.settle_frame(|bytes| {
        contains_bytes(bytes, STACKED_OVERLAY_LINE.as_bytes())
            && contains_bytes(bytes, STACKED_DIALOG_TITLE.as_bytes())
    })?;
    frame_lines_contain(scenario, &frame, STACKED_OVERLAY_LINE)?;
    frame_lines_contain(scenario, &frame, STACKED_DIALOG_TITLE)?;

    run.write_input(KEY_ESCAPE)?;
    // Liveness only: the app keeps rendering whichever surface owns the key.
    let _ = run.settle_frame(ready_predicate)?;

    // Dialog timeout auto-cancels even while the overlay holds focus.
    // One settle records one snapshot — a poll loop would emit a
    // timing-dependent number of frames and fail k-run digest equality.
    let frame = run.settle_frame(|bytes| {
        contains_bytes(bytes, STACKED_OVERLAY_LINE.as_bytes())
            && !contains_bytes(bytes, STACKED_DIALOG_TITLE.as_bytes())
    })?;
    frame_lines_contain(scenario, &frame, STACKED_OVERLAY_LINE)?;
    if frame
        .snapshot
        .lines
        .iter()
        .any(|line| line.contains(STACKED_DIALOG_TITLE))
    {
        return Err(CorpusError::Assert(format!(
            "{scenario}: stacked dialog did not timeout auto-cancel while overlay held focus"
        )));
    }

    // 'x' completes the overlay and restores editor routing.
    run.write_input(b"x")?;
    let _ = run.settle_output(ready_predicate)?;
    prove_editor_focus(&mut run, scenario, "stackfocus")?;

    quit_cleanly(&mut run)?;
    let (artifact, exit_code) = run.finish()?;
    if exit_code != Some(0) {
        return Err(CorpusError::Assert(format!(
            "{scenario}: expected clean exit 0, got {exit_code:?}"
        )));
    }
    write_artifact(row_label, scenario, iteration, &artifact)?;
    Ok(artifact)
}

// ---------------------------------------------------------------------------
// Scenario 6: rebind reflection in every rendered hint
// ---------------------------------------------------------------------------

fn run_keyboard_rebind_hints(
    iteration: usize,
    row_label: &str,
    row: &RunnerRow,
) -> Result<TranscriptArtifact, CorpusError> {
    let sandbox = create_sandbox()?;
    let launch = launch_env(
        &sandbox,
        &LaunchOpts {
            include_extension: false,
            rebind_keybindings: true,
            ..LaunchOpts::default()
        },
    )?;
    let mut run = ProductRun::open(launch, row.clone(), vec![ClaimClass::Execution])?;
    let scenario = "keyboard-rebind-hints";

    // Empty-state hint must render the rebound chords, never the defaults.
    let frame = run.settle_frame(ready_predicate)?;
    frame_lines_contain(
        scenario,
        &frame,
        "/hotkeys shortcuts · ctrl+m expand tools · f9 thinking",
    )?;
    for line in &frame.snapshot.lines {
        if line.contains("expand tools") && line.contains("ctrl+o") {
            return Err(CorpusError::Assert(format!(
                "{scenario}: empty-state hint still shows default ctrl+o: {line}"
            )));
        }
    }

    // /hotkeys overlay rows resolve from the same registry.
    run.send_line("/hotkeys")?;
    let frame = run.settle_frame(|bytes| contains_bytes(bytes, b"Keyboard shortcuts"))?;
    frame_lines_contain(scenario, &frame, "Keyboard shortcuts")?;
    let mut toggle_row = false;
    let mut thinking_row = false;
    for line in &frame.snapshot.lines {
        // Overlay rows use `keyDisplayText` (capitalized: Ctrl+M, F9); the
        // header empty-state hint uses lowercase. Both must reflect rebinds.
        if line.contains("Toggle tool output") {
            if !line.contains("Ctrl+M") {
                return Err(CorpusError::Assert(format!(
                    "{scenario}: Toggle tool output row missing rebound Ctrl+M: {line}"
                )));
            }
            toggle_row = true;
        }
        if line.contains("Cycle thinking") {
            if !line.contains("F9") {
                return Err(CorpusError::Assert(format!(
                    "{scenario}: Cycle thinking row missing rebound F9: {line}"
                )));
            }
            thinking_row = true;
        }
    }
    if !toggle_row || !thinking_row {
        return Err(CorpusError::Assert(format!(
            "{scenario}: /hotkeys overlay rows not found (toggle={toggle_row}, thinking={thinking_row})"
        )));
    }
    run.write_input(KEY_ESCAPE)?;
    let _ = run.settle_output(ready_predicate)?;

    quit_cleanly(&mut run)?;
    let (artifact, exit_code) = run.finish()?;
    if exit_code != Some(0) {
        return Err(CorpusError::Assert(format!(
            "{scenario}: expected clean exit 0, got {exit_code:?}"
        )));
    }
    write_artifact(row_label, scenario, iteration, &artifact)?;
    Ok(artifact)
}

/// Per-row verdict record: per-scenario pass/fail, k, digest, tier labels.
fn write_verdict(
    row_label: &str,
    row: &RunnerRow,
    digest: &str,
    scenarios: &[(&'static str, &'static str)],
) -> Result<PathBuf, CorpusError> {
    let path = target_root()
        .join("verification/tui-transcripts")
        .join(row_label)
        .join("keyboard-gauntlet")
        .join("verdict.json");
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let verdict = serde_json::json!({
        "stableId": "TUI-V2",
        "corpus": "keyboard-gauntlet",
        "row": {
            "label": row_label,
            "tier": format!("{:?}", row.tier).to_lowercase(),
            "id": format!("{:?}", row.id).to_lowercase(),
            "runnerImage": row.runner_image,
        },
        "k": K,
        "digest": digest,
        "scenarios": scenarios
            .iter()
            .map(|(scenario, verdict)| { (scenario.to_owned(), verdict.to_owned()) })
            .collect::<BTreeMap<_, _>>(),
    });
    let body = serde_json::to_vec_pretty(&verdict)
        .map_err(|error| CorpusError::Io(format!("serialize verdict: {error}")))?;
    let mut file = fs::File::create(&path)?;
    file.write_all(&body)?;
    file.write_all(b"\n")?;
    Ok(path)
}

#[expect(
    clippy::panic,
    reason = "test hard-fail: gauntlet failure is irrecoverable"
)]
fn hard_fail(error: &CorpusError) -> ! {
    panic!("tui keyboard gauntlet hard-fail: {error}");
}
#[expect(
    clippy::type_complexity,
    reason = "scenario table type is inherently complex; a type alias would be used only in this test"
)]
#[test]
fn tui_keyboard_gauntlet_wizard_slash_ctrl_d_streaming_overlay_rebind() {
    if let Err(error) = require_prerequisites() {
        hard_fail(&error);
    }
    let (row_label, row) = match resolve_row() {
        Ok(value) => value,
        Err(error) => hard_fail(&error),
    };
    if row_label == "local" {
        assert_eq!(
            row.tier,
            RowTier::Local,
            "local runs must never claim Tier N"
        );
    }

    let scenarios: [(
        &str,
        fn(usize, &str, &RunnerRow) -> Result<TranscriptArtifact, CorpusError>,
    ); 6] = [
        ("keyboard-wizard", run_keyboard_wizard),
        ("keyboard-slash-flows", run_keyboard_slash_flows),
        ("keyboard-ctrl-d-order", run_keyboard_ctrl_d_order),
        (
            "keyboard-streaming-interrupt",
            run_keyboard_streaming_interrupt,
        ),
        ("keyboard-overlay-focus", run_keyboard_overlay_focus),
        ("keyboard-rebind-hints", run_keyboard_rebind_hints),
    ];

    let mut verdicts: Vec<(&'static str, &'static str)> = Vec::new();
    let mut final_digest = String::new();
    for (scenario_dir, runner) in scenarios {
        let row_clone = row.clone();
        let label = row_label.clone();
        match run_scenario_k(scenario_dir, move |iteration| {
            runner(iteration, &label, &row_clone)
        }) {
            Ok(digest) => {
                verdicts.push((scenario_dir, "pass"));
                final_digest = digest;
            }
            Err(error) => hard_fail(&error),
        }
    }

    // Per-row verdict record: all six scenarios settled pass with k=3.
    if let Err(error) = write_verdict(&row_label, &row, &final_digest, &verdicts) {
        hard_fail(&error);
    }
}
