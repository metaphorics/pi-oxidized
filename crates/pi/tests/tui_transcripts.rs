#![allow(
    clippy::expect_used,
    clippy::panic,
    clippy::unwrap_used,
    clippy::too_many_lines,
    clippy::print_stderr
)]
//! Product transcript corpus driven through `pi_tui::testkit::RecordingSession`.
//!
//! Checkpoints use content predicates then quiescence — never timer-only waits.
//! Artifacts land under `target/verification/tui-transcripts/<row>/<scenario>/run-{1,2,3}/`.
//!
//! Named limitation: production interactive boot path passes `ui: None` into
//! `resolve_project_trusted`, so the interactive [`TrustUi`] prompt is absent.
//! This corpus records `/trust` selector + boot trust observations and does not
//! implement `TrustUi` dialog coverage.

use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use pi_tui::terminal::audit_bytes;
use pi_tui::testkit::driver::{
    Geometry as DriverGeometry, LaunchSpec, SettlePolicy, SettledFrame, TerminalDriver,
    TerminalSnapshot,
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
const READY_MARKERS_STR: &[&str] = &["type a message", "type a message to begin", "No messages"];
const KEY_ENTER: &[u8] = b"\r";
const KEY_ESCAPE: &[u8] = b"\x1b";

/// Absent production interactive `TrustUi` prompt (boot path uses `ui: None`).
const LIMITATION_ABSENT_PRODUCTION_TRUST_UI: &str =
    "limitation:absent-production-interactive-TrustUi-prompt";

/// Shared resize ladder ending at settled 1x1 (same as fixture corpus).
const RESIZE_LADDER: [(u16, u16); 8] = [
    (80, 24),
    (40, 12),
    (20, 8),
    (12, 6),
    (8, 4),
    (4, 2),
    (2, 1),
    (1, 1),
];

/// Atomic 24-size storm matching `pi_tui_pty_fixture` `resize_plan`.
const RESIZE_STORM: [(u16, u16); 24] = [
    (80, 24),
    (40, 12),
    (20, 8),
    (12, 6),
    (10, 5),
    (8, 4),
    (16, 10),
    (32, 14),
    (64, 20),
    (100, 30),
    (120, 40),
    (200, 50),
    (24, 8),
    (18, 7),
    (14, 6),
    (11, 5),
    (9, 4),
    (28, 12),
    (48, 16),
    (72, 22),
    (96, 28),
    (160, 36),
    (60, 18),
    (80, 24),
];

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
    let path = PathBuf::from(env!("CARGO_BIN_EXE_pi"));
    if path.is_file() {
        Ok(path)
    } else {
        Err(CorpusError::Prerequisite(format!(
            "product prerequisite missing: CARGO_BIN_EXE_pi points at missing binary {}; rebuild with cargo test -p pi --test tui_transcripts",
            path.display()
        )))
    }
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

fn seed_trust_requiring_project(work_dir: &Path) -> Result<(), CorpusError> {
    let pi_dir = work_dir.join(".pi");
    fs::create_dir_all(&pi_dir)?;
    fs::write(
        pi_dir.join("settings.json"),
        "{\n  \"theme\": \"dark\"\n}\n",
    )?;
    Ok(())
}

fn common_argv(include_extension: bool) -> Result<Vec<String>, CorpusError> {
    let mut argv = vec![
        pi_binary()?.to_string_lossy().into_owned(),
        "--provider".to_owned(),
        VERIFICATION_PROVIDER.to_owned(),
        "--model".to_owned(),
        VERIFICATION_MODEL.to_owned(),
        "--api-key".to_owned(),
        "verification-key".to_owned(),
    ];
    if include_extension {
        argv.push("--extension".to_owned());
        argv.push(extension_path()?.to_string_lossy().into_owned());
        argv.push(format!("--{VERIFICATION_PROFILE_FLAG}"));
        argv.push(VERIFICATION_PROFILE.to_owned());
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

fn launch_env(
    sandbox: &Sandbox,
    include_extension: bool,
    wizard: bool,
) -> Result<LaunchEnv, CorpusError> {
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
    env.insert("PI_VERIFICATION_CHUNK_COUNT".to_owned(), "3".to_owned());
    env.insert("PI_VERIFICATION_CHUNK_DELAY_MS".to_owned(), "0".to_owned());
    env.insert(
        "PI_VERIFICATION_FINAL_MARKER".to_owned(),
        FINAL_MARKER.to_owned(),
    );
    // Keep TERM overlays from LaunchSpec profile; do not force conflicting values.

    if wizard {
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
    }

    let argv = if wizard {
        // Wizard coverage still hard-requires extension host/extension existence,
        // but avoids loading the extension so setup UI stays deterministic.
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
        common_argv(include_extension)?
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

/// Row directory label + `RunnerRow`. Absent `PI_TUI_TIER_ROW` ⇒ `local` (never Tier N).
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

/// Ready markers: cell-diff paints split header text with cursor moves, so
/// raw bytes never carry a contiguous marker — checks run against the
/// reconstructed viewport.
fn ready_snapshot_predicate(snap: &TerminalSnapshot) -> bool {
    let lines = snap.lines.join("\n");
    READY_MARKERS_STR
        .iter()
        .any(|marker| lines.contains(marker))
        || lines.contains(FINAL_MARKER)
        || lines.contains("Choose a theme family")
}

fn observed_claims(raw: &[u8], saw_snapshot: bool) -> Vec<ClaimClass> {
    let mut claims = vec![
        ClaimClass::Execution,
        ClaimClass::Protocol,
        ClaimClass::Pty,
        ClaimClass::Render,
    ];
    let audit = audit_bytes(raw);
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
        scenario: Scenario,
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
            scenario,
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

    fn settle_frame_where<F>(&mut self, predicate: F) -> Result<SettledFrame, CorpusError>
    where
        F: FnMut(&TerminalSnapshot) -> bool,
    {
        let started = Instant::now();
        let frame =
            self.recording
                .read_settled_frame_where(&self.policy, predicate, &self.context)?;
        self.settle_windows_ms
            .push(u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX));
        self.raw_acc.extend_from_slice(&frame.batch.bytes);
        self.saw_snapshot = true;
        Ok(frame)
    }

    fn resize(&mut self, cols: u16, rows: u16) -> Result<(), CorpusError> {
        self.recording.resize(cols, rows)?;
        Ok(())
    }

    fn resize_storm(&mut self, sizes: &[(u16, u16)]) -> Result<(), CorpusError> {
        self.recording.resize_storm(sizes)?;
        Ok(())
    }

    fn finish(mut self) -> Result<TranscriptArtifact, CorpusError> {
        let status = self.recording.close()?;
        if !status.success() {
            return Err(CorpusError::Assert(format!(
                "product exited unsuccessfully: code={} signal={:?}",
                status.code, status.signal
            )));
        }
        let mut artifact = self.recording.finish()?;
        artifact.timing.wall_ms =
            u64::try_from(self.wall_started.elapsed().as_millis()).unwrap_or(u64::MAX);
        artifact.timing.settle_windows_ms = self.settle_windows_ms;
        artifact.claims = observed_claims(&self.raw_acc, self.saw_snapshot);
        // Recompute digest after rewriting observed claims.
        artifact.digest = pi_tui::testkit::transcript::digest_canonical(&artifact)
            .map_err(|error| CorpusError::Transcript(error.to_string()))?;
        Ok(artifact)
    }

    fn raw_so_far(&self) -> &[u8] {
        &self.raw_acc
    }
}

fn run_scenario_k(
    scenario_dir: &str,
    producer: impl FnMut(usize) -> Result<TranscriptArtifact, CorpusError>,
) -> Result<(), CorpusError> {
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
            Ok(())
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

fn run_cold_start(
    iteration: usize,
    row_label: &str,
    row: &RunnerRow,
) -> Result<TranscriptArtifact, CorpusError> {
    let sandbox = create_sandbox()?;
    let launch = launch_env(&sandbox, true, false)?;
    let mut run = ProductRun::open(
        launch,
        Scenario::ColdStart,
        row.clone(),
        vec![ClaimClass::Execution],
    )?;
    let frame = run.settle_frame_where(ready_snapshot_predicate)?;
    if !frame
        .snapshot
        .lines
        .iter()
        .any(|line| line.contains("type a message") || line.contains("No messages"))
    {
        return Err(CorpusError::Assert(
            "cold-start: ready snapshot missing interactive chrome".to_owned(),
        ));
    }
    quit_cleanly(&mut run)?;
    let artifact = run.finish()?;
    write_artifact(row_label, "cold-start", iteration, &artifact)?;
    Ok(artifact)
}

fn run_wizard(
    iteration: usize,
    row_label: &str,
    row: &RunnerRow,
) -> Result<TranscriptArtifact, CorpusError> {
    let sandbox = create_sandbox()?;
    let launch = launch_env(&sandbox, false, true)?;
    let mut run = ProductRun::open(
        launch,
        Scenario::Wizard,
        row.clone(),
        vec![ClaimClass::Execution],
    )?;
    let frame = run.settle_frame_where(|snap| {
        snap.lines
            .iter()
            .any(|line| line.contains("Choose a theme family"))
            || ready_snapshot_predicate(snap)
    })?;
    if !contains_bytes(run.raw_so_far(), b"Choose a theme family")
        && !frame
            .snapshot
            .lines
            .iter()
            .any(|line| line.contains("Choose a theme family"))
    {
        return Err(CorpusError::Assert(
            "wizard: first-time setup family step marker missing (require PI_EXPERIMENTAL=1, isolated HOME, no PI_CODING_AGENT_DIR, no settings.json)".to_owned(),
        ));
    }
    // Advance family → mode → analytics with Enter selections for deterministic completion.
    for _ in 0..3 {
        run.write_input(KEY_ENTER)?;
        let _ = run.settle_frame_where(|snap| {
            snap.lines.iter().any(|line| {
                line.contains("Choose a theme mode") || line.contains("anonymous usage")
            }) || ready_snapshot_predicate(snap)
        })?;
    }
    quit_cleanly(&mut run)?;
    let artifact = run.finish()?;
    write_artifact(row_label, "wizard", iteration, &artifact)?;
    Ok(artifact)
}

fn run_trust_selector(
    iteration: usize,
    row_label: &str,
    row: &RunnerRow,
) -> Result<TranscriptArtifact, CorpusError> {
    let sandbox = create_sandbox()?;
    seed_trust_requiring_project(&sandbox.work_dir)?;
    let launch = launch_env(&sandbox, true, false)?;
    let mut run = ProductRun::open(
        launch,
        Scenario::TrustSelector,
        row.clone(),
        vec![ClaimClass::Execution],
    )?;
    let _ = run.settle_frame_where(ready_snapshot_predicate)?;

    // Boot trust events: `--approve` overrides trust for the project that has
    // trust-requiring `.pi/settings.json`. Production interactive TrustUi is absent.
    if contains_bytes(run.raw_so_far(), b"Trust project folder?") {
        return Err(CorpusError::Assert(format!(
            "trust-selector: unexpected interactive TrustUi prompt present; {LIMITATION_ABSENT_PRODUCTION_TRUST_UI} expected absent"
        )));
    }
    let _ = LIMITATION_ABSENT_PRODUCTION_TRUST_UI;

    run.send_line("/trust")?;
    let frame = run.settle_frame_where(|snap| {
        snap.lines
            .iter()
            .any(|line| line.contains("Default project trust"))
    })?;
    if !frame
        .snapshot
        .lines
        .iter()
        .any(|line| line.contains("Default project trust"))
        && !contains_bytes(run.raw_so_far(), b"Default project trust")
    {
        return Err(CorpusError::Assert(
            "trust-selector: /trust did not surface Default project trust".to_owned(),
        ));
    }
    run.write_input(KEY_ESCAPE)?;
    let _ = run.settle_frame_where(ready_snapshot_predicate)?;
    quit_cleanly(&mut run)?;
    let artifact = run.finish()?;
    write_artifact(row_label, "trust-selector", iteration, &artifact)?;
    Ok(artifact)
}

fn run_streaming(
    iteration: usize,
    row_label: &str,
    row: &RunnerRow,
) -> Result<TranscriptArtifact, CorpusError> {
    let sandbox = create_sandbox()?;
    let launch = launch_env(&sandbox, true, false)?;
    let mut run = ProductRun::open(
        launch,
        Scenario::Streaming,
        row.clone(),
        vec![ClaimClass::Execution],
    )?;
    let _ = run.settle_frame_where(ready_snapshot_predicate)?;
    run.send_line("verification deterministic stream")?;
    let frame =
        run.settle_frame_where(|snap| snap.lines.iter().any(|line| line.contains(FINAL_MARKER)))?;
    if !contains_bytes(run.raw_so_far(), FINAL_MARKER.as_bytes())
        && !frame
            .snapshot
            .lines
            .iter()
            .any(|line| line.contains(FINAL_MARKER))
    {
        return Err(CorpusError::Assert(format!(
            "streaming: pinned final marker {FINAL_MARKER} missing"
        )));
    }
    if !contains_bytes(run.raw_so_far(), b"verification-chunk-0001") {
        return Err(CorpusError::Assert(
            "streaming: deterministic chunk marker verification-chunk-0001 missing".to_owned(),
        ));
    }
    quit_cleanly(&mut run)?;
    let artifact = run.finish()?;
    write_artifact(row_label, "streaming", iteration, &artifact)?;
    Ok(artifact)
}

fn run_selectors(
    iteration: usize,
    row_label: &str,
    row: &RunnerRow,
) -> Result<TranscriptArtifact, CorpusError> {
    let sandbox = create_sandbox()?;
    let launch = launch_env(&sandbox, true, false)?;
    let mut run = ProductRun::open(
        launch,
        Scenario::Selectors,
        row.clone(),
        vec![ClaimClass::Execution],
    )?;
    let _ = run.settle_frame_where(ready_snapshot_predicate)?;
    // Deterministic seeded turn so /resume has a session entry.
    run.send_line("verification seeded turn")?;
    let _ =
        run.settle_frame_where(|snap| snap.lines.iter().any(|line| line.contains(FINAL_MARKER)))?;

    run.send_line("/model")?;
    let _ = run.settle_frame_where(|snap| {
        snap.lines.iter().any(|line| {
            line.contains(VERIFICATION_MODEL)
                || line.contains(VERIFICATION_PROVIDER)
                || line.contains("model")
        })
    })?;
    run.write_input(KEY_ESCAPE)?;
    let _ = run.settle_frame_where(ready_snapshot_predicate)?;

    run.send_line("/settings")?;
    let _ = run.settle_frame_where(|snap| {
        snap.lines
            .iter()
            .any(|line| line.contains("Theme") || line.contains("Auto-compact"))
    })?;
    run.write_input(KEY_ESCAPE)?;
    let _ = run.settle_frame_where(ready_snapshot_predicate)?;

    run.send_line("/resume")?;
    // Snapshot-only: `settle_frame` ORs `raw_acc`, so the seeded FINAL_MARKER
    // can match before /resume paints. Empty picker text is a hard reject —
    // a seeded active session must produce an entry.
    let _ = run.settle_frame_where(|snap| {
        let visible: Vec<_> = snap
            .lines
            .iter()
            .rev()
            .take(usize::from(snap.geometry.rows))
            .collect();
        let empty = visible
            .iter()
            .any(|line| line.contains("No sessions found"));
        let picker_open = visible.iter().any(|line| line.contains("esc to cancel"));
        let seeded_row = visible
            .iter()
            .any(|line| line.contains("verification seeded"));
        matches!((empty, picker_open, seeded_row), (false, true, true))
    })?;
    run.write_input(KEY_ESCAPE)?;
    let _ = run.settle_frame_where(ready_snapshot_predicate)?;

    quit_cleanly(&mut run)?;
    let artifact = run.finish()?;
    write_artifact(row_label, "selectors", iteration, &artifact)?;
    Ok(artifact)
}

fn run_overlays(
    iteration: usize,
    row_label: &str,
    row: &RunnerRow,
) -> Result<TranscriptArtifact, CorpusError> {
    let sandbox = create_sandbox()?;
    let launch = launch_env(&sandbox, true, false)?;
    let mut run = ProductRun::open(
        launch,
        Scenario::Overlays,
        row.clone(),
        vec![ClaimClass::Execution],
    )?;
    let _ = run.settle_frame_where(ready_snapshot_predicate)?;

    run.send_line("/hotkeys")?;
    let _ = run.settle_frame_where(|snap| {
        snap.lines
            .iter()
            .any(|line| line.contains("Keyboard shortcuts"))
    })?;
    run.write_input(KEY_ESCAPE)?;
    let _ = run.settle_frame_where(ready_snapshot_predicate)?;

    run.send_line("/changelog")?;
    let _ = run.settle_frame_where(|snap| {
        snap.lines.iter().any(|line| {
            line.contains("No changelog entries found")
                || line.contains("# ")
                || line.contains("changelog")
        })
    })?;
    run.write_input(KEY_ESCAPE)?;
    let _ = run.settle_frame_where(ready_snapshot_predicate)?;

    quit_cleanly(&mut run)?;
    let artifact = run.finish()?;
    write_artifact(row_label, "overlays", iteration, &artifact)?;
    Ok(artifact)
}

fn run_product_resize_ladder(
    iteration: usize,
    row_label: &str,
    row: &RunnerRow,
) -> Result<TranscriptArtifact, CorpusError> {
    let sandbox = create_sandbox()?;
    let launch = launch_env(&sandbox, true, false)?;
    let mut run = ProductRun::open(
        launch,
        Scenario::ProductResizeLadder,
        row.clone(),
        vec![ClaimClass::Execution],
    )?;
    let _ = run.settle_frame_where(ready_snapshot_predicate)?;

    for (cols, rows) in RESIZE_LADDER {
        run.resize(cols, rows)?;
        // Contract: the app survives each resize and settles. Ready markers
        // wrap off-viewport at short heights and the <20-col floor blanks
        // the render entirely (the ladder ends at 1x1), so no content
        // predicate holds at every rung — quiescence after the resize
        // repaint is the observable. A crash or hang fails here by ceiling
        // or premature exit; geometry is driver bookkeeping by design.
        let _ = run.settle_frame_where(|_| true)?;
    }
    quit_cleanly(&mut run)?;
    let artifact = run.finish()?;
    write_artifact(row_label, "product-resize-ladder", iteration, &artifact)?;
    Ok(artifact)
}

fn run_product_resize_storm(
    iteration: usize,
    row_label: &str,
    row: &RunnerRow,
) -> Result<TranscriptArtifact, CorpusError> {
    let sandbox = create_sandbox()?;
    let launch = launch_env(&sandbox, true, false)?;
    let mut run = ProductRun::open(
        launch,
        Scenario::ProductResizeStorm,
        row.clone(),
        vec![ClaimClass::Execution],
    )?;
    let _ = run.settle_frame_where(ready_snapshot_predicate)?;
    run.resize_storm(&RESIZE_STORM)?;
    let frame = run.settle_frame_where(ready_snapshot_predicate)?;
    // Survival observation: process still rendering after the storm.
    if frame
        .snapshot
        .lines
        .iter()
        .all(|line| line.trim().is_empty())
        && run.raw_so_far().is_empty()
    {
        return Err(CorpusError::Assert(
            "product-resize-storm: no render output after storm".to_owned(),
        ));
    }
    quit_cleanly(&mut run)?;
    let artifact = run.finish()?;
    write_artifact(row_label, "product-resize-storm", iteration, &artifact)?;
    Ok(artifact)
}

#[expect(
    clippy::panic,
    reason = "test hard-fail: corpus failure is irrecoverable"
)]
fn hard_fail(error: &CorpusError) -> ! {
    panic!("tui product transcript corpus hard-fail: {error}");
}

#[expect(
    clippy::type_complexity,
    reason = "scenario table type is inherently complex; a type alias would be used only in this test"
)]
#[test]
fn tui_product_transcript_corpus_cold_wizard_trust_stream_selectors_overlays_resize() {
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
    ); 8] = [
        ("cold-start", run_cold_start),
        ("wizard", run_wizard),
        ("trust-selector", run_trust_selector),
        ("streaming", run_streaming),
        ("selectors", run_selectors),
        ("overlays", run_overlays),
        ("product-resize-ladder", run_product_resize_ladder),
        ("product-resize-storm", run_product_resize_storm),
    ];

    for (scenario_dir, runner) in scenarios {
        let row_clone = row.clone();
        let label = row_label.clone();
        if let Err(error) = run_scenario_k(scenario_dir, move |iteration| {
            runner(iteration, &label, &row_clone)
        }) {
            hard_fail(&error);
        }
    }
}
