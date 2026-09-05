//! Deterministic fixture transcript corpus via shared `RecordingSession`.
//!
//! Checkpoints use content predicates then quiescence — never timer-only waits.
//! Artifacts land under `target/verification/tui-transcripts/<row>/<scenario>/run-{1,2,3}/`.
#![cfg(all(unix, feature = "testkit"))]
#![allow(
    clippy::expect_used,
    clippy::panic,
    clippy::unwrap_used,
    clippy::too_many_lines
)]

use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

use pi_tui::terminal::audit_bytes;
use pi_tui::testkit::driver::{
    Geometry as DriverGeometry, LaunchSpec, SettlePolicy, TerminalDriver,
};
use pi_tui::testkit::posix::PosixPtyDriver;
use pi_tui::testkit::repeat::{RepeatError, run_k};
use pi_tui::testkit::transcript::{
    CapabilityProfile, ClaimClass, DriverKind, Geometry, NormalizationContext, OutputCanon, RowId,
    RowTier, RunnerRow, Scenario, TimingEnvelope, TranscriptArtifact, TranscriptMode,
    TranscriptRecorder, TranscriptSpec,
};
use pi_tui::testkit::{RecordingError, RecordingSession};

const INITIAL_COLS: u16 = 80;
const INITIAL_ROWS: u16 = 24;
const K: usize = 3;
const CURSOR_KEY_COUNT: u32 = 6;

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

/// Atomic 24-size storm matching the fixture's scripted plan shape.
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
            RecordingError::FinishBeforeClose => {
                Self::Assert("recording finish before close".to_owned())
            }
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

type HostSession = <PosixPtyDriver as TerminalDriver>::Session;

/// Scenario driver holding the shared `RecordingSession` (not a local recorder wrapper).
struct FixtureRun {
    recording: RecordingSession<HostSession>,
    context: NormalizationContext,
    policy: SettlePolicy,
    raw_acc: Vec<u8>,
    wall_started: Instant,
    settle_windows_ms: Vec<u64>,
}

impl FixtureRun {
    fn open(
        argv: Vec<String>,
        scenario: Scenario,
        row: RunnerRow,
        claims: Vec<ClaimClass>,
    ) -> Result<Self, CorpusError> {
        require_prerequisites(&argv[0])?;
        let geometry = Geometry {
            cols: INITIAL_COLS,
            rows: INITIAL_ROWS,
        };
        let profile = CapabilityProfile::Xterm256ColorTruecolor;
        let cwd = std::env::current_dir().map_err(|error| {
            CorpusError::Prerequisite(format!("current_dir unavailable: {error}"))
        })?;
        let context = NormalizationContext {
            home: std::env::var_os("HOME").map(|value| value.as_encoded_bytes().to_vec()),
            cwd: Some(cwd.as_os_str().as_encoded_bytes().to_vec()),
        };
        let mut env = BTreeMap::new();
        env.insert("PI_TUI_AUDIT".to_owned(), "1".to_owned());
        env.insert("TERM".to_owned(), "xterm-256color".to_owned());
        env.insert("COLORTERM".to_owned(), "truecolor".to_owned());

        let spec = LaunchSpec {
            argv: argv.clone(),
            cwd,
            env,
            geometry,
            profile,
        };
        let session = PosixPtyDriver.open(&spec)?;
        let recorder = TranscriptRecorder::new(TranscriptSpec {
            scenario,
            row,
            geometry,
            capability_profile: profile,
            driver_kind: DriverKind::PosixPty,
            mode: TranscriptMode::Standard,
            claims,
            timing: TimingEnvelope::default(),
            output_canon: OutputCanon::Bytes,
        });
        let recording = RecordingSession::new(session, recorder, argv, &context)?;
        Ok(Self {
            recording,
            context,
            policy: SettlePolicy::new(Duration::from_millis(120), Duration::from_secs(10))?,
            raw_acc: Vec::new(),
            wall_started: Instant::now(),
            settle_windows_ms: Vec::new(),
        })
    }

    fn write_input(&mut self, bytes: &[u8]) -> Result<(), CorpusError> {
        self.recording.write(bytes)?;
        Ok(())
    }

    fn resize(&mut self, cols: u16, rows: u16) -> Result<(), CorpusError> {
        self.recording.resize(cols, rows)?;
        Ok(())
    }

    fn resize_storm(&mut self, sizes: &[(u16, u16)]) -> Result<(), CorpusError> {
        self.recording.resize_storm(sizes)?;
        Ok(())
    }

    fn settle_output<F>(&mut self, mut predicate: F) -> Result<Vec<u8>, CorpusError>
    where
        F: FnMut(&[u8]) -> bool,
    {
        let started = Instant::now();
        let prior = self.raw_acc.clone();
        let batch = self.recording.read_output(
            &self.policy,
            |bytes| predicate(bytes) || predicate(&merge_acc(&prior, bytes)),
            &self.context,
        )?;
        self.settle_windows_ms
            .push(u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX));
        self.raw_acc.extend_from_slice(&batch.bytes);
        Ok(batch.bytes)
    }

    fn settle_frame<F>(
        &mut self,
        mut predicate: F,
    ) -> Result<pi_tui::testkit::driver::SettledFrame, CorpusError>
    where
        F: FnMut(&[u8]) -> bool,
    {
        let started = Instant::now();
        let prior = self.raw_acc.clone();
        let frame = self.recording.read_settled_frame(
            &self.policy,
            |bytes| predicate(bytes) || predicate(&merge_acc(&prior, bytes)),
            &self.context,
        )?;
        self.settle_windows_ms
            .push(u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX));
        self.raw_acc.extend_from_slice(&frame.batch.bytes);
        Ok(frame)
    }

    /// Settles on the predicate matching the CURRENT batch only — no
    /// merged prior bytes, so persistent chrome (STATUS, EDIT) in earlier
    /// output cannot satisfy a wait for a fresh repaint.
    fn settle_frame_fresh<F>(
        &mut self,
        mut predicate: F,
    ) -> Result<pi_tui::testkit::driver::SettledFrame, CorpusError>
    where
        F: FnMut(&[u8]) -> bool,
    {
        let started = Instant::now();
        let frame = self.recording.read_settled_frame(
            &self.policy,
            |bytes| predicate(bytes),
            &self.context,
        )?;
        self.settle_windows_ms
            .push(u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX));
        self.raw_acc.extend_from_slice(&frame.batch.bytes);
        Ok(frame)
    }

    fn finish(mut self) -> Result<TranscriptArtifact, CorpusError> {
        let status = self.recording.close()?;
        if !status.success() {
            return Err(CorpusError::Assert(format!(
                "fixture exited unsuccessfully: code={} signal={:?}",
                status.code, status.signal
            )));
        }
        let mut artifact = self.recording.finish()?;
        artifact.timing.wall_ms =
            u64::try_from(self.wall_started.elapsed().as_millis()).unwrap_or(u64::MAX);
        artifact.timing.settle_windows_ms = self.settle_windows_ms;
        Ok(artifact)
    }

    fn raw_so_far(&self) -> &[u8] {
        &self.raw_acc
    }
}

fn merge_acc(prefix: &[u8], pending: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(prefix.len() + pending.len());
    out.extend_from_slice(prefix);
    out.extend_from_slice(pending);
    out
}

fn contains_bytes(haystack: &[u8], needle: &[u8]) -> bool {
    haystack
        .windows(needle.len())
        .any(|window| window == needle)
}

fn count_subslice(haystack: &[u8], needle: &[u8]) -> usize {
    if needle.is_empty() || haystack.len() < needle.len() {
        return 0;
    }
    haystack
        .windows(needle.len())
        .filter(|window| *window == needle)
        .count()
}

fn require_prerequisites(fixture: &str) -> Result<(), CorpusError> {
    if !cfg!(unix) {
        return Err(CorpusError::Prerequisite(
            "PosixPtyDriver transcript corpus requires a unix host".to_owned(),
        ));
    }
    let path = Path::new(fixture);
    if !path.exists() {
        return Err(CorpusError::Prerequisite(format!(
            "fixture binary missing: {}",
            path.display()
        )));
    }
    let _ = DriverGeometry::new(1, 1).map_err(|error| {
        CorpusError::Prerequisite(format!("geometry prerequisite failed: {error}"))
    })?;
    Ok(())
}

fn fixture_binary() -> Result<PathBuf, CorpusError> {
    if let Ok(path) = std::env::var("CARGO_BIN_EXE_pi_tui_pty_fixture") {
        let path = PathBuf::from(path);
        if path.exists() {
            return Ok(path);
        }
        return Err(CorpusError::Prerequisite(format!(
            "CARGO_BIN_EXE_pi_tui_pty_fixture points at missing binary: {}",
            path.display()
        )));
    }
    let mut candidates = Vec::new();
    if let Ok(target) = std::env::var("CARGO_TARGET_DIR") {
        candidates.push(PathBuf::from(target));
    }
    candidates.push(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target"));
    candidates.push(PathBuf::from("target"));
    for root in candidates {
        for profile in ["debug", "release"] {
            let path = root.join(profile).join("pi_tui_pty_fixture");
            if path.exists() {
                return Ok(path);
            }
        }
    }
    let status = Command::new("cargo")
        .args([
            "build",
            "-p",
            "pi-tui",
            "--bin",
            "pi_tui_pty_fixture",
            "--quiet",
        ])
        .status()
        .map_err(|error| {
            CorpusError::Prerequisite(format!("fixture build spawn failed: {error}"))
        })?;
    if !status.success() {
        return Err(CorpusError::Prerequisite(
            "fixture build failed; hard-failing transcript corpus prerequisites".to_owned(),
        ));
    }
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/debug/pi_tui_pty_fixture");
    if path.exists() {
        Ok(path)
    } else {
        Err(CorpusError::Prerequisite(format!(
            "fixture binary missing after build at {}",
            path.display()
        )))
    }
}

fn host_row_id() -> RowId {
    match (std::env::consts::OS, std::env::consts::ARCH) {
        ("linux", "aarch64") => RowId::GnuArm64,
        ("macos", "x86_64") => RowId::DarwinX64,
        ("macos", "aarch64") => RowId::DarwinArm64,
        ("windows", _) => RowId::WindowsX64,
        // Unknown hosts (including linux/x86_64) fall back to the gnu x64 row.
        _ => RowId::GnuX64,
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

fn standard_claims() -> Vec<ClaimClass> {
    vec![
        ClaimClass::Execution,
        ClaimClass::Protocol,
        ClaimClass::Pty,
        ClaimClass::Render,
        ClaimClass::SynchronizedOutput,
        ClaimClass::NoClear,
        ClaimClass::Snapshot,
    ]
}

fn fixture_argv(serve: bool) -> Result<Vec<String>, CorpusError> {
    let binary = fixture_binary()?;
    let mut argv = vec![
        binary.to_string_lossy().into_owned(),
        "--exit=success".to_owned(),
    ];
    if serve {
        argv.push("--serve".to_owned());
    }
    Ok(argv)
}

fn assert_no_clear_balanced(raw: &[u8], label: &str) -> Result<(), CorpusError> {
    let audit = audit_bytes(raw);
    if audit.clear_2j != 0 || audit.clear_3j != 0 {
        return Err(CorpusError::Assert(format!(
            "{label}: clear sequences forbidden (2J={}, 3J={})",
            audit.clear_2j, audit.clear_3j
        )));
    }
    if audit.sync_begin != audit.sync_end {
        return Err(CorpusError::Assert(format!(
            "{label}: unbalanced sync markers begin={} end={}",
            audit.sync_begin, audit.sync_end
        )));
    }
    Ok(())
}

fn run_stream_settle(
    iteration: usize,
    row_label: &str,
    row: &RunnerRow,
) -> Result<TranscriptArtifact, CorpusError> {
    let argv = fixture_argv(false)?;
    let mut run = FixtureRun::open(
        argv,
        Scenario::FixtureStreamSettle,
        row.clone(),
        standard_claims(),
    )?;
    let frame = run.settle_frame(|bytes| contains_bytes(bytes, b"DONE-MARKER"))?;
    assert_no_clear_balanced(run.raw_so_far(), "stream-settle")?;
    if !frame
        .snapshot
        .lines
        .iter()
        .any(|line| line.contains("DONE") || line.contains("STATUS") || line.contains("FOOTER"))
    {
        return Err(CorpusError::Assert(
            "stream-settle: settled snapshot missing fixture content".to_owned(),
        ));
    }
    let artifact = run.finish()?;
    write_artifact(row_label, "stream-settle", iteration, &artifact)?;
    Ok(artifact)
}

fn run_resize_ladder(
    iteration: usize,
    row_label: &str,
    row: &RunnerRow,
) -> Result<TranscriptArtifact, CorpusError> {
    let argv = fixture_argv(true)?;
    let mut run = FixtureRun::open(
        argv,
        Scenario::FixtureResizeLadder,
        row.clone(),
        standard_claims(),
    )?;
    let _ = run.settle_output(|bytes| contains_bytes(bytes, b"SERVE-READY"))?;

    for (cols, rows) in RESIZE_LADDER {
        run.resize(cols, rows)?;
        // The ladder descends below the render floor (to 1x1) where the
        // fixture emits no repaint bytes at all, so no fresh-content
        // predicate holds at every rung. The contract mirrors the product
        // ladder: the child survives each resize and settles — a crash
        // fails via PrematureExit, a hang via the ceiling.
        let _ = run.settle_frame_fresh(|_| true)?;
        let _ = (cols, rows);
    }
    assert_no_clear_balanced(run.raw_so_far(), "resize-ladder")?;
    let artifact = run.finish()?;
    write_artifact(row_label, "resize-ladder", iteration, &artifact)?;
    Ok(artifact)
}

fn run_resize_storm(
    iteration: usize,
    row_label: &str,
    row: &RunnerRow,
) -> Result<TranscriptArtifact, CorpusError> {
    let argv = fixture_argv(true)?;
    let mut run = FixtureRun::open(
        argv,
        Scenario::FixtureResizeStorm,
        row.clone(),
        standard_claims(),
    )?;
    let ready = run.settle_output(|bytes| contains_bytes(bytes, b"SERVE-READY"))?;
    let txn_before = count_subslice(&ready, b"PI_TUI_TXN_BEGIN=");

    run.resize_storm(&RESIZE_STORM)?;
    let frame = run.settle_frame(|bytes| contains_bytes(bytes, b"STATUS"))?;
    let txn_after = count_subslice(&frame.batch.bytes, b"PI_TUI_TXN_BEGIN=");
    // Coalesced reanchor discipline: a single stage-3 transaction for the storm.
    if txn_after != 1 {
        return Err(CorpusError::Assert(format!(
            "resize-storm: expected exactly one coalesced reanchor txn after SERVE-READY, saw {txn_after} (pre-ready txns observed in ready batch={txn_before})"
        )));
    }
    assert_no_clear_balanced(run.raw_so_far(), "resize-storm")?;
    let artifact = run.finish()?;
    write_artifact(row_label, "resize-storm", iteration, &artifact)?;
    Ok(artifact)
}

fn run_paste_cursor(
    iteration: usize,
    row_label: &str,
    row: &RunnerRow,
) -> Result<TranscriptArtifact, CorpusError> {
    let argv = fixture_argv(true)?;
    let mut run = FixtureRun::open(
        argv,
        Scenario::FixturePasteCursor,
        row.clone(),
        standard_claims(),
    )?;
    let _ = run.settle_output(|bytes| contains_bytes(bytes, b"SERVE-READY"))?;

    let paste = b"\x1b[200~PASTED-BLOCK-line1\nline2\x1b[201~";
    run.write_input(paste)?;
    let _ = run.settle_output(|bytes| {
        contains_bytes(bytes, b"PASTED-BLOCK") || contains_bytes(bytes, b"paste=")
    })?;
    // Six cursor escape sequences are sent: left, right, up, down, home, end.
    let cursor = b"\x1b[D\x1b[C\x1b[A\x1b[B\x1b[H\x1b[F";
    let before = run.settle_frame(|_| true)?;
    let count_before = before
        .snapshot
        .lines
        .iter()
        .find_map(|line| line.split("cursor=").nth(1)?.split_whitespace().next())
        .and_then(|value| value.parse::<u32>().ok())
        .ok_or_else(|| {
            CorpusError::Assert("paste-cursor: cursor counter absent pre-keys".to_owned())
        })?;
    run.write_input(cursor)?;
    // Cell-diff repaints emit only the changed digit cells, so no contiguous
    // fresh marker exists; settle on quiet and prove all six cursor keys are
    // decoded: the counter must advance by exactly the six escape sequences sent.
    let after = run.settle_frame(|_| true)?;
    let count_after = after
        .snapshot
        .lines
        .iter()
        .find_map(|line| line.split("cursor=").nth(1)?.split_whitespace().next())
        .and_then(|value| value.parse::<u32>().ok())
        .ok_or_else(|| {
            CorpusError::Assert("paste-cursor: cursor counter absent post-keys".to_owned())
        })?;
    let expected_count = count_before.checked_add(CURSOR_KEY_COUNT).ok_or_else(|| {
        CorpusError::Assert(format!(
            "paste-cursor: cursor counter {count_before} + {CURSOR_KEY_COUNT} keys would overflow"
        ))
    })?;
    if count_after != expected_count {
        return Err(CorpusError::Assert(format!(
            "paste-cursor: cursor counter expected {expected_count}, observed {count_after} \
             ({count_before} before {CURSOR_KEY_COUNT} keys)"
        )));
    }
    if !after
        .snapshot
        .lines
        .iter()
        .any(|line| line.contains("paste=") && line.contains("cursor="))
    {
        return Err(CorpusError::Assert(
            "paste-cursor: status row counters missing after paste/cursor".to_owned(),
        ));
    }
    assert_no_clear_balanced(run.raw_so_far(), "paste-cursor")?;
    let artifact = run.finish()?;
    write_artifact(row_label, "paste-cursor", iteration, &artifact)?;
    Ok(artifact)
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

#[test]
fn transcript_fixture_corpus_stream_settle_resize_storm_paste_cursor() {
    let (row_label, row) = resolve_row().unwrap_or_else(|error| {
        panic!("hard-fail harness prerequisites / row config: {error}");
    });
    // Never label an absent-tier local row as Tier N.
    if row_label == "local" {
        assert_eq!(row.tier, RowTier::Local);
    }

    let row_settle = row.clone();
    let label_settle = row_label.clone();
    run_scenario_k("stream-settle", move |iteration| {
        run_stream_settle(iteration, &label_settle, &row_settle)
    })
    .unwrap_or_else(|error| panic!("{error}"));

    let row_ladder = row.clone();
    let label_ladder = row_label.clone();
    run_scenario_k("resize-ladder", move |iteration| {
        run_resize_ladder(iteration, &label_ladder, &row_ladder)
    })
    .unwrap_or_else(|error| panic!("{error}"));

    let row_storm = row.clone();
    let label_storm = row_label.clone();
    run_scenario_k("resize-storm", move |iteration| {
        run_resize_storm(iteration, &label_storm, &row_storm)
    })
    .unwrap_or_else(|error| panic!("{error}"));

    let row_paste = row;
    let label_paste = row_label;
    run_scenario_k("paste-cursor", move |iteration| {
        run_paste_cursor(iteration, &label_paste, &row_paste)
    })
    .unwrap_or_else(|error| panic!("{error}"));
}

#[test]
fn fixture_run_rejects_unsuccessful_child_exit() {
    let (_, row) = resolve_row().unwrap_or_else(|error| {
        panic!("hard-fail harness prerequisites / row config: {error}");
    });
    let mut argv = fixture_argv(false).expect("fixture argv");
    let exit = argv
        .iter_mut()
        .find(|arg| arg.as_str() == "--exit=success")
        .expect("fixture exit argument");
    *exit = "--exit=abort".to_owned();
    let mut run = FixtureRun::open(argv, Scenario::FixtureStreamSettle, row, standard_claims())
        .expect("fixture run");
    let _ = run
        .settle_output(|_| true)
        .expect("final fixture output may settle before exit status is checked");
    let error = run
        .finish()
        .expect_err("unsuccessful fixture exit must reject the transcript");
    assert!(
        error.to_string().contains("fixture exited unsuccessfully"),
        "unexpected error: {error}"
    );
}
