//! Schema-v1 transcript types, canonical encoding, and recorder normalization.

use std::collections::BTreeSet;

use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};

/// The sole accepted transcript schema identifier.
pub const SCHEMA_ID: &str = "pi-tui-transcript/1";

/// Every normalization available to schema v1, in application order.
pub const NORMALIZATION_TABLE_V1: &[NormalizationKind] = &[
    NormalizationKind::PathHome,
    NormalizationKind::PathCwd,
    NormalizationKind::TimeIso8601,
    NormalizationKind::TimeRelative,
    NormalizationKind::IdSession,
    NormalizationKind::SnapshotTrailingSpaceTrim,
    NormalizationKind::ResizeCollapse,
    NormalizationKind::OutputSettleCollapse,
];

/// A scenario represented by a transcript.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Scenario {
    /// Exercises stream settling against the deterministic fixture.
    FixtureStreamSettle,
    /// Exercises ordered resize behavior against the deterministic fixture.
    FixtureResizeLadder,
    /// Exercises resize coalescing against the deterministic fixture.
    FixtureResizeStorm,
    /// Exercises paste and cursor reporting against the deterministic fixture.
    FixturePasteCursor,
    /// Exercises the extension-UI gauntlet: railed messages, widget slots,
    /// stacked overlays with focus restore, `HostUiRequest` dialogs, extension
    /// footer shortcuts, hostile setTheme sanitization, and OSC 0 title
    /// injection with C0/C1 and >256 UTF-8 bytes.
    FixtureExtGauntlet,
    /// Exercises the two-tier state-matrix corpus (TUI-V1): empty, loading,
    /// retry, queue, streaming, error, focus-marked, and extension-UI states
    /// rendered through the deterministic fixture with per-state quality-bar
    /// checkpoints.
    FixtureStateMatrix,
    /// Records the musl release-row packaging/protocol lane (TUI-V1):
    /// host-native artifact execution, static-link/unpack/integrity, and the
    /// compiled-host plus bundled-Bun-fallback JSONL hello handshakes. Never
    /// carries a PTY, render, synchronized-output, no-clear, or snapshot
    /// claim (validator-enforced via `DriverKind::QemuUserSmoke`).
    MuslPackagingSmoke,
    /// Captures product startup before interactive input.
    ColdStart,
    /// Captures the first-run configuration flow.
    Wizard,
    /// Captures project trust selection.
    TrustSelector,
    /// Captures the project trust confirmation dialog.
    TrustDialog,
    /// Captures incremental model output.
    Streaming,
    /// Captures interactive selection controls.
    Selectors,
    /// Captures layered product surfaces.
    Overlays,
    /// Captures product behavior across ordered resizes.
    ProductResizeLadder,
    /// Captures product behavior while resizes are coalesced.
    ProductResizeStorm,
    /// Exercises the Unicode/width gauntlet (TUI-V3): the 13-probe corpus
    /// across rails, assistant markdown tables, editor cursor, overlay
    /// compositing, and paste-atomic segments; asserts column alignment of
    /// rails and table borders and drift-free cursor placement.
    FixtureUnicodeGauntlet,
    /// Exercises the keyboard and focus gauntlet (TUI-V2): keyboard-only
    /// first-run wizard completion, slash-command selector flows (/login,
    /// /tree, /settings, /import, /logout), per-context ctrl+d resolution
    /// order proofs, streaming interrupt, overlay-over-overlay focus
    /// restore, and rebind reflection in rendered hints — within current
    /// dispatch semantics (no semantics changes; those belong to TUI-G7).
    KeyboardGauntlet,
    /// Exercises the accessibility gauntlet (TUI-V6): transient-notice
    /// persistence across a scripted content change, spinner-status frames carrying
    /// kind + elapsed + cancel hint, and anti-chatter announcement
    /// sequencing per settled stage — the three automated accessibility
    /// invariants over canonical settled content, with the notice urgency
    /// window quarantined as a tolerated measured field.
    FixtureA11yGauntlet,
}

/// Evidence tier for the runner that produced an artifact.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum RowTier {
    /// Evidence produced directly on a supported host.
    Local,
    /// Evidence produced on a pinned CI runner image.
    TierN,
}

/// Closed runner row identity.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum RowId {
    /// GNU/Linux on x86-64.
    GnuX64,
    /// GNU/Linux on `AArch64`.
    GnuArm64,
    /// macOS on x86-64.
    DarwinX64,
    /// macOS on Apple silicon.
    DarwinArm64,
    /// Windows on x86-64.
    WindowsX64,
}

/// Driver implementation used to record an artifact.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum DriverKind {
    /// Records through a POSIX pseudoterminal.
    PosixPty,
    /// Records through the Windows pseudoconsole API.
    #[serde(rename = "conpty")]
    ConPty,
    /// Records execution-only contingency evidence through QEMU user mode.
    QemuUserSmoke,
}

/// Whether an artifact is primary evidence or contingency evidence.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum TranscriptMode {
    /// Primary evidence captured by a native terminal driver.
    Standard,
    /// Reduced evidence captured when a native runner is unavailable.
    Contingency,
}

/// Observable behavior claimed by an artifact.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ClaimClass {
    /// The child launched and completed with the recorded status.
    Execution,
    /// The child exchanged the recorded terminal protocol bytes.
    Protocol,
    /// The exchange occurred through a pseudoterminal.
    Pty,
    /// The artifact includes renderer-derived evidence.
    Render,
    /// Output used synchronized-update semantics.
    SynchronizedOutput,
    /// Rendering avoided destructive screen clears.
    NoClear,
    /// The artifact includes a settled visible-frame snapshot.
    Snapshot,
}

/// Pinned terminal capability profile.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum CapabilityProfile {
    /// xterm-256color with true-color capability advertised.
    Xterm256ColorTruecolor,
    /// Baseline xterm-256color capabilities.
    Xterm256Color,
    /// Minimal terminal without interactive capabilities.
    Dumb,
    /// macOS Terminal-compatible capabilities.
    TerminalApp,
    /// iTerm2-compatible capabilities.
    Iterm2,
    /// Windows Terminal-compatible VT capabilities.
    WindowsTerminalVt,
    /// Conhost profile that denies DEC synchronized-output support.
    ConhostVtDec2026Fallback,
}

/// Closed canonical event discriminants.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum EventKind {
    /// Child process launch boundary.
    Spawn,
    /// Bytes sent to the child.
    Input,
    /// Bytes observed from the child.
    Output,
    /// Settled visible-frame observation.
    Snapshot,
    /// Single terminal resize.
    Resize,
    /// Coalesced terminal resize sequence.
    ResizeStorm,
    /// Child process termination boundary.
    Exit,
}

/// Closed schema-v1 normalization discriminants.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum NormalizationKind {
    /// Replaces the runner's home directory with a stable token.
    PathHome,
    /// Replaces the runner's working directory with a stable token.
    PathCwd,
    /// Replaces ISO-8601 timestamps with a stable token.
    TimeIso8601,
    /// Replaces relative duration text with a stable token.
    TimeRelative,
    /// Replaces session UUIDs with a stable token.
    IdSession,
    /// Removes insignificant trailing snapshot spaces.
    SnapshotTrailingSpaceTrim,
    /// Retains only the terminal size observable after a resize storm.
    ResizeCollapse,
    /// Retains only the frame observable after settle.
    ///
    /// Scoped pins: any [`crate::components::DEFAULT_LOADER_FRAMES`] glyph after
    /// the first becomes `⠋`, and ` <digits>s ·` becomes ` <ELAPSED> ·`. This
    /// kind is applied only by settled-frame recorders and is never detected
    /// from raw bytes by [`detected_volatile_kinds`].
    OutputSettleCollapse,
}

/// Runner identity, excluded from canonical encoding.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct RunnerRow {
    /// Determines whether native or pinned-runner evidence rules apply.
    pub tier: RowTier,
    /// Selects the platform contract used to validate the driver pairing.
    pub id: RowId,
    /// Pins the CI environment for tier-N evidence.
    pub runner_image: Option<String>,
}

/// Terminal dimensions.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct Geometry {
    /// Visible terminal width in character cells.
    pub cols: u16,
    /// Visible terminal height in character cells.
    pub rows: u16,
}

/// Driver metadata.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct DriverDescriptor {
    /// Identifies which transport semantics constrain the artifact.
    pub kind: DriverKind,
}

/// One normalization actually applied to canonical content.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct NormalizationEntry {
    /// Names the pinned transform needed to reproduce canonical bytes.
    pub kind: NormalizationKind,
}

/// Runtime values required by path normalizers. This never enters canonical content.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct NormalizationContext {
    /// Raw home-directory bytes replaced by [`NormalizationKind::PathHome`].
    pub home: Option<Vec<u8>>,
    /// Raw working-directory bytes replaced by [`NormalizationKind::PathCwd`].
    pub cwd: Option<Vec<u8>>,
}

/// A closed canonical event. Sequence numbers start at zero.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
pub enum CanonicalEvent {
    /// Records the normalized command line at process launch.
    Spawn {
        /// Monotonic event position within the transcript.
        seq: u32,
        /// Normalized argument vector, including the program at index zero.
        argv: Vec<String>,
    },
    /// Records one input boundary.
    Input {
        /// Monotonic event position within the transcript.
        seq: u32,
        /// Exact input bytes encoded as standard base64.
        bytes_b64: String,
    },
    /// Records normalized output for one input boundary.
    Output {
        /// Monotonic event position within the transcript.
        seq: u32,
        /// Canonical output bytes encoded as standard base64.
        bytes_b64: String,
    },
    /// Records the visible frame after output reaches quiescence.
    Snapshot {
        /// Monotonic event position within the transcript.
        seq: u32,
        /// Width used to derive the frame.
        cols: u16,
        /// Height used to derive the frame.
        rows: u16,
        /// Zero-based cursor column and row.
        cursor: [u16; 2],
        /// Visible lines after pinned snapshot normalization.
        lines: Vec<String>,
    },
    /// Records one observable terminal-size transition.
    Resize {
        /// Monotonic event position within the transcript.
        seq: u32,
        /// Width after the transition.
        cols: u16,
        /// Height after the transition.
        rows: u16,
    },
    /// Records the final observable size from a burst of resizes.
    ResizeStorm {
        /// Monotonic event position within the transcript.
        seq: u32,
        /// Canonicalized size sequence, normally containing only the final size.
        sizes: Vec<Geometry>,
    },
    /// Records the child termination boundary.
    Exit {
        /// Monotonic event position within the transcript.
        seq: u32,
        /// Numeric child exit code when the platform supplied one.
        code: Option<i32>,
        /// Cross-platform success classification.
        success: bool,
    },
}

impl CanonicalEvent {
    /// Returns this event's logical sequence number.
    #[must_use]
    pub const fn seq(&self) -> u32 {
        match self {
            Self::Spawn { seq, .. }
            | Self::Input { seq, .. }
            | Self::Output { seq, .. }
            | Self::Snapshot { seq, .. }
            | Self::Resize { seq, .. }
            | Self::ResizeStorm { seq, .. }
            | Self::Exit { seq, .. } => *seq,
        }
    }

    /// Returns this event's closed discriminant.
    #[must_use]
    pub const fn kind(&self) -> EventKind {
        match self {
            Self::Spawn { .. } => EventKind::Spawn,
            Self::Input { .. } => EventKind::Input,
            Self::Output { .. } => EventKind::Output,
            Self::Snapshot { .. } => EventKind::Snapshot,
            Self::Resize { .. } => EventKind::Resize,
            Self::ResizeStorm { .. } => EventKind::ResizeStorm,
            Self::Exit { .. } => EventKind::Exit,
        }
    }
}

/// The canonical, digested event document.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct CanonicalDoc {
    /// Ordered canonical events, starting at sequence zero.
    pub events: Vec<CanonicalEvent>,
    /// Distinct normalizations applied while building those events.
    pub normalizations: Vec<NormalizationEntry>,
}

/// Timing for one observed output chunk.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct ChunkTiming {
    /// Canonical output event this chunk contributed to.
    pub event_seq: u32,
    /// Observed byte count before normalization.
    pub byte_len: u64,
    /// Milliseconds since the preceding chunk or boundary.
    pub delta_ms: u64,
}

/// A settle ceiling observation, outside canonical content.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct AbortCeiling {
    /// Configured settle deadline in milliseconds.
    pub ceiling_ms: u64,
    /// Elapsed milliseconds when the deadline fired.
    pub observed_ms: u64,
}

/// Exact runtime values used to normalize one output event.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct NormalizationAuditContext {
    /// Base64 home path used for re-deriving the audited output.
    pub home_b64: Option<String>,
    /// Base64 working directory used for re-deriving the audited output.
    pub cwd_b64: Option<String>,
}

/// Non-canonical evidence from which one canonical output is re-derived.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct OutputAudit {
    /// Output or settled-frame Snapshot sequence this raw evidence reconstructs.
    pub event_seq: u32,
    /// Unnormalized observed bytes encoded as standard base64.
    pub raw_bytes_b64: String,
    /// Exact path context needed to reproduce the normalization.
    pub context: NormalizationAuditContext,
    /// Normalizations applied while producing the canonical output.
    pub applied: Vec<NormalizationEntry>,
}

/// Non-canonical timing and audit data.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct TimingEnvelope {
    /// End-to-end wall time for the recording, outside the digest.
    pub wall_ms: u64,
    /// Per-chunk timing observations excluded from canonical bytes.
    pub chunk_log: Vec<ChunkTiming>,
    /// Quiet-window durations that completed successfully.
    pub settle_windows_ms: Vec<u64>,
    /// Present only when a settle attempt hit its hard ceiling.
    pub abort_ceiling: Option<AbortCeiling>,
    /// Concatenated raw PTY/stdio log encoded as standard base64.
    pub raw_log_b64: String,
    /// Raw-to-canonical reconstruction evidence for every output event
    /// and, under settled-frame canon, every snapshot event.
    pub output_audits: Vec<OutputAudit>,
}

/// Canonical form of each settle boundary.
///
/// This is a recorder construction choice, not an artifact field. Settled-frame
/// artifacts signal the mode by enumerating [`NormalizationKind::OutputSettleCollapse`].
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum OutputCanon {
    /// Schema-v1 Output+Snapshot pairs from raw settle bytes.
    #[default]
    Bytes,
    /// One Snapshot per settle boundary; raw evidence lives in audits.
    SettledFrame,
}

/// A complete schema-v1 transcript artifact.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct TranscriptArtifact {
    /// Must equal [`SCHEMA_ID`] for validators to accept the artifact.
    pub schema: String,
    /// Scenario this recording was produced for.
    pub scenario: Scenario,
    /// Runner metadata excluded from the canonical digest.
    pub row: RunnerRow,
    /// Initial terminal geometry for the recording.
    pub geometry: Geometry,
    /// Capability profile that shaped env and probe replies.
    pub capability_profile: CapabilityProfile,
    /// Transport that captured the recording.
    pub driver: DriverDescriptor,
    /// Whether this is primary or contingency evidence.
    pub mode: TranscriptMode,
    /// Behaviors the recording is allowed to assert.
    pub claims: Vec<ClaimClass>,
    /// Digested event document.
    pub canonical: CanonicalDoc,
    /// SHA-256 digest over the canonical encoding.
    pub digest: String,
    /// Non-canonical timing and audit envelope.
    pub timing: TimingEnvelope,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct CanonicalDigestInput<'a> {
    schema: &'a str,
    scenario: Scenario,
    geometry: Geometry,
    capability_profile: CapabilityProfile,
    driver_kind: DriverKind,
    mode: TranscriptMode,
    claims: &'a [ClaimClass],
    events: &'a [CanonicalEvent],
    applied_normalizations: &'a [NormalizationEntry],
}

/// Serialization or construction failure.
#[derive(Debug, thiserror::Error)]
pub enum TranscriptError {
    /// Canonical encoding failed while building digest input.
    #[error("canonical JSON serialization failed: {0}")]
    Serialization(#[from] serde_json::Error),
    /// More than `u32::MAX` events were appended to one recorder.
    #[error("event sequence overflowed u32")]
    SequenceOverflow,
    /// Settled-frame canon records only `output_and_snapshot` boundaries.
    #[error("settled-frame canon records only output_and_snapshot boundaries")]
    SettledFrameOnly,
}

/// Produces compact deterministic JSON for exactly the digested fields.
///
/// # Errors
///
/// Returns [`TranscriptError::Serialization`] when the digest input cannot be encoded.
pub fn encode_canonical(artifact: &TranscriptArtifact) -> Result<Vec<u8>, TranscriptError> {
    let input = CanonicalDigestInput {
        schema: &artifact.schema,
        scenario: artifact.scenario,
        geometry: artifact.geometry,
        capability_profile: artifact.capability_profile,
        driver_kind: artifact.driver.kind,
        mode: artifact.mode,
        claims: &artifact.claims,
        events: &artifact.canonical.events,
        applied_normalizations: &artifact.canonical.normalizations,
    };
    Ok(serde_json::to_vec(&input)?)
}

/// Computes the schema-v1 SHA-256 digest over canonical encoding.
///
/// # Errors
///
/// Returns [`TranscriptError::Serialization`] when canonical encoding fails.
pub fn digest_canonical(artifact: &TranscriptArtifact) -> Result<String, TranscriptError> {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let digest = Sha256::digest(encode_canonical(artifact)?);
    let mut encoded = String::with_capacity("sha256:".len() + digest.len() * 2);
    encoded.push_str("sha256:");
    for byte in digest {
        encoded.push(HEX[(byte >> 4) as usize] as char);
        encoded.push(HEX[(byte & 0x0f) as usize] as char);
    }
    Ok(encoded)
}

/// Result of applying byte-level schema-v1 normalization.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NormalizedBytes {
    /// Bytes after applying every triggered schema-v1 transform.
    pub bytes: Vec<u8>,
    /// Distinct transforms that changed the input.
    pub applied: Vec<NormalizationEntry>,
}

/// Applies pinned byte-level normalizations before an output event is constructed.
#[must_use]
pub fn normalize_raw_bytes(raw: &[u8], context: &NormalizationContext) -> NormalizedBytes {
    let mut bytes = raw.to_vec();
    let mut applied = BTreeSet::new();

    // Replace the more specific path first when CWD is nested under HOME.
    replace_context(
        &mut bytes,
        context.cwd.as_deref(),
        b"<CWD>",
        NormalizationKind::PathCwd,
        &mut applied,
    );
    replace_context(
        &mut bytes,
        context.home.as_deref(),
        b"<HOME>",
        NormalizationKind::PathHome,
        &mut applied,
    );
    normalize_tokens(&mut bytes, &mut applied);

    NormalizedBytes {
        bytes,
        applied: applied
            .into_iter()
            .map(|kind| NormalizationEntry { kind })
            .collect(),
    }
}

fn replace_context(
    bytes: &mut Vec<u8>,
    needle: Option<&[u8]>,
    replacement: &[u8],
    kind: NormalizationKind,
    applied: &mut BTreeSet<NormalizationKind>,
) {
    if let Some(needle) = needle.filter(|needle| !needle.is_empty())
        && replace_all(bytes, needle, replacement)
    {
        applied.insert(kind);
    }
}

fn replace_all(bytes: &mut Vec<u8>, needle: &[u8], replacement: &[u8]) -> bool {
    let mut output = Vec::with_capacity(bytes.len());
    let mut offset = 0;
    let mut changed = false;
    while let Some(relative) = find_subslice(&bytes[offset..], needle) {
        let index = offset + relative;
        output.extend_from_slice(&bytes[offset..index]);
        output.extend_from_slice(replacement);
        offset = index + needle.len();
        changed = true;
    }
    if changed {
        output.extend_from_slice(&bytes[offset..]);
        *bytes = output;
    }
    changed
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() {
        return None;
    }
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

fn normalize_tokens(bytes: &mut Vec<u8>, applied: &mut BTreeSet<NormalizationKind>) {
    let mut output = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        if let Some((len, kind, replacement)) = match_volatile_at(bytes, index) {
            output.extend_from_slice(replacement);
            applied.insert(kind);
            index += len;
            continue;
        }
        output.push(bytes[index]);
        index += 1;
    }
    *bytes = output;
}

pub(crate) fn detected_volatile_kinds(raw: &[u8]) -> BTreeSet<NormalizationKind> {
    let mut detected = BTreeSet::new();
    let mut index = 0;
    while index < raw.len() {
        if let Some((len, kind, _)) = match_volatile_at(raw, index) {
            detected.insert(kind);
            index += len;
            continue;
        }
        index += 1;
    }
    if has_user_home_path(raw) {
        detected.insert(NormalizationKind::PathHome);
    }
    detected
}

fn has_user_home_path(raw: &[u8]) -> bool {
    find_subslice(raw, b"/home/").is_some()
        || find_subslice(raw, b"/Users/").is_some()
        || find_subslice(raw, br"\Users\").is_some()
}

fn normalize_text(text: &str, context: &NormalizationContext) -> (String, Vec<NormalizationEntry>) {
    let normalized = normalize_raw_bytes(text.as_bytes(), context);
    match String::from_utf8(normalized.bytes) {
        Ok(value) => (value, normalized.applied),
        Err(_) => (text.to_owned(), Vec::new()),
    }
}

fn match_volatile_at(
    bytes: &[u8],
    index: usize,
) -> Option<(usize, NormalizationKind, &'static [u8])> {
    if let Some(len) = match_uuid_at(bytes, index) {
        return Some((len, NormalizationKind::IdSession, b"<SESSION>"));
    }
    if let Some(len) = match_iso8601_at(bytes, index) {
        return Some((len, NormalizationKind::TimeIso8601, b"<TS>"));
    }
    if let Some(len) = match_relative_time_at(bytes, index) {
        return Some((len, NormalizationKind::TimeRelative, b"<AGO>"));
    }
    None
}

fn match_uuid_at(bytes: &[u8], index: usize) -> Option<usize> {
    if index + 36 > bytes.len() {
        return None;
    }
    is_uuid(&bytes[index..index + 36]).then_some(36)
}

fn match_iso8601_at(bytes: &[u8], index: usize) -> Option<usize> {
    if index + 20 > bytes.len()
        || bytes[index + 4] != b'-'
        || bytes[index + 7] != b'-'
        || bytes[index + 10] != b'T'
    {
        return None;
    }
    let mut end = index + 20;
    while end < bytes.len() && !bytes[end].is_ascii_whitespace() {
        end += 1;
    }
    is_iso8601(&bytes[index..end]).then_some(end - index)
}

fn match_relative_time_at(bytes: &[u8], index: usize) -> Option<usize> {
    let mut cursor = index;
    if cursor >= bytes.len() || !bytes[cursor].is_ascii_digit() {
        return None;
    }
    while cursor < bytes.len() && bytes[cursor].is_ascii_digit() {
        cursor += 1;
    }
    let unit_len = if bytes.get(cursor..cursor + 2) == Some(b"ms".as_slice()) {
        2
    } else if matches!(bytes.get(cursor), Some(b's' | b'm' | b'h')) {
        1
    } else {
        return None;
    };
    cursor += unit_len;
    if bytes.get(cursor..cursor + 4) != Some(b" ago".as_slice()) {
        return None;
    }
    cursor += 4;
    Some(cursor - index)
}

fn is_uuid(token: &[u8]) -> bool {
    token.len() == 36
        && [8, 13, 18, 23]
            .into_iter()
            .all(|index| token[index] == b'-')
        && token
            .iter()
            .enumerate()
            .all(|(index, byte)| [8, 13, 18, 23].contains(&index) || byte.is_ascii_hexdigit())
}

fn is_iso8601(token: &[u8]) -> bool {
    token.len() >= 20
        && token.get(4) == Some(&b'-')
        && token.get(7) == Some(&b'-')
        && token.get(10) == Some(&b'T')
        && token.last() == Some(&b'Z')
}

/// Builds events in sequence and enforces driver-specific evidence restrictions.
pub struct TranscriptRecorder {
    artifact: TranscriptArtifact,
    next_seq: u32,
    applied: BTreeSet<NormalizationEntry>,
    raw_log: Vec<u8>,
    output_audits: Vec<OutputAudit>,
    output_canon: OutputCanon,
    context: NormalizationContext,
}

/// Named construction inputs for [`TranscriptRecorder`].
#[derive(Clone, Debug)]
pub struct TranscriptSpec {
    /// Scenario identity stamped onto the finished artifact.
    pub scenario: Scenario,
    /// Runner identity excluded from the digest.
    pub row: RunnerRow,
    /// Initial geometry recorded before any resize events.
    pub geometry: Geometry,
    /// Capability profile stamped onto the finished artifact.
    pub capability_profile: CapabilityProfile,
    /// Transport used for capture; also constrains allowed claims.
    pub driver_kind: DriverKind,
    /// Requested evidence mode; QEMU forces contingency.
    pub mode: TranscriptMode,
    /// Requested claims; QEMU narrows them to the allowed set.
    pub claims: Vec<ClaimClass>,
    /// Initial non-canonical timing envelope.
    pub timing: TimingEnvelope,
    /// Canonical form of each settle boundary. `Bytes` keeps schema-v1
    /// Output+Snapshot pairs.
    pub output_canon: OutputCanon,
}

impl TranscriptRecorder {
    /// Creates an empty recorder. QEMU mode and claims are forcibly narrowed.
    #[must_use]
    pub fn new(spec: TranscriptSpec) -> Self {
        let TranscriptSpec {
            scenario,
            row,
            geometry,
            capability_profile,
            driver_kind,
            mode,
            claims,
            timing,
            output_canon,
        } = spec;
        let (mode, claims) = constrain_driver(driver_kind, mode, claims);
        Self {
            artifact: TranscriptArtifact {
                schema: SCHEMA_ID.to_owned(),
                scenario,
                row,
                geometry,
                capability_profile,
                driver: DriverDescriptor { kind: driver_kind },
                mode,
                claims,
                canonical: CanonicalDoc {
                    events: Vec::new(),
                    normalizations: Vec::new(),
                },
                digest: String::new(),
                timing,
            },
            next_seq: 0,
            applied: BTreeSet::new(),
            raw_log: Vec::new(),
            output_audits: Vec::new(),
            output_canon,
            context: NormalizationContext::default(),
        }
    }

    /// Records the spawn boundary after normalizing argv path tokens.
    ///
    /// # Errors
    ///
    /// Returns [`TranscriptError::SequenceOverflow`] when the event sequence is exhausted.
    pub fn spawn(
        &mut self,
        argv: Vec<String>,
        context: &NormalizationContext,
    ) -> Result<(), TranscriptError> {
        self.context = context.clone();
        let mut normalized_argv = Vec::with_capacity(argv.len());
        for arg in argv {
            let (value, applied) = normalize_text(&arg, context);
            self.applied.extend(applied);
            normalized_argv.push(value);
        }
        let seq = self.take_seq()?;
        self.artifact.canonical.events.push(CanonicalEvent::Spawn {
            seq,
            argv: normalized_argv,
        });
        Ok(())
    }

    /// Records one input boundary. Home and cwd path tokens are replaced
    /// with the same pinned markers used for spawn argv and snapshots.
    ///
    /// # Errors
    ///
    /// Returns [`TranscriptError::SequenceOverflow`] when the event sequence is exhausted.
    pub fn input(&mut self, bytes: &[u8]) -> Result<(), TranscriptError> {
        let normalized = normalize_raw_bytes(bytes, &self.context);
        self.applied.extend(normalized.applied);
        let seq = self.take_seq()?;
        self.artifact.canonical.events.push(CanonicalEvent::Input {
            seq,
            bytes_b64: BASE64.encode(normalized.bytes),
        });
        Ok(())
    }

    /// Normalizes and merges output since the preceding input boundary into one event.
    ///
    /// # Errors
    ///
    /// Returns [`TranscriptError::SequenceOverflow`] when the event sequence is exhausted.
    pub fn output(
        &mut self,
        chunks: &[&[u8]],
        context: &NormalizationContext,
    ) -> Result<(), TranscriptError> {
        if self.output_canon == OutputCanon::SettledFrame {
            return Err(TranscriptError::SettledFrameOnly);
        }
        let raw_len = chunks.iter().map(|chunk| chunk.len()).sum();
        let mut raw = Vec::with_capacity(raw_len);
        for chunk in chunks {
            raw.extend_from_slice(chunk);
        }
        let normalized = normalize_raw_bytes(&raw, context);
        let applied = normalized.applied;
        let audit = OutputAudit {
            event_seq: 0,
            raw_bytes_b64: BASE64.encode(&raw),
            context: NormalizationAuditContext {
                home_b64: context.home.as_ref().map(|value| BASE64.encode(value)),
                cwd_b64: context.cwd.as_ref().map(|value| BASE64.encode(value)),
            },
            applied: applied.clone(),
        };
        let bytes_b64 = BASE64.encode(normalized.bytes);
        let seq = self.take_seq()?;
        let mut audit = audit;
        audit.event_seq = seq;
        self.raw_log.extend_from_slice(&raw);
        self.applied.extend(applied);
        self.output_audits.push(audit);
        self.artifact
            .canonical
            .events
            .push(CanonicalEvent::Output { seq, bytes_b64 });
        Ok(())
    }

    /// Records a settled snapshot. QEMU recorders cannot add render events.
    ///
    /// # Errors
    ///
    /// Returns [`TranscriptError::SequenceOverflow`] when the event sequence is exhausted.
    pub fn snapshot(
        &mut self,
        cols: u16,
        rows: u16,
        cursor: [u16; 2],
        lines: Vec<String>,
        context: &NormalizationContext,
    ) -> Result<bool, TranscriptError> {
        if self.output_canon == OutputCanon::SettledFrame {
            return Err(TranscriptError::SettledFrameOnly);
        }
        if self.artifact.driver.kind == DriverKind::QemuUserSmoke {
            return Ok(false);
        }
        let (normalized_lines, applied) = normalize_snapshot_lines(lines, context);
        self.applied.extend(applied);
        let seq = self.take_seq()?;
        self.artifact
            .canonical
            .events
            .push(CanonicalEvent::Snapshot {
                seq,
                cols,
                rows,
                cursor,
                lines: normalized_lines,
            });
        Ok(true)
    }

    /// Records one settled output boundary and its snapshot as one atomic pair.
    ///
    /// In [`OutputCanon::Bytes`], consecutive sequence numbers are reserved
    /// before any raw log, normalization, audit, or event mutation. QEMU
    /// recorders record only the output event and return `false`, matching
    /// [`Self::snapshot`]. In [`OutputCanon::SettledFrame`], a single sequence
    /// is reserved and only a Snapshot is appended.
    ///
    /// # Errors
    ///
    /// Returns [`TranscriptError::SequenceOverflow`] when a required sequence
    /// number is unavailable, or [`TranscriptError::SettledFrameOnly`] when a
    /// QEMU recorder is constructed in settled-frame canon.
    pub fn output_and_snapshot(
        &mut self,
        chunks: &[&[u8]],
        cols: u16,
        rows: u16,
        cursor: [u16; 2],
        lines: Vec<String>,
        context: &NormalizationContext,
    ) -> Result<bool, TranscriptError> {
        if self.artifact.driver.kind == DriverKind::QemuUserSmoke {
            self.output(chunks, context)?;
            return Ok(false);
        }

        match self.output_canon {
            OutputCanon::Bytes => {
                self.output_and_snapshot_bytes(chunks, cols, rows, cursor, lines, context)
            }
            OutputCanon::SettledFrame => {
                self.output_and_snapshot_settled(chunks, cols, rows, cursor, lines, context)
            }
        }
    }

    fn output_and_snapshot_bytes(
        &mut self,
        chunks: &[&[u8]],
        cols: u16,
        rows: u16,
        cursor: [u16; 2],
        lines: Vec<String>,
        context: &NormalizationContext,
    ) -> Result<bool, TranscriptError> {
        let output_seq = self.next_seq;
        let snapshot_seq = output_seq
            .checked_add(1)
            .ok_or(TranscriptError::SequenceOverflow)?;
        let advanced = snapshot_seq
            .checked_add(1)
            .ok_or(TranscriptError::SequenceOverflow)?;

        let raw_len = chunks.iter().map(|chunk| chunk.len()).sum();
        let mut raw = Vec::with_capacity(raw_len);
        for chunk in chunks {
            raw.extend_from_slice(chunk);
        }
        let normalized = normalize_raw_bytes(&raw, context);
        let (normalized_lines, snapshot_applied) = normalize_snapshot_lines(lines, context);

        self.next_seq = advanced;
        self.raw_log.extend_from_slice(&raw);
        self.applied.extend(normalized.applied.iter().copied());
        self.applied.extend(snapshot_applied);
        self.output_audits.push(OutputAudit {
            event_seq: output_seq,
            raw_bytes_b64: BASE64.encode(&raw),
            context: NormalizationAuditContext {
                home_b64: context.home.as_ref().map(|value| BASE64.encode(value)),
                cwd_b64: context.cwd.as_ref().map(|value| BASE64.encode(value)),
            },
            applied: normalized.applied.clone(),
        });
        self.artifact.canonical.events.push(CanonicalEvent::Output {
            seq: output_seq,
            bytes_b64: BASE64.encode(normalized.bytes),
        });
        self.artifact
            .canonical
            .events
            .push(CanonicalEvent::Snapshot {
                seq: snapshot_seq,
                cols,
                rows,
                cursor,
                lines: normalized_lines,
            });
        Ok(true)
    }

    fn output_and_snapshot_settled(
        &mut self,
        chunks: &[&[u8]],
        cols: u16,
        rows: u16,
        _cursor: [u16; 2],
        lines: Vec<String>,
        context: &NormalizationContext,
    ) -> Result<bool, TranscriptError> {
        let seq = self.take_seq()?;
        let raw_len = chunks.iter().map(|chunk| chunk.len()).sum();
        let mut raw = Vec::with_capacity(raw_len);
        for chunk in chunks {
            raw.extend_from_slice(chunk);
        }
        let enumerated = normalize_raw_bytes(&raw, context).applied;
        let (lines, applied) = settled_frame_lines(lines, context);
        self.raw_log.extend_from_slice(&raw);
        self.applied.extend(enumerated);
        self.applied.extend(applied.iter().copied());
        self.output_audits.push(OutputAudit {
            event_seq: seq,
            raw_bytes_b64: BASE64.encode(&raw),
            context: NormalizationAuditContext {
                home_b64: context.home.as_ref().map(|value| BASE64.encode(value)),
                cwd_b64: context.cwd.as_ref().map(|value| BASE64.encode(value)),
            },
            applied: applied.into_iter().collect(),
        });
        self.artifact
            .canonical
            .events
            .push(CanonicalEvent::Snapshot {
                seq,
                cols,
                rows,
                // AVT hardware cursor row/column jitters across otherwise
                // identical settled frames (Kitty hide/show, reverse-video
                // cell vs terminal cursor). Pin it so k-run digests compare
                // visible lines only.
                cursor: [0, 0],
                lines,
            });
        Ok(true)
    }

    /// Records one resize. Returns `false` without appending for QEMU recorders.
    ///
    /// # Errors
    ///
    /// Returns [`TranscriptError::SequenceOverflow`] when the event sequence is exhausted.
    pub fn resize(&mut self, cols: u16, rows: u16) -> Result<bool, TranscriptError> {
        if self.artifact.driver.kind == DriverKind::QemuUserSmoke {
            return Ok(false);
        }
        let seq = self.take_seq()?;
        self.artifact
            .canonical
            .events
            .push(CanonicalEvent::Resize { seq, cols, rows });
        Ok(true)
    }

    /// Records a resize storm, collapsing intermediate sizes when present.
    ///
    /// # Errors
    ///
    /// Returns [`TranscriptError::SequenceOverflow`] when the event sequence is exhausted.
    pub fn resize_storm(&mut self, sizes: &[Geometry]) -> Result<bool, TranscriptError> {
        if self.artifact.driver.kind == DriverKind::QemuUserSmoke {
            return Ok(false);
        }
        let collapsed = sizes.last().copied().into_iter().collect::<Vec<_>>();
        let collapse = sizes.len() > collapsed.len();
        let seq = self.take_seq()?;
        if collapse {
            self.applied.insert(NormalizationEntry {
                kind: NormalizationKind::ResizeCollapse,
            });
        }
        self.artifact
            .canonical
            .events
            .push(CanonicalEvent::ResizeStorm {
                seq,
                sizes: collapsed,
            });
        Ok(true)
    }

    /// Records the exit boundary for the child process.
    ///
    /// # Errors
    ///
    /// Returns [`TranscriptError::SequenceOverflow`] when the event sequence is exhausted.
    pub fn exit(&mut self, code: Option<i32>, success: bool) -> Result<(), TranscriptError> {
        let seq = self.take_seq()?;
        self.artifact
            .canonical
            .events
            .push(CanonicalEvent::Exit { seq, code, success });
        Ok(())
    }

    /// Finalizes normalization metadata, raw audit bytes, and digest.
    ///
    /// # Errors
    ///
    /// Returns [`TranscriptError::Serialization`] when the final digest cannot be encoded.
    pub fn finish(mut self) -> Result<TranscriptArtifact, TranscriptError> {
        self.artifact.claims = {
            let claims: BTreeSet<_> = self.artifact.claims.into_iter().collect();
            claims.into_iter().collect()
        };
        self.artifact.canonical.normalizations = self.applied.into_iter().collect();
        self.artifact.timing.raw_log_b64 = BASE64.encode(self.raw_log);
        self.artifact.timing.output_audits = self.output_audits;
        self.artifact.digest = digest_canonical(&self.artifact)?;
        Ok(self.artifact)
    }

    fn take_seq(&mut self) -> Result<u32, TranscriptError> {
        let seq = self.next_seq;
        self.next_seq = self
            .next_seq
            .checked_add(1)
            .ok_or(TranscriptError::SequenceOverflow)?;
        Ok(seq)
    }

    #[cfg(test)]
    pub(crate) fn force_next_seq_for_test(&mut self, seq: u32) {
        self.next_seq = seq;
    }
}

pub(crate) fn normalize_snapshot_lines(
    lines: Vec<String>,
    context: &NormalizationContext,
) -> (Vec<String>, BTreeSet<NormalizationEntry>) {
    let mut applied = BTreeSet::new();
    let mut trimmed = false;
    let mut normalized_lines = Vec::with_capacity(lines.len());
    for line in lines {
        let (mut value, line_applied) = normalize_text(&line, context);
        applied.extend(line_applied);
        let len = value.len();
        value.truncate(value.trim_end_matches(' ').len());
        trimmed |= len != value.len();
        normalized_lines.push(value);
    }
    if trimmed {
        applied.insert(NormalizationEntry {
            kind: NormalizationKind::SnapshotTrailingSpaceTrim,
        });
    }
    (normalized_lines, applied)
}

fn pin_elapsed_token(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let mut chars = line.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch != ' ' {
            out.push(ch);
            continue;
        }

        let mut probe = chars.clone();
        let mut saw_digit = false;
        while probe.next_if(char::is_ascii_digit).is_some() {
            saw_digit = true;
        }
        if saw_digit
            && probe.next() == Some('s')
            && probe.next() == Some(' ')
            && probe.next() == Some('·')
        {
            out.push_str(" <ELAPSED> ·");
            chars = probe;
        } else {
            out.push(' ');
        }
    }
    out
}
fn pin_composer_border_leak(line: &str) -> String {
    let trimmed = line.trim();
    if let Some(rest) = trimmed.strip_prefix('❯')
        && rest.chars().all(|ch| ch == ' ' || ch == '─')
    {
        return "❯".to_owned();
    }
    line.to_owned()
}

pub(crate) fn settled_frame_lines(
    lines: Vec<String>,
    context: &NormalizationContext,
) -> (Vec<String>, BTreeSet<NormalizationEntry>) {
    let (mut lines, mut applied) = normalize_snapshot_lines(lines, context);
    if let Some((first, glyphs)) = crate::components::DEFAULT_LOADER_FRAMES.split_first() {
        for line in &mut lines {
            for glyph in glyphs {
                if line.contains(*glyph) {
                    *line = line.replace(*glyph, first);
                }
            }
            *line = pin_composer_border_leak(&pin_elapsed_token(line));
        }
    }
    applied.insert(NormalizationEntry {
        kind: NormalizationKind::OutputSettleCollapse,
    });
    (lines, applied)
}

fn constrain_driver(
    driver: DriverKind,
    mode: TranscriptMode,
    claims: Vec<ClaimClass>,
) -> (TranscriptMode, Vec<ClaimClass>) {
    if driver != DriverKind::QemuUserSmoke {
        return (mode, claims);
    }
    (
        TranscriptMode::Contingency,
        claims
            .into_iter()
            .filter(|claim| matches!(claim, ClaimClass::Execution | ClaimClass::Protocol))
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn recorder(driver: DriverKind) -> TranscriptRecorder {
        recorder_with_canon(driver, OutputCanon::Bytes)
    }

    fn recorder_with_canon(driver: DriverKind, output_canon: OutputCanon) -> TranscriptRecorder {
        TranscriptRecorder::new(TranscriptSpec {
            scenario: Scenario::ColdStart,
            row: RunnerRow {
                tier: RowTier::Local,
                id: RowId::GnuX64,
                runner_image: None,
            },
            geometry: Geometry { cols: 80, rows: 24 },
            capability_profile: CapabilityProfile::Xterm256Color,
            driver_kind: driver,
            mode: TranscriptMode::Standard,
            claims: vec![ClaimClass::Execution, ClaimClass::Render],
            timing: TimingEnvelope::default(),
            output_canon,
        })
    }

    #[test]
    fn unknown_fields_are_rejected() -> Result<(), Box<dyn std::error::Error>> {
        let json = r#"{"cols":80,"rows":24,"extra":true}"#;
        let error = serde_json::from_str::<Geometry>(json)
            .err()
            .ok_or("unknown field unexpectedly accepted")?;
        assert!(error.to_string().contains("unknown field"));
        Ok(())
    }

    #[test]
    fn trust_dialog_scenario_is_distinct_from_trust_selector()
    -> Result<(), Box<dyn std::error::Error>> {
        assert_ne!(Scenario::TrustDialog, Scenario::TrustSelector);
        let encoded = serde_json::to_string(&Scenario::TrustDialog)?;
        assert_eq!(encoded, "\"trust-dialog\"");
        Ok(())
    }

    #[test]
    fn timing_and_runner_identity_do_not_change_canonical_bytes()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut first = complete(recorder(DriverKind::PosixPty))?;
        let before = encode_canonical(&first)?;
        first.row.runner_image = Some("other-image".to_owned());
        first.timing.wall_ms = 99_999;
        first.timing.raw_log_b64 = BASE64.encode(b"different audit bytes");
        first.timing.output_audits.clear();
        assert_eq!(before, encode_canonical(&first)?);
        Ok(())
    }

    #[test]
    fn normalization_happens_before_output_event_and_digest()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut value = recorder(DriverKind::PosixPty);
        value.spawn(vec!["pi".to_owned()], &NormalizationContext::default())?;
        value.output(
            &[b"/home/alice/project 550e8400-e29b-41d4-a716-446655440000"],
            &NormalizationContext {
                home: Some(b"/home/alice".to_vec()),
                cwd: None,
            },
        )?;
        value.exit(Some(0), true)?;
        let artifact = value.finish()?;
        let output = artifact.canonical.events.get(1).ok_or("missing output")?;
        let CanonicalEvent::Output { bytes_b64, .. } = output else {
            return Err("wrong event".into());
        };
        let decoded = BASE64.decode(bytes_b64)?;
        assert_eq!(decoded, b"<HOME>/project <SESSION>");
        assert!(
            artifact
                .canonical
                .normalizations
                .iter()
                .any(|entry| entry.kind == NormalizationKind::PathHome)
        );
        assert!(
            artifact
                .canonical
                .normalizations
                .iter()
                .any(|entry| entry.kind == NormalizationKind::IdSession)
        );
        assert_eq!(artifact.digest, digest_canonical(&artifact)?);
        assert_eq!(artifact.timing.output_audits.len(), 1);
        Ok(())
    }

    #[test]
    fn spawn_argv_and_snapshot_lines_are_normalized() -> Result<(), Box<dyn std::error::Error>> {
        let context = NormalizationContext {
            home: Some(b"/home/alice".to_vec()),
            cwd: Some(b"/home/alice/project".to_vec()),
        };
        let mut value = recorder(DriverKind::PosixPty);
        value.spawn(
            vec!["/home/alice/.cargo/bin/pi".to_owned(), "--cwd".to_owned()],
            &context,
        )?;
        value.input(b"/import /home/alice/project/export.jsonl\r")?;
        value.snapshot(
            80,
            24,
            [0, 0],
            vec!["cwd=/home/alice/project  ".to_owned()],
            &context,
        )?;
        value.exit(Some(0), true)?;
        let artifact = value.finish()?;
        let CanonicalEvent::Spawn { argv, .. } = &artifact.canonical.events[0] else {
            return Err("missing spawn".into());
        };
        assert_eq!(argv[0], "<HOME>/.cargo/bin/pi");
        let CanonicalEvent::Input { bytes_b64, .. } = &artifact.canonical.events[1] else {
            return Err("missing input".into());
        };
        assert_eq!(bytes_b64, &BASE64.encode(b"/import <CWD>/export.jsonl\r"));
        let CanonicalEvent::Snapshot { lines, .. } = &artifact.canonical.events[2] else {
            return Err("missing snapshot".into());
        };
        assert_eq!(lines[0], "cwd=<CWD>");
        assert!(
            artifact
                .canonical
                .normalizations
                .iter()
                .any(|entry| entry.kind == NormalizationKind::PathHome)
        );
        assert!(
            artifact
                .canonical
                .normalizations
                .iter()
                .any(|entry| entry.kind == NormalizationKind::PathCwd)
        );
        assert!(
            artifact
                .canonical
                .normalizations
                .iter()
                .any(|entry| entry.kind == NormalizationKind::SnapshotTrailingSpaceTrim)
        );
        Ok(())
    }

    #[test]
    fn windows_user_paths_are_detected_as_home_volatile() {
        let kinds = detected_volatile_kinds(br"C:\Users\alice\project");
        assert!(kinds.contains(&NormalizationKind::PathHome));
    }

    #[test]
    fn unchanged_bytes_enumerate_no_normalizations() {
        let normalized = normalize_raw_bytes(b"stable output", &NormalizationContext::default());
        assert_eq!(normalized.bytes, b"stable output");
        assert!(normalized.applied.is_empty());
    }

    #[test]
    fn invalid_utf8_bytes_are_preserved_through_normalization() {
        let raw = b"prefix\xff/home/alice\xfe suffix";
        let normalized = normalize_raw_bytes(
            raw,
            &NormalizationContext {
                home: Some(b"/home/alice".to_vec()),
                cwd: None,
            },
        );
        assert_eq!(normalized.bytes, b"prefix\xff<HOME>\xfe suffix");
        assert!(
            normalized
                .applied
                .iter()
                .any(|entry| entry.kind == NormalizationKind::PathHome)
        );
    }

    #[test]
    fn claims_are_sorted_and_deduplicated_before_digest() -> Result<(), Box<dyn std::error::Error>>
    {
        let mut value = TranscriptRecorder::new(TranscriptSpec {
            scenario: Scenario::ColdStart,
            row: RunnerRow {
                tier: RowTier::Local,
                id: RowId::GnuX64,
                runner_image: None,
            },
            geometry: Geometry { cols: 80, rows: 24 },
            capability_profile: CapabilityProfile::Xterm256Color,
            driver_kind: DriverKind::PosixPty,
            mode: TranscriptMode::Standard,
            claims: vec![
                ClaimClass::Snapshot,
                ClaimClass::Execution,
                ClaimClass::Pty,
                ClaimClass::Execution,
                ClaimClass::Render,
            ],
            timing: TimingEnvelope::default(),
            output_canon: OutputCanon::Bytes,
        });
        value.spawn(vec!["pi".to_owned()], &NormalizationContext::default())?;
        value.exit(Some(0), true)?;
        let artifact = value.finish()?;
        assert_eq!(
            artifact.claims,
            vec![
                ClaimClass::Execution,
                ClaimClass::Pty,
                ClaimClass::Render,
                ClaimClass::Snapshot,
            ]
        );
        let mut scrambled = artifact.clone();
        scrambled.claims = vec![
            ClaimClass::Snapshot,
            ClaimClass::Render,
            ClaimClass::Pty,
            ClaimClass::Execution,
            ClaimClass::Execution,
        ];
        assert_ne!(encode_canonical(&artifact)?, encode_canonical(&scrambled)?);
        scrambled.claims = artifact.claims.clone();
        assert_eq!(artifact.digest, digest_canonical(&scrambled)?);
        Ok(())
    }

    #[test]
    fn qemu_builder_forces_contingency_and_non_render_claims()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut value = recorder(DriverKind::QemuUserSmoke);
        value.spawn(vec!["qemu".to_owned()], &NormalizationContext::default())?;
        assert!(!value.snapshot(
            80,
            24,
            [0, 0],
            vec!["frame".to_owned()],
            &NormalizationContext::default(),
        )?);
        assert!(!value.resize(40, 12)?);
        value.exit(Some(0), true)?;
        let artifact = value.finish()?;
        assert_eq!(artifact.mode, TranscriptMode::Contingency);
        assert_eq!(artifact.claims, vec![ClaimClass::Execution]);
        let has_snapshot = artifact
            .canonical
            .events
            .iter()
            .any(|event| event.kind() == EventKind::Snapshot);
        assert!(!has_snapshot);
        Ok(())
    }

    #[test]
    fn relative_time_phrase_is_normalized() {
        let normalized = normalize_raw_bytes(b"done 5ms ago", &NormalizationContext::default());
        assert_eq!(normalized.bytes, b"done <AGO>");
        assert!(
            normalized
                .applied
                .iter()
                .any(|entry| entry.kind == NormalizationKind::TimeRelative)
        );
    }

    #[test]
    fn output_and_snapshot_assigns_consecutive_sequences() -> Result<(), TranscriptError> {
        let mut value = recorder(DriverKind::PosixPty);
        value.spawn(vec!["pi".to_owned()], &NormalizationContext::default())?;
        assert!(value.output_and_snapshot(
            &[b"ready"],
            80,
            24,
            [1, 2],
            vec!["ready  ".to_owned()],
            &NormalizationContext::default(),
        )?);
        assert_eq!(value.next_seq, 3);
        assert_eq!(value.artifact.canonical.events.len(), 3);
        assert_eq!(value.artifact.canonical.events[1].seq(), 1);
        assert_eq!(value.artifact.canonical.events[2].seq(), 2);
        assert_eq!(value.raw_log, b"ready");
        assert_eq!(value.output_audits.len(), 1);
        assert!(
            value
                .applied
                .iter()
                .any(|entry| entry.kind == NormalizationKind::SnapshotTrailingSpaceTrim)
        );
        Ok(())
    }

    #[test]
    fn output_and_snapshot_rejects_max_minus_one_without_mutation() -> Result<(), TranscriptError> {
        let mut value = recorder(DriverKind::PosixPty);
        value.spawn(vec!["pi".to_owned()], &NormalizationContext::default())?;
        let events_before = value.artifact.canonical.events.clone();
        let raw_before = value.raw_log.clone();
        let audits_before = value.output_audits.clone();
        let applied_before = value.applied.clone();
        value.force_next_seq_for_test(u32::MAX - 1);
        // The panic on `Ok` IS the assertion: overflow must be rejected.
        #[expect(
            clippy::expect_used,
            reason = "panic is the assertion: overflow at u32::MAX-1 must be rejected"
        )]
        let error = value
            .output_and_snapshot(
                &[b"/home/alice/ready"],
                80,
                24,
                [0, 0],
                vec!["/home/alice/ready  ".to_owned()],
                &NormalizationContext {
                    home: Some(b"/home/alice".to_vec()),
                    cwd: None,
                },
            )
            .expect_err("expected sequence overflow at u32::MAX-1");
        assert!(matches!(error, TranscriptError::SequenceOverflow));
        assert_eq!(value.next_seq, u32::MAX - 1);
        assert_eq!(value.artifact.canonical.events, events_before);
        assert_eq!(value.raw_log, raw_before);
        assert_eq!(value.output_audits, audits_before);
        assert_eq!(value.applied, applied_before);
        Ok(())
    }

    #[test]
    fn output_and_snapshot_rejects_max_without_mutation() -> Result<(), TranscriptError> {
        let mut value = recorder(DriverKind::PosixPty);
        value.spawn(vec!["pi".to_owned()], &NormalizationContext::default())?;
        let events_before = value.artifact.canonical.events.clone();
        let raw_before = value.raw_log.clone();
        let audits_before = value.output_audits.clone();
        let applied_before = value.applied.clone();
        value.force_next_seq_for_test(u32::MAX);
        // The panic on `Ok` IS the assertion: overflow must be rejected.
        #[expect(
            clippy::expect_used,
            reason = "panic is the assertion: overflow at u32::MAX must be rejected"
        )]
        let error = value
            .output_and_snapshot(
                &[b"/home/alice/ready"],
                80,
                24,
                [0, 0],
                vec!["/home/alice/ready  ".to_owned()],
                &NormalizationContext {
                    home: Some(b"/home/alice".to_vec()),
                    cwd: None,
                },
            )
            .expect_err("expected sequence overflow at u32::MAX");
        assert!(matches!(error, TranscriptError::SequenceOverflow));
        assert_eq!(value.next_seq, u32::MAX);
        assert_eq!(value.artifact.canonical.events, events_before);
        assert_eq!(value.raw_log, raw_before);
        assert_eq!(value.output_audits, audits_before);
        assert_eq!(value.applied, applied_before);
        Ok(())
    }

    fn snapshot_recorder_state(
        value: &TranscriptRecorder,
    ) -> (
        u32,
        Vec<CanonicalEvent>,
        Vec<u8>,
        Vec<OutputAudit>,
        BTreeSet<NormalizationEntry>,
    ) {
        (
            value.next_seq,
            value.artifact.canonical.events.clone(),
            value.raw_log.clone(),
            value.output_audits.clone(),
            value.applied.clone(),
        )
    }

    fn assert_recorder_state_unchanged(
        value: &TranscriptRecorder,
        before: &(
            u32,
            Vec<CanonicalEvent>,
            Vec<u8>,
            Vec<OutputAudit>,
            BTreeSet<NormalizationEntry>,
        ),
    ) {
        let after = snapshot_recorder_state(value);
        assert_eq!(after.0, before.0, "next_seq mutated");
        assert_eq!(after.1, before.1, "canonical events mutated");
        assert_eq!(after.2, before.2, "raw_log mutated");
        assert_eq!(after.3, before.3, "output_audits mutated");
        assert_eq!(after.4, before.4, "applied normalizations mutated");
    }

    #[test]
    fn output_rejects_sequence_overflow_without_mutation() -> Result<(), TranscriptError> {
        let mut value = recorder(DriverKind::PosixPty);
        value.spawn(vec!["pi".to_owned()], &NormalizationContext::default())?;
        // Seed one successful output so overflow must preserve existing audits/raw/applied.
        value.output(
            &[b"prefix"],
            &NormalizationContext {
                home: Some(b"/home/alice".to_vec()),
                cwd: Some(b"/home/alice/project".to_vec()),
            },
        )?;
        value.force_next_seq_for_test(u32::MAX);
        let before = snapshot_recorder_state(&value);
        // The panic on `Ok` IS the assertion: overflow must be rejected.
        #[expect(
            clippy::expect_used,
            reason = "panic is the assertion: output must reject sequence overflow"
        )]
        let error = value
            .output(
                &[b"/home/alice/project/ready"],
                &NormalizationContext {
                    home: Some(b"/home/alice".to_vec()),
                    cwd: Some(b"/home/alice/project".to_vec()),
                },
            )
            .expect_err("expected sequence overflow");
        assert!(matches!(error, TranscriptError::SequenceOverflow));
        assert_recorder_state_unchanged(&value, &before);
        // Mutation-capable payload would have extended raw_log and applied PathHome/PathCwd.
        assert_eq!(value.raw_log, b"prefix");
        assert_eq!(value.output_audits.len(), 1);
        Ok(())
    }

    #[test]
    fn resize_storm_rejects_sequence_overflow_without_mutation() -> Result<(), TranscriptError> {
        let mut value = recorder(DriverKind::PosixPty);
        value.spawn(vec!["pi".to_owned()], &NormalizationContext::default())?;
        assert!(value.resize_storm(&[Geometry { cols: 80, rows: 24 }])?);
        value.force_next_seq_for_test(u32::MAX);
        let before = snapshot_recorder_state(&value);
        // The panic on `Ok` IS the assertion: overflow must be rejected.
        #[expect(
            clippy::expect_used,
            reason = "panic is the assertion: resize_storm must reject sequence overflow"
        )]
        let error = value
            .resize_storm(&[
                Geometry { cols: 40, rows: 12 },
                Geometry { cols: 20, rows: 8 },
                Geometry { cols: 10, rows: 4 },
            ])
            .expect_err("expected sequence overflow");
        assert!(matches!(error, TranscriptError::SequenceOverflow));
        assert_recorder_state_unchanged(&value, &before);
        assert!(
            !value
                .applied
                .iter()
                .any(|entry| entry.kind == NormalizationKind::ResizeCollapse)
        );
        Ok(())
    }

    #[test]
    fn settled_frame_collapses_divergent_raw_to_identical_canonical()
    -> Result<(), Box<dyn std::error::Error>> {
        let geometry = Geometry { cols: 80, rows: 24 };
        let raw_blank_then_ready: &[u8] = b"xxxx\rready";
        let raw_ready: &[u8] = b"ready";
        let snap_a = super::super::session::snapshot_from_raw(raw_blank_then_ready, geometry);
        let snap_b = super::super::session::snapshot_from_raw(raw_ready, geometry);
        assert_eq!(snap_a.lines, snap_b.lines);

        let mut left = recorder_with_canon(DriverKind::PosixPty, OutputCanon::SettledFrame);
        let mut right = recorder_with_canon(DriverKind::PosixPty, OutputCanon::SettledFrame);
        for (recorder, raw, snap) in [
            (&mut left, raw_blank_then_ready, &snap_a),
            (&mut right, raw_ready, &snap_b),
        ] {
            recorder.spawn(vec!["pi".to_owned()], &NormalizationContext::default())?;
            recorder.output_and_snapshot(
                &[raw],
                geometry.cols,
                geometry.rows,
                [
                    u16::try_from(snap.cursor_col).unwrap_or(0),
                    u16::try_from(snap.cursor_row).unwrap_or(0),
                ],
                snap.lines.clone(),
                &NormalizationContext::default(),
            )?;
            recorder.exit(Some(0), true)?;
        }
        let left = left.finish()?;
        let right = right.finish()?;
        assert_eq!(encode_canonical(&left)?, encode_canonical(&right)?);

        let mut left_bytes = recorder(DriverKind::PosixPty);
        let mut right_bytes = recorder(DriverKind::PosixPty);
        for (recorder, raw, snap) in [
            (&mut left_bytes, raw_blank_then_ready, &snap_a),
            (&mut right_bytes, raw_ready, &snap_b),
        ] {
            recorder.spawn(vec!["pi".to_owned()], &NormalizationContext::default())?;
            recorder.output_and_snapshot(
                &[raw],
                geometry.cols,
                geometry.rows,
                [
                    u16::try_from(snap.cursor_col).unwrap_or(0),
                    u16::try_from(snap.cursor_row).unwrap_or(0),
                ],
                snap.lines.clone(),
                &NormalizationContext::default(),
            )?;
            recorder.exit(Some(0), true)?;
        }
        assert_ne!(
            encode_canonical(&left_bytes.finish()?)?,
            encode_canonical(&right_bytes.finish()?)?
        );
        Ok(())
    }

    #[test]
    fn settled_frame_lines_pins_loader_clock_and_marks_collapse() {
        let context = NormalizationContext::default();
        let first = crate::components::DEFAULT_LOADER_FRAMES[0];
        for glyph in crate::components::DEFAULT_LOADER_FRAMES {
            let (lines, applied) =
                settled_frame_lines(vec![format!("{glyph} Working… 1s ·")], &context);
            assert_eq!(lines, vec![format!("{first} Working… <ELAPSED> ·")]);
            assert!(
                applied
                    .iter()
                    .any(|entry| entry.kind == NormalizationKind::OutputSettleCollapse)
            );
        }
        let (lines, applied) = settled_frame_lines(vec![" 12s ·".to_owned()], &context);
        assert_eq!(lines, vec![" <ELAPSED> ·".to_owned()]);
        assert!(
            applied
                .iter()
                .any(|entry| entry.kind == NormalizationKind::OutputSettleCollapse)
        );
        let a11y = " ⠋ Working… 4s · esc to cancel".to_owned();
        let (lines, applied) = normalize_snapshot_lines(vec![a11y.clone()], &context);
        assert_eq!(lines, vec![a11y]);
        assert!(
            !applied
                .iter()
                .any(|entry| entry.kind == NormalizationKind::OutputSettleCollapse)
        );
        assert!(
            !detected_volatile_kinds("⠋ Working… 4s ·".as_bytes())
                .contains(&NormalizationKind::OutputSettleCollapse)
        );
    }

    #[test]
    fn settled_frame_lines_pins_empty_composer_border_leak() {
        let context = NormalizationContext::default();
        let (lines, _) = settled_frame_lines(vec!["❯ ─".to_owned(), "❯".to_owned()], &context);
        assert_eq!(lines, vec!["❯".to_owned(), "❯".to_owned()]);
        let (kept, _) = settled_frame_lines(vec!["❯ hello".to_owned()], &context);
        assert_eq!(kept, vec!["❯ hello".to_owned()]);
    }

    #[test]
    fn settled_frame_rejects_bare_output_and_snapshot() -> Result<(), TranscriptError> {
        let mut value = recorder_with_canon(DriverKind::PosixPty, OutputCanon::SettledFrame);
        value.spawn(vec!["pi".to_owned()], &NormalizationContext::default())?;
        let before = value.artifact.canonical.events.len();
        assert!(matches!(
            value.output(&[b"ready"], &NormalizationContext::default()),
            Err(TranscriptError::SettledFrameOnly)
        ));
        assert!(matches!(
            value.snapshot(
                80,
                24,
                [0, 0],
                vec!["ready".to_owned()],
                &NormalizationContext::default()
            ),
            Err(TranscriptError::SettledFrameOnly)
        ));
        assert_eq!(value.artifact.canonical.events.len(), before);
        Ok(())
    }

    #[test]
    fn settled_output_and_snapshot_consumes_one_seq_keyed_to_snapshot()
    -> Result<(), TranscriptError> {
        let mut value = recorder_with_canon(DriverKind::PosixPty, OutputCanon::SettledFrame);
        value.spawn(vec!["pi".to_owned()], &NormalizationContext::default())?;
        assert_eq!(value.next_seq, 1);
        assert!(value.output_and_snapshot(
            &[b"ready"],
            80,
            24,
            [0, 0],
            vec!["ready".to_owned()],
            &NormalizationContext::default(),
        )?);
        assert_eq!(value.next_seq, 2);
        assert_eq!(value.artifact.canonical.events.len(), 2);
        let event = &value.artifact.canonical.events[1];
        assert!(matches!(event, CanonicalEvent::Snapshot { .. }));
        assert_eq!(event.seq(), 1);
        assert_eq!(value.output_audits.len(), 1);
        assert_eq!(value.output_audits[0].event_seq, event.seq());
        Ok(())
    }

    fn complete(mut value: TranscriptRecorder) -> Result<TranscriptArtifact, TranscriptError> {
        value.spawn(vec!["pi".to_owned()], &NormalizationContext::default())?;
        value.exit(Some(0), true)?;
        value.finish()
    }
}
