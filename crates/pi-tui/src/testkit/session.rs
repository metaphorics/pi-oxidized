//! Shared reader pumping, quiescence settling, cleanup, and AVT snapshots.

use std::io::{Read, Write};
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use avt::Vt;

use super::profile::CapabilityProfile;
use crate::testkit::driver::{
    DriverError, DriverSession, ExitStatus, Geometry, OutputBatch, RenderSession, SettlePolicy,
    SettledFrame, TerminalSnapshot,
};
use crate::testkit::qemu::QemuUserSmokeSession;
use crate::testkit::transcript::{
    ClaimClass, DriverKind, NormalizationContext, TranscriptArtifact, TranscriptError,
    TranscriptRecorder, TranscriptSpec,
};

/// Failure from either the live terminal driver or transcript construction.
#[derive(Debug, thiserror::Error)]
pub enum RecordingError {
    /// The underlying terminal session operation failed.
    #[error("driver operation failed: {0}")]
    Driver(#[from] DriverError),
    /// Recording or finalizing a canonical transcript event failed.
    #[error("transcript operation failed: {0}")]
    Transcript(#[from] TranscriptError),
    /// The artifact was requested before the child closed successfully.
    #[error("recording cannot finish before the session closes successfully")]
    FinishBeforeClose,
}

/// Couples one live driver session to its canonical transcript recorder.
pub struct RecordingSession<S: DriverSession> {
    session: Option<S>,
    recorder: TranscriptRecorder,
    closed: bool,
}

impl<S: DriverSession> RecordingSession<S> {
    fn from_recorder(
        session: S,
        mut recorder: TranscriptRecorder,
        argv: Vec<String>,
        context: &NormalizationContext,
    ) -> Result<Self, RecordingError> {
        recorder.spawn(argv, context)?;
        Ok(Self {
            session: Some(session),
            recorder,
            closed: false,
        })
    }

    #[cfg(test)]
    fn force_next_seq(&mut self, seq: u32) {
        self.recorder.force_next_seq_for_test(seq);
    }

    /// Writes one input boundary and records it after the driver accepts it.
    ///
    /// # Errors
    ///
    /// Returns [`RecordingError::Driver`] on write failure or
    /// [`RecordingError::Transcript`] if the accepted input cannot be recorded.
    pub fn write(&mut self, bytes: &[u8]) -> Result<(), RecordingError> {
        let session = self.session.as_mut().ok_or(DriverError::Closed)?;
        session.write(bytes)?;
        self.recorder.input(bytes)?;
        Ok(())
    }

    /// Settles one output boundary and records the returned batch as one event.
    ///
    /// # Errors
    ///
    /// Returns [`RecordingError::Driver`] on settle failure or
    /// [`RecordingError::Transcript`] if the successful batch cannot be recorded.
    pub fn read_output<F>(
        &mut self,
        policy: &SettlePolicy,
        predicate: F,
        context: &NormalizationContext,
    ) -> Result<OutputBatch, RecordingError>
    where
        F: FnMut(&[u8]) -> bool,
    {
        let session = self.session.as_mut().ok_or(DriverError::Closed)?;
        let batch = session.read_output(policy, predicate)?;
        self.recorder.output(&[batch.bytes.as_slice()], context)?;
        Ok(batch)
    }

    /// Closes the driver, then records its successful exit boundary.
    ///
    /// # Errors
    ///
    /// Returns [`RecordingError::Driver`] if close fails or the session was
    /// already consumed, and [`RecordingError::Transcript`] if the exit event
    /// cannot be recorded.
    pub fn close(&mut self) -> Result<ExitStatus, RecordingError> {
        let session = self.session.take().ok_or(DriverError::Closed)?;
        let status = session.close()?;
        self.recorder
            .exit(i32::try_from(status.code).ok(), status.success())?;
        self.closed = true;
        Ok(status)
    }

    /// Finalizes the transcript after a successful [`Self::close`].
    ///
    /// # Errors
    ///
    /// Returns [`RecordingError::FinishBeforeClose`] while the session is open
    /// or after close failed, and [`RecordingError::Transcript`] if canonical
    /// encoding fails.
    pub fn finish(self) -> Result<TranscriptArtifact, RecordingError> {
        if !self.closed {
            return Err(RecordingError::FinishBeforeClose);
        }
        Ok(self.recorder.finish()?)
    }
}

fn constrain_qemu_spec(mut spec: TranscriptSpec) -> TranscriptSpec {
    spec.driver_kind = DriverKind::QemuUserSmoke;
    spec.claims
        .retain(|claim| matches!(claim, ClaimClass::Execution | ClaimClass::Protocol));
    spec
}

impl RecordingSession<QemuUserSmokeSession> {
    /// Starts a non-render QEMU recording with claims constrained to observable evidence.
    ///
    /// The driver kind is forced to QEMU smoke, and render-only claims are
    /// discarded before the recorder is constructed.
    ///
    /// # Errors
    ///
    /// Returns [`RecordingError::Transcript`] if the spawn event cannot be recorded.
    pub fn new_qemu(
        session: QemuUserSmokeSession,
        spec: TranscriptSpec,
        argv: Vec<String>,
        context: &NormalizationContext,
    ) -> Result<Self, RecordingError> {
        Self::from_recorder(
            session,
            TranscriptRecorder::new(constrain_qemu_spec(spec)),
            argv,
            context,
        )
    }
}

impl<S: RenderSession> RecordingSession<S> {
    /// Starts recording an already-open render session with the caller's launch identity.
    ///
    /// # Errors
    ///
    /// Returns [`RecordingError::Transcript`] if the spawn event cannot be recorded.
    pub fn new(
        session: S,
        recorder: TranscriptRecorder,
        argv: Vec<String>,
        context: &NormalizationContext,
    ) -> Result<Self, RecordingError> {
        Self::from_recorder(session, recorder, argv, context)
    }

    /// Resizes the live terminal, then records the successful transition.
    ///
    /// # Errors
    ///
    /// Returns [`RecordingError::Driver`] on resize failure or
    /// [`RecordingError::Transcript`] if the transition cannot be recorded.
    pub fn resize(&mut self, cols: u16, rows: u16) -> Result<(), RecordingError> {
        let session = self.session.as_mut().ok_or(DriverError::Closed)?;
        session.resize(cols, rows)?;
        self.recorder.resize(cols, rows)?;
        Ok(())
    }

    /// Applies a resize storm, then records its successful logical boundary.
    ///
    /// # Errors
    ///
    /// Returns [`RecordingError::Driver`] on resize failure or
    /// [`RecordingError::Transcript`] if the storm cannot be recorded.
    pub fn resize_storm(&mut self, sizes: &[(u16, u16)]) -> Result<(), RecordingError> {
        let session = self.session.as_mut().ok_or(DriverError::Closed)?;
        session.resize_storm(sizes)?;
        let geometries = sizes
            .iter()
            .map(|&(cols, rows)| Geometry { cols, rows })
            .collect::<Vec<_>>();
        self.recorder.resize_storm(&geometries)?;
        Ok(())
    }

    /// Settles one output boundary and records its output and visible snapshot.
    ///
    /// # Errors
    ///
    /// Returns [`RecordingError::Driver`] on settle failure or invalid snapshot
    /// coordinates, and [`RecordingError::Transcript`] if either event cannot
    /// be recorded.
    pub fn read_settled_frame<F>(
        &mut self,
        policy: &SettlePolicy,
        predicate: F,
        context: &NormalizationContext,
    ) -> Result<SettledFrame, RecordingError>
    where
        F: FnMut(&[u8]) -> bool,
    {
        let session = self.session.as_mut().ok_or(DriverError::Closed)?;
        let frame = session.read_settled_frame(policy, predicate)?;
        self.record_frame(frame, context)
    }

    /// Settles when the quiet window holds and the snapshot rebuilt from the
    /// full raw log satisfies `predicate`.
    ///
    /// One logical checkpoint yields exactly one Snapshot boundary regardless
    /// of how many quiet windows elapse.
    ///
    /// # Errors
    ///
    /// Returns [`RecordingError::Driver`] on settle failure or invalid snapshot
    /// coordinates, and [`RecordingError::Transcript`] if the event cannot
    /// be recorded.
    pub fn read_settled_frame_where<F>(
        &mut self,
        policy: &SettlePolicy,
        predicate: F,
        context: &NormalizationContext,
    ) -> Result<SettledFrame, RecordingError>
    where
        F: FnMut(&TerminalSnapshot) -> bool,
    {
        let session = self.session.as_mut().ok_or(DriverError::Closed)?;
        let frame = session.read_settled_frame_where(policy, predicate)?;
        self.record_frame(frame, context)
    }

    fn record_frame(
        &mut self,
        frame: SettledFrame,
        context: &NormalizationContext,
    ) -> Result<SettledFrame, RecordingError> {
        Geometry::new(frame.snapshot.geometry.cols, frame.snapshot.geometry.rows)?;
        let cursor_col = u16::try_from(frame.snapshot.cursor_col).map_err(|_| {
            DriverError::InvalidSpec("snapshot cursor column exceeds u16".to_owned())
        })?;
        let cursor_row = u16::try_from(frame.snapshot.cursor_row)
            .map_err(|_| DriverError::InvalidSpec("snapshot cursor row exceeds u16".to_owned()))?;
        self.recorder.output_and_snapshot(
            &[frame.batch.bytes.as_slice()],
            frame.snapshot.geometry.cols,
            frame.snapshot.geometry.rows,
            [cursor_col, cursor_row],
            frame.snapshot.lines.clone(),
            context,
        )?;
        Ok(frame)
    }
}

/// Shared writer that allows multiple owners to write to the same underlying
/// PTY master. Used to give the DSR auto-responder thread its own write handle
/// without calling `take_writer` twice (which portable-pty forbids).
pub(crate) struct SharedWriter {
    inner: Arc<Mutex<Box<dyn Write + Send>>>,
}

impl SharedWriter {
    pub(crate) fn new(writer: Box<dyn Write + Send>) -> Self {
        Self {
            inner: Arc::new(Mutex::new(writer)),
        }
    }

    /// Clones the shared handle for use by another thread.
    pub(crate) fn clone_handle(&self) -> SharedWriter {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl Write for SharedWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .write(buf)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .flush()
    }
}
/// One chunk, or a terminal reader failure that must not validate as settle.
pub(crate) type ReaderEvent = Result<Vec<u8>, std::io::Error>;

/// Byte channel fed by one or more reader threads.
pub(crate) struct ReaderPump {
    rx: Receiver<ReaderEvent>,
    joins: Vec<JoinHandle<()>>,
}
struct ProbeResponder {
    writer: Box<dyn Write + Send>,
    profile: CapabilityProfile,
}

impl ProbeResponder {
    fn respond(&mut self, query: &[u8]) {
        let Some(reply) = self.profile.probe_response(query) else {
            return;
        };
        let _ = self.writer.write_all(reply);
        let _ = self.writer.flush();
    }
}

impl ReaderPump {
    /// Builds a pump from a single readable end.
    pub(crate) fn from_reader<R>(reader: R) -> Self
    where
        R: Read + Send + 'static,
    {
        Self::from_reader_inner(reader, None)
    }

    /// Builds a pump that answers terminal capability queries as they appear.
    pub(crate) fn from_reader_with_probe_responder<R, W>(
        reader: R,
        responder: W,
        profile: CapabilityProfile,
    ) -> Self
    where
        R: Read + Send + 'static,
        W: Write + Send + 'static,
    {
        Self::from_reader_inner(
            reader,
            Some(ProbeResponder {
                writer: Box::new(responder),
                profile,
            }),
        )
    }

    fn from_reader_inner<R>(reader: R, mut responder: Option<ProbeResponder>) -> Self
    where
        R: Read + Send + 'static,
    {
        const QUERIES: [&[u8]; 6] = [
            b"\x1b[?u",
            b"\x1b[c",
            b"\x1b[16t",
            b"\x1b]11;?\x07",
            b"\x1b]11;?\x1b\\",
            b"\x1b[6n",
        ];
        const MAX_QUERY_LEN: usize = 8;
        let (tx, rx) = mpsc::channel();
        let join = thread::spawn(move || {
            let mut reader = reader;
            let mut buf = [0u8; 8192];
            let mut residual: Vec<u8> = Vec::new();
            loop {
                match reader.read(&mut buf) {
                    Ok(0) => break,
                    Ok(n) => {
                        let chunk = &buf[..n];

                        if let Some(probe) = &mut responder {
                            let scan: Vec<u8> =
                                residual.iter().chain(chunk.iter()).copied().collect();
                            for query in QUERIES {
                                let mut idx = 0;
                                while let Some(pos) = scan[idx..]
                                    .windows(query.len())
                                    .position(|window| window == query)
                                {
                                    let abs = idx + pos;
                                    if abs >= residual.len().saturating_sub(query.len() - 1) {
                                        probe.respond(query);
                                    }
                                    idx = abs + query.len();
                                }
                            }
                            let take = chunk.len().min(MAX_QUERY_LEN - 1);
                            residual.clear();
                            residual.extend_from_slice(&chunk[chunk.len() - take..]);
                        }

                        if tx.send(Ok(chunk.to_vec())).is_err() {
                            break;
                        }
                    }
                    Err(err) => {
                        let _ = tx.send(Err(err));
                        break;
                    }
                }
            }
        });
        Self {
            rx,
            joins: vec![join],
        }
    }

    /// Joins every reader thread, surfacing the first panic as an I/O error.
    pub(crate) fn join(&mut self) -> Result<(), DriverError> {
        let mut first_err: Option<DriverError> = None;
        for join in self.joins.drain(..) {
            if let Err(panic) = join.join() {
                let msg = panic_message(&panic);
                if first_err.is_none() {
                    first_err = Some(DriverError::Io(std::io::Error::other(format!(
                        "reader thread panicked: {msg}"
                    ))));
                }
            }
        }
        match first_err {
            Some(err) => Err(err),
            None => Ok(()),
        }
    }
}

fn panic_message(panic: &Box<dyn std::any::Any + Send>) -> String {
    if let Some(msg) = panic.downcast_ref::<&str>() {
        (*msg).to_owned()
    } else if let Some(msg) = panic.downcast_ref::<String>() {
        msg.clone()
    } else {
        "unknown panic payload".to_owned()
    }
}

/// Accumulates raw logs and per-boundary pending bytes.
#[derive(Debug, Default)]
pub(crate) struct OutputLedger {
    raw_log: Vec<u8>,
    pending: Vec<u8>,
}

impl OutputLedger {
    pub(crate) fn push(&mut self, chunk: &[u8]) {
        self.raw_log.extend_from_slice(chunk);
        self.pending.extend_from_slice(chunk);
    }

    pub(crate) fn pending(&self) -> &[u8] {
        &self.pending
    }

    pub(crate) fn raw_log(&self) -> &[u8] {
        &self.raw_log
    }

    pub(crate) fn take_batch(&mut self) -> OutputBatch {
        OutputBatch {
            bytes: std::mem::take(&mut self.pending),
        }
    }
}

/// Shared mutable I/O state for driver sessions.
pub(crate) struct SessionIo {
    pub(crate) writer: Option<Box<dyn Write + Send>>,
    pub(crate) pump: Option<ReaderPump>,
    pub(crate) ledger: OutputLedger,
    pub(crate) closed: bool,
}

impl SessionIo {
    pub(crate) fn new(writer: Box<dyn Write + Send>, pump: ReaderPump) -> Self {
        Self {
            writer: Some(writer),
            pump: Some(pump),
            ledger: OutputLedger::default(),
            closed: false,
        }
    }

    pub(crate) fn write_all(&mut self, bytes: &[u8]) -> Result<(), DriverError> {
        if self.closed {
            return Err(DriverError::Closed);
        }
        let writer = self.writer.as_mut().ok_or(DriverError::Closed)?;
        writer.write_all(bytes)?;
        writer.flush()?;
        Ok(())
    }

    pub(crate) fn read_output<F>(
        &mut self,
        policy: &SettlePolicy,
        mut predicate: F,
    ) -> Result<OutputBatch, DriverError>
    where
        F: FnMut(&[u8]) -> bool,
    {
        self.read_output_where(policy, |ledger| predicate(ledger.pending()))
    }

    pub(crate) fn read_output_where<F>(
        &mut self,
        policy: &SettlePolicy,
        mut predicate: F,
    ) -> Result<OutputBatch, DriverError>
    where
        F: FnMut(&OutputLedger) -> bool,
    {
        if self.closed {
            return Err(DriverError::Closed);
        }
        let pump = self.pump.as_mut().ok_or(DriverError::Closed)?;
        settle_read(&pump.rx, &mut self.ledger, policy, &mut predicate)?;
        Ok(self.ledger.take_batch())
    }

    pub(crate) fn close_writer(&mut self) {
        drop(self.writer.take());
    }

    /// Joins reader threads. Explicit close must propagate panics; Drop may ignore.
    pub(crate) fn join_readers(&mut self) -> Result<(), DriverError> {
        if let Some(mut pump) = self.pump.take() {
            pump.join()
        } else {
            Ok(())
        }
    }
}

impl Drop for SessionIo {
    fn drop(&mut self) {
        // Never join readers here: a still-live child can keep the reader blocked.
        // Session Drop/close must terminate/wait the child before join_readers().
        self.close_writer();
        self.closed = true;
        // Dropping JoinHandle detaches; EOF arrives after the child exits.
        drop(self.pump.take());
    }
}

/// Quiescence-bounded read: quiet window, then predicate against the settled ledger.
pub(crate) fn settle_read<F>(
    rx: &Receiver<ReaderEvent>,
    ledger: &mut OutputLedger,
    policy: &SettlePolicy,
    predicate: &mut F,
) -> Result<(), DriverError>
where
    F: FnMut(&OutputLedger) -> bool,
{
    let started = Instant::now();
    let mut last_data = Instant::now();

    loop {
        let since_data = last_data.elapsed();
        // Predicate evaluation can rebuild a full-log VT snapshot; skip it
        // while output is still arriving (quiet window not yet elapsed).
        let matched = since_data >= policy.quiet && predicate(ledger);
        if matched {
            return Ok(());
        }
        if started.elapsed() >= policy.ceiling {
            let pending = String::from_utf8_lossy(ledger.pending());
            return Err(DriverError::SettleCeiling(format!(
                "last_data_ms={} raw_len={} pending_len={} predicate={matched} pending={pending:?}",
                last_data.elapsed().as_millis(),
                ledger.raw_log().len(),
                ledger.pending().len()
            )));
        }

        let until_ceiling = policy
            .ceiling
            .checked_sub(started.elapsed())
            .unwrap_or(Duration::ZERO);
        let until_quiet = policy
            .quiet
            .checked_sub(since_data)
            .filter(|remaining| !remaining.is_zero())
            .unwrap_or(policy.quiet.max(Duration::from_millis(1)));
        let wait = until_ceiling.min(until_quiet);
        match rx.recv_timeout(wait) {
            Ok(Ok(chunk)) => {
                ledger.push(&chunk);
                last_data = Instant::now();
            }
            Ok(Err(err)) => {
                return Err(DriverError::Io(std::io::Error::new(
                    err.kind(),
                    format!("reader failed before settle: {err}"),
                )));
            }
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => {
                if predicate(ledger) {
                    return Ok(());
                }
                return Err(DriverError::PrematureExit);
            }
        }
    }
}

/// Rebuilds a scrollback-inclusive AVT snapshot from the full raw log.
///
/// Bytes-predicate settles match raw frame text that cell-diff repaints may
/// leave partly outside the live viewport, so assertions may need scrollback.
pub(crate) fn snapshot_from_raw(raw: &[u8], geometry: Geometry) -> TerminalSnapshot {
    build_snapshot(raw, geometry, false)
}

/// Rebuilds a viewport-only AVT snapshot (the live screen a user sees).
///
/// Snapshot-predicate settles must not see stale overlay/editor paints that
/// dismissal and clears scrolled out of the viewport.
pub(crate) fn viewport_snapshot_from_raw(raw: &[u8], geometry: Geometry) -> TerminalSnapshot {
    build_snapshot(raw, geometry, true)
}

fn build_snapshot(raw: &[u8], geometry: Geometry, viewport_only: bool) -> TerminalSnapshot {
    let cols = usize::from(geometry.cols.max(1));
    let rows = usize::from(geometry.rows.max(1));
    let mut vt = Vt::builder()
        .size(cols, rows)
        .scrollback_limit(10_000)
        .build();
    let lossy = String::from_utf8_lossy(raw);
    let _ = vt.feed_str(&lossy);
    let cursor = vt.cursor();
    let lines = if viewport_only {
        // `view()` is the authoritative viewport. Never fall back to
        // `vt.text()` here: it walks the whole buffer, so a blank live
        // viewport would re-import scrolled-out stale paints — exactly
        // what viewport-only settles must not see.
        vt.view()
            .map(|line| line.text().trim_end().to_owned())
            .collect()
    } else {
        let mut all: Vec<String> = vt
            .lines()
            .map(|line| line.text().trim_end().to_owned())
            .collect();
        if all.iter().all(|line| line.trim().is_empty()) {
            all = vt
                .text()
                .into_iter()
                .map(|line| line.trim_end().to_owned())
                .collect();
        }
        all
    };
    TerminalSnapshot {
        geometry,
        cursor_col: cursor.col,
        cursor_row: cursor.row,
        cursor_visible: cursor.visible,
        lines,
    }
}

/// Applies profile env then launch-spec overlays onto a portable-pty command.
pub(crate) fn apply_env(
    cmd: &mut portable_pty::CommandBuilder,
    spec: &crate::testkit::driver::LaunchSpec,
) {
    for (key, value) in spec.profile.env() {
        cmd.env(key, value);
    }
    for (key, value) in &spec.env {
        cmd.env(key, value);
    }
}

/// Applies profile env then launch-spec overlays onto a std Command.
pub(crate) fn apply_std_env(
    cmd: &mut std::process::Command,
    spec: &crate::testkit::driver::LaunchSpec,
) {
    for (key, value) in spec.profile.env() {
        cmd.env(key, value);
    }
    for (key, value) in &spec.env {
        cmd.env(key, value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::CapabilityProfile;
    use crate::testkit::transcript::{
        CanonicalEvent, EventKind, OutputCanon, RowId, RowTier, RunnerRow, Scenario,
        TimingEnvelope, TranscriptMode,
    };

    #[derive(Default)]
    struct FakeDriver {
        fail_write: bool,
        fail_read: bool,
        fail_close: bool,
        writes: Vec<Vec<u8>>,
        output: Vec<u8>,
    }

    impl DriverSession for FakeDriver {
        fn write(&mut self, bytes: &[u8]) -> Result<(), DriverError> {
            if self.fail_write {
                return Err(DriverError::Io(std::io::Error::other("write failed")));
            }
            self.writes.push(bytes.to_vec());
            Ok(())
        }

        fn read_output<F>(
            &mut self,
            _policy: &SettlePolicy,
            mut predicate: F,
        ) -> Result<OutputBatch, DriverError>
        where
            F: FnMut(&[u8]) -> bool,
        {
            if self.fail_read {
                return Err(DriverError::SettleCeiling("fake fail_read".to_owned()));
            }
            if !predicate(&self.output) {
                return Err(DriverError::SettleCeiling("fake predicate miss".to_owned()));
            }
            Ok(OutputBatch {
                bytes: self.output.clone(),
            })
        }

        fn close(self) -> Result<ExitStatus, DriverError> {
            if self.fail_close {
                return Err(DriverError::Closed);
            }
            Ok(ExitStatus::from_code(0))
        }
    }

    struct FakeRender {
        inner: FakeDriver,
        geometry: Geometry,
        fail_resize: bool,
        lines: Vec<String>,
        cursor_col: usize,
        cursor_row: usize,
        chunks: Vec<Vec<u8>>,
        raw_log: Vec<u8>,
    }

    impl Default for FakeRender {
        fn default() -> Self {
            Self {
                inner: FakeDriver::default(),
                geometry: Geometry { cols: 80, rows: 24 },
                fail_resize: false,
                lines: Vec::new(),
                cursor_col: 1,
                cursor_row: 2,
                chunks: Vec::new(),
                raw_log: Vec::new(),
            }
        }
    }

    impl DriverSession for FakeRender {
        fn write(&mut self, bytes: &[u8]) -> Result<(), DriverError> {
            self.inner.write(bytes)
        }

        fn read_output<F>(
            &mut self,
            policy: &SettlePolicy,
            predicate: F,
        ) -> Result<OutputBatch, DriverError>
        where
            F: FnMut(&[u8]) -> bool,
        {
            self.inner.read_output(policy, predicate)
        }

        fn close(self) -> Result<ExitStatus, DriverError> {
            self.inner.close()
        }
    }

    impl RenderSession for FakeRender {
        fn resize(&mut self, cols: u16, rows: u16) -> Result<(), DriverError> {
            if self.fail_resize {
                return Err(DriverError::InvalidSpec("resize failed".to_owned()));
            }
            self.geometry = Geometry::new(cols, rows)?;
            Ok(())
        }

        fn resize_storm(&mut self, sizes: &[(u16, u16)]) -> Result<(), DriverError> {
            for &(cols, rows) in sizes {
                self.resize(cols, rows)?;
            }
            Ok(())
        }

        fn read_settled_frame<F>(
            &mut self,
            policy: &SettlePolicy,
            predicate: F,
        ) -> Result<SettledFrame, DriverError>
        where
            F: FnMut(&[u8]) -> bool,
        {
            let batch = self.read_output(policy, predicate)?;
            Ok(SettledFrame {
                batch,
                snapshot: TerminalSnapshot {
                    geometry: self.geometry,
                    cursor_col: self.cursor_col,
                    cursor_row: self.cursor_row,
                    cursor_visible: true,
                    lines: self.lines.clone(),
                },
            })
        }

        fn read_settled_frame_where<F>(
            &mut self,
            _policy: &SettlePolicy,
            mut predicate: F,
        ) -> Result<SettledFrame, DriverError>
        where
            F: FnMut(&TerminalSnapshot) -> bool,
        {
            let mut drained = Vec::new();
            while !self.chunks.is_empty() {
                let chunk = self.chunks.remove(0);
                drained.extend_from_slice(&chunk);
                self.raw_log.extend_from_slice(&chunk);
                let snapshot = snapshot_from_raw(&self.raw_log, self.geometry);
                if predicate(&snapshot) {
                    return Ok(SettledFrame {
                        batch: OutputBatch { bytes: drained },
                        snapshot,
                    });
                }
            }
            Err(DriverError::SettleCeiling("fake drained".to_owned()))
        }
    }

    fn transcript_spec(driver: DriverKind, claims: Vec<ClaimClass>) -> TranscriptSpec {
        TranscriptSpec {
            scenario: Scenario::FixtureStreamSettle,
            row: RunnerRow {
                tier: RowTier::Local,
                id: RowId::GnuX64,
                runner_image: None,
            },
            geometry: Geometry { cols: 80, rows: 24 },
            capability_profile: CapabilityProfile::Xterm256Color,
            driver_kind: driver,
            mode: TranscriptMode::Standard,
            claims,
            timing: TimingEnvelope::default(),
            output_canon: OutputCanon::Bytes,
        }
    }

    fn recorder(driver: DriverKind) -> TranscriptRecorder {
        TranscriptRecorder::new(transcript_spec(
            driver,
            vec![ClaimClass::Execution, ClaimClass::Render],
        ))
    }

    fn kinds(artifact: &TranscriptArtifact) -> Vec<EventKind> {
        artifact
            .canonical
            .events
            .iter()
            .map(CanonicalEvent::kind)
            .collect()
    }

    #[test]
    // The panic arm IS the assertion: it fires only when the recorder emitted
    // a canonical event other than the expected Output boundary.
    #[expect(
        clippy::panic,
        reason = "panic is the assertion: rejects a non-Output canonical event at index 2"
    )]
    fn driver_only_fake_records_exact_base_sequence() -> Result<(), RecordingError> {
        let context = NormalizationContext::default();
        let mut session = RecordingSession::from_recorder(
            FakeDriver {
                output: b"chunk-a chunk-b".to_vec(),
                ..FakeDriver::default()
            },
            recorder(DriverKind::QemuUserSmoke),
            vec!["pi".to_owned(), "--offline".to_owned()],
            &context,
        )?;
        session.write(b"hello")?;
        let batch = session.read_output(
            &SettlePolicy::default(),
            |bytes| bytes.windows(7).any(|window| window == b"chunk-b"),
            &context,
        )?;
        assert_eq!(batch.bytes, b"chunk-a chunk-b");
        let status = session.close()?;
        assert!(status.success());
        let artifact = session.finish()?;
        assert_eq!(
            kinds(&artifact),
            vec![
                EventKind::Spawn,
                EventKind::Input,
                EventKind::Output,
                EventKind::Exit,
            ]
        );
        assert_eq!(artifact.canonical.events.len(), 4);
        match &artifact.canonical.events[2] {
            CanonicalEvent::Output { bytes_b64, .. } => {
                assert_eq!(
                    bytes_b64,
                    &base64::Engine::encode(
                        &base64::engine::general_purpose::STANDARD,
                        b"chunk-a chunk-b"
                    )
                );
            }
            other => panic!("expected one output boundary, got {other:?}"),
        }
        assert!(!artifact.timing.raw_log_b64.is_empty());
        assert_eq!(artifact.timing.output_audits.len(), 1);
        Ok(())
    }

    #[test]
    fn failed_driver_operations_add_no_events() -> Result<(), RecordingError> {
        let context = NormalizationContext::default();
        let mut session = RecordingSession::from_recorder(
            FakeDriver {
                fail_write: true,
                fail_read: true,
                output: b"ready".to_vec(),
                ..FakeDriver::default()
            },
            recorder(DriverKind::QemuUserSmoke),
            vec!["pi".to_owned()],
            &context,
        )?;
        assert!(matches!(
            session.write(b"nope"),
            Err(RecordingError::Driver(_))
        ));
        assert!(matches!(
            session.read_output(&SettlePolicy::default(), |_| true, &context),
            Err(RecordingError::Driver(_))
        ));
        let status = session.close()?;
        assert!(status.success());
        let artifact = session.finish()?;
        assert_eq!(kinds(&artifact), vec![EventKind::Spawn, EventKind::Exit]);
        Ok(())
    }

    #[test]
    fn finish_requires_successful_close() -> Result<(), RecordingError> {
        let context = NormalizationContext::default();
        let session = RecordingSession::from_recorder(
            FakeDriver::default(),
            recorder(DriverKind::QemuUserSmoke),
            vec!["pi".to_owned()],
            &context,
        )?;
        assert!(matches!(
            session.finish(),
            Err(RecordingError::FinishBeforeClose)
        ));

        let mut session = RecordingSession::from_recorder(
            FakeDriver {
                fail_close: true,
                ..FakeDriver::default()
            },
            recorder(DriverKind::QemuUserSmoke),
            vec!["pi".to_owned()],
            &context,
        )?;
        assert!(matches!(session.close(), Err(RecordingError::Driver(_))));
        assert!(matches!(
            session.finish(),
            Err(RecordingError::FinishBeforeClose)
        ));
        Ok(())
    }

    #[test]
    // The panic arm IS the assertion: it fires only when the recorder emitted
    // a canonical event other than the expected collapsed ResizeStorm.
    #[expect(
        clippy::panic,
        reason = "panic is the assertion: rejects a non-ResizeStorm canonical event at index 5"
    )]
    fn render_fake_records_exact_event_sequence() -> Result<(), RecordingError> {
        let context = NormalizationContext::default();
        let mut session = RecordingSession::new(
            FakeRender {
                inner: FakeDriver {
                    output: b"visible".to_vec(),
                    ..FakeDriver::default()
                },
                geometry: Geometry { cols: 80, rows: 24 },
                lines: vec!["visible".to_owned()],
                ..FakeRender::default()
            },
            recorder(DriverKind::PosixPty),
            vec!["pi".to_owned()],
            &context,
        )?;
        session.write(b"type")?;
        session.read_settled_frame(
            &SettlePolicy::default(),
            |bytes| bytes == b"visible",
            &context,
        )?;
        session.resize(40, 12)?;
        session.resize_storm(&[(30, 10), (20, 8), (10, 4)])?;
        session.close()?;
        let artifact = session.finish()?;
        assert_eq!(
            kinds(&artifact),
            vec![
                EventKind::Spawn,
                EventKind::Input,
                EventKind::Output,
                EventKind::Snapshot,
                EventKind::Resize,
                EventKind::ResizeStorm,
                EventKind::Exit,
            ]
        );
        match &artifact.canonical.events[5] {
            CanonicalEvent::ResizeStorm { sizes, .. } => {
                assert_eq!(sizes, &[Geometry { cols: 10, rows: 4 }]);
            }
            other => panic!("expected collapsed resize storm, got {other:?}"),
        }
        Ok(())
    }

    #[test]
    fn failed_render_operations_add_no_events() -> Result<(), RecordingError> {
        let context = NormalizationContext::default();
        let mut session = RecordingSession::new(
            FakeRender {
                fail_resize: true,
                geometry: Geometry { cols: 80, rows: 24 },
                ..FakeRender::default()
            },
            recorder(DriverKind::PosixPty),
            vec!["pi".to_owned()],
            &context,
        )?;
        assert!(matches!(
            session.resize(40, 12),
            Err(RecordingError::Driver(_))
        ));
        assert!(matches!(
            session.resize_storm(&[(40, 12), (20, 8)]),
            Err(RecordingError::Driver(_))
        ));
        session.close()?;
        let artifact = session.finish()?;
        assert_eq!(kinds(&artifact), vec![EventKind::Spawn, EventKind::Exit]);
        Ok(())
    }

    #[test]
    fn qemu_constructor_strips_render_claims() -> Result<(), RecordingError> {
        let context = NormalizationContext::default();
        let spec = constrain_qemu_spec(transcript_spec(
            DriverKind::PosixPty,
            vec![
                ClaimClass::Execution,
                ClaimClass::Render,
                ClaimClass::Protocol,
                ClaimClass::Snapshot,
                ClaimClass::Pty,
            ],
        ));
        assert_eq!(spec.driver_kind, DriverKind::QemuUserSmoke);
        assert_eq!(
            spec.claims,
            vec![ClaimClass::Execution, ClaimClass::Protocol]
        );
        let mut session = RecordingSession::from_recorder(
            FakeDriver::default(),
            TranscriptRecorder::new(spec),
            vec!["qemu-pi".to_owned()],
            &context,
        )?;
        session.close()?;
        let artifact = session.finish()?;
        assert_eq!(artifact.driver.kind, DriverKind::QemuUserSmoke);
        assert_eq!(artifact.mode, TranscriptMode::Contingency);
        assert_eq!(
            artifact.claims,
            vec![ClaimClass::Execution, ClaimClass::Protocol]
        );
        assert_eq!(kinds(&artifact), vec![EventKind::Spawn, EventKind::Exit]);
        assert!(artifact.timing.output_audits.is_empty());
        assert_eq!(artifact.timing.raw_log_b64, "");
        Ok(())
    }

    #[test]
    fn oversized_snapshot_cursor_records_no_output() -> Result<(), RecordingError> {
        let context = NormalizationContext::default();
        let mut session = RecordingSession::new(
            FakeRender {
                inner: FakeDriver {
                    output: b"visible".to_vec(),
                    ..FakeDriver::default()
                },
                lines: vec!["visible".to_owned()],
                cursor_col: usize::from(u16::MAX) + 1,
                cursor_row: 2,
                ..FakeRender::default()
            },
            recorder(DriverKind::PosixPty),
            vec!["pi".to_owned()],
            &context,
        )?;
        assert!(matches!(
            session.read_settled_frame(
                &SettlePolicy::default(),
                |bytes| bytes == b"visible",
                &context,
            ),
            Err(RecordingError::Driver(_))
        ));
        session.close()?;
        let artifact = session.finish()?;
        assert_eq!(kinds(&artifact), vec![EventKind::Spawn, EventKind::Exit]);
        assert!(artifact.timing.output_audits.is_empty());
        assert_eq!(artifact.timing.raw_log_b64, "");
        Ok(())
    }

    #[test]
    fn settled_frame_seq_overflow_records_no_output_or_snapshot() -> Result<(), RecordingError> {
        let context = NormalizationContext::default();
        for boundary in [u32::MAX - 1, u32::MAX] {
            let mut session = RecordingSession::new(
                FakeRender {
                    inner: FakeDriver {
                        output: b"visible".to_vec(),
                        ..FakeDriver::default()
                    },
                    lines: vec!["visible".to_owned()],
                    ..FakeRender::default()
                },
                recorder(DriverKind::PosixPty),
                vec!["pi".to_owned()],
                &context,
            )?;
            session.force_next_seq(boundary);
            assert!(matches!(
                session.read_settled_frame(
                    &SettlePolicy::default(),
                    |bytes| bytes == b"visible",
                    &context,
                ),
                Err(RecordingError::Transcript(
                    TranscriptError::SequenceOverflow
                ))
            ));
            // Overflow leaves the post-spawn sequence unused; restore it so close
            // can record exit without inventing events from the rejected settle.
            session.force_next_seq(1);
            session.close()?;
            let artifact = session.finish()?;
            assert_eq!(kinds(&artifact), vec![EventKind::Spawn, EventKind::Exit]);
            assert!(artifact.timing.output_audits.is_empty());
            assert_eq!(artifact.timing.raw_log_b64, "");
            assert!(artifact.canonical.normalizations.is_empty());
        }
        Ok(())
    }

    #[test]
    fn settle_read_accepts_preloaded_channel_when_log_len_matches() -> Result<(), DriverError> {
        let (tx, rx) = std::sync::mpsc::channel();
        tx.send(Ok(vec![b'a']))
            .map_err(|error| DriverError::Io(std::io::Error::other(error.to_string())))?;
        tx.send(Ok(vec![b'b']))
            .map_err(|error| DriverError::Io(std::io::Error::other(error.to_string())))?;
        tx.send(Ok(vec![b'c']))
            .map_err(|error| DriverError::Io(std::io::Error::other(error.to_string())))?;
        drop(tx);
        let mut ledger = OutputLedger::default();
        let policy = SettlePolicy::new(Duration::from_millis(1), Duration::from_secs(1))?;
        settle_read(&rx, &mut ledger, &policy, &mut |ledger: &OutputLedger| {
            ledger.raw_log().len() == 3
        })?;
        assert_eq!(ledger.pending(), b"abc");
        Ok(())
    }

    #[test]
    fn settle_read_premature_exit_when_predicate_never_holds() -> Result<(), DriverError> {
        let (tx, rx) = std::sync::mpsc::channel();
        let _ = tx.send(Ok(vec![b'a']));
        drop(tx);
        let mut ledger = OutputLedger::default();
        let policy = SettlePolicy::new(Duration::from_millis(1), Duration::from_secs(1))?;
        let result = settle_read(&rx, &mut ledger, &policy, &mut |_: &OutputLedger| false);
        assert!(matches!(result, Err(DriverError::PrematureExit)));
        Ok(())
    }

    #[test]
    fn settled_frame_where_records_one_snapshot_for_concatenated_chunks()
    -> Result<(), RecordingError> {
        let context = NormalizationContext::default();
        let mut spec = transcript_spec(
            DriverKind::PosixPty,
            vec![ClaimClass::Execution, ClaimClass::Render],
        );
        spec.output_canon = OutputCanon::SettledFrame;
        let mut session = RecordingSession::new(
            FakeRender {
                chunks: vec![b"a".to_vec(), b"b".to_vec(), b"c".to_vec()],
                ..FakeRender::default()
            },
            TranscriptRecorder::new(spec),
            vec!["pi".to_owned()],
            &context,
        )?;
        let frame = session.read_settled_frame_where(
            &SettlePolicy::default(),
            |snap| snap.lines.first().is_some_and(|line| line == "abc"),
            &context,
        )?;
        assert_eq!(frame.batch.bytes, b"abc");
        session.close()?;
        let artifact = session.finish()?;
        assert_eq!(
            kinds(&artifact),
            vec![EventKind::Spawn, EventKind::Snapshot, EventKind::Exit]
        );
        let snapshot = &artifact.canonical.events[1];
        assert!(
            matches!(
                snapshot,
                CanonicalEvent::Snapshot { seq, lines, .. }
                    if *seq == 1 && lines.first().map(String::as_str) == Some("abc")
            ),
            "expected snapshot with seq 1 and line abc, got {snapshot:?}"
        );
        assert_eq!(artifact.timing.output_audits.len(), 1);
        assert_eq!(artifact.timing.output_audits[0].event_seq, 1);
        assert_eq!(artifact.timing.output_audits[0].raw_bytes_b64, "YWJj");
        Ok(())
    }

    /// DSR auto-responder writes `\x1b[1;1R` for each `\x1b[6n` in the stream.
    #[test]
    fn dsr_responder_replies_to_cursor_query() {
        use std::io::Cursor;
        use std::sync::{Arc, Mutex};

        // Input: some text, a DSR query, more text, another DSR query.
        let input = b"hello\x1b[6nworld\x1b[6n!";
        let reader = Cursor::new(input.to_vec());

        let received: Arc<Mutex<Vec<u8>>> = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&received);
        let responder = DsrSink(sink);

        let pump = ReaderPump::from_reader_with_probe_responder(
            reader,
            responder,
            CapabilityProfile::Xterm256Color,
        );

        // Drain all chunks from the channel.
        let mut collected = Vec::new();
        while let Ok(Ok(chunk)) = pump.rx.recv_timeout(Duration::from_secs(1)) {
            collected.extend_from_slice(&chunk);
        }

        // The raw data passes through unchanged.
        assert_eq!(&collected[..], &input[..]);

        // Two DSR replies were written.
        let replies = received
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        assert_eq!(
            &replies[..],
            b"\x1b[1;1R\x1b[1;1R",
            "expected two DSR replies"
        );
    }

    #[test]
    fn probe_responder_waits_for_first_cursor_query() {
        use std::io::Cursor;
        use std::sync::{Arc, Mutex};

        let received = Arc::new(Mutex::new(Vec::new()));
        let input = b"\x1b[?u\x1b[c\x1b[16t\x1b]11;?\x1b\\\x1b[6n";
        let pump = ReaderPump::from_reader_with_probe_responder(
            Cursor::new(input.to_vec()),
            DsrSink(Arc::clone(&received)),
            CapabilityProfile::Xterm256Color,
        );

        while pump.rx.recv_timeout(Duration::from_secs(1)).is_ok() {}

        let replies = received
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        assert_eq!(
            &replies[..],
            b"\x1b[?0u\x1b[?1;2c\x1b[6;10;20t\x1b]11;rgb:0000/0000/0000\x07\x1b[1;1R"
        );
    }

    /// DSR auto-responder handles queries split across chunk boundaries.
    #[test]
    fn dsr_responder_handles_cross_chunk_split() {
        use std::sync::{Arc, Mutex};
        struct SplitReader {
            chunks: Vec<Vec<u8>>,
            idx: usize,
        }
        impl Read for SplitReader {
            fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
                if self.idx >= self.chunks.len() {
                    return Ok(0);
                }
                let chunk = &self.chunks[self.idx];
                let n = chunk.len().min(buf.len());
                buf[..n].copy_from_slice(&chunk[..n]);
                self.idx += 1;
                Ok(n)
            }
        }

        let reader = SplitReader {
            chunks: vec![
                b"abc\x1b".to_vec(), // partial: ESC at end
                b"[6n".to_vec(),     // rest of DSR query
                b"def".to_vec(),     // normal text
            ],
            idx: 0,
        };

        let received: Arc<Mutex<Vec<u8>>> = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&received);
        let responder = DsrSink(sink);

        let pump = ReaderPump::from_reader_with_probe_responder(
            reader,
            responder,
            CapabilityProfile::Xterm256Color,
        );

        let mut collected = Vec::new();
        while let Ok(Ok(chunk)) = pump.rx.recv_timeout(Duration::from_secs(1)) {
            collected.extend_from_slice(&chunk);
        }

        assert_eq!(&collected[..], b"abc\x1b[6ndef");

        let replies = received
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        assert_eq!(
            &replies[..],
            b"\x1b[1;1R",
            "expected one DSR reply for split query"
        );
    }

    /// Writer sink that captures bytes for DSR reply verification.
    struct DsrSink(std::sync::Arc<std::sync::Mutex<Vec<u8>>>);
    impl Write for DsrSink {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
}
