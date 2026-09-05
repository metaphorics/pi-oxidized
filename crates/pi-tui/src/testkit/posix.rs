//! POSIX PTY adapter backed by `portable-pty` `UnixPtySystem`.

#![cfg(unix)]

use std::fs::OpenOptions;
use std::os::fd::AsFd;

use nix::sys::signal::{Signal, killpg};
use nix::sys::termios::{LocalFlags, SetArg, tcgetattr, tcsetattr};
use nix::unistd::{Pid, getpid};
use portable_pty::unix::UnixPtySystem;
use portable_pty::{CommandBuilder, MasterPty, PtySize, PtySystem};

use super::transcript::DriverKind;
use crate::testkit::driver::{
    DriverError, DriverSession, ExitStatus, Geometry, LaunchSpec, OutputBatch, RenderSession,
    SettlePolicy, SettledFrame, TerminalDriver, TerminalSnapshot,
};
use crate::testkit::session::{
    SessionIo, apply_env, snapshot_from_raw, viewport_snapshot_from_raw,
};

/// POSIX PTY driver using `portable-pty`'s Unix backend.
#[derive(Debug, Default, Clone, Copy)]
pub struct PosixPtyDriver;

impl TerminalDriver for PosixPtyDriver {
    type Session = PosixPtySession;

    fn kind(&self) -> DriverKind {
        DriverKind::PosixPty
    }

    fn open(&self, spec: &LaunchSpec) -> Result<Self::Session, DriverError> {
        spec.validate()?;
        let system = UnixPtySystem::default();
        let pair = system
            .openpty(PtySize {
                rows: spec.geometry.rows,
                cols: spec.geometry.cols,
                pixel_width: 0,
                pixel_height: 0,
            })
            .map_err(|err| DriverError::pty(&err))?;

        let mut argv = Vec::with_capacity(spec.argv.len());
        for arg in &spec.argv {
            argv.push(std::ffi::OsString::from(arg));
        }
        let mut cmd = CommandBuilder::from_argv(argv);
        cmd.cwd(&spec.cwd);
        apply_env(&mut cmd, spec);

        disable_pty_echo(pair.master.as_ref())?;
        let child = pair
            .slave
            .spawn_command(cmd)
            .map_err(|err| DriverError::pty(&err))?;
        drop(pair.slave);
        disable_pty_echo(pair.master.as_ref())?;

        let raw_writer = pair
            .master
            .take_writer()
            .map_err(|err| DriverError::pty(&err))?;
        let reader = pair
            .master
            .try_clone_reader()
            .map_err(|err| DriverError::pty(&err))?;

        // Wrap in SharedWriter so the DSR auto-responder can share the single
        // PTY master writer (portable-pty allows only one take_writer call).
        let shared = crate::testkit::session::SharedWriter::new(raw_writer);
        let dsr_writer = shared.clone_handle();

        let pump = crate::testkit::session::ReaderPump::from_reader_with_probe_responder(
            reader,
            dsr_writer,
            spec.profile,
        );
        let pgid = child.process_id().and_then(|id| {
            i32::try_from(id).ok().and_then(|raw| {
                let pid = Pid::from_raw(raw);
                (raw > 1 && pid != getpid()).then_some(pid)
            })
        });
        Ok(PosixPtySession {
            master: pair.master,
            child: Some(child),
            pgid,
            io: SessionIo::new(Box::new(shared), pump),
            geometry: spec.geometry,
        })
    }
}

/// Render-capable POSIX PTY session.
pub struct PosixPtySession {
    master: Box<dyn MasterPty + Send>,
    child: Option<Box<dyn portable_pty::Child + Send + Sync>>,
    pgid: Option<Pid>,
    io: SessionIo,
    geometry: Geometry,
}

impl PosixPtySession {
    fn ensure_open(&self) -> Result<(), DriverError> {
        if self.io.closed {
            Err(DriverError::Closed)
        } else {
            Ok(())
        }
    }

    fn kill_session(&mut self) {
        if let Some(pgid) = self.pgid {
            let _ = killpg(pgid, Signal::SIGKILL);
        }
        if let Some(child) = self.child.as_mut() {
            let _ = child.kill();
        }
    }
}

impl DriverSession for PosixPtySession {
    fn write(&mut self, bytes: &[u8]) -> Result<(), DriverError> {
        self.io.write_all(bytes)
    }

    fn read_output<F>(
        &mut self,
        policy: &SettlePolicy,
        predicate: F,
    ) -> Result<OutputBatch, DriverError>
    where
        F: FnMut(&[u8]) -> bool,
    {
        self.io
            .read_output(policy, predicate)
            .map_err(|error| match error {
                DriverError::SettleCeiling(detail) => {
                    let child = match self.child.as_mut() {
                        Some(child) => match child.try_wait() {
                            Ok(Some(status)) => format!("exited:{status:?}"),
                            Ok(None) => "running".to_owned(),
                            Err(err) => format!("try_wait:{err}"),
                        },
                        None => "missing".to_owned(),
                    };
                    DriverError::SettleCeiling(format!("{detail} child={child}"))
                }
                other => other,
            })
    }

    fn close(mut self) -> Result<ExitStatus, DriverError> {
        self.ensure_open()?;
        // Emit portable-pty's master-EOF stand-in (`\n` + VEOT; see its
        // `UnixMasterWriter::drop`) explicitly before dropping the writer: the
        // DSR auto-responder's shared handle keeps the inner master writer
        // alive past `close_writer`, so its Drop-time stand-in cannot fire
        // until the reader pump is joined — which happens only after
        // `child.wait()`. Serve-mode children wait for the Ctrl+D terminator,
        // so without this a live child deadlocks the wait. Errors (child
        // already exited) are ignored, matching the Drop-time behavior.
        let _ = self.io.write_all(b"\n\x04");
        self.io.closed = true;
        self.io.close_writer();
        let mut child = self.child.take().ok_or(DriverError::Closed)?;
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        let wait_result = loop {
            match child.try_wait() {
                Ok(Some(status)) => break Ok(status),
                Ok(None) if std::time::Instant::now() < deadline => {
                    std::thread::sleep(std::time::Duration::from_millis(50));
                }
                Ok(None) => {
                    self.kill_session();
                    break child.wait();
                }
                Err(err) => break Err(err),
            }
        }
        .map_err(|err| {
            DriverError::Io(std::io::Error::new(
                err.kind(),
                format!("posix pty child wait failed: {err}"),
            ))
        });
        let join_result = self.io.join_readers();
        let status = wait_result?;
        join_result?;
        // Reap leftover session members (extension host) after pi exits.
        self.kill_session();
        Ok(status.into())
    }
}

impl RenderSession for PosixPtySession {
    fn resize(&mut self, cols: u16, rows: u16) -> Result<(), DriverError> {
        self.ensure_open()?;
        let geometry = Geometry::new(cols, rows)?;
        self.master
            .resize(PtySize {
                rows: geometry.rows,
                cols: geometry.cols,
                pixel_width: 0,
                pixel_height: 0,
            })
            .map_err(|err| DriverError::pty(&err))?;
        self.geometry = geometry;
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
        let snapshot = snapshot_from_raw(self.io.ledger.raw_log(), self.geometry);
        Ok(SettledFrame { batch, snapshot })
    }

    fn read_settled_frame_where<F>(
        &mut self,
        policy: &SettlePolicy,
        mut predicate: F,
    ) -> Result<SettledFrame, DriverError>
    where
        F: FnMut(&TerminalSnapshot) -> bool,
    {
        let geometry = self.geometry;
        // Rebuilding the viewport from the full raw log is O(log); the
        // settle loop re-evaluates the predicate on every quiet-window
        // expiry, so cache until new bytes arrive.
        let mut cached_len = usize::MAX;
        let mut cached = viewport_snapshot_from_raw(&[], geometry);
        let batch = self
            .io
            .read_output_where(policy, |ledger| {
                if ledger.raw_log().len() != cached_len {
                    cached_len = ledger.raw_log().len();
                    cached = viewport_snapshot_from_raw(ledger.raw_log(), geometry);
                }
                predicate(&cached)
            })
            .map_err(|error| match error {
                DriverError::SettleCeiling(detail) => {
                    let child = match self.child.as_mut() {
                        Some(child) => match child.try_wait() {
                            Ok(Some(status)) => format!("exited:{status:?}"),
                            Ok(None) => "running".to_owned(),
                            Err(err) => format!("try_wait:{err}"),
                        },
                        None => "missing".to_owned(),
                    };
                    let screen = viewport_snapshot_from_raw(self.io.ledger.raw_log(), geometry)
                        .lines
                        .join("\n");
                    DriverError::SettleCeiling(format!("{detail} child={child} screen:\n{screen}"))
                }
                other => other,
            })?;
        let snapshot = viewport_snapshot_from_raw(self.io.ledger.raw_log(), self.geometry);
        Ok(SettledFrame { batch, snapshot })
    }
}

impl Drop for PosixPtySession {
    fn drop(&mut self) {
        if !self.io.closed {
            self.io.closed = true;
            self.io.close_writer();
        }
        self.kill_session();
        if let Some(mut child) = self.child.take() {
            let _ = child.wait();
        }
        let _ = self.io.join_readers();
    }
}

fn disable_pty_echo(master: &dyn MasterPty) -> Result<(), DriverError> {
    let raw = master.as_raw_fd().ok_or_else(|| {
        DriverError::Pty("posix pty master has no raw fd for echo disable".to_owned())
    })?;
    // Re-open the master descriptor without unsafe FromRawFd.
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(format!("/dev/fd/{raw}"))?;
    let mut termios = tcgetattr(file.as_fd()).map_err(|err| {
        DriverError::Io(std::io::Error::other(format!(
            "tcgetattr failed while disabling echo: {err}"
        )))
    })?;
    termios.local_flags.remove(LocalFlags::ECHO);
    termios.local_flags.remove(LocalFlags::ECHOE);
    termios.local_flags.remove(LocalFlags::ECHOK);
    termios.local_flags.remove(LocalFlags::ECHONL);
    tcsetattr(file.as_fd(), SetArg::TCSANOW, &termios).map_err(|err| {
        DriverError::Io(std::io::Error::other(format!(
            "tcsetattr failed while disabling echo: {err}"
        )))
    })?;
    Ok(())
}
