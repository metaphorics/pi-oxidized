//! Cross-process exclusive lock using the `proper-lockfile` mkdir protocol.
//!
//! Compatibility target: npm `proper-lockfile@4.1.2` as used by coding-agent
//! settings, trust, and auth storage.
//!
//! Protocol:
//! - lock artifact is the directory ``${target}.lock`` (or an explicit
//!   `lockfile_path`), created with atomic `create_dir`
//! - `realpath: false` — the target path is made absolute without requiring
//!   the target to exist and without resolving symlinks
//! - contention (`EEXIST` while the lock is still fresh) is the only
//!   condition that is retried; other I/O errors propagate immediately
//! - default sync policy matches the coding-agent wrappers: 10 attempts with
//!   a 20 ms delay between attempts
//! - default stale threshold is 10 s; while held, this guard refreshes the
//!   lock directory mtime on an interval of `stale / 2` so a live holder is
//!   not stolen
//! - release removes only the directory this guard created, and only while
//!   that directory is still the same filesystem identity captured at acquire
//!
//! Callers that only need a read of a missing file (for example settings load
//! when no file exists yet) stay lock-free themselves; this module never
//! invents that policy.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant, SystemTime};

use filetime::{FileTime, set_file_mtime};
use same_file::Handle;
use thiserror::Error;

/// Default acquisition attempts used by coding-agent settings/trust/auth.
pub const DEFAULT_ATTEMPTS: u32 = 10;

/// Delay between contention retries in the default sync policy.
pub const DEFAULT_RETRY_DELAY: Duration = Duration::from_millis(20);

/// Duration after which an un-refreshed lock directory is treated as stale.
///
/// Matches `proper-lockfile`'s default `stale` duration of 10 seconds.
pub const DEFAULT_STALE: Duration = Duration::from_secs(10);

/// Default mtime refresh interval (`stale / 2`) of 5 seconds.
pub const DEFAULT_UPDATE: Duration = Duration::from_secs(5);

/// Configurable acquisition and ownership policy.
///
/// Production callers use [`LockOptions::default`]. Focused tests shorten
/// `attempts`, `retry_delay`, `stale`, and `update` so they stay within the
/// 10 × 20 ms production sleep budget without an injectable clock.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LockOptions {
    /// Maximum create attempts, including the first try. Must be ≥ 1.
    pub attempts: u32,
    /// Sleep between contention retries. Skipped when zero.
    pub retry_delay: Duration,
    /// Age at which an existing lock directory may be reclaimed.
    pub stale: Duration,
    /// Heartbeat interval for refreshing the held lock directory mtime.
    ///
    /// When zero, no background refresh is started (the holder can still be
    /// stolen after `stale` elapses).
    pub update: Duration,
    /// Override for the lock directory path.
    ///
    /// When `None`, the lock path is ``${absolute_target}.lock``. Trust storage
    /// passes an explicit sibling such as `trust.json.lock` while locking the
    /// parent directory.
    pub lockfile_path: Option<PathBuf>,
}

impl Default for LockOptions {
    fn default() -> Self {
        Self {
            attempts: DEFAULT_ATTEMPTS,
            retry_delay: DEFAULT_RETRY_DELAY,
            stale: DEFAULT_STALE,
            update: DEFAULT_UPDATE,
            lockfile_path: None,
        }
    }
}

impl LockOptions {
    /// Default production policy.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the maximum number of create attempts (including the first).
    #[must_use]
    pub fn attempts(mut self, attempts: u32) -> Self {
        self.attempts = attempts.max(1);
        self
    }

    /// Set the delay between contention retries.
    #[must_use]
    pub fn retry_delay(mut self, retry_delay: Duration) -> Self {
        self.retry_delay = retry_delay;
        self
    }

    /// Set the stale threshold used when reclaiming an abandoned lock.
    #[must_use]
    pub fn stale(mut self, stale: Duration) -> Self {
        self.stale = stale;
        self
    }

    /// Set the mtime refresh interval while the lock is held.
    #[must_use]
    pub fn update(mut self, update: Duration) -> Self {
        self.update = update;
        self
    }

    /// Override the lock directory path.
    #[must_use]
    pub fn lockfile_path(mut self, lockfile_path: impl Into<PathBuf>) -> Self {
        self.lockfile_path = Some(lockfile_path.into());
        self
    }
}

/// Failure to acquire or release a path lock.
#[derive(Debug, Error)]
pub enum LockError {
    /// Another live holder owns the lock (proper-lockfile `ELOCKED`).
    #[error("Lock file is already being held")]
    Contended {
        /// Path the caller asked to lock.
        target: PathBuf,
        /// Lock directory that could not be created.
        lock_path: PathBuf,
    },
    /// Non-contention filesystem failure.
    #[error("lock I/O failed for {}: {source}", lock_path.display())]
    Io {
        /// Path the caller asked to lock.
        target: PathBuf,
        /// Lock directory involved in the failure.
        lock_path: PathBuf,
        /// Underlying OS error.
        #[source]
        source: io::Error,
    },
}

impl LockError {
    /// Returns true when the failure is pure contention.
    #[must_use]
    pub const fn is_contended(&self) -> bool {
        matches!(self, Self::Contended { .. })
    }

    /// Stable error code matching proper-lockfile (`ELOCKED` / `EIO`).
    #[must_use]
    pub fn code(&self) -> &'static str {
        match self {
            Self::Contended { .. } => "ELOCKED",
            Self::Io { .. } => "EIO",
        }
    }
}

/// Shared stop signal for the mtime heartbeat: flag + condvar.
#[derive(Debug)]
struct StopSignal {
    stopped: Mutex<bool>,
    cvar: Condvar,
}

impl StopSignal {
    fn new() -> Self {
        Self {
            stopped: Mutex::new(false),
            cvar: Condvar::new(),
        }
    }

    /// Mark stopped and wake any waiter.
    fn stop(&self) {
        if let Ok(mut guard) = self.stopped.lock() {
            *guard = true;
            self.cvar.notify_all();
        }
    }

    /// Wait up to `total`, returning `true` when stop was observed.
    fn wait_timeout(&self, total: Duration) -> bool {
        let Ok(mut guard) = self.stopped.lock() else {
            // Poisoned mutex: treat as stopped so the heartbeat exits.
            return true;
        };
        if *guard {
            return true;
        }
        let deadline = Instant::now() + total;
        loop {
            if *guard {
                return true;
            }
            let now = Instant::now();
            if now >= deadline {
                return false;
            }
            let remaining = deadline.saturating_duration_since(now);
            let Ok((next, wait_result)) = self.cvar.wait_timeout(guard, remaining) else {
                return true;
            };
            guard = next;
            if *guard {
                return true;
            }
            if wait_result.timed_out() {
                return false;
            }
        }
    }
}

/// RAII exclusive lock over a proper-lockfile-compatible lock directory.
///
/// Dropping the guard stops the mtime heartbeat (if any) and removes only the
/// lock directory this guard created, and only while that directory is still
/// the same filesystem identity captured at acquire time.
#[derive(Debug)]
pub struct LockGuard {
    target: PathBuf,
    lock_path: PathBuf,
    /// Filesystem identity of the lock directory created by this guard.
    identity: Arc<Handle>,
    stop: Arc<StopSignal>,
    refresh_thread: Option<JoinHandle<()>>,
    released: bool,
}

impl LockGuard {
    /// Acquire an exclusive lock with the production default policy.
    ///
    /// # Errors
    ///
    /// Returns [`LockError::Contended`] when another live holder owns the lock
    /// after the configured retries, or [`LockError::Io`] for non-contention
    /// filesystem failures (including failure to capture lock identity).
    pub fn acquire(target: impl AsRef<Path>) -> Result<Self, LockError> {
        Self::acquire_with(target, &LockOptions::default())
    }

    /// Acquire an exclusive lock with a custom policy.
    ///
    /// # Errors
    ///
    /// Returns [`LockError::Contended`] when another live holder owns the lock
    /// after the configured retries, or [`LockError::Io`] for non-contention
    /// filesystem failures (including failure to capture lock identity).
    pub fn acquire_with(
        target: impl AsRef<Path>,
        options: &LockOptions,
    ) -> Result<Self, LockError> {
        let target_ref = target.as_ref();
        let target = absolute_path(target_ref);
        let lock_path = match &options.lockfile_path {
            Some(custom) => absolute_path(custom),
            None => default_lock_path(&target),
        };

        let attempts = options.attempts.max(1);
        let mut last_contended = false;

        for attempt in 1..=attempts {
            match try_create_lock(&lock_path, options.stale, true) {
                Ok(()) => {
                    let identity = match Handle::from_path(&lock_path) {
                        Ok(handle) => Arc::new(handle),
                        Err(source) => {
                            // Failed to capture identity: tear down the dir we
                            // just created so we do not leave an orphan lock.
                            let _ = fs::remove_dir(&lock_path);
                            return Err(LockError::Io {
                                target,
                                lock_path,
                                source,
                            });
                        }
                    };
                    return Ok(Self::from_acquired(
                        target,
                        lock_path,
                        identity,
                        options.update,
                    ));
                }
                Err(OnceError::Contended) => {
                    last_contended = true;
                    if attempt == attempts {
                        break;
                    }
                    if !options.retry_delay.is_zero() {
                        thread::sleep(options.retry_delay);
                    }
                }
                Err(OnceError::Io(source)) => {
                    return Err(LockError::Io {
                        target,
                        lock_path,
                        source,
                    });
                }
            }
        }

        if last_contended {
            Err(LockError::Contended { target, lock_path })
        } else {
            Err(LockError::Io {
                target,
                lock_path,
                source: io::Error::other("lock acquisition produced no result"),
            })
        }
    }

    /// Path the caller asked to lock (absolute, unresolved).
    #[must_use]
    pub fn target(&self) -> &Path {
        &self.target
    }

    /// Lock directory owned by this guard.
    #[must_use]
    pub fn lock_path(&self) -> &Path {
        &self.lock_path
    }

    /// Verify that this guard still owns the lock directory.
    ///
    /// # Errors
    ///
    /// Returns [`LockError::Io`] when the path vanished, changed identity, or
    /// cannot be inspected.
    pub fn check_ownership(&self) -> Result<(), LockError> {
        match still_our_lock(&self.lock_path, &self.identity) {
            Ok(true) => Ok(()),
            Ok(false) => Err(LockError::Io {
                target: self.target.clone(),
                lock_path: self.lock_path.clone(),
                source: io::Error::other("lock ownership was lost"),
            }),
            Err(source) => Err(LockError::Io {
                target: self.target.clone(),
                lock_path: self.lock_path.clone(),
                source,
            }),
        }
    }

    /// Explicitly release the lock. Equivalent to dropping the guard.
    ///
    /// # Errors
    ///
    /// Returns [`LockError::Io`] only when the lock directory is still owned by
    /// this guard and `remove_dir` fails for a reason other than not found.
    /// Ownership loss is treated as a successful no-op.
    pub fn release(mut self) -> Result<(), LockError> {
        self.release_inner()
    }

    fn from_acquired(
        target: PathBuf,
        lock_path: PathBuf,
        identity: Arc<Handle>,
        update: Duration,
    ) -> Self {
        let stop = Arc::new(StopSignal::new());
        let refresh_thread = start_refresh_thread(
            lock_path.clone(),
            Arc::clone(&identity),
            update,
            Arc::clone(&stop),
        );
        Self {
            target,
            lock_path,
            identity,
            stop,
            refresh_thread,
            released: false,
        }
    }

    fn release_inner(&mut self) -> Result<(), LockError> {
        if self.released {
            return Ok(());
        }
        self.released = true;
        self.stop.stop();
        if let Some(handle) = self.refresh_thread.take() {
            // Join so drop does not race a concurrent utime with remove_dir.
            let _ = handle.join();
        }

        // Only remove the directory if it is still the same filesystem object
        // we created. A stale reclaim by another process replaces the path with
        // a new directory; deleting that would steal the new owner's lock.
        match still_our_lock(&self.lock_path, &self.identity) {
            Ok(true) => match fs::remove_dir(&self.lock_path) {
                Ok(()) => Ok(()),
                Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(()),
                Err(source) => Err(LockError::Io {
                    target: self.target.clone(),
                    lock_path: self.lock_path.clone(),
                    source,
                }),
            },
            // Ownership lost or path gone: release is a no-op for the path.
            Ok(false) | Err(_) => Ok(()),
        }
    }
}

impl Drop for LockGuard {
    fn drop(&mut self) {
        let _ = self.release_inner();
    }
}

/// Internal outcome of a single create/reclaim attempt.
enum OnceError {
    Contended,
    Io(io::Error),
}

fn try_create_lock(
    lock_path: &Path,
    stale: Duration,
    allow_stale_reclaim: bool,
) -> Result<(), OnceError> {
    match fs::create_dir(lock_path) {
        Ok(()) => Ok(()),
        Err(err) if is_already_exists(&err) => {
            if !allow_stale_reclaim || stale.is_zero() {
                return Err(OnceError::Contended);
            }
            // Windows has no safe std API that atomically renames the directory
            // identified during the stale check. A metadata or handle precheck
            // followed by path-based removal still lets a new holder replace
            // the path in between. Fail closed until reclaim can be one
            // captured-handle operation; manual cleanup is safer than split
            // credential or settings writes.
            #[cfg(windows)]
            return Err(OnceError::Contended);
            #[cfg(not(windows))]
            match fs::metadata(lock_path) {
                Err(meta_err) if meta_err.kind() == io::ErrorKind::NotFound => {
                    // Lost the race with a releaser; one non-reclaiming retry.
                    try_create_lock(lock_path, Duration::ZERO, false)
                }
                Err(meta_err) => Err(OnceError::Io(meta_err)),
                Ok(meta) => {
                    if !is_lock_stale(&meta, stale) {
                        return Err(OnceError::Contended);
                    }
                    // Identity-safe reclamation: a contender may have reclaimed
                    // and recreated this path since the staleness check. Only
                    // remove the directory if it is still the same stale object
                    // we measured; a fresh replacement must not be deleted.
                    match reclaim_stale_lock(lock_path, &meta) {
                        Ok(true) => {}
                        Ok(false) => return Err(OnceError::Contended),
                        Err(source) => return Err(OnceError::Io(source)),
                    }
                    // After reclaim, never reclaim again in this attempt chain
                    // (mirrors proper-lockfile's `stale: 0` follow-up).
                    try_create_lock(lock_path, Duration::ZERO, false)
                }
            }
        }
        Err(err) => Err(OnceError::Io(err)),
    }
}

fn is_already_exists(err: &io::Error) -> bool {
    err.kind() == io::ErrorKind::AlreadyExists
}

fn is_lock_stale(meta: &fs::Metadata, stale: Duration) -> bool {
    let Ok(mtime) = meta.modified() else {
        // Unreadable mtime: treat as live so we do not steal blindly.
        return false;
    };
    match SystemTime::now().duration_since(mtime) {
        // proper-lockfile: `mtime.getTime() < Date.now() - stale`
        Ok(age) => age > stale,
        // Future mtime → not stale.
        Err(_) => false,
    }
}

#[cfg(not(windows))]
/// Reclaim a stale lock directory after re-reading its filesystem identity.
///
/// `stale_meta` is the metadata captured when the directory at `lock_path` was
/// judged stale. The path is re-read immediately before removal so that a
/// contender which reclaimed and recreated this path is *usually* detected and
/// its fresh lock spared. This narrows the TOCTOU window but does not close
/// it: `fs::metadata` and `fs::remove_dir` are separate syscalls, and a
/// contender can install a fresh lock in the gap between them. POSIX provides
/// no atomic compare-and-unlink for directories, so the race is inherent.
/// The identity check mitigates the consequence to a spurious error rather
/// than split state, but holders must still verify ownership via
/// [`LockGuard::check_ownership`] after acquisition.
///
/// Returns `Ok(true)` when the stale directory was removed (or had already
/// vanished) and the caller may retry creation, `Ok(false)` when a contender
/// has replaced it (the caller must treat this as contention), or `Err` for an
/// intervening I/O failure.
fn reclaim_stale_lock(lock_path: &Path, stale_meta: &fs::Metadata) -> Result<bool, io::Error> {
    let still_stale = match fs::metadata(lock_path) {
        Ok(current) => same_dir_identity(&current, stale_meta),
        // Vanished between staleness check and removal: a releaser finished.
        Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(true),
        Err(err) => return Err(err),
    };
    if !still_stale {
        // Path now names a different directory: a contender reclaimed it. Do
        // not delete the replacement lock.
        return Ok(false);
    }
    match fs::remove_dir(lock_path) {
        Ok(()) => Ok(true),
        Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(true),
        Err(err) => Err(err),
    }
}

#[cfg(not(windows))]
/// Compare two directory metadata objects for the same filesystem identity.
fn same_dir_identity(a: &fs::Metadata, b: &fs::Metadata) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        a.dev() == b.dev() && a.ino() == b.ino()
    }
    #[cfg(not(any(unix, windows)))]
    {
        a.len() == b.len()
            && a.modified().ok() == b.modified().ok()
            && a.created().ok() == b.created().ok()
    }
}

fn default_lock_path(target: &Path) -> PathBuf {
    let mut os = target.as_os_str().to_os_string();
    os.push(".lock");
    PathBuf::from(os)
}

fn absolute_path(path: &Path) -> PathBuf {
    // `std::path::absolute` does not touch the filesystem and does not resolve
    // symlinks — the Rust equivalent of Node `path.resolve` with
    // `realpath: false`.
    match std::path::absolute(path) {
        Ok(abs) => abs,
        Err(_) => {
            if path.is_absolute() {
                path.to_path_buf()
            } else {
                std::env::current_dir().map_or_else(|_| path.to_path_buf(), |cwd| cwd.join(path))
            }
        }
    }
}

/// Returns `Ok(true)` when `lock_path` still names the same directory identity.
fn still_our_lock(lock_path: &Path, identity: &Handle) -> io::Result<bool> {
    match Handle::from_path(lock_path) {
        Ok(current) => Ok(current == *identity),
        Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(err) => Err(err),
    }
}

fn start_refresh_thread(
    lock_path: PathBuf,
    identity: Arc<Handle>,
    update: Duration,
    stop: Arc<StopSignal>,
) -> Option<JoinHandle<()>> {
    if update.is_zero() {
        return None;
    }

    Some(thread::spawn(move || {
        loop {
            if stop.wait_timeout(update) {
                break;
            }
            // Refresh only while the path still names our original directory.
            // A reclaimed/replaced lock must not receive our mtime pulse, and
            // we must not invent a child marker file that would break rmdir.
            match still_our_lock(&lock_path, &identity) {
                Ok(true) => {
                    let _ = set_file_mtime(&lock_path, FileTime::now());
                }
                Ok(false) | Err(_) => break,
            }
        }
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;
    use std::sync::mpsc;
    use std::time::Duration;

    type TestResult = Result<(), Box<dyn Error + Send + Sync>>;

    #[derive(Debug)]
    struct TestFailure(String);

    impl std::fmt::Display for TestFailure {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str(&self.0)
        }
    }

    impl Error for TestFailure {}

    fn fail(message: impl Into<String>) -> Box<dyn Error + Send + Sync> {
        Box::new(TestFailure(message.into()))
    }

    fn make_temp_dir() -> Result<PathBuf, Box<dyn Error + Send + Sync>> {
        let nanos = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        let base =
            std::env::temp_dir().join(format!("pi-lockfile-{}-{}", std::process::id(), nanos));
        fs::create_dir_all(&base)?;
        Ok(base)
    }

    fn short_options() -> LockOptions {
        LockOptions::new()
            .attempts(3)
            .retry_delay(Duration::from_millis(5))
            .stale(Duration::from_millis(40))
            .update(Duration::from_millis(10))
    }

    fn require_contended(
        result: Result<LockGuard, LockError>,
    ) -> Result<LockError, Box<dyn Error + Send + Sync>> {
        match result {
            Ok(_) => Err(fail("expected contention, acquired lock")),
            Err(err) if err.is_contended() => Ok(err),
            Err(err) => Err(fail(format!("expected contention, got {err}"))),
        }
    }

    fn require_io(
        result: Result<LockGuard, LockError>,
    ) -> Result<LockError, Box<dyn Error + Send + Sync>> {
        match result {
            Ok(_) => Err(fail("expected I/O error, acquired lock")),
            Err(err) if !err.is_contended() => Ok(err),
            Err(err) => Err(fail(format!("expected I/O error, got {err}"))),
        }
    }

    #[test]
    fn acquires_when_target_does_not_exist() -> TestResult {
        let dir = make_temp_dir()?;
        let target = dir.join("settings.json");
        if target.exists() {
            return Err(fail("target should not exist before acquire"));
        }

        let guard = LockGuard::acquire(&target)?;
        if !guard.lock_path().exists() {
            return Err(fail("lock directory missing after acquire"));
        }
        if !guard.lock_path().is_dir() {
            return Err(fail("lock path is not a directory"));
        }
        let ends_with_lock = guard
            .lock_path()
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.ends_with("settings.json.lock"));
        if !ends_with_lock {
            return Err(fail("lock path suffix mismatch"));
        }
        drop(guard);
        if default_lock_path(&absolute_path(&target)).exists() {
            return Err(fail("lock directory remained after drop"));
        }
        let _ = fs::remove_dir_all(&dir);
        Ok(())
    }

    #[test]
    fn release_on_drop_removes_only_owned_lock_dir() -> TestResult {
        let dir = make_temp_dir()?;
        let target = dir.join("trust.json");
        let foreign = dir.join("other.lock");
        fs::create_dir(&foreign)?;

        {
            let guard = LockGuard::acquire(&target)?;
            if !guard.lock_path().exists() {
                return Err(fail("owned lock missing"));
            }
            if !foreign.exists() {
                return Err(fail("foreign lock missing during hold"));
            }
        }

        if default_lock_path(&absolute_path(&target)).exists() {
            return Err(fail("owned lock remained after drop"));
        }
        if !foreign.exists() {
            return Err(fail("unrelated lock dir must remain"));
        }
        let _ = fs::remove_dir_all(&dir);
        Ok(())
    }

    #[test]
    fn mutual_exclusion_second_holder_is_contended() -> TestResult {
        let dir = make_temp_dir()?;
        let target = dir.join("settings.json");
        let first = LockGuard::acquire(&target)?;

        let target_for_thread = target.clone();
        let (tx, rx) = mpsc::channel();
        let handle = thread::spawn(move || {
            let result = LockGuard::acquire_with(
                &target_for_thread,
                &LockOptions::new()
                    .attempts(2)
                    .retry_delay(Duration::from_millis(5)),
            );
            let is_err = result.is_err();
            let _ = tx.send(is_err);
            result
        });

        let contended = match rx.recv_timeout(Duration::from_millis(200)) {
            Ok(value) => value,
            Err(err) => return Err(fail(format!("join signal: {err}"))),
        };
        if !contended {
            return Err(fail("second acquire must fail while first holds"));
        }

        let Ok(thread_result) = handle.join() else {
            return Err(fail("second-holder thread panicked"));
        };
        let err = require_contended(thread_result)?;
        if err.code() != "ELOCKED" {
            return Err(fail(format!("expected ELOCKED, got {}", err.code())));
        }

        drop(first);
        let second = LockGuard::acquire(&target)?;
        drop(second);
        let _ = fs::remove_dir_all(&dir);
        Ok(())
    }

    #[cfg(not(windows))]
    #[test]
    fn stale_lock_is_reclaimed() -> TestResult {
        let dir = make_temp_dir()?;
        let target = dir.join("settings.json");
        let lock_path = default_lock_path(&absolute_path(&target));
        fs::create_dir(&lock_path)?;

        let past = SystemTime::now()
            .checked_sub(Duration::from_secs(30))
            .ok_or_else(|| fail("past time underflow"))?;
        set_file_mtime(&lock_path, FileTime::from_system_time(past))?;

        let guard = LockGuard::acquire_with(
            &target,
            &LockOptions::new()
                .attempts(2)
                .retry_delay(Duration::ZERO)
                .stale(Duration::from_millis(50))
                .update(Duration::from_millis(20)),
        )?;
        if !guard.lock_path().exists() {
            return Err(fail("reclaimed lock missing"));
        }
        drop(guard);
        let _ = fs::remove_dir_all(&dir);
        Ok(())
    }

    #[cfg(not(windows))]
    #[test]
    fn stale_reclaim_removes_unchanged_stale_directory() -> TestResult {
        let dir = make_temp_dir()?;
        let target = dir.join("settings.json");
        let lock_path = default_lock_path(&absolute_path(&target));
        fs::create_dir(&lock_path)?;
        let past = SystemTime::now()
            .checked_sub(Duration::from_secs(30))
            .ok_or_else(|| fail("past time underflow"))?;
        set_file_mtime(&lock_path, FileTime::from_system_time(past))?;
        let stale_meta = fs::metadata(&lock_path)?;

        match reclaim_stale_lock(&lock_path, &stale_meta) {
            Ok(true) => {}
            Ok(false) => return Err(fail("unchanged stale directory must be reclaimed")),
            Err(err) => return Err(fail(format!("unexpected reclaim error: {err}"))),
        }
        if lock_path.exists() {
            return Err(fail("stale directory must be removed by reclaim"));
        }
        let _ = fs::remove_dir_all(&dir);
        Ok(())
    }

    #[cfg(not(windows))]
    #[test]
    fn stale_reclaim_keeps_replacement_lock_directory() -> TestResult {
        let dir = make_temp_dir()?;
        let target = dir.join("settings.json");
        let lock_path = default_lock_path(&absolute_path(&target));
        fs::create_dir(&lock_path)?;

        // Age the directory past the stale threshold and capture its identity.
        let past = SystemTime::now()
            .checked_sub(Duration::from_secs(30))
            .ok_or_else(|| fail("past time underflow"))?;
        set_file_mtime(&lock_path, FileTime::from_system_time(past))?;
        let stale_meta = fs::metadata(&lock_path)?;

        // A contender reclaims the stale directory and installs a fresh
        // replacement at the same path, changing its filesystem identity.
        fs::remove_dir(&lock_path)?;
        fs::create_dir(&lock_path)?;
        let replacement = Handle::from_path(&lock_path)?;
        // Precondition, not assertion: this needs a filesystem that issues a
        // fresh identity on remove+create. A filesystem that recycles inodes
        // cannot distinguish contender from stale record, so there is nothing
        // identity-safe reclaim could prove here; pass vacuously instead of
        // failing on the filesystem's behavior.
        if same_dir_identity(&fs::metadata(&lock_path)?, &stale_meta) {
            return Ok(());
        }

        // Reclamation must observe the changed identity and refuse to delete the
        // contender's fresh lock, reporting contention instead.
        match reclaim_stale_lock(&lock_path, &stale_meta) {
            Ok(false) => {}
            Ok(true) => return Err(fail("reclaim deleted a replacement lock")),
            Err(err) => return Err(fail(format!("unexpected reclaim error: {err}"))),
        }
        if !lock_path.exists() {
            return Err(fail("replacement lock must survive identity-safe reclaim"));
        }
        if Handle::from_path(&lock_path)? != replacement {
            return Err(fail("replacement lock identity changed after reclaim"));
        }
        let _ = fs::remove_dir_all(&dir);
        Ok(())
    }

    #[cfg(windows)]
    #[test]
    fn stale_windows_lock_is_not_reclaimed_without_atomic_handle_rename() -> TestResult {
        let dir = make_temp_dir()?;
        let target = dir.join("settings.json");
        let lock_path = default_lock_path(&absolute_path(&target));
        fs::create_dir(&lock_path)?;
        let past = SystemTime::now()
            .checked_sub(Duration::from_secs(30))
            .ok_or_else(|| fail("past time underflow"))?;
        set_file_mtime(&lock_path, FileTime::from_system_time(past))?;
        let identity = Handle::from_path(&lock_path)?;

        let err = require_contended(LockGuard::acquire_with(
            &target,
            &LockOptions::new()
                .attempts(1)
                .retry_delay(Duration::ZERO)
                .stale(Duration::from_millis(50))
                .update(Duration::ZERO),
        ))?;
        if !err.is_contended() {
            return Err(fail("stale Windows lock must fail closed as contention"));
        }
        if Handle::from_path(&lock_path)? != identity {
            return Err(fail(
                "stale Windows lock identity changed during acquisition",
            ));
        }
        let _ = fs::remove_dir_all(&dir);
        Ok(())
    }

    #[test]
    fn fresh_foreign_lock_times_out_as_contended() -> TestResult {
        let dir = make_temp_dir()?;
        let target = dir.join("settings.json");
        let lock_path = default_lock_path(&absolute_path(&target));
        fs::create_dir(&lock_path)?;

        let started = Instant::now();
        let err = require_contended(LockGuard::acquire_with(
            &target,
            &LockOptions::new()
                .attempts(3)
                .retry_delay(Duration::from_millis(5))
                .stale(Duration::from_secs(30))
                .update(Duration::from_secs(15)),
        ))?;
        let elapsed = started.elapsed();

        if err.code() != "ELOCKED" {
            return Err(fail(format!("expected ELOCKED, got {}", err.code())));
        }
        if elapsed > Duration::from_millis(200) {
            return Err(fail(format!(
                "retries must stay within 10*20ms production bound, elapsed={elapsed:?}"
            )));
        }
        if !lock_path.exists() {
            return Err(fail("foreign lock must remain"));
        }
        let _ = fs::remove_dir_all(&dir);
        Ok(())
    }

    #[test]
    fn held_lock_stays_fresh_past_stale_threshold() -> TestResult {
        let dir = make_temp_dir()?;
        let target = dir.join("settings.json");
        let opts = short_options();

        let guard = LockGuard::acquire_with(&target, &opts)?;

        // Wait longer than the stale threshold. Heartbeat refresh at
        // `update=10ms` must keep mtime fresh so a second process cannot steal.
        thread::sleep(Duration::from_millis(100));

        let err = require_contended(LockGuard::acquire_with(
            &target,
            &LockOptions::new()
                .attempts(1)
                .retry_delay(Duration::ZERO)
                .stale(opts.stale)
                .update(opts.update),
        ))?;
        if !err.is_contended() {
            return Err(fail("live holder must not be stolen after stale window"));
        }

        drop(guard);
        let reacquired = LockGuard::acquire_with(&target, &opts)?;
        drop(reacquired);
        let _ = fs::remove_dir_all(&dir);
        Ok(())
    }

    #[test]
    fn unrelated_io_errors_propagate_without_retry_delay() -> TestResult {
        let dir = make_temp_dir()?;
        // Parent of the lock directory does not exist → create_dir fails with
        // NotFound (or equivalent), which is not contention and must not be
        // retried for hundreds of milliseconds.
        let target = dir.join("missing").join("nested").join("settings.json");
        let started = Instant::now();
        let err = require_io(LockGuard::acquire_with(
            &target,
            &LockOptions::new()
                .attempts(10)
                .retry_delay(Duration::from_millis(20)),
        ))?;
        let elapsed = started.elapsed();

        if err.is_contended() {
            return Err(fail("non-contention path returned contended"));
        }
        if err.code() != "EIO" {
            return Err(fail(format!("expected EIO, got {}", err.code())));
        }
        if elapsed >= Duration::from_millis(50) {
            return Err(fail(format!(
                "non-contention errors must not sleep through the retry budget, elapsed={elapsed:?}"
            )));
        }
        let _ = fs::remove_dir_all(&dir);
        Ok(())
    }

    #[test]
    fn custom_lockfile_path_is_honored() -> TestResult {
        let dir = make_temp_dir()?;
        let target = dir.join("store");
        fs::create_dir_all(&target)?;
        let custom = dir.join("trust.json.lock");

        let guard =
            LockGuard::acquire_with(&target, &LockOptions::new().lockfile_path(custom.clone()))?;
        if guard.lock_path() != absolute_path(&custom).as_path() {
            return Err(fail("custom lock path not honored"));
        }
        if !(custom.exists() || absolute_path(&custom).exists()) {
            return Err(fail("custom lock directory missing while held"));
        }
        drop(guard);
        if absolute_path(&custom).exists() {
            return Err(fail("custom lock directory remained after drop"));
        }
        let _ = fs::remove_dir_all(&dir);
        Ok(())
    }

    #[test]
    fn explicit_release_is_idempotent_with_drop() -> TestResult {
        let dir = make_temp_dir()?;
        let target = dir.join("settings.json");
        let guard = LockGuard::acquire(&target)?;
        let lock_path = guard.lock_path().to_path_buf();
        guard.release()?;
        if lock_path.exists() {
            return Err(fail("lock remained after explicit release"));
        }
        let _ = fs::remove_dir_all(&dir);
        Ok(())
    }

    #[test]
    fn replaced_lock_dir_is_neither_refreshed_nor_deleted_by_old_guard() -> TestResult {
        let dir = make_temp_dir()?;
        let target = dir.join("settings.json");
        let lock_path = default_lock_path(&absolute_path(&target));

        // Slow heartbeat so we can replace the dir between ticks and assert
        // the old guard does not re-touch or delete the replacement.
        let guard = LockGuard::acquire_with(
            &target,
            &LockOptions::new()
                .attempts(1)
                .retry_delay(Duration::ZERO)
                .stale(Duration::from_millis(30))
                .update(Duration::from_millis(200)),
        )?;

        let original_identity = Handle::from_path(&lock_path)?;

        // Simulate a stale reclaim: remove the original dir and create a
        // replacement at the same path owned by a new process.
        fs::remove_dir(&lock_path)?;
        fs::create_dir(&lock_path)?;
        let replacement_identity = Handle::from_path(&lock_path)?;
        if original_identity == replacement_identity {
            return Err(fail("replacement must be a distinct filesystem object"));
        }

        // Age the replacement mtime into the past. If the old heartbeat still
        // refreshes the path, mtime will jump forward again.
        let past = SystemTime::now()
            .checked_sub(Duration::from_mins(1))
            .ok_or_else(|| fail("past time underflow"))?;
        set_file_mtime(&lock_path, FileTime::from_system_time(past))?;
        let aged_mtime = fs::metadata(&lock_path).and_then(|m| m.modified())?;

        // Wait longer than one update interval so a buggy heartbeat would have
        // fired against the replacement path.
        thread::sleep(Duration::from_millis(250));

        let after_wait_mtime = fs::metadata(&lock_path).and_then(|m| m.modified())?;
        if aged_mtime != after_wait_mtime {
            return Err(fail("old guard must not refresh a replaced lock directory"));
        }
        if Handle::from_path(&lock_path)? != replacement_identity {
            return Err(fail("replacement identity changed during wait"));
        }

        // Drop must not delete the replacement either.
        drop(guard);
        if !lock_path.exists() {
            return Err(fail("old guard must not remove a replaced lock directory"));
        }
        if Handle::from_path(&lock_path)? != replacement_identity {
            return Err(fail("replacement identity lost after old guard drop"));
        }

        let _ = fs::remove_dir_all(&dir);
        Ok(())
    }
}
