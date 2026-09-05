//! Session listing, info extraction, and most-recent lookup.
//!
//! Ports `buildSessionInfo`, `findMostRecentSession`, `list`/`listAll` helpers
//! from `.references/pi-2.0/packages/coding-agent/src/core/session-manager.ts`.

use std::cmp::Reverse;
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use serde_json::Value;
use tokio::task::JoinSet;

use super::super::config::{PathInputOptions, normalize_path, resolve_path};
use super::entries::{
    NO_MESSAGES_PLACEHOLDER, iso_to_millis, mtime_millis, path_exists, read_session_header,
};

/// Maximum concurrent session-info loads (TypeScript `MAX_CONCURRENT_SESSION_INFO_LOADS`).
pub const MAX_CONCURRENT_SESSION_INFO_LOADS: usize = 10;

/// Progress callback: `(loaded, total)`.
pub type SessionListProgress<'a> = &'a (dyn Fn(usize, usize) + Send + Sync);

/// Summary info for one session file (used by selectors / list UIs).
#[derive(Clone, Debug, PartialEq)]
pub struct SessionInfo {
    /// Absolute or normalized path to the session file.
    pub path: String,
    /// Session header id (may be missing on pathological files).
    pub id: Option<String>,
    /// Working directory from the header (empty string when absent).
    pub cwd: String,
    /// Latest non-empty `session_info` name, if any.
    pub name: Option<String>,
    /// Parent session path from the header, if any.
    pub parent_session_path: Option<String>,
    /// Header timestamp as Unix ms (`None` when unparseable).
    pub created: Option<i64>,
    /// Last user/assistant activity ms, else header time, else file mtime.
    pub modified: i64,
    /// Count of `type: "message"` entries.
    pub message_count: usize,
    /// First user-message text, or [`NO_MESSAGES_PLACEHOLDER`].
    pub first_message: String,
    /// All user/assistant text contents joined with spaces. Populated by
    /// [`build_session_info`]; directory listing APIs leave it empty to keep
    /// listing memory proportional to session count.
    pub all_messages_text: String,
}

/// Find the most recently modified valid session file in a directory.
///
/// When `cwd` is `Some`, only sessions whose header cwd resolves to that path
/// are considered.
#[must_use]
pub fn find_most_recent_session(session_dir: &Path, cwd: Option<&str>) -> Option<PathBuf> {
    find_most_recent_session_inner(session_dir, cwd).unwrap_or(None)
}

fn find_most_recent_session_inner(
    session_dir: &Path,
    cwd: Option<&str>,
) -> Result<Option<PathBuf>, ()> {
    let resolved_session_dir =
        normalize_path(&session_dir.to_string_lossy(), PathInputOptions::new());
    let resolved_cwd = cwd.map(resolve_path);

    let entries = fs::read_dir(&resolved_session_dir).map_err(|_| ())?;
    let mut candidates: Vec<PathBuf> = Vec::new();

    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("jsonl") {
            continue;
        }
        let Some(header) = read_session_header(&path) else {
            continue;
        };
        if let Some(rcwd) = &resolved_cwd {
            let header_cwd = header.cwd.as_deref();
            if !session_cwd_matches(header_cwd, rcwd) {
                continue;
            }
        }
        candidates.push(path);
    }
    Ok(most_recent_candidate(candidates, |path| {
        fs::metadata(path)
            .ok()
            .map(|meta| mtime_millis(&meta).unwrap_or(0))
    }))
}

fn most_recent_candidate(
    candidates: Vec<PathBuf>,
    mut modified: impl FnMut(&Path) -> Option<i64>,
) -> Option<PathBuf> {
    let mut best: Option<(PathBuf, i64)> = None;
    for path in candidates {
        let Some(mtime) = modified(&path) else {
            continue;
        };
        if best
            .as_ref()
            .is_none_or(|(_, best_mtime)| mtime > *best_mtime)
        {
            best = Some((path, mtime));
        }
    }
    best.map(|(path, _)| path)
}

/// True when `cwd` is non-empty and resolves equal to `resolved_cwd`.
#[must_use]
pub fn session_cwd_matches(cwd: Option<&str>, resolved_cwd: &Path) -> bool {
    match cwd {
        Some(c) if !c.is_empty() => resolve_path(c) == resolved_cwd,
        _ => false,
    }
}

/// Build [`SessionInfo`] for one session file (sync, pure file IO).
#[must_use]
pub fn build_session_info(file_path: &Path) -> Option<SessionInfo> {
    build_session_info_inner(file_path, true).ok().flatten()
}

#[derive(Default)]
struct SessionScan {
    header: Option<Value>,
    message_count: usize,
    first_message: String,
    all_messages: Vec<String>,
    name: Option<String>,
    last_activity: Option<i64>,
}

impl SessionScan {
    /// Ingest one JSONL entry. Returns `false` when the file is invalid
    /// (caller should yield `Ok(None)`).
    fn ingest_entry(&mut self, entry: Value, include_all_messages: bool) -> bool {
        if self.header.is_none() {
            if entry.get("type").and_then(Value::as_str) != Some("session") {
                return false;
            }
            self.header = Some(entry);
            return true;
        }

        let ty = entry.get("type").and_then(Value::as_str);
        if ty == Some("session_info") {
            self.name = entry
                .get("name")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_owned);
            return true;
        }

        if ty != Some("message") {
            return true;
        }
        self.ingest_message_entry(&entry, include_all_messages)
    }

    fn ingest_message_entry(&mut self, entry: &Value, include_all_messages: bool) -> bool {
        self.message_count += 1;

        let Some(message) = entry.get("message") else {
            return true;
        };
        if !is_message_with_content(message) {
            return true;
        }
        let role = message.get("role").and_then(Value::as_str);
        if role != Some("user") && role != Some("assistant") {
            return true;
        }

        if let Some(activity) = get_message_activity_time(message, entry) {
            self.last_activity = Some(
                self.last_activity
                    .map_or(activity, |prev| prev.max(activity)),
            );
        }

        // extractTextContent — non-string non-array content throws in TS and
        // aborts the whole info build (caught → null).
        let Some(text) = extract_text_content(message) else {
            return false;
        };
        if text.is_empty() {
            return true;
        }
        if include_all_messages {
            self.all_messages.push(text.clone());
        }
        if self.first_message.is_empty() && role == Some("user") {
            self.first_message = text;
        }
        true
    }

    fn into_session_info(
        self,
        file_path: &Path,
        file_mtime: i64,
    ) -> Result<Option<SessionInfo>, ()> {
        let header = self.header.ok_or(())?;
        let cwd = header
            .get("cwd")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_owned();
        let parent_session_path = header
            .get("parentSession")
            .and_then(Value::as_str)
            .map(str::to_owned);
        let id = header.get("id").and_then(Value::as_str).map(str::to_owned);
        let header_time = header
            .get("timestamp")
            .and_then(Value::as_str)
            .and_then(iso_to_millis);
        let created = header_time;
        let modified = if let Some(t) = self.last_activity.filter(|&t| t > 0) {
            t
        } else if let Some(t) = header_time {
            t
        } else {
            file_mtime
        };

        Ok(Some(SessionInfo {
            path: file_path.to_string_lossy().into_owned(),
            id,
            cwd,
            name: self.name,
            parent_session_path,
            created,
            modified,
            message_count: self.message_count,
            first_message: if self.first_message.is_empty() {
                NO_MESSAGES_PLACEHOLDER.to_owned()
            } else {
                self.first_message
            },
            all_messages_text: self.all_messages.join(" "),
        }))
    }
}

fn build_session_info_inner(
    file_path: &Path,
    include_all_messages: bool,
) -> Result<Option<SessionInfo>, ()> {
    let meta = fs::metadata(file_path).map_err(|_| ())?;
    let file_mtime = mtime_millis(&meta).unwrap_or(0);

    let file = File::open(file_path).map_err(|_| ())?;
    let reader = BufReader::new(file);

    let mut scan = SessionScan::default();
    for line_result in reader.lines() {
        let line = line_result.map_err(|_| ())?;
        if line.trim().is_empty() {
            continue;
        }
        let entry: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(_) => continue,
        };
        if !scan.ingest_entry(entry, include_all_messages) {
            return Ok(None);
        }
    }

    scan.into_session_info(file_path, file_mtime)
}

fn is_message_with_content(message: &Value) -> bool {
    message.get("role").and_then(Value::as_str).is_some() && message.get("content").is_some()
}

fn finite_f64_to_i64(f: f64) -> Option<i64> {
    if !f.is_finite() {
        return None;
    }
    let r = f.round();
    format!("{r:.0}").parse().ok()
}

fn get_message_activity_time(message: &Value, entry: &Value) -> Option<i64> {
    if let Some(n) = message.get("timestamp").and_then(Value::as_i64) {
        return Some(n);
    }
    if let Some(n) = message.get("timestamp").and_then(Value::as_f64) {
        return finite_f64_to_i64(n);
    }
    entry
        .get("timestamp")
        .and_then(Value::as_str)
        .and_then(iso_to_millis)
}

/// Extract user/assistant text content. Returns `None` when content is neither
/// a string nor an array (TypeScript throws → whole info fails).
fn extract_text_content(message: &Value) -> Option<String> {
    let content = message.get("content")?;
    if let Some(s) = content.as_str() {
        return Some(s.to_owned());
    }
    let arr = content.as_array()?;
    let mut parts = Vec::new();
    for block in arr {
        if block.get("type").and_then(Value::as_str) == Some("text")
            && let Some(t) = block.get("text").and_then(Value::as_str)
        {
            parts.push(t);
        }
    }
    Some(parts.join(" "))
}

fn is_jsonl_path(path: &Path) -> bool {
    path.extension().and_then(|e| e.to_str()) == Some("jsonl")
}

/// List sessions in one directory with bounded concurrency (max 10).
pub async fn list_sessions_from_dir(
    dir: &Path,
    on_progress: Option<SessionListProgress<'_>>,
    progress_offset: usize,
    progress_total: Option<usize>,
) -> Vec<SessionInfo> {
    if !path_exists(dir) {
        return Vec::new();
    }

    let files: Vec<PathBuf> = match fs::read_dir(dir) {
        Ok(entries) => entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| is_jsonl_path(p))
            .collect(),
        Err(_) => return Vec::new(),
    };

    let total = progress_total.unwrap_or(files.len());
    let results = build_session_infos_with_concurrency(files, |loaded| {
        if let Some(cb) = on_progress {
            cb(progress_offset + loaded, total);
        }
    })
    .await;

    results.into_iter().flatten().collect()
}

async fn build_session_infos_with_concurrency(
    files: Vec<PathBuf>,
    mut on_loaded: impl FnMut(usize),
) -> Vec<Option<SessionInfo>> {
    let n = files.len();
    let mut results: Vec<Option<SessionInfo>> = (0..n).map(|_| None).collect();
    let mut set: JoinSet<(usize, Option<SessionInfo>)> = JoinSet::new();
    let mut next = 0_usize;
    let mut completed = 0_usize;

    while next < n || !set.is_empty() {
        while next < n && set.len() < MAX_CONCURRENT_SESSION_INFO_LOADS {
            let i = next;
            let file = files[i].clone();
            set.spawn_blocking(move || (i, build_session_info_inner(&file, false).ok().flatten()));
            next += 1;
        }
        if let Some(joined) = set.join_next().await {
            if let Ok((i, info)) = joined {
                results[i] = info;
            }
            // panic in worker → leave None
            completed += 1;
            on_loaded(completed);
        }
    }
    results
}

/// List all sessions under the agent sessions root (or a custom single dir).
pub async fn list_all_sessions(
    sessions_root: &Path,
    custom_session_dir: Option<&Path>,
    on_progress: Option<SessionListProgress<'_>>,
) -> Vec<SessionInfo> {
    if let Some(custom) = custom_session_dir {
        let mut sessions = list_sessions_from_dir(custom, on_progress, 0, None).await;
        sessions.sort_by_key(|s| Reverse(s.modified));
        return sessions;
    }

    if !path_exists(sessions_root) {
        return Vec::new();
    }

    let dirs: Vec<PathBuf> = match fs::read_dir(sessions_root) {
        Ok(entries) => entries
            .flatten()
            .filter(|e| e.file_type().is_ok_and(|t| t.is_dir()))
            .map(|e| e.path())
            .collect(),
        Err(_) => return Vec::new(),
    };

    let mut all_files: Vec<PathBuf> = Vec::new();
    for dir in &dirs {
        if let Ok(entries) = fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if is_jsonl_path(&path) {
                    all_files.push(path);
                }
            }
        }
    }

    let total = all_files.len();
    let results = build_session_infos_with_concurrency(all_files, |loaded| {
        if let Some(cb) = on_progress {
            cb(loaded, total);
        }
    })
    .await;

    let mut sessions: Vec<SessionInfo> = results.into_iter().flatten().collect();
    sessions.sort_by_key(|s| Reverse(s.modified));
    sessions
}

/// List sessions for a cwd with optional custom session dir and cwd filter.
pub async fn list_sessions_for_cwd(
    cwd: &str,
    session_dir: &Path,
    filter_cwd: bool,
    on_progress: Option<SessionListProgress<'_>>,
) -> Vec<SessionInfo> {
    let resolved_cwd = resolve_path(cwd);
    let mut sessions = list_sessions_from_dir(session_dir, on_progress, 0, None).await;
    if filter_cwd {
        sessions.retain(|s| session_cwd_matches(Some(&s.cwd), &resolved_cwd));
    }
    sessions.sort_by_key(|s| Reverse(s.modified));
    sessions
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::tempdir;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    fn write_session(path: &Path, id: &str, cwd: &str, messages: &[(&str, &str)]) -> TestResult {
        let mut f = File::create(path)?;
        writeln!(
            f,
            r#"{{"type":"session","version":3,"id":"{id}","timestamp":"2025-01-01T00:00:00.000Z","cwd":"{cwd}"}}"#
        )?;
        let mut parent = "null".to_owned();
        for (i, (role, text)) in messages.iter().enumerate() {
            let eid = format!("e{i}");
            if *role == "user" {
                writeln!(
                    f,
                    r#"{{"type":"message","id":"{eid}","parentId":{parent},"timestamp":"2025-01-01T00:00:0{i}.000Z","message":{{"role":"user","content":"{text}","timestamp":{ts}}}}}"#,
                    ts = 1_000 + i
                )?;
            } else {
                writeln!(
                    f,
                    r#"{{"type":"message","id":"{eid}","parentId":{parent},"timestamp":"2025-01-01T00:00:0{i}.000Z","message":{{"role":"assistant","content":[{{"type":"text","text":"{text}"}}],"api":"test","provider":"test","model":"test","usage":{{"input":1,"output":1,"cacheRead":0,"cacheWrite":0,"totalTokens":2,"cost":{{"input":0,"output":0,"cacheRead":0,"cacheWrite":0,"total":0}}}},"stopReason":"stop","timestamp":{ts}}}}}"#,
                    ts = 1_000 + i
                )?;
            }
            parent = format!(r#""{eid}""#);
        }
        Ok(())
    }

    #[test]
    fn find_most_recent_returns_null_for_empty() -> TestResult {
        let dir = tempdir()?;
        assert!(find_most_recent_session(dir.path(), None).is_none());
        Ok(())
    }

    #[test]
    fn find_most_recent_returns_valid() -> TestResult {
        let dir = tempdir()?;
        let file = dir.path().join("s.jsonl");
        write_session(&file, "abc", "/tmp", &[])?;
        assert_eq!(find_most_recent_session(dir.path(), None), Some(file));
        Ok(())
    }

    #[test]
    fn most_recent_skips_candidate_with_unreadable_metadata() {
        let unreadable = PathBuf::from("unreadable.jsonl");
        let valid = PathBuf::from("valid.jsonl");
        let selected =
            most_recent_candidate(vec![unreadable.clone(), valid.clone()], |path| match path {
                p if p == unreadable => None,
                p if p == valid => Some(42),
                _ => None,
            });
        assert_eq!(selected, Some(valid));
    }

    #[test]
    fn find_most_recent_filters_by_cwd() -> TestResult {
        let dir = tempdir()?;
        let a = dir.path().join("a.jsonl");
        let b = dir.path().join("b.jsonl");
        let project_a = dir.path().join("project-a");
        let project_b = dir.path().join("project-b");
        fs::create_dir_all(&project_a)?;
        fs::create_dir_all(&project_b)?;
        write_session(&a, "a", &project_a.to_string_lossy(), &[])?;
        std::thread::sleep(std::time::Duration::from_millis(15));
        write_session(&b, "b", &project_b.to_string_lossy(), &[])?;
        assert_eq!(
            find_most_recent_session(dir.path(), Some(&project_a.to_string_lossy())),
            Some(a)
        );
        assert_eq!(
            find_most_recent_session(dir.path(), Some(&project_b.to_string_lossy())),
            Some(b)
        );
        Ok(())
    }

    #[test]
    fn build_session_info_first_message() -> TestResult {
        let dir = tempdir()?;
        let file = dir.path().join("s.jsonl");
        write_session(
            &file,
            "abc",
            "/tmp",
            &[("user", "hello"), ("assistant", "hi")],
        )?;
        let info = build_session_info(&file).ok_or("info")?;
        assert_eq!(info.first_message, "hello");
        assert_eq!(info.message_count, 2);
        assert_eq!(info.id.as_deref(), Some("abc"));
        assert!(info.all_messages_text.contains("hello"));
        assert!(info.all_messages_text.contains("hi"));
        Ok(())
    }

    #[test]
    fn build_session_info_no_messages_placeholder() -> TestResult {
        let dir = tempdir()?;
        let file = dir.path().join("s.jsonl");
        write_session(&file, "abc", "/tmp", &[])?;
        let info = build_session_info(&file).ok_or("info")?;
        assert_eq!(info.first_message, NO_MESSAGES_PLACEHOLDER);
        assert_eq!(info.message_count, 0);
        Ok(())
    }

    #[tokio::test]
    async fn list_sessions_sorted_by_modified() -> TestResult {
        let dir = tempdir()?;
        let older = dir.path().join("older.jsonl");
        let newer = dir.path().join("newer.jsonl");
        write_session(&older, "old", "/tmp", &[("user", "a"), ("assistant", "b")])?;
        // Give a distinct mtime / message timestamp
        std::thread::sleep(std::time::Duration::from_millis(15));
        write_session(&newer, "new", "/tmp", &[("user", "c"), ("assistant", "d")])?;

        // Patch newer with a higher message timestamp so modified uses it
        let mut f = File::create(&newer)?;
        writeln!(
            f,
            r#"{{"type":"session","version":3,"id":"new","timestamp":"2025-01-01T00:00:00.000Z","cwd":"/tmp"}}
{{"type":"message","id":"e0","parentId":null,"timestamp":"2025-01-01T00:00:00.000Z","message":{{"role":"user","content":"c","timestamp":9999}}}}
{{"type":"message","id":"e1","parentId":"e0","timestamp":"2025-01-01T00:00:01.000Z","message":{{"role":"assistant","content":[{{"type":"text","text":"d"}}],"api":"test","provider":"test","model":"test","usage":{{"input":1,"output":1,"cacheRead":0,"cacheWrite":0,"totalTokens":2,"cost":{{"input":0,"output":0,"cacheRead":0,"cacheWrite":0,"total":0}}}},"stopReason":"stop","timestamp":10000}}}}"#
        )?;
        drop(f);

        let mut sessions = list_sessions_from_dir(dir.path(), None, 0, None).await;
        // from_dir preserves readdir order (filesystem-dependent); sort by
        // modified exactly as the product callers do before asserting order.
        sessions.sort_by_key(|s| Reverse(s.modified));
        assert_eq!(sessions.len(), 2);
        assert_eq!(sessions[0].id.as_deref(), Some("new"));
        assert_eq!(sessions[0].modified, 10000);
        assert!(
            sessions
                .iter()
                .all(|session| session.all_messages_text.is_empty()),
            "directory listing must not retain transcript text"
        );
        Ok(())
    }

    #[test]
    fn max_concurrent_is_ten() {
        assert_eq!(MAX_CONCURRENT_SESSION_INFO_LOADS, 10);
    }
}
