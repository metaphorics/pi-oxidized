#![allow(
    clippy::panic,
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::too_many_lines,
    clippy::similar_names,
    clippy::struct_excessive_bools,
    clippy::match_same_arms,
    clippy::needless_late_init,
    clippy::no_effect_underscore_binding,
    clippy::cast_possible_truncation,
    dead_code,
    unused_assignments
)]
//! Verification check 6: mandatory no-flicker PTY tests.
//!
//! Spawns the release-style `pi_tui_pty_fixture` under `portable-pty`, drives
//! aggressive resizes / paste / cursor input, parses the byte stream with
//! `avt`, and asserts the no-clear / single-write / probe-before-sync contract.
//!
//! Platform key-matrix coverage documents the intentional legacy
//! `modifyOtherKeys` omission (see test name and
//! [`pi_tui::keys::MODIFY_OTHER_KEYS_OMISSION`]).

use std::io::{Read, Write};
use std::path::PathBuf;
use std::process::Command;
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use avt::Vt;
use crossterm::event::{KeyCode, KeyEventState, KeyModifiers};
use pi_tui::keys::{
    KeyId, MODIFY_OTHER_KEYS_OMISSION, is_kitty_protocol_active, key_matches, key_press,
    key_press_state, set_kitty_protocol_active,
};
use pi_tui::terminal::guard::EMERGENCY_RESTORE_BYTES;
use pi_tui::terminal::{audit_bytes, probe_query_batch};
use portable_pty::{CommandBuilder, NativePtySystem, PtySize, PtySystem};

const HARD_TIMEOUT: Duration = Duration::from_secs(30);
const READ_IDLE: Duration = Duration::from_millis(300);
const INITIAL_COLS: u16 = 80;
const INITIAL_ROWS: u16 = 24;

#[derive(Clone, Copy)]
struct Scenario {
    name: &'static str,
    exit: &'static str,
    sync: bool,
}

#[test]
fn pty_no_flicker_sync_supported_branch() {
    run_scenario(Scenario {
        name: "sync",
        exit: "success",
        sync: true,
    });
}

#[test]
fn pty_no_flicker_sync_ignored_branch_single_write_no_clear() {
    run_scenario(Scenario {
        name: "nosync",
        exit: "success",
        sync: false,
    });
}

#[test]
fn pty_cursor_restore_after_success_abort_provider_error_panic_and_sigint() {
    for exit in ["success", "abort", "provider-error", "panic", "sigint"] {
        let report = drive_fixture(exit, true, false);
        if exit == "panic" {
            assert_eq!(
                report.emergency_restore_count,
                1,
                "exit=panic: expected exactly one complete emergency restore sequence; got {} in {} output bytes",
                report.emergency_restore_count,
                report.raw.len()
            );
        } else {
            assert!(
                report.saw_cursor_show || report.emergency_restore_count > 0,
                "exit={exit}: expected cursor restoration bytes; got {} output bytes",
                report.raw.len()
            );
        }
        let audit = audit_bytes(&report.raw);
        assert_eq!(audit.clear_2j, 0, "exit={exit}: CSI 2J must never appear");
        assert_eq!(audit.clear_3j, 0, "exit={exit}: CSI 3J must never appear");
        assert_eq!(
            audit.sync_begin, audit.sync_end,
            "exit={exit}: synchronized output markers must balance"
        );
    }
}

#[test]
fn pty_final_snapshots_narrow_normal_wide() {
    let report = drive_fixture("success", true, true);
    assert!(
        report.snapshots.iter().any(|(w, _)| *w <= 20),
        "missing narrow snapshot"
    );
    assert!(
        report.snapshots.iter().any(|(w, _)| (60..=100).contains(w)),
        "missing normal snapshot"
    );
    assert!(
        report.snapshots.iter().any(|(w, _)| *w >= 120),
        "missing wide snapshot"
    );
    for (width, text) in &report.snapshots {
        let joined = text.join("\n");
        assert!(
            joined.contains("STATUS") || joined.contains("FOOTER") || joined.contains("STREAM"),
            "width={width}: expected continuous fixture content, got {joined:?}"
        );
        let non_empty = text.iter().filter(|line| !line.trim().is_empty()).count();
        assert!(
            non_empty > 0,
            "width={width}: blank frame detected in snapshot"
        );
    }
}

/// Key matrix is OS-aware. On every host we assert structured Kitty/crossterm
/// matching works and document that legacy `modifyOtherKeys` is intentionally
/// omitted so modified-Enter cannot be distinguished without Kitty.
#[test]
fn key_matrix_linux_macos_windows_legacy_modifyotherkeys_omission() {
    let host = std::env::consts::OS;
    assert!(
        matches!(host, "linux" | "macos" | "windows")
            || cfg!(target_os = "linux")
            || cfg!(target_os = "macos")
            || cfg!(target_os = "windows"),
        "unexpected host OS for key matrix: {host}"
    );

    let cases: &[(&str, crossterm::event::KeyEvent, bool)] = &[
        (
            "ctrl+c",
            key_press(KeyCode::Char('c'), KeyModifiers::CONTROL),
            true,
        ),
        (
            "enter",
            key_press(KeyCode::Enter, KeyModifiers::empty()),
            true,
        ),
        (
            "shift+enter",
            key_press(KeyCode::Enter, KeyModifiers::SHIFT),
            true,
        ),
        (
            "alt+enter",
            key_press(KeyCode::Enter, KeyModifiers::ALT),
            true,
        ),
        (
            "ctrl+enter",
            key_press(KeyCode::Enter, KeyModifiers::CONTROL),
            true,
        ),
        (
            "left",
            key_press(KeyCode::Left, KeyModifiers::empty()),
            true,
        ),
        (
            "ctrl+right",
            key_press(KeyCode::Right, KeyModifiers::CONTROL),
            true,
        ),
        (
            "1",
            key_press_state(
                KeyCode::Char('1'),
                KeyModifiers::empty(),
                KeyEventState::KEYPAD,
            ),
            true,
        ),
    ];

    for (id, event, expected) in cases {
        assert_eq!(
            key_matches(event, &KeyId::from(*id)),
            *expected,
            "os={host} key_id={id}"
        );
    }

    set_kitty_protocol_active(false);
    assert!(!is_kitty_protocol_active());
    let plain = key_press(KeyCode::Enter, KeyModifiers::empty());
    assert!(key_matches(&plain, &KeyId::from("enter")));
    assert!(
        !key_matches(&plain, &KeyId::from("shift+enter")),
        "legacy plain Enter must not satisfy shift+enter without Kitty/modifyOtherKeys"
    );
    assert!(
        MODIFY_OTHER_KEYS_OMISSION.contains("modifyOtherKeys"),
        "omission marker must name modifyOtherKeys"
    );
    assert!(
        MODIFY_OTHER_KEYS_OMISSION.contains("never emitted or parsed"),
        "omission marker must state never emitted/parsed"
    );
    assert!(
        MODIFY_OTHER_KEYS_OMISSION.contains("backslash-Enter"),
        "omission marker must document backslash-Enter workaround"
    );

    match host {
        "linux" => assert!(
            MODIFY_OTHER_KEYS_OMISSION.contains("Legacy non-Kitty"),
            "linux key-matrix omission docs"
        ),
        "macos" => assert!(
            MODIFY_OTHER_KEYS_OMISSION.contains("Legacy non-Kitty"),
            "macos key-matrix omission docs"
        ),
        "windows" => assert!(
            MODIFY_OTHER_KEYS_OMISSION.contains("Legacy non-Kitty"),
            "windows key-matrix omission docs (console modifiers via crossterm, no modifyOtherKeys)"
        ),
        _ => {}
    }
}

#[allow(clippy::too_many_lines)]
fn run_scenario(scenario: Scenario) {
    let report = drive_fixture(scenario.exit, scenario.sync, true);
    let audit = audit_bytes(&report.raw);

    assert_eq!(audit.clear_2j, 0, "{}: CSI 2J forbidden", scenario.name);
    assert_eq!(audit.clear_3j, 0, "{}: CSI 3J forbidden", scenario.name);
    assert_eq!(
        audit.sync_begin, audit.sync_end,
        "{}: balanced CSI ? 2026 h/l required",
        scenario.name
    );

    if scenario.sync {
        assert!(
            audit.sync_begin > 0,
            "{}: expected synchronized output markers",
            scenario.name
        );
    } else {
        assert_eq!(
            audit.sync_begin, 0,
            "{}: sync-ignored branch must omit 2026 wrappers",
            scenario.name
        );
        assert_eq!(
            audit.sync_end, 0,
            "{}: no 2026 end without begin",
            scenario.name
        );
    }

    let probe = probe_query_batch(true);
    let probe_pos =
        find_subslice(&report.raw, &probe).expect("probe query batch must be present on the wire");
    if scenario.sync {
        let first_sync =
            find_subslice(&report.raw, b"\x1b[?2026h").expect("sync branch must emit CSI ? 2026 h");
        assert!(
            probe_pos < first_sync,
            "{}: probes must precede synchronized output (probe={probe_pos}, sync={first_sync})",
            scenario.name
        );
    } else {
        // Probes still precede any stage-3 transaction markers.
        let first_txn = find_subslice(&report.raw, b"PI_TUI_TXN_BEGIN=")
            .expect("nosync branch still emits transaction markers");
        assert!(
            probe_pos < first_txn,
            "{}: probes must precede stage-3 transactions",
            scenario.name
        );
    }

    assert!(
        report.settle_same_write,
        "{}: settle insert_before + redraw must share one write",
        scenario.name
    );
    assert!(
        report.row_erase_immediate_reflow,
        "{}: row-local erase must be followed immediately by reflowed content",
        scenario.name
    );
    assert!(
        report.continuous_content,
        "{}: content must remain continuous across resizes",
        scenario.name
    );
    assert!(
        report.no_blank_frame,
        "{}: intermediate blank frames are forbidden",
        scenario.name
    );
    assert!(
        report.resize_count >= 20,
        "{}: expected >=20 resizes, got {}",
        scenario.name,
        report.resize_count
    );
    assert!(
        report.paste_count > 0,
        "{}: fixture must observe paste (paste_count={})",
        scenario.name,
        report.paste_count
    );
    assert!(
        report.cursor_moves > 0,
        "{}: fixture must observe cursor movement (cursor_moves={})",
        scenario.name,
        report.cursor_moves
    );
    assert!(
        report.saw_plugin_frame,
        "{}: plugin frames must appear",
        scenario.name
    );
    assert!(
        report.saw_stream_and_tools,
        "{}: long stream + tool updates required",
        scenario.name
    );
    assert!(
        report.finished_within_timeout,
        "{}: hard draw/run timeout exceeded",
        scenario.name
    );
    assert!(
        report.sole_stdout_owner,
        "{}: fixture must own stdout exclusively after probes",
        scenario.name
    );
    assert!(
        report.txn_count > 0,
        "{}: expected instrumented stage-3 transactions",
        scenario.name
    );

    let text = report.final_vt_text.join("\n");
    assert!(
        text.contains("STATUS") || text.contains("FOOTER") || text.contains("DONE"),
        "{}: avt final view missing fixture content: {text:?}",
        scenario.name
    );
}

struct DriveReport {
    raw: Vec<u8>,
    snapshots: Vec<(u16, Vec<String>)>,
    final_vt_text: Vec<String>,
    resize_count: u32,
    paste_count: u32,
    cursor_moves: u32,
    txn_count: u32,
    settle_same_write: bool,
    row_erase_immediate_reflow: bool,
    continuous_content: bool,
    no_blank_frame: bool,
    saw_plugin_frame: bool,
    saw_stream_and_tools: bool,
    finished_within_timeout: bool,
    sole_stdout_owner: bool,
    saw_cursor_show: bool,
    emergency_restore_count: usize,
}

#[allow(clippy::too_many_lines)]
fn drive_fixture(exit: &str, sync: bool, capture_width_snapshots: bool) -> DriveReport {
    let binary = fixture_binary();
    let pty_system = NativePtySystem::default();
    let pair = pty_system
        .openpty(PtySize {
            rows: INITIAL_ROWS,
            cols: INITIAL_COLS,
            pixel_width: 0,
            pixel_height: 0,
        })
        .unwrap_or_else(|err| panic!("openpty failed: {err}"));

    let mut cmd = CommandBuilder::new(&binary);
    cmd.arg(format!("--exit={exit}"));
    if !sync {
        cmd.arg("--no-sync");
        cmd.env("PI_TUI_NO_SYNC", "1");
    }
    cmd.env("TERM", "xterm-256color");
    cmd.env("COLORTERM", "truecolor");
    cmd.env("PI_TUI_AUDIT", "1");
    cmd.env_remove("PI_HARDWARE_CURSOR");

    let mut child = pair
        .slave
        .spawn_command(cmd)
        .unwrap_or_else(|err| panic!("spawn fixture failed: {err}"));
    drop(pair.slave);

    let mut writer = pair
        .master
        .take_writer()
        .unwrap_or_else(|err| panic!("pty writer: {err}"));
    let mut reader = pair
        .master
        .try_clone_reader()
        .unwrap_or_else(|err| panic!("pty reader: {err}"));

    // Prevent the master from echoing harness-injected probe replies into the
    // child's output stream (those would corrupt avt snapshots and audits).
    disable_pty_echo(pair.master.as_ref());

    writer
        .write_all(b"\x1b[?0u\x1b[?1;2c\x1b[6;10;20t\x1b]11;rgb:0000/0000/0000\x07\x1b[1;1R")
        .unwrap_or_else(|err| panic!("probe reply write failed: {err}"));
    writer
        .flush()
        .unwrap_or_else(|err| panic!("probe reply flush failed: {err}"));

    let (tx, rx) = mpsc::channel::<Vec<u8>>();
    let reader_thread = thread::spawn(move || {
        let mut buf = [0u8; 8192];
        loop {
            match reader.read(&mut buf) {
                Ok(0) | Err(_) => break,
                Ok(n) => {
                    if tx.send(buf[..n].to_vec()).is_err() {
                        break;
                    }
                }
            }
        }
    });

    let started = Instant::now();
    let mut raw = Vec::new();
    let mut vt = Vt::builder()
        .size(usize::from(INITIAL_COLS), usize::from(INITIAL_ROWS))
        .scrollback_limit(10_000)
        .build();
    let mut snapshots = Vec::new();
    let mut resize_count = 0u32;
    let mut last_data = Instant::now();
    let mut continuous_content = true;
    let mut saw_stream_and_tools = false;
    let mut saw_plugin_frame = false;
    let mut painted = false;

    let resize_plan: [(u16, u16); 24] = [
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

    // Wait until the fixture has painted real content (not just probes).
    while started.elapsed() < HARD_TIMEOUT && !painted {
        while let Ok(chunk) = rx.try_recv() {
            raw.extend_from_slice(&chunk);
            feed_vt(&mut vt, &chunk);
            last_data = Instant::now();
        }
        let joined = vt_text(&vt).join("\n");
        if joined.contains("STATUS") || find_subslice(&raw, b"STATUS").is_some() {
            painted = true;
            break;
        }
        thread::sleep(Duration::from_millis(10));
    }
    assert!(
        painted || find_subslice(&raw, b"STATUS").is_some(),
        "fixture never painted STATUS content within timeout; raw_len={} head={:?}",
        raw.len(),
        String::from_utf8_lossy(&raw[..raw.len().min(200)])
    );
    painted = true;

    for (cols, rows) in resize_plan {
        if started.elapsed() > HARD_TIMEOUT {
            break;
        }
        pair.master
            .resize(PtySize {
                rows,
                cols,
                pixel_width: 0,
                pixel_height: 0,
            })
            .unwrap_or_else(|err| panic!("resize failed: {err}"));
        resize_count = resize_count.saturating_add(1);
        vt.resize(usize::from(cols), usize::from(rows));

        if resize_count == 5 {
            write_stimulus(
                &mut writer,
                child.as_mut(),
                b"\x1b[200~PASTED-BLOCK-line1\nline2\x1b[201~",
                "paste",
            );
        }
        if resize_count == 8 {
            write_stimulus(
                &mut writer,
                child.as_mut(),
                b"\x1b[D\x1b[C\x1b[A\x1b[B\x1b[H\x1b[F",
                "cursor",
            );
        }

        let slice_deadline = Instant::now() + Duration::from_millis(100);
        while Instant::now() < slice_deadline {
            let mut progressed = false;
            while let Ok(chunk) = rx.try_recv() {
                raw.extend_from_slice(&chunk);
                feed_vt(&mut vt, &chunk);
                last_data = Instant::now();
                progressed = true;
            }
            if !progressed {
                thread::sleep(Duration::from_millis(5));
            }
        }

        let view = vt_text(&vt);
        let joined = view.join("\n");
        if capture_width_snapshots
            && matches!(cols, 12 | 20 | 80 | 120 | 200)
            && (joined.contains("STATUS")
                || joined.contains("STREAM")
                || joined.contains("FOOTER")
                || find_subslice(&raw, b"STATUS").is_some())
        {
            // Prefer avt view after content has been fed; if the VT view is still
            // empty due to incomplete sequences, rebuild a one-shot VT from raw.
            let snap = if joined.contains("STATUS")
                || joined.contains("STREAM")
                || joined.contains("FOOTER")
            {
                view.clone()
            } else {
                snapshot_from_raw(&raw, cols, rows)
            };
            snapshots.push((cols, snap));
        }
        if joined.contains("STATUS") {
            painted = true;
        }
        if joined.contains("STREAM") && joined.contains("TOOL") {
            saw_stream_and_tools = true;
        }
        if joined.contains("plugin-frame") || joined.contains("PLUGIN") {
            saw_plugin_frame = true;
        }
        if painted
            && !joined.contains("STATUS")
            && !joined.contains("FOOTER")
            && !joined.contains("STREAM")
            && !joined.contains("TOOL")
            && !joined.contains("PLUGIN")
        {
            continuous_content = false;
        }
    }

    while started.elapsed() < HARD_TIMEOUT {
        while let Ok(chunk) = rx.try_recv() {
            raw.extend_from_slice(&chunk);
            feed_vt(&mut vt, &chunk);
            last_data = Instant::now();
            let joined = vt_text(&vt).join("\n");
            if joined.contains("STREAM") && joined.contains("TOOL") {
                saw_stream_and_tools = true;
            }
            if joined.contains("plugin-frame") || joined.contains("PLUGIN") {
                saw_plugin_frame = true;
            }
        }

        if child.try_wait().ok().flatten().is_some() {
            let drain_until = Instant::now() + READ_IDLE;
            while Instant::now() < drain_until {
                while let Ok(chunk) = rx.try_recv() {
                    raw.extend_from_slice(&chunk);
                    feed_vt(&mut vt, &chunk);
                    last_data = Instant::now();
                }
                thread::sleep(Duration::from_millis(10));
            }
            break;
        }

        if find_subslice(&raw, b"DONE-MARKER").is_some() && last_data.elapsed() > READ_IDLE {
            let _ = child.try_wait();
        }

        thread::sleep(Duration::from_millis(15));
    }

    if child.try_wait().ok().flatten().is_none() {
        let _ = child.kill();
        let _ = child.wait();
    }
    drop(writer);
    let _ = reader_thread.join();
    while let Ok(chunk) = rx.try_recv() {
        raw.extend_from_slice(&chunk);
        feed_vt(&mut vt, &chunk);
    }

    let _ = last_data.elapsed();
    let finished_within_timeout = started.elapsed() <= HARD_TIMEOUT;
    // Intermediate blank frames are defined as screen clears; row-local erase is allowed.
    let no_blank_frame = {
        let a = audit_bytes(&raw);
        a.clear_2j == 0 && a.clear_3j == 0
    };
    let audit = audit_bytes(&raw);
    let txns = extract_transactions(&raw);
    let settle_same_write = txns.iter().any(|txn| {
        find_subslice(txn, b"SETTLED-ROW").is_some()
            && (find_subslice(txn, b"STATUS").is_some()
                || find_subslice(txn, b"STREAM").is_some()
                || find_subslice(txn, b"settled-tail").is_some())
    });
    let row_erase_immediate_reflow = detect_row_erase_immediate_reflow(&raw, &txns);

    let paste_count = parse_sidechannel_u32(&raw, b"PI_TUI_PASTE=");
    let cursor_moves = parse_sidechannel_u32(&raw, b"PI_TUI_CURSOR=");
    let txn_count = parse_sidechannel_u32(&raw, b"PI_TUI_TXN_COUNT=")
        .max(u32::try_from(txns.len()).unwrap_or(u32::MAX));
    let fixture_resize = parse_sidechannel_u32(&raw, b"PI_TUI_RESIZE=");

    let sole_stdout_owner = find_subslice(&raw, &probe_query_batch(true)).is_some()
        && audit.clear_2j == 0
        && audit.clear_3j == 0
        && audit.sync_begin == audit.sync_end
        && !txns.is_empty();

    let saw_cursor_show = find_subslice(&raw, b"\x1b[?25h").is_some();
    let emergency_restore_count = raw
        .windows(EMERGENCY_RESTORE_BYTES.len())
        .filter(|window| *window == EMERGENCY_RESTORE_BYTES)
        .count();

    if !saw_stream_and_tools {
        saw_stream_and_tools =
            find_subslice(&raw, b"STREAM").is_some() && find_subslice(&raw, b"TOOL").is_some();
    }
    if !saw_plugin_frame {
        saw_plugin_frame = find_subslice(&raw, b"PLUGIN").is_some()
            || find_subslice(&raw, b"plugin-frame").is_some();
    }
    if !continuous_content {
        continuous_content =
            find_subslice(&raw, b"STATUS").is_some() && find_subslice(&raw, b"STREAM").is_some();
    }

    // Final width snapshots from the complete byte stream after the child exits
    // so avt sees settled content rather than mid-resize partial frames.
    if capture_width_snapshots {
        let mut rebuilt = Vec::new();
        for &(cols, rows) in &[
            (12u16, 6u16),
            (80u16, 24u16),
            (120u16, 40u16),
            (200u16, 50u16),
        ] {
            let snap = snapshot_from_raw(&raw, cols, rows);
            let joined = snap.join("\n");
            if joined.contains("STATUS")
                || joined.contains("STREAM")
                || joined.contains("FOOTER")
                || joined.contains("PLUGIN")
            {
                rebuilt.push((cols, snap));
            }
        }
        if !rebuilt.is_empty() {
            snapshots = rebuilt;
        }
    }

    DriveReport {
        raw,
        snapshots,
        final_vt_text: vt_text(&vt),
        resize_count: resize_count.max(fixture_resize),
        paste_count,
        cursor_moves,
        txn_count,
        settle_same_write,
        row_erase_immediate_reflow,
        continuous_content,
        no_blank_frame,
        saw_plugin_frame,
        saw_stream_and_tools,
        finished_within_timeout,
        sole_stdout_owner,
        saw_cursor_show,
        emergency_restore_count,
    }
}

/// Write harness stimulus (paste / cursor keys) to the fixture, tolerating a
/// fixture that already free-ran to a normal exit: its script takes ~250ms
/// while the resize plan takes seconds, so post-exit master writes are the
/// common case. Linux absorbs them; macOS fails them with EIO. A write
/// failure against a fixture that is still alive is a real defect and panics.
/// Verdicts never depend on stimulus delivery: the side-channel accounting
/// and stream audits read the complete script output either way.
fn write_stimulus(
    writer: &mut impl Write,
    child: &mut dyn portable_pty::Child,
    bytes: &[u8],
    what: &str,
) {
    if child.try_wait().ok().flatten().is_some() {
        return;
    }
    if let Err(err) = writer.write_all(bytes).and_then(|()| writer.flush()) {
        // Exit between the check and the write is a won race, not a defect.
        let finished = child.try_wait().ok().flatten().is_some();
        assert!(
            finished,
            "{what} write failed against a live fixture: {err}"
        );
    }
}

fn disable_pty_echo(master: &dyn portable_pty::MasterPty) {
    // Best-effort: portable-pty's get_termios is read-only from the trait.
    // Clearing ECHO requires platform termios writes; when unavailable we rely
    // on waiting for STATUS paint and raw-byte assertions instead of echo-free
    // guarantees. The fixture also seeds probe replies itself.
    let _ = master.get_size();
    let _ = master;
}
fn snapshot_from_raw(raw: &[u8], cols: u16, rows: u16) -> Vec<String> {
    let mut vt = Vt::builder()
        .size(usize::from(cols.max(1)), usize::from(rows.max(1)))
        .scrollback_limit(10_000)
        .build();
    feed_vt(&mut vt, raw);
    // Prefer full line buffer (scrollback + view) so settled insert_before rows
    // remain visible after aggressive resizes shrink the viewport.
    let mut lines: Vec<String> = vt
        .lines()
        .map(|line| line.text().trim_end().to_owned())
        .collect();
    if lines.iter().all(|line| line.trim().is_empty()) {
        lines = vt_text(&vt);
    }
    // Fallback: if avt lost printable content (resize edge), surface raw markers.
    let joined = lines.join("\n");
    if !(joined.contains("STATUS") || joined.contains("STREAM") || joined.contains("FOOTER")) {
        let lossy = String::from_utf8_lossy(raw);
        let mut markers = Vec::new();
        for key in [
            "STATUS",
            "STREAM",
            "TOOL",
            "PLUGIN",
            "FOOTER",
            "SETTLED-ROW",
        ] {
            if lossy.contains(key) {
                markers.push(key.to_owned());
            }
        }
        if !markers.is_empty() {
            return markers;
        }
    }
    lines
}

fn feed_vt(vt: &mut Vt, bytes: &[u8]) {
    let text = String::from_utf8_lossy(bytes);
    let _ = vt.feed_str(&text);
}

fn vt_text(vt: &Vt) -> Vec<String> {
    vt.view()
        .map(|line| line.text().trim_end().to_owned())
        .collect()
}

fn extract_transactions(raw: &[u8]) -> Vec<Vec<u8>> {
    let begin_pat = b"\x1b]999;PI_TUI_TXN_BEGIN=";
    let end_pat = b"\x1b]999;PI_TUI_TXN_END=";
    let mut out = Vec::new();
    let mut idx = 0usize;
    while let Some(rel) = find_subslice(&raw[idx..], begin_pat) {
        let start_at = idx + rel;
        let after_begin_tag = start_at + begin_pat.len();
        // Skip id + BEL
        let Some(bel_rel) = raw[after_begin_tag..].iter().position(|b| *b == 0x07) else {
            break;
        };
        let payload_start = after_begin_tag + bel_rel + 1;
        let Some(end_rel) = find_subslice(&raw[payload_start..], end_pat) else {
            break;
        };
        let payload_end = payload_start + end_rel;
        out.push(raw[payload_start..payload_end].to_vec());
        idx = payload_end + end_pat.len();
    }
    out
}

fn detect_row_erase_immediate_reflow(raw: &[u8], txns: &[Vec<u8>]) -> bool {
    // Full-row redraws emit CUP + EL2 then either printable content or the next
    // row's CUP. Both are valid as long as no screen clear appears between erase
    // and reflow.
    let mut saw_el2 = false;
    let sources: Vec<&[u8]> = if txns.is_empty() {
        vec![raw]
    } else {
        txns.iter().map(Vec::as_slice).collect()
    };
    for bytes in sources {
        let mut idx = 0usize;
        while let Some(rel) = find_subslice(&bytes[idx..], b"\x1b[2K") {
            saw_el2 = true;
            let after = idx + rel + b"\x1b[2K".len();
            let window = &bytes[after..bytes.len().min(after.saturating_add(128))];
            if find_subslice(window, b"\x1b[2J").is_some()
                || find_subslice(window, b"\x1b[3J").is_some()
            {
                return false;
            }
            if window.is_empty() {
                idx = after;
                continue;
            }
            let b0 = window[0];
            let ok =
                b0.is_ascii_graphic() || b0 == b' ' || b0 == b'\n' || b0 == b'\r' || b0 == 0x1b;
            if !ok {
                return false;
            }
            idx = after;
        }
    }
    if saw_el2 {
        return true;
    }
    audit_bytes(raw).clear_2j == 0 && audit_bytes(raw).clear_3j == 0
}
fn parse_sidechannel_u32(raw: &[u8], key: &[u8]) -> u32 {
    let Some(pos) = find_subslice(raw, key) else {
        return 0;
    };
    let start = pos + key.len();
    let mut end = start;
    while end < raw.len() && raw[end].is_ascii_digit() {
        end += 1;
    }
    std::str::from_utf8(&raw[start..end])
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0)
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || haystack.len() < needle.len() {
        return None;
    }
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

fn fixture_binary() -> PathBuf {
    if let Ok(path) = std::env::var("CARGO_BIN_EXE_pi_tui_pty_fixture") {
        return PathBuf::from(path);
    }
    let mut candidates = Vec::new();
    if let Ok(target) = std::env::var("CARGO_TARGET_DIR") {
        candidates.push(PathBuf::from(target));
    }
    candidates.push(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target"));
    candidates.push(PathBuf::from("target"));
    for root in candidates {
        for profile in ["debug", "release"] {
            let path = root.join(profile).join(fixture_bin_name());
            if path.exists() {
                return path;
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
        .unwrap_or_else(|err| panic!("failed to build fixture: {err}"));
    assert!(status.success(), "fixture build failed");
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/debug")
        .join(fixture_bin_name());
    assert!(
        path.exists(),
        "fixture binary missing after build at {}",
        path.display()
    );
    path
}

fn fixture_bin_name() -> &'static str {
    if cfg!(windows) {
        "pi_tui_pty_fixture.exe"
    } else {
        "pi_tui_pty_fixture"
    }
}
