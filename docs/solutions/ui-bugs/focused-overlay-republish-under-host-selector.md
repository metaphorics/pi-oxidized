---
title: "Preserve Extension Overlay Focus Across Host Preemption"
date: "2026-09-05"
category: "ui-bugs"
module: "crates/pi/src/modes/interactive"
problem_type: "ui_bug"
component: "overlay-focus"
symptoms:
  - "A republished extension overlay stayed visible but stopped receiving keys after a host selector closed"
  - "Visual focus returned to the overlay while its logical routing token remained empty"
root_cause: "logic_error"
resolution_type: "code_fix"
severity: "high"
tags:
  - "tui"
  - "focus-management"
  - "overlay"
  - "extension-slots"
  - "key-routing"
---

# Preserve Extension Overlay Focus Across Host Preemption

## Problem

An extension overlay can own input before a host selector opens. The extension can then publish a replacement for the same slot while the selector owns input. The replacement must not take input from the selector, but it must remain the logical owner that receives input after the selector closes.

The runtime stored these two facts in separate fields:

- `view.focus` identified the surface that currently received input.
- `focused_extension_slot` identified the extension slot that could receive routed input.

The replacement path treated the publish as a new focus request. It disposed the old slot and cleared `focused_extension_slot`. The host selector correctly kept `view.focus`, so the replacement did not acquire the token again. When the selector closed, `view.focus` returned to the overlay, but the routing token stayed empty.

## Symptoms

- `host_selector_outranks_focused_extension_slot` failed after the selector closed:

  ```text
  assertion `left == right` failed: republished overlay must regain the extension routing ownership token
    left: None
   right: Some("overlay.grab")
  ```

- The overlay was visible and had `FocusArea::Overlay`, but `route_extension_input` could not find a focused extension slot.

## What Didn't Work

- Disposal followed by ordinary focus acquisition was not sufficient. The host correctly blocked focus acquisition while it owned input.
- Restoring only `view.focus` in `close_selector` was not sufficient. Visual focus does not identify which extension slot owned routing before the host preempted it.
- Giving every focusable publish a token would be wrong. A new or previously unfocused slot could then steal logical ownership while a host surface was active.

## Solution

Capture whether the replaced key owns the logical token before disposal. Restore that token only when the replacement has the same key and still captures focus. Do not change the current host-owned `view.focus`.

```
let replacement_had_focus_token =
    self.focused_extension_slot.as_deref() == Some(slot.key.as_str());
self.dispose_extension_slot(&slot.key);

// Project the replacement.

if takes_focus {
    self.focused_extension_slot = Some(slot.key.clone());
    self.view.focus = if slot.placement == SlotPlacement::Overlay {
        FocusArea::Overlay
    } else {
        FocusArea::Widget
    };
} else if captures_focus && replacement_had_focus_token {
    self.focused_extension_slot = Some(slot.key.clone());
}
```

The regression test uses this order:

1. Publish generation 1 of a focusable extension overlay.
2. Open a host selector.
3. Publish generation 2 with the same slot key.
4. Close the selector with `Esc`.
5. Require both `FocusArea::Overlay` and `focused_extension_slot == Some("overlay.grab")`.

The test failed before the repair and passed after the repair.

## Why This Works

Host preemption changes the active input surface. It does not transfer the prior extension ownership to another extension slot. A same-key publish is a replacement of the owner, not a new contender for ownership.

The preserved token stays inactive while the selector is open. `route_extension_input` also calls `extension_slot_owns_focus`, which requires the matching active `FocusArea`. The selector therefore keeps input until it closes. A different key cannot acquire the token because `replacement_had_focus_token` is false.

## Prevention

- Model active surface focus and logical routing ownership as separate state-machine facts.
- Preserve ownership across an in-place replacement. Do not reacquire it as if the replacement were a new component.
- Test the full ordering: focus, host preemption, asynchronous same-key replacement, host close, and routed input ownership.
- Test the inverse case so a new or unfocused slot cannot acquire ownership during host preemption.
