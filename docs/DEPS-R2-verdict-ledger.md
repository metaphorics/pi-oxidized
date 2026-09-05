# DEPS-R2 verdict ledger

Per-change record of every shipped-exposure classification made under the DEPS-R2
remediation runbook (`docs/DEPS-R2-remediation-runbook.md`, issue #128). This is the
DEPS-R2 view of the campaign's invariant ledger required by EXT-23: every shipped-input
change commit carries either seven-target evidence or a complete E1–E4 exemption bundle,
and the corresponding row here is the citation. Verdicts are **per change** — no package
carries a permanent class label; a package reclassified after a runtime import or field
move immediately takes its new verdict.

## Row format

Rows are emitted by the checker itself and pasted verbatim:

```
bun run verify:dependency-exposure classify --subject <kind:name> \
  --reference scripts/verification/fixtures/dependency-exposure/reference --emit-ledger-row
```

| Column | Meaning |
|---|---|
| `head` | Commit the classification ran against. For sanity rows this is the reference capture head (the classification describes the captured baseline); for live remediation rows, the post-fix commit the verdict was recorded on. |
| `date` | Reference capture date (the baseline the verdict was decided against). |
| `subject` | `npm:<name>` / `crate:<name>` / `tool:rust-toolchain\|bun-runtime\|bun-bundler`. |
| `class` | `S` (full seven-target post-audit) or `E` (complete E1–E4 bundle; only the lane is skippable). |
| `checks` | `E1..E4` statuses; any `fail` **or** `undecidable` forces `S`. |

Every live row must be accompanied by an entry in the records list below (advisory/yank
citation, gates actually run, commit SHA of the remediation). A row without its record
entry is an audit failure at DEPS-D1.

## Ledger

| head | date | subject | class | checks |
|---|---|---|---|---|
| a8896826f4b0 | 2026-09-05 | npm:typebox | S | E1:fail E2:fail E3:pass E4:pass |
| a8896826f4b0 | 2026-09-05 | npm:@types/bun | E | E1:pass E2:pass E3:pass E4:pass |
| a8896826f4b0 | 2026-09-05 | tool:bun-runtime | S | E1:pass E2:fail E3:fail E4:fail |

## Records
- **a8896826f4b0 / platform-fix re-capture (2026-09-05) — sanity rows re-decided, not new verdicts.** The CI platform batch (repo-wide LF pin, posix surface paths, exec-bit host gate in `stage.ts`) changed an authority module, invalidating the bundle pin by design; every non-musl leg failed the self-check on authority drift. Mechanical regeneration at `a889682` captured 2,451 metafile inputs with captureHead `a8896826f4b0`; all three classes stayed unchanged. `verify:dependency-exposure` 33/33 green.

- **ab1a3524f876 / witness-accuracy re-capture (2026-09-05) — sanity rows re-decided, not new verdicts.** The `cc0181b` record named a captureHead that cannot reproduce its own bundle: the tmpdir staging change landed in `2be7ca2` while the capture ran with the script dirty, and the capture tool did not count itself as a relevant pathspec so its own assert let it pass. The tool now lists its own path, and mechanical regeneration at `ab1a352` captured 2,451 metafile inputs (zero `providers/data/` paths, no staging residue) with captureHead `ab1a3524f876`; all three classes stayed unchanged. `verify:dependency-exposure` 33/33 green. Superseded same-day by the platform-fix re-capture above, whose head carries the CI platform batch.

- **cc0181bc46b4 / staging-hygiene re-capture (2026-09-05) — sanity rows re-decided, not new verdicts.** Council review found the `9d67f0e` bundle pinned the author's absolute checkout path in its argv (in-repo `.capture-staging`) and a use-after-head rows/bundle mismatch. The capture now stages under the OS temp dir, leaving no residue and no checkout path in the bundle; mechanical regeneration captured 2,451 metafile inputs with captureHead `cc0181bc46b4`; all three classes stayed unchanged. `verify:dependency-exposure` 33/33 green. Superseded same-day by the witness-accuracy re-capture above, whose captureHead names the head that actually contains the staging change.

- **9d67f0eff96d / hydrated-data exclusion re-anchor (2026-09-05) — sanity rows re-decided, not new verdicts.** The checked-in reference (last captured at `2c944d965b96`) pinned live-hydrated catalog JSON under the reference data dir (gitignored upstream, rewritten wholesale by any hydration); `openrouter.json` drifted twice in one session and the fail-closed self-check emitted E2-undecidable for `npm:@types/bun`, forcing Class S — a stale-pin artifact, not a real classification. Durable fix per the G3 open item: the capture now skips the generated data dir (same doctrine as the existing `.manifest.json` carve-out; model-list data carries no exposure signal), then mechanical regeneration captured 2,451 metafile inputs (bundle captureHead `cd80b8d`; rows re-emitted at `9d67f0eff96d`); all three classes stayed unchanged. `verify:dependency-exposure` 33/33 green, SBOM baseline green (no drift). Superseded same-day by the staging-hygiene re-capture below, which also moved capture staging out of the repo tree.

- **2c944d965b96 / reference re-anchor (2026-09-03) — sanity rows re-decided, not new verdicts.** The checked-in reference (last captured at `eb91d6b1d4fa`) trailed the current tree: the metafile projection pinned `openrouter.json` at sha `a9a1e3cf…` but the file on disk hashes `f370fd84…`. The fail-closed self-check emitted E2-undecidable for `npm:@types/bun` (stale metafile input), which the verdict algebra forces to Class S — this was a stale-reference artifact, not a real classification. Mechanical regeneration captured 2,490 metafile inputs at `2c944d965b96`; all three classes stayed unchanged. SBOM baseline `verify:sbom` green (no drift).

- **eb91d6b1d4fa / DOC-F final-tree re-ground (2026-08-29) — sanity rows
  re-decided, not new verdicts.** The final parity and performance work added a
  benchmark manifest edge after DEPS-D1 and restored the canonical reference
  checkout. The fail-closed self-check rejected the stale projection.
  Mechanical regeneration captured 2,491 metafile inputs at `eb91d6b1d4fa`;
  all three classes stayed unchanged.

- **849122647411 / DEPS-D1 closing re-ground (2026-08-28) — sanity rows re-decided, not new verdicts.** The
  DEPS-D1 closing git-log audit found the checked-in reference (last captured at `7f325058`) trailing the
  post-Bin-M shipped-input commits (`4ee9916` serde-rc, `75996e1` toolchain
  re-pin to the Rust floor registered in [compatibility.md § Engine Floors](compatibility.md#engine-floors)): the fail-closed checker
  classified every npm subject Class S on rust-input drift and the checked-in self-check went red — the
  reference-refresh-law state (`DEPS-R2-remediation-runbook.md` §5), resolved by mechanical regeneration
  (`16d2ad9`), never by widening the gate. The three anchor rows above were re-emitted by the checker at
  capture head 8491226 (2491 metafile inputs); classes are identical to the b90362dc-era rows. SBOM baseline
  re-anchored in the same closing pass (`80ac57f`; delta = exactly the toolchain channel re-pin).

- **b90362dc / npm:typebox — sanity, not a remediation.** Known-member anchor from the
  checker `self-check` at DEPS-R2 landing: production-field position
  (`packages/extension-host/package.json` `dependencies`, pre and post) and bundled into
  the shipped sidecar (metafile inputs under `.references/pi-2.0/node_modules/typebox/`).
  Any future typebox remediation is Class S: full seven-target lane including both musl
  per-artifact proofs.
- **b90362dc / npm:@types/bun — sanity, its recorded verdict.** Complete E1–E4 bundle:
  devDependencies-only across all three surfaces, zero of the 2493 metafile inputs, no
  shipped-byte-producing invocation, none of the staged inputs. A future @types/bun bump
  may skip only the seven-target lane; lockfile law, advisory scans, and SBOM diff still
  apply. (In the scheduled Bin M epoch it nonetheless keeps its full seven-target gate —
  zero scheduled epoch member is pre-classified exempt; Class E exists only for
  execution-time out-of-band/lifecycle changes carrying this evidence bundle.)
- **b90362dc / tool:bun-runtime — sanity.** Bun embedded-runtime bumps change the
  compiled sidecar bytes and stage the runtime into the runtime-bundle archive: Class S
  by definition.
