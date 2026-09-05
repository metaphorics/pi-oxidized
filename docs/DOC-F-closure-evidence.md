# DOC-F publication closure evidence

Issue: [#138](https://github.com/metaphorics/pi-oxidized/issues/138)

Evidence date: 2026-08-29

## Local verdict

The local DOC-F gates pass. The release consumer checks the current
documentation tree in every dry-run target archive. The compatibility generator
is byte-stable. All 77 ledger rows are `present` and share one fresh run ID. The
full-tree audit has no HIGH or MEDIUM finding. DOC-F remains open until the
release workflow produces and the consumer verifies all seven CI artifacts.

## Seven local archive consumers

`scripts/tests/release-docs.test.ts` runs the real
`scripts/package-release.ts --dry-run` path for each target. Each test verifies:

- the expected archive and its matching `.sha256` sidecar;
- the sidecar's exact checksum line against the archive bytes;
- `README.md`, `CHANGELOG.md`, `release.json`, and every current `docs/**` file;
- exact file size, SHA-256, and archived bytes for each documentation member;
- exact equality between the extracted archive tree and the manifest tree plus
  `release.json`.

| Rust target | Dry-run archive consumer |
|---|---|
| `x86_64-unknown-linux-gnu` | PASS |
| `x86_64-unknown-linux-musl` | PASS |
| `aarch64-unknown-linux-gnu` | PASS |
| `aarch64-unknown-linux-musl` | PASS |
| `x86_64-apple-darwin` | PASS |
| `aarch64-apple-darwin` | PASS |
| `x86_64-pc-windows-msvc` | PASS |

REL-DOCS closed the same release implementation and archive contract in
[#111](https://github.com/metaphorics/pi-oxidized/issues/111#issuecomment-5446802937).
REL-CLOSE consumed its release artifacts in
[#119](https://github.com/metaphorics/pi-oxidized/issues/119#issuecomment-5447640526).
The current DOC-F tree still requires its own seven CI artifacts. DOC-F changes
no file under `scripts/release/` or `.github/workflows/`.

## Final compatibility generation

DEPS-D1 handed off the final dependency and toolchain state at
`849122647411`. DOC-F ran the generator twice after that handoff.

| Input or output | SHA-256 |
|---|---|
| Committed `docs/compatibility.md` before generation | `e00d69e6178855e28c8352bc68fe2733ebe9f0a6e07d3b3a98da8e3fdec8da39` |
| First generation | `e00d69e6178855e28c8352bc68fe2733ebe9f0a6e07d3b3a98da8e3fdec8da39` |
| Second generation | `e00d69e6178855e28c8352bc68fe2733ebe9f0a6e07d3b3a98da8e3fdec8da39` |

`scripts/tests/generate-compat-docs.test.ts` now compares all three byte
sequences. A stale committed document cannot pass by producing two equal new
outputs.

## Doc-evidence run

| Field | Value |
|---|---|
| Schema | `pi.docs.evidence.run.v1` |
| Local run ID | `2026-08-29T14:32:01.743Z` |
| Rows | 77 |
| `present` rows | 77 |
| Canonical ledger hash | `1c1d5479cecca4568888b3929b7e0648ea653493f595f4890c8d8aa125e11ac3` |
| Ledger file SHA-256 | `b05b3948e6cec7294d8027abf33ebe567eed05482095754032dd32af0b0a459e` |
| Run manifest | `target/verification/docs-evidence/run-manifest.json` |

## Prerequisite closure evidence

DOC-F rechecked the seven required closures through their retained evidence and
fresh verifier runs. The hashes below bind this record to the consumed files.

| Gate | Consumed evidence | SHA-256 |
|---|---|---|
| PAR-CLOSE | `docs/PARITY_LEDGER.md`; `bun run verify:parity` | `e0070d2b0d2ae389957d48d474e730d061d6daa143f546cc23574085919cfd4e` |
| XC-CLOSE | `docs/xc-mutation-log.md`; extension compatibility witnesses | `c5b2925ac610a670c5baefa6f8cbaab3f37ce3e79a350b1d72b60916adbfe86a` |
| TUI-CLOSE | `docs/TUI-CLOSE-evidence.md`; terminal witness suites | `b773467509763d503f5e993c75b5575e4d03b80536177001fdd015f9f991f4c1` |
| PERF-CLOSE | `docs/performance/PERF-CLOSE-evidence.md`; retained benchmark and e2e artifacts | `586cefeca4259b961ea88a3e1dc9a3454c09e7c96173c68256a8ace521660502` |
| REL-CLOSE | `scripts/tests/compat-matrix.test.ts`; seven-row release matrix gate | `1b45f707e48c52650eb2cb89f2fdc0cceee70b91aac0d2a23607cd98b56b1dd1` |
| REL-DOCS | `scripts/tests/package-release.test.ts`; seven archive staging contract | `19eae76dc60c3dec1ab54f8fcdb0eaa06549dd730d5e1015f1c3a7a7005b43db` |
| DEPS-D1 | `docs/DEPS-R2-verdict-ledger.md`; exposure and SBOM self-checks | `603f0ac68090d704e2bd528496c47b285038f700fd9ba667ec37ea0333469bc5` |

The final-tree exposure reference was refreshed at `eb91d6b1d4fa` after the
performance benchmark added a shipped manifest edge. The self-check preserves
the DEPS-D1 classifications: `npm:typebox` is S, `npm:@types/bun` is E, and the
Bun runtime is S.

## Full-tree audit

[`DOC-F-sync-docs-audit.md`](DOC-F-sync-docs-audit.md) records the final audit:

- HIGH: 0
- MEDIUM: 0
- LOW: 3, all retained historical records with an owner and explicit
  disposition

The audit file SHA-256 is
`a44fc232cd6919fab7de99b7535b5b24d4b457374987cb19b22144bfe4060848`.

## Boundary

The DOC-F change contains no `examplesSource`, no `examples/` tree. Release
code remains owned by REL-DOCS and REL-CLOSE.

## Addendum: final-tree re-verification (2026-09-05)

The 08-29 record above is retained verbatim. After the intervening tree work
(overlay routing-token repair, model-runtime lock-order repair, testkit
hardening, transcript exit guards, exposure hydrated-data exclusion), every
locally-executable DOC-F item was re-run against the final tree:

| Item | Result |
|---|---|
| Seven dry-run archive consumers (`release-docs.test.ts`) | 42/42 pass, docs/ tree consumed incl. the new solution note |
| Compat-docs triple generation | byte-stable (`f0f6044e…`), tree clean |
| Generated fixtures (`check:fixtures`) | all FRESH after catalog regen 1337→1353 models (pin test follows) |
| Doc-evidence run | 77/77 present, runId `2026-09-05T11:00:15.970Z`, manifest emitted |
| Root README crate catalog | 5 rows = workspace members, verified |
| Exposure ledger | sanity rows re-emitted verbatim at `9d67f0eff96d` with re-anchor record; self-check 33/33 |
| SBOM baseline | `14ed8987` still describes the tree |
| Final-tree drift sweep | no new HIGH/MEDIUM: no doc states the old 1337 count, no doc states the old 2491-input projection as current (dated re-anchor records only), solution-note identifiers match production symbols, no new rust/ts fences under docs/ |

Remaining open: the seven CI artifacts (release workflow billing-locked per
[#138](https://github.com/metaphorics/pi-oxidized/issues/138)) and the musl
native-run leg (no loader on this host). Both are environmental, not tree state.

## Addendum 2 (2026-09-05): witness-accuracy re-capture + CI unblocked

- Exposure ledger: rows re-emitted at `ab1a3524f876` with the
  witness-accuracy record; the `9d67f0e` line above is superseded, and the
  DEPS-D1 digest `603f0ac6…` bound in the 08-29 body predates the `2a35e26`,
  `cc0181b`, `2be7ca2`, and `ab1a352` ledger rewrites. Current ledger digest:
  `53862c9db3a9f771…` (first 16 hex). Ledger chain itself carries the full
  per-era record; this addendum only re-binds the pointer.
- CI status correction: the "billing-locked" premise in Addendum 1 is dead —
  `actions/permissions` returns `enabled:true, allowed_actions:"all"`, and
  draft PR #181 (`release/0.1.1` → `main`) now runs all seven
  release-verification legs. The musl leg needs no host loader: the workflow
  builds its loader in-workflow (release-verification.yml:359-426); the "no
  loader on this host" line above describes local runs only.
- Line endings: `.gitattributes` pins `scripts/verification/fixtures/**` and
  `docs/PARITY_LEDGER.md` to LF — Windows checkouts otherwise break the
  exposure hash chain (proven by CRLF repro) and the content-addressed
  execution-map generations the Windows leg loads via `bun test scripts`.
