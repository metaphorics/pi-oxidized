#!/usr/bin/env bun
/**
 * Doc-evidence ledger checker entrypoint (DOC-A, issue #129).
 *
 * Loads the docs-evidence ledger, validates the schema (no command/argv
 * strings, every row carries an owner and a closed class, exactly one
 * reference-pin literal), runs each row through its closed evidence-class
 * runner, checks sidecar staleness (contentHash + toolVersion + runId), and
 * writes fresh sidecar artifacts under target/verification/docs-evidence/.
 * A clean run also emits run-manifest.json (schema pi.docs.evidence.run.v1)
 * beside the sidecars; any stale manifest is removed up front so a failing
 * run can never leave one behind.
 *
 * The ledger must contain exactly 77 rows and match the inventory artifact
 * (scripts/verification/fixtures/docs-inventory.json).
 *
 * Usage:
 *   bun run scripts/verification/docs-evidence.ts
 *   bun run scripts/verification/docs-evidence.ts --ledger <path> --sidecar-dir <dir> --inventory <path>
 */

import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { isAbsolute, join, resolve } from "node:path";

import {
	CANONICAL_REFERENCE_SHA,
	LEGACY_REFERENCE_SHA,
	RETIRED_REFERENCE_SHA,
	assertCanonicalReference,
} from "../reference-identity.ts";
import {
	DEFAULT_REPROOF_INTERVAL_MS,
	FORBIDDEN_FIELDS,
	RUN_MANIFEST_SCHEMA,
	TOOL_VERSION,
	type LedgerRow,
	type RunManifest,
	type RunManifestEntry,
	type RunnerResult,
	type Sidecar,
	checkStaleness,
	isEvidenceClass,
	isEvidenceStatus,
	runEvidence,
	sha256,
	scanForExampleProductImports,
} from "./docs-evidence-runners.ts";

export const REPO_ROOT = resolve(import.meta.dirname, "../..");

export const DEFAULT_LEDGER_PATH = "scripts/verification/docs-evidence.json";
export const DEFAULT_INVENTORY_PATH = "scripts/verification/fixtures/docs-inventory.json";
export const DEFAULT_SIDECAR_DIR = "target/verification/docs-evidence";
export const EXPECTED_LEDGER_ROW_COUNT = 77;

export const SENTINEL_OK = "DOCS_EVIDENCE_OK";

/** Filename of the run manifest emitted beside the sidecars on a clean run. */
export const RUN_MANIFEST_FILENAME = "run-manifest.json";

// ---------------------------------------------------------------------------
// CLI parsing
// ---------------------------------------------------------------------------

interface CliArgs {
	readonly ledgerPath: string;
	readonly inventoryPath: string;
	readonly sidecarDir: string;
}

function parseArgs(argv: readonly string[]): CliArgs {
	const args: CliArgs = {
		ledgerPath: DEFAULT_LEDGER_PATH,
		inventoryPath: DEFAULT_INVENTORY_PATH,
		sidecarDir: DEFAULT_SIDECAR_DIR,
	};
	for (let i = 0; i < argv.length; i++) {
		const flag = argv[i];
		const next = argv[i + 1];
		if (flag === "--ledger" && next) {
			(args as { ledgerPath: string }).ledgerPath = next;
			i++;
		} else if (flag === "--inventory" && next) {
			(args as { inventoryPath: string }).inventoryPath = next;
			i++;
		} else if (flag === "--sidecar-dir" && next) {
			(args as { sidecarDir: string }).sidecarDir = next;
			i++;
		}
	}
	return args;
}

// ---------------------------------------------------------------------------
// Ledger loading and validation
// ---------------------------------------------------------------------------

export interface Ledger {
	readonly schema: string;
	readonly referencePin: string;
	readonly rows: readonly LedgerRow[];
}

export interface InventoryArtifact {
	readonly schema: string;
	readonly categories: readonly {
		readonly id: string;
		readonly name: string;
		readonly surfaces: readonly string[];
	}[];
}

export function loadLedger(root: string, relPath: string): Ledger {
	// relative() across Windows drives yields an absolute path; honor it
	// instead of joining it onto the root into garbage.
	const abs = isAbsolute(relPath) ? relPath : join(root, relPath);
	if (!existsSync(abs)) {
		throw new Error(`ledger not found: ${relPath}`);
	}
	const raw = JSON.parse(readFileSync(abs, "utf8")) as Record<string, unknown>;
	if (raw["schema"] !== "pi.docs.evidence.v1") {
		throw new Error(`ledger schema mismatch: ${String(raw["schema"])}`);
	}
	if (typeof raw["referencePin"] !== "string") {
		throw new Error("ledger missing referencePin");
	}
	const rows = raw["rows"];
	if (!Array.isArray(rows)) {
		throw new Error("ledger rows is not an array");
	}
	return {
		schema: raw["schema"] as string,
		referencePin: raw["referencePin"] as string,
		rows: rows as LedgerRow[],
	};
}

export function loadInventory(root: string, relPath: string): InventoryArtifact {
	const abs = isAbsolute(relPath) ? relPath : join(root, relPath);
	if (!existsSync(abs)) {
		throw new Error(`inventory not found: ${relPath}`);
	}
	const raw = JSON.parse(readFileSync(abs, "utf8")) as Record<string, unknown>;
	if (raw["schema"] !== "pi.docs.inventory.v1") {
		throw new Error(`inventory schema mismatch: ${String(raw["schema"])}`);
	}
	return raw as unknown as InventoryArtifact;
}

/** Count total surfaces in the inventory artifact. */
export function inventorySurfaceCount(inv: InventoryArtifact): number {
	let count = 0;
	for (const cat of inv.categories) {
		count += cat.surfaces.length;
	}
	return count;
}

// ---------------------------------------------------------------------------
// Schema validation
// ---------------------------------------------------------------------------

export interface ValidationProblem {
	readonly rowId: string;
	readonly message: string;
}

export function validateLedger(ledger: Ledger): readonly ValidationProblem[] {
	const problems: ValidationProblem[] = [];

	// Reference-pin literal check
	if (ledger.referencePin !== CANONICAL_REFERENCE_SHA) {
		problems.push({
			rowId: "(ledger)",
			message: `referencePin is ${ledger.referencePin}, expected ${CANONICAL_REFERENCE_SHA}`,
		});
	}
	if (ledger.referencePin === LEGACY_REFERENCE_SHA) {
		problems.push({
			rowId: "(ledger)",
			message: `referencePin is the legacy pin ${LEGACY_REFERENCE_SHA}; the legacy checkout is not an active ledger identity`,
		});
	}
	if (ledger.referencePin === RETIRED_REFERENCE_SHA) {
		problems.push({
			rowId: "(ledger)",
			message: `referencePin is the retired pin ${RETIRED_REFERENCE_SHA}; retired identity must never reappear in an active ledger`,
		});
	}

	const seenIds = new Set<string>();
	for (const row of ledger.rows) {
		// Unique id
		if (seenIds.has(row.id)) {
			problems.push({ rowId: row.id, message: "duplicate row id" });
		}
		seenIds.add(row.id);

		// Owner required
		if (typeof row.owner !== "string" || row.owner.length === 0) {
			problems.push({ rowId: row.id, message: "missing owner" });
		}

		// Lifecycle statuses remain representable, but only a fully present
		// ledger can produce a successful closure manifest.
		if (!isEvidenceStatus(row.status)) {
			problems.push({
				rowId: row.id,
				message: `unknown or missing status: ${String(row.status)}`,
			});
		} else if (row.status !== "present") {
			problems.push({
				rowId: row.id,
				message: `status ${row.status} is not final`,
			});
		}

		// The target names the exact registered surface; a second path would
		// create an unchecked mapping and let the manifest attest the wrong bytes.
		if (typeof row.target !== "string" || row.target.length === 0) {
			problems.push({ rowId: row.id, message: "missing target" });
		} else if (row.target !== row.surface) {
			problems.push({
				rowId: row.id,
				message: `target ${row.target} does not match surface ${row.surface}`,
			});
		}

		// Closed class required
		if (!isEvidenceClass(row.class)) {
			problems.push({
				rowId: row.id,
				message: `unknown or missing evidence class: ${String(row.class)}`,
			});
		}

		// No forbidden fields (command/argv strings)
		const rowRecord = row as unknown as Record<string, unknown>;
		for (const field of FORBIDDEN_FIELDS) {
			if (Object.hasOwn(rowRecord, field) || Object.hasOwn(row.params, field)) {
				problems.push({
					rowId: row.id,
					message: `forbidden field present: ${field}`,
				});
			}
		}

		// Params must be an object
		if (typeof row.params !== "object" || row.params === null) {
			problems.push({ rowId: row.id, message: "params is not an object" });
		}
	}

	return problems;
}

// ---------------------------------------------------------------------------
// Sidecar I/O
// ---------------------------------------------------------------------------

function sidecarPath(sidecarDir: string, rowId: string): string {
	return join(sidecarDir, `${rowId}.json`);
}

function readPriorSidecar(sidecarDir: string, rowId: string): Sidecar | null {
	const p = sidecarPath(sidecarDir, rowId);
	if (!existsSync(p)) return null;
	try {
		return JSON.parse(readFileSync(p, "utf8")) as Sidecar;
	} catch {
		return null;
	}
}

function writeSidecar(sidecarDir: string, sidecar: Sidecar): void {
	mkdirSync(sidecarDir, { recursive: true });
	writeFileSync(sidecarPath(sidecarDir, sidecar.rowId), JSON.stringify(sidecar, null, 2) + "\n");
}

// ---------------------------------------------------------------------------
// Run manifest
// ---------------------------------------------------------------------------

/**
 * Deterministic JSON serialization of a loaded value: object keys sorted
 * recursively, arrays kept in order, no whitespace. Used to hash the ledger.
 */
export function canonicalJson(value: unknown): string {
	if (value === null || typeof value !== "object") {
		return JSON.stringify(value) ?? "null";
	}
	if (Array.isArray(value)) {
		return `[${value.map(canonicalJson).join(",")}]`;
	}
	const record = value as Record<string, unknown>;
	const body = Object.keys(record)
		.sort()
		.map((k) => `${JSON.stringify(k)}:${canonicalJson(record[k])}`)
		.join(",");
	return `{${body}}`;
}

/** Drop any prior run manifest so a failed run cannot leave one falsely current. */
function removeRunManifest(sidecarDir: string): void {
	const manifestPath = join(sidecarDir, RUN_MANIFEST_FILENAME);
	if (existsSync(manifestPath)) {
		rmSync(manifestPath);
	}
}

/**
 * Build the manifest for a completed run: one entry per ledger row with the
 * fresh sidecar contentHash, sorted by rowId. Written only when the run had
 * no problems.
 */
function buildRunManifest(
	ledger: Ledger,
	contentHashByRowId: ReadonlyMap<string, string>,
	runId: string,
): RunManifest {
	const entries: RunManifestEntry[] = [];
	for (const row of ledger.rows) {
		const contentHash = contentHashByRowId.get(row.id);
		if (contentHash === undefined) continue;
		entries.push({ rowId: row.id, status: row.status, contentHash });
	}
	entries.sort((a, b) => (a.rowId < b.rowId ? -1 : a.rowId > b.rowId ? 1 : 0));
	return {
		schema: RUN_MANIFEST_SCHEMA,
		runId,
		referencePin: ledger.referencePin,
		ledgerHash: sha256(canonicalJson(ledger)),
		rowCount: ledger.rows.length,
		presentCount: ledger.rows.filter((row) => row.status === "present").length,
		entries,
	};
}

/** Write the run manifest and return its path. */
function writeRunManifest(sidecarDir: string, manifest: RunManifest): string {
	mkdirSync(sidecarDir, { recursive: true });
	const manifestPath = join(sidecarDir, RUN_MANIFEST_FILENAME);
	writeFileSync(manifestPath, JSON.stringify(manifest, null, 2) + "\n");
	return manifestPath;
}

export function loadRunManifest(manifestPath: string): RunManifest {
	const raw = JSON.parse(readFileSync(manifestPath, "utf8")) as Record<string, unknown>;
	const entries = raw["entries"];
	const entriesValid =
		Array.isArray(entries) &&
		entries.length === EXPECTED_LEDGER_ROW_COUNT &&
		entries.every((entry) => {
			if (entry === null || typeof entry !== "object" || Array.isArray(entry)) return false;
			const record = entry as Record<string, unknown>;
			return (
				typeof record["rowId"] === "string" &&
				record["status"] === "present" &&
				typeof record["contentHash"] === "string" &&
				/^[0-9a-f]{64}$/.test(record["contentHash"])
			);
		});
	const rowIds = entriesValid
		? new Set((entries as RunManifestEntry[]).map(({ rowId }) => rowId))
		: new Set<string>();
	if (
		raw["schema"] !== RUN_MANIFEST_SCHEMA ||
		typeof raw["runId"] !== "string" ||
		Number.isNaN(Date.parse(raw["runId"])) ||
		raw["referencePin"] !== CANONICAL_REFERENCE_SHA ||
		typeof raw["ledgerHash"] !== "string" ||
		!/^[0-9a-f]{64}$/.test(raw["ledgerHash"]) ||
		raw["rowCount"] !== EXPECTED_LEDGER_ROW_COUNT ||
		raw["presentCount"] !== EXPECTED_LEDGER_ROW_COUNT ||
		!entriesValid ||
		rowIds.size !== EXPECTED_LEDGER_ROW_COUNT
	) {
		throw new Error(`invalid docs-evidence run manifest: ${manifestPath}`);
	}
	return raw as unknown as RunManifest;
}

// ---------------------------------------------------------------------------
// Main check loop
// ---------------------------------------------------------------------------

export interface CheckResult {
	readonly ok: boolean;
	readonly problems: readonly string[];
	readonly sidecars: readonly Sidecar[];
	readonly manifestPath: string | null;
}

/**
 * Run the full doc-evidence check: validate the ledger, run each evidence
 * class, check sidecar staleness, and enforce the fixed and inventory counts.
 */
export function runCheck(
	ledger: Ledger,
	inventory: InventoryArtifact,
	root: string,
	sidecarDir: string,
	runId: string,
	requiredRowCount: number = EXPECTED_LEDGER_ROW_COUNT,
): CheckResult {
	// Drop any prior run manifest first: a failing run must never leave one.
	removeRunManifest(sidecarDir);

	const problems: string[] = [];
	const sidecars: Sidecar[] = [];
	const contentHashByRowId = new Map<string, string>();

	// 1. Schema validation
	const validationProblems = validateLedger(ledger);
	for (const vp of validationProblems) {
		problems.push(`[validation] ${vp.rowId}: ${vp.message}`);
	}

	// 2. Row count vs fixed contract and inventory
	if (ledger.rows.length !== requiredRowCount) {
		problems.push(
			`[inventory] ledger has ${ledger.rows.length} rows, contract requires ${requiredRowCount}`,
		);
	}
	const inventoryCount = inventorySurfaceCount(inventory);
	if (ledger.rows.length !== inventoryCount) {
		problems.push(
			`[inventory] ledger has ${ledger.rows.length} rows, inventory has ${inventoryCount} surfaces`,
		);
	}

	// 2b. Scan for disguised example-product imports (DOC-G2 adversarial hardening)
	const importFindings = scanForExampleProductImports(root);
	for (const finding of importFindings) {
		problems.push(`[example-product-import] ${finding}`);
	}

	// 3. Run each row's evidence class runner
	for (const row of ledger.rows) {
		let result: RunnerResult;
		try {
			result = runEvidence(row, root, runId);
		} catch (error) {
			const detail = error instanceof Error ? error.message : String(error);
			problems.push(`[runner] ${row.id}: runner threw: ${detail}`);
			continue;
		}

		if (!result.ok) {
			for (const p of result.problems) {
				problems.push(`[runner] ${p}`);
			}
		}

		// 4. Check staleness against prior sidecar
		const prior = readPriorSidecar(sidecarDir, row.id);
		if (prior !== null) {
			const staleness = checkStaleness(prior, result.sidecar, DEFAULT_REPROOF_INTERVAL_MS);
			for (const reason of staleness.reasons) {
				problems.push(`[stale] ${reason}`);
			}
		}

		sidecars.push(result.sidecar);
		contentHashByRowId.set(row.id, result.sidecar.contentHash);
	}

	// 5. Write fresh sidecars (even if there were problems, so the next run can compare)
	for (const sc of sidecars) {
		writeSidecar(sidecarDir, sc);
	}

	// 6. Emit the run manifest only for a clean run: the stale manifest was
	// removed at run start, so a failing run leaves none behind.
	let manifestPath: string | null = null;
	if (problems.length === 0) {
		manifestPath = writeRunManifest(sidecarDir, buildRunManifest(ledger, contentHashByRowId, runId));
	}

	return { ok: problems.length === 0, problems, sidecars, manifestPath };
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

async function main(): Promise<void> {
	const args = parseArgs(process.argv.slice(2));
	const root = REPO_ROOT;
	const sidecarDir = resolve(root, args.sidecarDir);
	removeRunManifest(sidecarDir);

	let ledger: Ledger;
	let inventory: InventoryArtifact;
	try {
		ledger = loadLedger(root, args.ledgerPath);
		inventory = loadInventory(root, args.inventoryPath);
	} catch (error) {
		const detail = error instanceof Error ? error.message : String(error);
		console.error(`docs-evidence: failed to load inputs: ${detail}`);
		process.exit(1);
	}

	// Fail closed before any current evidence reads: the canonical reference
	// checkout must sit at the pinned commit.
	try {
		assertCanonicalReference(root);
	} catch (error) {
		const detail = error instanceof Error ? error.message : String(error);
		console.error(`docs-evidence: ${detail}`);
		process.exit(1);
	}

	const runId = new Date().toISOString();
	const result = runCheck(ledger, inventory, root, sidecarDir, runId);

	if (result.ok) {
		// One line: sentinel, runId, row count, manifest path — CI logs can
		// locate the emitted run-manifest.json directly.
		process.stdout.write(
			`${SENTINEL_OK} runId=${runId} rows=${ledger.rows.length} manifest=${result.manifestPath ?? "<missing>"}\n`,
		);
		return;
	}

	console.error(`docs-evidence: ${result.problems.length} problem(s):`);
	for (const p of result.problems) {
		console.error(`  - ${p}`);
	}
	process.exit(1);
}

if (import.meta.main) main();
