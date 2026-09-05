#!/usr/bin/env bun
/**
 * Alignment witness suite (VER-ALIGN, issue #145).
 *
 * Freezes the verification/workflow reference pin and the live canonical
 * reference checkout (`.references/pi-2.0`) to the canonical baseline SHA,
 * checks that the portable seven-tool schema selection contract still holds
 * against the canonical registry surface, and runs the exact-file legacy
 * classifier: retired reference identity survives only inside the
 * enumerated historical witnesses this module allows — never in active
 * paths or closure inputs.
 *
 * Identity constants (canonical root/SHA, legacy root/SHA, retired SHA)
 * come from scripts/reference-identity.ts, the dependency-free authority
 * leaf; this module never re-exports them.
 */

import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { join, resolve } from "node:path";
import {
	EXECUTION_MAP_CURRENT_PATH,
	EXECUTION_MAP_GENERATIONS_DIRECTORY,
	computeExecutionMapGenerationId,
	extractExecutionMapBundle,
	isExecutionMapGenerationPath,
} from "./map.ts";
import { REQUIRED_TOOL_NAMES, loadCanonicalToolRegistry, selectPortableToolParameters } from "../generate-tool-schemas.ts";
import {
	CANONICAL_REFERENCE_ROOT,
	CANONICAL_REFERENCE_SHA,
	LEGACY_REFERENCE_ROOT,
	LEGACY_REFERENCE_SHA,
	LEGACY_REFERENCE_SHA_SHORT,
	RETIRED_REFERENCE_SHA,
	RETIRED_REFERENCE_SHA_SHORT,
	assertCanonicalReference,
	readReferenceHead,
} from "../reference-identity.ts";

export const REPO_ROOT = resolve(import.meta.dirname, "../..");

/** Declarative carriers that must spell the canonical identity directly. */
export const PIN_LITERAL_PATHS = [
	".github/workflows/release-verification.yml",
	".github/workflows/musl-bakeoff.yml",
	"scripts/reference-identity.ts",
] as const;

const PIN_LITERAL_OCCURRENCES: Readonly<
	Record<(typeof PIN_LITERAL_PATHS)[number], { readonly sha: number; readonly root: number }>
> = {
	".github/workflows/release-verification.yml": { sha: 2, root: 4 },
	".github/workflows/musl-bakeoff.yml": { sha: 4, root: 4 },
	"scripts/reference-identity.ts": { sha: 1, root: 2 },
};

/** Ledger carrier whose referencePin must equal the canonical SHA exactly. */
export const LEDGER_CARRIER_PATH = "scripts/verification/docs-evidence.json";

/** Classifier fixture witness: the alignment suite's own regression literals. */
export const ALIGNMENT_POLICY_PATH = "scripts/verification/alignment.ts";
export const CLASSIFIER_FIXTURE_PATH = "scripts/verification/alignment.test.ts";

/** Inline marker every counted legacy occurrence must carry on its line. */
export const HISTORICAL_LABEL = "historical witness";

export interface AlignmentInputs {
	readonly files: Readonly<Record<string, string>>;
	readonly referenceHead: string;
	readonly registryTools: Readonly<Record<string, unknown>>;
	readonly trackedFiles: Readonly<Record<string, string>>;
}

/** Fail when an owned carrier mis-pins the reference identity. */
export function verifyPinLiterals(files: Readonly<Record<string, string>>): string[] {
	const problems: string[] = [];
	const retiredShas = [LEGACY_REFERENCE_SHA, RETIRED_REFERENCE_SHA];
	for (const path of PIN_LITERAL_PATHS) {
		const body = files[path];
		if (body === undefined) {
			problems.push(`${path} is not readable`);
			continue;
		}
		const expected = PIN_LITERAL_OCCURRENCES[path];
		const shaOccurrences = body.split(CANONICAL_REFERENCE_SHA).length - 1;
		if (shaOccurrences !== expected.sha) {
			problems.push(
				`${path} must carry ${CANONICAL_REFERENCE_SHA} exactly ${expected.sha} time(s); found ${shaOccurrences}`,
			);
		}
		const rootOccurrences = body.split(CANONICAL_REFERENCE_ROOT).length - 1;
		if (rootOccurrences !== expected.root) {
			problems.push(
				`${path} must carry ${CANONICAL_REFERENCE_ROOT} exactly ${expected.root} time(s); found ${rootOccurrences}`,
			);
		}
		if (path !== "scripts/reference-identity.ts") {
			for (const retired of retiredShas) {
				if (body.includes(retired)) {
					problems.push(`${path} still contains retired reference SHA ${retired}`);
				}
			}
		}
	}
	const ledger = files[LEDGER_CARRIER_PATH];
	if (ledger === undefined) {
		problems.push(`${LEDGER_CARRIER_PATH} is not readable`);
	} else {
		let referencePin: unknown;
		try {
			referencePin = (JSON.parse(ledger) as Record<string, unknown>)["referencePin"];
		} catch (error) {
			problems.push(
				`${LEDGER_CARRIER_PATH} is not valid JSON: ${error instanceof Error ? error.message : String(error)}`,
			);
		}
		if (referencePin !== undefined && referencePin !== CANONICAL_REFERENCE_SHA) {
			problems.push(
				`${LEDGER_CARRIER_PATH} referencePin is ${String(referencePin)}, expected ${CANONICAL_REFERENCE_SHA}`,
			);
		}
		const occurrences = ledger.split(CANONICAL_REFERENCE_SHA).length - 1;
		if (occurrences !== 1) {
			problems.push(
				`${LEDGER_CARRIER_PATH} must carry ${CANONICAL_REFERENCE_SHA} exactly once (referencePin); found ${occurrences}`,
			);
		}
	}
	return problems;
}

/** Fail when the checked-out canonical reference is not at the baseline. */
export function verifyReferenceCheckout(headSha: string): string[] {
	if (headSha !== CANONICAL_REFERENCE_SHA) {
		return [
			`${CANONICAL_REFERENCE_ROOT} HEAD is ${headSha === "" ? "(missing or unreadable)" : headSha}, expected ${CANONICAL_REFERENCE_SHA}`,
		];
	}
	return [];
}

/**
 * Fail when the registry cannot supply the seven portable tools, or when
 * selection accidentally keeps a reference-only platform tool.
 */
export function verifyPortableToolSelection(registryTools: Readonly<Record<string, unknown>>): string[] {
	const problems: string[] = [];
	let selected: Record<string, unknown>;
	try {
		selected = selectPortableToolParameters({ ...registryTools });
	} catch (error) {
		const detail = error instanceof Error ? error.message : String(error);
		return [`portable tool selection failed: ${detail}`];
	}
	const selectedNames = Object.keys(selected).sort();
	const expected = [...REQUIRED_TOOL_NAMES].sort();
	if (selectedNames.join("\0") !== expected.join("\0")) {
		problems.push(
			`portable tool selection mismatch (expected ${expected.join(", ")}; got ${selectedNames.join(", ")})`,
		);
	}
	for (const name of Object.keys(registryTools)) {
		if (!(REQUIRED_TOOL_NAMES as readonly string[]).includes(name) && Object.hasOwn(selected, name)) {
			problems.push(`portable tool selection retained reference-only tool ${name}`);
		}
	}
	return problems;
}

/** Canonical checkout HEAD, or "" when missing/unreadable (witness reports it). */
export function readCanonicalReferenceHead(root: string): string {
	try {
		return readReferenceHead(join(root, CANONICAL_REFERENCE_ROOT));
	} catch {
		return "";
	}
}

// ---------------------------------------------------------------------------
// Legacy identity classifier (exact-file historical witness boundary)
// ---------------------------------------------------------------------------

/** Match kinds the classifier recognizes for retired reference identity. */
export type LegacyMatchKind =
	| "legacy-root-direct"
	| "legacy-root-split"
	| "legacy-sha-full"
	| "legacy-sha-short"
	| "retired-sha-full"
	| "retired-sha-short";

export interface LegacyOccurrence {
	readonly kind: LegacyMatchKind;
	readonly line: number;
}

export interface LegacyAllowance {
	/** Short classification of the historical witness. */
	readonly label: string;
	/** Why this file may still carry retired identity. */
	readonly reason: string;
	/** Historical witnesses never back active evidence or closure inputs. */
	readonly closureEligible: false;
	/** Immutable whole-file SHA-256 digest (exact witness). */
	readonly digest?: string;
	/** Exact per-kind occurrence counts (live fixture witness). */
	readonly counts?: Readonly<Partial<Record<LegacyMatchKind, number>>>;
}

const escapeRegExp = (value: string): string => value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

const LEGACY_SHA_FULL_RE = new RegExp(escapeRegExp(LEGACY_REFERENCE_SHA), "g");
const LEGACY_SHA_SHORT_RE = new RegExp(`(?<![0-9A-Fa-f])${LEGACY_REFERENCE_SHA_SHORT}(?![0-9A-Fa-f])`, "g");
const RETIRED_SHA_FULL_RE = new RegExp(escapeRegExp(RETIRED_REFERENCE_SHA), "g");
const RETIRED_SHA_SHORT_RE = new RegExp(`(?<![0-9A-Fa-f])${RETIRED_REFERENCE_SHA_SHORT}(?![0-9A-Fa-f])`, "g");
const LEGACY_ROOT_DIRECT_RE = new RegExp(`${escapeRegExp(LEGACY_REFERENCE_ROOT)}(?![0-9A-Za-z_-])`, "g");

/** The reference root without its leaf segment, for split-form detection. */
const LEGACY_ROOT_HEAD = LEGACY_REFERENCE_ROOT.slice(0, LEGACY_REFERENCE_ROOT.length - "/pi".length);

/** Split root forms: the retired root reassembled from string pieces. */
const LEGACY_ROOT_SPLIT_RES: readonly RegExp[] = [
	// join(repo, ".references", "pi") // historical witness
	new RegExp(`${escapeRegExp(LEGACY_ROOT_HEAD)}["']\\s*,\\s*["']pi["']`, "g"),
	// join(repo, "references", "pi") — dotless first segment // historical witness
	new RegExp(`["']references["']\\s*,\\s*["']pi["']`, "g"),
	// ".references/" + "pi" // historical witness
	new RegExp(`${escapeRegExp(LEGACY_ROOT_HEAD)}\\/["']\\s*\\+\\s*["']pi["']`, "g"),
	// ".references" + "/pi" // historical witness
	new RegExp(`${escapeRegExp(LEGACY_ROOT_HEAD)}["']\\s*\\+\\s*["']\\/pi["']`, "g"),
];

/**
 * Exact per-file allowances for retired identity. Every entry is a narrow
 * historical witness with closureEligible false: nothing here may back
 * active evidence, closure inputs, or any current DOC-F / PERF-CLOSE
 * source. Anything not enumerated below fails closed.
 */
export const LEGACY_ALLOWANCES: Readonly<Record<string, LegacyAllowance>> = {
	[ALIGNMENT_POLICY_PATH]: {
		label: "legacy classifier patterns",
		reason:
			"the classifier must represent split legacy roots to reject them in every other tracked file",
		closureEligible: false,
		counts: {
			"legacy-root-split": 4,
		},
	},
	[CLASSIFIER_FIXTURE_PATH]: {
		label: "alignment regression fixtures",
		reason:
			"the classifier's own suite pins each retired identity literal once so a corrupted reference-identity module cannot redefine history silently",
		closureEligible: false,
		counts: {
			"legacy-root-direct": 1,
			"legacy-root-split": 1,
			"legacy-sha-full": 1,
			"legacy-sha-short": 1,
			"retired-sha-full": 1,
			"retired-sha-short": 1,
		},
	},
	"scripts/reference-identity.ts": {
		label: "reference identity rejection literals",
		reason:
			"the authority leaf must name each rejected identity so active consumers can fail closed without duplicating those literals",
		closureEligible: false,
		counts: {
			"legacy-root-direct": 2,
			"legacy-sha-full": 1,
			"retired-sha-full": 1,
		},
	},
	"docs/PERF-R2-workload-surface-ranking.md": {
		label: "historical performance ranking",
		reason:
			"the ranking records measurements made against the legacy checkout and is excluded from current performance closure",
		closureEligible: false,
		counts: {
			"legacy-root-direct": 6,
			"legacy-sha-full": 1,
		},
	},
	"docs/PERF-R8-paired-baselines.md": {
		label: "historical paired baselines",
		reason:
			"the paired baseline records legacy measurements and is excluded from current performance closure",
		closureEligible: false,
		counts: {
			"legacy-root-direct": 4,
			"legacy-sha-full": 1,
		},
	},
	"docs/performance/t11-iterations.md": {
		label: "historical optimization iterations",
		reason:
			"the iteration log preserves commands for legacy measurements and is excluded from current performance closure",
		closureEligible: false,
		counts: {
			"legacy-root-direct": 6,
		},
	},
	"docs/performance/floors/memory-resource-units.md": {
		label: "historical memory floor",
		reason:
			"the memory floor preserves its legacy measurement source and is excluded from current performance closure",
		closureEligible: false,
		counts: {
			"legacy-root-direct": 1,
		},
	},
};

/** Current workstream sources that must never consume a legacy witness. */
const CLOSURE_SOURCE_RES: readonly (readonly [workstream: string, re: RegExp])[] = [
	["DOC-F", /^docs\/DOC-F-/],
	["PERF-CLOSE", /^docs\/performance\/PERF-CLOSE-/],
];

/** Classify every retired-identity occurrence in a text body. */
export function scanLegacyIdentity(content: string): LegacyOccurrence[] {
	const occurrences: LegacyOccurrence[] = [];
	const occupied: Array<readonly [number, number]> = [];
	const lineAt = (index: number): number => content.slice(0, index).split("\n").length;
	const record = (pattern: RegExp, kind: LegacyMatchKind): void => {
		for (const match of content.matchAll(pattern)) {
			const start = match.index;
			if (start === undefined) continue;
			const end = start + match[0].length;
			if (occupied.some(([from, to]) => start < to && from < end)) continue;
			occupied.push([start, end]);
			occurrences.push({ kind, line: lineAt(start) });
		}
	};
	record(LEGACY_SHA_FULL_RE, "legacy-sha-full");
	record(RETIRED_SHA_FULL_RE, "retired-sha-full");
	record(LEGACY_ROOT_DIRECT_RE, "legacy-root-direct");
	for (const pattern of LEGACY_ROOT_SPLIT_RES) record(pattern, "legacy-root-split");
	record(LEGACY_SHA_SHORT_RE, "legacy-sha-short");
	record(RETIRED_SHA_SHORT_RE, "retired-sha-short");
	return occurrences;
}

/** Collect the historical-witness paths a closure source depends on. */
function historicalWitnessReferences(content: string, allowances: Readonly<Record<string, LegacyAllowance>>): string[] {
	const references = Object.keys(allowances).filter((witness) => content.includes(witness));
	if (content.includes(EXECUTION_MAP_CURRENT_PATH)) references.push(EXECUTION_MAP_CURRENT_PATH);
	for (const match of content.matchAll(new RegExp(`${escapeRegExp(EXECUTION_MAP_GENERATIONS_DIRECTORY)}/[0-9a-f]{64}\\.md`, "g"))) {
		references.push(match[0]);
	}
	return references;
}

/**
 * Exact-file legacy classifier over tracked text. Retired identity is
 * accepted only inside an enumerated allowance (exact counts on labelled
 * lines, or the pinned whole-file digest), or the canonical-witness JSON of
 * a digest-verified, strictly parsed immutable map generation. Unknown files,
 * extra or unlabelled occurrences, stale allowances, digest drift, and any
 * DOC-F or PERF-CLOSE source consuming a historical witness are violations.
 */
export function verifyLegacyIdentity(
	trackedFiles: Readonly<Record<string, string>>,
	allowances: Readonly<Record<string, LegacyAllowance>> = LEGACY_ALLOWANCES,
): string[] {
	const problems: string[] = [];
	const paths = Object.keys(trackedFiles).sort();
	for (const path of paths) {
		const content = trackedFiles[path];
		if (content === undefined) continue;
		const publicationRelativePath = path.startsWith(`${EXECUTION_MAP_GENERATIONS_DIRECTORY}/`)
			? `generations/${path.slice(EXECUTION_MAP_GENERATIONS_DIRECTORY.length + 1)}`
			: "";
		if (isExecutionMapGenerationPath(publicationRelativePath)) {
			const generationId = publicationRelativePath.slice("generations/".length, -3);
			const actualId = computeExecutionMapGenerationId(content);
			if (actualId !== generationId) {
				problems.push(`${path} generation digest drift: expected ${generationId}, found ${actualId}`);
				continue;
			}
			try {
				const bundle = extractExecutionMapBundle(content);
				for (const occurrence of scanLegacyIdentity(bundle.mapText)) {
					problems.push(`unclassified legacy ${occurrence.kind} occurrence at ${path}:${occurrence.line}`);
				}
			} catch (error) {
				problems.push(`${path} generation bundle is malformed: ${String(error)}`);
			}
			continue;
		}
		const occurrences = scanLegacyIdentity(content);
		const closureSource = CLOSURE_SOURCE_RES.find(([, re]) => re.test(path));
		if (closureSource !== undefined) {
			const witnessRefs = historicalWitnessReferences(content, allowances);
			if (occurrences.length > 0 || witnessRefs.length > 0) {
				const detail = [
					...occurrences.map((occurrence) => `${occurrence.kind}@${occurrence.line}`),
					...witnessRefs.map((witness) => `references historical witness ${witness}`),
				].join(", ");
				problems.push(`${path} is a current ${closureSource[0]} source consuming a legacy witness (${detail})`);
			}
			continue;
		}
		const allowance = allowances[path];
		if (allowance === undefined) {
			for (const occurrence of occurrences) {
				problems.push(`unclassified legacy ${occurrence.kind} occurrence at ${path}:${occurrence.line}`);
			}
			continue;
		}
		if (allowance.digest !== undefined) {
			const actual = createHash("sha256").update(content, "utf8").digest("hex");
			if (actual !== allowance.digest) {
				problems.push(`${path} (${allowance.label}) digest drift: expected ${allowance.digest}, found ${actual}`);
			}
			continue;
		}
		const counts = allowance.counts ?? {};
		const lines = content.split("\n");
		const actualByKind = new Map<LegacyMatchKind, [LegacyOccurrence, ...LegacyOccurrence[]]>();
		for (const occurrence of occurrences) {
			const bucket = actualByKind.get(occurrence.kind);
			if (bucket === undefined) {
				actualByKind.set(occurrence.kind, [occurrence]);
				continue;
			}
			bucket.push(occurrence);
		}
		for (const [kind, found] of actualByKind) {
			const allowed = counts[kind];
			if (allowed === undefined) {
				problems.push(
					`${path} carries unlabelled legacy ${kind} occurrence at line ${found[0].line} (allowance kinds: ${Object.keys(counts).join(", ") || "none"})`,
				);
				continue;
			}
			const unlabelled = found.filter((occurrence) => !(lines[occurrence.line - 1] ?? "").includes(HISTORICAL_LABEL));
			for (const occurrence of unlabelled) {
				problems.push(
					`${path} legacy ${kind} occurrence at line ${occurrence.line} is unlabelled (missing ${HISTORICAL_LABEL} marker)`,
				);
			}
			if (found.length > allowed) {
				problems.push(`${path} has ${found.length} legacy ${kind} occurrence(s), allowance allows ${allowed} (extra)`);
			}
		}
		for (const [kind, allowed] of Object.entries(counts) as ReadonlyArray<readonly [LegacyMatchKind, number]>) {
			const found = actualByKind.get(kind)?.length ?? 0;
			if (found < allowed) {
				problems.push(`${path} allowance expects ${allowed} legacy ${kind} occurrence(s), found ${found} (unused)`);
			}
		}
	}
	for (const path of Object.keys(allowances).sort()) {
		if (!(path in trackedFiles)) {
			problems.push(`${path} legacy witness allowance is unused: file is absent from tracked text`);
		}
	}
	return problems;
}

const TRACKED_TEXT_PATHS = [
	".github/workflows",
	"scripts",
	"docs",
	"packages",
	"README.md",
	"CONTRIBUTING.md",
	"package.json",
	"bun.lock",
	"Cargo.toml",
	"Cargo.lock",
	"rust-toolchain.toml",
	"deny.toml",
	"rustfmt.toml",
] as const;

/** Enumerate the approved tracked-text authority surface. */
export function loadTrackedTextFiles(root: string): Record<string, string> {
	let listing: string;
	try {
		listing = execFileSync(
			"git",
			["-C", root, "ls-files", "-z", "--", ...TRACKED_TEXT_PATHS],
			{ encoding: "utf8" },
		);
	} catch {
		return {};
	}
	const files: Record<string, string> = {};
	for (const entry of listing.split("\0")) {
		if (entry === "") continue;
		try {
			const content = readFileSync(join(root, entry), "utf8");
			if (content.includes("\u0000")) continue;
			files[entry] = content;
		} catch {
			// unreadable or removed mid-run: not tracked text we can classify
		}
	}
	return files;
}

export async function loadAlignmentInputs(root: string): Promise<AlignmentInputs> {
	const files: Record<string, string> = {};
	for (const path of [...PIN_LITERAL_PATHS, LEDGER_CARRIER_PATH]) {
		try {
			files[path] = readFileSync(join(root, path), "utf8");
		} catch {
			// verifyPinLiterals reports missing files
		}
	}
	// Fail closed before any reference-derived data is read: the canonical
	// checkout must sit at the exact baseline HEAD.
	assertCanonicalReference(root);
	const { definitions } = await loadCanonicalToolRegistry();
	return {
		files,
		referenceHead: readCanonicalReferenceHead(root),
		registryTools: definitions,
		trackedFiles: loadTrackedTextFiles(root),
	};
}

/** Run every alignment witness; empty means green. */
export function runAlignmentWitnesses(inputs: AlignmentInputs): string[] {
	return [
		...verifyPinLiterals(inputs.files).map((problem) => `[pin-literals] ${problem}`),
		...verifyReferenceCheckout(inputs.referenceHead).map((problem) => `[reference-checkout] ${problem}`),
		...verifyPortableToolSelection(inputs.registryTools).map((problem) => `[portable-tools] ${problem}`),
		...verifyLegacyIdentity(inputs.trackedFiles).map((problem) => `[legacy-identity] ${problem}`),
	];
}

async function main(): Promise<void> {
	let inputs: AlignmentInputs;
	try {
		inputs = await loadAlignmentInputs(REPO_ROOT);
	} catch (error) {
		const detail = error instanceof Error ? error.message : String(error);
		console.error(`alignment witness suite failed to load inputs: ${detail}`);
		process.exit(1);
	}
	const violations = runAlignmentWitnesses(inputs);
	if (violations.length > 0) {
		console.error(`alignment witness suite failed with ${violations.length} violation(s):`);
		for (const violation of violations) console.error(`  - ${violation}`);
		process.exit(1);
	}
	process.stdout.write("ALIGNMENT_WITNESSES_OK\n");
}

if (import.meta.main) main();
