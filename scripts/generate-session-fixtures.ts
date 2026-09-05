#!/usr/bin/env bun
/**
 * Offline deterministic generator for Phase 3 session compatibility fixtures.
 *
 * Source of truth: the reference SessionManager and public helpers under
 * `.references/pi-2.0/packages/coding-agent/src/core/session-manager.ts`. Network
 * fetches are forbidden; runtime Rust never needs Bun.
 *
 * Emits:
 *   crates/pi/tests/fixtures/sessions/
 *     v1/linear-with-compaction.jsonl
 *     v2/branched.jsonl
 *     v3/{basic,branched-labels,compacted-twice,unknown-entries,
 *         forked-header,custom-messages,branched-session}.jsonl
 *     <same>.expected.json  (context/tree/labels/name/leaf + originalLines)
 *     manifest.json (published last, after stale-pair pruning)
 *
 * Reruns converge on the same tree: after every current `.jsonl` and
 * matching `.expected.json` pair is written, unlisted stale pairs under
 * the output directory are removed and only then is the manifest
 * published. The output directory itself is never deleted, so a failed
 * run preserves the previous complete tree and manifest.
 *
 * Normalization (must be reproducible by Rust-side test helpers):
 *   - session id  → 00000000-0000-7000-8000-0000000000NN (per fixture)
 *   - entry id    → e + 7-digit zero-padded 1-based position among non-header lines
 *   - timestamp   → header = 2025-01-01T00:00:00.000Z;
 *                   non-header line k = base + k seconds (ISO ms Z);
 *                   message.timestamp (ms) = same instant as entry ISO
 *   - cwd         → /tmp/pi-fixtures/cwd  (fork target: .../cwd-forked)
 *   - parentSession → /tmp/pi-fixtures/parent-session.jsonl
 *                     (branched-session: .../source-session.jsonl)
 * Cross-line relationships (parentId, firstKeptEntryId, targetId, fromId≠"root")
 * are remapped through the same id map.
 *
 * Usage: bun scripts/generate-session-fixtures.ts
 */

import { access, mkdir, readFile, readdir, rename, stat, unlink, writeFile } from "node:fs/promises";
import {
	appendFileSync,
	constants as fsConstants,
	mkdirSync,
	mkdtempSync,
	readFileSync,
	rmSync,
	writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { randomUUID } from "node:crypto";
import { assertCanonicalReference, canonicalReferenceRoot } from "./reference-identity.ts";

// ---------------------------------------------------------------------------
// Paths
// ---------------------------------------------------------------------------

const SCRIPT_DIR = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = resolve(SCRIPT_DIR, "..");
const REF_ROOT = canonicalReferenceRoot(REPO_ROOT);
const REF_SESSION_MANAGER = join(
	REF_ROOT,
	"packages/coding-agent/src/core/session-manager.ts",
);
const REF_UUID = join(REF_ROOT, "packages/ai/src/utils/uuid.ts");
const OUT_DIR = join(
	REPO_ROOT,
	"crates/pi/tests/fixtures/sessions",
);

// ---------------------------------------------------------------------------
// Normalization constants
// ---------------------------------------------------------------------------

const BASE_ISO = "2025-01-01T00:00:00.000Z";
const BASE_MS = Date.parse(BASE_ISO);
const CWD_PLACEHOLDER = "/tmp/pi-fixtures/cwd";
const CWD_FORKED_PLACEHOLDER = "/tmp/pi-fixtures/cwd-forked";
const PARENT_SESSION_PLACEHOLDER = "/tmp/pi-fixtures/parent-session.jsonl";
const SOURCE_SESSION_PLACEHOLDER = "/tmp/pi-fixtures/source-session.jsonl";

/** Per-fixture session id (v7-shaped, stable). Index 1..N. */
function sessionIdFor(index: number): string {
	const nn = String(index).padStart(2, "0");
	return `00000000-0000-7000-8000-0000000000${nn}`;
}

/** Entry id for 1-based non-header position k. */
function entryIdFor(k: number): string {
	return `e${String(k).padStart(7, "0")}`;
}

/** ISO timestamp for position k (header = 0). */
function isoFor(k: number): string {
	return new Date(BASE_MS + k * 1000).toISOString();
}

// ---------------------------------------------------------------------------
// Fail / Bun / path guards (match generate-builtin-models.ts style)
// ---------------------------------------------------------------------------

function fail(message: string): never {
	console.error(message);
	process.exit(1);
}

function assertBunRuntime(): void {
	if (!("Bun" in globalThis) || globalThis.Bun === undefined) {
		fail(
			"missing prerequisite: Bun runtime required (run with `bun scripts/generate-session-fixtures.ts`)",
		);
	}
}

async function assertPathReadable(path: string, label: string): Promise<void> {
	try {
		await access(path, fsConstants.R_OK);
	} catch {
		fail(`missing prerequisite: ${label} not found or unreadable: ${path}`);
	}
}

// ---------------------------------------------------------------------------
// Reference import via Bun runtime plugin
// ---------------------------------------------------------------------------

function registerReferenceResolver(): void {
	Bun.plugin({
		name: "pi-reference-resolver",
		setup(build) {
			// SessionManager only needs uuidv7 at runtime from pi-ai. Mapping the
			// package root to the implementation keeps ID generation faithful
			// without pulling the full provider graph.
			build.onResolve({ filter: /^@earendil-works\/pi-ai$/ }, () => ({
				path: REF_UUID,
			}));
			// child-process.ts imports cross-spawn; on Linux it is never called
			// (win32-only branch). Shim satisfies resolution without npm install.
			build.onResolve({ filter: /^cross-spawn$/ }, () => ({
				path: "cross-spawn",
				namespace: "pi-shim",
			}));
			build.onLoad({ filter: /.*/, namespace: "pi-shim" }, () => ({
				contents:
					'import { spawn, spawnSync } from "node:child_process";\nexport default Object.assign(spawn, { sync: spawnSync });\n',
				loader: "js",
			}));
		},
	});
}

interface SessionHeader {
	type: "session";
	version?: number;
	id: string;
	timestamp: string;
	cwd: string;
	parentSession?: string;
}

interface SessionEntryBase {
	type: string;
	id: string;
	parentId: string | null;
	timestamp: string;
	[key: string]: unknown;
}

type FileEntry = SessionHeader | SessionEntryBase;

interface SessionTreeNode {
	entry: SessionEntryBase;
	children: SessionTreeNode[];
	label?: string;
	labelTimestamp?: string;
}

interface SessionContext {
	messages: unknown[];
	thinkingLevel: string;
	model: { provider: string; modelId: string } | null;
}

interface SessionManagerLike {
	appendMessage(message: unknown): string;
	appendThinkingLevelChange(thinkingLevel: string): string;
	appendModelChange(provider: string, modelId: string): string;
	appendCompaction(
		summary: string,
		firstKeptEntryId: string,
		tokensBefore: number,
		details?: unknown,
		fromHook?: boolean,
	): string;
	appendCustomEntry(customType: string, data?: unknown): string;
	appendSessionInfo(name: string): string;
	appendCustomMessageEntry(
		customType: string,
		content: string | unknown[],
		display: boolean,
		details?: unknown,
	): string;
	appendLabelChange(targetId: string, label: string | undefined): string;
	branch(branchFromId: string): void;
	branchWithSummary(
		branchFromId: string | null,
		summary: string,
		details?: unknown,
		fromHook?: boolean,
	): string;
	createBranchedSession(leafId: string): string | undefined;
	getLeafId(): string | null;
	getLeafEntry(): SessionEntryBase | undefined;
	getEntry(id: string): SessionEntryBase | undefined;
	getLabel(id: string): string | undefined;
	getHeader(): SessionHeader | null;
	getEntries(): SessionEntryBase[];
	getTree(): SessionTreeNode[];
	getSessionName(): string | undefined;
	buildSessionContext(): SessionContext;
	getSessionFile(): string | undefined;
	getCwd(): string;
	getSessionDir(): string;
}

interface SessionManagerStatic {
	create(cwd: string, sessionDir?: string): SessionManagerLike;
	open(path: string, sessionDir?: string, cwdOverride?: string): SessionManagerLike;
	inMemory(cwd?: string): SessionManagerLike;
	forkFrom(
		sourcePath: string,
		targetCwd: string,
		sessionDir?: string,
	): SessionManagerLike;
}

interface ReferenceSessionModule {
	SessionManager: SessionManagerStatic;
	migrateSessionEntries: (entries: FileEntry[]) => void;
	buildSessionContext: (
		entries: SessionEntryBase[],
		leafId?: string | null,
	) => SessionContext;
	buildContextEntries: (
		entries: SessionEntryBase[],
		leafId?: string | null,
	) => SessionEntryBase[];
	sessionEntryToContextMessages: (entry: SessionEntryBase) => unknown[];
	loadEntriesFromFile: (filePath: string) => FileEntry[];
	CURRENT_SESSION_VERSION: number;
}

let ref: ReferenceSessionModule;

async function loadReference(): Promise<void> {
	// Fail closed before the reference SessionManager module is read.
	assertCanonicalReference(REPO_ROOT);
	await assertPathReadable(REF_SESSION_MANAGER, "reference SessionManager");
	await assertPathReadable(REF_UUID, "reference uuidv7");

	registerReferenceResolver();

	let imported: unknown;
	try {
		imported = await import(pathToFileURL(REF_SESSION_MANAGER).href);
	} catch (error) {
		const detail = error instanceof Error ? error.message : String(error);
		fail(
			`missing prerequisite: failed to import reference SessionManager ${REF_SESSION_MANAGER}: ${detail}`,
		);
	}

	if (typeof imported !== "object" || imported === null) {
		fail(
			`missing prerequisite: reference SessionManager export is not an object: ${REF_SESSION_MANAGER}`,
		);
	}

	const mod = imported as Record<string, unknown>;
	const SessionManager = mod.SessionManager;
	const migrateSessionEntries = mod.migrateSessionEntries;
	const buildSessionContext = mod.buildSessionContext;
	const buildContextEntries = mod.buildContextEntries;
	const sessionEntryToContextMessages = mod.sessionEntryToContextMessages;
	const loadEntriesFromFile = mod.loadEntriesFromFile;
	const CURRENT_SESSION_VERSION = mod.CURRENT_SESSION_VERSION;

	if (typeof SessionManager !== "function") {
		fail(
			"missing prerequisite: reference SessionManager class not exported",
		);
	}
	if (typeof migrateSessionEntries !== "function") {
		fail(
			"missing prerequisite: reference migrateSessionEntries not exported",
		);
	}
	if (typeof buildSessionContext !== "function") {
		fail(
			"missing prerequisite: reference buildSessionContext not exported",
		);
	}
	if (typeof buildContextEntries !== "function") {
		fail(
			"missing prerequisite: reference buildContextEntries not exported",
		);
	}
	if (typeof sessionEntryToContextMessages !== "function") {
		fail(
			"missing prerequisite: reference sessionEntryToContextMessages not exported",
		);
	}
	if (typeof loadEntriesFromFile !== "function") {
		fail(
			"missing prerequisite: reference loadEntriesFromFile not exported",
		);
	}
	if (CURRENT_SESSION_VERSION !== 3) {
		fail(
			`missing prerequisite: reference CURRENT_SESSION_VERSION is ${String(CURRENT_SESSION_VERSION)}, expected 3`,
		);
	}

	ref = {
		// Escape hatch: the read-only `.references/pi-2.0` module is imported
		// untyped at runtime; the typeof check above is the runtime gate.
		SessionManager: SessionManager as unknown as SessionManagerStatic,
		migrateSessionEntries: migrateSessionEntries as (
			entries: FileEntry[],
		) => void,
		buildSessionContext: buildSessionContext as ReferenceSessionModule["buildSessionContext"],
		buildContextEntries: buildContextEntries as ReferenceSessionModule["buildContextEntries"],
		sessionEntryToContextMessages:
			sessionEntryToContextMessages as ReferenceSessionModule["sessionEntryToContextMessages"],
		loadEntriesFromFile: loadEntriesFromFile as (
			filePath: string,
		) => FileEntry[],
		CURRENT_SESSION_VERSION: 3,
	};
}

// ---------------------------------------------------------------------------
// Message factories (fixed wire shapes for deterministic content)
// ---------------------------------------------------------------------------

function userMessage(text: string, timestampMs: number): Record<string, unknown> {
	return {
		role: "user",
		content: text,
		timestamp: timestampMs,
	};
}

function assistantMessage(
	text: string,
	timestampMs: number,
	opts?: { provider?: string; model?: string },
): Record<string, unknown> {
	return {
		role: "assistant",
		content: [{ type: "text", text }],
		api: "anthropic-messages",
		provider: opts?.provider ?? "anthropic",
		model: opts?.model ?? "claude-sonnet-4-5",
		usage: { input: 10, output: 5, cacheRead: 0, cacheWrite: 0 },
		stopReason: "stop",
		timestamp: timestampMs,
	};
}

function hookMessage(
	customType: string,
	content: string,
	timestampMs: number,
): Record<string, unknown> {
	// v2-era custom message role; migrateV2ToV3 renames role → "custom".
	return {
		role: "hookMessage",
		customType,
		content,
		display: true,
		timestamp: timestampMs,
	};
}

// ---------------------------------------------------------------------------
// Normalization
// ---------------------------------------------------------------------------

interface NormalizeOpts {
	sessionId: string;
	cwd: string;
	parentSession?: string;
}

/**
 * Structural, file-order normalization of raw JSONL lines.
 * Returns normalized raw line strings (no trailing newline).
 */
function normalizeLines(rawLines: string[], opts: NormalizeOpts): string[] {
	const objs: Record<string, unknown>[] = [];
	for (const line of rawLines) {
		if (line.trim().length === 0) continue;
		let parsed: unknown;
		try {
			parsed = JSON.parse(line);
		} catch (error) {
			const detail = error instanceof Error ? error.message : String(error);
			fail(`normalize: JSONL line does not parse: ${detail}`);
		}
		if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
			fail("normalize: JSONL line root is not an object");
		}
		objs.push(parsed as Record<string, unknown>);
	}

	// Pass 1: build id map for non-header entries that already have an id.
	const idMap = new Map<string, string>();
	let entryPos = 0;
	for (const o of objs) {
		if (o.type === "session") continue;
		entryPos += 1;
		if (typeof o.id === "string") {
			idMap.set(o.id, entryIdFor(entryPos));
		}
	}

	// Pass 2: rewrite fields in place.
	entryPos = 0;
	for (const o of objs) {
		if (o.type === "session") {
			o.id = opts.sessionId;
			o.timestamp = isoFor(0);
			if (typeof o.cwd === "string") {
				o.cwd = opts.cwd;
			}
			if (typeof o.parentSession === "string") {
				if (opts.parentSession === undefined) {
					fail(
						"normalize: header has parentSession but no placeholder provided",
					);
				}
				o.parentSession = opts.parentSession;
			}
			continue;
		}

		entryPos += 1;
		const entryTs = isoFor(entryPos);
		const entryMs = BASE_MS + entryPos * 1000;

		if (typeof o.id === "string") {
			const mapped = idMap.get(o.id);
			if (mapped === undefined) {
				fail(`normalize: entry id ${o.id} missing from id map`);
			}
			o.id = mapped;
		}
		if (typeof o.parentId === "string") {
			const mapped = idMap.get(o.parentId);
			if (mapped === undefined) {
				fail(
					`normalize: parentId ${o.parentId} does not reference a known entry`,
				);
			}
			o.parentId = mapped;
		} else if (o.parentId !== null && o.parentId !== undefined) {
			// leave null as-is; anything else is unexpected
			fail(
				`normalize: unexpected parentId type ${typeof o.parentId} on entry pos ${entryPos}`,
			);
		}
		o.timestamp = entryTs;

		if (typeof o.firstKeptEntryId === "string") {
			const mapped = idMap.get(o.firstKeptEntryId);
			if (mapped === undefined) {
				fail(
					`normalize: firstKeptEntryId ${o.firstKeptEntryId} does not reference a known entry`,
				);
			}
			o.firstKeptEntryId = mapped;
		}
		if (typeof o.targetId === "string") {
			const mapped = idMap.get(o.targetId);
			if (mapped === undefined) {
				fail(
					`normalize: targetId ${o.targetId} does not reference a known entry`,
				);
			}
			o.targetId = mapped;
		}
		if (typeof o.fromId === "string" && o.fromId !== "root") {
			const mapped = idMap.get(o.fromId);
			if (mapped === undefined) {
				// Dangling fromId: the referenced entry was on an abandoned
				// path and is not present after createBranchedSession's
				// path-only copy. Reset to "root" — fromId is metadata, not
				// used for tree construction, and "root" is the canonical
				// fallback for a branch origin absent from the file.
				o.fromId = "root";
			} else {
				o.fromId = mapped;
			}
		}

		if (o.type === "message") {
			const message = o.message;
			if (
				typeof message === "object" &&
				message !== null &&
				!Array.isArray(message)
			) {
				const msg = message as Record<string, unknown>;
				if (typeof msg.timestamp === "number") {
					msg.timestamp = entryMs;
				}
			}
		}
	}

	return objs.map((o) => JSON.stringify(o));
}

/**
 * Downgrade a normalized v3 session to v1 form:
 *   - strip header.version
 *   - strip entry id/parentId
 *   - compaction firstKeptEntryId → firstKeptEntryIndex (index into full array)
 */
function downgradeToV1(normalizedLines: string[]): string[] {
	const objs: Record<string, unknown>[] = [];
	for (const line of normalizedLines) {
		objs.push(JSON.parse(line) as Record<string, unknown>);
	}

	// Compute firstKeptEntryIndex from current firstKeptEntryId BEFORE stripping ids.
	for (let i = 0; i < objs.length; i += 1) {
		const o = objs[i];
		if (o === undefined || o.type !== "compaction") continue;
		const keptId = o.firstKeptEntryId;
		if (typeof keptId !== "string") {
			fail("downgradeToV1: compaction entry missing firstKeptEntryId");
		}
		let found = -1;
		for (let j = 0; j < objs.length; j += 1) {
			const candidate = objs[j];
			if (candidate !== undefined && candidate.id === keptId) {
				found = j;
				break;
			}
		}
		if (found < 0) {
			fail(
				`downgradeToV1: firstKeptEntryId ${keptId} not found in entries`,
			);
		}
		o.firstKeptEntryIndex = found;
		delete o.firstKeptEntryId;
	}

	for (const o of objs) {
		if (o.type === "session") {
			delete o.version;
			continue;
		}
		delete o.id;
		delete o.parentId;
	}

	return objs.map((o) => JSON.stringify(o));
}

/**
 * Downgrade a normalized v3 session to v2 form:
 *   - header.version = 2
 *   - keep ids/parentIds
 *   - message roles with "hookMessage" stay as authored (migration renames them)
 */
function downgradeToV2(normalizedLines: string[]): string[] {
	const objs: Record<string, unknown>[] = [];
	for (const line of normalizedLines) {
		objs.push(JSON.parse(line) as Record<string, unknown>);
	}
	for (const o of objs) {
		if (o.type === "session") {
			o.version = 2;
		}
	}
	return objs.map((o) => JSON.stringify(o));
}

/**
 * Apply reference migrateSessionEntries to fixture lines, then re-normalize
 * entry ids (migration assigns random ids for v1; v2 keeps them).
 * Returns rewritten (post-migration, normalized) raw line strings.
 */
function migrateAndNormalize(
	fixtureLines: string[],
	opts: NormalizeOpts,
): string[] {
	const entries: FileEntry[] = [];
	for (const line of fixtureLines) {
		entries.push(JSON.parse(line) as FileEntry);
	}
	ref.migrateSessionEntries(entries);
	// Re-serialize then run the structural normalizer so ids/timestamps/paths
	// are stable regardless of migration-assigned random ids.
	const migratedRaw = entries.map((e) => JSON.stringify(e));
	return normalizeLines(migratedRaw, opts);
}

// ---------------------------------------------------------------------------
// Expected-view collection
// ---------------------------------------------------------------------------

export interface CollectedView {
	header: SessionHeader;
	entries: SessionEntryBase[];
	tree: SessionTreeNode[];
	context: SessionContext;
	labels: Record<string, string>;
	name: string | null;
	leaf: string | null;
}

function collectView(mgr: SessionManagerLike): CollectedView {
	const header = mgr.getHeader();
	if (header === null) {
		fail("collectView: SessionManager has no header");
	}
	const entries = mgr.getEntries();
	const labels: Record<string, string> = {};
	for (const entry of entries) {
		const label = mgr.getLabel(entry.id);
		if (label !== undefined) {
			labels[entry.id] = label;
		}
	}
	return {
		header,
		entries,
		tree: mgr.getTree(),
		context: mgr.buildSessionContext(),
		labels,
		name: mgr.getSessionName() ?? null,
		leaf: mgr.getLeafId(),
	};
}

/**
 * Write lines to a temp file, open with reference SessionManager, collect view.
 * Caller owns cleanup of parent tempDir.
 */
function reopenAndCollect(
	tempDir: string,
	fileName: string,
	lines: string[],
): CollectedView {
	mkdirSync(tempDir, { recursive: true });
	const filePath = join(tempDir, fileName);
	writeFileSync(filePath, `${lines.join("\n")}\n`, { encoding: "utf8" });
	const mgr = ref.SessionManager.open(filePath);
	return collectView(mgr);
}

// ---------------------------------------------------------------------------
// Expected JSON assembly
// ---------------------------------------------------------------------------

interface ExpectedJson {
	fixture: string;
	formatVersion: number;
	normalization: {
		sessionId: string;
		entryIdScheme: string;
		timestampBase: string;
		timestampScheme: string;
		cwd: string;
		parentSession: string | null;
	};
	sessionId: string;
	originalLines: string[];
	rewrittenLines?: string[];
	header: SessionHeader;
	entries: SessionEntryBase[];
	tree: SessionTreeNode[];
	context: SessionContext;
	labels: Record<string, string>;
	name: string | null;
	leaf: string | null;
}

function buildExpected(args: {
	fixture: string;
	formatVersion: number;
	sessionId: string;
	cwd: string;
	parentSession: string | null;
	originalLines: string[];
	rewrittenLines?: string[];
	view: CollectedView;
}): ExpectedJson {
	const expected: ExpectedJson = {
		fixture: args.fixture,
		formatVersion: args.formatVersion,
		normalization: {
			sessionId: args.sessionId,
			entryIdScheme:
				"e + 7-digit zero-padded 1-based position among non-header lines",
			timestampBase: BASE_ISO,
			timestampScheme:
				"header = base; non-header line k = base + k seconds; message.timestamp ms = same instant",
			cwd: args.cwd,
			parentSession: args.parentSession,
		},
		sessionId: args.sessionId,
		originalLines: args.originalLines,
		header: args.view.header,
		entries: args.view.entries,
		tree: args.view.tree,
		context: args.view.context,
		labels: args.view.labels,
		name: args.view.name,
		leaf: args.view.leaf,
	};
	if (args.rewrittenLines !== undefined) {
		// Insert rewrittenLines after originalLines by rebuilding key order.
		// ExpectedJson type has rewrittenLines optional; place it by reassigning.
		const withRewritten: ExpectedJson = {
			fixture: expected.fixture,
			formatVersion: expected.formatVersion,
			normalization: expected.normalization,
			sessionId: expected.sessionId,
			originalLines: expected.originalLines,
			rewrittenLines: args.rewrittenLines,
			header: expected.header,
			entries: expected.entries,
			tree: expected.tree,
			context: expected.context,
			labels: expected.labels,
			name: expected.name,
			leaf: expected.leaf,
		};
		return withRewritten;
	}
	return expected;
}

function encodeExpected(expected: ExpectedJson): string {
	return `${JSON.stringify(expected, null, 2)}\n`;
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

function validateFixture(args: {
	fixture: string;
	formatVersion: number;
	lines: string[];
	expected: ExpectedJson;
	tmpLeakNeedle: string;
}): void {
	const { fixture, formatVersion, lines, expected, tmpLeakNeedle } = args;

	// Every JSONL line parses.
	const parsed: Record<string, unknown>[] = [];
	for (let i = 0; i < lines.length; i += 1) {
		const line = lines[i];
		if (line === undefined || line.trim().length === 0) {
			fail(`validate ${fixture}: empty line at index ${i}`);
		}
		let obj: unknown;
		try {
			obj = JSON.parse(line);
		} catch (error) {
			const detail = error instanceof Error ? error.message : String(error);
			fail(`validate ${fixture}: line ${i} does not parse: ${detail}`);
		}
		if (typeof obj !== "object" || obj === null || Array.isArray(obj)) {
			fail(`validate ${fixture}: line ${i} root is not an object`);
		}
		parsed.push(obj as Record<string, unknown>);
	}

	// No temp-path leakage into fixture bytes.
	const joined = lines.join("\n");
	if (joined.includes(tmpLeakNeedle)) {
		fail(
			`validate ${fixture}: fixture content leaks temp path ${tmpLeakNeedle}`,
		);
	}

	// Header checks.
	const header = parsed[0];
	if (header === undefined || header.type !== "session") {
		fail(`validate ${fixture}: first line is not a session header`);
	}
	if (header.id !== expected.sessionId) {
		fail(
			`validate ${fixture}: header.id ${String(header.id)} != expected.sessionId ${expected.sessionId}`,
		);
	}
	if (header.timestamp !== BASE_ISO) {
		fail(
			`validate ${fixture}: header.timestamp ${String(header.timestamp)} != ${BASE_ISO}`,
		);
	}
	if (formatVersion === 1) {
		if (header.version !== undefined) {
			fail(`validate ${fixture}: v1 header must not have version`);
		}
	} else if (formatVersion === 2) {
		if (header.version !== 2) {
			fail(
				`validate ${fixture}: v2 header.version is ${String(header.version)}, expected 2`,
			);
		}
	} else if (header.version !== 3) {
		fail(
			`validate ${fixture}: v3 header.version is ${String(header.version)}, expected 3`,
		);
	}

	// Collect entry ids (v1 has none).
	const ids = new Set<string>();
	let entryPos = 0;
	for (let i = 1; i < parsed.length; i += 1) {
		const o = parsed[i];
		if (o === undefined) continue;
		entryPos += 1;
		if (typeof o.timestamp === "string" && o.timestamp !== isoFor(entryPos)) {
			fail(
				`validate ${fixture}: entry at pos ${entryPos} timestamp ${String(o.timestamp)} != ${isoFor(entryPos)}`,
			);
		}
		if (typeof o.id === "string") {
			if (ids.has(o.id)) {
				fail(`validate ${fixture}: duplicate entry id ${o.id}`);
			}
			ids.add(o.id);
			if (formatVersion >= 2) {
				const expectedId = entryIdFor(entryPos);
				if (o.id !== expectedId) {
					fail(
						`validate ${fixture}: entry pos ${entryPos} id ${o.id} != ${expectedId}`,
					);
				}
			}
		}
	}

	// Cross-line references (only when ids are present).
	if (formatVersion >= 2) {
		for (let i = 1; i < parsed.length; i += 1) {
			const o = parsed[i];
			if (o === undefined) continue;
			if (typeof o.parentId === "string" && !ids.has(o.parentId)) {
				fail(
					`validate ${fixture}: parentId ${o.parentId} not in fixture ids`,
				);
			}
			if (
				typeof o.firstKeptEntryId === "string" &&
				!ids.has(o.firstKeptEntryId)
			) {
				fail(
					`validate ${fixture}: firstKeptEntryId ${o.firstKeptEntryId} not in fixture ids`,
				);
			}
			if (typeof o.targetId === "string" && !ids.has(o.targetId)) {
				fail(
					`validate ${fixture}: targetId ${o.targetId} not in fixture ids`,
				);
			}
			if (
				typeof o.fromId === "string" &&
				o.fromId !== "root" &&
				!ids.has(o.fromId)
			) {
				fail(
					`validate ${fixture}: fromId ${o.fromId} not in fixture ids`,
				);
			}
		}
	}

	// Expected view references valid normalized ids (post-migration for v1/v2).
	const expectedIds = new Set(expected.entries.map((e) => e.id));
	if (expected.leaf !== null && !expectedIds.has(expected.leaf)) {
		fail(
			`validate ${fixture}: expected.leaf ${expected.leaf} not in expected.entries`,
		);
	}
	for (const entry of expected.entries) {
		if (typeof entry.parentId === "string" && !expectedIds.has(entry.parentId)) {
			fail(
				`validate ${fixture}: expected entry ${entry.id} parentId ${entry.parentId} not in expected.entries`,
			);
		}
		if (
			typeof entry.firstKeptEntryId === "string" &&
			!expectedIds.has(entry.firstKeptEntryId)
		) {
			fail(
				`validate ${fixture}: expected entry ${entry.id} firstKeptEntryId ${entry.firstKeptEntryId} not in expected.entries`,
			);
		}
		if (
			typeof entry.targetId === "string" &&
			!expectedIds.has(entry.targetId)
		) {
			fail(
				`validate ${fixture}: expected entry ${entry.id} targetId ${entry.targetId} not in expected.entries`,
			);
		}
		if (
			typeof entry.fromId === "string" &&
			entry.fromId !== "root" &&
			!expectedIds.has(entry.fromId)
		) {
			fail(
				`validate ${fixture}: expected entry ${entry.id} fromId ${entry.fromId} not in expected.entries`,
			);
		}
	}
	for (const labelTarget of Object.keys(expected.labels)) {
		if (!expectedIds.has(labelTarget)) {
			fail(
				`validate ${fixture}: labels key ${labelTarget} not in expected.entries`,
			);
		}
	}
	if (expected.context.messages.length === 0) {
		fail(`validate ${fixture}: expected.context.messages is empty`);
	}
	if (expected.originalLines.length !== lines.length) {
		fail(
			`validate ${fixture}: originalLines length ${expected.originalLines.length} != fixture lines ${lines.length}`,
		);
	}
	for (let i = 0; i < lines.length; i += 1) {
		if (expected.originalLines[i] !== lines[i]) {
			fail(
				`validate ${fixture}: originalLines[${i}] diverges from fixture line`,
			);
		}
	}
}

// ---------------------------------------------------------------------------
// Atomic write
// ---------------------------------------------------------------------------

async function writeAtomically(path: string, contents: string): Promise<void> {
	const dir = dirname(path);
	await mkdir(dir, { recursive: true });
	const tempPath = join(
		dir,
		`.session-fixture.${process.pid}.${Date.now()}.${Math.random().toString(16).slice(2)}.tmp`,
	);
	try {
		await writeFile(tempPath, contents, { encoding: "utf8" });
		await rename(tempPath, path);
	} catch (error) {
		try {
			const { unlink } = await import("node:fs/promises");
			await unlink(tempPath);
		} catch {
			// best-effort temp cleanup
		}
		const detail = error instanceof Error ? error.message : String(error);
		fail(`failed to write fixture atomically to ${path}: ${detail}`);
	}
}

// ---------------------------------------------------------------------------
// Stale fixture pruning
// ---------------------------------------------------------------------------

const FIXTURE_MANIFEST_NAME = "manifest.json";

/**
 * Enumerate generator-owned fixture files under `outDir`, grouped by the
 * `.jsonl` rel that owns them.
 *
 * Ownership is conservative: only regular files ending in `.jsonl`, or in
 * `.expected.json` (owned by the sibling `.jsonl` rel), are listed.
 * Unrelated files and directories never appear here and are therefore
 * never removed.
 */
async function listOwnedFixtureFiles(
	outDir: string,
): Promise<Map<string, string[]>> {
	const owned = new Map<string, string[]>();
	let names: string[];
	try {
		names = await readdir(outDir, { recursive: true });
	} catch (error) {
		const code = error instanceof Error
			? (error as Error & { code?: string }).code
			: undefined;
		if (code === "ENOENT") return owned;
		throw error;
	}
	for (const name of names) {
		// Manifest entries are posix-style ("v3/basic.jsonl"); normalize disk
		// rels so the keep-set comparison holds on Windows too.
		const rel = String(name).replaceAll("\\", "/");
		let isFile: boolean;
		try {
			isFile = (await stat(join(outDir, rel))).isFile();
		} catch {
			continue; // vanished between readdir and stat
		}
		if (!isFile) continue;
		let owner: string | null = null;
		if (rel.endsWith(".jsonl")) {
			owner = rel;
		} else if (rel.endsWith(".expected.json")) {
			owner = `${rel.slice(0, -".expected.json".length)}.jsonl`;
		}
		if (owner === null) continue;
		const files = owned.get(owner) ?? [];
		files.push(rel);
		owned.set(owner, files);
	}
	return owned;
}

/**
 * Remove stale generator-owned fixture pairs after every current pair has
 * been written successfully. A pair is stale when its `.jsonl` rel is not
 * in `listedRels`; a lone `.expected.json` whose `.jsonl` is missing is
 * stale too. Listed pairs, unrelated files, and `outDir` itself are never
 * touched, so fresh and dirty checkouts converge on the same tree and a
 * failed run leaves the previous complete tree intact. The caller
 * publishes the manifest only after this returns.
 */
export async function pruneStaleFixturePairs(
	outDir: string,
	listedRels: readonly string[],
): Promise<string[]> {
	const listed = new Set(listedRels);
	const owned = await listOwnedFixtureFiles(outDir);
	const removed: string[] = [];
	for (const [owner, files] of [...owned.entries()].sort(([a], [b]) =>
		a.localeCompare(b),
	)) {
		if (listed.has(owner)) continue;
		for (const rel of files.sort()) {
			await unlink(join(outDir, rel));
			removed.push(rel);
		}
	}
	return removed;
}

// ---------------------------------------------------------------------------
// Scenario builders
// ---------------------------------------------------------------------------

export interface BuiltFixture {
	/** Relative path under OUT_DIR, e.g. "v3/basic.jsonl" */
	rel: string;
	formatVersion: number;
	sessionId: string;
	cwd: string;
	parentSession: string | null;
	/** Normalized (or downgraded) fixture lines. */
	lines: string[];
	/** Post-migration rewritten lines (v1/v2 only). */
	rewrittenLines?: string[];
	view: CollectedView;
}

function readSessionFileLines(filePath: string): string[] {
	const content = readFileSync(filePath, { encoding: "utf8" });
	const lines: string[] = [];
	for (const line of content.split("\n")) {
		if (line.trim().length > 0) lines.push(line);
	}
	if (lines.length === 0) {
		fail(`readSessionFileLines: empty session file ${filePath}`);
	}
	return lines;
}

function requireSessionFile(mgr: SessionManagerLike, label: string): string {
	const file = mgr.getSessionFile();
	if (file === undefined || file.length === 0) {
		fail(`${label}: SessionManager has no session file (assistant not flushed?)`);
	}
	return file;
}

function requireLeafId(mgr: SessionManagerLike, label: string): string {
	const leaf = mgr.getLeafId();
	if (leaf === null) {
		fail(`${label}: SessionManager has null leaf`);
	}
	return leaf;
}

// ---- v3/basic ----

function buildBasic(workRoot: string, fixtureIndex: number): BuiltFixture {
	const sessionDir = join(workRoot, "basic");
	const cwd = join(workRoot, "cwd-basic");
	const mgr = ref.SessionManager.create(cwd, sessionDir);

	mgr.appendMessage(userMessage("What is 2+2?", BASE_MS + 1000));
	mgr.appendMessage(
		assistantMessage("2+2 = 4.", BASE_MS + 2000),
	);
	mgr.appendMessage(userMessage("And 3+3?", BASE_MS + 3000));
	mgr.appendMessage(
		assistantMessage("3+3 = 6.", BASE_MS + 4000),
	);

	const file = requireSessionFile(mgr, "basic");
	const realLines = readSessionFileLines(file);
	const sessionId = sessionIdFor(fixtureIndex);
	const lines = normalizeLines(realLines, {
		sessionId,
		cwd: CWD_PLACEHOLDER,
	});
	const view = reopenAndCollect(join(workRoot, "reopen"), "basic.jsonl", lines);
	return {
		rel: "v3/basic.jsonl",
		formatVersion: 3,
		sessionId,
		cwd: CWD_PLACEHOLDER,
		parentSession: null,
		lines,
		view,
	};
}

// ---- v3/branched-labels ----

function buildBranchedLabels(
	workRoot: string,
	fixtureIndex: number,
): BuiltFixture {
	const sessionDir = join(workRoot, "branched-labels");
	const cwd = join(workRoot, "cwd-branched-labels");
	const mgr = ref.SessionManager.create(cwd, sessionDir);

	const u1 = mgr.appendMessage(
		userMessage("Start of main path.", BASE_MS + 1000),
	);
	const a1 = mgr.appendMessage(
		assistantMessage("Acknowledged start.", BASE_MS + 2000),
	);
	mgr.appendMessage(userMessage("Continue main path.", BASE_MS + 3000));
	const a2 = mgr.appendMessage(
		assistantMessage("Main path reply.", BASE_MS + 4000),
	);

	// Branch off a1 with a summary of the abandoned path.
	mgr.branchWithSummary(a1, "Summary of abandoned main path after A1.");
	const u3 = mgr.appendMessage(
		userMessage("Branch question after summary.", BASE_MS + 6000),
	);
	mgr.appendMessage(
		assistantMessage("Branch answer.", BASE_MS + 7000),
	);

	// Labels: set, set, set, then clear one.
	mgr.appendLabelChange(u1, "start");
	mgr.appendLabelChange(a2, "abandoned");
	mgr.appendLabelChange(u3, "branch-question");
	mgr.appendLabelChange(a2, undefined);
	mgr.appendSessionInfo("Branched labels fixture");

	const file = requireSessionFile(mgr, "branched-labels");
	const realLines = readSessionFileLines(file);
	const sessionId = sessionIdFor(fixtureIndex);
	const lines = normalizeLines(realLines, {
		sessionId,
		cwd: CWD_PLACEHOLDER,
	});
	const view = reopenAndCollect(
		join(workRoot, "reopen"),
		"branched-labels.jsonl",
		lines,
	);
	return {
		rel: "v3/branched-labels.jsonl",
		formatVersion: 3,
		sessionId,
		cwd: CWD_PLACEHOLDER,
		parentSession: null,
		lines,
		view,
	};
}

// ---- v3/compacted-twice ----

function buildCompactedTwice(
	workRoot: string,
	fixtureIndex: number,
): BuiltFixture {
	const sessionDir = join(workRoot, "compacted-twice");
	const cwd = join(workRoot, "cwd-compacted-twice");
	const mgr = ref.SessionManager.create(cwd, sessionDir);

	mgr.appendMessage(userMessage("Topic A intro.", BASE_MS + 1000));
	mgr.appendMessage(assistantMessage("Topic A reply.", BASE_MS + 2000));
	const u2 = mgr.appendMessage(
		userMessage("Topic A follow-up.", BASE_MS + 3000),
	);
	mgr.appendMessage(assistantMessage("Topic A follow-up reply.", BASE_MS + 4000));

	// First compaction: keep from U2 onward.
	mgr.appendCompaction(
		"First summary: early discussion about topic A.",
		u2,
		12000,
	);

	mgr.appendMessage(userMessage("Topic B intro.", BASE_MS + 6000));
	const a3 = mgr.appendMessage(
		assistantMessage("Topic B reply.", BASE_MS + 7000),
	);
	mgr.appendMessage(userMessage("Topic B follow-up.", BASE_MS + 8000));
	mgr.appendMessage(
		assistantMessage("Topic B follow-up reply.", BASE_MS + 9000),
	);

	// Second compaction: keep from A3; extension-generated details.
	mgr.appendCompaction(
		"Second summary: topics A and B compacted.",
		a3,
		24500,
		{ trigger: "auto", cycle: 2 },
		true,
	);

	mgr.appendMessage(userMessage("Topic C after second compact.", BASE_MS + 11000));
	mgr.appendMessage(
		assistantMessage("Topic C reply.", BASE_MS + 12000),
	);

	const file = requireSessionFile(mgr, "compacted-twice");
	const realLines = readSessionFileLines(file);
	const sessionId = sessionIdFor(fixtureIndex);
	const lines = normalizeLines(realLines, {
		sessionId,
		cwd: CWD_PLACEHOLDER,
	});
	const view = reopenAndCollect(
		join(workRoot, "reopen"),
		"compacted-twice.jsonl",
		lines,
	);
	return {
		rel: "v3/compacted-twice.jsonl",
		formatVersion: 3,
		sessionId,
		cwd: CWD_PLACEHOLDER,
		parentSession: null,
		lines,
		view,
	};
}

// ---- v3/unknown-entries ----

function buildUnknownEntries(
	workRoot: string,
	fixtureIndex: number,
): BuiltFixture {
	const sessionDir = join(workRoot, "unknown-entries");
	const cwd = join(workRoot, "cwd-unknown-entries");
	const mgr = ref.SessionManager.create(cwd, sessionDir);

	mgr.appendMessage(userMessage("Before unknown entry.", BASE_MS + 1000));
	mgr.appendMessage(
		assistantMessage("Reply before unknown.", BASE_MS + 2000),
	);
	const a2Id = requireLeafId(mgr, "unknown-entries after A2");

	// Flush so the file exists, then append a foreign entry line by hand.
	const file = requireSessionFile(mgr, "unknown-entries pre-unknown");
	const unknownId = randomUUID().slice(0, 8);
	const unknownEntry = {
		type: "future_thing",
		id: unknownId,
		parentId: a2Id,
		timestamp: new Date().toISOString(),
		payload: {
			kind: "quantum-annotations",
			nested: { values: [1, 2, 3], flag: true },
		},
		enabled: true,
	};
	appendFileSync(file, `${JSON.stringify(unknownEntry)}\n`);

	// Reopen so the manager indexes the unknown entry; it becomes the leaf.
	const mgr2 = ref.SessionManager.open(file, sessionDir);
	// Confirm unknown is the leaf and is on the path.
	if (mgr2.getLeafId() !== unknownId) {
		fail(
			`unknown-entries: expected leaf ${unknownId} after reopen, got ${String(mgr2.getLeafId())}`,
		);
	}
	mgr2.appendMessage(
		userMessage("After unknown entry.", BASE_MS + 4000),
	);
	mgr2.appendMessage(
		assistantMessage("Reply after unknown.", BASE_MS + 5000),
	);

	const file2 = requireSessionFile(mgr2, "unknown-entries final");
	const realLines = readSessionFileLines(file2);
	const sessionId = sessionIdFor(fixtureIndex);
	const lines = normalizeLines(realLines, {
		sessionId,
		cwd: CWD_PLACEHOLDER,
	});
	const view = reopenAndCollect(
		join(workRoot, "reopen"),
		"unknown-entries.jsonl",
		lines,
	);

	// Sanity: unknown type must survive on the path and contribute nothing to context.
	const hasFuture = view.entries.some((e) => e.type === "future_thing");
	if (!hasFuture) {
		fail("unknown-entries: future_thing entry missing from expected entries");
	}
	const contextRoles = view.context.messages.map((m) => {
		if (typeof m === "object" && m !== null && "role" in m) {
			return String((m as { role: unknown }).role);
		}
		return "?";
	});
	if (contextRoles.includes("future_thing")) {
		fail(
			"unknown-entries: future_thing leaked into context.messages",
		);
	}

	return {
		rel: "v3/unknown-entries.jsonl",
		formatVersion: 3,
		sessionId,
		cwd: CWD_PLACEHOLDER,
		parentSession: null,
		lines,
		view,
	};
}

// ---- v3/forked-header ----

function buildForkedHeader(
	workRoot: string,
	fixtureIndex: number,
): BuiltFixture {
	const sourceDir = join(workRoot, "fork-source");
	const sourceCwd = join(workRoot, "cwd-fork-source");
	const source = ref.SessionManager.create(sourceCwd, sourceDir);
	source.appendMessage(
		userMessage("Source session user message.", BASE_MS + 1000),
	);
	source.appendMessage(
		assistantMessage("Source session assistant reply.", BASE_MS + 2000),
	);
	source.appendMessage(
		userMessage("Source second user turn.", BASE_MS + 3000),
	);
	source.appendMessage(
		assistantMessage("Source second assistant reply.", BASE_MS + 4000),
	);
	const sourceFile = requireSessionFile(source, "forked-header source");

	const forkDir = join(workRoot, "fork-target");
	const forkCwd = join(workRoot, "cwd-fork-target");
	const forked = ref.SessionManager.forkFrom(sourceFile, forkCwd, forkDir);
	const forkedFile = requireSessionFile(forked, "forked-header forked");

	const realLines = readSessionFileLines(forkedFile);
	const sessionId = sessionIdFor(fixtureIndex);
	const lines = normalizeLines(realLines, {
		sessionId,
		cwd: CWD_FORKED_PLACEHOLDER,
		parentSession: PARENT_SESSION_PLACEHOLDER,
	});
	const view = reopenAndCollect(
		join(workRoot, "reopen"),
		"forked-header.jsonl",
		lines,
	);
	if (view.header.parentSession !== PARENT_SESSION_PLACEHOLDER) {
		fail(
			`forked-header: expected parentSession ${PARENT_SESSION_PLACEHOLDER}, got ${String(view.header.parentSession)}`,
		);
	}
	return {
		rel: "v3/forked-header.jsonl",
		formatVersion: 3,
		sessionId,
		cwd: CWD_FORKED_PLACEHOLDER,
		parentSession: PARENT_SESSION_PLACEHOLDER,
		lines,
		view,
	};
}

// ---- v3/custom-messages ----

function buildCustomMessages(
	workRoot: string,
	fixtureIndex: number,
): BuiltFixture {
	const sessionDir = join(workRoot, "custom-messages");
	const cwd = join(workRoot, "cwd-custom-messages");
	const mgr = ref.SessionManager.create(cwd, sessionDir);

	mgr.appendMessage(userMessage("User before custom.", BASE_MS + 1000));
	mgr.appendMessage(
		assistantMessage("Assistant before custom.", BASE_MS + 2000),
	);

	// custom_message participates in context.
	mgr.appendCustomMessageEntry(
		"quote-ext",
		"quoted context text from extension",
		true,
		{ source: "handbook", page: 12 },
	);
	// custom_message with content array (text + image), display=false.
	mgr.appendCustomMessageEntry(
		"screenshot-ext",
		[
			{ type: "text", text: "see screenshot" },
			{ type: "image", data: "aGVsbG8=", mimeType: "image/png" },
		],
		false,
		{ captureId: "cap-1" },
	);
	// plain custom entry — state only, NOT in context.
	mgr.appendCustomEntry("state-ext", { counter: 3, flags: ["a", "b"] });

	mgr.appendMessage(userMessage("User after custom.", BASE_MS + 6000));
	mgr.appendMessage(
		assistantMessage("Assistant after custom.", BASE_MS + 7000),
	);

	const file = requireSessionFile(mgr, "custom-messages");
	const realLines = readSessionFileLines(file);
	const sessionId = sessionIdFor(fixtureIndex);
	const lines = normalizeLines(realLines, {
		sessionId,
		cwd: CWD_PLACEHOLDER,
	});
	const view = reopenAndCollect(
		join(workRoot, "reopen"),
		"custom-messages.jsonl",
		lines,
	);

	// Context must include custom messages (role "custom") but not plain custom entries.
	const roles = view.context.messages.map((m) => {
		if (typeof m === "object" && m !== null && "role" in m) {
			return String((m as { role: unknown }).role);
		}
		return "?";
	});
	const customCount = roles.filter((r) => r === "custom").length;
	if (customCount !== 2) {
		fail(
			`custom-messages: expected 2 custom context messages, got ${customCount} (roles=${roles.join(",")})`,
		);
	}
	const hasStateEntry = view.entries.some(
		(e) => e.type === "custom" && e.customType === "state-ext",
	);
	if (!hasStateEntry) {
		fail("custom-messages: state-ext custom entry missing from entries");
	}

	return {
		rel: "v3/custom-messages.jsonl",
		formatVersion: 3,
		sessionId,
		cwd: CWD_PLACEHOLDER,
		parentSession: null,
		lines,
		view,
	};
}

// ---- v3/branched-session (createBranchedSession golden) ----

function buildBranchedSession(
	workRoot: string,
	fixtureIndex: number,
	branchedLabelsLines: string[],
): BuiltFixture {
	// Seed from the already-normalized branched-labels fixture so createBranchedSession
	// exercises path-only copy + label re-chain against stable input.
	const seedDir = join(workRoot, "branched-session-seed");
	const seedFile = join(seedDir, "seed.jsonl");
	mkdirSync(seedDir, { recursive: true });
	writeFileSync(seedFile, `${branchedLabelsLines.join("\n")}\n`, {
		encoding: "utf8",
	});
	const mgr = ref.SessionManager.open(seedFile, seedDir);
	const leaf = requireLeafId(mgr, "branched-session seed");
	const newFile = mgr.createBranchedSession(leaf);
	if (newFile === undefined) {
		fail("branched-session: createBranchedSession returned undefined");
	}

	const realLines = readSessionFileLines(newFile);
	const sessionId = sessionIdFor(fixtureIndex);
	const lines = normalizeLines(realLines, {
		sessionId,
		cwd: CWD_PLACEHOLDER,
		parentSession: SOURCE_SESSION_PLACEHOLDER,
	});
	const view = reopenAndCollect(
		join(workRoot, "reopen"),
		"branched-session.jsonl",
		lines,
	);
	if (view.header.parentSession !== SOURCE_SESSION_PLACEHOLDER) {
		fail(
			`branched-session: expected parentSession ${SOURCE_SESSION_PLACEHOLDER}, got ${String(view.header.parentSession)}`,
		);
	}
	// Path-only: abandoned main-path labels (cleared A2) must not reappear;
	// "start" and "branch-question" must survive.
	const labelValues = Object.values(view.labels);
	if (!labelValues.includes("start")) {
		fail(
			`branched-session: expected label "start" to survive path copy; labels=${JSON.stringify(view.labels)}`,
		);
	}
	if (!labelValues.includes("branch-question")) {
		fail(
			`branched-session: expected label "branch-question" to survive path copy; labels=${JSON.stringify(view.labels)}`,
		);
	}
	if (labelValues.includes("abandoned")) {
		fail(
			"branched-session: cleared label 'abandoned' reappeared after createBranchedSession",
		);
	}

	return {
		rel: "v3/branched-session.jsonl",
		formatVersion: 3,
		sessionId,
		cwd: CWD_PLACEHOLDER,
		parentSession: SOURCE_SESSION_PLACEHOLDER,
		lines,
		view,
	};
}

// ---- v1/linear-with-compaction ----

function buildV1LinearWithCompaction(
	workRoot: string,
	fixtureIndex: number,
): BuiltFixture {
	const sessionDir = join(workRoot, "v1-linear");
	const cwd = join(workRoot, "cwd-v1-linear");
	const mgr = ref.SessionManager.create(cwd, sessionDir);

	mgr.appendMessage(userMessage("V1 U1.", BASE_MS + 1000));
	mgr.appendMessage(assistantMessage("V1 A1.", BASE_MS + 2000));
	mgr.appendMessage(userMessage("V1 U2.", BASE_MS + 3000));
	mgr.appendMessage(assistantMessage("V1 A2.", BASE_MS + 4000));
	const u3 = mgr.appendMessage(userMessage("V1 U3 kept.", BASE_MS + 5000));
	mgr.appendMessage(assistantMessage("V1 A3 kept.", BASE_MS + 6000));
	mgr.appendMessage(userMessage("V1 U4 kept.", BASE_MS + 7000));
	mgr.appendMessage(assistantMessage("V1 A4 kept.", BASE_MS + 8000));

	// Compaction keeps from U3; no details/fromHook (v1-era shape).
	mgr.appendCompaction(
		"V1 linear compaction summary of early turns.",
		u3,
		54321,
	);

	mgr.appendMessage(userMessage("V1 U5 post-compact.", BASE_MS + 10000));
	mgr.appendMessage(
		assistantMessage("V1 A5 post-compact.", BASE_MS + 11000),
	);

	const file = requireSessionFile(mgr, "v1-linear");
	const realLines = readSessionFileLines(file);
	const sessionId = sessionIdFor(fixtureIndex);
	const normalizedV3 = normalizeLines(realLines, {
		sessionId,
		cwd: CWD_PLACEHOLDER,
	});
	const v1Lines = downgradeToV1(normalizedV3);

	// Expected = post-migration rewritten form.
	const rewrittenLines = migrateAndNormalize(v1Lines, {
		sessionId,
		cwd: CWD_PLACEHOLDER,
	});
	const view = reopenAndCollect(
		join(workRoot, "reopen"),
		"v1-linear-migrated.jsonl",
		rewrittenLines,
	);

	// Sanity: compaction firstKeptEntryId must resolve after migration.
	const compaction = view.entries.find((e) => e.type === "compaction");
	if (compaction === undefined) {
		fail("v1-linear: no compaction entry after migration");
	}
	if (typeof compaction.firstKeptEntryId !== "string") {
		fail("v1-linear: compaction missing firstKeptEntryId after migration");
	}
	if (view.entries.every((e) => e.id !== compaction.firstKeptEntryId)) {
		fail(
			`v1-linear: firstKeptEntryId ${compaction.firstKeptEntryId} not in migrated entries`,
		);
	}

	return {
		rel: "v1/linear-with-compaction.jsonl",
		formatVersion: 1,
		sessionId,
		cwd: CWD_PLACEHOLDER,
		parentSession: null,
		lines: v1Lines,
		rewrittenLines,
		view,
	};
}

// ---- v2/branched (hookMessage + branch) ----

function buildV2Branched(
	workRoot: string,
	fixtureIndex: number,
): BuiltFixture {
	const sessionDir = join(workRoot, "v2-branched");
	const cwd = join(workRoot, "cwd-v2-branched");
	const mgr = ref.SessionManager.create(cwd, sessionDir);

	mgr.appendMessage(userMessage("V2 U1 main.", BASE_MS + 1000));
	// v2-era custom message: role "hookMessage" (renamed to "custom" by v2→v3).
	// Placed before the branch point so it remains on the active path after branch().
	mgr.appendMessage(
		hookMessage("onboarding-hook", "Welcome hint from hook.", BASE_MS + 2000),
	);
	const a1 = mgr.appendMessage(
		assistantMessage("V2 A1 main.", BASE_MS + 3000),
	);
	mgr.appendMessage(userMessage("V2 U2 main.", BASE_MS + 4000));
	mgr.appendMessage(assistantMessage("V2 A2 main.", BASE_MS + 5000));

	// Branch off A1 (abandons U2/A2; keeps U1/hook/A1 on the active path).
	mgr.branch(a1);
	mgr.appendMessage(userMessage("V2 U3 branch.", BASE_MS + 6000));
	mgr.appendMessage(
		assistantMessage("V2 A3 branch.", BASE_MS + 7000),
	);

	const file = requireSessionFile(mgr, "v2-branched");
	const realLines = readSessionFileLines(file);
	const sessionId = sessionIdFor(fixtureIndex);
	const normalizedV3 = normalizeLines(realLines, {
		sessionId,
		cwd: CWD_PLACEHOLDER,
	});
	const v2Lines = downgradeToV2(normalizedV3);

	// Confirm fixture still carries hookMessage before migration.
	const hasHook = v2Lines.some((line) => {
		const o = JSON.parse(line) as Record<string, unknown>;
		if (o.type !== "message") return false;
		const message = o.message;
		return (
			typeof message === "object" &&
			message !== null &&
			(message as { role?: unknown }).role === "hookMessage"
		);
	});
	if (!hasHook) {
		fail("v2-branched: fixture missing hookMessage role message");
	}

	const rewrittenLines = migrateAndNormalize(v2Lines, {
		sessionId,
		cwd: CWD_PLACEHOLDER,
	});
	const view = reopenAndCollect(
		join(workRoot, "reopen"),
		"v2-branched-migrated.jsonl",
		rewrittenLines,
	);

	// After migration: hookMessage → custom; must appear in context.
	const roles = view.context.messages.map((m) => {
		if (typeof m === "object" && m !== null && "role" in m) {
			return String((m as { role: unknown }).role);
		}
		return "?";
	});
	if (!roles.includes("custom")) {
		fail(
			`v2-branched: expected custom role in context after migration; roles=${roles.join(",")}`,
		);
	}
	// Tree must have a branch (A1 has two children: hook→... and U3→...).
	if (view.tree.length === 0) {
		fail("v2-branched: empty tree after migration");
	}

	return {
		rel: "v2/branched.jsonl",
		formatVersion: 2,
		sessionId,
		cwd: CWD_PLACEHOLDER,
		parentSession: null,
		lines: v2Lines,
		rewrittenLines,
		view,
	};
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

interface WriteResult {
	rel: string;
	jsonlLines: number;
	entries: number;
	contextMessages: number;
	formatVersion: number;
}

export interface GenerateSessionFixturesOptions {
	/** Output directory override (tests); defaults to OUT_DIR. */
	outDir?: string;
	/** Fixture set override (tests); defaults to the nine scenarios. */
	fixtures?: BuiltFixture[];
	/**
	 * Check mode: compare on-disk files against generated content without
	 * writing to the repository. Candidate derivation still runs the
	 * session-manager pipeline in its own temp scratch directory (that
	 * roundtrip is how fixture content is produced); the output tree is
	 * read-only in this mode.
	 */
	check?: boolean;
}

export interface GenerateSessionFixturesResult {
	outDir: string;
	results: WriteResult[];
	pruned: string[];
	/** Stale file list (check mode only). */
	stale?: string[];
}

function buildDefaultFixtures(workRoot: string): BuiltFixture[] {
	// Fixture index order (session id NN):
	// 01 v1/linear-with-compaction
	// 02 v2/branched
	// 03 v3/basic
	// 04 v3/branched-labels
	// 05 v3/compacted-twice
	// 06 v3/unknown-entries
	// 07 v3/forked-header
	// 08 v3/custom-messages
	// 09 v3/branched-session
	const fixtures: BuiltFixture[] = [];
	fixtures.push(buildV1LinearWithCompaction(workRoot, 1));
	fixtures.push(buildV2Branched(workRoot, 2));
	fixtures.push(buildBasic(workRoot, 3));
	const branchedLabels = buildBranchedLabels(workRoot, 4);
	fixtures.push(branchedLabels);
	fixtures.push(buildCompactedTwice(workRoot, 5));
	fixtures.push(buildUnknownEntries(workRoot, 6));
	fixtures.push(buildForkedHeader(workRoot, 7));
	fixtures.push(buildCustomMessages(workRoot, 8));
	fixtures.push(
		buildBranchedSession(workRoot, 9, branchedLabels.lines),
	);
	return fixtures;
}

export async function generateSessionFixtures(
	options: GenerateSessionFixturesOptions = {},
): Promise<GenerateSessionFixturesResult> {
	assertBunRuntime();
	await loadReference();

	const outDir = options.outDir ?? OUT_DIR;
	const workRoot = mkdtempSync(join(tmpdir(), "pi-session-fixtures-"));
	const results: WriteResult[] = [];
	const stale: string[] = [];

	try {
		const fixtures = options.fixtures ?? buildDefaultFixtures(workRoot);

		for (const built of fixtures) {
			const fixtureName = built.rel.replace(/\.jsonl$/, "");
			const expected = buildExpected({
				fixture: fixtureName,
				formatVersion: built.formatVersion,
				sessionId: built.sessionId,
				cwd: built.cwd,
				parentSession: built.parentSession,
				originalLines: built.lines,
				rewrittenLines: built.rewrittenLines,
				view: built.view,
			});

			validateFixture({
				fixture: fixtureName,
				formatVersion: built.formatVersion,
				lines: built.lines,
				expected,
				tmpLeakNeedle: workRoot,
			});

			const jsonlPath = join(outDir, built.rel);
			const expectedPath = join(
				outDir,
				built.rel.replace(/\.jsonl$/, ".expected.json"),
			);
			const jsonlContent = `${built.lines.join("\n")}\n`;
			const expectedContent = encodeExpected(expected);

			if (options.check) {
				const jsonlOnDisk = await readFile(jsonlPath, "utf8").catch(() => null);
				const expectedOnDisk = await readFile(expectedPath, "utf8").catch(() => null);
				if (jsonlOnDisk !== jsonlContent) stale.push(built.rel);
				if (expectedOnDisk !== expectedContent)
					stale.push(built.rel.replace(/\.jsonl$/, ".expected.json"));
			} else {
				await writeAtomically(jsonlPath, jsonlContent);
				await writeAtomically(expectedPath, expectedContent);
			}

			results.push({
				rel: built.rel,
				jsonlLines: built.lines.length,
				entries: built.view.entries.length,
				contextMessages: built.view.context.messages.length,
				formatVersion: built.formatVersion,
			});
		}
	} finally {
		try {
			rmSync(workRoot, { recursive: true, force: true });
		} catch {
			// best-effort cleanup
		}
	}

	const manifest = {
		count: results.length,
		fixtures: results.map((r) => r.rel),
	};
	const manifestContent = `${JSON.stringify(manifest, null, 2)}\n`;

	if (options.check) {
		const manifestOnDisk = await readFile(
			join(outDir, FIXTURE_MANIFEST_NAME),
			"utf8",
		).catch(() => null);
		if (manifestOnDisk !== manifestContent) stale.push(FIXTURE_MANIFEST_NAME);
		return { outDir, results, pruned: [], stale };
	}

	// Prune stale generator-owned pairs only after every current pair was
	// written successfully; publish the manifest only after pruning. OUT_DIR
	// is never deleted: a failed run leaves the previous complete tree and
	// manifest intact.
	const pruned = await pruneStaleFixturePairs(
		outDir,
		results.map((r) => r.rel),
	);

	// Authoritative manifest: the Rust interop test reads this at runtime
	// instead of scraping the generator source, so the expected fixture
	// count always reflects what was actually written to disk.
	await writeAtomically(
		join(outDir, FIXTURE_MANIFEST_NAME),
		manifestContent,
	);

	return { outDir, results, pruned };
}

async function main(): Promise<void> {
	const check = process.argv.includes("--check");
	const { outDir, results, pruned, stale } = await generateSessionFixtures({ check });

	if (check) {
		if (stale && stale.length > 0) {
			process.stderr.write(
				`stale session fixtures under ${outDir}:\n${stale.map((f) => `  ${f}`).join("\n")}\n`,
			);
			process.exit(1);
		}
		process.stdout.write(`SESSION_FIXTURES_FRESH ${outDir}\n`);
		return;
	}

	// Summary
	const totalJsonl = results.length;
	const totalExpected = results.length;
	const totalLines = results.reduce((n, r) => n + r.jsonlLines, 0);
	const totalEntries = results.reduce((n, r) => n + r.entries, 0);
	const totalContext = results.reduce((n, r) => n + r.contextMessages, 0);

	const linesOut: string[] = [
		`Wrote ${totalJsonl} JSONL + ${totalExpected} expected under ${outDir}`,
		`totals: lines=${totalLines} entries=${totalEntries} contextMessages=${totalContext}`,
		`source: ${REF_SESSION_MANAGER}`,
	];
	if (pruned.length > 0) {
		linesOut.push(
			`pruned ${pruned.length} stale fixture file(s): ${pruned.join(", ")}`,
		);
	}
	for (const r of results) {
		linesOut.push(
			`  ${r.rel}: v${r.formatVersion} lines=${r.jsonlLines} entries=${r.entries} context=${r.contextMessages}`,
		);
	}
	process.stdout.write(`${linesOut.join("\n")}\n`);
}

if (import.meta.main) {
	await main();
}
