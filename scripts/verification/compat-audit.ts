#!/usr/bin/env bun
/**
 * PAR-COMPAT-AUDIT witness suite (issue #59).
 *
 * Six executable negative witnesses adjudicate the upstream `./compat`
 * legacy global provider registry (ledger row A8) before any deletion or
 * port decision.  Each witness returns an empty array when green; any
 * non-empty result is a violation that blocks the A8 ledger flip.
 *
 *  1. Source evidence — compat.ts export map and side-effect inventory.
 *  2. Downstream-import evidence — exhaustive importer enumeration,
 *     classified TS-side-runtime vs Rust-surface.
 *  3. Extension-host routing — Mode 1 alias maps `@earendil-works/pi-ai`
 *     to the JS bundle's `./compat` entry, NOT a Rust port.
 *  4. Config corpus — env-key resolution already in A7 auth
 *     (`auth/env_keys.rs`); `Model.compat` already adapter-local in Rust.
 *  5. Rust-surface negative witness — no Rust source references the
 *     `./compat` module.
 *  6. PAR-COMPAT-DISPO single-owner witness — the dead
 *     `pi::core::config_value` wrapper stays deleted; exactly one parser
 *     and one process-wide command cache exist, both owned by
 *     `crates/pi-ai/src/auth/config_value.rs`.
 */

import { readFileSync, readdirSync, statSync } from "node:fs";
import type { Dirent } from "node:fs";
import { join, relative, resolve, sep } from "node:path";
import { CANONICAL_REFERENCE_ROOT, assertCanonicalReference } from "../reference-identity.ts";

export const REPO_ROOT = resolve(import.meta.dirname, "../..");
const REF_ROOT = join(REPO_ROOT, CANONICAL_REFERENCE_ROOT, "packages");
const COMPAT_TS = join(REF_ROOT, "ai", "src", "compat.ts");

// ---------------------------------------------------------------------------
// Source evidence: compat.ts export map and side-effect inventory
// ---------------------------------------------------------------------------

/** Pinned `export *` re-export sources from compat.ts (lines 13–29). */
const EXPECTED_REEXPORTS: readonly string[] = [
	"./api/anthropic-messages.lazy.ts",
	"./api/azure-openai-responses.lazy.ts",
	"./api/bedrock-converse-stream.lazy.ts",
	"./api/google-generative-ai.lazy.ts",
	"./api/google-vertex.lazy.ts",
	"./api/mistral-conversations.lazy.ts",
	"./api/openai-codex-responses.lazy.ts",
	"./api/openai-completions.lazy.ts",
	"./api/openai-responses.lazy.ts",
	"./api/pi-messages.lazy.ts",
	"./env-api-keys.ts",
	"./image-models.ts",
	"./images.ts",
	"./images-api-registry.ts",
	"./index.ts",
	"./legacy-api-aliases.ts",
	"./providers/images/register-builtins.ts",
];

/** Pinned named exports declared directly in compat.ts. */
const EXPECTED_DIRECT_EXPORTS: readonly string[] = [
	"getModel",
	"getModels",
	"getProviders",
	"ApiStreamFunction",
	"ApiStreamSimpleFunction",
	"ApiProvider",
	"registerApiProvider",
	"getApiProvider",
	"getApiProviders",
	"unregisterApiProviders",
	"registerFauxProvider",
	"registerBuiltInApiProviders",
	"resetApiProviders",
	"stream",
	"complete",
	"streamSimple",
	"completeSimple",
	"BuiltinProvider",
];

/** Pinned module-level side effects (executed on import, not inside a function). */
const EXPECTED_SIDE_EFFECTS: readonly string[] = [
	"registerBuiltInApiProviders();",
	"const compatModels = builtinModels();",
];

/** Pinned env-key resolution import. */
const EXPECTED_ENV_IMPORT = "getEnvApiKey";

export interface SourceInventory {
	reexports: string[];
	directExports: string[];
	sideEffects: string[];
	hasEnvImport: boolean;
}

export function parseCompatSource(source: string): SourceInventory {
	// Strip line comments so regex matching does not pick up commented-out
	// exports or side effects.  A `//` outside a string literal starts a comment.
	const strippedLines: string[] = [];
	for (const line of source.split("\n")) {
		let inString = false;
		let cutAt = line.length;
		for (let i = 0; i < line.length; i++) {
			const ch = line[i];
			if (ch === '"') inString = !inString;
			if (ch === "/" && line[i + 1] === "/" && !inString) {
				cutAt = i;
				break;
			}
		}
		strippedLines.push(line.slice(0, cutAt));
	}
	const stripped = strippedLines.join("\n");

	const reexportRe = /export\s+\*\s+from\s+"([^"]+)"/g;
	const reexports: string[] = [];
	let m: RegExpExecArray | null;
	while ((m = reexportRe.exec(stripped)) !== null) {
		const captured = m[1];
		if (captured !== undefined) reexports.push(captured);
	}

	// Named export declarations: `export function`, `export async function`,
	// `export const`, `export type X = ...`, `export interface`, and
	// `export type { X }` re-exports.
	const exportRe =
		/export\s+(?:async\s+)?(?:function|const|type|interface)\s+(\w+)/g;
	const directExports: string[] = [];
	while ((m = exportRe.exec(stripped)) !== null) {
		const captured = m[1];
		if (captured !== undefined) directExports.push(captured);
	}

	// `export type { Name, ... }` re-exports.
	const typeReexportRe = /export\s+type\s+\{([^}]+)\}/g;
	while ((m = typeReexportRe.exec(stripped)) !== null) {
		const captured = m[1];
		if (captured === undefined) continue;
		const names = captured.split(",").map((s) => s.trim());
		for (const name of names) {
			if (name) directExports.push(name);
		}
	}

	// Module-level side effects: top-level statements that execute on import.
	// We detect lines at column 0 (no indentation) that are not declarations,
	// imports, exports, or comments, and that call a function or initialise
	// a const with a function call.
	const sideEffects: string[] = [];
	const lines = stripped.split("\n");
	let inBlock = false;
	for (const line of lines) {
		const trimmed = line.trim();
		if (
			trimmed.startsWith("function ") ||
			trimmed.startsWith("export function") ||
			trimmed.startsWith("export async function") ||
			trimmed.startsWith("interface ") ||
			trimmed.startsWith("export interface") ||
			trimmed.startsWith("type ") ||
			trimmed.startsWith("export type")
		) {
			inBlock = true;
		}
		if (inBlock && trimmed === "}") inBlock = false;
		if (
			!inBlock &&
			line.length > 0 &&
			!line.startsWith(" ") &&
			!line.startsWith("\t") &&
			!line.startsWith("//") &&
			!line.startsWith("import") &&
			!line.startsWith("export") &&
			!line.startsWith("/**") &&
			!line.startsWith("/*") &&
			!line.startsWith("*") &&
			!line.startsWith("}")
		) {
			// Side effects: bare function calls and const-with-call initialisers.
			if (trimmed.endsWith(";")) {
				sideEffects.push(trimmed);
			}
		}
	}

	const hasEnvImport = source.includes(EXPECTED_ENV_IMPORT);

	return { reexports, directExports, sideEffects, hasEnvImport };
}

export function verifySourceEvidence(source: string): string[] {
	const violations: string[] = [];
	const inv = parseCompatSource(source);

	for (const expected of EXPECTED_REEXPORTS) {
		if (!inv.reexports.includes(expected)) {
			violations.push(`compat.ts missing re-export of "${expected}"`);
		}
	}
	for (const expected of EXPECTED_DIRECT_EXPORTS) {
		if (!inv.directExports.includes(expected)) {
			violations.push(`compat.ts missing direct export "${expected}"`);
		}
	}
	for (const expected of EXPECTED_SIDE_EFFECTS) {
		if (!inv.sideEffects.includes(expected)) {
			violations.push(`compat.ts missing side effect "${expected}"`);
		}
	}
	if (!inv.hasEnvImport) {
		violations.push(`compat.ts missing env-key import "${EXPECTED_ENV_IMPORT}"`);
	}

	return violations;
}

// ---------------------------------------------------------------------------
// Downstream-import evidence: exhaustive importer enumeration
// ---------------------------------------------------------------------------

/** Allowed TS-side-runtime consumer packages in the reference checkout. */
const ALLOWED_CONSUMER_PACKAGES: readonly string[] = [
	"ai",
	"agent",
	"coding-agent",
];

/** Allowed subdirectories within each consumer package. */
const ALLOWED_SUBDIRS: readonly string[] = ["src", "test", "examples"];

function listFilesRecursive(dir: string, ext: string): string[] {
	const results: string[] = [];
	let entries: Dirent[];
	try {
		entries = readdirSync(dir, { withFileTypes: true });
	} catch {
		return results;
	}
	for (const entry of entries) {
		const fullPath = join(dir, entry.name);
		if (entry.isDirectory()) {
			if (entry.name === "node_modules" || entry.name === "dist") continue;
			results.push(...listFilesRecursive(fullPath, ext));
		} else if (entry.isFile() && entry.name.endsWith(ext)) {
			results.push(fullPath);
		}
	}
	return results;
}

export interface ImporterRecord {
	file: string;
	pkg: string;
	subdir: string;
	specifier: string;
}

/** Patterns that match compat imports but NOT unrelated "compat" substrings. */
const COMPAT_IMPORT_RE =
	/from\s+["'](?:@earendil-works\/pi-ai\/compat|@mariozechner\/pi-ai\/compat|\.\.\/\.\.\/src\/compat\.ts|\.\.\/src\/compat\.ts|\.\.\/\.\.\/\.\.\/src\/compat\.ts)["']/g;

export function enumerateCompatImporters(refRoot: string): {
	importers: ImporterRecord[];
	problems: string[];
} {
	const importers: ImporterRecord[] = [];
	const problems: string[] = [];

	// Scan every package directory in the reference checkout, not just the
	// allowed ones, so an importer in an unexpected package is caught.
	let pkgEntries: Dirent[];
	try {
		pkgEntries = readdirSync(refRoot, { withFileTypes: true });
	} catch {
		problems.push(`reference root "${refRoot}" not readable`);
		return { importers, problems };
	}

	for (const pkgEntry of pkgEntries) {
		if (!pkgEntry.isDirectory()) continue;
		const pkgDir = join(refRoot, pkgEntry.name);
		const files = listFilesRecursive(pkgDir, ".ts");
		for (const file of files) {
			const content = readFileSync(file, "utf8");
			COMPAT_IMPORT_RE.lastIndex = 0;
			let m: RegExpExecArray | null;
			while ((m = COMPAT_IMPORT_RE.exec(content)) !== null) {
				const relPath = relative(refRoot, file).replaceAll("\\", "/");
				const parts = relPath.split("/");
				const pkgName = parts[0] ?? "";
				const subdir = parts[1] ?? "";
				importers.push({
					file: relPath,
					pkg: pkgName,
					subdir,
					specifier: m[0],
				});
		}
	}
	}
	return { importers, problems };
}

export function verifyDownstreamImporters(refRoot: string): string[] {
	const violations: string[] = [];
	const { importers, problems } = enumerateCompatImporters(refRoot);
	violations.push(...problems);

	if (importers.length === 0) {
		violations.push("no compat importers found — enumeration may be broken");
	}

	for (const imp of importers) {
		if (!ALLOWED_CONSUMER_PACKAGES.includes(imp.pkg)) {
			violations.push(`compat importer in unexpected package "${imp.pkg}": ${imp.file}`);
		}
		if (!ALLOWED_SUBDIRS.includes(imp.subdir)) {
			violations.push(`compat importer in unexpected subdir "${imp.subdir}": ${imp.file}`);
		}
	}

	return violations;
}

// ---------------------------------------------------------------------------
// Extension-host routing witness
// ---------------------------------------------------------------------------

export function verifyExtensionHostRouting(repoRoot: string): string[] {
	const violations: string[] = [];

	const virtualModulesPath = join(
		repoRoot,
		"packages",
		"extension-host",
		"src",
		"virtual-modules.ts",
	);
	let vmSource: string;
	try {
		vmSource = readFileSync(virtualModulesPath, "utf8");
	} catch {
		violations.push("packages/extension-host/src/virtual-modules.ts not readable");
		return violations;
	}

	// Compiled mode: _bundledPiAiCompat is served for both @earendil-works/pi-ai
	// and @earendil-works/pi-ai/compat.
	if (!vmSource.includes('"@earendil-works/pi-ai": _bundledPiAiCompat')) {
		violations.push("virtual-modules.ts does not route @earendil-works/pi-ai to _bundledPiAiCompat");
	}
	if (!vmSource.includes('"@earendil-works/pi-ai/compat": _bundledPiAiCompat')) {
		violations.push("virtual-modules.ts does not route @earendil-works/pi-ai/compat to _bundledPiAiCompat");
	}

	// Source mode: alias maps to compat.ts reference source.
	if (!vmSource.includes('const aiCompat = `${REF_ROOT}/ai/src/compat.ts`')) {
		violations.push("virtual-modules.ts source-mode alias does not point to ai/src/compat.ts");
	}

	// The host itself imports validateToolArguments from compat.
	const hostPath = join(repoRoot, "packages", "extension-host", "src", "host.ts");
	let hostSource: string;
	try {
		hostSource = readFileSync(hostPath, "utf8");
	} catch {
		violations.push("packages/extension-host/src/host.ts not readable");
		return violations;
	}
	if (!hostSource.includes('from "@earendil-works/pi-ai/compat"')) {
		violations.push("host.ts does not import from @earendil-works/pi-ai/compat");
	}

	return violations;
}

// ---------------------------------------------------------------------------
// Config corpus witness: env-key resolution in A7, Model.compat adapter-local
// ---------------------------------------------------------------------------

export function verifyConfigCorpus(repoRoot: string): string[] {
	const violations: string[] = [];

	// A7 auth: env_keys.rs must contain get_env_api_key and find_env_keys.
	const envKeysPath = join(repoRoot, "crates", "pi-ai", "src", "auth", "env_keys.rs");
	let envKeysSource: string;
	try {
		envKeysSource = readFileSync(envKeysPath, "utf8");
	} catch {
		violations.push("crates/pi-ai/src/auth/env_keys.rs not readable");
		return violations;
	}
	if (!envKeysSource.includes("pub fn get_env_api_key")) {
		violations.push("auth/env_keys.rs missing get_env_api_key (A7 env-key resolution)");
	}
	if (!envKeysSource.includes("pub fn find_env_keys")) {
		violations.push("auth/env_keys.rs missing find_env_keys (A7 env-key discovery)");
	}
	// Spot-check a few provider env keys that must match upstream.
	for (const envKey of ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"]) {
		if (!envKeysSource.includes(envKey)) {
			violations.push(`auth/env_keys.rs missing env key "${envKey}"`);
		}
	}

	// Model.compat field is adapter-local in Rust types.
	const typesPath = join(repoRoot, "crates", "pi-ai", "src", "types.rs");
	let typesSource: string;
	try {
		typesSource = readFileSync(typesPath, "utf8");
	} catch {
		violations.push("crates/pi-ai/src/types.rs not readable");
		return violations;
	}
	if (!typesSource.includes("pub compat: Option<Value>")) {
		violations.push("types.rs missing Model.compat field (adapter-local compat settings)");
	}

	// At least one Rust provider must consume model.compat.
	const providersDir = join(repoRoot, "crates", "pi-ai", "src", "providers");
	let hasCompatConsumer = false;
	try {
		for (const file of listFilesRecursive(providersDir, ".rs")) {
			if (readFileSync(file, "utf8").includes(".compat")) {
				hasCompatConsumer = true;
				break;
			}
		}
	} catch {
		// ignore
	}
	if (!hasCompatConsumer) {
		violations.push("no Rust provider consumes model.compat (adapter-local consumption missing)");
	}

	return violations;
}

// ---------------------------------------------------------------------------
// Rust-surface negative witness: no Rust source references the compat module
// ---------------------------------------------------------------------------

export function verifyNoRustCompatConsumer(repoRoot: string): string[] {
	const violations: string[] = [];
	const cratesDir = join(repoRoot, "crates");

	const rustFiles = listFilesRecursive(cratesDir, ".rs");
	for (const file of rustFiles) {
		const source = readFileSync(file, "utf8");
		const lines = source.split("\n");
		for (let i = 0; i < lines.length; i++) {
			const line = lines[i] ?? "";
			// Check for pi_ai::compat module references
			if (/\bpi_ai::compat\b/.test(line)) {
				violations.push(`${file}:${i + 1}: Rust references pi_ai::compat module`);
			}
			// Check for mod compat declaration (module-level, not struct field)
			if (/^\s*(pub\s+)?mod\s+compat\b/.test(line)) {
				violations.push(`${file}:${i + 1}: Rust declares mod compat`);
			}
		}
	}

	return violations;
}

// ---------------------------------------------------------------------------
// PAR-COMPAT-DISPO witness: single-owner config-value resolution (issue #45)
// ---------------------------------------------------------------------------

/** The one module allowed to parse config values and own the command cache. */
const CANONICAL_CONFIG_VALUE = join("crates", "pi-ai", "src", "auth", "config_value.rs").split(sep).join("/");

/** Siblings inside the auth module may `use super::config_value`. */
const AUTH_DIR = join("crates", "pi-ai", "src", "auth").split(sep).join("/");

function isWithin(path: string, dir: string): boolean {
	const relative = path.startsWith(dir + "/") || path.startsWith(dir + "\\");
	return relative;
}

/** Cut a Rust line at the first `//` that is outside a string literal. */
function stripLineComment(line: string): string {
	let inString = false;
	for (let i = 0; i < line.length; i++) {
		const ch = line[i];
		if (ch === '"') inString = !inString;
		if (ch === '/' && line[i + 1] === '/' && !inString) return line.slice(0, i);
	}
	return line;
}

/**
 * Scan every Rust source under crates/ and attribute each single-owner
 * definition (parser, command cache, cache-clear) to its defining file.
 * Paths are repo-root relative with forward slashes.
 */
function scanConfigValueOwnership(repoRoot: string) {
	const cratesDir = join(repoRoot, "crates");
	const rustFiles = listFilesRecursive(cratesDir, ".rs");
	const modules: string[] = [];
	const parserFiles: string[] = [];
	const cacheFiles: string[] = [];
	const clearFiles: string[] = [];
	const strayDeclarations: string[] = [];
	const strayImports: string[] = [];

	for (const file of rustFiles) {
		// Native separators on both sides: every comparison below (join-built
		// constants, isWithin) already speaks native. String-stripping the
		// root with a posix "/" never matches on Windows and leaks absolute
		// paths into every witness verdict.
		const rel = relative(repoRoot, file).split(sep).join("/");
		if (file.endsWith("config_value.rs") || file.endsWith(join("config_value", "mod.rs"))) modules.push(rel);
		const source = readFileSync(file, "utf8");
		if (source.includes("fn parse_config_value_reference")) parserFiles.push(rel);
		if (source.includes("static COMMAND_CACHE")) cacheFiles.push(rel);
		if (source.includes("fn clear_config_value_cache")) clearFiles.push(rel);
		const lines = source.split("\n");
		for (let i = 0; i < lines.length; i++) {
			const line = stripLineComment(lines[i] ?? "");
			if (/^\s*(pub(?:\s*\([$\w]+\))?\s+)?mod\s+config_value\b/.test(line) && rel !== `${AUTH_DIR}/mod.rs`) {
				strayDeclarations.push(`${rel}:${i + 1}: ${line.trim()}`);
			}
		}
		const codeOnly = lines.map((l) => stripLineComment(l ?? "")).join("\n");
		if (!isWithin(rel, AUTH_DIR) && /\b(?:crate|super|self)(?:::\w+)*::config_value\b|\bpi::core::config_value\b/.test(codeOnly)) {
			strayImports.push(rel);
		}
	}
	return { modules, parserFiles, cacheFiles, clearFiles, strayDeclarations, strayImports };
}

/**
 * Witness 6 — the PAR-COMPAT-DISPO delete-not-port disposition, made
 * permanent: the dead `pi::core::config_value` HashMap-shaped wrapper stays
 * deleted, and config-value resolution keeps exactly one parser and one
 * process-wide command cache, both in `pi-ai/src/auth/config_value.rs`.
 */
export function verifyConfigValueSingleOwner(repoRoot: string): string[] {
	const violations: string[] = [];
	const ownership = scanConfigValueOwnership(repoRoot);

	if (ownership.modules.length !== 1 || ownership.modules[0] !== CANONICAL_CONFIG_VALUE) {
		violations.push(
			`expected exactly one config_value module (${CANONICAL_CONFIG_VALUE}); found: ${
				ownership.modules.length === 0 ? "none" : ownership.modules.join(", ")
			}`,
		);
	}
	if (ownership.parserFiles.length !== 1 || ownership.parserFiles[0] !== CANONICAL_CONFIG_VALUE) {
		violations.push(
			`expected exactly one config-value parser in ${CANONICAL_CONFIG_VALUE}; found: ${
				ownership.parserFiles.length === 0 ? "none" : ownership.parserFiles.join(", ")
			}`,
		);
	}
	if (ownership.cacheFiles.length !== 1 || ownership.cacheFiles[0] !== CANONICAL_CONFIG_VALUE) {
		violations.push(
			`expected exactly one process-wide command cache (static COMMAND_CACHE) in ${CANONICAL_CONFIG_VALUE}; found: ${
				ownership.cacheFiles.length === 0 ? "none" : ownership.cacheFiles.join(", ")
			}`,
		);
	}
	if (ownership.clearFiles.length !== 1 || ownership.clearFiles[0] !== CANONICAL_CONFIG_VALUE) {
		violations.push(
			`expected exactly one clear_config_value_cache definition in ${CANONICAL_CONFIG_VALUE}; found: ${
				ownership.clearFiles.length === 0 ? "none" : ownership.clearFiles.join(", ")
			}`,
		);
	}
	for (const declaration of ownership.strayDeclarations) {
		violations.push(`second config_value module declared outside the canonical auth owner: ${declaration}`);
	}
	for (const file of ownership.strayImports) {
		violations.push(`${file}: imports a local config_value module instead of pi_ai::auth::config_value`);
	}

	return violations;
}

// ---------------------------------------------------------------------------
// Orchestration
// ---------------------------------------------------------------------------

export function runCompatAuditWitnesses(repoRoot: string): string[] {
	assertCanonicalReference(repoRoot);
	const violations: string[] = [];
	const add = (witness: string, results: readonly string[]): void => {
		for (const result of results) violations.push(`[${witness}] ${result}`);
	};

	// Witness 1: source evidence
	let compatSource: string;
	try {
		compatSource = readFileSync(COMPAT_TS, "utf8");
	} catch {
		add("source-evidence", [`${CANONICAL_REFERENCE_ROOT}/packages/ai/src/compat.ts not readable`]);
		return violations;
	}
	add("source-evidence", verifySourceEvidence(compatSource));

	// Witness 2: downstream importers
	add("downstream-importers", verifyDownstreamImporters(REF_ROOT));

	// Witness 3: extension-host routing
	add("extension-host-routing", verifyExtensionHostRouting(repoRoot));

	// Witness 4: config corpus
	add("config-corpus", verifyConfigCorpus(repoRoot));

	// Witness 5: Rust-surface negative
	add("rust-negative", verifyNoRustCompatConsumer(repoRoot));

	// Witness 6: PAR-COMPAT-DISPO single-owner disposition
	add("config-value-single-owner", verifyConfigValueSingleOwner(repoRoot));

	return violations;
}

function main(): void {
	const violations = runCompatAuditWitnesses(REPO_ROOT);
	if (violations.length > 0) {
		console.error(`compat audit witness suite failed with ${violations.length} violation(s):`);
		for (const violation of violations) console.error(`  - ${violation}`);
		process.exit(1);
	}
	process.stdout.write("COMPAT_AUDIT_WITNESSES_OK\n");
}

if (import.meta.main) main();
