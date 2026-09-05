#!/usr/bin/env bun
/**
 * Shipped-exposure predicate checker (DEPS-R2, issue #128).
 *
 * Classifies a proposed dependency remediation (CVE / yanked version) as
 * Class E (complete E1–E4 pass bundle; only the seven-target lane is
 * skippable) or Class S (shipped-exposed; full seven-target post-audit
 * including both musl artifact proofs). Fail-closed: every undecidable
 * check maps its subject to Class S, and a crash anywhere in this
 * classifier still emits `DEPENDENCY_EXPOSURE_FAILED_CLOSED` with exit 1.
 *
 * Redesign disposition (issue #128 review waves — the prior prototype was
 * reverted at 8051e59 for systemic fail-open paths; this implementation
 * closes each finding by construction):
 *
 * 1. No `--input auto`, no change-detection short-circuit. Subjects are
 *    explicit (`--subject npm:typebox`), every invocation recomputes over
 *    the full current tree, and cross-ecosystem byte-identity is enforced:
 *    an npm subject requires every Cargo manifest + Cargo.lock to be
 *    byte-identical to the reference capture, so a Cargo.toml-only edge or
 *    feature change fails E1 closed even when Cargo.lock is unchanged.
 *    Rust edges are read exclusively from `cargo metadata` dep-graph
 *    projections (captured with `--all-features --locked --offline`), never
 *    from manifest text.
 * 2. E3 covers the release pipeline's actual `CommandRunner.run` seam: it
 *    scans every `.run(` call site in scripts/release/** and both release
 *    entry scripts, plus every direct `bun build` / `cargo build`
 *    invocation in .github/workflows/*.yml. A build-capable site whose
 *    arguments cannot be attributed to literals or the authority argvs is
 *    undecidable.
 * 3. The classifier never executes head-side package code: no `bun
 *    install`, no `bun build`, no lifecycle scripts. The bundler metafile
 *    comes from the pre-change reference capture. The only child process
 *    is `cargo metadata --locked --offline --all-features` (no build
 *    scripts, no package code, no daemonization), overridable with
 *    `--cargo-metadata-file` for hermetic runs.
 * 4. The canonical reference is a hash-chained manifest: reference.json
 *    pins the sha256 of both projections, and the metafile projection pins
 *    the sha256 of every module-graph input plus the metafile itself.
 *    Authority modules (scripts/release/{host,targets,stage}.ts) are
 *    byte-compared against the reference before they are imported.
 * 5. No process-group teardown surface: nothing is built here, so there is
 *    no descendant tree to contain. The single child (cargo metadata) is a
 *    direct non-shell child killed with SIGKILL on timeout.
 *
 * Commands:
 *   capture-reference --out <dir>   capture the pre-change reference (runs
 *                                   the authority bun build once, on the
 *                                   trusted pre-change tree, to emit the
 *                                   bundler metafile)
 *   classify --subject <kind:name> --reference <dir>
 *                                   emit the E1–E4 verdict (exit 0 for any
 *                                   decided class; exit 1 + sentinel only
 *                                   when the classifier itself crashed)
 *   self-check                      known-member sanity + fail-closed
 *                                   probes against the checked-in
 *                                   canonical reference
 *
 * Output schema: pi.deps.exposure.v1.
 */
import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, readdirSync, statSync, writeFileSync } from "node:fs";
import { dirname, join, posix, relative, resolve } from "node:path";
import { tmpdir } from "node:os";

import type { hostBundleCommands } from "../release/host.ts";
import type { planFor, TARGET_PLANS } from "../release/targets.ts";
import type { stagedInputs } from "../release/stage.ts";
import type { Fs } from "../release/runner.ts";

export const SCHEMA = "pi.deps.exposure.v1" as const;
export const REFERENCE_SCHEMA = "pi.deps.exposure-reference.v1" as const;
export const METAFILE_PROJECTION_SCHEMA = "pi.deps.exposure-metafile.v1" as const;
export const CARGO_GRAPH_PROJECTION_SCHEMA = "pi.deps.exposure-cargo-graph.v1" as const;
export const SENTINEL_OK = "DEPENDENCY_EXPOSURE_OK";
export const SENTINEL_FAILED_CLOSED = "DEPENDENCY_EXPOSURE_FAILED_CLOSED";

export const REPO_ROOT = resolve(import.meta.dirname, "../..");
export const HOST_PACKAGE_DIR = "packages/extension-host";
/** Host dir the bundler resolves metafile input paths against. */
export const HOST_DIR = join(REPO_ROOT, HOST_PACKAGE_DIR);
/** Release authority modules byte-compared against the reference before import. */
export const AUTHORITY_REL_PATHS = [
	"scripts/release/host.ts",
	"scripts/release/targets.ts",
	"scripts/release/stage.ts",
] as const;
/** Toolchain-shaped subjects; all are Class S by the decision table. */
export const TOOL_NAMES = ["rust-toolchain", "bun-runtime", "bun-bundler"] as const;
export const CHECK_NAMES = ["E1", "E2", "E3", "E4"] as const;
/** Checked-in canonical reference used by `self-check` and CI. */
export const CANONICAL_REFERENCE_DIR = join(
	REPO_ROOT,
	"scripts/verification/fixtures/dependency-exposure/reference",
);
/** cargo metadata argv captured and replayed for the Rust dep graph. */
export const CARGO_METADATA_ARGV = [
	"metadata",
	"--format-version",
	"1",
	"--locked",
	"--offline",
	"--all-features",
] as const;
/** Commands that can produce build outputs when driven through the seam. */
const BUILD_CAPABLE_COMMANDS = new Set(["bun", "cargo", "tsc", "npm", "npx", "yarn", "pnpm"]);
/** Argument tokens that mark a seam site as shipped-byte producing. */
const EMIT_TOKENS = new Set(["build", "--compile", "--outfile", "--outdir", "--release"]);
/** npm dependency fields that are NOT dev-only. */
const NON_DEV_FIELDS = ["dependencies", "optionalDependencies", "peerDependencies"] as const;
const DEP_FIELDS = [...NON_DEV_FIELDS, "devDependencies"] as const;
const SHA256_RE = /^[0-9a-f]{64}$/;

export class ExposureError extends Error {
	constructor(message: string) {
		super(message);
		this.name = "ExposureError";
	}
}

// ---------------------------------------------------------------------------
// JSON boundary parsers (strict: unknown in, typed out, throw on mismatch)
// ---------------------------------------------------------------------------

function asRecord(value: unknown, what: string): Record<string, unknown> {
	if (value === null || typeof value !== "object" || Array.isArray(value)) {
		throw new ExposureError(`${what}: expected an object`);
	}
	// Validated boundary: the checks above leave exactly `object`; assert the
	// index signature once, here, rather than guarding at every field access.
	return value as Record<string, unknown>;
}

function reqString(holder: Record<string, unknown>, key: string, what: string): string {
	const value = holder[key];
	if (typeof value !== "string" || value.length === 0) {
		throw new ExposureError(`${what}: field "${key}" must be a non-empty string`);
	}
	return value;
}

function optString(holder: Record<string, unknown>, key: string): string | undefined {
	const value = holder[key];
	return typeof value === "string" ? value : undefined;
}

function reqStringArray(holder: Record<string, unknown>, key: string, what: string): string[] {
	const value = holder[key];
	if (!Array.isArray(value) || value.some((entry) => typeof entry !== "string")) {
		throw new ExposureError(`${what}: field "${key}" must be an array of strings`);
	}
	return value as string[];
}

function reqSha256(holder: Record<string, unknown>, key: string, what: string): string {
	const value = reqString(holder, key, what);
	if (!SHA256_RE.test(value)) {
		throw new ExposureError(`${what}: field "${key}" must be a lowercase sha256 hex digest`);
	}
	return value;
}

function parseJson(text: string, what: string): Record<string, unknown> {
	let parsed: unknown;
	try {
		parsed = JSON.parse(text);
	} catch (error) {
		throw new ExposureError(`${what}: invalid JSON (${errorText(error)})`);
	}
	return asRecord(parsed, what);
}

// ---------------------------------------------------------------------------
// Shared result vocabulary
// ---------------------------------------------------------------------------

export type SubjectKind = "npm" | "crate" | "tool";
export type CheckStatus = "pass" | "fail" | "undecidable";
export type ExposureClass = "S" | "E";
export type CheckName = (typeof CHECK_NAMES)[number];

export interface Subject {
	readonly kind: SubjectKind;
	readonly name: string;
	readonly raw: string;
}

export interface CheckResult {
	readonly status: CheckStatus;
	readonly detail: string;
}

export interface Verdict {
	readonly subject: string;
	readonly exposureClass: ExposureClass;
	readonly reason: string;
}

export interface ExposureReport {
	readonly schema: typeof SCHEMA;
	readonly subject: string;
	readonly verdict: Verdict;
	readonly checks: Record<CheckName, CheckResult>;
	readonly referenceDir: string;
	readonly capturedAt: string;
	readonly sentinel: typeof SENTINEL_OK;
}

const NPM_NAME_RE = /^(?:@[a-z0-9-~][a-z0-9-._~]*\/)?[a-z0-9-~][a-z0-9-._~]*$/;
const CARGO_NAME_RE = /^[A-Za-z][A-Za-z0-9_-]*$/;

export function parseSubject(raw: string): Subject {
	const kindSep = raw.indexOf(":");
	const invalid = (): never => {
		throw new ExposureError(
			`invalid subject "${raw}" (expected npm:<name>, crate:<name>, or tool:<${TOOL_NAMES.join("|")}>)`,
		);
	};
	if (kindSep <= 0) invalid();
	const kindText = raw.slice(0, kindSep);
	const name = raw.slice(kindSep + 1);
	if (name.length === 0) invalid();
	if (kindText === "npm") {
		if (!NPM_NAME_RE.test(name)) invalid();
		return { kind: "npm", name, raw };
	}
	if (kindText === "crate") {
		if (!CARGO_NAME_RE.test(name)) invalid();
		return { kind: "crate", name, raw };
	}
	const isTool: Record<string, true> = { "rust-toolchain": true, "bun-runtime": true, "bun-bundler": true };
	if (kindText === "tool") {
		if (isTool[name] !== true) invalid();
		return { kind: "tool", name, raw };
	}
	throw new ExposureError(
		`invalid subject "${raw}" (expected npm:<name>, crate:<name>, or tool:<${TOOL_NAMES.join("|")}>)`,
	);
}

export function pass(detail: string): CheckResult {
	return { status: "pass", detail };
}
export function fail(detail: string): CheckResult {
	return { status: "fail", detail };
}
export function undecidable(detail: string): CheckResult {
	return { status: "undecidable", detail };
}

export function verdictFromChecks(subject: Subject, checks: Record<CheckName, CheckResult>): Verdict {
	const failed = CHECK_NAMES.filter((name) => checks[name].status === "fail");
	if (failed.length > 0) {
		return {
			subject: subject.raw,
			exposureClass: "S",
			reason: `${failed.join(", ")} failed: ${failed.map((n) => checks[n].detail).join(" | ")}`,
		};
	}
	const undecided = CHECK_NAMES.filter((name) => checks[name].status === "undecidable");
	if (undecided.length > 0) {
		return {
			subject: subject.raw,
			exposureClass: "S",
			reason: `fail-closed (${undecided.join(", ")} undecidable): ${undecided
				.map((n) => checks[n].detail)
				.join(" | ")}`,
		};
	}
	return {
		subject: subject.raw,
		exposureClass: "E",
		reason: "complete E1–E4 exemption bundle; only the seven-target lane is skippable",
	};
}

export function errorText(error: unknown): string {
	return error instanceof Error ? error.message : String(error);
}

// ---------------------------------------------------------------------------
// Hashing helpers
// ---------------------------------------------------------------------------

export function sha256Bytes(data: Uint8Array): string {
	return createHash("sha256").update(data).digest("hex");
}

export function sha256Text(text: string): string {
	return sha256Bytes(Buffer.from(text, "utf8"));
}

export function sha256FileAt(path: string): string {
	if (!existsSync(path) || !statSync(path).isFile()) {
		throw new ExposureError(`missing file to hash: ${path}`);
	}
	return sha256Bytes(readFileSync(path));
}

// ---------------------------------------------------------------------------
// npm surfaces (E1): root + every workspace package.json
// ---------------------------------------------------------------------------

export interface NpmSurface {
	readonly path: string;
	readonly relPath: string;
	readonly sha256: string;
	/** dep-field name -> package names declared in that field. */
	readonly depFields: Readonly<Record<string, readonly string[]>>;
	/** package.json "name" field (workspace package identity). */
	readonly packageName: string | undefined;
	readonly scripts: Readonly<Record<string, string>>;
}

/** Discover the package.json surfaces: the root manifest plus every workspace glob member. */
export function discoverSurfacePaths(root: string): string[] {
	const rootPath = join(root, "package.json");
	if (!existsSync(rootPath)) throw new ExposureError(`no root package.json under ${root}`);
	const rootJson = parseJson(readFileSync(rootPath, "utf8"), rootPath);
	const workspaces = reqStringArray(rootJson, "workspaces", rootPath);
	const surfaces = [rootPath];
	const seen = new Set([rootPath]);
	for (const pattern of workspaces) {
		const dir = pattern.endsWith("/package.json") ? dirname(pattern) : pattern;
		if (!/[*?[{]/.test(dir)) {
			const pkgPath = join(root, dir, "package.json");
			if (existsSync(pkgPath) && !seen.has(pkgPath)) {
				seen.add(pkgPath);
				surfaces.push(pkgPath);
			}
			continue;
		}
		// Glob member: expand one level (packages/*) without a glob dependency.
		const [head, literal] = dir.split("/*", 2);
		if (head === undefined || literal === undefined) {
			throw new ExposureError(`unsupported workspace glob "${pattern}" in ${rootPath}`);
		}
		const baseDir = join(root, head);
		if (!existsSync(baseDir)) continue;
		for (const entry of readdirSync(baseDir, { withFileTypes: true }).sort((a, b) => (a.name < b.name ? -1 : 1))) {
			if (!entry.isDirectory()) continue;
			const pkgPath = join(baseDir, entry.name, literal.replace(/^\//, ""), "package.json");
			if (existsSync(pkgPath) && !seen.has(pkgPath)) {
				seen.add(pkgPath);
				surfaces.push(pkgPath);
			}
		}
	}
	return surfaces.sort();
}

export function parseNpmSurface(path: string, root: string): NpmSurface {
	const text = readFileSync(path, "utf8");
	const json = parseJson(text, path);
	const depFields: Record<string, readonly string[]> = {};
	for (const field of DEP_FIELDS) {
		const value = json[field];
		if (value === undefined) {
			depFields[field] = [];
			continue;
		}
		const record = asRecord(value, `${path} field "${field}"`);
		depFields[field] = Object.keys(record).sort();
	}
	const nameValue = json["name"];
	const scriptsValue = json["scripts"];
	const scripts: Record<string, string> = {};
	if (scriptsValue !== undefined) {
		for (const [key, value] of Object.entries(asRecord(scriptsValue, `${path} field "scripts"`))) {
			if (typeof value !== "string") {
				throw new ExposureError(`${path}: scripts["${key}"] must be a string`);
			}
			scripts[key] = value;
		}
	}
	return {
		path,
		// Posix separators: the checked-in bundle pins `/` paths, and
		// downstream uses (surface-set equality, dirname splits) assume
		// them. Node's relative() yields `\` on Windows (proven: surface
		// set compared packages\extension-host\... against packages/...).
		relPath: relative(root, path).replace(/\\/g, "/"),
		sha256: sha256Text(text),
		depFields,
		packageName: typeof nameValue === "string" ? nameValue : undefined,
		scripts,
	};
}

export function loadNpmSurfaces(root: string): NpmSurface[] {
	return discoverSurfacePaths(root).map((path) => parseNpmSurface(path, root));
}

// ---------------------------------------------------------------------------
// Reference manifest (hash-chained, immutable boundary)
// ---------------------------------------------------------------------------

export interface ReferenceFilePin {
	readonly path: string;
	readonly sha256: string;
}

export interface ReferenceNpmSurface extends ReferenceFilePin {
	readonly packageName: string | undefined;
	readonly depFields: Readonly<Record<string, readonly string[]>>;
}

export interface ReferenceManifest {
	readonly schema: typeof REFERENCE_SCHEMA;
	readonly capturedAt: string;
	readonly captureHead: string | undefined;
	/** Porcelain output recorded when capture ran with --allow-dirty-relevant. */
	readonly relevantTreeStatus: string | undefined;
	readonly metafile: {
		readonly projectionPath: string;
		readonly sha256: string;
		readonly entry: string;
		readonly hostDirRel: string;
		readonly argv: readonly string[];
		readonly metafileSha256: string;
	};
	readonly cargo: {
		readonly projectionPath: string;
		readonly sha256: string;
		readonly argv: readonly string[];
	};
	readonly npmSurfaces: readonly ReferenceNpmSurface[];
	readonly cargoFiles: readonly ReferenceFilePin[];
	readonly authority: readonly ReferenceFilePin[];
}

export function parseReferenceManifest(text: string, what: string): ReferenceManifest {
	const json = parseJson(text, what);
	if (json["schema"] !== REFERENCE_SCHEMA) {
		throw new ExposureError(`${what}: unsupported schema ${String(json["schema"])}`);
	}
	const metafile = asRecord(json["metafile"], `${what} metafile`);
	const cargo = asRecord(json["cargo"], `${what} cargo`);
	const surfacesRaw = json["npmSurfaces"];
	if (!Array.isArray(surfacesRaw)) throw new ExposureError(`${what}: npmSurfaces must be an array`);
	const cargoFilesRaw = json["cargoFiles"];
	if (!Array.isArray(cargoFilesRaw)) throw new ExposureError(`${what}: cargoFiles must be an array`);
	const authorityRaw = json["authority"];
	if (!Array.isArray(authorityRaw)) throw new ExposureError(`${what}: authority must be an array`);
	const npmSurfaces: ReferenceNpmSurface[] = surfacesRaw.map((entry, index) => {
		const record = asRecord(entry, `${what} npmSurfaces[${index}]`);
		const depFieldsValue = record["depFields"];
		const depFields = asRecord(depFieldsValue, `${what} npmSurfaces[${index}] depFields`);
		const parsed: Record<string, readonly string[]> = {};
		for (const [field, names] of Object.entries(depFields)) {
			if (!Array.isArray(names) || !names.every((name): name is string => typeof name === "string")) {
				throw new ExposureError(`${what}: depFields.${field} must be an array of strings`);
			}
			parsed[field] = names;
		}
		return {
			path: reqString(record, "path", what),
			sha256: reqSha256(record, "sha256", what),
			packageName: optString(record, "packageName"),
			depFields: parsed,
		};
	});
	const pin = (entry: unknown, label: string): ReferenceFilePin => {
		const record = asRecord(entry, `${what} ${label}`);
		return { path: reqString(record, "path", what), sha256: reqSha256(record, "sha256", what) };
	};
	const headValue = optString(json, "captureHead");
	return {
		schema: REFERENCE_SCHEMA,
		capturedAt: reqString(json, "capturedAt", what),
		captureHead: headValue,
		relevantTreeStatus: optString(json, "relevantTreeStatus"),
		metafile: {
			projectionPath: reqString(metafile, "projectionPath", what),
			sha256: reqSha256(metafile, "sha256", what),
			entry: reqString(metafile, "entry", what),
			hostDirRel: reqString(metafile, "hostDirRel", what),
			argv: reqStringArray(metafile, "argv", what),
			metafileSha256: reqSha256(metafile, "metafileSha256", what),
		},
		cargo: {
			projectionPath: reqString(cargo, "projectionPath", what),
			sha256: reqSha256(cargo, "sha256", what),
			argv: reqStringArray(cargo, "argv", what),
		},
		npmSurfaces,
		cargoFiles: cargoFilesRaw.map((entry, index) => pin(entry, `cargoFiles[${index}]`)),
		authority: authorityRaw.map((entry, index) => pin(entry, `authority[${index}]`)),
	};
}

export interface MetafileProjection {
	readonly schema: typeof METAFILE_PROJECTION_SCHEMA;
	readonly entry: string;
	readonly hostDirRel: string;
	readonly argv: readonly string[];
	readonly metafileSha256: string;
	/** metafile input path (relative to the host dir) -> sha256 of its bytes. */
	readonly inputs: Readonly<Record<string, string>>;
}

export function parseMetafileProjection(text: string, what: string): MetafileProjection {
	const json = parseJson(text, what);
	if (json["schema"] !== METAFILE_PROJECTION_SCHEMA) {
		throw new ExposureError(`${what}: unsupported schema ${String(json["schema"])}`);
	}
	const inputsRaw = json["inputs"];
	const inputsRecord = asRecord(inputsRaw, `${what} inputs`);
	const inputs: Record<string, string> = {};
	for (const [path, hash] of Object.entries(inputsRecord)) {
		if (typeof hash !== "string" || !SHA256_RE.test(hash)) {
			throw new ExposureError(`${what}: inputs["${path}"] must be a sha256 digest`);
		}
		inputs[path] = hash;
	}
	return {
		schema: METAFILE_PROJECTION_SCHEMA,
		entry: reqString(json, "entry", what),
		hostDirRel: reqString(json, "hostDirRel", what),
		argv: reqStringArray(json, "argv", what),
		metafileSha256: reqSha256(json, "metafileSha256", what),
		inputs,
	};
}

export interface CargoEdge {
	readonly from: string;
	readonly to: string;
	readonly kinds: readonly string[];
}

export interface CargoGraphProjection {
	readonly schema: typeof CARGO_GRAPH_PROJECTION_SCHEMA;
	readonly argv: readonly string[];
	readonly workspaceMembers: readonly string[];
	readonly edges: readonly CargoEdge[];
}

export function parseCargoGraphProjection(text: string, what: string): CargoGraphProjection {
	const json = parseJson(text, what);
	if (json["schema"] !== CARGO_GRAPH_PROJECTION_SCHEMA) {
		throw new ExposureError(`${what}: unsupported schema ${String(json["schema"])}`);
	}
	const edgesRaw = json["edges"];
	if (!Array.isArray(edgesRaw)) throw new ExposureError(`${what}: edges must be an array`);
	const edges: CargoEdge[] = edgesRaw.map((entry, index) => {
		const record = asRecord(entry, `${what} edges[${index}]`);
		return {
			from: reqString(record, "from", what),
			to: reqString(record, "to", what),
			kinds: reqStringArray(record, "kinds", what),
		};
	});
	for (const edge of edges) {
		for (const kind of edge.kinds) {
			if (kind !== "normal" && kind !== "build" && kind !== "dev") {
				throw new ExposureError(`${what}: edge ${edge.from}->${edge.to} has unknown kind "${kind}"`);
			}
		}
	}
	return {
		schema: CARGO_GRAPH_PROJECTION_SCHEMA,
		argv: reqStringArray(json, "argv", what),
		workspaceMembers: reqStringArray(json, "workspaceMembers", what),
		edges,
	};
}

/** Project raw `cargo metadata` JSON into the graph shape this checker reasons over. */
export function projectCargoMetadata(rawText: string, argv: readonly string[], what: string): CargoGraphProjection {
	const json = parseJson(rawText, what);
	const packagesRaw = json["packages"];
	if (!Array.isArray(packagesRaw)) throw new ExposureError(`${what}: packages must be an array`);
	const nameById = new Map<string, string>();
	const members: string[] = [];
	const workspaceRoot = optString(json, "workspace_root");
	for (const entry of packagesRaw) {
		const record = asRecord(entry, `${what} packages[]`);
		const id = reqString(record, "id", what);
		const name = reqString(record, "name", what);
		nameById.set(id, name);
		const manifestPath = optString(record, "manifest_path");
		if (workspaceRoot !== undefined && manifestPath !== undefined && manifestPath.startsWith(workspaceRoot)) {
			members.push(name);
		}
	}
	const resolveRaw = json["resolve"];
	const resolve = asRecord(resolveRaw, `${what} resolve`);
	const nodesRaw = resolve["nodes"];
	if (!Array.isArray(nodesRaw)) throw new ExposureError(`${what}: resolve.nodes must be an array`);
	const edges: CargoEdge[] = [];
	for (const nodeEntry of nodesRaw) {
		const node = asRecord(nodeEntry, `${what} resolve.nodes[]`);
		const fromId = reqString(node, "id", what);
		const from = nameById.get(fromId);
		if (from === undefined) throw new ExposureError(`${what}: unknown package id "${fromId}"`);
		const depsRaw = node["deps"];
		if (!Array.isArray(depsRaw)) throw new ExposureError(`${what}: resolve node deps must be an array`);
		for (const depEntry of depsRaw) {
			const dep = asRecord(depEntry, `${what} resolve node dep`);
			const toId = reqString(dep, "pkg", what);
			const to = nameById.get(toId);
			if (to === undefined) throw new ExposureError(`${what}: unknown package id "${toId}"`);
			const kindsRaw = dep["dep_kinds"];
			const kinds = new Set<string>();
			if (!Array.isArray(kindsRaw) || kindsRaw.length === 0) {
				kinds.add("normal");
			} else {
				for (const kindEntry of kindsRaw) {
					const kindRecord = asRecord(kindEntry, `${what} dep_kinds[]`);
					const kind = optString(kindRecord, "kind");
					kinds.add(kind === undefined ? "normal" : kind);
				}
			}
			edges.push({ from, to, kinds: [...kinds].sort() });
		}
	}
	return {
		schema: CARGO_GRAPH_PROJECTION_SCHEMA,
		argv: [...argv],
		workspaceMembers: [...new Set(members)].sort(),
		edges,
	};
}

// ---------------------------------------------------------------------------
// Reference bundle: load + verify the hash chain end to end
// ---------------------------------------------------------------------------

export interface ReferenceBundle {
	readonly dir: string;
	readonly manifest: ReferenceManifest;
	readonly metafile: MetafileProjection;
	readonly cargoGraph: CargoGraphProjection;
}

export function loadReferenceBundle(dir: string): ReferenceBundle {
	const manifestPath = join(dir, "reference.json");
	const manifest = parseReferenceManifest(readFileSync(manifestPath, "utf8"), manifestPath);
	const metafileText = readFileSync(join(dir, manifest.metafile.projectionPath), "utf8");
	if (sha256Text(metafileText) !== manifest.metafile.sha256) {
		throw new ExposureError(
			`reference hash-chain broken: ${manifest.metafile.projectionPath} does not match reference.json`,
		);
	}
	const cargoText = readFileSync(join(dir, manifest.cargo.projectionPath), "utf8");
	if (sha256Text(cargoText) !== manifest.cargo.sha256) {
		throw new ExposureError(
			`reference hash-chain broken: ${manifest.cargo.projectionPath} does not match reference.json`,
		);
	}
	return {
		dir,
		manifest,
		metafile: parseMetafileProjection(metafileText, manifest.metafile.projectionPath),
		cargoGraph: parseCargoGraphProjection(cargoText, manifest.cargo.projectionPath),
	};
}

/** Byte-compare the release authority modules against the reference pins (before any import). */
export function checkAuthorityIntegrity(root: string, manifest: ReferenceManifest): CheckResult {
	try {
		for (const pin of manifest.authority) {
			const current = sha256FileAt(join(root, pin.path));
			if (current !== pin.sha256) {
				return undecidable(
					`authority module ${pin.path} drifted from the reference capture (${current} != ${pin.sha256}); refusing to import`,
				);
			}
		}
	} catch (error) {
		return undecidable(`authority integrity: ${errorText(error)}`);
	}
	return pass(`${manifest.authority.length} authority modules byte-identical to the reference`);
}

/** Re-hash every metafile input; any drift means the pre-change graph no longer describes the tree. */
export function checkMetafileInputIntegrity(
	root: string,
	projection: MetafileProjection,
): CheckResult {
	const hostDir = resolve(root, projection.hostDirRel);
	let checked = 0;
	try {
		for (const [inputPath, expected] of Object.entries(projection.inputs)) {
			const current = sha256FileAt(resolve(hostDir, inputPath));
			if (current !== expected) {
				return undecidable(
					`metafile input ${inputPath} drifted from the reference capture (${current} != ${expected}); pre-change module graph is stale`,
				);
			}
			checked += 1;
		}
	} catch (error) {
		return undecidable(`metafile input integrity: ${errorText(error)}`);
	}
	return pass(`${checked} metafile inputs byte-identical to the reference`);
}

/** Verify a list of file pins against the current tree. Returns drifted paths. */
export function driftedPins(root: string, pins: readonly ReferenceFilePin[]): string[] {
	const drifted: string[] = [];
	for (const pin of pins) {
		const current = sha256FileAt(join(root, pin.path));
		if (current !== pin.sha256) drifted.push(pin.path);
	}
	return drifted;
}

// ---------------------------------------------------------------------------
// E1: field / edge position
// ---------------------------------------------------------------------------

/** npm side: the subject must be devDependencies-only across every surface, before AND after the change. */
export function e1Npm(
	subject: Subject,
	preSurfaces: readonly ReferenceNpmSurface[],
	postSurfaces: readonly NpmSurface[],
): CheckResult {
	try {
		const prePaths = preSurfaces.map((surface) => surface.path).sort().join(",");
		const postPaths = postSurfaces.map((surface) => surface.relPath).sort().join(",");
		if (prePaths !== postPaths) {
			return undecidable(
				`package.json surface set changed ([${prePaths}] -> [${postPaths}]); re-capture the reference`,
			);
		}
		const hits: string[] = [];
		for (const surface of preSurfaces) {
			for (const field of NON_DEV_FIELDS) {
				if ((surface.depFields[field] ?? []).includes(subject.name)) {
					hits.push(`pre ${surface.path} ${field}`);
				}
			}
		}
		for (const surface of postSurfaces) {
			for (const field of NON_DEV_FIELDS) {
				if ((surface.depFields[field] ?? []).includes(subject.name)) {
					hits.push(`post ${surface.relPath} ${field}`);
				}
			}
		}
		if (hits.length > 0) {
			return fail(`non-dev field position: ${hits.join("; ")}`);
		}
		return pass(`devDependencies-only across all ${postSurfaces.length} package.json surfaces (pre and post)`);
	} catch (error) {
		return undecidable(`E1 npm: ${errorText(error)}`);
	}
}

function subjectEdges(graph: CargoGraphProjection, name: string): CargoEdge[] {
	return graph.edges.filter((edge) => edge.to === name);
}

/**
 * Rust side: dep-graph position only — the subject's edges in BOTH the pre
 * and post `cargo metadata` graphs must be `kind = "dev"`. Manifest text is
 * never consulted.
 */
export function e1Cargo(
	subject: Subject,
	preGraph: CargoGraphProjection | undefined,
	postGraph: CargoGraphProjection | undefined,
): CheckResult {
	try {
		if (preGraph === undefined) return undecidable("pre-change cargo graph projection missing");
		if (postGraph === undefined) return undecidable("post-change cargo graph unavailable");
		const offenders: string[] = [];
		for (const [label, graph] of [
			["pre", preGraph],
			["post", postGraph],
		] as const) {
			for (const edge of subjectEdges(graph, subject.name)) {
				if (edge.kinds.some((kind) => kind !== "dev")) {
					offenders.push(`${label}: ${edge.from} -> ${subject.name} kinds=[${edge.kinds.join(",")}]`);
				}
			}
		}
		if (offenders.length > 0) {
			return fail(`non-dev graph edges: ${offenders.join("; ")}`);
		}
		return pass(
			`no non-dev edges into ${subject.name} in either cargo metadata graph (all-features, locked)`,
		);
	} catch (error) {
		return undecidable(`E1 cargo: ${errorText(error)}`);
	}
}

// ---------------------------------------------------------------------------
// E2: zero bundler-metafile reachability on every --compile entry
// ---------------------------------------------------------------------------

/**
 * Resolve the owning npm package of one metafile input path using npm
 * resolution semantics (innermost `node_modules/<pkg>` wins). Repo-local
 * source inputs own nothing (null).
 */
export function resolveInputOwner(inputPath: string): string | null {
	const segments = inputPath.split("/");
	for (let i = segments.length - 2; i >= 0; i -= 1) {
		if (segments[i] !== "node_modules") continue;
		const next = segments[i + 1];
		if (next === undefined) return null;
		if (next.startsWith("@")) {
			const scope = segments[i + 2];
			return scope === undefined ? null : `${next}/${scope}`;
		}
		return next;
	}
	return null;
}

/** Repo-local (workspace) ownership: an input under a surface's package dir. */
export function workspaceOwner(
	inputPath: string,
	surfaces: readonly { relPath: string; packageName: string | undefined }[],
): string | null {
	// Metafile inputs are host-dir-relative ("../pi-tui-protocol/src/x.ts");
	// normalize against the host dir before matching a surface prefix.
	for (const surface of surfaces) {
		if (surface.packageName === undefined) continue;
		const dirSegments = dirname(surface.relPath).split("/");
		const last = dirSegments[dirSegments.length - 1];
		if (last === undefined || last === ".") continue;
		const repoRel = repoRelativeFromHost(inputPath);
		if (repoRel === dirSegments.join("/") || repoRel.startsWith(`${dirSegments.join("/")}/`)) {
			return surface.packageName;
		}
	}
	return null;
}

/** Normalize a host-dir-relative metafile input into a repo-relative path. */
export function repoRelativeFromHost(inputPath: string): string {
	return posix.normalize(posix.join(HOST_PACKAGE_DIR, inputPath));
}

export function e2Reachability(
	subject: Subject,
	metafileInputs: readonly string[],
	surfaces: readonly { relPath: string; packageName: string | undefined }[],
): CheckResult {
	if (subject.kind === "crate") {
		return pass("no bundler surface: crate subjects do not appear in the extension-host metafile");
	}
	if (subject.kind === "tool") {
		if (subject.name === "rust-toolchain") {
			return pass("rust toolchain does not participate in the bundler metafile (see E3)");
		}
		return fail(
			`${subject.name} version bump changes the compiled sidecar bytes (bundler compiles / runtime is embedded via --compile)`,
		);
	}
	const hits: string[] = [];
	let total = 0;
	for (const input of metafileInputs) {
		const owner = resolveInputOwner(input) ?? workspaceOwner(input, surfaces);
		if (owner === null) continue;
		total += 1;
		if (owner === subject.name && hits.length < 3) hits.push(input);
	}
	if (hits.length > 0) {
		return fail(`bundled into the shipped sidecar: ${hits.join(", ")}${total > 3 ? ", …" : ""}`);
	}
	return pass(`zero metafile reachability: ${subject.name} owns none of the ${metafileInputs.length} bundled inputs`);
}

// ---------------------------------------------------------------------------
// --compile entry enumeration (authority + package.json scripts + CI)
// ---------------------------------------------------------------------------

export interface CompileEntry {
	readonly origin: string;
	readonly argv: readonly string[];
}

/**
 * Authority module shapes. These modules are loaded through dynamic
 * `import(file://…)` ONLY after their bytes match the reference pins (see
 * checkAuthorityIntegrity) — a static import would execute drifted release
 * authority, which is exactly what the pin check exists to prevent.
 */
interface HostAuthority {
	readonly hostBundleCommands: typeof hostBundleCommands;
}
interface TargetsAuthority {
	readonly planFor: typeof planFor;
	readonly TARGET_PLANS: typeof TARGET_PLANS;
}
interface StageAuthority {
	readonly stagedInputs: typeof stagedInputs;
}

/** Tokens whose values are entry-independent (target/output/metafile locations). */
const VALUE_FLAGS = new Set(["--target", "--outfile", "--metafile"]);
export function normalizeCompileArgv(argv: readonly string[]): { entry: string; flags: string[] } {
	const tokens = argv.filter((token) => token !== "bun" && token !== "build");
	let entry = "";
	const flags: string[] = [];
	for (let i = 0; i < tokens.length; i += 1) {
		const token = tokens[i];
		if (token === undefined) continue;
		if (token.startsWith("--")) {
			const [flag] = token.split("=", 2);
			if (flag === undefined) continue;
			if (VALUE_FLAGS.has(flag)) {
				if (!token.includes("=")) i += 1; // skip the value token
				flags.push(flag);
				continue;
			}
			flags.push(token);
			continue;
		}
		if (/\.(?:ts|tsx|mts|cts)$/.test(token)) {
			entry = token;
			continue;
		}
		flags.push(token);
	}
	return { entry, flags: [...new Set(flags)] };
}

/**
 * A --compile entry conforms to the authority when it compiles the same
 * entrypoint using only flags the authority knows. Local variants may omit
 * flags (e.g. --target, whose value only selects the codegen platform); any
 * flag outside the authority vocabulary (authority flags ∪ value-flags)
 * changes the bundling contract and is a divergence (undecidable).
 */
export function compileEntryConforms(
	entry: { entry: string; flags: string[] },
	authority: { entry: string; flags: string[] },
): boolean {
	if (entry.entry !== authority.entry) return false;
	return entry.flags.every((flag) => authority.flags.includes(flag) || VALUE_FLAGS.has(flag));
}

/** Enumerate every --compile entry in the tree: authority argv, package.json build scripts, CI workflows. */
export async function enumerateCompileEntries(
	root: string,
	authorityOk: CheckResult,
	surfaces: readonly NpmSurface[],
): Promise<{ entries: CompileEntry[]; authority: readonly string[]; problem: string | undefined }> {
	if (authorityOk.status !== "pass") {
		return { entries: [], authority: [], problem: `authority integrity: ${authorityOk.detail}` };
	}
	try {
		// Dynamic by design: authority modules load only after byte-verification
		// against the reference pin — a static import would execute drifted code.
		const hostUrl = new URL(`file://${join(root, "scripts/release/host.ts")}`).href;
		const hostModule = (await import(hostUrl)) as HostAuthority;
		const targetsUrl = new URL(`file://${join(root, "scripts/release/targets.ts")}`).href;
		const targetsModule = (await import(targetsUrl)) as TargetsAuthority;
		const triple = localRustTriple();
		if (triple === undefined) throw new ExposureError("cannot map the local platform to a release triple");
		const plan = targetsModule.planFor(triple);
		const commands = hostModule.hostBundleCommands(plan, "/tmp/de-out");
		const authority = [...commands.compiled];
		const entries: CompileEntry[] = [
			{ origin: "scripts/release/host.ts hostBundleCommands().compiled", argv: authority },
		];
		for (const surface of surfaces) {
			for (const [scriptName, script] of Object.entries(surface.scripts)) {
				if (script.includes("--compile")) {
					entries.push({ origin: `${surface.relPath} scripts.${scriptName}`, argv: script.split(/\s+/) });
				}
			}
		}
		for (const line of workflowBuildLines(root)) {
			if (line.text.includes("--compile")) {
				entries.push({ origin: `${line.file}:${line.line}`, argv: line.text.trim().split(/\s+/) });
			}
		}
		return { entries, authority, problem: undefined };
	} catch (error) {
		return { entries: [], authority: [], problem: `compile-entry enumeration: ${errorText(error)}` };
	}
}

export function localRustTriple(): string | undefined {
	if (process.platform === "linux" && process.arch === "x64") return "x86_64-unknown-linux-gnu";
	if (process.platform === "linux" && process.arch === "arm64") return "aarch64-unknown-linux-gnu";
	if (process.platform === "darwin" && process.arch === "x64") return "x86_64-apple-darwin";
	if (process.platform === "darwin" && process.arch === "arm64") return "aarch64-apple-darwin";
	if (process.platform === "win32" && process.arch === "x64") return "x86_64-pc-windows-msvc";
	return undefined;
}

/** Every `bun build` / `cargo build` statement in the workflow files (line continuations joined). */
export function workflowBuildLines(root: string): { file: string; line: number; text: string }[] {
	const workflowsDir = join(root, ".github/workflows");
	const results: { file: string; line: number; text: string }[] = [];
	if (!existsSync(workflowsDir)) return results;
	for (const name of readdirSync(workflowsDir).filter((n) => n.endsWith(".yml") || n.endsWith(".yaml")).sort()) {
		const file = join(workflowsDir, name);
		const raw = readFileSync(file, "utf8");
		const physical = raw.split("\n");
		let merged = "";
		let startLine = 0;
		for (let i = 0; i < physical.length; i += 1) {
			const text = physical[i];
			if (text === undefined) continue;
			if (merged === "") startLine = i + 1;
			if (text.trimEnd().endsWith("\\")) {
				merged += `${text.trimEnd().slice(0, -1)} `;
				continue;
			}
			const statement = merged + text;
			merged = "";
			if (/\b(?:bun|cargo)\s+build\b/.test(statement)) {
				results.push({ file: relative(root, file), line: startLine, text: statement.trim() });
			}
		}
	}
	return results;
}

// ---------------------------------------------------------------------------
// E3: no shipped-byte production (CommandRunner.run seam + CI invocations)
// ---------------------------------------------------------------------------

export interface SeamSite {
	readonly file: string;
	readonly line: number;
	readonly command: string | undefined;
	readonly literalArgs: readonly string[];
	readonly spreadNames: readonly string[];
	readonly unresolved: boolean;
}

/** Balanced-paren argument extraction for one `.run(` call site. */
function extractCall(text: string, openIdx: number): string {
	let depth = 0;
	let inString: string | undefined;
	for (let i = openIdx; i < text.length; i += 1) {
		const ch = text[i];
		if (inString !== undefined) {
			if (ch === inString) inString = undefined;
			continue;
		}
		if (ch === '"' || ch === "'" || ch === "`") {
			inString = ch;
			continue;
		}
		if (ch === "(") depth += 1;
		if (ch === ")") {
			depth -= 1;
			if (depth === 0) return text.slice(openIdx + 1, i);
		}
	}
	throw new ExposureError(`unbalanced .run( call starting at offset ${openIdx}`);
}

/** Parse one `.run(cmd, [args])` argument string into a seam site description. */
export function parseSeamCall(argsText: string, file: string, line: number): SeamSite {
	const topLevel = splitTopLevel(argsText);
	const first = topLevel[0]?.trim() ?? "";
	let command: string | undefined;
	let unresolvedCommand = false;
	const firstLit = staticStringLiteral(first);
	if (firstLit !== undefined) {
		command = firstLit;
	} else {
		unresolvedCommand = true;
	}
	const literalArgs: string[] = [];
	const spreadNames: string[] = [];
	let unresolved = false;
	const second = topLevel[1]?.trim() ?? "";
	if (second.startsWith("[")) {
		for (const element of splitTopLevel(second.slice(1, -1))) {
			const token = element.trim();
			if (token.length === 0) continue;
			const lit = staticStringLiteral(token);
			if (lit !== undefined) {
				literalArgs.push(lit);
			} else if (token.startsWith("...") && /^[A-Za-z_$][A-Za-z0-9_$]*$/.test(token.slice(3))) {
				spreadNames.push(token.slice(3));
			} else {
				unresolved = true;
			}
		}
	} else if (second.length > 0) {
		unresolved = true;
	}
	return {
		file,
		line,
		command: unresolvedCommand ? undefined : command,
		literalArgs,
		spreadNames,
		unresolved: unresolved || unresolvedCommand,
	};
}

/**
 * Extract the content of a string literal token (single/double quoted, or a
 * template literal with no `${…}` interpolation). Interpolated templates and
 * non-string tokens return `undefined` (unattributable). Static template
 * literals must be attributed like any other literal: otherwise a
 * `runner.run("bun", [`build`, `--compile`, …])` site hides its emit intent
 * from the E3 scanner and fails open.
 */
function staticStringLiteral(token: string): string | undefined {
	if (token.length < 2) return undefined;
	const quote = token[0];
	if (quote !== '"' && quote !== "'" && quote !== "`") return undefined;
	if (token[token.length - 1] !== quote) return undefined;
	const inner = token.slice(1, -1);
	if (quote === "`" && inner.includes("${")) return undefined;
	return inner;
}

function splitTopLevel(text: string): string[] {
	const parts: string[] = [];
	let depth = 0;
	let inString: string | undefined;
	let current = "";
	for (const ch of text) {
		if (inString !== undefined) {
			current += ch;
			if (ch === inString) inString = undefined;
			continue;
		}
		if (ch === '"' || ch === "'" || ch === "`") {
			inString = ch;
			current += ch;
			continue;
		}
		if (ch === "(" || ch === "[" || ch === "{") depth += 1;
		if (ch === ")" || ch === "]" || ch === "}") depth -= 1;
		if (ch === "," && depth === 0) {
			parts.push(current);
			current = "";
			continue;
		}
		current += ch;
	}
	parts.push(current);
	return parts;
}

/** Authority spread names accepted when attributing seam argvs. */
const AUTHORITY_SPREADS = new Set(["compiled", "runtimeBundle"]);

/**
 * E3 scan: every `.run(` seam site in the release scripts must be
 * attributable — literal argv, authority spread, or a non-build execution of
 * already-produced artifacts. Build-capable sites with unattributable
 * arguments are undecidable (fail closed to Class S).
 */
export function scanSeamSites(sources: Readonly<Record<string, string>>): { sites: SeamSite[]; problems: string[] } {
	const sites: SeamSite[] = [];
	const problems: string[] = [];
	for (const [relPath, text] of Object.entries(sources)) {
		const lines = text.split("\n");
		for (let lineIdx = 0; lineIdx < lines.length; lineIdx += 1) {
			const lineText = lines[lineIdx];
			if (lineText === undefined || !lineText.includes(".run(")) continue;
			const offsetInLine = lineText.indexOf(".run(");
			const globalOffset =
				lines.slice(0, lineIdx).reduce((sum, l) => sum + l.length + 1, 0) + offsetInLine;
			try {
				const argsText = extractCall(text, globalOffset + ".run(".length - 1);
				sites.push(parseSeamCall(argsText, relPath, lineIdx + 1));
			} catch (error) {
				problems.push(`${relPath}:${lineIdx + 1} ${errorText(error)}`);
			}
		}
	}
	for (const site of sites) {
		if (site.command !== undefined && BUILD_CAPABLE_COMMANDS.has(site.command)) {
			const spreadOk = site.spreadNames.every((name) => AUTHORITY_SPREADS.has(name));
			if (site.unresolved || !spreadOk) {
				// Only module-graph commands (bun/tsc/npm/…) can pull an npm
				// subject's bytes into shipped output through their argv; a
				// cargo argv with dynamic segments (target triples, package
				// selectors) cannot, because Rust linkage is decided by the
				// cargo metadata graph in E1/E3, never by argv text.
				if (site.command !== "cargo" && hasEmitIntent([...site.literalArgs, ...site.spreadNames])) {
					problems.push(
						`${site.file}:${site.line} build-capable .run("${site.command}", …) has unattributable arguments [${[...site.literalArgs, ...site.spreadNames].join(", ")}]`,
					);
				}
			}
		}
	}
	return { sites, problems };
}

function hasEmitIntent(tokens: readonly string[]): boolean {
	return tokens.some((token) => EMIT_TOKENS.has(token));
}

/** Is a seam/workflow site shipped-byte producing? */
export function siteProducesShippedBytes(site: { command?: string; literalArgs: readonly string[] }): boolean {
	if (site.command === "cargo") return site.literalArgs.includes("build");
	if (site.command === "bun") return site.literalArgs.includes("build");
	if (site.command === "tsc" || site.command === "npm" || site.command === "npx") return true;
	return false;
}

// ---------------------------------------------------------------------------
// E4: no archive staging (staged-input table from the assembly script source)
// ---------------------------------------------------------------------------

export interface StagedRow {
	readonly kind: string;
	readonly source: string;
	readonly destRel: string;
	readonly hostKind: string;
}

/** Import the byte-verified stage.ts and enumerate the staged-input table for BOTH host kinds. */
export async function enumerateStagedInputs(
	root: string,
	authorityOk: CheckResult,
): Promise<{ rows: StagedRow[]; problem: string | undefined }> {
	if (authorityOk.status !== "pass") {
		return { rows: [], problem: `authority integrity: ${authorityOk.detail}` };
	}
	try {
		// Dynamic by design: byte-verified release authority (see HostAuthority).
		const stageUrl = new URL(`file://${join(root, "scripts/release/stage.ts")}`).href;
		const stageModule = (await import(stageUrl)) as StageAuthority;
		const targetsUrl = new URL(`file://${join(root, "scripts/release/targets.ts")}`).href;
		const targetsModule = (await import(targetsUrl)) as TargetsAuthority;
		const triple = localRustTriple();
		if (triple === undefined) throw new ExposureError("cannot map the local platform to a release triple");
		const plan = targetsModule.planFor(triple);
		// stagedInputs is pure table construction; the Fs seam it requires is
		// never invoked. Every method throws so an accidental call fails loud.
		const inertFs: Fs = {
			mkdir: async () => {},
			rm: async () => {},
			writeFile: async () => {},
			readFile: async () => {
				throw new ExposureError("inertFs.readFile must not be called");
			},
			copyFile: async () => {},
			cp: async () => {},
			chmod: async () => {},
			stat: async () => {
				throw new ExposureError("inertFs.stat must not be called");
			},
			readdir: async () => {
				throw new ExposureError("inertFs.readdir must not be called");
			},
		};
		const base = {
			fs: inertFs,
			plan,
			version: "0.0.0-checker",
			piBinaryPath: "/check/target/pi",
			repoRoot: root,
			bunRuntimePath: "/check/staging/bun",
			docsSource: "/check/docs",
			sourceDateEpoch: 0,
			compatibilityVersion: "0.0.0",
			protocolVersion: 1,
			createdAt: "1970-01-01T00:00:00Z",
		};
		const rows: StagedRow[] = [];
		const hosts = [
			{ kind: "compiled" as const, binaryPath: "/check/staging/pi-extension-host" },
			{
				kind: "runtime-bundle" as const,
				runtimePath: "/check/staging/bun",
				scriptPath: "/check/staging/pi-extension-host.js",
			},
		];
		for (const host of hosts) {
			for (const staged of stageModule.stagedInputs({ ...base, host })) {
				rows.push({ kind: staged.kind, source: staged.source, destRel: staged.destRel, hostKind: host.kind });
			}
		}
		return { rows, problem: undefined };
	} catch (error) {
		return { rows: [], problem: `staged-input enumeration: ${errorText(error)}` };
	}
}
export function npmStagingHit(subject: Subject, rows: readonly StagedRow[]): StagedRow | undefined {
	const needle = `node_modules/${subject.name}/`;
	return rows.find((row) => row.source.replaceAll("\\", "/").includes(needle));
}

const TOOL_PRODUCTS: Record<string, readonly string[]> = {
	"rust-toolchain": ["rust-binary"],
	"bun-bundler": ["host-binary", "host-bundle"],
	"bun-runtime": ["bun-runtime"],
};

export function e4Verdict(subject: Subject, rows: readonly StagedRow[], problem: string | undefined): CheckResult {
	if (problem !== undefined) return undecidable(problem);
	if (subject.kind === "tool") {
		const products = TOOL_PRODUCTS[subject.name] ?? [];
		const staged = rows.filter((row) => products.includes(row.kind));
		if (staged.length > 0) {
			return fail(
				`${subject.name} product staged into the archive: ${staged
					.map((row) => `${row.kind} -> ${row.destRel} (${row.hostKind})`)
					.join("; ")}`,
			);
		}
		return pass("no staged product");
	}
	if (subject.kind === "npm") {
		const hit = npmStagingHit(subject, rows);
		if (hit !== undefined) return fail(`staged from subject install path: ${hit.source} -> ${hit.destRel}`);
		return pass(`none of the ${rows.length} staged inputs source bytes from node_modules/${subject.name}`);
	}
	return pass(`no archive staging of crate bytes (rust-binary source is the linked pi build; see E3)`);
}

// ---------------------------------------------------------------------------
// Classify orchestration
// ---------------------------------------------------------------------------

export interface ClassifyOptions {
	readonly subject: Subject;
	readonly referenceDir: string;
	readonly root?: string;
	/** Hermetic override: path to raw `cargo metadata` JSON for the current tree. */
	readonly cargoMetadataFile?: string;
	/** Skip spawning cargo (npm/tool subjects never need it). */
	readonly spawnCargo?: boolean;
}

function readScriptSources(root: string): Record<string, string> {
	const sources: Record<string, string> = {};
	const dirs = [join(root, "scripts/release")];
	const files = [
		join(root, "scripts/package-release.ts"),
		join(root, "scripts/build-extension-host.ts"),
	];
	for (const dir of dirs) {
		for (const name of readdirSync(dir).filter((n) => n.endsWith(".ts")).sort()) {
			sources[relative(root, join(dir, name)).replace(/\\/g, "/")] = readFileSync(join(dir, name), "utf8");
		}
	}
	for (const file of files) {
		if (existsSync(file)) sources[relative(root, file).replace(/\\/g, "/")] = readFileSync(file, "utf8");
	}
	return sources;
}

async function currentCargoGraph(
	options: ClassifyOptions,
	referenceArgv: readonly string[],
): Promise<CargoGraphProjection | undefined> {
	const root = options.root ?? REPO_ROOT;
	if (options.cargoMetadataFile !== undefined) {
		return projectCargoMetadata(
			readFileSync(options.cargoMetadataFile, "utf8"),
			referenceArgv,
			options.cargoMetadataFile,
		);
	}
	if (options.spawnCargo === false) return undefined;
	const result = spawnSync("cargo", [...CARGO_METADATA_ARGV], {
		cwd: root,
		encoding: "utf8",
		timeout: 120_000,
		maxBuffer: 64 * 1024 * 1024,
		killSignal: "SIGKILL",
	});
	if (result.error !== undefined) {
		throw new ExposureError(`cargo metadata spawn failed: ${errorText(result.error)}`);
	}
	if (result.status !== 0) {
		throw new ExposureError(
			`cargo metadata ${CARGO_METADATA_ARGV.join(" ")} exited ${result.status}: ${(result.stderr ?? "").slice(0, 500)}`,
		);
	}
	return projectCargoMetadata(result.stdout ?? "", referenceArgv, "cargo metadata (spawned)");
}

async function guard(name: CheckName, run: () => Promise<CheckResult> | CheckResult): Promise<CheckResult> {
	try {
		return await run();
	} catch (error) {
		return undecidable(`${name}: ${errorText(error)}`);
	}
}

export async function classify(options: ClassifyOptions): Promise<ExposureReport> {
	const root = options.root ?? REPO_ROOT;
	// Fail-closed from the first byte: an unreadable/tampered reference
	// yields a decided Class S report, never an exemption-shaped crash.
	let bundle: ReferenceBundle;
	try {
		bundle = loadReferenceBundle(options.referenceDir);
	} catch (error) {
		const detail = `reference load: ${errorText(error)}`;
		return finalizeWithChecks({
			subject: options.subject,
			referenceDir: options.referenceDir,
			capturedAt: "unknown",
			checks: {
				E1: undecidable(detail),
				E2: undecidable(detail),
				E3: undecidable(detail),
				E4: undecidable(detail),
			},
		});
	}
	const manifest = bundle.manifest;


	const authority = checkAuthorityIntegrity(root, manifest);
	const postSurfaces = loadNpmSurfaces(root);

	// Cross-ecosystem byte-identity: an npm subject must not ride along with
	// Rust manifest drift, and vice versa. This is what makes a
	// Cargo.toml-only edge/feature change fail closed even when Cargo.lock
	// is untouched.
	const crossIdentity = await guard("E1", (): CheckResult => {
		if (options.subject.kind === "npm") {
			const drifted = driftedPins(root, manifest.cargoFiles);
			if (drifted.length > 0) {
				return undecidable(
					`npm subject but Rust inputs drifted since capture: ${drifted.join(", ")} (Cargo.toml-only changes fail closed)`,
				);
			}
		} else if (options.subject.kind === "crate") {
			const drifted = driftedPins(
				root,
				manifest.npmSurfaces.map((surface) => ({ path: surface.path, sha256: surface.sha256 })),
			);
			if (drifted.length > 0) {
				return undecidable(`crate subject but npm surfaces drifted since capture: ${drifted.join(", ")}`);
			}
		}
		return pass("cross-ecosystem inputs unchanged since reference capture");
	});

	// Post-change cargo graph: crate subjects read the current tree's graph
	// (spawned `cargo metadata --locked --offline --all-features`, or the
	// hermetic file override). A fetch failure is an undecidable E1/E3 —
	// never an exemption.
	let postGraph: CargoGraphProjection | undefined;
	let postGraphError: string | undefined;
	if (options.subject.kind === "crate") {
		try {
			postGraph = await currentCargoGraph(options, manifest.cargo.argv);
		} catch (error) {
			postGraphError = errorText(error);
		}
	}

	const e1 = await guard("E1", (): CheckResult => {
		if (crossIdentity.status !== "pass") return crossIdentity;
		if (options.subject.kind === "tool") {
			return pass("toolchain subject: field/edge position is not the deciding check (see E3/E4)");
		}
		if (options.subject.kind === "npm") {
			return e1Npm(options.subject, manifest.npmSurfaces, postSurfaces);
		}
		if (postGraphError !== undefined) {
			return undecidable(`post-change cargo graph unavailable: ${postGraphError}`);
		}
		return e1Cargo(options.subject, bundle.cargoGraph, postGraph);
	});

	const metafileIntegrity = checkMetafileInputIntegrity(root, bundle.metafile);
	const compileEntries = await enumerateCompileEntries(root, authority, postSurfaces);

	const e2 = await guard("E2", (): CheckResult => {
		if (compileEntries.problem !== undefined) return undecidable(compileEntries.problem);
		const authorityNorm = normalizeCompileArgv(compileEntries.authority);
		for (const entry of compileEntries.entries) {
			const norm = normalizeCompileArgv(entry.argv);
			if (!compileEntryConforms(norm, authorityNorm)) {
				return undecidable(
					`--compile entry at ${entry.origin} diverges from the authority argv (entry=${norm.entry}, flags=[${norm.flags.join(" ")}] vs authority [${authorityNorm.flags.join(" ")}])`,
				);
			}
		}
		if (metafileIntegrity.status !== "pass") return metafileIntegrity;
		return e2Reachability(options.subject, Object.keys(bundle.metafile.inputs), postSurfaces);
	});

	const scriptSources = readScriptSources(root);
	const seamScan = scanSeamSites(scriptSources);
	const workflowBuilds = workflowBuildLines(root);

	const e3 = await guard("E3", (): CheckResult => {
		if (options.subject.kind === "tool") {
			return fail(
				`${options.subject.name} produces shipped bytes by definition (pi binary / sidecar compile / embedded runtime)`,
			);
		}
		const problems = [...seamScan.problems];
		if (problems.length > 0) return undecidable(problems.join("; "));
		// Subject literal in any shipped-byte-producing invocation?
		for (const site of seamScan.sites) {
			if (!siteProducesShippedBytes({ command: site.command, literalArgs: site.literalArgs })) continue;
			if ([...site.literalArgs, ...site.spreadNames].some((token) => token.includes(options.subject.name))) {
				return fail(`named in shipped-byte-producing invocation at ${site.file}:${site.line}`);
			}
		}
		for (const build of workflowBuilds) {
			if (build.text.includes(options.subject.name)) {
				return fail(`named in CI build statement ${build.file}:${build.line}`);
			}
		}
		if (options.subject.kind === "crate") {
			if (postGraph === undefined) {
				return undecidable(
					`post-change cargo graph unavailable for linkage closure: ${postGraphError ?? "not fetched"}`,
				);
			}
			if (linkedIntoShippedBinary(postGraph, options.subject.name)) {
				return fail(`${options.subject.name} is linked into the shipped pi binary via non-dev edges`);
			}
			return pass("crate not linked into the shipped binary (dev-only in the post-change graph)");
		}
		return pass(
			`no shipped-byte-producing invocation names ${options.subject.name} (${seamScan.sites.length} seam sites, ${workflowBuilds.length} CI build statements attributed)`,
		);
	});

	const staged = await enumerateStagedInputs(root, authority);
	const e4 = e4Verdict(options.subject, staged.rows, staged.problem);

	return finalizeWithChecks({
		subject: options.subject,
		referenceDir: options.referenceDir,
		capturedAt: manifest.capturedAt,
		checks: { E1: e1, E2: e2, E3: e3, E4: e4 },
	});
}


export function linkedIntoShippedBinary(graph: CargoGraphProjection, crate: string): boolean {
	const adjacent = new Set<string>(graph.workspaceMembers);
	const nonDev = new Map<string, string[]>();
	for (const edge of graph.edges) {
		if (edge.kinds.some((kind) => kind !== "dev")) {
			nonDev.set(edge.from, [...(nonDev.get(edge.from) ?? []), edge.to]);
		}
	}
	const seen = new Set<string>();
	const queue = [...adjacent];
	while (queue.length > 0) {
		const current = queue.shift();
		if (current === undefined || seen.has(current)) continue;
		seen.add(current);
		if (current === crate) return true;
		for (const next of nonDev.get(current) ?? []) {
			if (!seen.has(next)) queue.push(next);
		}
	}
	return false;
}

function finalizeWithChecks(input: {
	subject: Subject;
	referenceDir: string;
	capturedAt: string;
	checks: Record<CheckName, CheckResult>;
}): ExposureReport {
	const verdict = verdictFromChecks(input.subject, input.checks);
	return {
		schema: SCHEMA,
		subject: input.subject.raw,
		verdict,
		checks: input.checks,
		referenceDir: input.referenceDir,
		capturedAt: input.capturedAt,
		sentinel: SENTINEL_OK,
	};
}

export interface CaptureOptions {
	/** Record (rather than refuse) a dirty relevant tree; provenance is stored in the manifest. */
	readonly allowDirtyRelevant?: boolean;
}

export function assertRelevantTreeClean(root: string, allowDirtyRelevant: boolean): string | undefined {
	const status = gitOutput(root, ["status", "--porcelain", "--", ...RELEVANT_PATHSPECS]);
	if (status.trim().length === 0) return undefined;
	if (!allowDirtyRelevant) {
		throw new ExposureError(
			`refusing to capture a reference from a dirty relevant tree:\n${status.trim()}\n(capture must run on the clean pre-change commit; pass --allow-dirty-relevant only when the dirt is provably dep-free and record why)`,
		);
	}
	return status.trim();
}

// ---------------------------------------------------------------------------
// capture-reference
// ---------------------------------------------------------------------------

const RELEVANT_PATHSPECS = [
	"package.json",
	"packages/extension-host/package.json",
	"packages/pi-tui-protocol/package.json",
	"packages/*/package.json",
	"bun.lock",
	"packages/extension-host/bun.lock",
	"Cargo.toml",
	"Cargo.lock",
	"crates/*/Cargo.toml",
	"scripts/release",
	"scripts/package-release.ts",
	"scripts/build-extension-host.ts",
	"scripts/verification/dependency-exposure.ts",
	"packages/extension-host/src",
	"packages/pi-tui-protocol/src",
] as const;

function gitOutput(root: string, args: readonly string[]): string {
	const result = spawnSync("git", args, { cwd: root, encoding: "utf8" });
	if (result.status !== 0 || result.stdout === null) {
		throw new ExposureError(`git ${args.join(" ")} failed: ${(result.stderr ?? "").slice(0, 300)}`);
	}
	return result.stdout;
}

function listCargoFiles(root: string): string[] {
	const paths = ["Cargo.toml", "Cargo.lock"];
	const cratesDir = join(root, "crates");
	for (const entry of readdirSync(cratesDir, { withFileTypes: true })) {
		if (entry.isDirectory()) {
			const manifest = `crates/${entry.name}/Cargo.toml`;
			if (existsSync(join(root, manifest))) paths.push(manifest);
		}
	}
	return paths.sort();
}

export async function captureReference(
	root: string,
	outDir: string,
	options?: CaptureOptions,
): Promise<ReferenceManifest> {
	const relevantTreeStatus = assertRelevantTreeClean(root, options?.allowDirtyRelevant === true);
	const hostDir = join(root, HOST_PACKAGE_DIR);
	mkdirSync(outDir, { recursive: true });
	// 1. Authority argv (trusted pre-change tree; byte pinning happens below).
	// Dynamic by design: capture runs on the clean pre-change tree, and these
	// modules are the same byte-pinned authorities classification loads.
	const hostUrl = new URL(`file://${join(root, "scripts/release/host.ts")}`).href;
	const hostModule = (await import(hostUrl)) as HostAuthority;
	const targetsUrl = new URL(`file://${join(root, "scripts/release/targets.ts")}`).href;
	const targetsModule = (await import(targetsUrl)) as TargetsAuthority;
	const triple = localRustTriple();
	if (triple === undefined) throw new ExposureError("cannot map the local platform to a release triple");
	const plan = targetsModule.planFor(triple);
	// Stage outside the repo tree: the staging paths are pinned into the
	// committed argv, so an in-repo staging dir would bake the author's
	// checkout path into the bundle and leave build residue in the tree.
	const staging = join(tmpdir(), "exposure-capture-staging");
	mkdirSync(staging, { recursive: true });
	const commands = hostModule.hostBundleCommands(plan, join(staging, plan.hostBinaryName));
	const argv = [...commands.compiled, `--metafile=${join(staging, "metafile.json")}`];

	// 2. Run the authority bun build once, on the trusted pre-change tree.
	const build = spawnSync("bun", argv, { cwd: hostDir, encoding: "utf8", timeout: 10 * 60_000 });
	if (build.status !== 0) {
		throw new ExposureError(`bun build (authority argv) failed: ${(build.stderr ?? "").slice(0, 800)}`);
	}
	const metafileText = readFileSync(join(staging, "metafile.json"), "utf8");
	const metafile = parseJson(metafileText, "metafile.json");
	const inputsRaw = metafile["inputs"];
	const inputsRecord = asRecord(inputsRaw, "metafile inputs");
	const inputs: Record<string, string> = {};
	for (const path of Object.keys(inputsRecord)) {
		// The provider-data manifest embeds its own generation timestamp
		// (generatedAt), so its digest changes on every data hydration and
		// can never reproduce a capture. Pinning it tests recency, not
		// integrity. The manifest's source files are pinned individually
		// below, so skipping it loses no coverage.
		if (path.endsWith("packages/ai/src/providers/data/.manifest.json")) continue;
		// The live provider catalogs beside it are hydrated from the vendor
		// APIs on every data hydration (the whole directory is generated and
		// gitignored upstream), so their digests drift with each hydration
		// and can never reproduce a capture either. They carry model-list
		// data, not staged code, so skipping them loses no exposure signal.
		if (path.includes("packages/ai/src/providers/data/") && path.endsWith(".json")) continue;
		inputs[path] = sha256FileAt(resolve(hostDir, path));
	}
	const metafileProjection: MetafileProjection = {
		schema: METAFILE_PROJECTION_SCHEMA,
		entry: "./src/main.ts",
		hostDirRel: HOST_PACKAGE_DIR,
		argv,
		metafileSha256: sha256Text(metafileText),
		inputs,
	};
	const metafileProjectionText = `${JSON.stringify(metafileProjection, null, "\t")}\n`;
	writeFileSync(join(outDir, "metafile-projection.json"), metafileProjectionText);

	// 3. cargo metadata for the pre-change graph.
	const cargoRaw = spawnSync("cargo", [...CARGO_METADATA_ARGV], {
		cwd: root,
		encoding: "utf8",
		timeout: 120_000,
		maxBuffer: 64 * 1024 * 1024,
		killSignal: "SIGKILL",
	});
	if (cargoRaw.status !== 0) {
		throw new ExposureError(`cargo metadata failed: ${(cargoRaw.stderr ?? "").slice(0, 800)}`);
	}
	const cargoGraph = projectCargoMetadata(cargoRaw.stdout ?? "", CARGO_METADATA_ARGV, "cargo metadata");
	const cargoProjectionText = `${JSON.stringify(
		{
			schema: CARGO_GRAPH_PROJECTION_SCHEMA,
			argv: [...CARGO_METADATA_ARGV],
			workspaceMembers: cargoGraph.workspaceMembers,
			edges: cargoGraph.edges,
		},
		null,
		"\t",
	)}\n`;
	writeFileSync(join(outDir, "cargo-graph-projection.json"), cargoProjectionText);

	// 4. Surfaces + authority + cargo file pins.
	const surfaces = loadNpmSurfaces(root);
	const manifest: ReferenceManifest = {
		schema: REFERENCE_SCHEMA,
		capturedAt: new Date().toISOString(),
		captureHead: gitOutput(root, ["rev-parse", "HEAD"]).trim(),
		relevantTreeStatus,
		metafile: {
			projectionPath: "metafile-projection.json",
			sha256: sha256Text(metafileProjectionText),
			entry: "./src/main.ts",
			hostDirRel: HOST_PACKAGE_DIR,
			argv,
			metafileSha256: metafileProjection.metafileSha256,
		},
		cargo: {
			projectionPath: "cargo-graph-projection.json",
			sha256: sha256Text(cargoProjectionText),
			argv: [...CARGO_METADATA_ARGV],
		},
		npmSurfaces: surfaces.map((surface) => ({
			path: surface.relPath,
			sha256: surface.sha256,
			packageName: surface.packageName,
			depFields: surface.depFields,
		})),
		cargoFiles: listCargoFiles(root).map((path) => ({
			path,
			sha256: sha256FileAt(join(root, path)),
		})),
		authority: AUTHORITY_REL_PATHS.map((path) => ({
			path,
			sha256: sha256FileAt(join(root, path)),
		})),
	};
	const manifestText = `${JSON.stringify(manifest, null, "\t")}\n`;
	writeFileSync(join(outDir, "reference.json"), manifestText);
	return manifest;
}

// ---------------------------------------------------------------------------
// self-check
// ---------------------------------------------------------------------------

export interface SelfCheckOutcome {
	readonly name: string;
	readonly expected: string;
	readonly actual: string;
	readonly ok: boolean;
	readonly detail: string;
}

/**
 * Known-member sanity + fail-closed probes against the canonical reference:
 * typebox must classify Class S (prod-field + metafile reachability) and
 * @types/bun must classify Class E (its recorded verdict). The synthetic
 * probe tampers the reference hash chain and must still yield Class S.
 */
export async function selfCheck(referenceDir: string, tmpDir: string): Promise<SelfCheckOutcome[]> {
	const outcomes: SelfCheckOutcome[] = [];

	const typebox = await classify({ subject: parseSubject("npm:typebox"), referenceDir });
	outcomes.push({
		name: "npm:typebox",
		expected: "S",
		actual: typebox.verdict.exposureClass,
		ok: typebox.verdict.exposureClass === "S",
		detail: typebox.verdict.reason,
	});

	const typesBun = await classify({ subject: parseSubject("npm:@types/bun"), referenceDir });
	outcomes.push({
		name: "npm:@types/bun",
		expected: "E",
		actual: typesBun.verdict.exposureClass,
		ok: typesBun.verdict.exposureClass === "E",
		detail: typesBun.verdict.reason,
	});

	const bunRuntime = await classify({ subject: parseSubject("tool:bun-runtime"), referenceDir });
	outcomes.push({
		name: "tool:bun-runtime",
		expected: "S",
		actual: bunRuntime.verdict.exposureClass,
		ok: bunRuntime.verdict.exposureClass === "S",
		detail: bunRuntime.verdict.reason,
	});

	// Synthetic fail-closed probe: tamper the canonical reference copy so the
	// cargo projection no longer matches its pinned hash; classification must
	// fail closed to Class S instead of exempting.
	const tamperedDir = join(tmpDir, "tampered-reference");
	mkdirSync(tamperedDir, { recursive: true });
	for (const name of ["reference.json", "metafile-projection.json", "cargo-graph-projection.json"]) {
		writeFileSync(join(tamperedDir, name), readFileSync(join(referenceDir, name)));
	}
	const cargoPath = join(tamperedDir, "cargo-graph-projection.json");
	writeFileSync(cargoPath, `${readFileSync(cargoPath, "utf8")}\n`);
	const probe = await classify({ subject: parseSubject("npm:@types/bun"), referenceDir: tamperedDir });
	outcomes.push({
		name: "fail-closed probe (tampered reference hash chain)",
		expected: "S",
		actual: probe.verdict.exposureClass,
		ok: probe.verdict.exposureClass === "S",
		detail: probe.verdict.reason,
	});

	return outcomes;
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

function usage(): never {
	process.stderr.write(
		[
			"usage:",
			"  dependency-exposure.ts capture-reference --out <dir>",
			"  dependency-exposure.ts classify --subject <kind:name> --reference <dir> [--cargo-metadata-file <path>] [--emit-ledger-row]",
			"  dependency-exposure.ts self-check [--reference <dir>]",
			"",
		].join("\n"),
	);
	process.exit(2);
}

function argValue(args: readonly string[], name: string): string | undefined {
	const index = args.indexOf(name);
	if (index === -1) return undefined;
	const value = args[index + 1];
	if (value === undefined || value.startsWith("--")) return undefined;
	return value;
}

function renderReport(report: ExposureReport): string {
	const lines = [
		`subject:    ${report.subject}`,
		`reference:  ${report.referenceDir} (captured ${report.capturedAt})`,
		`class:      ${report.verdict.exposureClass}`,
		`reason:     ${report.verdict.reason}`,
		"",
	];
	for (const name of CHECK_NAMES) {
		const check = report.checks[name];
		lines.push(`  ${name} ${check.status.padEnd(11)} ${check.detail}`);
	}
	return lines.join("\n");
}

function ledgerRow(report: ExposureReport): string {
	const summary = report.checks[CHECK_NAMES[0]] !== undefined
		? CHECK_NAMES.map((name) => `${name}:${report.checks[name].status}`).join(" ")
		: "";
	const head = gitOutput(REPO_ROOT, ["rev-parse", "HEAD"]).trim().slice(0, 12);
	const date = report.capturedAt.slice(0, 10);
	return `| ${head} | ${date} | ${report.subject} | ${report.verdict.exposureClass} | ${summary} |`;
}

async function main(): Promise<number> {
	const args = process.argv.slice(2);
	const command = args[0];
	if (command === "capture-reference") {
		const out = argValue(args, "--out");
		if (out === undefined) usage();
		const manifest = await captureReference(REPO_ROOT, resolve(out), {
			allowDirtyRelevant: args.includes("--allow-dirty-relevant"),
		});
		const bundle = await loadReferenceBundle(resolve(out));
		process.stdout.write(
			`captured reference at ${resolve(out)} (head ${manifest.captureHead ?? "?"}, ${Object.keys(bundle.metafile.inputs).length} metafile inputs${manifest.relevantTreeStatus !== undefined ? ", DIRTY-RELEVANT-RECORDED" : ""})\n`,
		);
		return 0;
	}
	if (command === "classify") {
		const subjectRaw = argValue(args, "--subject");
		const referenceDir = argValue(args, "--reference");
		if (subjectRaw === undefined || referenceDir === undefined) usage();
		const report = await classify({
			subject: parseSubject(subjectRaw),
			referenceDir: resolve(referenceDir),
			cargoMetadataFile:
				argValue(args, "--cargo-metadata-file") === undefined
					? undefined
					: resolve(argValue(args, "--cargo-metadata-file") as string),
		});
		process.stdout.write(`${renderReport(report)}\n${report.sentinel}\n`);
		if (args.includes("--emit-ledger-row")) {
			process.stdout.write(`ledger-row: ${ledgerRow(report)}\n`);
		}
		return 0;
	}
	if (command === undefined || command === "self-check") {
		const referenceDir = resolve(argValue(args, "--reference") ?? CANONICAL_REFERENCE_DIR);
		const tmpDir = join(REPO_ROOT, "target", "dependency-exposure-selfcheck");
		const outcomes = await selfCheck(referenceDir, tmpDir);
		for (const outcome of outcomes) {
			const mark = outcome.ok ? "ok  " : "FAIL";
			process.stdout.write(`${mark} ${outcome.name}: expected ${outcome.expected}, got ${outcome.actual} — ${outcome.detail}\n`);
		}
		if (outcomes.every((outcome) => outcome.ok)) {
			process.stdout.write(`${SENTINEL_OK}\n`);
			return 0;
		}
		return 1;
	}
	usage();
}

if (import.meta.main) {
	try {
		process.exit(await main());
	} catch (error) {
		// Fail closed: an operator must never read a missing verdict as an exemption.
		process.stderr.write(`${SENTINEL_FAILED_CLOSED}: ${errorText(error)}\n`);
		process.exit(1);
	}
}
