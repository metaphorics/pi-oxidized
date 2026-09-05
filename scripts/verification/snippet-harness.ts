#!/usr/bin/env bun
/**
 * Two-lane fenced snippet compiler (DOC-G1, issue #132).
 *
 * `runSnippetHarness(root)` returns the timestamp-free, JSON-serializable
 * `SnippetReport` consumed by DOC-D's future fenced-compile evidence runner.
 * This prototype does not write evidence or integrate with the blocked DOC-A
 * ledger. Rust snippets compile in an isolated temporary Cargo project against
 * live workspace crates. TypeScript snippets syntax-check and type-check against
 * shipped workspace entrypoints with the existing tsc.
 *
 * Go/no-go findings for DOC-D: the current docs have no Rust or TypeScript
 * fences; the protocol package has no `Codec` export, so fixtures cover its
 * codec functions; pi-ext and pi-tui expose the fixture surfaces through module
 * paths; tsc, not Bun's type-stripping transpiler, is the type authority;
 * offline Cargo requires a warm cache; fixed temporary paths assume one runner
 * per checkout.
 */

import { existsSync, mkdirSync, readFileSync, readdirSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { basename, isAbsolute, join, relative, resolve, sep } from "node:path";
import { SpawnRunner, type RunResult } from "../release/runner.ts";

export const REPO_ROOT = resolve(import.meta.dirname, "../..");
const FIXTURE_ROOT = "scripts/verification/fixtures/docs-snippets";
const RUST_TIMEOUT_MS = 10 * 60_000;
const TS_TIMEOUT_MS = 2 * 60_000;

export const EXCLUDED_EXAMPLE_PRODUCTS = [
	"with-deps",
	"custom-provider-anthropic",
	"custom-provider-gitlab-duo",
	"sandbox",
	"gondolin",
] as const;

export interface RequiredSnippetFixture {
	readonly lane: "rust" | "ts";
	readonly path: string;
	readonly fenceIndex: number;
	readonly probes: readonly string[];
}

/** Smallest fixture contract for every DOC-G1 public entrypoint group. */
export const REQUIRED_SNIPPET_FIXTURES: readonly RequiredSnippetFixture[] = [
	{
		lane: "rust",
		path: `${FIXTURE_ROOT}/rust/pi-ai.md`,
		fenceIndex: 0,
		probes: ["use pi_ai::estimate_text_tokens;", "estimate_text_tokens(\"hello\")"],
	},
	{
		lane: "rust",
		path: `${FIXTURE_ROOT}/rust/pi-agent.md`,
		fenceIndex: 0,
		probes: ["PendingMessageQueue", "QueueMode"],
	},
	{
		lane: "rust",
		path: `${FIXTURE_ROOT}/rust/pi-ext.md`,
		fenceIndex: 0,
		probes: ["pi_ext::protocol::FLAGS_SET_METHOD"],
	},
	{
		lane: "rust",
		path: `${FIXTURE_ROOT}/rust/pi-tui.md`,
		fenceIndex: 0,
		probes: ["pi_tui::keys::is_kitty_protocol_active"],
	},
	{
		lane: "rust",
		path: `${FIXTURE_ROOT}/rust/pi.md`,
		fenceIndex: 0,
		probes: ["pi::VERSION"],
	},
	{
		lane: "ts",
		path: `${FIXTURE_ROOT}/ts/protocol.md`,
		fenceIndex: 0,
		probes: [
			'from "@earendil-works/pi-tui-protocol"',
			"encodeFrame",
			"decodeFrameLine",
			"FrameDecoder",
			"ProtocolClient",
			"validateFrame",
		],
	},
	{
		lane: "ts",
		path: `${FIXTURE_ROOT}/ts/extension-host.md`,
		fenceIndex: 0,
		probes: [
			'from "@earendil-works/pi-extension-host"',
			"ExtensionHost",
			"TerminalInputHandler",
			"parseAnsiLine",
		],
	},
	{
		lane: "ts",
		path: `${FIXTURE_ROOT}/ts/extension-host.md`,
		fenceIndex: 1,
		probes: ['from "@earendil-works/pi-coding-agent"', "ExtensionMode", "SourceInfo"],
	},
] as const;

export interface PublicShellSnippet {
	readonly path: string;
	readonly fenceIndex: number;
	readonly probes: readonly string[];
}

/**
 * Contract for the public `bash` command blocks in README.md and
 * docs/getting-started.md. Every probe must coexist inside the single bash
 * fence at `fenceIndex`. Commands are registered, never executed; the real
 * commands are verified separately.
 */
export const PUBLIC_SHELL_SNIPPETS: readonly PublicShellSnippet[] = [
	{
		path: "README.md",
		fenceIndex: 0,
		probes: [
			"mkdir -p .references",
			"git clone https://github.com/earendil-works/pi.git .references/pi-2.0",
			"git -C .references/pi-2.0 checkout --detach 853a80d26c90a14c1886f0ebb8ffaae133ca2185",
			'test "$(git -C .references/pi-2.0 rev-parse HEAD)" = "853a80d26c90a14c1886f0ebb8ffaae133ca2185"',
			"bun install --frozen-lockfile",
			"bun run scripts/reconstruct-provider-data.ts",
			"npm ci --ignore-scripts --prefix .references/pi-2.0",
			"bun run build:extension-host --target x86_64-unknown-linux-gnu",
			"cargo build -p pi --release --locked",
		],
	},
	{
		path: "README.md",
		fenceIndex: 1,
		probes: ['read -rsp "Enter Gemini API key: " GEMINI_API_KEY && export GEMINI_API_KEY', "printf '\\n'"],
	},
	{
		path: "README.md",
		fenceIndex: 2,
		probes: [
			'PI_EXTENSION_HOST="$PWD/dist/release/.staging-host/host/x86_64-unknown-linux-gnu/pi-extension-host"',
			"target/release/pi --provider google --model gemini-flash-latest",
		],
	},
	{
		path: "docs/getting-started.md",
		fenceIndex: 0,
		probes: [
			"git clone https://github.com/gosuda/pi-oxidized.git",
			"cd pi-oxidized",
			"mkdir -p .references",
			"git clone https://github.com/earendil-works/pi.git .references/pi-2.0",
			"git -C .references/pi-2.0 checkout --detach 853a80d26c90a14c1886f0ebb8ffaae133ca2185",
			'test "$(git -C .references/pi-2.0 rev-parse HEAD)" = "853a80d26c90a14c1886f0ebb8ffaae133ca2185"',
			"bun install --frozen-lockfile",
			"bun run scripts/reconstruct-provider-data.ts",
			"npm ci --ignore-scripts --prefix .references/pi-2.0",
			"bun run build:extension-host --target x86_64-unknown-linux-gnu",
			"cargo build -p pi --release --locked",
			"target/release/pi --help",
		],
	},
	{
		path: "docs/getting-started.md",
		fenceIndex: 1,
		probes: ['read -rsp "Enter Gemini API key: " GEMINI_API_KEY && export GEMINI_API_KEY', "printf '\\n'"],
	},
	{
		path: "docs/getting-started.md",
		fenceIndex: 2,
		probes: [
			'PI_EXTENSION_HOST="$PWD/dist/release/.staging-host/host/x86_64-unknown-linux-gnu/pi-extension-host"',
			"target/release/pi --provider google --model gemini-flash-latest",
		],
	},
	{
		path: "docs/getting-started.md",
		fenceIndex: 3,
		probes: ["target/release/pi --provider google --model gemini-flash-latest"],
	},
	{
		path: "docs/getting-started.md",
		fenceIndex: 4,
		probes: [
			"bun run build:extension-host --target x86_64-unknown-linux-gnu",
			'PI_EXTENSION_HOST="$PWD/dist/release/.staging-host/host/x86_64-unknown-linux-gnu/pi-extension-host"',
			"target/release/pi --provider google --model gemini-flash-latest",
		],
	},
] as const;

export interface Fence {
	readonly docPath: string;
	readonly openLine: number;
	readonly bodyStartLine: number;
	readonly infoString: string;
	readonly body: string;
}

export type FenceClass = "rust" | "rust-skip" | "ts" | "ignore" | "unsupported";
export type SnippetTool = "rustc" | "tsc" | "extract" | "env";
export type SnippetLane = "rust" | "ts" | "extract" | "env";

export interface SnippetFailure {
	readonly docPath: string;
	readonly line: number;
	readonly column?: number;
	readonly lane: SnippetLane;
	readonly snippetId: string;
	readonly tool: SnippetTool;
	readonly code?: string;
	readonly message: string;
}

export interface ExtractResult {
	readonly fences: Fence[];
	readonly failures: SnippetFailure[];
}

export interface LaneResult {
	readonly lane: "rust" | "ts";
	readonly extracted: number;
	readonly documents: number;
	readonly fixtures: number;
	readonly compiled: number;
	readonly skipped: number;
	readonly failures: SnippetFailure[];
}

export interface SnippetReport {
	readonly ok: boolean;
	readonly lanes: {
		readonly rust: LaneResult;
		readonly ts: LaneResult;
	};
	readonly violations: string[];
}

interface RegisteredFence extends Fence {
	readonly kind: FenceClass;
	readonly snippetId: string;
}

interface CargoDiagnostic {
	readonly reason?: string;
	readonly message?: {
		readonly code?: { readonly code?: string } | null;
		readonly message?: string;
		readonly spans?: readonly {
			readonly file_name?: string;
			readonly line_start?: number;
			readonly column_start?: number;
			readonly is_primary?: boolean;
		}[];
	};
}

function normalizedRelativePath(path: string): string | undefined {
	if (path === "" || isAbsolute(path) || path.includes("\\")) return undefined;
	const normalized = path.split("/").filter((part) => part !== ".").join("/");
	if (normalized === "" || normalized.split("/").includes("..")) return undefined;
	return normalized;
}

export function validateSourcePath(root: string, path: string): string | undefined {
	const normalized = normalizedRelativePath(path);
	if (normalized === undefined) return undefined;
	const absoluteRoot = resolve(root);
	const absolutePath = resolve(absoluteRoot, normalized);
	const fromRoot = relative(absoluteRoot, absolutePath);
	if (fromRoot === "" || fromRoot === ".." || fromRoot.startsWith(`..${sep}`) || isAbsolute(fromRoot)) {
		return undefined;
	}
	return normalized;
}

function failure(
	docPath: string,
	line: number,
	lane: SnippetLane,
	tool: SnippetTool,
	message: string,
	extra: { snippetId?: string; column?: number; code?: string } = {},
): SnippetFailure {
	return {
		docPath,
		line,
		lane,
		snippetId: extra.snippetId ?? `${docPath}:${line}`,
		tool,
		message,
		...(extra.column === undefined ? {} : { column: extra.column }),
		...(extra.code === undefined ? {} : { code: extra.code }),
	};
}

export function extractFences(source: string, docPath: string): ExtractResult {
	const safePath = normalizedRelativePath(docPath);
	if (safePath === undefined) {
		return { fences: [], failures: [failure(docPath, 1, "extract", "extract", "source path must be repo-relative and cannot traverse parents")] };
	}
	const lines = source.split(/\r?\n/);
	const fences: Fence[] = [];
	const failures: SnippetFailure[] = [];
	let open: { ticks: number; line: number; info: string; body: string[] } | undefined;
	for (let index = 0; index < lines.length; index += 1) {
		const text = lines[index] ?? "";
		if (open === undefined) {
			const match = /^ {0,3}(`{3,})(.*)$/.exec(text);
			if (match === null) continue;
			const ticks = match[1]?.length ?? 0;
			const info = (match[2] ?? "").trim();
			if (info.includes("`")) {
				failures.push(failure(safePath, index + 1, "extract", "extract", "fence info string cannot contain backticks"));
				continue;
			}
			open = { ticks, line: index + 1, info, body: [] };
			continue;
		}
		const close = /^ {0,3}(`{3,})\s*$/.exec(text);
		if (close !== null && (close[1]?.length ?? 0) >= open.ticks) {
			fences.push({
				docPath: safePath,
				openLine: open.line,
				bodyStartLine: open.line + 1,
				infoString: open.info,
				body: open.body.join("\n"),
			});
			open = undefined;
			continue;
		}
		open.body.push(text);
	}
	if (open !== undefined) {
		failures.push(failure(safePath, open.line, "extract", "extract", "unclosed backtick fence"));
	}
	return { fences, failures };
}

export function classifyFence(infoString: string): FenceClass {
	const parts = infoString.split(",").map((part) => part.trim()).filter(Boolean);
	const language = parts[0]?.toLowerCase() ?? "";
	const metadata = parts.slice(1);
	if (language === "rust") {
		if (metadata.length === 0 || (metadata.length === 1 && metadata[0] === "no_run")) return "rust";
		if (metadata.length === 1 && (metadata[0] === "ignore" || metadata[0] === "text")) return "rust-skip";
		return "unsupported";
	}
	if (language === "ts" || language === "typescript") return metadata.length === 0 ? "ts" : "unsupported";
	return "ignore";
}

function collectMarkdownFiles(root: string, relativeDirectory: string): string[] {
	const safeDirectory = validateSourcePath(root, relativeDirectory);
	if (safeDirectory === undefined) return [];
	const absoluteDirectory = join(root, safeDirectory);
	if (!existsSync(absoluteDirectory)) return [];
	const files: string[] = [];
	const visit = (directory: string): void => {
		for (const entry of readdirSync(directory, { withFileTypes: true })) {
			const absolute = join(directory, entry.name);
			if (entry.isDirectory()) visit(absolute);
			else if (entry.isFile() && entry.name.endsWith(".md") && !entry.name.startsWith("negative-")) {
				files.push(relative(root, absolute).split(sep).join("/"));
			}
		}
	};
	visit(absoluteDirectory);
	return files.sort();
}

export function collectDocFences(root: string): { fences: RegisteredFence[]; failures: SnippetFailure[] } {
	const files = [...collectMarkdownFiles(root, "docs"), ...collectMarkdownFiles(root, FIXTURE_ROOT)].sort();
	const fences: RegisteredFence[] = [];
	const failures: SnippetFailure[] = [];
	for (const docPath of files) {
		const safePath = validateSourcePath(root, docPath);
		if (safePath === undefined) {
			failures.push(failure(docPath, 1, "extract", "extract", "source path escaped repository root"));
			continue;
		}
		const extracted = extractFences(readFileSync(join(root, safePath), "utf8"), safePath);
		failures.push(...extracted.failures);
		for (const fence of extracted.fences) {
			const kind = classifyFence(fence.infoString);
			const snippetId = `${fence.docPath}:${fence.openLine}`;
			if (kind === "unsupported") {
				failures.push(failure(fence.docPath, fence.openLine, "extract", "extract", `unsupported fence metadata: ${fence.infoString}`, { snippetId }));
			}
			fences.push({ ...fence, kind, snippetId });
		}
	}
	fences.sort((a, b) => a.docPath.localeCompare(b.docPath) || a.openLine - b.openLine);
	return { fences, failures: sortFailures(failures) };
}

export function wrapRustSnippet(body: string): { code: string; headerLines: number } {
	const visible = body.split("\n").map((line) => line.startsWith("# ") ? line.slice(2) : line).join("\n");
	if (/\bfn\s+main\s*\(/.test(visible)) return { code: `${visible}\n`, headerLines: 0 };
	const indented = visible.split("\n").map((line) => line === "" ? "" : `    ${line}`).join("\n");
	return { code: `fn main() {\n${indented}\n}\n`, headerLines: 1 };
}

const RUST_CRATES = [
	["pi_agent", "pi-agent"],
	["pi_ai", "pi-ai"],
	["pi_ext", "pi-ext"],
	["pi_tui", "pi-tui"],
	["pi", "pi"],
] as const;

export function inferRustDeps(body: string): string[] {
	return RUST_CRATES.filter(([identifier]) => new RegExp(`(?:^|[^A-Za-z0-9_])${identifier}(?=::|[^A-Za-z0-9_])`, "m").test(body))
		.map(([, packageName]) => packageName)
		.sort();
}

function laneCounts(fences: readonly RegisteredFence[], kind: "rust" | "ts"): Pick<LaneResult, "extracted" | "documents" | "fixtures"> {
	const selected = fences.filter((fence) => fence.kind === kind);
	return {
		extracted: selected.length,
		documents: selected.filter((fence) => fence.docPath.startsWith("docs/")).length,
		fixtures: selected.filter((fence) => fence.docPath.startsWith(`${FIXTURE_ROOT}/`)).length,
	};
}

function emptyLane(lane: "rust" | "ts", fences: readonly RegisteredFence[]): LaneResult {
	return { lane, ...laneCounts(fences, lane), compiled: 0, skipped: 0, failures: [] };
}

export function mapCargoDiagnostic(
	line: string,
	byGeneratedFile: ReadonlyMap<string, { fence: Fence; headerLines: number }>,
): SnippetFailure | undefined {
	let diagnostic: CargoDiagnostic;
	try {
		diagnostic = JSON.parse(line) as CargoDiagnostic;
	} catch {
		return undefined;
	}
	if (diagnostic.reason !== "compiler-message" || diagnostic.message === undefined) return undefined;
	const span = diagnostic.message.spans?.find((candidate) => candidate.is_primary && candidate.file_name !== undefined);
	if (span?.file_name === undefined || span.line_start === undefined) return undefined;
	const mapping = byGeneratedFile.get(basename(span.file_name));
	if (mapping === undefined) return undefined;
	const lineInBody = span.line_start - mapping.headerLines;
	const docLine = mapping.fence.bodyStartLine + Math.max(0, lineInBody - 1);
	return failure(mapping.fence.docPath, docLine, "rust", "rustc", diagnostic.message.message ?? "Rust compilation failed", {
		snippetId: `${mapping.fence.docPath}:${mapping.fence.openLine}`,
		code: diagnostic.message.code?.code,
	});
}

function tail(text: string, limit = 2000): string {
	const trimmed = text.trim();
	return trimmed.length <= limit ? trimmed : trimmed.slice(-limit);
}

export async function runRustLane(root: string, allFences: readonly RegisteredFence[]): Promise<LaneResult> {
	const lane = emptyLane("rust", allFences);
	const fences = allFences.filter((fence) => fence.kind === "rust");
	const skipped = allFences.filter((fence) => fence.kind === "rust-skip" || (fence.kind === "unsupported" && fence.infoString.startsWith("rust"))).length;
	if (fences.length === 0) return { ...lane, skipped };
	const laneRoot = join(resolve(root), "target/snippet-harness/lane1");
	const byGeneratedFile = new Map<string, { fence: Fence; headerLines: number }>();
	try {
		rmSync(laneRoot, { recursive: true, force: true });
		mkdirSync(join(laneRoot, "src/bin"), { recursive: true });
		const dependencies = new Set<string>();
		for (const [index, fence] of fences.entries()) {
			const generatedName = `snippet_${String(index).padStart(3, "0")}.rs`;
			const wrapped = wrapRustSnippet(fence.body);
			writeFileSync(join(laneRoot, "src/bin", generatedName), wrapped.code);
			byGeneratedFile.set(generatedName, { fence, headerLines: wrapped.headerLines });
			for (const dependency of inferRustDeps(fence.body)) dependencies.add(dependency);
		}
		const manifest = [
			"[package]",
			'name = "pi-snippet-harness"',
			'version = "0.0.0"',
			'edition = "2024"',
			'rust-version = "1.97.1"',
			"publish = false",
			"",
			"[dependencies]",
			...[...dependencies].sort().map((name) => `${name} = { path = ${JSON.stringify(join(resolve(root), "crates", name))} }`),
			"",
			"[workspace]",
			"",
		].join("\n");
		writeFileSync(join(laneRoot, "Cargo.toml"), manifest);
		let result: RunResult;
		try {
			result = await new SpawnRunner().run("cargo", ["check", "--offline", "--message-format=json"], {
				cwd: laneRoot,
				env: { CARGO_TARGET_DIR: join(laneRoot, "target") },
				timeoutMs: RUST_TIMEOUT_MS,
			});
		} catch (error) {
			return { ...lane, skipped, failures: [failure("target/snippet-harness/lane1", 1, "env", "env", `cargo could not run: ${error instanceof Error ? error.message : String(error)}`)] };
		}
		const mapped = result.stdout.split(/\r?\n/).map((line) => mapCargoDiagnostic(line, byGeneratedFile)).filter((item): item is SnippetFailure => item !== undefined);
		if (result.exitCode !== 0 && mapped.length === 0) {
			mapped.push(failure("target/snippet-harness/lane1", 1, "env", "env", `cargo check failed without a snippet span: ${tail(result.stderr || result.stdout)}`));
		}
		return { ...lane, compiled: result.exitCode === 0 ? fences.length : 0, skipped, failures: sortFailures(mapped) };
	} finally {
		rmSync(laneRoot, { recursive: true, force: true });
	}
}

export function mapTscDiagnostic(
	line: string,
	byGeneratedFile: ReadonlyMap<string, Fence>,
): SnippetFailure | undefined {
	const match = /^(.*?)\((\d+),(\d+)\): error (TS\d+): (.*)$/.exec(line.trim());
	if (match === null) return undefined;
	const mapping = byGeneratedFile.get(basename(match[1] ?? ""));
	if (mapping === undefined) return undefined;
	const generatedLine = Number(match[2]);
	return failure(mapping.docPath, mapping.bodyStartLine + generatedLine - 1, "ts", "tsc", match[5] ?? "TypeScript compilation failed", {
		snippetId: `${mapping.docPath}:${mapping.openLine}`,
		column: Number(match[3]),
		code: match[4],
	});
}

export function verifyShippedEntrypointsExist(root: string): string[] {
	const required = [
		"packages/pi-tui-protocol/dist/index.d.ts",
		"packages/extension-host/src/index.ts",
		"packages/extension-host/src/refs.d.ts",
		"node_modules/typescript/bin/tsc",
	];
	return required.filter((path) => !existsSync(join(root, path))).map((path) => `${path}: required TypeScript entrypoint or compiler is missing`);
}

export async function runTypeScriptLane(root: string, allFences: readonly RegisteredFence[]): Promise<LaneResult> {
	const lane = emptyLane("ts", allFences);
	const fences = allFences.filter((fence) => fence.kind === "ts");
	const skipped = allFences.filter((fence) => fence.kind === "unsupported" && /^(ts|typescript)(?:,|$)/.test(fence.infoString)).length;
	if (fences.length === 0) return { ...lane, skipped };
	const laneRoot = join(resolve(root), "target/snippet-harness/lane2");
	const byGeneratedFile = new Map<string, Fence>();
	try {
		rmSync(laneRoot, { recursive: true, force: true });
		mkdirSync(laneRoot, { recursive: true });
		const packageScope = join(laneRoot, "node_modules/@earendil-works");
		mkdirSync(packageScope, { recursive: true });
		symlinkSync(join(resolve(root), "packages/pi-tui-protocol"), join(packageScope, "pi-tui-protocol"), "dir");
		symlinkSync(join(resolve(root), "packages/extension-host"), join(packageScope, "pi-extension-host"), "dir");
		for (const [index, fence] of fences.entries()) {
			const generatedName = `snippet_${String(index).padStart(3, "0")}.ts`;
			writeFileSync(join(laneRoot, generatedName), `${fence.body}\n`);
			byGeneratedFile.set(generatedName, fence);
		}
		const paths = {
			"@earendil-works/pi-coding-agent": [join(resolve(root), "packages/extension-host/src/refs.d.ts")],
			"@earendil-works/pi-coding-agent/*": [join(resolve(root), "packages/extension-host/src/refs.d.ts")],
			"@earendil-works/pi-ai": [join(resolve(root), "packages/extension-host/src/refs.d.ts")],
			"@earendil-works/pi-ai/*": [join(resolve(root), "packages/extension-host/src/refs.d.ts")],
		};
		writeFileSync(join(laneRoot, "tsconfig.json"), `${JSON.stringify({ compilerOptions: { target: "ESNext", module: "ESNext", moduleResolution: "bundler", lib: ["ESNext"], types: ["bun"], typeRoots: [join(resolve(root), "node_modules/@types")], strict: true, noUncheckedIndexedAccess: true, exactOptionalPropertyTypes: false, noEmit: true, skipLibCheck: true, allowImportingTsExtensions: true, paths }, include: ["snippet_*.ts"] }, null, 2)}\n`);
		let result: RunResult;
		try {
			// Run the compiler through bun, not the .bin shim: .bin/tsc has no
			// extensionless executable on Windows (tsc.cmd only), while the
			// package entry runs everywhere bun does.
			result = await new SpawnRunner().run("bun", [join(resolve(root), "node_modules/typescript/bin/tsc"), "--noEmit", "-p", laneRoot, "--pretty", "false"], { cwd: resolve(root), timeoutMs: TS_TIMEOUT_MS });
		} catch (error) {
			return { ...lane, skipped, failures: [failure("target/snippet-harness/lane2", 1, "env", "env", `tsc could not run: ${error instanceof Error ? error.message : String(error)}`)] };
		}
		const output = `${result.stdout}\n${result.stderr}`;
		const mapped = output.split(/\r?\n/).map((line) => mapTscDiagnostic(line, byGeneratedFile)).filter((item): item is SnippetFailure => item !== undefined);
		if (result.exitCode !== 0 && mapped.length === 0) {
			mapped.push(failure("target/snippet-harness/lane2", 1, "env", "env", `tsc failed without a snippet span: ${tail(output)}`));
		}
		return { ...lane, compiled: result.exitCode === 0 ? fences.length : 0, skipped, failures: sortFailures(mapped) };
	} finally {
		rmSync(laneRoot, { recursive: true, force: true });
	}
}

export function verifyNoExcludedExampleProducts(root: string): string[] {
	const fixtureRoot = join(resolve(root), FIXTURE_ROOT);
	if (!existsSync(fixtureRoot)) return [`${FIXTURE_ROOT}: fixture root is missing`];
	const violations: string[] = [];
	const visit = (directory: string): void => {
		for (const entry of readdirSync(directory, { withFileTypes: true })) {
			const absolute = join(directory, entry.name);
			if (entry.isDirectory()) visit(absolute);
			else if (entry.isFile()) {
				const fixturePath = relative(resolve(root), absolute).split(sep).join("/");
				const body = readFileSync(absolute, "utf8");
				for (const name of EXCLUDED_EXAMPLE_PRODUCTS) {
					if (body.includes(name)) violations.push(`${fixturePath}: excluded product name ${name}`);
				}
				if (body.includes(".references/")) violations.push(`${fixturePath}: reference-tree import is forbidden`);
				if (/(^|["'`\s])(?:\.\.\/|\.\/|\/)?examples\//m.test(body)) violations.push(`${fixturePath}: examples path is forbidden`);
			}
		}
	};
	visit(fixtureRoot);
	return violations.sort();
}

/** Directories excluded from the examples/ directory check. */
const EXAMPLES_DIR_EXCLUDES: Record<string, true> = {
	".references": true,
	"node_modules": true,
	".git": true,
	"target": true,
	"prototype": true,
};

/**
 * Assert no `examples/` directory exists in the tree (excluding reference,
 * build, and dependency directories). DOC-D acceptance: the boundary excludes
 * any examples/ tree.
 */
export function verifyNoExamplesDirectory(root: string): string[] {
	const absoluteRoot = resolve(root);
	const violations: string[] = [];
	const visit = (directory: string): void => {
		for (const entry of readdirSync(directory, { withFileTypes: true })) {
			if (entry.isDirectory()) {
				if (entry.name === "examples") {
					const rel = relative(absoluteRoot, join(directory, entry.name)).split(sep).join("/");
					violations.push(`${rel}: examples/ directory is forbidden in the tree`);
					continue;
				}
				if (!(entry.name in EXAMPLES_DIR_EXCLUDES)) visit(join(directory, entry.name));
			}
		}
	};
	visit(absoluteRoot);
	return violations.sort();
}

/** Rust crate identifiers that can appear in `use` statements. */
const RUST_CRATE_IDENTIFIERS: readonly string[] = RUST_CRATES.map(([identifier]) => identifier);

/** TypeScript package specifiers that can appear in `import` statements. */
const TS_PACKAGE_SPECIFIERS = [
	"@earendil-works/pi-tui-protocol",
	"@earendil-works/pi-extension-host",
	"@earendil-works/pi-coding-agent",
	"@earendil-works/pi-ai",
] as const;

export interface FenceImport {
	readonly fenceId: string;
	readonly lane: "rust" | "ts";
	readonly specifier: string;
	readonly items: readonly string[];
}

/**
 * Enumerate imports across all registered fences (fixtures only, excluding
 * negative-* mutation fixtures). Returns one entry per fence with the resolved
 * module specifier and the imported item names.
 */
export function enumerateFenceImports(root: string): { imports: FenceImport[]; problems: string[] } {
	const { fences } = collectDocFences(root);
	const imports: FenceImport[] = [];
	for (const fence of fences) {
		if (fence.docPath.includes("negative-")) continue;
		if (fence.kind === "rust") {
			const useMatch = /^\s*use\s+([A-Za-z0-9_]+)(?:::|;|\s)/m.exec(fence.body);
			const crateName = useMatch?.[1];
			if (crateName !== undefined && RUST_CRATE_IDENTIFIERS.includes(crateName)) {
				const braceMatch = /use\s+[A-Za-z0-9_]+\s*::\s*\{([^}]*)\}/.exec(fence.body);
				if (braceMatch !== null) {
					const items = braceMatch[1]?.split(",").map((item) => item.trim()).filter((item) => item.length > 0) ?? [];
					imports.push({ fenceId: fence.snippetId, lane: "rust", specifier: crateName, items });
				} else {
					const singleMatch = /use\s+[A-Za-z0-9_]+\s*::\s*([A-Za-z0-9_]+)/.exec(fence.body);
					const items = singleMatch?.[1] !== undefined ? [singleMatch[1]] : [];
					imports.push({ fenceId: fence.snippetId, lane: "rust", specifier: crateName, items });
				}
			} else if (/\bpi_\w+::/.test(fence.body)) {
				const inlineMatch = /\b(pi_\w+)::/.exec(fence.body);
				if (inlineMatch?.[1] !== undefined && RUST_CRATE_IDENTIFIERS.includes(inlineMatch[1])) {
					imports.push({ fenceId: fence.snippetId, lane: "rust", specifier: inlineMatch[1], items: [] });
				}
			}
		} else if (fence.kind === "ts") {
			const importMatch = /^\s*import\s+(?:type\s+)?\{([^}]*)\}\s+from\s+["']([^"']+)["']/m.exec(fence.body);
			const specifier = importMatch?.[2];
			if (specifier !== undefined && (TS_PACKAGE_SPECIFIERS as readonly string[]).includes(specifier)) {
				const items = (importMatch?.[1] ?? "").split(",").map((item) => item.trim().replace(/^type\s+/, "")).filter((item) => item.length > 0);
				imports.push({ fenceId: fence.snippetId, lane: "ts", specifier, items });
			} else {
				const bareImport = /^\s*import\s+["']([^"']+)["']/m.exec(fence.body);
				if (bareImport?.[1] !== undefined && (TS_PACKAGE_SPECIFIERS as readonly string[]).includes(bareImport[1])) {
					imports.push({ fenceId: fence.snippetId, lane: "ts", specifier: bareImport[1], items: [] });
				}
			}
		}
	}
	imports.sort((a, b) => a.fenceId.localeCompare(b.fenceId) || a.specifier.localeCompare(b.specifier));
	return { imports, problems: [] };
}

/** Live Rust workspace exports keyed by crate identifier. */
const RUST_WORKSPACE_EXPORTS: Record<string, readonly string[]> = {
	pi_ai: ["estimate_text_tokens", "estimate_message_tokens", "estimate_messages_tokens", "estimate_context_tokens", "calculate_context_tokens", "estimate_text_and_image_content_tokens", "ContextUsageEstimate", "Provider", "ProviderError", "ProviderResponse", "StreamOptions", "SimpleStreamOptions", "ThinkingBudgets", "ThinkingBudgetsResolved", "AdjustedMaxTokens", "CONTEXT_SAFETY_TOKENS", "DEFAULT_CACHE_RETENTION", "DEFAULT_MAX_RETRY_DELAY_MS", "DEFAULT_THINKING_BUDGET_HIGH", "DEFAULT_THINKING_BUDGET_LOW", "DEFAULT_THINKING_BUDGET_MEDIUM", "DEFAULT_THINKING_BUDGET_MINIMAL", "adjust_max_tokens_for_thinking", "apply_simple_max_tokens_clamp", "apply_thinking_and_context_clamp", "build_base_options", "clamp_max_tokens_to_context", "clamp_reasoning", "default_thinking_budgets"],
	pi_agent: ["Agent", "AgentOptions", "AGENT_EVENT_CAPACITY", "AgentEventSink", "AgentEventSubscription", "EXTENSION_EVENT_CAPACITY", "EventSink", "ExtensionEvent", "ExtensionSubscription", "AfterToolCall", "AfterToolCallContext", "AfterToolCallResult", "AgentContext", "AgentLoopConfig", "AgentLoopTurnUpdate", "BeforeToolCall", "BeforeToolCallContext", "BeforeToolCallResult", "ConvertToLlm", "GetApiKey", "GetMessages", "PrepareNextTurn", "PrepareNextTurnContext", "ShouldStopAfterTurn", "ShouldStopAfterTurnContext", "TransformContext", "build_stream_options", "default_convert_to_llm_hook", "DRAIN_EVENT_CAPACITY", "DrainItem", "ProviderDrain", "AgentLoopError", "ToolError", "AgentEvent", "AgentMessage", "CustomAgentMessage", "default_convert_to_llm", "now_millis", "user_text", "PendingMessageQueue", "QueueMode", "RunIo", "run_agent_loop", "run_agent_loop_continue", "EmitAgentEvent", "ExecutedToolCallBatch", "MAX_PARALLEL_TOOL_CALLS", "PARALLEL_TOOL_UPDATE_CAPACITY", "execute_tool_calls", "fail_tool_calls_from_truncated_message", "should_terminate_tool_batch", "AgentState", "AgentStateSnapshot", "AGENT_TELEMETRY_SCHEMAS", "AI_TELEMETRY_SCHEMA", "AttributeValue", "HARNESS_TELEMETRY_SCHEMA", "InMemoryTelemetryContext", "RecordedEvent", "RecordedSpan", "SpanAttributes", "SpanOptions", "SpanStatus", "TelemetryContext", "TelemetrySchema", "TelemetrySpan", "noop_context", "AgentTool", "AgentToolResult", "ToolExecutionMode", "ToolUpdates", "error_tool_result", "to_pi_tool", "pi_ai"],
	pi_ext: ["adapters", "client", "host", "protocol", "sanitize", "server"],
	pi_tui: ["component", "components", "editor_support", "focus", "frame", "fuzzy", "image", "keybindings", "keys", "layout", "link", "overlay", "terminal", "text", "testkit"],
	pi: ["cli", "core", "modes", "remote", "VERSION"],
};

/** Live TypeScript workspace exports keyed by package specifier. */
const TS_WORKSPACE_EXPORTS: Record<string, readonly string[]> = {
	"@earendil-works/pi-tui-protocol": ["decodeFrameLine", "decodeFrameStr", "decodeFrameStrStrict", "encodeFrame", "encodeFrameString", "errorFrame", "eventFrame", "FrameDecoder", "ProtocolError", "requestFrame", "responseFrame", "validateFrame", "ByteReadable", "ByteWritable", "FrameHandler", "ProtocolClient", "RequestOptions", "COMPATIBILITY_VERSION", "ConfirmRequest", "ConfirmResponse", "DialogOptions", "DisposeSlot", "EditorRequest", "EditorResponse", "ErrorPayload", "ExtensionErrorEvent", "Frame", "FrameId", "FrameKind", "Hello", "HelloAck", "Hyperlink", "InputRequest", "InputResponse", "isMethod", "KeyEventKindWire", "KeyModifiersWire", "localHello", "localHelloAck", "MAX_FRAME_BYTES", "MeasureResponse", "Method", "METHODS", "NamedColor", "NotifyLevel", "NotifyRequest", "OverlayAnchor", "OverlayMargin", "OverlayOptions", "PROTOCOL_VERSION", "ProviderEvent", "SelectRequest", "SelectResponse", "SizeValue", "SlotCursor", "SlotPlacement", "SlotRenderRequest", "Style", "StyledRun", "TerminalInputResult", "ToolUpdate", "UiEventWire", "UiSlot", "WireColor"],
	"@earendil-works/pi-extension-host": ["ExtensionHost", "EXTENSION_HOOK_TIMEOUT_MS", "EXTENSION_INPUT_TIMEOUT_MS", "EXTENSION_INPUT_QUEUE_CAPACITY", "TerminalInputHandler", "TerminalInputHandlerResult", "parseAnsiLine", "parseAnsiLines", "MAX_HYPERLINK_ID_BYTES", "MAX_HYPERLINK_URI_BYTES", "COMPATIBILITY_VERSION", "getExtensionAliases", "createExtensionJiti"],
	"@earendil-works/pi-coding-agent": ["ExtensionMode", "SourceInfo", "TuiBridge", "ExtensionUIContext", "SourceScope", "SourceOrigin"],
};

/**
 * Assert every enumerated fence import resolves against live workspace
 * exports. DOC-D acceptance: a test enumerates imports across all registered
 * fences and asserts resolution against live workspace exports in both lanes.
 */
export function assertFenceImportsResolve(root: string): string[] {
	const { imports } = enumerateFenceImports(root);
	const violations: string[] = [];
	for (const imp of imports) {
		const exports = imp.lane === "rust"
			? RUST_WORKSPACE_EXPORTS[imp.specifier]
			: TS_WORKSPACE_EXPORTS[imp.specifier];
		if (exports === undefined) {
			violations.push(`${imp.fenceId}: ${imp.specifier} is not a known workspace export surface`);
			continue;
		}
		for (const item of imp.items) {
			if (!exports.includes(item)) {
				violations.push(`${imp.fenceId}: ${imp.specifier} does not export ${item}`);
			}
		}
	}
	return violations.sort();
}

export function verifyRequiredSnippetFixtures(root: string): string[] {
	const violations: string[] = [];
	for (const required of REQUIRED_SNIPPET_FIXTURES) {
		const safePath = validateSourcePath(root, required.path);
		if (safePath === undefined || !safePath.startsWith(`${FIXTURE_ROOT}/`)) {
			violations.push(`${required.path}: required fixture path escaped fixture root`);
			continue;
		}
		const absolute = join(resolve(root), safePath);
		if (!existsSync(absolute)) {
			violations.push(`${safePath}: required fixture is missing`);
			continue;
		}
		const extracted = extractFences(readFileSync(absolute, "utf8"), safePath);
		for (const item of extracted.failures) {
			violations.push(`${safePath}:${item.line}: ${item.message}`);
		}
		const laneFences = extracted.fences.filter((fence) => classifyFence(fence.infoString) === required.lane);
		const fence = laneFences[required.fenceIndex];
		if (fence === undefined) {
			violations.push(`${safePath}: missing ${required.lane} fence #${required.fenceIndex}`);
			continue;
		}
		for (const probe of required.probes) {
			if (!fence.body.includes(probe)) {
				violations.push(`${safePath}: fence #${required.fenceIndex} missing probe ${JSON.stringify(probe)}`);
			}
		}
	}
	return violations.sort();
}

function isBashFence(infoString: string): boolean {
	const language = infoString.split(",")[0]?.trim().toLowerCase() ?? "";
	return language === "bash";
}

/**
 * Assert the registered public `bash` command blocks exist intact. Reports
 * missing documents, missing bash fences, missing probes, and probes that
 * drifted into a different bash fence of the same document.
 */
export function verifyPublicShellSnippets(root: string): string[] {
	const violations: string[] = [];
	for (const required of PUBLIC_SHELL_SNIPPETS) {
		const safePath = validateSourcePath(root, required.path);
		if (safePath === undefined) {
			violations.push(`${required.path}: public shell snippet path escaped repo root`);
			continue;
		}
		const absolute = join(resolve(root), safePath);
		if (!existsSync(absolute)) {
			violations.push(`${safePath}: required public shell document is missing`);
			continue;
		}
		const extracted = extractFences(readFileSync(absolute, "utf8"), safePath);
		for (const item of extracted.failures) {
			violations.push(`${safePath}:${item.line}: ${item.message}`);
		}
		const bashFences = extracted.fences.filter((fence) => isBashFence(fence.infoString));
		const fence = bashFences[required.fenceIndex];
		if (fence === undefined) {
			violations.push(`${safePath}: missing public bash fence #${required.fenceIndex}`);
			continue;
		}
		for (const probe of required.probes) {
			if (fence.body.includes(probe)) continue;
			const elsewhere = bashFences.findIndex((other, index) => index !== required.fenceIndex && other.body.includes(probe));
			violations.push(
				elsewhere === -1
					? `${safePath}: bash fence #${required.fenceIndex} missing probe ${JSON.stringify(probe)}`
					: `${safePath}: probe ${JSON.stringify(probe)} split out of bash fence #${required.fenceIndex} into bash fence #${elsewhere}`,
			);
		}
	}
	return violations.sort();
}

function sortFailures(failures: readonly SnippetFailure[]): SnippetFailure[] {
	return [...failures].sort((a, b) => a.docPath.localeCompare(b.docPath) || a.line - b.line || (a.column ?? 0) - (b.column ?? 0) || (a.code ?? "").localeCompare(b.code ?? "") || a.message.localeCompare(b.message));
}

function renderFailure(item: SnippetFailure): string {
	const tag = item.tool === "rustc" ? "rust-doctest" : item.tool === "tsc" ? "ts-ext-snippet" : item.tool === "extract" ? "extract" : `${item.lane}-lane-env`;
	const code = item.code === undefined ? "" : ` ${item.code}`;
	return `[${tag}] ${item.docPath}:${item.line}${item.column === undefined ? "" : `:${item.column}`}${code} — ${item.message}`;
}

export async function runSnippetHarness(root = REPO_ROOT): Promise<SnippetReport> {
	const absoluteRoot = resolve(root);
	const collected = collectDocFences(absoluteRoot);
	const fixtureViolations = verifyNoExcludedExampleProducts(absoluteRoot).map((message) => failure(FIXTURE_ROOT, 1, "env", "env", message));
	const registryViolations = verifyRequiredSnippetFixtures(absoluteRoot).map((message) => failure(FIXTURE_ROOT, 1, "env", "env", message));
	const shellRegistryViolations = verifyPublicShellSnippets(absoluteRoot).map((message) => failure("docs", 1, "env", "env", message));
	const entrypointViolations = verifyShippedEntrypointsExist(absoluteRoot).map((message) => failure("packages", 1, "env", "env", message));
	const examplesViolations = verifyNoExamplesDirectory(absoluteRoot).map((message) => failure("tree", 1, "env", "env", message));
	const importResolutionViolations = assertFenceImportsResolve(absoluteRoot).map((message) => failure(FIXTURE_ROOT, 1, "env", "env", message));
	const [rust, ts] = await Promise.all([runRustLane(absoluteRoot, collected.fences), runTypeScriptLane(absoluteRoot, collected.fences)]);
	const failures = sortFailures([...collected.failures, ...fixtureViolations, ...registryViolations, ...shellRegistryViolations, ...entrypointViolations, ...examplesViolations, ...importResolutionViolations, ...rust.failures, ...ts.failures]);
	const violations = failures.map(renderFailure).sort();
	return { ok: violations.length === 0, lanes: { rust, ts }, violations };
}

async function main(): Promise<void> {
	const report = await runSnippetHarness(REPO_ROOT);
	if (!report.ok) {
		console.error(`snippet harness failed with ${report.violations.length} violation(s):`);
		for (const violation of report.violations) console.error(`  - ${violation}`);
		process.exit(1);
	}
	process.stdout.write("SNIPPET_HARNESS_OK\n");
}

if (import.meta.main) await main();
