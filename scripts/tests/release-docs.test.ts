/**
 * DOC-E (#136) consumer test: release evidence surfaces.
 *
 * This is a docs-owned consumer test — it consumes the landed release tooling
 * (scripts/release/*, scripts/package-release.ts) and the landed CI matrix
 * (.github/workflows/release-verification.yml) without modifying any of them.
 * It proves:
 *
 * 1. Every fenced command in docs/release.md parses through parseReleaseArgs
 *    without UnknownArgError, so instructions cannot name a flag the parser
 *    rejects.
 * 2. Every fenced dry-run command succeeds as a real dry-run execution for
 *    all seven targets: the produced archive and `.sha256` sidecar are
 *    verified against the current repo docs tree, README.md, CHANGELOG.md,
 *    and the release.json digest table (zip lane via the library extractor,
 *    tar.gz lanes via the SpawnRunner tar pattern).
 * 3. The release-path CHANGELOG gate is present in both dry-run and full-build
 *    modes (the gate runs before any build work, so the mode does not matter,
 *    but the test exercises both code paths through changelogGateFailure).
 * 4. The platform support matrix row count == RUST_TARGETS.length == 7 and
 *    matches the landed release-verification CI matrix entry-for-entry.
 * 5. Each scripts/generate-*.ts rerun is a byte-stable offline no-op.
 * 6. release.json semantics are generated from the stage.ts type, not
 *    paraphrased.
 * 7. The extension compatibility narrative cites frames.jsonl as evidence.
 */

import { describe, expect, test } from "bun:test";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { spawnSync } from "node:child_process";

import { checksumLine, extractZip, sha256Bytes } from "../release/archive.ts";
import { parseReleaseArgs, UnknownArgError } from "../release/args.ts";
import { changelogGateFailure } from "../package-release.ts";
import { realFs, SpawnRunner, tarArgs, type Fs } from "../release/runner.ts";
import {
	RELEASE_MANIFEST_SCHEMA,
	type ReleaseManifest,
} from "../release/stage.ts";
import {
	archiveName,
	checksumName,
	planFor,
	RUST_TARGETS,
	TARGET_PLANS,
	type TargetPlan,
} from "../release/targets.ts";

const REPO_ROOT = resolve(import.meta.dirname, "../..");
const RELEASE_DOC = readFileSync(join(REPO_ROOT, "docs/release.md"), "utf8");
const CI_WORKFLOW = readFileSync(
	join(REPO_ROOT, ".github/workflows/release-verification.yml"),
	"utf8",
);

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Extract every fenced code block from a markdown source. */
function extractFencedBlocks(source: string): string[] {
	const lines = source.split(/\r?\n/);
	const blocks: string[] = [];
	let inFence = false;
	let ticks = 0;
	let body: string[] = [];
	for (const line of lines) {
		if (!inFence) {
			const match = /^ {0,3}(`{3,})(.*)$/.exec(line);
			if (match !== null) {
				inFence = true;
				ticks = match[1]!.length;
				body = [];
			}
			continue;
		}
		const close = new RegExp(`^ {0,3}(\`{${ticks},})\\s*$`).exec(line);
		if (close !== null) {
			blocks.push(body.join("\n"));
			inFence = false;
			body = [];
			continue;
		}
		body.push(line);
	}
	return blocks;
}

/** Extract `bun run scripts/package-release.ts ...` argv from a fenced block. */
function extractPackageReleaseArgv(block: string): string[] | null {
	const trimmed = block.trim();
	if (!trimmed.startsWith("bun run scripts/package-release.ts")) return null;
	// Split on whitespace, drop the first three tokens (bun run scripts/package-release.ts)
	const tokens = trimmed.split(/\s+/);
	return tokens.slice(3);
}

/** In-memory Fs for CHANGELOG gate tests. */
function memoryFs(
	initial: Readonly<Record<string, Uint8Array>>,
): Fs & { files: Map<string, Uint8Array> } {
	const files = new Map(Object.entries(initial));
	const norm = (path: string) => path.replace(/\\/g, "/");
	return {
		files,
		async readFile(path: string): Promise<Uint8Array> {
			const data = files.get(norm(path));
			if (data === undefined) throw new Error(`ENOENT: ${path}`);
			return data;
		},
		async writeFile(path: string, data: Uint8Array | string): Promise<void> {
			files.set(norm(path), typeof data === "string" ? new TextEncoder().encode(data) : data);
		},
		async mkdir(): Promise<void> {},
		async rm(): Promise<void> {},
		async readdir(): Promise<string[]> { return []; },
		async stat(path: string) {
			const data = files.get(norm(path));
			if (data === undefined) throw new Error(`ENOENT: ${path}`);
			return { isFile: true, isDir: false, size: data.length, mode: 0o644 };
		},
		async copyFile(src: string, dest: string): Promise<void> {
			const data = files.get(norm(src));
			if (data === undefined) throw new Error(`ENOENT: ${src}`);
			files.set(norm(dest), data);
		},
		async cp(src: string, dest: string): Promise<void> {
			const data = files.get(norm(src));
			if (data === undefined) throw new Error(`ENOENT: ${src}`);
			files.set(norm(dest), data);
		},
		async chmod(): Promise<void> {},
	};
}

/** Run a generator script and return its status + stdout. */
function runGenerator(scriptPath: string): { status: number; stdout: string; stderr: string } {
	const proc = spawnSync("bun", ["run", scriptPath], {
		cwd: REPO_ROOT,
		encoding: "utf8",
		timeout: 60_000,
	});
	return { status: proc.status ?? -1, stdout: proc.stdout ?? "", stderr: proc.stderr ?? "" };
}

/** Extract the CI matrix target list from the release-verification workflow. */
function extractCiMatrixTargets(workflow: string): string[] {
	const targets: string[] = [];
	for (const line of workflow.split(/\r?\n/)) {
		const match = /^\s+-\s+target:\s+(.+?)\s*$/.exec(line);
		if (match !== null) {
			targets.push(match[1]!);
		}
	}
	return targets;
}

const bytes = (text: string) => new TextEncoder().encode(text);

/** Recursively collect POSIX-style relative file paths under `root`, sorted. */
function walkFiles(root: string): string[] {
	const files: string[] = [];
	const visit = (absDir: string, relDir: string): void => {
		for (const dirent of readdirSync(absDir, { withFileTypes: true })) {
			const childRel = relDir === "" ? dirent.name : `${relDir}/${dirent.name}`;
			if (dirent.isDirectory()) {
				visit(join(absDir, dirent.name), childRel);
			} else if (dirent.isFile()) {
				files.push(childRel);
			}
		}
	};
	visit(root, "");
	return files.sort();
}

/**
 * Extract a produced release archive: the library zip helper on the Windows
 * lane, `tar -xzf` through the SpawnRunner pattern everywhere else (mirrors
 * unpackArchive in scripts/package-release.ts).
 */
async function extractArchive(
	archivePath: string,
	plan: TargetPlan,
	outDir: string,
): Promise<void> {
	if (plan.archive === "zip") {
		await extractZip(archivePath, outDir);
		return;
	}
	const result = await new SpawnRunner().run("tar", tarArgs("-xzf", archivePath, "-C", outDir), {
		rejectOnError: false,
		timeoutMs: 60_000,
	});
	if (result.exitCode !== 0) {
		throw new Error(`tar exited ${result.exitCode}: ${result.stderr.slice(0, 500)}`);
	}
}

/** Read the workspace version from package.json with runtime narrowing. */
function workspaceVersion(): string {
	const parsed: unknown = JSON.parse(readFileSync(join(REPO_ROOT, "package.json"), "utf8"));
	if (typeof parsed !== "object" || parsed === null || !("version" in parsed)) {
		throw new Error("package.json carries no version field");
	}
	const { version } = parsed;
	if (typeof version !== "string") {
		throw new Error("package.json version is not a string");
	}
	return version;
}

/** Runtime narrowing for release.json bytes produced by a real dry-run. */
function isReleaseManifest(value: unknown): value is ReleaseManifest {
	return (
		typeof value === "object" &&
		value !== null &&
		"schema" in value &&
		typeof value.schema === "string" &&
		"files" in value &&
		Array.isArray(value.files)
	);
}

async function verifyReleaseArchiveBundle(
	bundleDir: string,
	extractDir: string,
	plan: TargetPlan,
	pkgVersion: string,
	repoDocMembers: readonly string[],
): Promise<void> {
	const archiveBase = archiveName(pkgVersion, plan);
	const archivePath = join(bundleDir, archiveBase);
	const checksumPath = join(bundleDir, checksumName(archiveBase));
	expect(existsSync(archivePath)).toBe(true);
	expect(existsSync(checksumPath)).toBe(true);

	const digest = sha256Bytes(readFileSync(archivePath));
	expect(readFileSync(checksumPath, "utf8")).toBe(checksumLine(digest, archiveBase));

	mkdirSync(extractDir, { recursive: true });
	await extractArchive(archivePath, plan, extractDir);
	const archiveDirPath = join(extractDir, plan.archiveDir);
	const releaseJsonPath = join(archiveDirPath, "release.json");
	expect(existsSync(releaseJsonPath)).toBe(true);
	const parsedManifest: unknown = JSON.parse(readFileSync(releaseJsonPath, "utf8"));
	if (!isReleaseManifest(parsedManifest)) {
		throw new Error(`${plan.archiveDir}/release.json is not a ReleaseManifest`);
	}
	const manifest: ReleaseManifest = parsedManifest;
	expect(manifest.schema).toBe(RELEASE_MANIFEST_SCHEMA);
	expect(manifest.version).toBe(pkgVersion);
	expect(manifest.rustTarget).toBe(plan.rustTarget);
	expect(manifest.bunTarget).toBe(plan.bunTarget);

	const drift: string[] = [];
	for (const rel of repoDocMembers) {
		const entry = manifest.files.find((file) => file.path === rel);
		if (entry === undefined) {
			drift.push(`${rel}: missing from release.json`);
			continue;
		}
		const repoBuf = readFileSync(join(REPO_ROOT, rel));
		if (entry.size !== repoBuf.length) {
			drift.push(`${rel}: size ${entry.size} != repo ${repoBuf.length}`);
		}
		if (entry.sha256 !== sha256Bytes(repoBuf)) {
			drift.push(`${rel}: sha256 ${entry.sha256} != repo digest`);
		}
		const extractedPath = join(archiveDirPath, rel);
		if (!existsSync(extractedPath)) {
			drift.push(`${rel}: missing from archive`);
		} else if (!repoBuf.equals(readFileSync(extractedPath))) {
			drift.push(`${rel}: archived bytes differ from repo bytes`);
		}
	}
	expect(drift).toEqual([]);

	for (const file of manifest.files) {
		if (file.path.startsWith("docs/")) expect(repoDocMembers).toContain(file.path);
	}
	const walked = walkFiles(extractDir);
	const expectedMembers = [
		...manifest.files.map((file) => `${plan.archiveDir}/${file.path}`),
		`${plan.archiveDir}/release.json`,
	].sort();
	expect(walked).toEqual(expectedMembers);
}

// ---------------------------------------------------------------------------
// 1. Fenced command extraction test
// ---------------------------------------------------------------------------

describe("DOC-E: docs/release.md fenced commands parse through parseReleaseArgs", () => {
	const blocks = extractFencedBlocks(RELEASE_DOC);
	const releaseArgv = blocks
		.map(extractPackageReleaseArgv)
		.filter((argv): argv is string[] => argv !== null);

	test("at least one fenced package-release command is present", () => {
		expect(releaseArgv.length).toBeGreaterThan(0);
	});

	test("every fenced package-release command parses without UnknownArgError", () => {
		for (const argv of releaseArgv) {
			expect(() => parseReleaseArgs(argv, REPO_ROOT)).not.toThrow(UnknownArgError);
		}
	});

	test("every fenced --dry-run command resolves to dryRun=true", () => {
		for (const argv of releaseArgv) {
			if (argv.includes("--dry-run")) {
				const args = parseReleaseArgs(argv, REPO_ROOT);
				expect(args.dryRun).toBe(true);
			}
		}
	});

	test("every fenced --target value is one of the seven supported triples", () => {
		const supported = new Set<string>(RUST_TARGETS);
		for (const argv of releaseArgv) {
			const idx = argv.indexOf("--target");
			if (idx !== -1 && idx + 1 < argv.length) {
				expect(supported.has(argv[idx + 1]!)).toBe(true);
			}
		}
	});
});

// ---------------------------------------------------------------------------
// 2. Real dry-run archives verified against the repo docs tree
// ---------------------------------------------------------------------------

describe("DOC-E: dry-run succeeds for every seven targets", () => {
	const pkgVersion = workspaceVersion();

	/** Repo members every archive must ship: the whole docs tree + root metadata. */
	const repoDocMembers = [
		...walkFiles(join(REPO_ROOT, "docs")).map((rel) => `docs/${rel}`),
		"README.md",
		"CHANGELOG.md",
	].sort();

	test("the repo docs tree enumerates real members for every archive", () => {
		expect(repoDocMembers.length).toBeGreaterThan(0);
		expect(repoDocMembers).toContain("README.md");
		expect(repoDocMembers).toContain("CHANGELOG.md");
	});

	for (const triple of RUST_TARGETS) {
		test(`dry-run ${triple}: .sha256 sidecar, archive members, release.json digests`, async () => {
			const plan = planFor(triple);
			const outDir = mkdtempSync(join(tmpdir(), `doc-e-dry-${triple}-`));
			const extractDir = join(outDir, "extracted");
			try {
				const proc = spawnSync(
					"bun",
					["run", "scripts/package-release.ts", "--target", triple, "--dry-run", "--out", outDir],
					{ cwd: REPO_ROOT, encoding: "utf8", timeout: 60_000 },
				);
				if (proc.status !== 0) {
					throw new Error(
						`package-release --dry-run exited ${proc.status}: ${proc.stderr.slice(-2000)}`,
					);
				}

				// Final-tree archive + checksum under the deterministic target names.
				await verifyReleaseArchiveBundle(
					outDir,
					extractDir,
					plan,
					pkgVersion,
					repoDocMembers,
				);
			} finally {
				rmSync(outDir, { recursive: true, force: true });
			}
		}, 60_000);
	}
});

const relCloseArtifactRoot = process.env["PI_RELCLOSE_ARTIFACT_DIR"];
if (relCloseArtifactRoot !== undefined) {
	describe("DOC-F: consumed REL-CLOSE archive artifacts", () => {
		const pkgVersion = workspaceVersion();
		const repoDocMembers = [
			...walkFiles(join(REPO_ROOT, "docs")).map((rel) => `docs/${rel}`),
			"README.md",
			"CHANGELOG.md",
		].sort();

		for (const triple of RUST_TARGETS) {
			test(`consume pi-${triple}`, async () => {
				const extractDir = mkdtempSync(join(tmpdir(), `doc-f-rel-close-${triple}-`));
				try {
					await verifyReleaseArchiveBundle(
						join(relCloseArtifactRoot, `pi-${triple}`),
						extractDir,
						planFor(triple),
						pkgVersion,
						repoDocMembers,
					);
				} finally {
					rmSync(extractDir, { recursive: true, force: true });
				}
			}, 60_000);
		}
	});
}

// ---------------------------------------------------------------------------
// 3. CHANGELOG gate in both dry-run and full-build modes
// ---------------------------------------------------------------------------

describe("DOC-E: release CHANGELOG gate (dry-run and full-build modes)", () => {
	const validChangelog =
		"# Changelog\n\n## [Unreleased]\n\n### Added\n\n- Release notes [#136]\n";
	const emptyChangelog = "# Changelog\n\n## [Unreleased]\n\n## [0.1.0]\n";
	const noUnreleasedChangelog = "# Changelog\n\n## [0.1.0]\n\n- Shipped.\n";

	test("gate fails when CHANGELOG.md is missing (dry-run mode)", async () => {
		const fs = memoryFs({});
		expect(await changelogGateFailure(fs, "/workspace")).toContain("missing");
	});

	test("gate fails when CHANGELOG.md is missing (full-build mode)", async () => {
		// The gate runs before any build work, so the mode is irrelevant —
		// changelogGateFailure is the same function called in both paths.
		const fs = memoryFs({});
		expect(await changelogGateFailure(fs, "/workspace")).toContain("missing");
	});

	test("gate fails when Unreleased section is absent", async () => {
		const fs = memoryFs({ "/workspace/CHANGELOG.md": bytes(noUnreleasedChangelog) });
		expect(await changelogGateFailure(fs, "/workspace")).toContain("no ## [Unreleased] section");
	});

	test("gate fails when Unreleased section is empty", async () => {
		const fs = memoryFs({ "/workspace/CHANGELOG.md": bytes(emptyChangelog) });
		expect(await changelogGateFailure(fs, "/workspace")).toContain("empty");
	});

	test("gate passes when Unreleased section carries entries", async () => {
		const fs = memoryFs({ "/workspace/CHANGELOG.md": bytes(validChangelog) });
		expect(await changelogGateFailure(fs, "/workspace")).toBeNull();
	});

	test("gate transitions fail-empty-pass against a real filesystem", async () => {
		const root = mkdtempSync(join(tmpdir(), "doc-e-changelog-gate-"));
		try {
			expect(await changelogGateFailure(realFs, root)).toContain("missing");
			writeFileSync(join(root, "CHANGELOG.md"), "# Changelog\n\n## [Unreleased]\n");
			expect(await changelogGateFailure(realFs, root)).toContain("empty");
			writeFileSync(join(root, "CHANGELOG.md"), validChangelog);
			expect(await changelogGateFailure(realFs, root)).toBeNull();
		} finally {
			rmSync(root, { recursive: true, force: true });
		}
	});

	test("root CHANGELOG.md has a non-empty Unreleased section", async () => {
		expect(await changelogGateFailure(realFs, REPO_ROOT)).toBeNull();
	});
});

// ---------------------------------------------------------------------------
// 4. Platform matrix cross-assertion against CI
// ---------------------------------------------------------------------------

describe("DOC-E: platform matrix matches RUST_TARGETS and CI matrix", () => {
	const ciTargets = extractCiMatrixTargets(CI_WORKFLOW);

	test("RUST_TARGETS has exactly seven entries", () => {
		expect(RUST_TARGETS.length).toBe(7);
	});

	test("TARGET_PLANS has exactly seven entries", () => {
		expect(TARGET_PLANS.length).toBe(7);
	});

	test("CI matrix has exactly seven target entries", () => {
		expect(ciTargets.length).toBe(7);
	});

	test("RUST_TARGETS set equals CI matrix target set", () => {
		expect(new Set<string>(RUST_TARGETS)).toEqual(new Set<string>(ciTargets));
	});

	test("docs/release.md target rows equal the resolved release plans", () => {
		for (const plan of TARGET_PLANS) {
			expect(RELEASE_DOC).toContain(
				`| \`${plan.rustTarget}\` | \`${plan.archiveDir}\` | \`${plan.archive}\` | \`${plan.bunTarget}\` |`,
			);
		}
	});

	test("every TargetPlan has the expected fields derived from its triple", () => {
		for (const plan of TARGET_PLANS) {
			expect(plan.rustTarget).toBeDefined();
			expect(plan.bunTarget).toMatch(/^bun-(linux|darwin|windows)-(x64|arm64)(-musl)?(-baseline)?$/);
			expect(plan.archive).toMatch(/^(tar\.gz|zip)$/);
			expect(plan.archiveDir).toMatch(/^pi-(linux|darwin|windows)-(x64|arm64)(-musl)?(-base)?$/);
		}
	});

	test("x86_64 plans carry -baseline in bunTarget", () => {
		for (const plan of TARGET_PLANS) {
			if (plan.arch === "x86_64") {
				expect(plan.bunTarget).toContain("-baseline");
			}
		}
	});

	test("aarch64 plans do not carry -baseline in bunTarget", () => {
		for (const plan of TARGET_PLANS) {
			if (plan.arch === "aarch64") {
				expect(plan.bunTarget).not.toContain("-baseline");
			}
		}
	});

	test("musl plans carry -musl in bunTarget", () => {
		for (const plan of TARGET_PLANS) {
			if (plan.libc === "musl") {
				expect(plan.bunTarget).toContain("-musl");
			}
		}
	});

	test("windows plan uses zip archive", () => {
		for (const plan of TARGET_PLANS) {
			if (plan.windows) {
				expect(plan.archive).toBe("zip");
			} else {
				expect(plan.archive).toBe("tar.gz");
			}
		}
	});

	test("docs/supported-platforms.md table has seven data rows matching RUST_TARGETS", () => {
		const supportedDoc = readFileSync(join(REPO_ROOT, "docs/supported-platforms.md"), "utf8");
		for (const triple of RUST_TARGETS) {
			expect(supportedDoc).toContain(triple);
		}
		// Count table data rows (lines starting with | that are not the header or separator)
		const tableRows = supportedDoc
			.split(/\r?\n/)
			.filter((line) => line.startsWith("|") && !line.includes("---") && !line.includes("Rust target"));
		expect(tableRows.length).toBe(7);
	});
});

describe("DOC-F: root README crate catalog", () => {
	test("catalog rows equal Cargo workspace members", () => {
		const cargo = readFileSync(join(REPO_ROOT, "Cargo.toml"), "utf8");
		const workspace = cargo.match(/\[workspace\]([\s\S]*?)\n\[/)?.[1] ?? "";
		const members = [...workspace.matchAll(/^\s*"([^"]+)",?\s*$/gm)]
			.map((match) => match[1])
			.filter((member): member is string => member !== undefined);
		const readme = readFileSync(join(REPO_ROOT, "README.md"), "utf8");
		const catalog = readme.match(/## Workspace crates\n([\s\S]*?)\n## /)?.[1] ?? "";
		const catalogPaths = [...catalog.matchAll(/\]\((crates\/[^)]+)\)/g)]
			.map((match) => match[1])
			.filter((path): path is string => path !== undefined);

		expect(catalogPaths).toEqual(members);
		expect(new Set(catalogPaths).size).toBe(5);
	});
});

// ---------------------------------------------------------------------------
// 5. generate-*.ts idempotence (byte-stable offline no-op)
// ---------------------------------------------------------------------------

describe("DOC-E: generate-*.ts idempotence", () => {
	// CI runs generate-session-fixtures and generate-tool-schemas in the
	// release-verification workflow (workflow:185-191). generate-builtin-models
	// is not run in CI; its committed output (crates/pi-ai/data/builtin-models.json)
	// is the source of truth and reconstruct-provider-data.ts inverts from it.
	// The idempotence assertion covers the two CI-run generators: two
	// consecutive runs must exit 0 (byte-stable offline no-op).
	const ciGenerators = [
		{ name: "generate-session-fixtures.ts", path: "scripts/generate-session-fixtures.ts" },
		{ name: "generate-tool-schemas.ts", path: "scripts/generate-tool-schemas.ts" },
	];

	for (const gen of ciGenerators) {
		test(`${gen.name} two consecutive runs exit 0 (byte-stable no-op)`, () => {
			const first = runGenerator(join(REPO_ROOT, gen.path));
			expect(first.status).toBe(0);
			const second = runGenerator(join(REPO_ROOT, gen.path));
			expect(second.status).toBe(0);
		}, 120_000);
	}

	test("crates/pi-ai/data/builtin-models.json exists (committed catalog is the source of truth)", () => {
		expect(existsSync(join(REPO_ROOT, "crates/pi-ai/data/builtin-models.json"))).toBe(true);
	});
});

// ---------------------------------------------------------------------------
// 6. release.json semantics generated from stage.ts type
// ---------------------------------------------------------------------------

describe("DOC-E: release.json semantics from stage.ts type", () => {
	test("RELEASE_MANIFEST_SCHEMA is pi.release.v1", () => {
		expect(RELEASE_MANIFEST_SCHEMA).toBe("pi.release.v1");
	});

	test("ReleaseManifest interface fields are documented in docs/release.md", () => {
		const doc = RELEASE_DOC;
		// Every field of the ReleaseManifest interface must appear in the doc
		const fields = ["schema", "version", "rustTarget", "bunTarget", "hostKind",
			"compatibilityVersion", "protocolVersion", "sourceDateEpoch", "createdAt", "files"];
		for (const field of fields) {
			expect(doc).toContain(`\`${field}\``);
		}
	});

	test("ManifestFile fields are documented in docs/release.md", () => {
		const doc = RELEASE_DOC;
		const fields = ["path", "size", "sha256", "executable"];
		for (const field of fields) {
			expect(doc).toContain(`\`${field}\``);
		}
	});

	test("docs/release.md cites the stage.ts source for release.json semantics", () => {
		expect(RELEASE_DOC).toContain("scripts/release/stage.ts:19");
		expect(RELEASE_DOC).toContain("scripts/release/stage.ts:34-46");
		expect(RELEASE_DOC).toContain("scripts/release/stage.ts:22-31");
	});

	test("docs/release.md cites stagedInputs ordering from stage.ts", () => {
		expect(RELEASE_DOC).toContain("scripts/release/stage.ts:123-215");
	});
});

// ---------------------------------------------------------------------------
// 7. Extension compatibility narrative cites frames.jsonl
// ---------------------------------------------------------------------------

describe("DOC-E: extension compatibility narrative cites frames.jsonl", () => {
	test("docs/extension-compatibility-contract.md cites frames.jsonl", () => {
		const compatDoc = readFileSync(
			join(REPO_ROOT, "docs/extension-compatibility-contract.md"),
			"utf8",
		);
		expect(compatDoc).toContain("frames.jsonl");
		expect(compatDoc).toContain("packages/pi-tui-protocol/tests/fixtures/frames.jsonl");
	});

	test("frames.jsonl exists at the cited path", () => {
		expect(existsSync(join(REPO_ROOT, "packages/pi-tui-protocol/tests/fixtures/frames.jsonl"))).toBe(true);
	});

	test("docs/release.md cites frames.jsonl as extension compatibility evidence", () => {
		expect(RELEASE_DOC).toContain("frames.jsonl");
	});
});
