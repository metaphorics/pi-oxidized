import { describe, expect, test } from "bun:test";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, unlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import {
	decodeZipArchive,
	extractZip,
	sha256Bytes,
	writeTarGz,
	writeZip,
} from "../release/archive.ts";
import { changelogGateFailure, enforceChangelogGate, smokeUnpacked } from "../package-release.ts";
import {
	helloRequestLine,
	HOST_COMPATIBILITY_VERSION,
	HOST_PROTOCOL_VERSION,
} from "../release/host.ts";
import { RecordingRunner, realFs, safeJoinPath, SpawnRunner, tarArgs, type Fs, type RunResult } from "../release/runner.ts";
import { assembleRelease } from "../release/stage.ts";
import {
	BUN_RUNTIME_VERSION,
	BunRuntimeProvisionError,
	bunRuntimeAsset,
	provisionBunRuntime,
	type RuntimeFetcher,
} from "../release/runtime.ts";
import { planFor, RUST_TARGETS, type TargetPlan } from "../release/targets.ts";

const FILE_STAT = { isFile: true, isDir: false, size: 1, mode: 0o755 } as const;

function existingFilesFs(paths: readonly string[]): Fs {
	const files = new Set(paths);
	return {
		async mkdir() {},
		async rm() {},
		async writeFile(path) {
			files.add(path);
		},
		async readFile() {
			return new Uint8Array();
		},
		async copyFile() {},
		async cp() {},
		async chmod() {},
		async stat(path) {
			if (files.has(path)) return FILE_STAT;
			throw new Error(`ENOENT: ${path}`);
		},
		async readdir() {
			return [];
		},
	};
}

/** In-memory `Fs` fake serving `initial`; writes land in the returned `files`. */
function memoryFs(
	initial: Readonly<Record<string, Uint8Array>>,
): Fs & { files: Map<string, Uint8Array> } {
	const files = new Map(Object.entries(initial));
	return {
		files,
		async mkdir() {},
		async rm() {},
		async writeFile(path, data) {
			files.set(
				path,
				typeof data === "string" ? new TextEncoder().encode(data) : data,
			);
		},
		async readFile(path) {
			const data = files.get(path);
			if (data === undefined) throw new Error(`ENOENT: ${path}`);
			return data;
		},
		async copyFile() {},
		async cp() {},
		async chmod() {},
		async stat(path) {
			if (files.has(path)) return FILE_STAT;
			throw new Error(`ENOENT: ${path}`);
		},
		async readdir() {
			return [...files.keys()];
		},
	};
}

/** Await a promise expected to reject and surface the rejection as an Error. */
async function rejectionOf(promise: Promise<unknown>): Promise<Error> {
	return await promise.then(
		() => {
			throw new Error("expected the promise to reject");
		},
		(reason: unknown) => (reason instanceof Error ? reason : new Error(String(reason))),
	);
}

function helloResult(
	payload: Record<string, unknown> = {
		protocolVersion: HOST_PROTOCOL_VERSION,
		compatibilityVersion: HOST_COMPATIBILITY_VERSION,
	},
	exitCode = 0,
): RunResult {
	return {
		exitCode,
		stdout: `${JSON.stringify({ id: 1, kind: "res", method: "hello", payload })}\n`,
		stderr: exitCode === 0 ? "" : "host crashed",
	};
}

describe("smokeUnpacked", () => {
	test("runs pi --version and a strict compiled-host hello handshake", async () => {
		const plan = planFor("x86_64-unknown-linux-gnu");
		const archiveDir = "/unpacked/pi-linux-x64-base";
		const pi = join(archiveDir, plan.piBinaryName);
		const host = join(archiveDir, plan.hostBinaryName);
		const fs = existingFilesFs([pi, host]);
		const runner = new RecordingRunner((call) => {
			if (call.command === pi) return { exitCode: 0, stdout: "pi 0.1.0\n", stderr: "" };
			if (call.command === host) return helloResult();
			throw new Error(`ENOENT: ${call.command}`);
		});

		await smokeUnpacked({ fs, runner, archiveDir, plan, dryRun: false });

		expect(runner.calls.map(({ command, args }) => [command, args])).toEqual([
			[pi, ["--version"]],
			[host, []],
		]);
		const handshake = runner.calls[1];
		expect(handshake?.options?.stdin).toBe(helloRequestLine());
		expect(JSON.parse(handshake?.options?.stdin?.trim() ?? "{}")).toEqual({
			id: 1,
			kind: "req",
			method: "hello",
			payload: {
				protocolVersion: HOST_PROTOCOL_VERSION,
				compatibilityVersion: HOST_COMPATIBILITY_VERSION,
			},
		});
		expect(runner.calls.every((call) => call.options?.timeoutMs === 30_000)).toBe(true);
	});

	test("smokes the Bun plus JavaScript fallback when the compiled host is absent", async () => {
		const plan = planFor("x86_64-unknown-linux-gnu");
		const archiveDir = "/unpacked/pi-linux-x64-base";
		const pi = join(archiveDir, plan.piBinaryName);
		const runtime = join(archiveDir, plan.bunRuntimeName);
		const script = join(archiveDir, plan.hostBundleName);
		const runner = new RecordingRunner((call) =>
			call.command === pi
				? { exitCode: 0, stdout: "pi 0.1.0\n", stderr: "" }
				: helloResult(),
		);

		await smokeUnpacked({
			fs: existingFilesFs([pi, runtime, script]),
			runner,
			archiveDir,
			plan,
			dryRun: false,
		});

		expect(runner.calls[1]?.command).toBe(runtime);
		expect(runner.calls[1]?.args).toEqual([script]);
		expect(runner.calls[1]?.options?.stdin).toBe(helloRequestLine());
	});

	test("rejects a missing pi binary before spawning", async () => {
		const plan = planFor("x86_64-unknown-linux-gnu");
		const runner = new RecordingRunner(() => {
			throw new Error("must not spawn");
		});
		await expect(
			smokeUnpacked({
				fs: existingFilesFs([]),
				runner,
				archiveDir: "/unpacked/pi-linux-x64-base",
				plan,
				dryRun: false,
			}),
		).rejects.toThrow("missing pi");
		expect(runner.calls).toHaveLength(0);
	});

	test("rejects a nonzero pi --version result", async () => {
		const plan = planFor("x86_64-unknown-linux-gnu");
		const archiveDir = "/unpacked/pi-linux-x64-base";
		const pi = join(archiveDir, plan.piBinaryName);
		const host = join(archiveDir, plan.hostBinaryName);
		const runner = new RecordingRunner(() => ({ exitCode: 7, stdout: "", stderr: "boom" }));
		await expect(
			smokeUnpacked({
				fs: existingFilesFs([pi, host]),
				runner,
				archiveDir,
				plan,
				dryRun: false,
			}),
		).rejects.toThrow("pi --version failed (exit 7)");
		expect(runner.calls).toHaveLength(1);
	});

	test("rejects malformed or incompatible hello acknowledgements", async () => {
		const invalidLines = [
			"not-json",
			JSON.stringify({ kind: "event", method: "hello", payload: {} }),
			JSON.stringify({
				kind: "res",
				method: "hello",
				payload: { protocolVersion: 2, compatibilityVersion: HOST_COMPATIBILITY_VERSION },
			}),
			JSON.stringify({
				kind: "res",
				method: "hello",
				payload: { protocolVersion: HOST_PROTOCOL_VERSION },
			}),
			JSON.stringify({
				kind: "res",
				method: "hello",
				payload: { protocolVersion: HOST_PROTOCOL_VERSION, compatibilityVersion: "wrong" },
			}),
		];
		for (const line of invalidLines) {
			const plan = planFor("x86_64-unknown-linux-gnu");
			const archiveDir = "/unpacked/pi-linux-x64-base";
			const pi = join(archiveDir, plan.piBinaryName);
			const host = join(archiveDir, plan.hostBinaryName);
			const runner = new RecordingRunner((call) =>
				call.command === pi
					? { exitCode: 0, stdout: "pi 0.1.0\n", stderr: "" }
					: { exitCode: 0, stdout: `${line}\n`, stderr: "" },
			);
			await expect(
				smokeUnpacked({
					fs: existingFilesFs([pi, host]),
					runner,
					archiveDir,
					plan,
					dryRun: false,
				}),
			).rejects.toThrow("host hello handshake failed");
		}
	});

	test("rejects a host that acknowledges hello and then exits nonzero", async () => {
		const plan = planFor("x86_64-unknown-linux-gnu");
		const archiveDir = "/unpacked/pi-linux-x64-base";
		const pi = join(archiveDir, plan.piBinaryName);
		const host = join(archiveDir, plan.hostBinaryName);
		const runner = new RecordingRunner((call) =>
			call.command === pi
				? { exitCode: 0, stdout: "pi 0.1.0\n", stderr: "" }
				: helloResult(undefined, 9),
		);
		await expect(
			smokeUnpacked({
				fs: existingFilesFs([pi, host]),
				runner,
				archiveDir,
				plan,
				dryRun: false,
			}),
		).rejects.toThrow("exit 9");
	});

	test("dry-run verifies compiled and fallback layouts without spawning", async () => {
		const plan = planFor("x86_64-unknown-linux-gnu");
		const archiveDir = "/unpacked/pi-linux-x64-base";
		const pi = join(archiveDir, plan.piBinaryName);
		const host = join(archiveDir, plan.hostBinaryName);
		const runtime = join(archiveDir, plan.bunRuntimeName);
		const script = join(archiveDir, plan.hostBundleName);
		for (const paths of [[pi, host], [pi, runtime, script]]) {
			const runner = new RecordingRunner(() => {
				throw new Error("must not spawn");
			});
			await smokeUnpacked({ fs: existingFilesFs(paths), runner, archiveDir, plan, dryRun: true });
			expect(runner.calls).toHaveLength(0);
		}
		await expect(
			smokeUnpacked({
				fs: existingFilesFs([pi, runtime]),
				runner: new RecordingRunner(() => undefined),
				archiveDir,
				plan,
				dryRun: true,
			}),
		).rejects.toThrow("incomplete runtime-bundle fallback");
	});

	test("uses Windows executable names from the target plan", async () => {
		const plan = planFor("x86_64-pc-windows-msvc");
		const archiveDir = "/unpacked/pi-windows-x64-base";
		const pi = join(archiveDir, "pi.exe");
		const host = join(archiveDir, "pi-extension-host.exe");
		const runner = new RecordingRunner((call) =>
			call.command === pi
				? { exitCode: 0, stdout: "pi 0.1.0\n", stderr: "" }
				: helloResult(),
		);
		await smokeUnpacked({
			fs: existingFilesFs([pi, host]),
			runner,
			archiveDir,
			plan,
			dryRun: false,
		});
		expect(runner.calls.map((call) => call.command)).toEqual([pi, host]);
	});
});

describe("pinned Bun runtime provisioning", () => {
	test("maps every release target to a checksum-pinned official asset", () => {
		const expectedFile: Readonly<Record<(typeof RUST_TARGETS)[number], string>> = {
			"x86_64-unknown-linux-gnu": "bun-linux-x64-baseline.zip",
			"x86_64-unknown-linux-musl": "bun-linux-x64-musl-baseline.zip",
			"aarch64-unknown-linux-gnu": "bun-linux-aarch64.zip",
			"aarch64-unknown-linux-musl": "bun-linux-aarch64-musl.zip",
			"x86_64-apple-darwin": "bun-darwin-x64-baseline.zip",
			"aarch64-apple-darwin": "bun-darwin-aarch64.zip",
			"x86_64-pc-windows-msvc": "bun-windows-x64-baseline.zip",
		};
		for (const target of RUST_TARGETS) {
			const plan = planFor(target);
			const asset = bunRuntimeAsset(plan);
			expect(asset.bunTarget).toBe(plan.bunTarget);
			expect(asset.fileName).toBe(expectedFile[target]);
			expect(asset.sha256).toMatch(/^[0-9a-f]{64}$/);
			expect(asset.runtimeMember).toEndWith(`/${plan.bunRuntimeName}`);
			expect(asset.url).toContain(`/bun-v${BUN_RUNTIME_VERSION}/`);
		}
	});

	test("pins the exact musl runtime assets and filenames", () => {
		const x64 = bunRuntimeAsset(planFor("x86_64-unknown-linux-musl"));
		expect(x64.bunTarget).toBe("bun-linux-x64-musl-baseline");
		expect(x64.fileName).toBe("bun-linux-x64-musl-baseline.zip");
		expect(x64.sha256).toBe(
			"56a7d6806cf155536c0178f0ea5fbd098e684fa509ebdb4fc0a7e19fb65382dc",
		);
		expect(x64.runtimeMember).toBe("bun-linux-x64-musl-baseline/bun");
		expect(x64.url).toBe(
			`https://github.com/oven-sh/bun/releases/download/bun-v${BUN_RUNTIME_VERSION}/bun-linux-x64-musl-baseline.zip`,
		);

		const arm64 = bunRuntimeAsset(planFor("aarch64-unknown-linux-musl"));
		expect(arm64.bunTarget).toBe("bun-linux-arm64-musl");
		expect(arm64.fileName).toBe("bun-linux-aarch64-musl.zip");
		expect(arm64.sha256).toBe(
			"b98e0ad3625c5c00d1d5b5ff55605c7adddbfae151861e68ade57b2d3b8703bb",
		);
		expect(arm64.runtimeMember).toBe("bun-linux-aarch64-musl/bun");
		expect(arm64.url).toBe(
			`https://github.com/oven-sh/bun/releases/download/bun-v${BUN_RUNTIME_VERSION}/bun-linux-aarch64-musl.zip`,
		);
	});

	test("rejects downloaded runtime bytes before extraction when checksum differs", async () => {
		const plan = planFor("x86_64-unknown-linux-gnu");
		await expect(
			provisionBunRuntime({
				plan,
				destination: "/tmp/bun",
				fs: existingFilesFs([]),
				fetcher: async () => ({
					ok: true,
					status: 200,
					async arrayBuffer() {
						return Uint8Array.from([1, 2, 3]).buffer;
					},
				}),
			}),
		).rejects.toThrow("checksum mismatch");
	});

	test("installs from a checksum-valid offline cache without fetching", async () => {
		const plan = planFor("x86_64-unknown-linux-gnu");
		const asset = bunRuntimeAsset(plan);
		const work = mkdtempSync(join(tmpdir(), "pi-runtime-cache-"));
		try {
			const zipPath = join(work, asset.fileName);
			const runtimeBytes = new TextEncoder().encode("cached-bun-runtime\n");
			await writeZip(
				[{ path: asset.runtimeMember, data: runtimeBytes, mode: 0o755 }],
				zipPath,
				{ sourceDateEpoch: 0 },
			);
			const fs = memoryFs({
				[safeJoinPath("/cache", asset.fileName)]: new Uint8Array(readFileSync(zipPath)),
			});
			const destination = await provisionBunRuntime({
				plan,
				destination: "/out/bun",
				cacheDir: "/cache",
				fs,
				// A throwing fetcher proves the cache is consulted before any
				// fetch: a cache miss would surface as a wrapped fetch failure.
				fetcher: () => Promise.reject(new Error("network unavailable")),
				// The pinned official archive bytes are unforgeable offline, so
				// the digest seam vouches for the hand-built archive instead.
				digest: () => asset.sha256,
			});
			expect(destination).toBe("/out/bun");
			expect(fs.files.get("/out/bun")).toEqual(runtimeBytes);
		} finally {
			rmSync(work, { recursive: true, force: true });
		}
	});

	test("installs from a checksum-valid offline cache for all seven targets without fetching", async () => {
		for (const target of RUST_TARGETS) {
			const plan = planFor(target);
			const asset = bunRuntimeAsset(plan);
			const work = mkdtempSync(join(tmpdir(), "pi-runtime-cache-7-"));
			try {
				const zipPath = join(work, asset.fileName);
				const runtimeBytes = new TextEncoder().encode(`cached-bun-${target}\n`);
				await writeZip(
					[{ path: asset.runtimeMember, data: runtimeBytes, mode: 0o755 }],
					zipPath,
					{ sourceDateEpoch: 0 },
				);
				const fs = memoryFs({
					[safeJoinPath("/cache", asset.fileName)]: new Uint8Array(readFileSync(zipPath)),
				});
				const destination = await provisionBunRuntime({
					plan,
					destination: "/out/bun",
					cacheDir: "/cache",
					fs,
					fetcher: () => Promise.reject(new Error("network unavailable")),
					digest: () => asset.sha256,
				});
				expect(destination).toBe("/out/bun");
				expect(fs.files.get("/out/bun")).toEqual(runtimeBytes);
			} finally {
				rmSync(work, { recursive: true, force: true });
			}
		}
	});

	test("wraps non-OK and throwing fetch failures with cache path and filename", async () => {
		const plan = planFor("x86_64-unknown-linux-gnu");
		const asset = bunRuntimeAsset(plan);
		const cachePath = safeJoinPath("/cache", asset.fileName);
		const shared = {
			plan,
			destination: "/out/bun",
			cacheDir: "/cache",
			fs: memoryFs({}),
		};

		const nonOk = await rejectionOf(
			provisionBunRuntime({
				...shared,
				fetcher: async () => ({
					ok: false,
					status: 503,
					async arrayBuffer() {
						return new ArrayBuffer(0);
					},
				}),
			}),
		);
		expect(nonOk).toBeInstanceOf(BunRuntimeProvisionError);
		expect(nonOk.message).toContain(cachePath);
		expect(nonOk.message).toContain(asset.fileName);

		const thrown = await rejectionOf(
			provisionBunRuntime({
				...shared,
				fetcher: () => Promise.reject(new Error("DNS resolution failed")),
			}),
		);
		expect(thrown).toBeInstanceOf(BunRuntimeProvisionError);
		expect(thrown.message).toContain(cachePath);
		expect(thrown.message).toContain(asset.fileName);
	});

	test("a corrupted cache entry rejects identically to a corrupted download", async () => {
		const plan = planFor("x86_64-unknown-linux-gnu");
		const asset = bunRuntimeAsset(plan);
		const corrupt = Uint8Array.from([1, 2, 3]);
		const expected = `checksum mismatch for ${asset.fileName}: expected ${asset.sha256}, got ${sha256Bytes(corrupt)}`;
		const fetcher: RuntimeFetcher = async () => ({
			ok: true,
			status: 200,
			async arrayBuffer() {
				return corrupt.buffer;
			},
		});

		const fromCache = await rejectionOf(
			provisionBunRuntime({
				plan,
				destination: "/out/bun",
				cacheDir: "/cache",
				fs: memoryFs({ [safeJoinPath("/cache", asset.fileName)]: corrupt }),
				fetcher,
			}),
		);
		expect(fromCache).toBeInstanceOf(BunRuntimeProvisionError);
		expect(fromCache.message).toBe(expected);

		const fromDownload = await rejectionOf(
			provisionBunRuntime({
				plan,
				destination: "/out/bun",
				fs: existingFilesFs([]),
				fetcher,
			}),
		);
		expect(fromDownload).toBeInstanceOf(BunRuntimeProvisionError);
		expect(fromDownload.message).toBe(expected);
	});
});

describe("portable ZIP validation", () => {
	test("rejects traversal members before extraction", async () => {
		const work = mkdtempSync(join(tmpdir(), "pi-release-zip-"));
		try {
			const archive = join(work, "traversal.zip");
			await writeZip(
				[{ path: "safe.tx", data: new TextEncoder().encode("payload"), mode: 0o644 }],
				archive,
				{ sourceDateEpoch: 0 },
			);
			const bytes = new Uint8Array(readFileSync(archive));
			const archiveBuffer = Buffer.from(bytes.buffer, bytes.byteOffset, bytes.byteLength);
			const safeName = Buffer.from("safe.tx");
			const unsafeName = Buffer.from("../evil");
			for (let offset = archiveBuffer.indexOf(safeName); offset !== -1; ) {
				bytes.set(unsafeName, offset);
				offset = archiveBuffer.indexOf(safeName, offset + safeName.length);
			}
			expect(() => decodeZipArchive(bytes)).toThrow("archive path escapes root");
		} finally {
			rmSync(work, { recursive: true, force: true });
		}
	});
});

describe("release CHANGELOG gate", () => {
	const bytes = (text: string) => new TextEncoder().encode(text);

	test("fails when the root CHANGELOG.md is missing", async () => {
		const failure = await changelogGateFailure(memoryFs({}), "/workspace");
		expect(failure).toContain("CHANGELOG.md");
		expect(failure).toContain("missing");
	});

	test("fails when the Unreleased section is absent", async () => {
		const fs = memoryFs({
			[join("/workspace", "CHANGELOG.md")]: bytes(
				"# Changelog\n\n## [0.1.0] - 2026-01-01\n\n- Shipped something.\n",
			),
		});
		expect(await changelogGateFailure(fs, "/workspace")).toContain(
			"no ## [Unreleased] section",
		);
	});

	test("fails when the Unreleased section carries no entries", async () => {
		const fs = memoryFs({
			[join("/workspace", "CHANGELOG.md")]: bytes(
				"# Changelog\n\n## [Unreleased]\n\n### Added\n\n## [0.1.0] - 2026-01-01\n\n- Shipped something.\n",
			),
		});
		expect(await changelogGateFailure(fs, "/workspace")).toContain(
			"empty ## [Unreleased] section",
		);
	});

	test("passes when the Unreleased section carries entries", async () => {
		const fs = memoryFs({
			[join("/workspace", "CHANGELOG.md")]: bytes(
				"# Changelog\n\n## [Unreleased]\n\n### Added\n\n- Docs staged into release archives [#111]\n\n## [0.1.0] - 2026-01-01\n\n- Shipped something.\n",
			),
		});
		expect(await changelogGateFailure(fs, "/workspace")).toBeNull();
	});

	test("enforceChangelogGate throws the gate reason on failure", async () => {
		const error = await rejectionOf(enforceChangelogGate(memoryFs({}), "/workspace"));
		expect(error.message).toContain("release CHANGELOG gate");
	});

	test("gate transitions fail-empty-pass against a real filesystem", async () => {
		const root = mkdtempSync(join(tmpdir(), "pi-changelog-gate-"));
		try {
			expect(await changelogGateFailure(realFs, root)).toContain("missing");
			writeFileSync(join(root, "CHANGELOG.md"), "# Changelog\n\n## [Unreleased]\n");
			expect(await changelogGateFailure(realFs, root)).toContain("empty");
			writeFileSync(
				join(root, "CHANGELOG.md"),
				"# Changelog\n\n## [Unreleased]\n\n- Release notes [#111]\n",
			);
			expect(await changelogGateFailure(realFs, root)).toBeNull();
		} finally {
			rmSync(root, { recursive: true, force: true });
		}
	});
});

describe("seven-archive release path", () => {
	const EPOCH = 1_700_000_000;

	/** Fixture repo root with every member a release archive must ship. */
	function writeDocsFixtureRepo(root: string, stagingRoot: string, plan: TargetPlan): void {
		mkdirSync(join(root, "target", plan.rustTarget, "release"), { recursive: true });
		writeFileSync(
			join(root, "target", plan.rustTarget, "release", plan.piBinaryName),
			`mock-pi ${plan.rustTarget}\n`,
		);
		mkdirSync(join(stagingRoot, "host"), { recursive: true });
		writeFileSync(join(stagingRoot, "pi-extension-host"), "mock-extension-host\n");
		mkdirSync(join(root, "docs", "guide"), { recursive: true });
		writeFileSync(join(root, "docs", "intro.md"), "# Introduction\n\nShipped docs member.\n");
		writeFileSync(join(root, "docs", "guide", "deep.md"), "# Deep guide\n\nNested docs member.\n");
		writeFileSync(
			join(root, "CHANGELOG.md"),
			"# Changelog\n\n## [Unreleased]\n\n- Docs staged into release archives [#111]\n",
		);
		writeFileSync(join(root, "README.md"), "# pi\n\nRust port of the pi coding agent.\n");
	}

	async function assembleFixture(root: string, stagingRoot: string, plan: TargetPlan) {
		return assembleRelease(stagingRoot, {
			plan,
			version: "0.0.0-gate",
			piBinaryPath: join(root, "target", plan.rustTarget, "release", plan.piBinaryName),
			repoRoot: root,
			host: { kind: "compiled", binaryPath: join(stagingRoot, "pi-extension-host") },
			fs: realFs,
			sourceDateEpoch: EPOCH,
			compatibilityVersion: HOST_COMPATIBILITY_VERSION,
			protocolVersion: HOST_PROTOCOL_VERSION,
			createdAt: new Date(EPOCH * 1000).toISOString(),
			docsSource: join(root, "docs"),
		});
	}

	/** Pack the assembled tree exactly like package-release main() and extract it. */
	async function packAndExtract(
		root: string,
		stagingRoot: string,
		plan: TargetPlan,
	): Promise<string> {
		const assembly = await assembleFixture(root, stagingRoot, plan);
		const entries: { path: string; data: Uint8Array; mode: number }[] = [];
		for (const file of assembly.manifest.files) {
			entries.push({
				path: `${plan.archiveDir}/${file.path}`,
				data: new Uint8Array(await realFs.readFile(join(assembly.stagingDir, file.path))),
				mode: file.executable ? 0o755 : 0o644,
			});
		}
		entries.push({
			path: `${plan.archiveDir}/release.json`,
			data: new Uint8Array(await realFs.readFile(join(assembly.stagingDir, "release.json"))),
			mode: 0o644,
		});
		const archivePath = join(stagingRoot, plan.archive === "zip" ? "out.zip" : "out.tar.gz");
		const archiveOpts = { sourceDateEpoch: EPOCH };
		if (plan.archive === "zip") {
			await writeZip(entries, archivePath, archiveOpts);
		} else {
			await writeTarGz(entries, archivePath, archiveOpts);
		}
		const extractDir = join(stagingRoot, "extracted");
		await realFs.mkdir(extractDir, { recursive: true });
		if (plan.archive === "zip") {
			await extractZip(archivePath, extractDir);
		} else {
			const tar = await new SpawnRunner().run(
				"tar",
				tarArgs("-xzf", archivePath, "-C", extractDir),
				{ rejectOnError: false, timeoutMs: 30_000 },
			);
			if (tar.exitCode !== 0) {
				throw new Error(`tar exited ${tar.exitCode}: ${tar.stderr.slice(0, 500)}`);
			}
		}
		return extractDir;
	}

	const readExtracted = (dir: string, rel: string): Uint8Array =>
		new Uint8Array(readFileSync(join(dir, rel)));

	test("every archive ships docs, README, CHANGELOG, and a digest-matching release.json", async () => {
		for (const target of RUST_TARGETS) {
			const plan = planFor(target);
			const root = mkdtempSync(join(tmpdir(), `pi-rel-docs-${target}-`));
			const stagingRoot = join(root, "staging");
			writeDocsFixtureRepo(root, stagingRoot, plan);
			try {
				const archiveRoot = join(await packAndExtract(root, stagingRoot, plan), plan.archiveDir);
				const releaseJson = JSON.parse(
					new TextDecoder().decode(readExtracted(archiveRoot, "release.json")),
				) as { files: { path: string; size: number; sha256: string; executable: boolean }[] };
				const byPath = new Map(releaseJson.files.map((file) => [file.path, file]));
				for (const member of [
					"CHANGELOG.md",
					"README.md",
					"docs/intro.md",
					"docs/guide/deep.md",
				]) {
					const entry = byPath.get(member);
					if (entry === undefined) {
						throw new Error(`${target}: release.json digest table omits ${member}`);
					}
					const memberBytes = readExtracted(archiveRoot, member);
					expect(sha256Bytes(memberBytes)).toBe(entry.sha256);
					expect(memberBytes.length).toBe(entry.size);
					expect(entry.executable).toBe(false);
				}
			} finally {
				rmSync(root, { recursive: true, force: true });
			}
		}
	}, 60_000);

	test("deleting one docs source file drops exactly that member from archive and digest table", async () => {
		const plan = planFor("x86_64-unknown-linux-gnu");
		const root = mkdtempSync(join(tmpdir(), "pi-rel-docs-mutation-"));
		const stagingRoot = join(root, "staging");
		writeDocsFixtureRepo(root, stagingRoot, plan);
		try {
			const before = await assembleFixture(root, stagingRoot, plan);
			expect(before.manifest.files.find((file) => file.path === "docs/guide/deep.md")).toBeDefined();
			unlinkSync(join(root, "docs", "guide", "deep.md"));
			// package-release main() always stages into a fresh staging root;
			// mirror that guarantee before re-assembling the mutated tree.
			rmSync(stagingRoot, { recursive: true, force: true });
			mkdirSync(join(stagingRoot, "host"), { recursive: true });
			writeFileSync(join(stagingRoot, "pi-extension-host"), "mock-extension-host\n");

			const archiveRoot = join(await packAndExtract(root, stagingRoot, plan), plan.archiveDir);
			const releaseJson = JSON.parse(
				new TextDecoder().decode(readExtracted(archiveRoot, "release.json")),
			) as { files: { path: string; sha256: string }[] };
			const paths = releaseJson.files.map((file) => file.path);
			expect(paths).not.toContain("docs/guide/deep.md");
			expect(paths).toContain("docs/intro.md");
			const intro = releaseJson.files.find((file) => file.path === "docs/intro.md");
			if (intro === undefined) throw new Error("digest table lost docs/intro.md");
			expect(sha256Bytes(readExtracted(archiveRoot, "docs/intro.md"))).toBe(intro.sha256);
		} finally {
			rmSync(root, { recursive: true, force: true });
		}
	}, 30_000);
});
