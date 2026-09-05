#!/usr/bin/env bun
/**
 * Master release script for pi-oxidized.
 *
 * Drives the cross-target release assembly defined in the master plan:
 *   1. `cargo build -p pi --release --locked --target <triple>`
 *   2. Host compile + runtime-import fixture + hello handshake (via host.ts).
 *   3. Artifact assembly (binary, host, docs, assets) into a staging tree.
 *   4. Verification (target agreement, executable bits, no host bytes in pi).
 *   5. Deterministic `.tar.gz` or `.zip` creation with `.sha256` sidecar.
 *   6. Copy artifacts into the final output directory and run unpack smoke.
 *
 * Usage:
 *   bun run scripts/package-release.ts --target <triple> [--out <dir>] [--dry-run]
 *
 *   --target <triple>        one of the seven supported Rust triples
 *   --out / --out-dir <dir>  output directory (default: <cwd>/dist/release)
 *   --dry-run                skip cargo + host build; assemble from stub binaries
 *   --no-cargo               skip cargo build, but still compile host and archive
 *   --no-handshake           skip the host `hello` handshake verification
 *   --source-date-epoch <s>  override SOURCE_DATE_EPOCH for archive mtimes
 *   --runtime-cache <dir>   offline cache for pinned Bun runtime assets
 *
 * Verification check 13 calls for, from the unpacked archive:
 *   - `pi --version` (binary runs and reports the workspace version)
 *   - host `hello` handshake (compiled sidecar responds with the expected version)
 *
 * `smokeUnpacked` unpacks the finalized archive into a fresh directory before
 * running any check so archive corruption, missing files, or wrong target
 * fail the release.
 */

import { mkdir, rm } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";

import { checksumLine, extractZip, sha256Bytes, writeTarGz, writeZip } from "./release/archive.ts";
import { parseReleaseArgs } from "./release/args.ts";
import {
	buildFallbackBundle,
	buildHost,
	helloRequestLine,
	HOST_COMPATIBILITY_VERSION,
	HOST_PROTOCOL_VERSION,
	isHelloAckLine,
} from "./release/host.ts";
import { pathExists, realFs, SpawnRunner, tarArgs, type CommandRunner, type Fs } from "./release/runner.ts";
import { provisionBunRuntime } from "./release/runtime.ts";
import { assembleRelease } from "./release/stage.ts";
import { archiveName, type TargetPlan } from "./release/targets.ts";
import { verifyUnpackedArchive } from "./release/verify.ts";

const CARGO_BUILD_TIMEOUT_MS = 30 * 60_000;
const ARCHIVE_TOOL_TIMEOUT_MS = 2 * 60_000;
const SMOKE_TIMEOUT_MS = 30_000;

/** The section a releasable CHANGELOG must carry with at least one entry. */
const UNRELEASED_SECTION = "## [Unreleased]";

/**
 * Release-path gate: return the failure reason when the root CHANGELOG.md is
 * missing or its `## [Unreleased]` section carries no entries, else `null`.
 * Enforced for every build mode (dry-run, no-cargo, full) before any build
 * work starts, so no release is cut without release notes.
 */
export async function changelogGateFailure(fs: Fs, repoRoot: string): Promise<string | null> {
	const path = join(repoRoot, "CHANGELOG.md");
	let text: string;
	try {
		text = new TextDecoder().decode(await fs.readFile(path));
	} catch {
		return (
			`release CHANGELOG gate: ${path} is missing; every release build ` +
			`(dry-run and full) requires a CHANGELOG.md with a non-empty ${UNRELEASED_SECTION} section`
		);
	}
	const lines = text.split(/\r?\n/);
	const start = lines.findIndex((line) => line.trim() === UNRELEASED_SECTION);
	if (start === -1) {
		return (
			`release CHANGELOG gate: ${path} has no ${UNRELEASED_SECTION} section; ` +
			`add release notes before building a release`
		);
	}
	const section = lines.slice(start + 1);
	const end = section.findIndex((line) => line.startsWith("## "));
	const body = end === -1 ? section : section.slice(0, end);
	const hasEntries = body.some((line) => {
		const trimmed = line.trim();
		return trimmed.length > 0 && !trimmed.startsWith("#");
	});
	if (!hasEntries) {
		return (
			`release CHANGELOG gate: ${path} has an empty ${UNRELEASED_SECTION} section; ` +
			`add release notes before building a release`
		);
	}
	return null;
}

/** Fail the build when the root CHANGELOG gate trips. */
export async function enforceChangelogGate(fs: Fs, repoRoot: string): Promise<void> {
	const failure = await changelogGateFailure(fs, repoRoot);
	if (failure !== null) throw new Error(failure);
}


async function main(): Promise<void> {
	const args = parseReleaseArgs(process.argv.slice(2));
	const repoRoot = resolve(import.meta.dirname, "..");
	const runner = new SpawnRunner();
	const fs = realFs;

	process.stdout.write(`=== Building release for ${args.plan.rustTarget} ===\n`);
	process.stdout.write(`Bun host target: ${args.plan.bunTarget}\n`);
	process.stdout.write(`Archive format:  ${args.plan.archive}\n`);
	process.stdout.write(`Output dir:      ${args.outDir}\n`);
	process.stdout.write(
		`Mode:            ${args.dryRun ? "dry-run" : args.noCargo ? "no-cargo" : "full"}\n\n`,
	);

	// Release CHANGELOG gate: fails every build mode (dry-run and full)
	// before any build work when release notes are missing or empty.
	await enforceChangelogGate(fs, repoRoot);

	const stagingRoot = join(args.outDir, `.staging-release-${args.plan.rustTarget}`);
	await fs.mkdir(stagingRoot, { recursive: true });

	const piBinaryPath = args.dryRun
		? join(stagingRoot, "pi-mock")
		: join(repoRoot, "target", args.plan.rustTarget, "release", args.plan.piBinaryName);

	try {
		if (args.dryRun) {
			await fs.writeFile(
				piBinaryPath,
				`mock-pi ${args.plan.rustTarget} source-date-epoch=${args.sourceDateEpoch}\n`,
			);
			process.stdout.write(`[1/6] Skipping cargo build (dry-run).\n`);
		} else if (args.noCargo) {
			if (!(await pathExists(fs, piBinaryPath))) {
				throw new Error(
					`--no-cargo requires a pre-built binary at ${piBinaryPath}. ` +
						`Run \`cargo build -p pi --release --locked --target ${args.plan.rustTarget}\` first.`,
				);
			}
			process.stdout.write(`[1/6] Using pre-built cargo artifact at ${piBinaryPath}.\n`);
		} else {
			process.stdout.write(`[1/6] Running cargo build --target ${args.plan.rustTarget}...\n`);
			const cargoRes = await runner.run(
				"cargo",
				["build", "-p", "pi", "--release", "--locked", "--target", args.plan.rustTarget],
				{ cwd: repoRoot, timeoutMs: CARGO_BUILD_TIMEOUT_MS },
			);
			if (cargoRes.exitCode !== 0) {
				throw new Error(`Cargo build failed: ${cargoRes.stderr.slice(0, 2000)}`);
			}
		}

		// 2. Host sidecar.
		process.stdout.write(`[2/6] Building extension host sidecar...\n`);
		const host = args.dryRun
			? {
					kind: "compiled" as const,
					binaryPath: join(stagingRoot, args.plan.hostBinaryName),
				}
			: await buildHost({
					repoRoot,
					stagingRoot,
					plan: args.plan,
					skipRuntimeImport: false,
					skipHandshake: !args.handshake,
					runner,
				});

		if (args.dryRun && host.kind === "compiled") {
			await fs.writeFile(
				host.binaryPath,
				`mock-host ${args.plan.bunTarget} source-date-epoch=${args.sourceDateEpoch}\n`,
			);
		}

		// A runtime-bundle is self-contained: provision the exact pinned Bun
		// executable at the path returned by the host builder.
		let bunRuntimePath: string | undefined;
		if (host.kind === "runtime-bundle") {
			bunRuntimePath = host.runtimePath;
			if (args.dryRun) {
				await fs.writeFile(
					bunRuntimePath,
					`mock-bun-runtime ${args.plan.bunTarget} source-date-epoch=${args.sourceDateEpoch}\n`,
				);
			} else {
				process.stdout.write(`  Provisioning checksum-verified Bun runtime ${args.plan.bunTarget}...\n`);
				await provisionBunRuntime({
					plan: args.plan,
					destination: bunRuntimePath,
					cacheDir: args.runtimeCache,
					fs,
				});
			}
		}
		// Musl rows ship both host execution paths: the compiled sidecar is
		// the primary resolver path, and the pinned musl Bun runtime plus JS
		// bundle ride along as the fallback (REL-T4 drives both `hello`
		// protocols from the same unpacked archive).
		let fallbackBundle:
			| { readonly scriptPath: string; readonly bunRuntimePath: string }
			| undefined;
		if (args.plan.libc === "musl") {
			if (host.kind !== "compiled") {
				throw new Error(
					`musl release for ${args.plan.rustTarget} requires a compiled sidecar; ` +
						`host build produced ${host.kind}`,
				);
			}
			const bunRuntimePath = join(
				stagingRoot,
				"host",
				args.plan.rustTarget,
				args.plan.bunRuntimeName,
			);
			let scriptPath: string;
			await fs.mkdir(dirname(bunRuntimePath), { recursive: true });
			if (args.dryRun) {
				scriptPath = join(stagingRoot, args.plan.hostBundleName);
				await fs.writeFile(
					scriptPath,
					`mock-host-bundle ${args.plan.bunTarget} source-date-epoch=${args.sourceDateEpoch}\n`,
				);
				await fs.writeFile(
					bunRuntimePath,
					`mock-bun-runtime ${args.plan.bunTarget} source-date-epoch=${args.sourceDateEpoch}\n`,
				);
			} else {
				scriptPath = (
					await buildFallbackBundle({
						repoRoot,
						stagingRoot,
						plan: args.plan,
						runner,
					})
				).scriptPath;
				process.stdout.write(
					`  Provisioning checksum-verified musl Bun runtime ${args.plan.bunTarget}...\n`,
				);
				await provisionBunRuntime({
					plan: args.plan,
					destination: bunRuntimePath,
					cacheDir: args.runtimeCache,
					fs,
				});
			}
			fallbackBundle = { scriptPath, bunRuntimePath };
		}

		// 3. Assembly & verification.
		process.stdout.write(`[3/6] Assembling artifacts and verifying invariants...\n`);
		const pkgJson = JSON.parse(
			new TextDecoder().decode(await fs.readFile(join(repoRoot, "package.json"))),
		) as { version: string };

		const assembly = await assembleRelease(stagingRoot, {
			plan: args.plan,
			version: pkgJson.version,
			piBinaryPath,
			repoRoot,
			host,
			bunRuntimePath,
			fallbackBundle,
			fs,
			sourceDateEpoch: parseInt(args.sourceDateEpoch, 10),
			compatibilityVersion: HOST_COMPATIBILITY_VERSION,
			protocolVersion: HOST_PROTOCOL_VERSION,
			createdAt: new Date(parseInt(args.sourceDateEpoch, 10) * 1000).toISOString(),
			docsSource: join(repoRoot, "docs"),
			assetsSource: join(repoRoot, "crates", "pi", "assets"),
		});

		// 4. Deterministic archive.
		process.stdout.write(`[4/6] Creating deterministic archive...\n`);
		const archiveBase = archiveName(pkgJson.version, args.plan);
		const archivePath = join(stagingRoot, archiveBase);
		const entries: { path: string; data: Uint8Array; mode: number }[] = [];
		for (const file of assembly.manifest.files) {
			const data = await fs.readFile(join(assembly.stagingDir, file.path));
			entries.push({
				path: `${args.plan.archiveDir}/${file.path}`,
				data,
				mode: file.executable ? 0o755 : 0o644,
			});
		}
		entries.push({
			path: `${args.plan.archiveDir}/release.json`,
			data: await fs.readFile(join(assembly.stagingDir, "release.json")),
			mode: 0o644,
		});
		const archiveOpts = { sourceDateEpoch: parseInt(args.sourceDateEpoch, 10) };
		if (args.plan.archive === "zip") {
			await writeZip(entries, archivePath, archiveOpts);
		} else {
			await writeTarGz(entries, archivePath, archiveOpts);
		}

		// 5. Checksum sidecar.
		const archiveBytes = await fs.readFile(archivePath);
		const digest = sha256Bytes(archiveBytes);
		const checksumPath = join(stagingRoot, `${archiveBase}.sha256`);
		await fs.writeFile(checksumPath, checksumLine(digest, archiveBase));

		// 6. Finalization.
		process.stdout.write(`[5/6] Finalizing release in ${args.outDir}...\n`);
		await fs.mkdir(args.outDir, { recursive: true });
		const finalArchivePath = join(args.outDir, archiveBase);
		const finalChecksumPath = join(args.outDir, `${archiveBase}.sha256`);
		await fs.copyFile(archivePath, finalArchivePath);
		await fs.copyFile(checksumPath, finalChecksumPath);

		// 7. Unpack smoke (verification check 13). We always extract the
		// finalized archive so a wrong target, missing host, or empty pi
		// binary fails the release.
		process.stdout.write(`[6/6] Running unpack smoke checks...\n`);
		const smokeRoot = join(args.outDir, `.smoke-${args.plan.rustTarget}`);
		await rm(smokeRoot, { recursive: true, force: true });
		await mkdir(smokeRoot, { recursive: true });
		try {
			await unpackArchive(archivePath, args.plan, smokeRoot, runner);
			const archiveDir = join(smokeRoot, args.plan.archiveDir);
			const integrity = await verifyUnpackedArchive(archiveDir);
			if (!integrity.ok) {
				throw new Error(
					`archive-integrity verification failed:\n${integrity.errors.map((e) => `  ${e}`).join("\n")}`,
				);
			}
			process.stdout.write(
				`  archive-integrity: ${integrity.fileCount} members verified against release.json\n`,
			);
			await smokeUnpacked({
				fs,
				runner,
				archiveDir,
				plan: args.plan,
				dryRun: args.dryRun,
			});
		} finally {
			await rm(smokeRoot, { recursive: true, force: true });
		}

		process.stdout.write(`\n=== Release Complete ===\n`);
		process.stdout.write(`Archive:  ${finalArchivePath}\n`);
		process.stdout.write(`Checksum: ${finalChecksumPath}\n`);
	} finally {
		await fs.rm(stagingRoot, { recursive: true, force: true });
	}
}

/** Inputs to {@link smokeUnpacked}. */
interface SmokeOptions {
	readonly fs: Fs;
	readonly runner: CommandRunner;
	/** Absolute path to the unpacked archive root (containing `pi`, etc.). */
	readonly archiveDir: string;
	readonly plan: TargetPlan;
	readonly dryRun: boolean;
}

/** Extract the finalized archive into the smoke directory. */
async function unpackArchive(
	archivePath: string,
	plan: TargetPlan,
	smokeRoot: string,
	runner: CommandRunner,
): Promise<void> {
	if (plan.archive === "zip") {
		await extractZip(archivePath, smokeRoot);
		return;
	}
	const result = await runner.run("tar", tarArgs("-xzf", archivePath, "-C", smokeRoot), {
		rejectOnError: false,
		timeoutMs: ARCHIVE_TOOL_TIMEOUT_MS,
	});
	if (result.exitCode !== 0) {
		throw new Error(`tar exited ${result.exitCode}: ${result.stderr.slice(0, 500)}`);
	}
}


/**
 * Run two smoke checks against the unpacked archive.
 * Real failures throw; architectural mismatches against the dev
 * host (e.g. extracting a darwin archive on linux) skip with a clear note.
 */
export async function smokeUnpacked(opts: SmokeOptions): Promise<void> {
	const { fs, runner, archiveDir, plan, dryRun } = opts;
	const piInArchive = join(archiveDir, plan.piBinaryName);
	if (!(await pathExists(fs, piInArchive))) {
		throw new Error(`unpack smoke: missing ${plan.piBinaryName} inside ${archiveDir}`);
	}

	// In dry-run the staged `pi` is a stub text file, not a real executable,
	// so we cannot invoke it. Verify the archive structure and skip the
	// process-level checks.
	if (dryRun) {
		const stat = await fs.stat(piInArchive);
		process.stdout.write(
			`  [dry-run] ${plan.piBinaryName}: present (${stat.size} bytes) — subprocess skipped\n`,
		);
		const hostBin = join(archiveDir, plan.hostBinaryName);
		if (await pathExists(fs, hostBin)) {
			const hstat = await fs.stat(hostBin);
			process.stdout.write(
				`  [dry-run] ${plan.hostBinaryName}: present (${hstat.size} bytes) — handshake skipped\n`,
			);
		} else {
			const runtime = join(archiveDir, plan.bunRuntimeName);
			const script = join(archiveDir, plan.hostBundleName);
			if (!(await pathExists(fs, runtime)) || !(await pathExists(fs, script))) {
				throw new Error(
					`unpack smoke: missing ${plan.hostBinaryName} and incomplete runtime-bundle fallback`,
				);
			}
			process.stdout.write(
				`  [dry-run] ${plan.bunRuntimeName} + ${plan.hostBundleName}: present — handshake skipped\n`,
			);
		}
		return;
	}

	const versionRes = await runner.run(piInArchive, ["--version"], {
		rejectOnError: false,
		cwd: archiveDir,
		timeoutMs: SMOKE_TIMEOUT_MS,
	});
	if (versionRes.exitCode !== 0) {
		throw new Error(
			`unpack smoke: pi --version failed (exit ${versionRes.exitCode}): ${versionRes.stderr.slice(0, 500)}`,
		);
	}
	process.stdout.write(`  pi --version: ${versionRes.stdout.trim() || "(no output)"}\n`);
	// Prefer the compiled sibling, then exercise the shipped Bun+JS fallback.
	const hostBin = join(archiveDir, plan.hostBinaryName);
	let hostProgram = hostBin;
	let hostArgs: readonly string[] = [];
	if (!(await pathExists(fs, hostBin))) {
		const runtime = join(archiveDir, plan.bunRuntimeName);
		const script = join(archiveDir, plan.hostBundleName);
		if (!(await pathExists(fs, runtime))) {
			throw new Error(`unpack smoke: missing ${plan.hostBinaryName} and ${plan.bunRuntimeName}`);
		}
		if (!(await pathExists(fs, script))) {
			throw new Error(`unpack smoke: missing ${plan.hostBinaryName} and ${plan.hostBundleName}`);
		}
		hostProgram = runtime;
		hostArgs = [script];
	}
	const hostRes = await runner.run(hostProgram, hostArgs, {
		stdin: helloRequestLine(),
		rejectOnError: false,
		cwd: archiveDir,
		timeoutMs: SMOKE_TIMEOUT_MS,
	});
	const firstLine = hostRes.stdout.split("\n", 1)[0] ?? "";
	if (hostRes.exitCode !== 0 || !isHelloAckLine(firstLine)) {
		throw new Error(
			`unpack smoke: host hello handshake failed (exit ${hostRes.exitCode}): ${firstLine.slice(0, 500)}; stderr=${hostRes.stderr.slice(0, 500)}`,
		);
	}
	process.stdout.write(
		hostProgram === hostBin
			? `  host hello handshake: OK\n`
			: `  runtime-bundle host hello handshake: OK\n`,
	);
}


if (import.meta.main) {
	main().catch((err: unknown) => {
		console.error(err);
		process.exit(1);
	});
}
