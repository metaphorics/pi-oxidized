/**
 * Release staging: assemble the on-disk directory tree that becomes the
 * archive, verify target agreement + executable bits + no-host-in-pi, and
 * emit the `release.json` manifest with per-file SHA-256 digests.
 *
 * Every external interaction flows through the {@link Fs} and
 * {@link CommandRunner} seams so tests can drive the assembly without
 * invoking cargo or bun.
 */

import { createHash } from "node:crypto";

import type { HostArtifact } from "./host.ts";
import { safeJoinPath } from "./runner.ts";
import type { Fs } from "./runner.ts";
import type { TargetPlan } from "./targets.ts";

/** Schema discriminator stamped into every `release.json`. */
export const RELEASE_MANIFEST_SCHEMA = "pi.release.v1" as const;

/** Per-file entry inside the manifest. */
export interface ManifestFile {
	/** POSIX-style path relative to the archive root. */
	readonly path: string;
	/** File size in bytes. */
	readonly size: number;
	/** Lowercase hex SHA-256 digest of file bytes. */
	readonly sha256: string;
	/** `true` for binaries that must carry the executable bit. */
	readonly executable: boolean;
}

/** Shape of `release.json` shipped inside every archive. */
export interface ReleaseManifest {
	readonly schema: typeof RELEASE_MANIFEST_SCHEMA;
	readonly version: string;
	readonly rustTarget: string;
	readonly bunTarget: string;
	/** `compiled` sidecar or `runtime-bundle` fallback. */
	readonly hostKind: HostArtifact["kind"];
	readonly compatibilityVersion: string;
	readonly protocolVersion: number;
	readonly sourceDateEpoch: number;
	readonly createdAt: string;
	readonly files: readonly ManifestFile[];
}

/** Inputs to {@link assembleRelease}. */
export interface AssembleInputs {
	/** Resolved release target. */
	readonly plan: TargetPlan;
	/** Workspace version (e.g. `0.1.0`). */
	readonly version: string;
	/** Absolute path to the freshly-built Rust binary for `plan.rustTarget`. */
	readonly piBinaryPath: string;
	/** Absolute path to the workspace root (for metadata sources). */
	readonly repoRoot: string;
	/** Built host artifact (compiled sidecar or runtime-bundle fallback). */
	readonly host: HostArtifact;
	/**
	 * Absolute path to a pre-built Bun runtime matching `plan.bunTarget`,
	 * required only when `host.kind === "runtime-bundle"`. The release script
	 * supplies it from the official Bun release archive.
	 */
	readonly bunRuntimePath?: string;
	/**
	 * Optional runtime-bundle fallback staged beside a compiled host: the
	 * host JavaScript bundle plus the provisioned Bun runtime. Musl rows
	 * ship both host execution paths so the release smoke drives each
	 * `hello` protocol from the same unpacked archive.
	 */
	readonly fallbackBundle?: {
		readonly scriptPath: string;
		readonly bunRuntimePath: string;
	};
	/** Filesystem seam. */
	readonly fs: Fs;
	/** Source-date-epoch stamp for the manifest + archive mtimes. */
	readonly sourceDateEpoch: number;
	/** Compatibility version recorded in the manifest. */
	readonly compatibilityVersion: string;
	/** Protocol version recorded in the manifest. */
	readonly protocolVersion: number;
	/**
	 * Built timestamp (ISO 8601). For reproducibility, pass a fixed value
	 * derived from `sourceDateEpoch`; tests inject deterministic strings.
	 */
	readonly createdAt: string;
	/**
	 * Absolute path to the docs tree copied verbatim into `<archiveDir>/docs/`
	 * of every archive. Required: a release without the shipped documentation
	 * fails staging rather than silently shipping a doc-less archive.
	 */
	readonly docsSource: string;
	/** Absolute path to the assets tree copied into `<archiveDir>/assets/`; skipped when absent. */
	readonly assetsSource?: string;
	/**
	 * Optional set of additional `(src, archiveRelPath)` pairs to copy in,
	 * used by tests to verify reproducibility without spinning up cargo.
	 * Reserved destinations (binary slots, manifest path, or duplicates) are
	 * rejected before any bytes are written.
	 */
	readonly extraFiles?: readonly { readonly src: string; readonly dest: string }[];
}

/** One ordered source-to-archive operation performed by {@link assembleRelease}. */
export interface StagedInput {
	readonly kind:
		| "rust-binary"
		| "host-binary"
		| "host-bundle"
		| "bun-runtime"
		| "metadata-file"
		| "tree"
		| "extra"
		| "manifest";
	readonly source: string;
	readonly destRel: string;
	readonly optional: boolean;
}

/** Ordered staging authority for one release assembly. */
export function stagedInputs(inputs: AssembleInputs): readonly StagedInput[] {
	const staged: StagedInput[] = [
		{
			kind: "rust-binary",
			source: inputs.piBinaryPath,
			destRel: inputs.plan.piBinaryName,
			optional: false,
		},
	];
	if (inputs.host.kind === "compiled") {
		staged.push({
			kind: "host-binary",
			source: inputs.host.binaryPath,
			destRel: inputs.plan.hostBinaryName,
			optional: false,
		});
	} else {
		staged.push(
			{
				kind: "host-bundle",
				source: inputs.host.scriptPath,
				destRel: inputs.plan.hostBundleName,
				optional: false,
			},
			{
				kind: "bun-runtime",
				source: inputs.bunRuntimePath ?? "",
				destRel: inputs.plan.bunRuntimeName,
				optional: false,
			},
		);
	}
	if (inputs.fallbackBundle !== undefined) {
		staged.push(
			{
				kind: "host-bundle",
				source: inputs.fallbackBundle.scriptPath,
				destRel: inputs.plan.hostBundleName,
				optional: false,
			},
			{
				kind: "bun-runtime",
				source: inputs.fallbackBundle.bunRuntimePath,
				destRel: inputs.plan.bunRuntimeName,
				optional: false,
			},
		);
	}
	// CHANGELOG.md, README.md, and the docs tree are mandatory release
	// members: the CHANGELOG gate refuses builds without release notes, and
	// no archive may ship without the README or the documentation. Only the
	// license files stay optional.
	for (const [name, optional] of [
		["CHANGELOG.md", false],
		["README.md", false],
		["LICENSE", true],
		["LICENSE-MIT", true],
	] as const) {
		staged.push({
			kind: "metadata-file",
			source: `${inputs.repoRoot}/${name}`,
			destRel: name,
			optional,
		});
	}
	for (const [source, destRel, optional] of [
		[inputs.docsSource, "docs", false],
		[inputs.assetsSource, "assets", true],
		[`${inputs.repoRoot}/crates/pi/assets/theme`, "theme", true],
	] as const) {
		staged.push({
			kind: "tree",
			source: source ?? "",
			destRel,
			optional,
		});
	}
	for (const extra of inputs.extraFiles ?? []) {
		staged.push({
			kind: "extra",
			source: extra.src,
			destRel: normalizeArchiveRel(extra.dest),
			optional: false,
		});
	}
	staged.push({
		kind: "manifest",
		source: "generated:release.json",
		destRel: "release.json",
		optional: false,
	});
	return staged;
}

/** Result of {@link assembleRelease}: staging directory + manifest. */
export interface AssembledRelease {
	/** Absolute path to the staged archive-root directory. */
	readonly stagingDir: string;
	/** Written `release.json` document. */
	readonly manifest: ReleaseManifest;
}

/** Error raised when a target-agreement or contamination check fails. */
export class ReleaseVerifyError extends Error {
	constructor(message: string) {
		super(message);
		this.name = "ReleaseVerifyError";
	}
}

/** Reserved top-level slots extras cannot overwrite. */
const RESERVED_TOP_LEVEL: Record<string, true> = {
	"pi": true,
	"pi.exe": true,
	"pi-extension-host": true,
	"pi-extension-host.exe": true,
	"pi-extension-host.js": true,
	"bun": true,
	"bun.exe": true,
	"release.json": true,
};

/** POSIX-style normalization of an archive-relative path. */
function normalizeArchiveRel(path: string): string {
	return path.split("\\").join("/").replace(/^\/+/, "");
}

/** Track which archive-relative paths have already been staged. */
class UsedPaths {
	private readonly seen = new Set<string>();

	claim(relPath: string): void {
		const norm = normalizeArchiveRel(relPath);
		if (this.seen.has(norm)) {
			throw new ReleaseVerifyError(`duplicate archive path: ${norm}`);
		}
		this.seen.add(norm);
	}

	assertNotReserved(relPath: string): void {
		const norm = normalizeArchiveRel(relPath);
		const top = norm.split("/")[0] ?? "";
		if (RESERVED_TOP_LEVEL[top] && !norm.startsWith("docs/") && !norm.startsWith("assets/")) {
			throw new ReleaseVerifyError(
				`extraFile destination collides with reserved slot: ${norm}`,
			);
		}
	}
}

/**
 * Assemble the release tree at `<stagingRoot>/<plan.archiveDir>/`:
 *
 *   <archiveDir>/
 *     pi[.exe]
 *     pi-extension-host[.exe]            (compiled path)
 *     bun[.exe], pi-extension-host.js    (fallback path)
 *     CHANGELOG.md (required), README.md, LICENSE
 *     docs/...                           (recursive, required)
 *     assets/...                         (recursive)
 *     release.json
 *
 * Every destination path is run through {@link safeJoinPath} so a malicious
 * source name cannot escape the staging root, and every stage step claims
 * its paths through {@link UsedPaths} so a caller-supplied `extraFiles`
 * entry cannot overwrite a binary or duplicate another file.
 */
export async function assembleRelease(
	stagingRoot: string,
	inputs: AssembleInputs,
): Promise<AssembledRelease> {
	const { fs, plan } = inputs;
	const archiveDir = safeJoinPath(stagingRoot, plan.archiveDir);
	await fs.mkdir(archiveDir, { recursive: true });

	const used = new UsedPaths();
	const copied: ManifestFile[] = [];

	for (const staged of stagedInputs(inputs)) {
		switch (staged.kind) {
			case "rust-binary":
			case "host-binary":
			case "host-bundle":
			case "bun-runtime": {
				if (staged.kind === "bun-runtime" && staged.source.length === 0) {
					throw new ReleaseVerifyError(
						`runtime-bundle host requires bunRuntimePath for ${plan.rustTarget}`,
					);
				}
				const executable =
					staged.kind === "rust-binary" ||
					staged.kind === "host-binary" ||
					staged.kind === "bun-runtime";
				const file = await copyBinary(
					fs,
					staged.source,
					archiveDir,
					staged.destRel,
					executable,
					staged.optional,
					plan.windows,
					used,
				);
				if (file) copied.push(file);
				break;
			}
			case "metadata-file": {
				const file = await copyStagedFile(
					fs,
					staged.source,
					archiveDir,
					staged.destRel,
					staged.optional,
					used,
				);
				if (file) copied.push(file);
				break;
			}
			case "tree":
				for (const file of await copyTreeOptional(
					fs,
					staged.source,
					archiveDir,
					staged.destRel,
					staged.optional,
					used,
				)) {
					copied.push(file);
				}
				break;
			case "extra": {
				used.assertNotReserved(staged.destRel);
				const data = await readStagedFile(fs, staged.source, staged.optional);
				if (data) {
					used.claim(staged.destRel);
					const dest = safeJoinPath(archiveDir, staged.destRel);
					await fs.mkdir(dest.split("/").slice(0, -1).join("/"), { recursive: true });
					await fs.writeFile(dest, data);
					copied.push(manifestEntryFromData(data, staged.destRel, false));
				}
				break;
			}
			case "manifest": {
				if (staged.optional) {
					throw new ReleaseVerifyError("release manifest cannot be optional");
				}
				await verifyNoHostInPi(fs, inputs.piBinaryPath, inputs.host);
				await verifyExecutableBits(fs, plan, archiveDir, inputs.host);
				const manifest: ReleaseManifest = {
					schema: RELEASE_MANIFEST_SCHEMA,
					version: inputs.version,
					rustTarget: plan.rustTarget,
					bunTarget: plan.bunTarget,
					hostKind: inputs.host.kind,
					compatibilityVersion: inputs.compatibilityVersion,
					protocolVersion: inputs.protocolVersion,
					sourceDateEpoch: inputs.sourceDateEpoch,
					createdAt: inputs.createdAt,
					files: copied
						.slice()
						.sort((a, b) => (a.path < b.path ? -1 : a.path > b.path ? 1 : 0)),
				};
				used.claim(staged.destRel);
				await fs.writeFile(
					safeJoinPath(archiveDir, staged.destRel),
					JSON.stringify(manifest, null, 2) + "\n",
				);
				return { stagingDir: archiveDir, manifest };
			}
		}
	}
	throw new ReleaseVerifyError("staging plan omitted release manifest");
}

/** Copy a binary into the staging tree, preserving the executable bit. */
async function copyBinary(
	fs: Fs,
	srcPath: string,
	archiveDir: string,
	destName: string,
	executable: boolean,
	optional: boolean,
	isWindows: boolean,
	used: UsedPaths,
): Promise<ManifestFile | null> {
	const data = await readStagedFile(fs, srcPath, optional);
	if (!data) return null;
	used.claim(destName);
	const dest = safeJoinPath(archiveDir, destName);
	await fs.writeFile(dest, data);
	// Windows has no chmod; the archive writer carries the bit via
	// manifest metadata and the installer restores it on POSIX.
	if (executable && !isWindows) {
		try {
			await fs.chmod(dest, 0o755);
		} catch {
			// Just in case it fails on a strange filesystem.
		}
	}
	return manifestEntryFromData(data, destName, executable);
}

/** Read a staged file, returning `null` only when its source is optional. */
async function readStagedFile(
	fs: Fs,
	src: string,
	optional: boolean,
): Promise<Uint8Array | null> {
	try {
		return await fs.readFile(src);
	} catch (error) {
		if (optional) return null;
		throw error;
	}
}

/** Copy one staged file; optional sources can be absent. */
async function copyStagedFile(
	fs: Fs,
	src: string,
	archiveDir: string,
	destRel: string,
	optional: boolean,
	used: UsedPaths,
): Promise<ManifestFile | null> {
	let data: Uint8Array | null;
	try {
		data = await readStagedFile(fs, src, optional);
	} catch (error) {
		throw new ReleaseVerifyError(
			`required staged file is missing: ${src} (${errMessage(error)})`,
		);
	}
	if (!data) return null;
	used.claim(destRel);
	const dest = safeJoinPath(archiveDir, destRel);
	await fs.writeFile(dest, data);
	return manifestEntryFromData(data, destRel, false);
}

/**
 * Recursively copy `src` into `<archiveDir>/<destSubdir>/`, returning one
 * manifest entry per file. Optional missing sources produce no entries.
 */
async function copyTreeOptional(
	fs: Fs,
	src: string | undefined,
	archiveDir: string,
	destSubdir: string,
	optional: boolean,
	used: UsedPaths,
): Promise<ManifestFile[]> {
	if (!src) {
		if (optional) return [];
		throw new ReleaseVerifyError(`required staged tree has no source: ${destSubdir}`);
	}
	try {
		const stat = await fs.stat(src);
		if (!stat.isDir) {
			if (optional) return [];
			throw new ReleaseVerifyError(`required staged tree is not a directory: ${src}`);
		}
	} catch (error) {
		if (optional) return [];
		throw new ReleaseVerifyError(
			`required staged tree is missing or unreadable: ${src} (${errMessage(error)})`,
		);
	}

	const destRoot = safeJoinPath(archiveDir, destSubdir);
	await fs.cp(src, destRoot, { recursive: true });
	const out: ManifestFile[] = [];
	const queue: string[] = [destRoot];
	while (queue.length > 0) {
		const dir = queue.shift();
		if (dir === undefined) break;
		let entries: string[];
		try {
			entries = await fs.readdir(dir);
		} catch {
			continue;
		}
		entries.sort();
		for (const name of entries) {
			const childAbs = `${dir}/${name}`;
			const s = await fs.stat(childAbs);
			const rel = archiveRelativePath(archiveDir, childAbs);
			if (s.isDir) {
				queue.push(childAbs);
				continue;
			}
			if (!s.isFile) continue;
			used.claim(rel);
			const data = await fs.readFile(childAbs);
			out.push(manifestEntryFromData(data, rel, false));
		}
	}
	return out;
}

/** Compute the archive-relative POSIX path for an absolute staged file. */
function archiveRelativePath(archiveDir: string, absPath: string): string {
	const rel = absPath.slice(archiveDir.length + 1);
	return rel.split("\\").join("/");
}

/**
 * Verification: a 64 KiB slice of the host must not appear contiguously in
 * the Rust binary. If it does, the build accidentally embedded the host via
 * `include_bytes!` (the master plan forbids embedding ~100 MB into the LTO
 * link).
 */
async function verifyNoHostInPi(fs: Fs, piPath: string, host: HostArtifact): Promise<void> {
	const piBytes = await fs.readFile(piPath);
	const hostSrc = host.kind === "compiled" ? host.binaryPath : host.scriptPath;
	const hostBytes = await fs.readFile(hostSrc);
	const probeLen = Math.min(64 * 1024, hostBytes.length);
	if (probeLen === 0) return;
	const probe = hostBytes.subarray(0, probeLen);
	if (Buffer.from(piBytes).indexOf(probe) !== -1) {
		throw new ReleaseVerifyError(
			`Rust binary at ${piPath} contains a contiguous ${probeLen}-byte slice of the host ${hostSrc}; the host must be shipped beside the binary, never embedded.`,
		);
	}
}

/**
 * Verification: pi (and the host binary in compiled mode) must exist and
 * carry the executable bit on POSIX. Windows archives skip the bit check,
 * and so does any run on a Windows host: exec bits are unobservable there
 * (Node reports 0o666 regardless of chmod), while the tar manifest still
 * carries the archived modes. Gating on the target alone breaks cross
 * dry-runs executed on Windows (proven: all six POSIX targets failed).
 */
async function verifyExecutableBits(
	fs: Fs,
	plan: TargetPlan,
	archiveDir: string,
	host: HostArtifact,
): Promise<void> {
	if (plan.windows) return; // Windows uses the manifest's executable flag.
	if (process.platform === "win32") return; // Host cannot observe exec bits.
	const required = [plan.piBinaryName];
	if (host.kind === "compiled") required.push(plan.hostBinaryName);
	if (host.kind === "runtime-bundle") required.push(plan.bunRuntimeName);
	for (const name of required) {
		const path = safeJoinPath(archiveDir, name);
		const s = await fs.stat(path);
		if (!s.isFile) {
			throw new ReleaseVerifyError(`expected file at ${path}`);
		}
		if ((s.mode & 0o111) === 0) {
			throw new ReleaseVerifyError(`${path} is missing the executable bit`);
		}
	}
}

/** Build a manifest entry from already-loaded bytes. */
function manifestEntryFromData(data: Uint8Array, relPath: string, executable: boolean): ManifestFile {
	const hash = createHash("sha256").update(data).digest("hex");
	return { path: relPath, size: data.length, sha256: hash, executable };
}

/** Render an unknown error as a short string for diagnostic messages. */
function errMessage(err: unknown): string {
	if (err instanceof Error) return err.message;
	return String(err);
}
