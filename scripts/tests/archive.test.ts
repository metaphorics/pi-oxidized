import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import {
	checksumLine,
	readBytes,
	safeRelativePath,
	sha256Bytes,
	writeTarGz,
	writeZip,
} from "../release/archive.ts";
import { TraversalError } from "../release/archive.ts";
import { tarArgs } from "../release/runner.ts";

let work: string;

beforeEach(() => {
	work = mkdtempSync(join(tmpdir(), "pi-release-archive-"));
});

afterEach(() => {
	rmSync(work, { recursive: true, force: true });
});

function ownedBytes(bytes: Uint8Array): Uint8Array<ArrayBuffer> {
	const copy = new Uint8Array(bytes.byteLength);
	copy.set(bytes);
	return copy;
}

describe("safeRelativePath", () => {
	test("accepts a clean relative POSIX path", () => {
		expect(safeRelativePath("dir/file.txt")).toBe("dir/file.txt");
	});

	test("rejects absolute paths", () => {
		expect(() => safeRelativePath("/etc/passwd")).toThrow(TraversalError);
	});

	test("rejects backslashes", () => {
		expect(() => safeRelativePath("dir\\file")).toThrow(TraversalError);
	});

	test("rejects empty segments and dot escapes", () => {
		expect(() => safeRelativePath("dir//file")).toThrow(TraversalError);
		expect(() => safeRelativePath("dir/./file")).toThrow(TraversalError);
		expect(() => safeRelativePath("dir/../escape")).toThrow(TraversalError);
	});

	test("rejects empty input", () => {
		expect(() => safeRelativePath("")).toThrow(TraversalError);
	});
});

describe("sha256Bytes + checksumLine", () => {
	test("hash is 64 lowercase hex chars", () => {
		const h = sha256Bytes(new TextEncoder().encode("hello"));
		expect(h).toHaveLength(64);
		expect(h).toMatch(/^[0-9a-f]+$/);
		expect(h).toBe("2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824");
	});

	test("checksumLine formats with two-space separator + newline", () => {
		expect(checksumLine("abcd", "file.zip")).toBe("abcd  file.zip\n");
	});
});

/**
 * Parse the USTAR mtime (12-byte octal field at offset 136) for the first
 * entry of a tar archive. Used to assert the deterministic mtime is what the
 * writer claims.
 */
function readFirstTarMtime(bytes: Uint8Array): number {
	// tar header is 512 bytes; mtime is at offset 136, 12 bytes.
	const mtimeStr = new TextDecoder().decode(bytes.subarray(136, 148)).replace(/\0+$/, "");
	return Number.parseInt(mtimeStr, 8);
}

/** Parse the DOS modification date (4 bytes at offset 12) of the first
 * local-file header in a zip archive. */
function readFirstZipDosDate(bytes: Uint8Array): { time: number; date: number } {
	// Skip leading zero bytes (the zip writer starts at offset 0 with the
	// local file header signature 0x04034b50).
	const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
	// Verify signature: 0x04034b50 little-endian.
	expect(view.getUint32(0, true)).toBe(0x04034b50);
	const time = view.getUint16(10, true);
	const date = view.getUint16(12, true);
	return { time, date };
}

/** Decode a DOS date+time pair (UTC) into an ISO 8601 string. */
function decodeDos(date: number, time: number): string {
	const year = ((date >> 9) & 0x7f) + 1980;
	const month = (date >> 5) & 0x0f;
	const day = date & 0x1f;
	const hour = (time >> 11) & 0x1f;
	const min = (time >> 5) & 0x3f;
	const sec = (time & 0x1f) * 2;
	const pad = (n: number): string => (n < 10 ? `0${n}` : `${n}`);
	return `${year}-${pad(month)}-${pad(day)}T${pad(hour)}:${pad(min)}:${pad(sec)}Z`;
}

describe("writeTarGz determinism", () => {
	test("produces byte-identical output for the same inputs", async () => {
		const entries = [
			{ path: "alpha", data: new TextEncoder().encode("one"), mode: 0o644 },
			{ path: "beta", data: new TextEncoder().encode("two"), mode: 0o755 },
		];
		await writeTarGz(entries, join(work, "a.tar.gz"), { sourceDateEpoch: 1234 });
		await writeTarGz(entries, join(work, "b.tar.gz"), { sourceDateEpoch: 1234 });
		const bytesA = await readBytes(join(work, "a.tar.gz"));
		const bytesB = await readBytes(join(work, "b.tar.gz"));
		expect(sha256Bytes(bytesA)).toBe(sha256Bytes(bytesB));
	});

	test("encodes the requested sourceDateEpoch in the USTAR header", async () => {
		const entries = [{ path: "f", data: new Uint8Array([1, 2, 3]), mode: 0o644 }];
		await writeTarGz(entries, join(work, "sde.tar.gz"), { sourceDateEpoch: 1234 });
		// Decompress the gzip wrapper first so we can read the raw tar header.
		const gzBytes = await readBytes(join(work, "sde.tar.gz"));
		const tarBytes = new Uint8Array(Bun.gunzipSync(ownedBytes(gzBytes)));
		expect(readFirstTarMtime(tarBytes)).toBe(1234);
	});

	test("clamps USTAR mtime to 0o77777777777 on extreme SDE", async () => {
		const entries = [{ path: "f", data: new Uint8Array([1]), mode: 0o644 }];
		await writeTarGz(
			entries,
			join(work, "clamp.tar.gz"),
			{ sourceDateEpoch: 10_000_000_000 },
		);
		const gzBytes = await readBytes(join(work, "clamp.tar.gz"));
		const tarBytes = new Uint8Array(Bun.gunzipSync(ownedBytes(gzBytes)));
		expect(readFirstTarMtime(tarBytes)).toBe(0o77777777777);
	});

	test("tar entry order matches sorted path", async () => {
		const entries = [
			{ path: "zeta", data: new Uint8Array([9]), mode: 0o644 },
			{ path: "alpha", data: new Uint8Array([1]), mode: 0o644 },
		];
		await writeTarGz(entries, join(work, "order.tar.gz"), { sourceDateEpoch: 0 });
		const proc = Bun.spawnSync(["tar", ...tarArgs("-tzf", join(work, "order.tar.gz"))]);
		if (proc.exitCode !== 0) {
			const stderr = (proc.stderr?.toString("utf8") ?? "").slice(0, 300);
			throw new Error(`tar exited ${proc.exitCode}: ${stderr}`);
		}
		const out = (proc.stdout?.toString("utf8") ?? "").trim().split("\n");
		expect(out).toEqual(["alpha", "zeta"]);
	});
});

describe("writeZip determinism", () => {
	test("produces byte-identical output for the same inputs", async () => {
		const entries = [
			{ path: "alpha", data: new TextEncoder().encode("one"), mode: 0o644 },
			{ path: "beta", data: new TextEncoder().encode("two"), mode: 0o755 },
		];
		await writeZip(entries, join(work, "a.zip"), { sourceDateEpoch: 1234 });
		await writeZip(entries, join(work, "b.zip"), { sourceDateEpoch: 1234 });
		const a = await readBytes(join(work, "a.zip"));
		const b = await readBytes(join(work, "b.zip"));
		expect(sha256Bytes(a)).toBe(sha256Bytes(b));
	});

	test("encodes the requested sourceDateEpoch as DOS date+time", async () => {
		const entries = [{ path: "f", data: new Uint8Array([1]), mode: 0o644 }];
		await writeZip(entries, join(work, "sde.zip"), { sourceDateEpoch: 1234567890 });
		const bytes = await readBytes(join(work, "sde.zip"));
		const { date, time } = readFirstZipDosDate(bytes);
		// 2009-02-13 23:31:30 UTC
		expect(decodeDos(date, time)).toBe("2009-02-13T23:31:30Z");
	});

	test("clamps DOS date to 1980-01-01 for pre-1980 epochs", async () => {
		const entries = [{ path: "f", data: new Uint8Array([1]), mode: 0o644 }];
		await writeZip(entries, join(work, "clamp-low.zip"), { sourceDateEpoch: -1 });
		const bytes = await readBytes(join(work, "clamp-low.zip"));
		const { date, time } = readFirstZipDosDate(bytes);
		expect(decodeDos(date, time)).toBe("1980-01-01T00:00:00Z");
	});

	test("clamps DOS date to 2107-12-31 for far-future epochs", async () => {
		const entries = [{ path: "f", data: new Uint8Array([1]), mode: 0o644 }];
		await writeZip(
			entries,
			join(work, "clamp-high.zip"),
			{ sourceDateEpoch: 10_000_000_000 },
		);
		const bytes = await readBytes(join(work, "clamp-high.zip"));
		const { date, time } = readFirstZipDosDate(bytes);
		// 10 billion is clamped to USTAR max (8589934591 = 2242-03-16 12:56:31Z).
		// The DOS year is then clamped to 127 (1980+127 = 2107).
		expect(decodeDos(date, time)).toBe("2107-03-16T12:56:30Z");
	});

	test("zip is readable by host `unzip` and lists entries in sorted order", async () => {
		const entries = [
			{ path: "zeta", data: new Uint8Array([9]), mode: 0o644 },
			{ path: "alpha", data: new Uint8Array([1]), mode: 0o644 },
		];
		await writeZip(entries, join(work, "order.zip"), { sourceDateEpoch: 0 });
		const proc = Bun.spawnSync(["unzip", "-l", join(work, "order.zip")]);
		const out = proc.stdout?.toString("utf8") ?? "";
		const fileLines = out
			.split("\n")
			.map((l) => l.trim())
			.filter((l) => l.includes("alpha") || l.includes("zeta"));
		expect(fileLines[0]).toContain("alpha");
		expect(fileLines[1]).toContain("zeta");
	});
});
