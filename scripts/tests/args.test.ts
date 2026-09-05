import { describe, expect, test } from "bun:test";
import { resolve } from "node:path";

import {
	ArgvHelpRequested,
	InvalidSourceDateEpochError,
	MissingArgValueError,
	MissingTargetError,
	parseReleaseArgs,
	UnknownArgError,
} from "../release/args.ts";
import { InvalidTargetError, RUST_TARGETS } from "../release/targets.ts";

describe("args", () => {
	test("resolves the minimal happy path with defaults", () => {
		// The release workflow exports SOURCE_DATE_EPOCH and the parser
		// forwards process.env by default (explicit undefined re-triggers
		// the default), so scope a clean env around the default leg.
		const savedEpoch = process.env.SOURCE_DATE_EPOCH;
		delete process.env.SOURCE_DATE_EPOCH;
		try {
			const args = parseReleaseArgs(["--target", "x86_64-unknown-linux-gnu"]);
			expect(args.plan.rustTarget).toBe("x86_64-unknown-linux-gnu");
			expect(args.dryRun).toBe(false);
			expect(args.noCargo).toBe(false);
			expect(args.handshake).toBe(true);
			expect(args.sourceDateEpoch).toBe("0");
		} finally {
			if (savedEpoch !== undefined) process.env.SOURCE_DATE_EPOCH = savedEpoch;
		}
	});

	test("honors --dry-run and --no-cargo independently", () => {
		const dryArgs = parseReleaseArgs([
			"--target",
			"x86_64-apple-darwin",
			"--dry-run",
		]);
		expect(dryArgs.dryRun).toBe(true);
		expect(dryArgs.noCargo).toBe(false);

		const noCargoArgs = parseReleaseArgs([
			"--target",
			"x86_64-apple-darwin",
			"--no-cargo",
		]);
		expect(noCargoArgs.dryRun).toBe(false);
		expect(noCargoArgs.noCargo).toBe(true);
	});

	test("expands --out to an absolute path", () => {
		// Expectations go through resolve like the parser does: identical on
		// POSIX, drive-aware on Windows ("/tmp/x" is drive-relative there).
		const abs = parseReleaseArgs(
			["--target", "x86_64-apple-darwin", "--out", "/tmp/release"],
			"/cwd",
		);
		expect(abs.outDir).toBe(resolve("/cwd", "/tmp/release"));

		const relative = parseReleaseArgs(
			["--target", "x86_64-apple-darwin", "--out", "artifacts"],
			"/cwd",
		);
		expect(relative.outDir).toBe(resolve("/cwd", "artifacts"));

		const defaultArgs = parseReleaseArgs(
			["--target", "x86_64-apple-darwin"],
			"/cwd",
		);
		expect(defaultArgs.outDir).toBe(resolve("/cwd", "dist/release"));
	});

	test("accepts the --out-dir alias", () => {
		const args = parseReleaseArgs(
			["--target", "x86_64-apple-darwin", "--out-dir", "/tmp/r2"],
			"/cwd",
		);
		expect(args.outDir).toBe(resolve("/cwd", "/tmp/r2"));
	});

	test("resolves --runtime-cache to an absolute path and defaults to undefined", () => {
		const absolute = parseReleaseArgs(
			["--target", "x86_64-unknown-linux-gnu", "--runtime-cache", "/var/cache/pi"],
			"/cwd",
		);
		expect(absolute.runtimeCache).toBe(resolve("/cwd", "/var/cache/pi"));

		const relative = parseReleaseArgs(
			["--target", "x86_64-unknown-linux-gnu", "--runtime-cache", "rel/cache"],
			"/cwd",
		);
		expect(relative.runtimeCache).toBe(resolve("/cwd", "rel/cache"));

		const omitted = parseReleaseArgs(["--target", "x86_64-unknown-linux-gnu"], "/cwd");
		expect(omitted.runtimeCache).toBeUndefined();

		expect(() =>
			parseReleaseArgs(["--target", "x86_64-unknown-linux-gnu", "--runtime-cache"]),
		).toThrow(MissingArgValueError);
	});

	test("rejects missing --target with MissingTargetError", () => {
		expect(() => parseReleaseArgs([])).toThrow(MissingTargetError);
	});

	test("rejects unsupported targets via planFor", () => {
		expect(() => parseReleaseArgs(["--target", "riscv64gc-unknown-linux-gnu"])).toThrow(
			InvalidTargetError,
		);
	});

	test("rejects unknown flags", () => {
		expect(() =>
			parseReleaseArgs(["--target", "x86_64-apple-darwin", "--bogus"]),
		).toThrow(UnknownArgError);
	});

	test("rejects missing values for value-taking flags", () => {
		expect(() => parseReleaseArgs(["--target"])).toThrow(MissingArgValueError);
		expect(() =>
			parseReleaseArgs(["--target", "x86_64-apple-darwin", "--out"]),
		).toThrow(MissingArgValueError);
		expect(() =>
			parseReleaseArgs(["--target", "x86_64-apple-darwin", "--out", "-x"]),
		).toThrow(MissingArgValueError);
	});

	test("rejects non-decimal --source-date-epoch", () => {
		expect(() =>
			parseReleaseArgs([
				"--target",
				"x86_64-apple-darwin",
				"--source-date-epoch",
				"1.5",
			]),
		).toThrow(InvalidSourceDateEpochError);
		expect(() =>
			parseReleaseArgs([
				"--target",
				"x86_64-apple-darwin",
				"--source-date-epoch",
				"abc",
			]),
		).toThrow(InvalidSourceDateEpochError);
		expect(() =>
			parseReleaseArgs(
				["--target", "x86_64-apple-darwin"],
				"/cwd",
				"-1",
			),
		).toThrow(InvalidSourceDateEpochError);
	});

	test("throws ArgvHelpRequested on --help / -h", () => {
		expect(() => parseReleaseArgs(["--help"])).toThrow(ArgvHelpRequested);
		expect(() => parseReleaseArgs(["-h"])).toThrow(ArgvHelpRequested);
	});

	test("forwards SOURCE_DATE_EPOCH env when --source-date-epoch is absent", () => {
		const args = parseReleaseArgs(["--target", "x86_64-apple-darwin"], "/cwd", "1700000000");
		expect(args.sourceDateEpoch).toBe("1700000000");
	});

	test("explicit --source-date-epoch overrides env", () => {
		const args = parseReleaseArgs(
			["--target", "x86_64-apple-darwin", "--source-date-epoch", "42"],
			"/cwd",
			"1700000000",
		);
		expect(args.sourceDateEpoch).toBe("42");
	});

	test("--no-handshake disables handshake", () => {
		const args = parseReleaseArgs(["--target", "x86_64-apple-darwin", "--no-handshake"]);
		expect(args.handshake).toBe(false);
	});

	test("rejects removed --skip-host-tests flag", () => {
		expect(() =>
			parseReleaseArgs(["--target", "x86_64-apple-darwin", "--skip-host-tests"]),
		).toThrow(UnknownArgError);
	});

	test("iterates over every supported triple without throwing", () => {
		for (const triple of RUST_TARGETS) {
			const args = parseReleaseArgs(["--target", triple]);
			expect(args.plan.rustTarget).toBe(triple);
		}
	});
});
