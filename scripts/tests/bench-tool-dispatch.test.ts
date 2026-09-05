import { describe, expect, test } from "bun:test";
import { existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import {
	Distribution,
	INVALID_ARGUMENTS,
	NOOP_PARAMETERS,
	VALID_ARGUMENTS,
	WorkerReport,
	distribution,
	implementationOrder,
	parseWorkerFlags,
	protocolExpectations,
	validateWorkerReport,
} from "../bench-tool-dispatch.ts";

const REPOSITORY_ROOT = resolve(import.meta.dirname, "../..");
const ARTIFACT_PATH = join(REPOSITORY_ROOT, "target/bench/tool-dispatch.json");

function sampleReport(overrides: Partial<WorkerReport> = {}): WorkerReport {
	return {
		implementation: "typescript",
		argumentsMode: "valid",
		warmupCalls: 2,
		callsPerBlock: 4,
		blocks: [
			{
				index: 0,
				calls: 4,
				wallMsPerCall: 0.05,
				wallMedianNs: 50_000,
				wallMinNs: 40_000,
				wallMaxNs: 60_000,
				cpuMsPerCall: 0.2,
			},
		],
		events: { start: 4, update: 4, end: 4, errorResults: 0 },
		appends: 8,
		session: { file: null, bytesDelta: 1024, headerEntries: 1 },
		ok: true,
		failure: null,
		...overrides,
	};
}

describe("bench-tool-dispatch pure helpers", () => {
	test("distribution preserves quantiles and spread fields", () => {
		const result: Distribution = distribution([1, 2, 3, 4, 5]);
		expect(result.median).toBe(3);
		expect(result.p95).toBe(4.8);
		expect(result.p99).toBeCloseTo(4.96, 12);
		expect(result.count).toBe(5);
		expect(result.stddev).toBeCloseTo(Math.sqrt(2), 12);
		expect(result.relativeSpread).toBeCloseTo(Math.sqrt(2) / 3, 12);
	});

	test("distribution rejects empty or negative samples", () => {
		expect(() => distribution([])).toThrow();
		expect(() => distribution([1, -1])).toThrow();
	});

	test("implementationOrder alternates starting sides per sample", () => {
		expect(implementationOrder(0)).toEqual(["rust", "typescript"]);
		expect(implementationOrder(1)).toEqual(["typescript", "rust"]);
		expect(implementationOrder(2)).toEqual(["rust", "typescript"]);
	});

	test("parseWorkerFlags parses the worker protocol and rejects bad flags", () => {
		const flags = parseWorkerFlags([
			"--worker",
			"--calls",
			"7",
			"--warmup",
			"3",
			"--blocks",
			"2",
			"--session-dir",
			"/tmp/x",
			"--arguments",
			"invalid",
		]);
		expect(flags).toMatchObject({
			worker: true,
			calls: 7,
			warmup: 3,
			blocks: 2,
			sessionDir: "/tmp/x",
			invalid: true,
		});
		expect(() => parseWorkerFlags(["--arguments", "sometimes"])).toThrow();
		expect(() => parseWorkerFlags(["--calls"])).toThrow();
		expect(() => parseWorkerFlags(["--calls", "0"])).toThrow();
	});

	test("protocolExpectations encode the shared dispatch contract", () => {
		expect(protocolExpectations("valid", 10)).toEqual({
			start: 10,
			update: 10,
			end: 10,
			errorResults: 0,
			appends: 20,
		});
		expect(protocolExpectations("invalid", 10)).toEqual({
			start: 10,
			update: 0,
			end: 10,
			errorResults: 10,
			appends: 20,
		});
	});

	test("validateWorkerReport accepts a contract-clean report", () => {
		expect(validateWorkerReport(sampleReport(), { implementation: "typescript", mode: "valid", calls: 4 })).toBeNull();
	});

	test("validateWorkerReport rejects protocol violations", () => {
		const wrongUpdate = sampleReport({
			events: { start: 4, update: 3, end: 4, errorResults: 0 },
		});
		expect(validateWorkerReport(wrongUpdate, { implementation: "typescript", mode: "valid", calls: 4 })).toContain(
			"update events",
		);
		const wrongAppends = sampleReport({ appends: 7 });
		expect(validateWorkerReport(wrongAppends, { implementation: "typescript", mode: "valid", calls: 4 })).toContain(
			"appends",
		);
		const failed = sampleReport({ ok: false, failure: "boom" });
		expect(validateWorkerReport(failed, { implementation: "typescript", mode: "valid", calls: 4 })).toContain("boom");
		const nonpositiveWall = sampleReport({
			blocks: [{ index: 0, calls: 4, wallMsPerCall: 0, wallMedianNs: 0, wallMinNs: 0, wallMaxNs: 0, cpuMsPerCall: null }],
		});
		expect(validateWorkerReport(nonpositiveWall, { implementation: "typescript", mode: "valid", calls: 4 })).toContain(
			"wallMsPerCall",
		);
	});

	test("argument payloads and schema match the Rust worker constants", () => {
		expect(VALID_ARGUMENTS).toEqual({ path: "bench/noop/input.txt", count: 3 });
		// Shared rejection case: range violation, not mistype (upstream coerces mistypes).
		expect(INVALID_ARGUMENTS).toEqual({ path: "bench/noop/input.txt", count: 999 });
		expect(NOOP_PARAMETERS).toEqual({
			type: "object",
			properties: {
				path: { type: "string", minLength: 1 },
				count: { type: "integer", minimum: 1, maximum: 64 },
			},
			required: ["path"],
			additionalProperties: false,
		});
	});

	test("importing the module does not run the benchmark", () => {
		const before = existsSync(ARTIFACT_PATH) ? readFileSync(ARTIFACT_PATH) : undefined;
		expect(typeof distribution).toBe("function");
		const after = existsSync(ARTIFACT_PATH) ? readFileSync(ARTIFACT_PATH) : undefined;
		expect(after).toEqual(before);
	});
});

describe("bench-tool-dispatch TypeScript worker", () => {
	test("drives upstream runAgentLoop with the shared dispatch protocol", async () => {
		const sessionDir = mkdtempSync(join(tmpdir(), "t5-worker-test-"));
		try {
			const spawned = Bun.spawnSync(
				[
					process.execPath,
					join(REPOSITORY_ROOT, "scripts/bench-tool-dispatch.ts"),
					"--worker",
					"--calls",
					"6",
					"--warmup",
					"2",
					"--blocks",
					"1",
					"--session-dir",
					sessionDir,
				],
				{ stdout: "pipe", stderr: "pipe" },
			);
			// Surface worker stderr: a bare exit-code assertion hides the cause
			// (proven blind on CI, where both worker legs exit 1 with no text).
			if (spawned.exitCode !== 0) {
				throw new Error(
					`worker exited ${spawned.exitCode}: ${new TextDecoder().decode(spawned.stderr).slice(-2_000)}`,
				);
			}
			const report = JSON.parse(new TextDecoder().decode(spawned.stdout).trim()) as WorkerReport;
			expect(validateWorkerReport(report, { implementation: "typescript", mode: "valid", calls: 6 })).toBeNull();
			expect(report.session.bytesDelta).toBeGreaterThan(0);
		} finally {
			rmSync(sessionDir, { recursive: true, force: true });
		}
	});

	test("rejects the shared invalid payload through argument validation", async () => {
		const sessionDir = mkdtempSync(join(tmpdir(), "t5-worker-invalid-"));
		try {
			const spawned = Bun.spawnSync(
				[
					process.execPath,
					join(REPOSITORY_ROOT, "scripts/bench-tool-dispatch.ts"),
					"--worker",
					"--calls",
					"5",
					"--warmup",
					"2",
					"--blocks",
					"1",
					"--session-dir",
					sessionDir,
					"--arguments",
					"invalid",
				],
				{ stdout: "pipe", stderr: "pipe" },
			);
			if (spawned.exitCode !== 0) {
				throw new Error(
					`worker exited ${spawned.exitCode}: ${new TextDecoder().decode(spawned.stderr).slice(-2_000)}`,
				);
			}
			const report = JSON.parse(new TextDecoder().decode(spawned.stdout).trim()) as WorkerReport;
			expect(validateWorkerReport(report, { implementation: "typescript", mode: "invalid", calls: 5 })).toBeNull();
		} finally {
			rmSync(sessionDir, { recursive: true, force: true });
		}
	});
});
