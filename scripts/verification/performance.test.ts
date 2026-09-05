import { afterAll, describe, expect, test } from "bun:test";
import { existsSync, mkdtempSync, readFileSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { spawnPty, type PtySnapshot } from "./pty.ts";
import {
	aggregateKeypressRounds,
	HarnessFailure,
	ProcTreeSampler,
	assembleProcessMemoryReading,
	distribution,
	exitCodeForFailure,
	frameObservation,
	isSharedCiEnvironment,
	keySyncTransaction,
	observeProcessTreeMemory,
	parseProcStatusPeakRssText,
	parseSmapsRollupText,
	planMemorySampleStarts,
	processTreeIdentity,
	recordedQuitTimeouts,
	recordEntrypointHarnessFailure,
	sampleProcessTreeMemoryWindow,
	settleExtensionStartup,
	terminateAndRequireCleanExit,
	STREAM_PTY_SIZE,
	validateMemoryCoverage,
	type KeypressRoundRecord,
} from "./performance.ts";
import {
	NOISE_EXIT_CODE,
	NOISE_RELATIVE_SPREAD_LIMIT,
	NoiseRejection,
	requireQuiet,
} from "../statistics.ts";

// T33: after capturing the first frame, the performance verifier sends /quit
// and requires a clean exit; a child that ignores /quit is a teardown
// problem, not a measurement failure, so the helper escalates to tree
// termination, keeps the captured sample, and records the escalation in
// recordedQuitTimeouts() (disclosed as harness.quitTimeouts). These tests
// exercise the same contract runFirstFrameSample now enforces, using
// synthetic children that emit a synchronized-output frame and then either
// honor or ignore /quit.

const SYNC_BEGIN = "\x1b[?2026h";
const SYNC_END = "\x1b[?2026l";
// spawnPty shells to util-linux setsid/script: absent on macOS and Windows.
const lacksUtilLinuxPty = process.platform !== "linux";
const bunExecutable = process.execPath;

const REPOSITORY_ROOT = resolve(import.meta.dirname, "../..");
const PERFORMANCE_MODULE = resolve(import.meta.dirname, "performance.ts");
const PERFORMANCE_ARTIFACT = resolve(REPOSITORY_ROOT, "target/bench/performance-comparison.json");

const temporaryPaths: string[] = [];

afterAll(() => {
	for (const path of temporaryPaths) rmSync(path, { recursive: true, force: true });
});

function temporaryDirectory(prefix: string): string {
	const path = mkdtempSync(join(tmpdir(), prefix));
	temporaryPaths.push(path);
	return path;
}

test("does not run the benchmark when performance verification is imported", () => {
	const sandbox = temporaryDirectory("perf-import-");
	const artifactBefore = existsSync(PERFORMANCE_ARTIFACT)
		? readFileSync(PERFORMANCE_ARTIFACT)
		: undefined;
	try {
		const imported = Bun.spawnSync(
			[
				bunExecutable,
				"-e",
				[
					"const exitCodeBeforeImport = process.exitCode;",
					`await import(${JSON.stringify(PERFORMANCE_MODULE)});`,
					"if (process.exitCode !== exitCodeBeforeImport) throw new Error('performance import changed process.exitCode');",
				].join("\n"),
			],
			{
				cwd: sandbox,
				env: { ...process.env, TMPDIR: sandbox },
				stdout: "pipe",
				stderr: "pipe",
				timeout: 10_000,
			},
		);
		expect(imported.exitCode).toBe(0);
		expect(new TextDecoder().decode(imported.stdout)).toBe("");
		expect(new TextDecoder().decode(imported.stderr)).toBe("");
		expect(readdirSync(sandbox).filter((entry) => entry.startsWith("pi-check9-"))).toEqual([]);
		const artifactExists = existsSync(PERFORMANCE_ARTIFACT);
		expect(artifactExists).toBe(artifactBefore !== undefined);
		if (artifactBefore !== undefined && artifactExists) {
			expect(Buffer.compare(readFileSync(PERFORMANCE_ARTIFACT), artifactBefore)).toBe(0);
		}
	} finally {
		const artifactExists = existsSync(PERFORMANCE_ARTIFACT);
		const artifactChanged =
			artifactExists !== (artifactBefore !== undefined) ||
			(artifactBefore !== undefined &&
				(!artifactExists || Buffer.compare(readFileSync(PERFORMANCE_ARTIFACT), artifactBefore) !== 0));
		if (artifactChanged) {
			if (artifactBefore !== undefined) writeFileSync(PERFORMANCE_ARTIFACT, artifactBefore);
			else rmSync(PERFORMANCE_ARTIFACT, { force: true });
		}
	}
}, 15_000);

const CLEAN_QUIT_CHILD = `
process.stdout.write(${JSON.stringify(SYNC_BEGIN)} + "frame" + ${JSON.stringify(SYNC_END)} + "\\n");
process.stdin.setEncoding("utf8");
process.stdin.resume();
const iterator = process.stdin[Symbol.asyncIterator]();
await iterator.next();
process.stdin.pause();
process.stdin.destroy();
process.exit(0);
`;

const IGNORE_QUIT_CHILD = `
process.stdout.write(${JSON.stringify(SYNC_BEGIN)} + "frame" + ${JSON.stringify(SYNC_END)} + "\\n");
process.stdin.setEncoding("utf8");
process.stdin.resume();
setInterval(() => {}, 1_000);
`;

const DELAYED_EXTENSIONS_CHILD = `
process.stdout.write(${JSON.stringify(SYNC_BEGIN)} + "first frame" + ${JSON.stringify(SYNC_END)} + "\\n");
process.stdin.setEncoding("utf8");
process.stdin.resume();
const iterator = process.stdin[Symbol.asyncIterator]();
await iterator.next();
process.stdout.write(${JSON.stringify(SYNC_BEGIN)} + "[Extensions]" + ${JSON.stringify(SYNC_END)} + "\\n");
await iterator.next();
`;

const TERMINAL_PROBE_CHILD = `
process.stdin.setRawMode?.(true);
process.stdin.setEncoding("utf8");
process.stdin.resume();
process.stdout.write("\\x1b[>1u\\x1b[?u\\x1b[c\\x1b[16t\\x1b]11;?\\x07\\x1b[6n");
const expected = ["\\x1b[?62;1;2;6;7;8;9c", "\\x1b[1;1R"];
let input = "";
for await (const chunk of process.stdin) {
	input += chunk;
	if (expected.every((reply) => input.includes(reply))) break;
}
process.stdout.write(${JSON.stringify(SYNC_BEGIN)} + "probe complete" + ${JSON.stringify(SYNC_END)} + "\\n");
`;

const CLEAN_EXIT_CHILD = "process.exit(0);";
const FAILURE_EXIT_CHILD = "process.exit(7);";

describe.skipIf(lacksUtilLinuxPty)("TypeScript extension startup settlement", () => {
	test("waits past the first frame for the extensions readiness marker", async () => {
		const sandbox = temporaryDirectory("perf-extension-startup-");
		const pty = spawnPty({
			argv: [bunExecutable, "-e", DELAYED_EXTENSIONS_CHILD],
			cwd: sandbox,
			size: { columns: 80, rows: 24 },
		});
		try {
			await pty.waitFor((candidate) => frameObservation(candidate) !== undefined, {
				deadlineMs: 5_000,
				source: "raw",
			});
			expect(pty.snapshot().rawText).not.toContain("[Extensions]");
			let settled = false;
			const settlement = settleExtensionStartup(pty, "typescript", "test:extension-startup").then(() => {
				settled = true;
			});
			await Promise.resolve();
			expect(settled).toBe(false);
			pty.writeKeys("r", "\r");
			await settlement;
			expect(pty.snapshot().rawText).toContain("[Extensions]");
		} finally {
			await pty.terminate();
		}
	}, 10_000);
});

describe.skipIf(lacksUtilLinuxPty)("performance first-frame lifecycle", () => {
	// Internal deadlines exercised by the ignore-quit test: 5_000ms frame
	// wait + 10_000ms /quit exit wait (terminateAndRequireCleanExit). The
	// test timeout is their sum plus 50% headroom so a slow runner fails
	// the assertion, not the harness timeout.
	const FRAME_WAIT_DEADLINE_MS = 5_000;
	const QUIT_EXIT_DEADLINE_MS = 10_000;
	const IGNORE_QUIT_TEST_TIMEOUT_MS = Math.round(
		(FRAME_WAIT_DEADLINE_MS + QUIT_EXIT_DEADLINE_MS) * 1.5,
	);
	test("accepts an already-settled clean exit through the production helper", async () => {
		const sandbox = temporaryDirectory("perf-settled-clean-");
		const pty = spawnPty({
			argv: [bunExecutable, "-e", CLEAN_EXIT_CHILD],
			cwd: sandbox,
			size: { columns: 100, rows: 32 },
		});
		try {
			expect(await pty.waitForExit(5_000)).toBe(0);
			await terminateAndRequireCleanExit(pty, "first-frame:settled-clean");
		} finally {
			await pty.terminate();
		}
	}, 15_000);

	test("rejects a nonzero already-settled exit through the production helper", async () => {
		const sandbox = temporaryDirectory("perf-settled-failure-");
		const pty = spawnPty({
			argv: [bunExecutable, "-e", FAILURE_EXIT_CHILD],
			cwd: sandbox,
			size: { columns: 100, rows: 32 },
		});
		try {
			expect(await pty.waitForExit(5_000)).toBe(7);
			await expect(terminateAndRequireCleanExit(pty, "first-frame:settled-failure")).rejects.toBeInstanceOf(
				HarnessFailure,
			);
		} finally {
			await pty.terminate();
		}
	}, 15_000);

	test("terminates a child that emits a frame but ignores /quit and discloses the escalation", async () => {
		const sandbox = temporaryDirectory("perf-ignore-quit-");
		const pty = spawnPty({
			argv: [bunExecutable, "-e", IGNORE_QUIT_CHILD],
			cwd: sandbox,
			size: { columns: 100, rows: 32 },
		});
		try {
			await pty.waitFor((candidate) => frameObservation(candidate) !== undefined, {
				deadlineMs: FRAME_WAIT_DEADLINE_MS,
				source: "raw",
			});
			const frame = frameObservation(pty.snapshot());
			expect(frame).toBeDefined();
			expect(frame?.bytes).toBeGreaterThan(0);
			const label = "first-frame:ignore-quit";
			await terminateAndRequireCleanExit(pty, label);
			expect(pty.exited).toBe(true);
			expect(recordedQuitTimeouts()).toContain(label);
		} finally {
			await pty.terminate();
		}
	}, IGNORE_QUIT_TEST_TIMEOUT_MS);

	test("passes a child that emits a frame and exits cleanly on /quit", async () => {
		const sandbox = temporaryDirectory("perf-clean-quit-");
		const pty = spawnPty({
			argv: [bunExecutable, "-e", CLEAN_QUIT_CHILD],
			cwd: sandbox,
			size: { columns: 100, rows: 32 },
		});
		try {
			const snapshot = await pty.waitFor((candidate) => frameObservation(candidate) !== undefined, {
				deadlineMs: 5_000,
				source: "raw",
			});
			const frame = frameObservation(snapshot);
			expect(frame).toBeDefined();
			expect(frame?.elapsedMs).toBeGreaterThanOrEqual(0);
			await terminateAndRequireCleanExit(pty, "first-frame:clean-quit");
			expect(pty.exited).toBe(true);
		} finally {
			await pty.terminate();
		}
	}, 15_000);
});

describe("distribution additive noise fields", () => {
	test("adds exact population stddev and median-relative spread", () => {
		const result = distribution([1, 2, 3, 4, 5]);
		expect(result.count).toBe(5);
		expect(result.median).toBe(3);
		expect(result.p95).toBe(4.8);
		expect(result.p99).toBe(4.96);
		expect(result.min).toBe(1);
		expect(result.max).toBe(5);
		expect(result.stddev).toBe(Math.sqrt(2));
		expect(result.relativeSpread).toBe(Math.sqrt(2) / 3);
	});

	test("constant zero is quiet; nonconstant median zero is undefined", () => {
		expect(distribution([0, 0, 0])).toEqual({
			count: 3,
			median: 0,
			p95: 0,
			p99: 0,
			min: 0,
			max: 0,
			stddev: 0,
			relativeSpread: 0,
		});
		const noisyZero = distribution([0, 0, 1]);
		expect(noisyZero.median).toBe(0);
		expect(noisyZero.stddev).toBeGreaterThan(0);
		expect(noisyZero.relativeSpread).toBeNull();
	});

	test("boundary relative spread 0.20 is quiet and epsilon above rejects", () => {
		expect(() =>
			requireQuiet([
				{
					label: "boundary",
					count: 2,
					median: 10,
					stddev: 2,
					relativeSpread: NOISE_RELATIVE_SPREAD_LIMIT,
				},
			]),
		).not.toThrow();
		expect(() =>
			requireQuiet([
				{
					label: "noisy",
					count: 2,
					median: 10,
					stddev: 2.0000001,
					relativeSpread: NOISE_RELATIVE_SPREAD_LIMIT + Number.EPSILON,
				},
			]),
		).toThrow(NoiseRejection);
	});
});

describe("exitCodeForFailure mapping", () => {
	test("maps NoiseRejection to 2 and other failures to 1", () => {
		const rejection = new NoiseRejection([
			{
				label: "x",
				count: 1,
				median: 1,
				stddev: 1,
				relativeSpread: 1,
			},
		]);
		expect(exitCodeForFailure(rejection)).toBe(NOISE_EXIT_CODE);
		expect(exitCodeForFailure(new HarnessFailure("statistics", "bad"))).toBe(1);
		expect(exitCodeForFailure(new Error("threshold-like"))).toBe(1);
		expect(rejection).toBeInstanceOf(Error);
		expect(rejection.name).toBe("NoiseRejection");
		expect(rejection instanceof HarnessFailure).toBe(false);
	});

	test("detects shared CI runners for the noise advisory", () => {
		expect(isSharedCiEnvironment({})).toBe(false);
		expect(isSharedCiEnvironment({ CI: "false" })).toBe(false);
		expect(isSharedCiEnvironment({ CI: "true" })).toBe(true);
		expect(isSharedCiEnvironment({ GITHUB_ACTIONS: "true" })).toBe(true);
	});
	});
describe("process memory parsers", () => {
	test("parseSmapsRollupText requires both fields and multiplies by 1024", () => {
		const parsed = parseSmapsRollupText("Rss: 10 kB\nPss: 7 kB\n");
		expect(parsed).toEqual({ rssBytes: 10 * 1024, pssBytes: 7 * 1024 });
		expect(parseSmapsRollupText("Rss: 10 kB\n")).toBeUndefined();
		expect(parseSmapsRollupText("Pss: 7 kB\n")).toBeUndefined();
		expect(parseSmapsRollupText("not-a-rollup")).toBeUndefined();
	});

	test("parseProcStatusPeakRssText reads VmHWM bytes or undefined", () => {
		expect(parseProcStatusPeakRssText("VmHWM:\t12 kB\nVmRSS:\t8 kB\n")).toBe(12 * 1024);
		expect(parseProcStatusPeakRssText("VmRSS:\t8 kB\n")).toBeUndefined();
		expect(parseProcStatusPeakRssText("garbage")).toBeUndefined();
	});
});

describe("process tree identity", () => {
	test("same pid with different startTime stays distinct", () => {
		expect(processTreeIdentity(42, "100")).toBe("42:100");
		expect(processTreeIdentity(42, "100")).not.toBe(processTreeIdentity(42, "101"));
	});
});

describe("timed CPU sampler purity", () => {
	test("ProcTreeSampler body does not read smaps_rollup or status", () => {
		const source = readFileSync(PERFORMANCE_MODULE, "utf8");
		const start = source.indexOf("export class ProcTreeSampler");
		const end = source.indexOf("function totalTicks", start);
		expect(start).toBeGreaterThanOrEqual(0);
		expect(end).toBeGreaterThan(start);
		const body = source.slice(start, end);
		expect(body).not.toContain("smaps_rollup");
		expect(body).not.toContain("/status");
		expect(body).not.toContain("parseSmaps");
		expect(body).not.toContain("observeProcessTreeMemory");
		expect(body).not.toContain("readProcFile");
	});

	test("snapshot is a pure read and does not advance procSamples", async () => {
		const child = Bun.spawn({
			cmd: [bunExecutable, "-e", "setInterval(() => {}, 1000)"],
			stdout: "ignore",
			stderr: "ignore",
		});
		try {
			const sampler = new ProcTreeSampler(child.pid!, 20);
			await Bun.sleep(25);
			const first = sampler.snapshot();
			const second = sampler.snapshot();
			expect(second.procSamples).toBe(first.procSamples);
			expect(second.observedProcesses).toBe(first.observedProcesses);
			expect(second.maxOwnTicks.size).toBe(first.maxOwnTicks.size);
			expect("memory" in second).toBe(false);
			await sampler.stop();
		} finally {
			child.kill();
			await child.exited;
		}
	}, 15_000);
});

describe.skipIf(lacksUtilLinuxPty)("terminal probe emulation", () => {
	test("answers completion-required probes through a real PTY", async () => {
		const sandbox = temporaryDirectory("perf-terminal-probes-");
		const pty = spawnPty({
			argv: [bunExecutable, "-e", TERMINAL_PROBE_CHILD],
			cwd: sandbox,
			size: { columns: 100, rows: 32 },
		});
		try {
			await pty.waitFor((snapshot) => snapshot.rawText.includes("probe complete"), {
				deadlineMs: 5_000,
				source: "raw",
			});
			expect(await pty.waitForExit(5_000)).toBe(0);
		} finally {
			await pty.terminate();
		}
	}, 15_000);

	test("first-frame collection uses the complete default probe profile", () => {
		const source = readFileSync(PERFORMANCE_MODULE, "utf8");
		const functionStart = source.indexOf("async function runFirstFrameSample(");
		const spawnStart = source.indexOf("const pty = spawnPty({", functionStart);
		const spawnEnd = source.indexOf("const sampler = new ProcTreeSampler", spawnStart);
		expect(source.slice(spawnStart, spawnEnd)).not.toContain("cursorPosition: false");
		expect(source).toContain('"verification PTY answers device-attribute and cursor-position');
	});
});

describe("stream PTY geometry", () => {
	test("both stream samplers use the shared 80x24 geometry", () => {
		expect(STREAM_PTY_SIZE).toEqual({ columns: 80, rows: 24 });
		const source = readFileSync(PERFORMANCE_MODULE, "utf8");
		const streamProcess = source.slice(
			source.indexOf("async function runStreamProcess("),
			source.indexOf("export interface KeypressRoundRecord"),
		);
		const streamMemory = source.slice(
			source.indexOf("async function runStreamLoadMemorySample("),
			source.indexOf("async function collectStreamLoadMemorySamples("),
		);
		expect(streamProcess).toContain("size: STREAM_PTY_SIZE");
		expect(streamMemory).toContain("size: STREAM_PTY_SIZE");
	});
});

describe("startup versus idle memory labels", () => {
	test("idle lane uses startupSumVmHwmBytes as non-simultaneous lifetime upper bound", () => {
		const source = readFileSync(PERFORMANCE_MODULE, "utf8");
		expect(source).toContain("startupSumVmHwmBytes");
		expect(source).toContain("steadyWindowMaxTreeRssBytes");
		expect(source).toContain("steadyWindowMaxTreePssBytes");
		expect(source).toContain("non-simultaneous sum of per-identity VmHWM");
		expect(source).not.toMatch(/startupPeakRssBytes/);
		expect(source).not.toMatch(/idlePeakRssBytes/);
		const idleArtifact = source.slice(source.indexOf("idleProcessTreeMemory"));
		expect(idleArtifact).toContain("startupSumVmHwm");
		expect(idleArtifact).toContain("steadyWindowRss");
		expect(idleArtifact).toContain("steadyWindowPss");
		expect(idleArtifact).toContain("lifetime upper bound");
		expect(idleArtifact).not.toMatch(/\bidlePeak/);
	});
});

describe("pre-memory verdict ordering", () => {
	test("evaluates requireQuiet and blockers before memory collectors", () => {
		const source = readFileSync(PERFORMANCE_MODULE, "utf8");
		const mainStart = source.indexOf("async function main()");
		const main = source.slice(mainStart);
		const quiet = main.indexOf("requireQuiet([");
		const evaluated = main.indexOf("const evaluatedVerdict");
		const streamMem = main.indexOf("collectStreamLoadMemorySamples()");
		const idleMem = main.indexOf("collectIdleMemorySamples()");
		expect(quiet).toBeGreaterThan(0);
		expect(evaluated).toBeGreaterThan(quiet);
		expect(streamMem).toBeGreaterThan(evaluated);
		expect(idleMem).toBeGreaterThan(streamMem);
		expect(main.indexOf("after evaluated verdict")).toBeGreaterThan(streamMem);
	});
});

describe("absolute memory sample schedule", () => {
	test("plans starts strictly before the deadline with no overrun slot", () => {
		expect(planMemorySampleStarts(1000, 50)).toEqual(
			Array.from({ length: 20 }, (_, index) => index * 50),
		);
		expect(planMemorySampleStarts(1000, 50).every((offset) => offset < 1000)).toBe(true);
		expect(planMemorySampleStarts(100, 50)).toEqual([0, 50]);
		expect(planMemorySampleStarts(50, 50)).toEqual([0]);
		expect(planMemorySampleStarts(0, 50)).toEqual([]);
	});
});

describe("process memory assembly policy", () => {
	test("discards identity races when reconfirm startTime differs", () => {
		const result = assembleProcessMemoryReading({
			pid: 7,
			initialStartTime: "100",
			root: false,
			smaps: { kind: "ok", text: "Rss: 10 kB\nPss: 7 kB\n" },
			status: { kind: "ok", text: "VmHWM:\t12 kB\n" },
			reconfirm: { kind: "ok", text: "7 (cmd) R 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 999" },
		});
		expect(result).toEqual({ kind: "discard-identity-race" });
	});

	test("live-root access denial is incomplete access-denied", () => {
		const result = assembleProcessMemoryReading({
			pid: 1,
			initialStartTime: "100",
			root: true,
			smaps: { kind: "access-denied" },
			status: { kind: "ok", text: "VmHWM:\t12 kB\n" },
			reconfirm: { kind: "ok", text: "1 (cmd) R 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 100" },
		});
		expect(result.kind).toBe("incomplete");
		if (result.kind === "incomplete") {
			expect(result.reason).toBe("access-denied");
		}
	});

	test("live-root parse failure is incomplete parse", () => {
		const result = assembleProcessMemoryReading({
			pid: 1,
			initialStartTime: "100",
			root: true,
			smaps: { kind: "ok", text: "Rss: 10 kB\n" },
			status: { kind: "ok", text: "VmHWM:\t12 kB\n" },
			reconfirm: { kind: "ok", text: "1 (cmd) R 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 100" },
		});
		expect(result.kind).toBe("incomplete");
		if (result.kind === "incomplete") {
			expect(result.reason).toBe("parse");
		}
	});

	test("descendant vanish after reconfirm is vanished not incomplete", () => {
		const result = assembleProcessMemoryReading({
			pid: 9,
			initialStartTime: "100",
			root: false,
			smaps: { kind: "ok", text: "Rss: 10 kB\nPss: 7 kB\n" },
			status: { kind: "ok", text: "VmHWM:\t12 kB\n" },
			reconfirm: { kind: "vanished" },
		});
		expect(result).toEqual({ kind: "vanished" });
	});

	test("validateMemoryCoverage fails when live identities lack complete memory", () => {
		expect(() =>
			validateMemoryCoverage(
				{
					processes: [],
					treeRssBytes: 0,
					treePssBytes: 0,
					sumPeakRssBytes: 0,
					observedLiveIdentities: 1,
					identitiesWithCompleteMemory: 0,
					vanishedDescendants: 0,
					coverageComplete: false,
				},
				"memory-coverage",
			),
		).toThrow(HarnessFailure);
	});
});

// Live /proc observation only exists on Linux (no /proc on macOS/Windows).
describe.skipIf(process.platform !== "linux")("process-tree memory observation", () => {
	test("populates complete coverage for a live synthetic root", async () => {
		const child = Bun.spawn({
			cmd: [bunExecutable, "-e", "setInterval(() => {}, 1000)"],
			stdout: "ignore",
			stderr: "ignore",
		});
		try {
			const observation = observeProcessTreeMemory(child.pid!, "synthetic-memory");
			expect(observation.coverageComplete).toBe(true);
			expect(observation.identitiesWithCompleteMemory).toBe(observation.observedLiveIdentities);
			expect(observation.treeRssBytes).toBeGreaterThan(0);
			expect(observation.treePssBytes).toBeGreaterThan(0);
			expect(observation.sumPeakRssBytes).toBeGreaterThan(0);
		} finally {
			child.kill();
			await child.exited;
		}
	}, 15_000);
});

describe("memory-path child enumeration policy", () => {
	function statText(pid: number, startTime: string): string {
		return `${pid} (cmd) R ${Array.from({ length: 18 }, () => "0").concat(startTime).join(" ")}`;
	}

	test("child EACCES on a live parent fails hard", () => {
		let statReads = 0;
		expect(() =>
			observeProcessTreeMemory(1, "child-eacces", {
				readProcFile: (path) => {
					if (path === "/proc/1/stat") {
						statReads += 1;
						return { kind: "ok", text: statText(1, "100") };
					}
					if (path.endsWith("smaps_rollup")) return { kind: "ok", text: "Rss: 10 kB\nPss: 7 kB\n" };
					if (path.endsWith("status")) return { kind: "ok", text: "VmHWM:\t12 kB\n" };
					return { kind: "vanished" };
				},
				enumerateChildren: () => ({ kind: "access-denied", detail: "/proc/1/task EACCES" }),
			}),
		).toThrow(/denied children enumeration/);
		expect(statReads).toBeGreaterThan(0);
	});

	test("reused parent identity does not enqueue children", () => {
		let childEnumCalls = 0;
		let statReads = 0;
		expect(() =>
			observeProcessTreeMemory(1, "identity-race", {
				readProcFile: (path) => {
					if (path === "/proc/1/stat") {
						statReads += 1;
						if (statReads === 1) return { kind: "ok", text: statText(1, "100") };
						return { kind: "ok", text: statText(1, "999") };
					}
					if (path.endsWith("smaps_rollup")) return { kind: "ok", text: "Rss: 10 kB\nPss: 7 kB\n" };
					if (path.endsWith("status")) return { kind: "ok", text: "VmHWM:\t12 kB\n" };
					return { kind: "vanished" };
				},
				enumerateChildren: () => {
					childEnumCalls += 1;
					return { kind: "ok", children: [2] };
				},
			}),
		).toThrow(HarnessFailure);
		expect(childEnumCalls).toBe(0);
	});

	test("zombie descendant with empty maps counts as vanished, not failure", () => {
		const stat = (pid: number, state: string, startTime: string) =>
			`${pid} (cmd) ${state} ${Array.from({ length: 18 }, () => "0").concat(startTime).join(" ")}`;
		const observation = observeProcessTreeMemory(1, "zombie-descendant", {
			readProcFile: (path) => {
				if (path === "/proc/1/stat") return { kind: "ok", text: stat(1, "R", "100") };
				if (path === "/proc/2/stat") return { kind: "ok", text: stat(2, "Z", "101") };
				if (path.endsWith("smaps_rollup")) {
					if (path.startsWith("/proc/2/")) return { kind: "ok", text: "" };
					return { kind: "ok", text: "Rss: 10 kB\nPss: 7 kB\n" };
				}
				if (path.endsWith("status")) {
					if (path.startsWith("/proc/2/")) return { kind: "ok", text: "Name:\tcmd\nState:\tZ (zombie)\n" };
					return { kind: "ok", text: "VmHWM:\t12 kB\n" };
				}
				return { kind: "vanished" };
			},
			enumerateChildren: (pid: number) => ({ kind: "ok", children: pid === 1 ? [2] : [] }),
		});
		expect(observation.coverageComplete).toBe(true);
		expect(observation.vanishedDescendants).toBe(1);
		expect(observation.identitiesWithCompleteMemory).toBe(1);
		expect(observation.processes.map((process) => process.pid)).toEqual([1]);
	});
});

describe("memory window coverage aggregation", () => {
	function statText(pid: number, startTime: string): string {
		return `${pid} (cmd) R ${Array.from({ length: 18 }, () => "0").concat(startTime).join(" ")}`;
	}

	test("records planned cadence, per-sample observations, and max-record indices", async () => {
		let ticks = 0;
		const rssKbValues = [10, 30, 20];
		const window = await sampleProcessTreeMemoryWindow(1, "agg-window", 150, 50, {
			readProcFile: (path) => {
				if (path.endsWith("/stat")) return { kind: "ok", text: statText(1, "100") };
				if (path.endsWith("smaps_rollup")) {
					const rssKb = rssKbValues[Math.min(ticks, rssKbValues.length - 1)]!;
					return { kind: "ok", text: `Rss: ${rssKb} kB\nPss: ${Math.floor(rssKb / 2)} kB\n` };
				}
				if (path.endsWith("status")) return { kind: "ok", text: "VmHWM:\t40 kB\n" };
				return { kind: "vanished" };
			},
			enumerateChildren: () => {
				ticks += 1;
				return { kind: "ok", children: [] };
			},
		});
		expect(window.plannedSampleStartsMs).toEqual([0, 50, 100]);
		expect(window.observations.length).toBe(window.aggregateCoverage.samples);
		expect(window.sampleStartOffsetsMs.length).toBe(window.observations.length);
		expect(window.sampleStartOffsetsMs.every((offset) => offset < 150)).toBe(true);
		expect(window.aggregateCoverage.allSamplesCoverageComplete).toBe(true);
		expect(window.aggregateCoverage.observedLiveIdentitiesMax).toBeGreaterThan(0);
		expect(window.maxTreeRss.sampleIndex).toBeGreaterThanOrEqual(0);
		expect(window.maxTreeRss.sampleIndex).toBeLessThan(window.observations.length);
		expect(window.observations[window.maxTreeRss.sampleIndex]?.treeRssBytes).toBe(window.maxTreeRss.bytes);
		expect(window.observations[window.maxTreePss.sampleIndex]?.treePssBytes).toBe(window.maxTreePss.bytes);
		expect(window.maxTreeRss.startOffsetMs).toBe(window.sampleStartOffsetsMs[window.maxTreeRss.sampleIndex] ?? -1);
		expect(window.maxTreeRss.bytes).toBe(30 * 1024);
	});
});

describe("post-enumeration parent identity recheck", () => {
	function statText(pid: number, startTime: string): string {
		return `${pid} (cmd) R ${Array.from({ length: 18 }, () => "0").concat(startTime).join(" ")}`;
	}

	test("parent startTime change after child enum discards children", () => {
		let parentStatReads = 0;
		let childReads = 0;
		const observation = observeProcessTreeMemory(1, "post-enum-change", {
			readProcFile: (path) => {
				if (path === "/proc/1/stat") {
					parentStatReads += 1;
					// initial + assemble reconfirm stay stable; post-enum recheck flips identity
					if (parentStatReads <= 2) return { kind: "ok", text: statText(1, "100") };
					return { kind: "ok", text: statText(1, "999") };
				}
				if (path === "/proc/2/stat" || path === "/proc/2/smaps_rollup" || path === "/proc/2/status") {
					childReads += 1;
					return { kind: "ok", text: path.endsWith("stat") ? statText(2, "200") : path.endsWith("status") ? "VmHWM:\t12 kB\n" : "Rss: 4 kB\nPss: 3 kB\n" };
				}
				if (path.endsWith("smaps_rollup")) return { kind: "ok", text: "Rss: 10 kB\nPss: 7 kB\n" };
				if (path.endsWith("status")) return { kind: "ok", text: "VmHWM:\t12 kB\n" };
				return { kind: "vanished" };
			},
			enumerateChildren: (pid) => {
				if (pid === 1) return { kind: "ok", children: [2] };
				return { kind: "ok", children: [] };
			},
		});
		expect(observation.processes.map((process) => process.pid)).toEqual([1]);
		expect(childReads).toBe(0);
		expect(parentStatReads).toBeGreaterThanOrEqual(3);
	});

	test("parent vanish after child enum discards children", () => {
		let parentStatReads = 0;
		let childReads = 0;
		const observation = observeProcessTreeMemory(1, "post-enum-vanish", {
			readProcFile: (path) => {
				if (path === "/proc/1/stat") {
					parentStatReads += 1;
					if (parentStatReads <= 2) return { kind: "ok", text: statText(1, "100") };
					return { kind: "vanished" };
				}
				if (path.startsWith("/proc/2/")) {
					childReads += 1;
					return { kind: "ok", text: path.endsWith("stat") ? statText(2, "200") : path.endsWith("status") ? "VmHWM:\t12 kB\n" : "Rss: 4 kB\nPss: 3 kB\n" };
				}
				if (path.endsWith("smaps_rollup")) return { kind: "ok", text: "Rss: 10 kB\nPss: 7 kB\n" };
				if (path.endsWith("status")) return { kind: "ok", text: "VmHWM:\t12 kB\n" };
				return { kind: "vanished" };
			},
			enumerateChildren: (pid) => (pid === 1 ? { kind: "ok", children: [2] } : { kind: "ok", children: [] }),
		});
		expect(observation.processes.map((process) => process.pid)).toEqual([1]);
		expect(childReads).toBe(0);
	});
});

describe("entrypoint harness failure preserves threshold blockers", () => {
	test("appends harness failure without replacing evaluated blockers", () => {
		const target: Parameters<typeof recordEntrypointHarnessFailure>[0] = {
			pass: true,
			blockers: [
				"Rust native keypress-to-paint p99: 12.000 ms >= required 10.000 ms (median 8.000 ms, p95 11.000 ms, 20 samples)",
			],
		};
		recordEntrypointHarnessFailure(
			target,
			new HarnessFailure("idle-memory", "process-tree memory coverage incomplete: 0/1 live identities had complete Rss/Pss/VmHWM"),
		);
		expect(target.pass).toBe(false);
		expect(target.blockers).toHaveLength(2);
		expect(target.blockers[0]).toContain("keypress-to-paint p99");
		expect(target.blockers[1]).toBe(
			"idle-memory: process-tree memory coverage incomplete: 0/1 live identities had complete Rss/Pss/VmHWM",
		);
		expect(target.failure).toEqual({
			stage: "idle-memory",
			message: "process-tree memory coverage incomplete: 0/1 live identities had complete Rss/Pss/VmHWM",
		});
	});

	test("entrypoint catch records via preserve-and-append helper", () => {
		const source = readFileSync(PERFORMANCE_MODULE, "utf8");
		const catchStart = source.indexOf("if (import.meta.main)");
		const catchBody = source.slice(catchStart);
		expect(catchBody).toContain("recordEntrypointHarnessFailure(artifact, failure)");
		expect(catchBody).not.toMatch(/artifact\.blockers = \[`\$\{stage\}: \$\{failure\.message\}`\]/);
	});
});
describe("strict keypress synchronized observer", () => {
	const encoder = new TextEncoder();

	function snapshotOf(...texts: readonly string[]): PtySnapshot {
		const chunks = texts.map((text, index) => ({
			stream: "pty" as const,
			text,
			bytes: encoder.encode(text),
			elapsedMs: (index + 1) * 5,
			unixMs: index + 1,
		}));
		const rawText = texts.join("");
		return {
			rawText,
			applicationText: rawText,
			echoText: "",
			chunks,
			elapsedMs: chunks.length * 5,
			exited: false,
			exitCode: null,
		};
	}

	test("returns the first balanced transaction with payload and completing-chunk timestamp across split markers", () => {
		expect(keySyncTransaction(snapshotOf(`junk${SYNC_BEGIN}`, "a\x1b[0mkey", `${SYNC_END}tail`), 0)).toEqual({
			kind: "transaction",
			payload: "a\x1b[0mkey",
			beginCount: 1,
			endCount: 1,
			elapsedMs: 15,
		});
	});

	test("ignores complete transactions before the receipt offset and scans only post-write bytes", () => {
		expect(
			keySyncTransaction(snapshotOf(`${SYNC_BEGIN}old${SYNC_END}`, `${SYNC_BEGIN}new${SYNC_END}`), 15),
		).toEqual({ kind: "transaction", payload: "new", beginCount: 1, endCount: 1, elapsedMs: 10 });
	});

	test("reports row-local printable output before any synchronized begin as a fallback, not a frame", () => {
		expect(keySyncTransaction(snapshotOf("\x1b[3;1Hcursor moved"), 0)).toEqual({ kind: "fallback", elapsedMs: 5 });
	});

	test("returns undefined while the markers are incomplete or absent", () => {
		expect(keySyncTransaction(snapshotOf(SYNC_BEGIN, "partial"), 0)).toBeUndefined();
		expect(keySyncTransaction(snapshotOf("plain text without control sequences"), 0)).toBeUndefined();
	});

	test("counts extra markers through the completing chunk so an extra completed frame can be rejected", () => {
		expect(
			keySyncTransaction(snapshotOf(`${SYNC_BEGIN}q${SYNC_END}${SYNC_BEGIN}extra${SYNC_END}`), 0),
		).toEqual({ kind: "transaction", payload: "q", beginCount: 2, endCount: 2, elapsedMs: 5 });
	});

	test("exposes the payload so the collector can require the typed key", () => {
		const observation = keySyncTransaction(snapshotOf(`${SYNC_BEGIN}editor [q] ready${SYNC_END}`), 0);
		expect(observation?.kind === "transaction" && observation.payload.includes("q")).toBe(true);
	});
});

describe("keypress round aggregation", () => {
	function syntheticRound(round: number, latencies: readonly number[]): KeypressRoundRecord {
		return {
			round,
			medianMs: distribution(latencies).median,
			samples: latencies.map((latencyMs, index) => ({
				latencyMs,
				key: String.fromCharCode(97 + (index % 26)),
				synchronizedFramesObserved: 1,
			})),
		};
	}

	test("summarizes all 27-style rounds without filtering any sample", () => {
		const rounds = [0, 1, 2].map((round) =>
			syntheticRound(
				round,
				Array.from({ length: 200 }, (_, index) => (index === 0 && round === 1 ? 999 : round + 1)),
			),
		);
		const aggregated = aggregateKeypressRounds(rounds);
		expect(aggregated.roundMedians).toEqual([1, 2, 3]);
		expect(aggregated.roundSummary.median).toBe(2);
		expect(aggregated.pooled.count).toBe(600);
		expect(aggregated.pooled.max).toBe(999);
	});

	test("rejects a partial round instead of summarizing missing samples", () => {
		const short = syntheticRound(0, Array.from({ length: 199 }, () => 1));
		expect(() => aggregateKeypressRounds([short])).toThrow(HarnessFailure);
	});

	test("round-median spread exactly at the 0.20 trust boundary stays quiet", () => {
		const spread = distribution([8, 12]);
		expect(spread.median).toBe(10);
		expect(spread.stddev).toBe(2);
		expect(spread.relativeSpread).toBe(0.2);
		expect(() =>
			requireQuiet([
				{ label: "boundary", count: spread.count, median: spread.median, stddev: spread.stddev, relativeSpread: spread.relativeSpread },
			]),
		).not.toThrow();
	});
});
