import { describe, expect, test } from "bun:test";
import {
	existsSync,
	linkSync,
	mkdirSync,
	mkdtempSync,
	readFileSync,
	readdirSync,
	renameSync,
	rmSync,
	unlinkSync,
	writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
	EXECUTION_MAP_DIRECTORY,
	computeExecutionMapGenerationId,
	extractExecutionMapBundle,
	loadCurrentExecutionMap,
	parseExecutionMapPointer,
	renderExecutionMapPointer,
} from "./map.ts";
import {
	type ExecutionMapFilesystem,
	type WitnessEnvelope,
	publishExecutionMap,
	renderExecutionMapGeneration,
} from "./publish-map.ts";
import { REPO_ROOT } from "./parity.ts";

// ============================================================================
// Fixtures and observable-state helpers
// ============================================================================

const GENERATION_ID = "ab".repeat(32);

function loadEnvelopeFixture(): WitnessEnvelope {
	const parsed: unknown = JSON.parse(loadCurrentExecutionMap(REPO_ROOT).witnessText);
	if (typeof parsed !== "object" || parsed === null) {
		throw new Error("execution-map witness fixture must parse into an object");
	}
	const envelope = parsed as WitnessEnvelope;
	if (!Array.isArray(envelope.records) || envelope.records.length === 0) {
		throw new Error("execution-map witness fixture must carry records");
	}
	return envelope;
}

const baseEnvelope: WitnessEnvelope = loadEnvelopeFixture();

/** A second valid publication: identical structure, different source hash. */
function envelopeTitled(envelope: WitnessEnvelope, titleSuffix: string): WitnessEnvelope {
	return {
		...envelope,
		records: envelope.records.map((record, index) =>
			index === 0 ? { ...record, title: `${record.title}${titleSuffix}` } : record,
		),
	};
}

function envelopeWithBogusKind(envelope: WitnessEnvelope): WitnessEnvelope {
	const parsed: unknown = JSON.parse(JSON.stringify(envelope));
	const container = parsed as { records: Array<{ kind?: unknown }> };
	const first = container.records[0];
	if (first === undefined) throw new Error("fixture must contain at least one record");
	first.kind = "bogus";
	return parsed as WitnessEnvelope;
}

function withTempRoot(run: (root: string) => void): void {
	const root = mkdtempSync(join(tmpdir(), "publish-execution-map-"));
	try {
		run(root);
	} finally {
		rmSync(root, { recursive: true, force: true });
	}
}

function pointerPath(root: string): string {
	return join(root, EXECUTION_MAP_DIRECTORY, "current.md");
}

function generationsPath(root: string): string {
	return join(root, EXECUTION_MAP_DIRECTORY, "generations");
}

function stagingPath(root: string): string {
	return join(root, EXECUTION_MAP_DIRECTORY, ".staging");
}

function installedGenerationIds(root: string): string[] {
	return readdirSync(generationsPath(root))
		.map((fileName) => {
			const match = /^([0-9a-f]{64})\.md$/.exec(fileName);
			if (match?.[1] === undefined) throw new Error(`unexpected generation file ${fileName}`);
			return match[1];
		})
		.sort();
}

function stageResidue(root: string): string[] {
	return readdirSync(stagingPath(root)).sort();
}

function realExecutionMapFilesystem(): ExecutionMapFilesystem {
	return {
		mkdir: (path) => mkdirSync(path, { recursive: true }),
		read: (path) => readFileSync(path, "utf8"),
		write: (path, data) => writeFileSync(path, data, "utf8"),
		link: linkSync,
		rename: renameSync,
		unlink: unlinkSync,
		exists: existsSync,
	};
}

function caughtFailure(run: () => unknown): unknown {
	try {
		run();
	} catch (error) {
		return error;
	}
	throw new Error("expected the operation to fail, but it succeeded");
}

function expectAggregateOf(failure: unknown, expectedMessages: readonly string[]): void {
	if (!(failure instanceof AggregateError)) {
		throw new Error(`expected AggregateError, got ${String(failure)}`);
	}
	expect(failure.errors.length).toBe(expectedMessages.length);
	for (const [index, expected] of expectedMessages.entries()) {
		const error = failure.errors[index];
		expect(error === undefined ? "" : String(error)).toContain(expected);
	}
}

// ============================================================================
// Deterministic delegating fault adapter
// ============================================================================

type SeamOperation = keyof ExecutionMapFilesystem;

interface ScriptedFault {
	readonly operation: SeamOperation;
	readonly matches: (paths: readonly string[]) => boolean;
	readonly mode: "before" | "after" | "partial-write";
	readonly error: Error;
	timesRemaining: number;
}

function injectFault(
	operation: SeamOperation,
	pathIncludes: string,
	mode: ScriptedFault["mode"],
	message: string,
): ScriptedFault {
	return {
		operation,
		matches: (paths) => paths.some((path) => path.replaceAll("\\", "/").includes(pathIncludes)),
		mode,
		error: new Error(message),
		timesRemaining: 1,
	};
}

/**
 * Delegates every operation to the real filesystem inside a temporary
 * directory and fires scripted faults at exact (operation, path) sites.
 * Faults are deterministic: first matching scripted fault wins, each fires
 * once. "after" lets the real effect happen before throwing; "partial-write"
 * writes half the bytes before throwing.
 */
class DeterministicFaultFilesystem implements ExecutionMapFilesystem {
	readonly journal: Array<{ operation: SeamOperation; paths: readonly string[] }> = [];

	constructor(
		private readonly inner: ExecutionMapFilesystem,
		private readonly faults: readonly ScriptedFault[],
	) {}

	private takeFault(operation: SeamOperation, paths: readonly string[]): ScriptedFault | null {
		for (const fault of this.faults) {
			if (fault.timesRemaining > 0 && fault.operation === operation && fault.matches(paths)) {
				fault.timesRemaining -= 1;
				return fault;
			}
		}
		return null;
	}

	private record(operation: SeamOperation, paths: readonly string[]): void {
		this.journal.push({ operation, paths });
	}

	mkdir(path: string): void {
		this.record("mkdir", [path]);
		const fault = this.takeFault("mkdir", [path]);
		if (fault !== null) throw fault.error;
		this.inner.mkdir(path);
	}

	read(path: string): string {
		this.record("read", [path]);
		return this.inner.read(path);
	}

	write(path: string, data: string): void {
		this.record("write", [path]);
		const fault = this.takeFault("write", [path]);
		if (fault?.mode === "before") throw fault.error;
		if (fault?.mode === "partial-write") {
			this.inner.write(path, data.slice(0, Math.max(1, Math.floor(data.length / 2))));
			throw fault.error;
		}
		this.inner.write(path, data);
		if (fault !== null) throw fault.error;
	}

	link(existingPath: string, newPath: string): void {
		this.record("link", [existingPath, newPath]);
		const fault = this.takeFault("link", [existingPath, newPath]);
		if (fault?.mode === "before") throw fault.error;
		this.inner.link(existingPath, newPath);
		if (fault !== null) throw fault.error;
	}

	rename(from: string, to: string): void {
		this.record("rename", [from, to]);
		const fault = this.takeFault("rename", [from, to]);
		if (fault?.mode === "before") throw fault.error;
		this.inner.rename(from, to);
		if (fault !== null) throw fault.error;
	}

	unlink(path: string): void {
		this.record("unlink", [path]);
		const fault = this.takeFault("unlink", [path]);
		if (fault !== null) throw fault.error;
		this.inner.unlink(path);
	}

	exists(path: string): boolean {
		this.record("exists", [path]);
		return this.inner.exists(path);
	}
}

// ============================================================================
// Pointer and bundle grammar (completed tracer)
// ============================================================================

describe("execution-map publication pointer grammar", () => {
	test("publication pointer names exactly one content-addressed generation", () => {
		const pointer = `[Current execution map](generations/${GENERATION_ID}.md)\n`;
		expect(parseExecutionMapPointer(pointer)).toBe(GENERATION_ID);
		expect(renderExecutionMapPointer(GENERATION_ID)).toBe(pointer);
		expect(() => parseExecutionMapPointer(pointer.trimEnd())).toThrow("malformed execution-map pointer");
		expect(() => parseExecutionMapPointer(pointer.replace(GENERATION_ID, GENERATION_ID.toUpperCase()))).toThrow(
			"malformed execution-map pointer",
		);
		expect(() => parseExecutionMapPointer(`${pointer}${pointer}`)).toThrow("malformed execution-map pointer");
	});

	test("generation bundle grammar is strict and the publisher's render round-trips", () => {
		const handBuilt = "# Execution map\n\nmulti\nline\nmap\n\n## Canonical witness\n\n```json\n{\"version\":2}\n```\n";
		const bundle = extractExecutionMapBundle(handBuilt);
		expect(bundle.mapText).toBe("# Execution map\n\nmulti\nline\nmap\n");
		expect(bundle.witnessText).toBe("{\"version\":2}\n");
		expect(computeExecutionMapGenerationId(handBuilt)).toMatch(/^[0-9a-f]{64}$/);

		const generation = renderExecutionMapGeneration(baseEnvelope);
		expect(generation.generationId).toBe(computeExecutionMapGenerationId(generation.bundleText));
		expect(generation.bundleText.endsWith("```\n")).toBe(true);
		expect(generation.bundleText.split("## Canonical witness").length).toBe(2);
		const rendered = extractExecutionMapBundle(generation.bundleText);
		expect(rendered.witnessText.endsWith("\n")).toBe(true);
		const witness = JSON.parse(rendered.witnessText) as { version: number };
		expect(witness.version).toBe(2);
	});

	test("tracked current generation is the exact deterministic render of its witness", () => {
		const current = loadCurrentExecutionMap(REPO_ROOT);
		const rendered = renderExecutionMapGeneration(baseEnvelope);
		expect(rendered.generationId).toBe(current.generationId);
		expect(rendered.bundleText).toBe(current.bundleText);
	});
});

// ============================================================================
// Publication state machine: observable behavior over real directories
// ============================================================================

describe("execution-map publication state machine", () => {
	test("invalid input rejects before any filesystem action", () => {
		withTempRoot((root) => {
			const filesystem = new DeterministicFaultFilesystem(realExecutionMapFilesystem(), []);
			const failure = caughtFailure(() => publishExecutionMap(envelopeWithBogusKind(baseEnvelope), filesystem, root));
			expect(String(failure)).toContain("validation problems");
			expect(filesystem.journal).toEqual([]);
			expect(existsSync(join(root, EXECUTION_MAP_DIRECTORY))).toBe(false);
		});
	});

	test("staging-directory creation failure installs nothing and selects nothing", () => {
		withTempRoot((root) => {
			const filesystem = new DeterministicFaultFilesystem(realExecutionMapFilesystem(), [
				injectFault("mkdir", ".staging", "before", "injected staging-directory failure"),
			]);
			const failure = caughtFailure(() => publishExecutionMap(baseEnvelope, filesystem, root));
			expect(String(failure)).toContain("injected staging-directory failure");
			expect(existsSync(generationsPath(root))).toBe(true);
			expect(existsSync(stagingPath(root))).toBe(false);
			expect(existsSync(pointerPath(root))).toBe(false);
		});
	});

	test("failed generation staging write cleans the stage and installs nothing", () => {
		withTempRoot((root) => {
			const filesystem = new DeterministicFaultFilesystem(realExecutionMapFilesystem(), [
				injectFault("write", ".staging/generation-", "before", "injected generation stage write failure"),
			]);
			const failure = caughtFailure(() => publishExecutionMap(baseEnvelope, filesystem, root));
			expect(String(failure)).toContain("injected generation stage write failure");
			expect(installedGenerationIds(root)).toEqual([]);
			expect(stageResidue(root)).toEqual([]);
			expect(existsSync(pointerPath(root))).toBe(false);
		});
	});

	test("partial generation staging write leaves no residue and installs nothing", () => {
		withTempRoot((root) => {
			const filesystem = new DeterministicFaultFilesystem(realExecutionMapFilesystem(), [
				injectFault("write", ".staging/generation-", "partial-write", "injected partial generation stage write"),
			]);
			const failure = caughtFailure(() => publishExecutionMap(baseEnvelope, filesystem, root));
			expect(String(failure)).toContain("injected partial generation stage write");
			expect(installedGenerationIds(root)).toEqual([]);
			expect(stageResidue(root)).toEqual([]);
			expect(existsSync(pointerPath(root))).toBe(false);
		});
	});

	test("generation link failure before effect fails closed and cleans the stage", () => {
		withTempRoot((root) => {
			const filesystem = new DeterministicFaultFilesystem(realExecutionMapFilesystem(), [
				injectFault("link", "generations", "before", "injected generation link failure"),
			]);
			const failure = caughtFailure(() => publishExecutionMap(baseEnvelope, filesystem, root));
			expect(String(failure)).toContain("injected generation link failure");
			expect(installedGenerationIds(root)).toEqual([]);
			expect(stageResidue(root)).toEqual([]);
			expect(existsSync(pointerPath(root))).toBe(false);
		});
	});

	test("generation link failure after effect reconciles exact bytes and completes publication", () => {
		withTempRoot((root) => {
			const filesystem = new DeterministicFaultFilesystem(realExecutionMapFilesystem(), [
				injectFault("link", "generations", "after", "injected post-link failure"),
			]);
			const result = publishExecutionMap(baseEnvelope, filesystem, root);
			expect(installedGenerationIds(root)).toEqual([result.generationId]);
			const installed = readFileSync(join(generationsPath(root), `${result.generationId}.md`), "utf8");
			expect(installed).toBe(result.bundleText);
			expect(computeExecutionMapGenerationId(installed)).toBe(result.generationId);
			expect(parseExecutionMapPointer(readFileSync(pointerPath(root), "utf8"))).toBe(result.generationId);
			expect(stageResidue(root)).toEqual([]);
		});
	});

	test("existing corrupt generation fails closed without selecting anything", () => {
		withTempRoot((root) => {
			const generation = renderExecutionMapGeneration(baseEnvelope);
			mkdirSync(generationsPath(root), { recursive: true });
			const target = join(generationsPath(root), `${generation.generationId}.md`);
			const corruptBytes = `${generation.bundleText}CORRUPTION\n`;
			writeFileSync(target, corruptBytes, "utf8");
			caughtFailure(() => publishExecutionMap(baseEnvelope, realExecutionMapFilesystem(), root));
			expect(readFileSync(target, "utf8")).toBe(corruptBytes);
			expect(existsSync(pointerPath(root))).toBe(false);
			expect(stageResidue(root)).toEqual([]);
		});
	});

	test("generation-stage cleanup failure is surfaced together with the primary failure", () => {
		withTempRoot((root) => {
			const filesystem = new DeterministicFaultFilesystem(realExecutionMapFilesystem(), [
				injectFault("link", "generations", "before", "injected generation link failure"),
				injectFault("unlink", ".staging/generation-", "before", "injected generation stage cleanup failure"),
			]);
			const failure = caughtFailure(() => publishExecutionMap(baseEnvelope, filesystem, root));
			expectAggregateOf(failure, ["injected generation link failure", "injected generation stage cleanup failure"]);
			expect(stageResidue(root).length).toBe(1);
			expect(installedGenerationIds(root)).toEqual([]);
			expect(existsSync(pointerPath(root))).toBe(false);
		});
	});

	test("failed pointer staging write retains the generation but selects nothing", () => {
		withTempRoot((root) => {
			const filesystem = new DeterministicFaultFilesystem(realExecutionMapFilesystem(), [
				injectFault("write", ".staging/pointer-", "before", "injected pointer stage write failure"),
			]);
			const failure = caughtFailure(() => publishExecutionMap(baseEnvelope, filesystem, root));
			expect(String(failure)).toContain("injected pointer stage write failure");
			expect(installedGenerationIds(root).length).toBe(1);
			expect(stageResidue(root)).toEqual([]);
			expect(existsSync(pointerPath(root))).toBe(false);
		});
	});

	test("partial pointer staging write leaves no residue and selects nothing", () => {
		withTempRoot((root) => {
			const filesystem = new DeterministicFaultFilesystem(realExecutionMapFilesystem(), [
				injectFault("write", ".staging/pointer-", "partial-write", "injected partial pointer stage write"),
			]);
			const failure = caughtFailure(() => publishExecutionMap(baseEnvelope, filesystem, root));
			expect(String(failure)).toContain("injected partial pointer stage write");
			expect(installedGenerationIds(root).length).toBe(1);
			expect(stageResidue(root)).toEqual([]);
			expect(existsSync(pointerPath(root))).toBe(false);
		});
	});

	test("pointer swap failure before effect keeps the pointer unselected and cleans the stage", () => {
		withTempRoot((root) => {
			const filesystem = new DeterministicFaultFilesystem(realExecutionMapFilesystem(), [
				injectFault("rename", "current.md", "before", "injected pointer rename failure"),
			]);
			const failure = caughtFailure(() => publishExecutionMap(baseEnvelope, filesystem, root));
			expect(String(failure)).toContain("injected pointer rename failure");
			expect(installedGenerationIds(root).length).toBe(1);
			expect(stageResidue(root)).toEqual([]);
			expect(existsSync(pointerPath(root))).toBe(false);
		});
	});

	test("pointer swap failure after effect reconciles the committed selection", () => {
		withTempRoot((root) => {
			const filesystem = new DeterministicFaultFilesystem(realExecutionMapFilesystem(), [
				injectFault("rename", "current.md", "after", "injected post-rename failure"),
			]);
			const result = publishExecutionMap(baseEnvelope, filesystem, root);
			expect(parseExecutionMapPointer(readFileSync(pointerPath(root), "utf8"))).toBe(result.generationId);
			expect(stageResidue(root)).toEqual([]);
			const loaded = loadCurrentExecutionMap(root);
			expect(loaded.generationId).toBe(result.generationId);
		});
	});

	test("pointer-stage cleanup failure is surfaced together with the primary failure", () => {
		withTempRoot((root) => {
			const filesystem = new DeterministicFaultFilesystem(realExecutionMapFilesystem(), [
				injectFault("rename", "current.md", "before", "injected pointer rename failure"),
				injectFault("unlink", ".staging/pointer-", "before", "injected pointer stage cleanup failure"),
			]);
			const failure = caughtFailure(() => publishExecutionMap(baseEnvelope, filesystem, root));
			expectAggregateOf(failure, ["injected pointer rename failure", "injected pointer stage cleanup failure"]);
			expect(stageResidue(root).length).toBe(1);
			expect(installedGenerationIds(root).length).toBe(1);
			expect(existsSync(pointerPath(root))).toBe(false);
		});
	});
});

// ============================================================================
// Publication lifecycle over real directories
// ============================================================================

describe("execution-map publication lifecycle", () => {
	test("first publication installs one exact generation and selects it through the pointer", () => {
		withTempRoot((root) => {
			const result = publishExecutionMap(baseEnvelope, realExecutionMapFilesystem(), root);
			expect(installedGenerationIds(root)).toEqual([result.generationId]);
			const installed = readFileSync(join(generationsPath(root), `${result.generationId}.md`), "utf8");
			expect(installed).toBe(result.bundleText);
			expect(readFileSync(pointerPath(root), "utf8")).toBe(renderExecutionMapPointer(result.generationId));
			const loaded = loadCurrentExecutionMap(root);
			expect(loaded.generationId).toBe(result.generationId);
			expect(loaded.mapText).toBe(extractExecutionMapBundle(result.bundleText).mapText);
			const witness = JSON.parse(loaded.witnessText) as { version: number };
			expect(witness.version).toBe(2);
		});
	});

	test("repeated identical publication over an existing valid generation keeps one file", () => {
		withTempRoot((root) => {
			const first = publishExecutionMap(baseEnvelope, realExecutionMapFilesystem(), root);
			const pointerBytes = readFileSync(pointerPath(root), "utf8");
			const second = publishExecutionMap(baseEnvelope, realExecutionMapFilesystem(), root);
			expect(second.generationId).toBe(first.generationId);
			expect(installedGenerationIds(root)).toEqual([first.generationId]);
			expect(readFileSync(pointerPath(root), "utf8")).toBe(pointerBytes);
		});
	});

	test("last completed pointer swap wins across distinct publications in either order", () => {
		withTempRoot((root) => {
			const alpha = publishExecutionMap(envelopeTitled(baseEnvelope, " (alpha)"), realExecutionMapFilesystem(), root);
			const beta = publishExecutionMap(envelopeTitled(baseEnvelope, " (beta)"), realExecutionMapFilesystem(), root);
			expect(alpha.generationId).not.toBe(beta.generationId);
			expect(installedGenerationIds(root).sort()).toEqual([alpha.generationId, beta.generationId].sort());
			expect(parseExecutionMapPointer(readFileSync(pointerPath(root), "utf8"))).toBe(beta.generationId);
		});
		withTempRoot((reversedRoot) => {
			const beta = publishExecutionMap(envelopeTitled(baseEnvelope, " (beta)"), realExecutionMapFilesystem(), reversedRoot);
			const alpha = publishExecutionMap(envelopeTitled(baseEnvelope, " (alpha)"), realExecutionMapFilesystem(), reversedRoot);
			expect(parseExecutionMapPointer(readFileSync(pointerPath(reversedRoot), "utf8"))).toBe(alpha.generationId);
		});
	});

	test("ignored staging residue never affects the reader or the publisher", () => {
		withTempRoot((root) => {
			const result = publishExecutionMap(baseEnvelope, realExecutionMapFilesystem(), root);
			mkdirSync(stagingPath(root), { recursive: true });
			writeFileSync(join(stagingPath(root), "generation-crashleft.md"), "half-written residue", "utf8");
			expect(loadCurrentExecutionMap(root).generationId).toBe(result.generationId);
			const again = publishExecutionMap(baseEnvelope, realExecutionMapFilesystem(), root);
			expect(again.generationId).toBe(result.generationId);
			expect(stageResidue(root)).toEqual(["generation-crashleft.md"]);
		});
	});
});

// ============================================================================
// Strict pointer reader: fail closed, no fallback
// ============================================================================

describe("current-generation pointer reader", () => {
	test("malformed pointer, missing generation, and digest mismatch fail closed", () => {
		withTempRoot((root) => {
			const result = publishExecutionMap(baseEnvelope, realExecutionMapFilesystem(), root);

			writeFileSync(pointerPath(root), "not a pointer\n", "utf8");
			let failure = caughtFailure(() => loadCurrentExecutionMap(root));
			expect(String(failure)).toContain("malformed execution-map pointer");

			writeFileSync(pointerPath(root), renderExecutionMapPointer("cd".repeat(32)), "utf8");
			failure = caughtFailure(() => loadCurrentExecutionMap(root));
			expect(String(failure)).toContain("generations");
			expect(String(failure)).toContain("cannot read required");

			const target = join(generationsPath(root), `${result.generationId}.md`);
			writeFileSync(target, `${result.bundleText}drift\n`, "utf8");
			writeFileSync(pointerPath(root), renderExecutionMapPointer(result.generationId), "utf8");
			failure = caughtFailure(() => loadCurrentExecutionMap(root));
			expect(String(failure)).toContain("digest mismatch");
		});
	});
});
