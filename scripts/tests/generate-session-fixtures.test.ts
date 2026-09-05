import { describe, expect, test } from "bun:test";
import {
	mkdir,
	mkdtemp,
	readFile,
	readdir,
	rm,
	stat,
	writeFile,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, sep } from "node:path";

import {
	generateSessionFixtures,
	pruneStaleFixturePairs,
	type BuiltFixture,
} from "../generate-session-fixtures.ts";

const BASE_ISO = "2025-01-01T00:00:00.000Z";
const BASE_MS = Date.parse(BASE_ISO);
const FAKE_CWD = "/tmp/pi-fixtures/cwd";

/** Minimal v3 fixture that satisfies validateFixture without the reference. */
function fakeFixture(rel: string, index: number): BuiltFixture {
	const sessionId = `00000000-0000-7000-8000-${String(index).padStart(12, "0")}`;
	const message = {
		type: "message",
		id: "e0000001",
		parentId: null,
		timestamp: new Date(BASE_MS + 1000).toISOString(),
		message: {
			role: "user",
			content: [{ type: "text", text: "fake fixture" }],
			timestamp: BASE_MS + 1000,
		},
	};
	const header = {
		type: "session" as const,
		version: 3,
		id: sessionId,
		timestamp: BASE_ISO,
		cwd: FAKE_CWD,
	};
	return {
		rel,
		formatVersion: 3,
		sessionId,
		cwd: FAKE_CWD,
		parentSession: null,
		lines: [JSON.stringify(header), JSON.stringify(message)],
		view: {
			header,
			entries: [message],
			tree: [{ entry: message, children: [] }],
			context: {
				messages: [message.message],
				thinkingLevel: "off",
				model: null,
			},
			labels: {},
			name: null,
			leaf: "e0000001",
		},
	};
}

async function listFiles(dir: string): Promise<string[]> {
	const names = await readdir(dir, { recursive: true });
	const files: string[] = [];
	for (const name of names) {
		const rel = String(name);
		if ((await stat(join(dir, rel))).isFile()) files.push(rel.split(sep).join("/"));
	}
	return files.sort();
}

describe("generateSessionFixtures stale-pair pruning", () => {
	test("renamed/removed fixtures leave no stale pair behind", async () => {
		const outDir = await mkdtemp(join(tmpdir(), "pi-session-fixtures-prune-"));
		try {
			// Dirty checkout: stale pair from a renamed scenario, a lone
			// expected file whose jsonl is already gone, plus unrelated
			// files that pruning must never touch.
			await mkdir(join(outDir, "v3"), { recursive: true });
			await writeFile(join(outDir, "v3", "renamed-away.jsonl"), '{"type":"session"}\n');
			await writeFile(join(outDir, "v3", "renamed-away.expected.json"), "{}\n");
			await writeFile(join(outDir, "v3", "orphan.expected.json"), "{}\n");
			await writeFile(join(outDir, "v3", "notes.txt"), "not a fixture\n");
			await writeFile(join(outDir, "README.md"), "human notes\n");

			const first = await generateSessionFixtures({
				outDir,
				fixtures: [fakeFixture("v3/basic.jsonl", 1)],
			});

			expect(first.pruned.sort()).toEqual([
				"v3/orphan.expected.json",
				"v3/renamed-away.expected.json",
				"v3/renamed-away.jsonl",
			]);

			const tree = await listFiles(outDir);
			expect(tree).toEqual([
				"README.md",
				"manifest.json",
				"v3/basic.expected.json",
				"v3/basic.jsonl",
				"v3/notes.txt",
			]);

			// Manifest count matches the current files on disk.
			const manifest = JSON.parse(
				await readFile(join(outDir, "manifest.json"), "utf8"),
			);
			const jsonlFiles = tree.filter((f) => f.endsWith(".jsonl"));
			expect(manifest.count).toBe(jsonlFiles.length);
			expect(manifest.fixtures).toEqual(jsonlFiles);

			// A rerun on the converged tree prunes nothing.
			const second = await generateSessionFixtures({
				outDir,
				fixtures: [fakeFixture("v3/basic.jsonl", 1)],
			});
			expect(second.pruned).toEqual([]);
			expect(await listFiles(outDir)).toEqual(tree);
		} finally {
			await rm(outDir, { recursive: true, force: true });
		}
	}, 60_000);

	test("a failed generation preserves prior outputs and manifest", async () => {
		const outDir = await mkdtemp(join(tmpdir(), "pi-session-fixtures-fail-"));
		try {
			await generateSessionFixtures({
				outDir,
				fixtures: [fakeFixture("v3/basic.jsonl", 1)],
			});
			const priorTree = await listFiles(outDir);
			const priorManifest = await readFile(join(outDir, "manifest.json"), "utf8");

			// Second fixture dies inside the write loop (after the first
			// pair was written, before prune/manifest). A real
			// writeAtomically I/O error exits the CLI at the same point by
			// construction, so neither path can prune or republish.
			const boom = fakeFixture("v3/boom.jsonl", 2);
			Object.defineProperty(boom, "lines", {
				get(): never {
					throw new Error("synthetic write-phase failure");
				},
			});

			await expect(
				generateSessionFixtures({
					outDir,
					fixtures: [fakeFixture("v3/basic.jsonl", 1), boom],
				}),
			).rejects.toThrow("synthetic write-phase failure");

			expect(await readFile(join(outDir, "manifest.json"), "utf8")).toBe(
				priorManifest,
			);
			expect(await listFiles(outDir)).toEqual(priorTree);
		} finally {
			await rm(outDir, { recursive: true, force: true });
		}
	}, 60_000);
});

describe("pruneStaleFixturePairs", () => {
	test("removes only unlisted fixture pairs and never the output directory", async () => {
		const outDir = await mkdtemp(join(tmpdir(), "pi-session-fixtures-unit-"));
		try {
			await mkdir(join(outDir, "v1"), { recursive: true });
			await mkdir(join(outDir, "v3"), { recursive: true });
			await writeFile(join(outDir, "v1", "keep.jsonl"), "{}\n");
			await writeFile(join(outDir, "v1", "keep.expected.json"), "{}\n");
			await writeFile(join(outDir, "v3", "gone.jsonl"), "{}\n");
			await writeFile(join(outDir, "v3", "gone.expected.json"), "{}\n");
			await writeFile(join(outDir, "v3", "lonely.expected.json"), "{}\n");
			await writeFile(join(outDir, "v3", "stray.jsonl.tmp"), "{}\n");
			await writeFile(join(outDir, "notes.md"), "keep me\n");

			const removed = await pruneStaleFixturePairs(outDir, ["v1/keep.jsonl"]);

			expect(removed.sort()).toEqual([
				"v3/gone.expected.json",
				"v3/gone.jsonl",
				"v3/lonely.expected.json",
			]);
			expect(await listFiles(outDir)).toEqual([
				"notes.md",
				"v1/keep.expected.json",
				"v1/keep.jsonl",
				"v3/stray.jsonl.tmp",
			]);

			// Even with nothing listed, unrelated files survive and the
			// output directory itself is retained.
			const removedAll = await pruneStaleFixturePairs(outDir, []);
			expect(removedAll.sort()).toEqual([
				"v1/keep.expected.json",
				"v1/keep.jsonl",
			]);
			expect((await stat(outDir)).isDirectory()).toBe(true);
			expect(await listFiles(outDir)).toEqual([
				"notes.md",
				"v3/stray.jsonl.tmp",
			]);
		} finally {
			await rm(outDir, { recursive: true, force: true });
		}
	});
});
