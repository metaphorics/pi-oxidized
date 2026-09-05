import { afterAll, describe, expect, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { PTY_KEYS, spawnPty } from "./pty.ts";

// spawnPty shells to util-linux setsid/script: absent on macOS and Windows.
const lacksUtilLinuxPty = process.platform !== "linux";


const temporaryPaths: string[] = [];

afterAll(() => {
	for (const path of temporaryPaths) rmSync(path, { recursive: true, force: true });
});

function temporaryDirectory(prefix: string): string {
	const path = mkdtempSync(join(tmpdir(), prefix));
	temporaryPaths.push(path);
	return path;
}

describe.skipIf(lacksUtilLinuxPty)("PTY driver", () => {
	test("preserves hostile argv, separates terminal echo, timestamps chunks, and exits cleanly", async () => {
		const root = temporaryDirectory("pi pty ' $() ");
		const process = spawnPty({
			argv: ["/bin/sh", "-c", "stty echo; printf APP_READY; IFS= read -r line; printf '\\nAPP:%s\\n' \"$line\""],
			cwd: root,
		});
		try {
			await process.waitFor(/APP_READY/, { deadlineMs: 5_000, source: "raw" });
			process.writeKeys("ECHO_SENT", PTY_KEYS.enter);
			const snapshot = await process.waitFor(/APP:ECHO_SENT/, { deadlineMs: 5_000 });
			expect(snapshot.echoText).toContain("ECHO_SENT");
			expect(snapshot.applicationText).toContain("APP:ECHO_SENT");
			expect(snapshot.chunks.length).toBeGreaterThan(0);
			expect(snapshot.chunks.every((chunk) => chunk.elapsedMs >= 0 && chunk.unixMs > 0)).toBe(true);
			expect(await process.waitForExit(5_000)).toBe(0);
		} finally {
			await process.terminate();
		}
	}, 15_000);

	test("terminates the complete PTY process group", async () => {
		const process = spawnPty({
			argv: ["/bin/sh", "-c", "sleep 300 & printf 'CHILD:%s\\n' \"$!\"; wait"],
			cwd: temporaryDirectory("pi-verification-tree-"),
		});
		try {
			const snapshot = await process.waitFor(/CHILD:(\d+)/, { deadlineMs: 5_000, source: "raw" });
			const match = /CHILD:(\d+)/.exec(snapshot.rawText);
			if (!match?.[1]) throw new Error("child pid was not reported");
			const childPid = Number(match[1]);
			await process.terminate();
		const deadline = Date.now() + 5_000;
		let alive = true;
		while (Date.now() < deadline) {
			try {
				globalThis.process.kill(childPid, 0);
			} catch {
				alive = false;
				break;
			}
			await Bun.sleep(25);
		}
		expect(alive).toBe(false);
		} finally {
			await process.terminate();
		}
	}, 15_000);

	test("writeKeys returns a receipt that excludes prior chunks, starts immediately before the write, and precedes the resulting output", async () => {
		const root = temporaryDirectory("pi pty ' $() ");
		const process = spawnPty({
			argv: [
				"/bin/sh",
				"-c",
				`stty echo; printf READY; IFS= read -r a; printf 'GOT:%s\\n' "$a"; IFS= read -r b; printf 'GOT2:%s\\n' "$b"`,
			],
			cwd: root,
		});
		try {
			await process.waitFor(/READY/, { deadlineMs: 5_000, source: "raw" });
			const first = process.writeKeys("first", PTY_KEYS.enter);
			await process.waitFor(/GOT:first/, { deadlineMs: 5_000 });
			const afterFirst = process.snapshot();
			expect(afterFirst.rawText.slice(0, first.outputOffset)).not.toContain("GOT:first");
			expect(afterFirst.rawText.slice(first.outputOffset)).toContain("GOT:first");

			const hostile = "sec\"'\\$(`)ond";
			const beforeSecond = process.snapshot();
			const second = process.writeKeys(hostile, PTY_KEYS.enter);
			expect(second.outputOffset).toBeGreaterThan(first.outputOffset);
			expect(second.outputOffset).toBe(beforeSecond.rawText.length);
			expect(second.startedElapsedMs).toBeGreaterThanOrEqual(beforeSecond.elapsedMs);
			await process.waitFor(/GOT2:/, { deadlineMs: 5_000 });
			const afterSecond = process.snapshot();
			expect(afterSecond.rawText.slice(second.outputOffset)).toContain(`GOT2:${hostile}`);
			const postWriteChunkElapsed: number[] = [];
			let consumed = 0;
			for (const chunk of afterSecond.chunks) {
				if (chunk.stream !== "pty") continue;
				consumed += chunk.text.length;
				if (consumed > second.outputOffset) postWriteChunkElapsed.push(chunk.elapsedMs);
			}
			expect(postWriteChunkElapsed.length).toBeGreaterThan(0);
			expect(postWriteChunkElapsed.every((elapsedMs) => elapsedMs >= second.startedElapsedMs)).toBe(true);
			expect(await process.waitForExit(5_000)).toBe(0);
		} finally {
			await process.terminate();
		}
	}, 15_000);
});
