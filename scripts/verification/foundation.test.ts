import { afterAll, describe, expect, test } from "bun:test";
import { existsSync, mkdtempSync, mkdirSync, readFileSync, readdirSync, rmSync, watch } from "node:fs";
import { tmpdir } from "node:os";
import { basename, dirname, join, resolve } from "node:path";
import type {
	Api,
	AssistantMessageEvent,
	Context,
	Model,
	SimpleStreamOptions,
} from "../../.references/pi-2.0/packages/ai/src/index.ts";
import type { ExtensionAPI } from "../../.references/pi-2.0/packages/coding-agent/src/core/extensions/types.ts";
import verificationExtension, {
	DEFAULT_FINAL_MARKER,
	VERIFICATION_MODEL,
	VERIFICATION_PROVIDER,
} from "./extension.ts";
import { PTY_KEYS, spawnPty } from "./pty.ts";
import { assertCanonicalReference, canonicalReferenceRoot } from "../reference-identity.ts";

// The PTY driver shells to util-linux `setsid`/`script`, absent on macOS
// (BSD userland) and Windows. Gate the suites that need it on Linux.
const lacksUtilLinuxPty = process.platform !== "linux";

interface RegisteredProvider {
	readonly models?: readonly { readonly id: string }[];
	readonly streamSimple?: (
		model: Model<Api>,
		context: Context,
		options?: SimpleStreamOptions,
	) => AsyncIterable<AssistantMessageEvent>;
}

const temporaryPaths: string[] = [];

afterAll(() => {
	for (const path of temporaryPaths) rmSync(path, { recursive: true, force: true });
});

function temporaryDirectory(prefix: string): string {
	const path = mkdtempSync(join(tmpdir(), prefix));
	temporaryPaths.push(path);
	return path;
}

function fixtureModel(): Model<Api> {
	return {
		id: VERIFICATION_MODEL,
		name: "Verification Model",
		api: "custom",
		provider: VERIFICATION_PROVIDER,
		baseUrl: "https://verification.invalid",
		reasoning: false,
		input: ["text"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 1_000_000,
		maxTokens: 100_000,
	};
}

async function collectEvents(provider: RegisteredProvider, context: Context): Promise<AssistantMessageEvent[]> {
	if (!provider.streamSimple) throw new Error("verification streamSimple was not registered");
	const events: AssistantMessageEvent[] = [];
	for await (const event of provider.streamSimple(fixtureModel(), context)) events.push(event);
	return events;
}

function registerFixture(): RegisteredProvider {
	let provider: RegisteredProvider | undefined;
	const api = {
		registerFlag() {},
		on() {},
		registerShortcut() {},
		registerCommand() {},
		registerProvider(name: string, config: RegisteredProvider) {
			expect(name).toBe(VERIFICATION_PROVIDER);
			provider = config;
		},
	} as ExtensionAPI;
	verificationExtension(api);
	if (!provider) throw new Error("verification provider was not registered");
	return provider;
}

function withEnvironment(values: Readonly<Record<string, string | undefined>>, run: () => Promise<void>): Promise<void> {
	const previous: Record<string, string | undefined> = {};
	for (const [name, value] of Object.entries(values)) {
		previous[name] = process.env[name];
		if (value === undefined) delete process.env[name];
		else process.env[name] = value;
	}
	return run().finally(() => {
		for (const [name, value] of Object.entries(previous)) {
			if (value === undefined) delete process.env[name];
			else process.env[name] = value;
		}
	});
}
async function waitForFileContent(path: string, expected: string, deadlineMs: number): Promise<void> {
	const matches = (): boolean => {
		try {
			return readFileSync(path, "utf8") === expected;
		} catch (error) {
			if (error instanceof Error && "code" in error && error.code === "ENOENT") return false;
			throw error;
		}
	};
	if (matches()) return;
	const { promise, resolve, reject } = Promise.withResolvers<void>();
	const watcher = watch(dirname(path), (_event, filename) => {
		if (filename !== null && filename.toString() !== basename(path)) return;
		try {
			if (matches()) resolve();
		} catch (error) {
			reject(error);
		}
	});
	// The deadline guards a real child-process event; fake time cannot drive it.
	const timeout = setTimeout(
		() => reject(new Error(`${path} did not reach the expected content within ${deadlineMs}ms`)),
		deadlineMs,
	);
	try {
		if (!matches()) await promise;
	} finally {
		clearTimeout(timeout);
		watcher.close();
	}
}

describe("verification extension", () => {
	test("registers a deterministic model, stream, tool stages, compaction, and load generation", async () => {
		const directory = temporaryDirectory("pi-verification-extension-");
		const loadCountPath = join(directory, "state", "loads.txt");
		await withEnvironment(
			{
				PI_VERIFICATION_LOAD_COUNT_PATH: loadCountPath,
				PI_VERIFICATION_CHUNK_COUNT: "2",
				PI_VERIFICATION_FINAL_MARKER: "UNIT_FINAL",
			},
			async () => {
				const provider = registerFixture();
				registerFixture();
				expect(readFileSync(loadCountPath, "utf8")).toBe("2\n");
				expect(provider.models?.map((model) => model.id)).toEqual([VERIFICATION_MODEL]);

				const textEvents = await collectEvents(provider, {
					messages: [{ role: "user", content: "ordinary prompt", timestamp: 0 }],
				});
				expect(textEvents.map((event) => event.type)).toEqual([
					"start",
					"text_start",
					"text_delta",
					"text_delta",
					"text_delta",
					"text_end",
					"done",
				]);
				const done = textEvents.at(-1);
				expect(done?.type).toBe("done");
				if (done?.type === "done") expect(done.message.content).toEqual([
					{ type: "text", text: "verification-chunk-0001\nverification-chunk-0002\nUNIT_FINAL" },
				]);

				const toolResults: Context["messages"] = [];
				for (const expectedTool of ["read", "edit", "bash"] as const) {
					const events = await collectEvents(provider, {
						messages: [{ role: "user", content: "verification:tools", timestamp: 0 }, ...toolResults],
					});
					const toolEnd = events.find((event) => event.type === "toolcall_end");
					expect(toolEnd?.type === "toolcall_end" ? toolEnd.toolCall.name : undefined).toBe(expectedTool);
					toolResults.push({
						role: "toolResult",
						toolCallId: `verification-${expectedTool}`,
						toolName: expectedTool,
						content: [{ type: "text", text: "ok" }],
						isError: false,
						timestamp: 0,
					});
				}
				const finalToolEvents = await collectEvents(provider, {
					messages: [{ role: "user", content: "verification:tools", timestamp: 0 }, ...toolResults],
				});
				expect(finalToolEvents.some((event) => event.type === "text_delta" && event.delta.includes("UNIT_FINAL"))).toBe(true);

				process.env.PI_VERIFICATION_MODE = "compaction";
				const compaction = await collectEvents(provider, { messages: [] });
				expect(compaction.some((event) => event.type === "text_delta" && event.delta.includes("## Goal"))).toBe(true);
				delete process.env.PI_VERIFICATION_MODE;
			},
		);
	}, 15_000);
});

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
		const snapshot = await process.waitFor(/CHILD:(\d+)/, { deadlineMs: 5_000, source: "raw" });
		const match = /CHILD:(\d+)/.exec(snapshot.rawText);
		if (!match?.[1]) throw new Error("child pid was not reported");
		const childPid = Number(match[1]);
		await process.terminate();
		expect(() => globalThis.process.kill(childPid, 0)).toThrow();
	}, 15_000);
});

interface CliFixture {
	readonly name: string;
	readonly argvPrefix: readonly [string, ...string[]];
}

async function smokeCli(fixture: CliFixture, sharedDirectory: string): Promise<void> {
	const extensionPath = resolve(import.meta.dirname, "extension.ts");
	const hostPath = resolve("packages/extension-host/dist/pi-extension-host");
	const agentDirectory = join(sharedDirectory, "agent");
	const readyPath = join(sharedDirectory, `${fixture.name}-ready`);
	const sessionDirectory = join(sharedDirectory, "sessions");
	mkdirSync(agentDirectory, { recursive: true });
	mkdirSync(sessionDirectory, { recursive: true });
	const cli = spawnPty({
		argv: [
			...fixture.argvPrefix,
			"--provider",
			VERIFICATION_PROVIDER,
			"--model",
			VERIFICATION_MODEL,
			"--api-key",
			"verification-key",
			"--extension",
			extensionPath,
			"--no-session",
			"--offline",
			"--no-context-files",
			"--no-skills",
			"--no-themes",
			"--approve",
		],
		cwd: sharedDirectory,
		env: {
			HOME: join(sharedDirectory, "home"),
			PI_CODING_AGENT_DIR: agentDirectory,
			PI_CODING_AGENT_SESSION_DIR: sessionDirectory,
			PI_EXTENSION_HOST: hostPath,
			PI_OFFLINE: "1",
			PI_VERIFICATION_MODE: "text",
			PI_VERIFICATION_CHUNK_COUNT: "1",
			PI_VERIFICATION_CHUNK_DELAY_MS: "0",
			PI_VERIFICATION_FINAL_MARKER: DEFAULT_FINAL_MARKER,
			PI_VERIFICATION_READY_PATH: readyPath,
		},
	});
	try {
		await waitForFileContent(readyPath, "ready\n", 20_000);
		cli.writeKeys(`foundation prompt for ${fixture.name}`, PTY_KEYS.enter);
		// Match the marker with terminal styling removed: the response
		// renderer can place a style boundary (e.g. streaming-cursor reset
		// `\x1b[;m`) inside the marker bytes, so a raw match hangs on text
		// that is visibly on screen (proven: split marker at 2.0s, 150s
		// deadline, PTY silent after). Echo stays raw: it carries no styling.
		const response = await cli.waitFor(
			(snapshot) =>
				snapshot.applicationText
					.replaceAll(/\x1b\[[0-9;?]*[a-zA-Z]/g, "")
					.replaceAll(/\x1b\]8;;.*?\x07/g, "")
					.includes(DEFAULT_FINAL_MARKER),
			{ deadlineMs: 120_000 },
		);
		expect(response.echoText).not.toContain(DEFAULT_FINAL_MARKER);
		cli.writeKeys("/quit", PTY_KEYS.enter);
		expect(await cli.waitForExit(10_000)).toBe(0);
	} catch (error) {
		const snapshot = cli.snapshot();
		const tail = snapshot.rawText.slice(-4_000);
		const lastChunk = snapshot.chunks.at(-1);
		const chunkState =
			`chunks=${snapshot.chunks.length} lastChunkElapsedMs=${lastChunk ? Math.round(lastChunk.elapsedMs) : -1} ` +
			`exited=${snapshot.exited} exitCode=${snapshot.exitCode}`;
		const markerState =
			`marker raw=${snapshot.rawText.includes(DEFAULT_FINAL_MARKER)} ` +
			`application=${snapshot.applicationText.includes(DEFAULT_FINAL_MARKER)} ` +
			`echo=${snapshot.echoText.includes(DEFAULT_FINAL_MARKER)}`;
		// Diagnostics for the next red run: the PTY tail shows the app is
		// alive, so name the hang from on-disk state (ready handshake,
		// agent/session residues). Never let diagnostics mask the cause.
		let diagnostics = "";
		try {
			const readyState = existsSync(readyPath)
				? `ready=${JSON.stringify(readFileSync(readyPath, "utf8").slice(0, 32))}`
				: "ready=MISSING";
			let agentList: string;
			try {
				agentList = readdirSync(agentDirectory).slice(0, 12).join(",");
			} catch {
				agentList = "UNREADABLE";
			}
			let sessionList: string;
			try {
				sessionList = readdirSync(sessionDirectory).slice(0, 12).join(",");
			} catch {
				sessionList = "UNREADABLE";
			}
			diagnostics = `\nready ${readyState}\nagent dir [${agentList}]\nsessions dir [${sessionList}]`;
		} catch {
			diagnostics = "\ndiagnostics unavailable";
		}
		throw new Error(
			`${fixture.name} smoke failed: ${error instanceof Error ? error.message : String(error)}\n${chunkState}\n${markerState}${diagnostics}\nPTY tail:\n${tail}`,
		);
	} finally {
		await cli.terminate();
	}
}

describe.skipIf(lacksUtilLinuxPty)("shared interactive provider smoke", () => {
	test("drives Rust and TypeScript CLIs with one extension and model", async () => {
		// Gate before the first reference spawn: the TypeScript fixture runs the
		// canonical checkout's CLI, so its HEAD must match the pinned SHA.
		assertCanonicalReference();
		const rustBinary = resolve("target/debug/pi");
		const hostBinary = resolve("packages/extension-host/dist/pi-extension-host");
		expect(existsSync(rustBinary), `missing ${rustBinary}; run cargo build -p pi`).toBe(true);
		expect(existsSync(hostBinary), `missing ${hostBinary}; run bun run build:extension-host`).toBe(true);
		const bun = Bun.which("bun");
		if (!bun) throw new Error("bun executable not found");
		const sharedDirectory = temporaryDirectory("pi-verification-shared-");
		mkdirSync(join(sharedDirectory, "home"), { recursive: true });
		const fixtures: readonly CliFixture[] = [
			{
				name: "typescript",
				argvPrefix: [bun, join(canonicalReferenceRoot(), "packages/coding-agent/src/cli.ts")],
			},
			{ name: "rust", argvPrefix: [rustBinary] },
		];
	for (const fixture of fixtures) await smokeCli(fixture, sharedDirectory);
	}, 300_000);
});
