import { afterAll, describe, expect, test } from "bun:test";
import { cpSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
	assertFenceImportsResolve,
	enumerateFenceImports,
	EXCLUDED_EXAMPLE_PRODUCTS,
	PUBLIC_SHELL_SNIPPETS,
	REQUIRED_SNIPPET_FIXTURES,
	REPO_ROOT,
	classifyFence,
	collectDocFences,
	extractFences,
	inferRustDeps,
	mapCargoDiagnostic,
	mapTscDiagnostic,
	runRustLane,
	runSnippetHarness,
	runTypeScriptLane,
	validateSourcePath,
	verifyNoExamplesDirectory,
	verifyNoExcludedExampleProducts,
	verifyPublicShellSnippets,
	verifyRequiredSnippetFixtures,
	wrapRustSnippet,
} from "./snippet-harness.ts";

const temporaryPaths: string[] = [];

afterAll(() => {
	for (const path of temporaryPaths) rmSync(path, { recursive: true, force: true });
});

function temporaryDirectory(prefix: string): string {
	const path = mkdtempSync(join(tmpdir(), prefix));
	temporaryPaths.push(path);
	return path;
}

const NEGATIVE_PATH = "scripts/verification/fixtures/docs-snippets/negative-stale-import.md";
const NEGATIVE_SOURCE = readFileSync(join(REPO_ROOT, NEGATIVE_PATH), "utf8");

describe("validateSourcePath", () => {
	test("accepts repo-relative paths and rejects traversal or absolute paths", () => {
		expect(validateSourcePath(REPO_ROOT, "docs/README.md")).toBe("docs/README.md");
		expect(validateSourcePath(REPO_ROOT, "scripts/verification/fixtures/docs-snippets/rust/pi.md")).toBe(
			"scripts/verification/fixtures/docs-snippets/rust/pi.md",
		);
		expect(validateSourcePath(REPO_ROOT, "../outside.md")).toBeUndefined();
		expect(validateSourcePath(REPO_ROOT, "/tmp/evil.md")).toBeUndefined();
		expect(validateSourcePath(REPO_ROOT, "docs/../../etc/passwd")).toBeUndefined();
	});
});

describe("extractFences", () => {
	test("captures info strings and exact open/body line numbers", () => {
		const source = ["# title", "", "```rust", "fn main() {}", "```", "", "```ts", "const x = 1;", "```"].join("\n");
		const { fences, failures } = extractFences(source, "docs/example.md");
		expect(failures).toEqual([]);
		expect(fences).toEqual([
			{
				docPath: "docs/example.md",
				openLine: 3,
				bodyStartLine: 4,
				infoString: "rust",
				body: "fn main() {}",
			},
			{
				docPath: "docs/example.md",
				openLine: 7,
				bodyStartLine: 8,
				infoString: "ts",
				body: "const x = 1;",
			},
		]);
	});

	test("rejects unclosed fences and backtick-bearing info strings", () => {
		const unclosed = extractFences("```rust\nfn main() {}\n", "docs/open.md");
		expect(unclosed.fences).toEqual([]);
		expect(unclosed.failures.some((item) => item.line === 1 && item.message.includes("unclosed"))).toBe(true);

		const badInfo = extractFences("```rust`bad\nfn main() {}\n```\n", "docs/bad.md");
		expect(badInfo.failures.some((item) => item.message.includes("backticks"))).toBe(true);
	});

	test("preserves empty bodies and indented opening fences", () => {
		const source = ["   ```rust", "```", "", "```ts", "", "```"].join("\n");
		const { fences, failures } = extractFences(source, "docs/empty.md");
		expect(failures).toEqual([]);
		expect(fences.map((fence) => ({ info: fence.infoString, body: fence.body, open: fence.openLine }))).toEqual([
			{ info: "rust", body: "", open: 1 },
			{ info: "ts", body: "", open: 4 },
		]);
	});
});

describe("classifyFence", () => {
	test("routes the supported corpus info strings", () => {
		expect(classifyFence("rust")).toBe("rust");
		expect(classifyFence("rust,no_run")).toBe("rust");
		expect(classifyFence("rust,ignore")).toBe("rust-skip");
		expect(classifyFence("rust,text")).toBe("rust-skip");
		expect(classifyFence("rust,compile_fail")).toBe("unsupported");
		expect(classifyFence("rust,should_panic")).toBe("unsupported");
		expect(classifyFence("ts")).toBe("ts");
		expect(classifyFence("typescript")).toBe("ts");
		expect(classifyFence("bash")).toBe("ignore");
		expect(classifyFence("json")).toBe("ignore");
		expect(classifyFence("text")).toBe("ignore");
	});
});

describe("wrapRustSnippet and inferRustDeps", () => {
	test("wraps fragments, keeps fn main verbatim, and uncomments hidden lines in place", () => {
		expect(wrapRustSnippet("let x = 1;")).toEqual({
			code: "fn main() {\n    let x = 1;\n}\n",
			headerLines: 1,
		});
		expect(wrapRustSnippet("fn main() {\n    let x = 1;\n}")).toEqual({
			code: "fn main() {\n    let x = 1;\n}\n",
			headerLines: 0,
		});
		expect(wrapRustSnippet("# use pi_ai::estimate_text_tokens;\nlet _ = estimate_text_tokens(\"x\");")).toEqual({
			code: "fn main() {\n    use pi_ai::estimate_text_tokens;\n    let _ = estimate_text_tokens(\"x\");\n}\n",
			headerLines: 1,
		});
	});

	test("maps crate tokens and leaves zero-dep snippets empty", () => {
		expect(inferRustDeps("use pi_ai::estimate_text_tokens;")).toEqual(["pi-ai"]);
		expect(inferRustDeps("let _ = pi::VERSION;\nlet _ = pi_ext::protocol::FLAGS_SET_METHOD;").sort()).toEqual([
			"pi",
			"pi-ext",
		]);
		expect(inferRustDeps("let x = 1;")).toEqual([]);
	});
});

describe("diagnostic mapping", () => {
	test("maps cargo and tsc diagnostics to original document lines", () => {
		const fence = {
			docPath: "scripts/verification/fixtures/docs-snippets/rust/pi-ai.md",
			openLine: 3,
			bodyStartLine: 4,
			infoString: "rust",
			body: "use pi_ai::missing;",
		};
		const cargoLine = JSON.stringify({
			reason: "compiler-message",
			message: {
				code: { code: "E0432" },
				message: "unresolved import",
				spans: [{ file_name: "src/bin/snippet_000.rs", line_start: 2, column_start: 99, is_primary: true }],
			},
		});
		const rustMapped = mapCargoDiagnostic(cargoLine, new Map([["snippet_000.rs", { fence, headerLines: 1 }]]));
		expect(rustMapped).toMatchObject({
			docPath: fence.docPath,
			line: 4,
			tool: "rustc",
			code: "E0432",
		});
		expect(rustMapped?.column).toBeUndefined();

		const tsFence = {
			docPath: "scripts/verification/fixtures/docs-snippets/ts/protocol.md",
			openLine: 7,
			bodyStartLine: 8,
			infoString: "ts",
			body: 'import { missing } from "@earendil-works/pi-tui-protocol";',
		};
		expect(
			mapTscDiagnostic(
				"snippet_000.ts(1,10): error TS2305: Module has no exported member 'missing'.",
				new Map([["snippet_000.ts", tsFence]]),
			),
		).toMatchObject({
			docPath: tsFence.docPath,
			line: 8,
			tool: "tsc",
			code: "TS2305",
		});
	});

	test(
		"attributes multiline TypeScript syntax errors to the later body line",
		async () => {
			const body = ["const ok = 1;", "function broken( {", "  return ok;", "}"].join("\n");
			const fence = {
				docPath: "scripts/verification/fixtures/docs-snippets/negative-stale-import.md",
				openLine: 10,
				bodyStartLine: 11,
				infoString: "ts",
				body,
				kind: "ts" as const,
				snippetId: "scripts/verification/fixtures/docs-snippets/negative-stale-import.md:10",
			};
			const result = await runTypeScriptLane(REPO_ROOT, [fence]);
			expect(result.failures.some((item) => item.docPath === fence.docPath && item.line === fence.bodyStartLine)).toBe(false);
			expect(result.failures.some((item) => item.docPath === fence.docPath && item.line > fence.bodyStartLine && item.tool === "tsc")).toBe(true);
		},
		{ timeout: 120_000 },
	);
});

describe("required fixture registry", () => {
	test("live fixture corpus satisfies every required entrypoint contract", () => {
		expect(verifyRequiredSnippetFixtures(REPO_ROOT)).toEqual([]);
		expect(REQUIRED_SNIPPET_FIXTURES.filter((entry) => entry.lane === "rust")).toHaveLength(5);
		expect(REQUIRED_SNIPPET_FIXTURES.filter((entry) => entry.lane === "ts")).toHaveLength(3);
		const readme = readFileSync(join(REPO_ROOT, "scripts/verification/fixtures/docs-snippets/README.md"), "utf8");
		for (const entry of REQUIRED_SNIPPET_FIXTURES) {
			expect(readme.includes(entry.path.replace("scripts/verification/fixtures/docs-snippets/", ""))).toBe(true);
		}
	});

	test("deleting one registered fixture fails the registry witness", () => {
		const root = temporaryDirectory("snippet-registry-");
		const relativeFixtureRoot = "scripts/verification/fixtures/docs-snippets";
		cpSync(join(REPO_ROOT, relativeFixtureRoot), join(root, relativeFixtureRoot), { recursive: true });
		const target = REQUIRED_SNIPPET_FIXTURES[0];
		if (target === undefined) throw new Error("required fixture registry is empty");
		rmSync(join(root, target.path), { force: true });
		const problems = verifyRequiredSnippetFixtures(root);
		expect(problems.some((problem) => problem.includes(target.path) && problem.includes("missing"))).toBe(true);
	});
});

describe("public shell snippet registry", () => {
	test("live README and getting-started docs satisfy every public bash contract", () => {
		expect(verifyPublicShellSnippets(REPO_ROOT)).toEqual([]);
		expect(PUBLIC_SHELL_SNIPPETS.filter((entry) => entry.path === "README.md")).toHaveLength(3);
		expect(PUBLIC_SHELL_SNIPPETS.filter((entry) => entry.path === "docs/getting-started.md")).toHaveLength(5);
	});

	test("missing public shell document fails the registry", () => {
		const root = temporaryDirectory("shell-registry-missing-");
		mkdirSync(join(root, "docs"), { recursive: true });
		writeFileSync(join(root, "docs/getting-started.md"), readFileSync(join(REPO_ROOT, "docs/getting-started.md"), "utf8"));
		const problems = verifyPublicShellSnippets(root);
		expect(problems.some((problem) => problem.includes("README.md") && problem.includes("missing"))).toBe(true);
		expect(problems.some((problem) => problem.includes("docs/getting-started.md"))).toBe(false);
	});

	test("deleting --model gemini-flash-latest from a temporary copy fails every launch contract", () => {
		const root = temporaryDirectory("shell-registry-drift-");
		mkdirSync(join(root, "docs"), { recursive: true });
		for (const path of ["README.md", "docs/getting-started.md"]) {
			writeFileSync(join(root, path), readFileSync(join(REPO_ROOT, path), "utf8").replaceAll("--model gemini-flash-latest", ""));
		}
		const problems = verifyPublicShellSnippets(root);
		const launchViolations = problems.filter((problem) => problem.includes("target/release/pi --provider google --model gemini-flash-latest"));
		expect(launchViolations).toHaveLength(4);
		for (const problem of launchViolations) expect(problem).toContain("missing probe");
	});

	test("a probe present only in a different bash fence is reported as split", () => {
		const root = temporaryDirectory("shell-registry-split-");
		writeFileSync(
			join(root, "README.md"),
			[
				"# pi",
				"",
				"```bash",
				"cargo build -p pi --release --locked",
				"```",
				"",
				"```bash",
				'read -rsp "Enter Gemini API key: " GEMINI_API_KEY && export GEMINI_API_KEY',
				"```",
				"",
				"```bash",
				"printf '\\n'",
				"target/release/pi --provider google --model gemini-flash-latest",
				"```",
				"",
			].join("\n"),
		);
		mkdirSync(join(root, "docs"), { recursive: true });
		writeFileSync(join(root, "docs/getting-started.md"), readFileSync(join(REPO_ROOT, "docs/getting-started.md"), "utf8"));
		const problems = verifyPublicShellSnippets(root);
		expect(problems.some((problem) => problem.includes("README.md") && problem.includes("split out of bash fence #1 into bash fence #2"))).toBe(true);
	});
});

describe("exclusion guard", () => {
	test("rejects excluded product names in fixture trees", () => {
		const root = temporaryDirectory("snippet-exclusion-");
		const fixtureDir = join(root, "scripts/verification/fixtures/docs-snippets");
		mkdirSync(fixtureDir, { recursive: true });
		writeFileSync(join(fixtureDir, "bad.md"), "mentions with-deps product\n");
		expect(verifyNoExcludedExampleProducts(root).some((problem) => problem.includes("with-deps"))).toBe(true);
	});

	test("real fixture corpus stays free of excluded products", () => {
		expect(verifyNoExcludedExampleProducts(REPO_ROOT)).toEqual([]);
		for (const name of EXCLUDED_EXAMPLE_PRODUCTS) {
			expect(NEGATIVE_SOURCE.includes(name)).toBe(false);
		}
	});
});

describe("corpus accounting and determinism", () => {
	test("current docs contribute zero rust/ts fences and fixture collection is sorted", () => {
		const { fences, failures } = collectDocFences(REPO_ROOT);
		expect(failures).toEqual([]);
		expect(fences.filter((fence) => fence.docPath.startsWith("docs/") && (fence.kind === "rust" || fence.kind === "ts"))).toEqual([]);
		const ordered = fences.map((fence) => `${fence.docPath}:${fence.openLine}`);
		expect(ordered).toEqual(
			[...fences]
				.sort((a, b) => a.docPath.localeCompare(b.docPath) || a.openLine - b.openLine)
				.map((fence) => `${fence.docPath}:${fence.openLine}`),
		);
		expect(fences.some((fence) => fence.docPath.includes("negative-"))).toBe(false);
	});

	test("pure collection composition is deterministic", () => {
		const first = collectDocFences(REPO_ROOT);
		const second = collectDocFences(REPO_ROOT);
		expect(second).toEqual(first);
	});
});

describe("snippet harness e2e", () => {
	test(
		"compiles both fixture lanes against live sources",
		async () => {
			const report = await runSnippetHarness(REPO_ROOT);
			if (!report.ok) {
				throw new Error(
					`snippet harness failed: violations=${JSON.stringify(report.violations.slice(0, 8))} ` +
						`rust=${JSON.stringify(report.lanes.rust)} ts=${JSON.stringify(report.lanes.ts)}`,
				);
			}
			expect(report.lanes.rust.documents).toBe(0);
			expect(report.lanes.ts.documents).toBe(0);
			expect(report.lanes.rust.fixtures).toBeGreaterThan(0);
			expect(report.lanes.ts.fixtures).toBeGreaterThan(0);
			expect(report.lanes.rust.compiled).toBe(report.lanes.rust.extracted);
			expect(report.lanes.ts.compiled).toBe(report.lanes.ts.extracted);
			const again = await runSnippetHarness(REPO_ROOT);
			expect(again).toEqual(report);
		},
		{ timeout: 600_000 },
	);

	test(
		"stale imports fail with exact document line attribution",
		async () => {
			const extracted = extractFences(NEGATIVE_SOURCE, NEGATIVE_PATH);
			expect(extracted.failures).toEqual([]);
			const rustFence = extracted.fences.find((fence) => classifyFence(fence.infoString) === "rust");
			const tsFence = extracted.fences.find((fence) => classifyFence(fence.infoString) === "ts");
			expect(rustFence).toBeDefined();
			expect(tsFence).toBeDefined();
			if (rustFence === undefined || tsFence === undefined) throw new Error("negative fixtures missing");

			const rustRegistered = [
				{
					...rustFence,
					kind: "rust" as const,
					snippetId: `${rustFence.docPath}:${rustFence.openLine}`,
				},
			];
			const tsRegistered = [
				{
					...tsFence,
					kind: "ts" as const,
					snippetId: `${tsFence.docPath}:${tsFence.openLine}`,
				},
			];
			const [rust, ts] = await Promise.all([
				runRustLane(REPO_ROOT, rustRegistered),
				runTypeScriptLane(REPO_ROOT, tsRegistered),
			]);
			expect(rust.failures.some((item) => item.docPath === NEGATIVE_PATH && item.line === rustFence.bodyStartLine)).toBe(true);
			expect(ts.failures.some((item) => item.docPath === NEGATIVE_PATH && item.line === tsFence.bodyStartLine)).toBe(true);
		},
		{ timeout: 600_000 },
	);
});

describe("DOC-D: import resolution across registered fences", () => {
	test("every fence import resolves against live workspace exports in both lanes", () => {
		const violations = assertFenceImportsResolve(REPO_ROOT);
		expect(violations).toEqual([]);
	});

	test("enumerateFenceImports covers both rust and ts lanes", () => {
		const { imports } = enumerateFenceImports(REPO_ROOT);
		const rustImports = imports.filter((imp) => imp.lane === "rust");
		const tsImports = imports.filter((imp) => imp.lane === "ts");
		expect(rustImports.length).toBeGreaterThan(0);
		expect(tsImports.length).toBeGreaterThan(0);
		for (const imp of imports) {
			expect(imp.fenceId).toMatch(/^scripts\/verification\/fixtures\/docs-snippets\//);
			expect(imp.specifier.length).toBeGreaterThan(0);
		}
	});

	test("stale import in a fence fails resolution", () => {
		const root = temporaryDirectory("snippet-import-res-");
		const fixtureDir = "scripts/verification/fixtures/docs-snippets/rust";
		mkdirSync(join(root, fixtureDir), { recursive: true });
		writeFileSync(join(root, fixtureDir, "stale.md"), "```rust\nuse pi_ai::nonexistent_symbol;\n\nfn main() { let _ = nonexistent_symbol(); }\n```\n");
		const violations = assertFenceImportsResolve(root);
		expect(violations.some((v) => v.includes("nonexistent_symbol"))).toBe(true);
	});
});

describe("DOC-D: ledger assertion for excluded products and examples/ directory", () => {
	test("zero fixtures reference any excluded example-product name", () => {
		expect(verifyNoExcludedExampleProducts(REPO_ROOT)).toEqual([]);
		for (const name of EXCLUDED_EXAMPLE_PRODUCTS) {
			expect(NEGATIVE_SOURCE.includes(name)).toBe(false);
		}
	});

	test("zero ledger rows reference any excluded example-product name", () => {
		const ledger = JSON.parse(readFileSync(join(REPO_ROOT, "scripts/verification/docs-evidence.json"), "utf8"));
		for (const row of ledger.rows) {
			const rowStr = JSON.stringify(row);
			for (const name of EXCLUDED_EXAMPLE_PRODUCTS) {
				expect(rowStr.includes(name)).toBe(false);
			}
		}
	});

	test("no examples/ directory exists in the tree", () => {
		const violations = verifyNoExamplesDirectory(REPO_ROOT);
		expect(violations).toEqual([]);
	});

	test("examples/ directory detection fires on a planted directory", () => {
		const root = temporaryDirectory("snippet-examples-");
		mkdirSync(join(root, "examples"), { recursive: true });
		const violations = verifyNoExamplesDirectory(root);
		expect(violations.some((v) => v.includes("examples/"))).toBe(true);
	});

	test("examples/ under excluded directories are not flagged", () => {
		const root = temporaryDirectory("snippet-examples-excluded-");
		mkdirSync(join(root, "target/debug/examples"), { recursive: true });
		mkdirSync(join(root, "node_modules/some-pkg/examples"), { recursive: true });
		const violations = verifyNoExamplesDirectory(root);
		expect(violations).toEqual([]);
	});
});
