import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import ts from "typescript";
import type { Issue } from "../compiler";
import { checkSceneSource } from "./sandbox";
import { resolveAsset } from "./project";

const HERE = path.dirname(fileURLToPath(import.meta.url));
export const RENDERER_DIR = path.resolve(HERE, "../..");
export const SRC_DIR = path.join(RENDERER_DIR, "src");
/** Staging area. Must live inside the renderer dir so bundled scene files resolve `remotion`, `three`, ... from its node_modules. */
export const CACHE_ROOT = path.join(RENDERER_DIR, ".render-cache");

export type CustomFile = { key: string; abs: string };

export type Stage = { dir: string; cleanup: () => void };

export function newStage(prefix: string): Stage {
  fs.mkdirSync(CACHE_ROOT, { recursive: true });
  const dir = fs.mkdtempSync(path.join(CACHE_ROOT, `${prefix}-`));
  return { dir, cleanup: () => fs.rmSync(dir, { recursive: true, force: true }) };
}

/** Remove stale stages left behind by a crashed worker (older than an hour). */
export function sweepStaleStages(maxAgeMs = 3600_000) {
  if (!fs.existsSync(CACHE_ROOT)) return;
  for (const name of fs.readdirSync(CACHE_ROOT)) {
    const p = path.join(CACHE_ROOT, name);
    try {
      if (Date.now() - fs.statSync(p).mtimeMs > maxAgeMs) fs.rmSync(p, { recursive: true, force: true });
    } catch {
      /* best effort */
    }
  }
}

/**
 * Resolve the project's custom scene files, re-run the static policy on each (so hand-edited files are covered too),
 * and copy them into the stage. Returns issues instead of throwing so callers can report them.
 */
export function stageCustomScenes(projectDir: string, stageDir: string, relFiles: string[]): { files: CustomFile[]; issues: Issue[] } {
  const files: CustomFile[] = [];
  const issues: Issue[] = [];
  for (const rel of relFiles) {
    const abs = resolveAsset(projectDir, rel);
    if (!abs || !fs.existsSync(abs)) {
      issues.push({ level: "error", path: rel, message: "scene file is missing or outside the project" });
      continue;
    }
    const source = fs.readFileSync(abs, "utf8");
    issues.push(...checkSceneSource(source, rel));
    const dst = path.join(stageDir, rel);
    fs.mkdirSync(path.dirname(dst), { recursive: true });
    fs.writeFileSync(dst, source);
    files.push({ key: rel, abs: dst });
  }
  return { files, issues };
}

/** Generated bundle entry: registers custom scenes, then the normal Root. */
export function writeEntry(stageDir: string, files: CustomFile[]): string {
  const rootImport = path.relative(stageDir, path.join(SRC_DIR, "Root")).replace(/\\/g, "/");
  const customImport = path.relative(stageDir, path.join(SRC_DIR, "scenes", "custom")).replace(/\\/g, "/");
  const lines = [
    `import { registerRoot } from "remotion";`,
    `import { Root } from "${rootImport}";`,
    `import { registerCustomScenes } from "${customImport}";`,
    ...files.map((f, i) => `import S${i} from "./${f.key.replace(/\.tsx$/, "")}";`),
    `registerCustomScenes({ ${files.map((f, i) => `${JSON.stringify(f.key)}: S${i}`).join(", ")} });`,
    `registerRoot(Root);`,
  ];
  const entry = path.join(stageDir, "entry.ts");
  fs.writeFileSync(entry, lines.join("\n") + "\n");
  return entry;
}

/** Names @oprel/kit exports (guarded by test/kit.test.ts). */
export const KIT_EXPORTS = [
  "tokens", "Backdrop", "Header", "TitleBlock", "FadeIn", "ease", "reveal", "clamp01", "seededPoints", "SceneProps",
  "AbsoluteFill", "Easing", "Sequence", "interpolate", "interpolateColors", "random", "spring", "useCurrentFrame", "useVideoConfig",
];
const SIGNATURES =
  "interpolate(value, inputRange, outputRange, { extrapolateLeft?, extrapolateRight?, easing? }) -> number; " +
  "spring({ frame, fps, config?: { damping } }) -> number 0..1; ease(frame, start, end) -> number 0..1; random(\"seed\") -> number 0..1; " +
  "useVideoConfig() -> { fps, width, height, durationInFrames }.";

/** Short, concrete advice for the TypeScript mistakes small models make most. Appended to the compiler message. */
export function hintFor(code: number, text: string): string {
  const name = /'(\w+)'/.exec(text)?.[1] ?? "";
  if (code === 2305 || code === 2724) {
    const mod = /Module '"([^"]+)"'/.exec(text)?.[1] ?? "";
    if (mod === "@oprel/kit") return `@oprel/kit exports only: ${KIT_EXPORTS.join(", ")}.`;
    if (mod.startsWith("@remotion/three")) return 'import { ThreeCanvas } from "@remotion/three" only; THREE classes (Vector3, Mesh...) come from "three" - or use plain arrays like [x, y, z].';
    if (mod === "remotion") return "remotion exports AbsoluteFill, Easing, Sequence, interpolate, interpolateColors, random, spring, useCurrentFrame, useVideoConfig (and more); check the spelling.";
    return `that name does not exist in "${mod}". Allowed modules: react, remotion, @oprel/kit, three, @remotion/three, @remotion/paths.`;
  }
  if (code === 2304 || code === 2552 || code === 2551 && /Cannot find name/.test(text))
    return `"${name}" is not defined: add it to an import. Header, Backdrop, tokens, ease from "@oprel/kit"; spring, interpolate, useCurrentFrame, useVideoConfig, AbsoluteFill from "remotion" (or "@oprel/kit").`;
  if (code === 2554 || code === 2555) return `wrong number of arguments. Signatures: ${SIGNATURES}`;
  if ((code === 2551 || code === 2339) && /'value'/.test(text)) return "spring(), interpolate() and ease() return plain numbers: there is no .value; use the number directly.";
  if (code === 2322 && /textAlign/.test(text)) return 'SVG <text> has no textAlign; use textAnchor="middle" (or "start"/"end").';
  if (code === 2322 && /fontSize/.test(text) && /string/.test(text)) return "in SVG/JSX props use numbers for sizes (fontSize={40}), not strings.";
  if (code === 2769 || code === 2353) return `an option or property used here does not exist. ${SIGNATURES}`;
  return "";
}

export function typecheckDiagnostics(stagedFile: string): { sf: ts.SourceFile | undefined; diags: ts.Diagnostic[] } {
  const options: ts.CompilerOptions = {
    target: ts.ScriptTarget.ES2022,
    module: ts.ModuleKind.ESNext,
    moduleResolution: ts.ModuleResolutionKind.Bundler,
    jsx: ts.JsxEmit.ReactJSX,
    strict: true,
    noImplicitAny: false, // missing type annotations never change what is rendered; real API errors are still caught
    noEmit: true,
    skipLibCheck: true,
    esModuleInterop: true,
    resolveJsonModule: true,
    lib: ["lib.es2022.d.ts", "lib.dom.d.ts", "lib.dom.iterable.d.ts"],
    types: [],
    baseUrl: RENDERER_DIR,
    paths: { "@oprel/kit": ["src/kit.ts"] },
  };
  const roots = [stagedFile, path.join(SRC_DIR, "assets.d.ts"), path.join(SRC_DIR, "r3f-augment.ts")];
  const program = ts.createProgram(roots, options);
  const sf = program.getSourceFile(stagedFile);
  if (!sf) return { sf, diags: [] };
  return { sf, diags: [...program.getSyntacticDiagnostics(sf), ...program.getSemanticDiagnostics(sf)] };
}

export function diagnosticsToIssues(sf: ts.SourceFile, diags: ts.Diagnostic[], rel: string): Issue[] {
  return diags.slice(0, 20).map((d) => {
    const { line } = sf.getLineAndCharacterOfPosition(d.start ?? 0);
    const text = ts.flattenDiagnosticMessageText(d.messageText, " ");
    const hint = hintFor(d.code, text);
    return { level: "error" as const, path: `${rel}:${line + 1}`, message: `TS${d.code}: ${text}${hint ? ` (Hint: ${hint})` : ""}` };
  });
}

/** Type-check one staged scene file with the same compiler options the renderer uses. */
export function typecheckScene(stagedFile: string, rel: string): Issue[] {
  const { sf, diags } = typecheckDiagnostics(stagedFile);
  if (!sf) return [{ level: "error", path: rel, message: "could not load the scene file for type checking" }];
  return diagnosticsToIssues(sf, diags, rel);
}
