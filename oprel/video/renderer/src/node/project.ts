import fs from "node:fs";
import path from "node:path";
import { compileStoryboard, type CompileResult, type CompiledTimeline, type Issue } from "../compiler";
import { checkSceneSource } from "./sandbox";

export const STORYBOARD_FILE = "storyboard.json";

/** Resolve a project-relative asset to an absolute path, refusing anything that escapes the project (incl. via symlinks). */
export function resolveAsset(projectDir: string, rel: string): string | null {
  const root = fs.realpathSync(projectDir);
  const abs = path.resolve(root, rel);
  if (!abs.startsWith(root + path.sep)) return null;
  if (!fs.existsSync(abs)) return abs; // caller decides what a missing file means
  const real = fs.realpathSync(abs);
  return real.startsWith(root + path.sep) ? real : null;
}

export function loadAndCompile(projectDir: string, file: string = STORYBOARD_FILE): CompileResult & { storyboardPath: string } {
  const storyboardPath = path.resolve(projectDir, file);
  if (!storyboardPath.startsWith(path.resolve(projectDir) + path.sep))
    return { ok: false, issues: [{ level: "error", path: file, message: "storyboard file must be inside the project" }], storyboardPath };
  let raw: unknown;
  try {
    raw = JSON.parse(fs.readFileSync(storyboardPath, "utf8"));
  } catch (e) {
    const issue: Issue = { level: "error", path: STORYBOARD_FILE, message: `cannot read storyboard: ${(e as Error).message}` };
    return { ok: false, issues: [issue], storyboardPath };
  }
  const res = compileStoryboard(raw, {
    assetExists: (rel) => {
      const abs = resolveAsset(projectDir, rel);
      return abs !== null && fs.existsSync(abs) && fs.statSync(abs).isFile();
    },
  });
  return { ...res, storyboardPath };
}

/** Static sandbox policy for every AI/user-written scene file the timeline references. */
export function policyIssues(projectDir: string, tl: CompiledTimeline): Issue[] {
  const out: Issue[] = [];
  for (const rel of tl.customFiles) {
    const abs = resolveAsset(projectDir, rel);
    if (!abs || !fs.existsSync(abs)) continue; // missing files are already reported by the compiler
    out.push(...checkSceneSource(fs.readFileSync(abs, "utf8"), rel));
  }
  return out;
}
