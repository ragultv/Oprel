import fs from "node:fs";
import path from "node:path";
import ts from "typescript";
import { PNG } from "pngjs";
import { openBrowser, renderStill, selectComposition, type BrowserLog } from "@remotion/renderer";
import type { CompiledTimeline, Issue } from "../compiler";
import { COMPOSITION_ID } from "../constants";
import { checkSceneSource } from "./sandbox";
import { chromium, prepare, type Emit } from "./render";
import { autofixScene, type Fix } from "./autofix";
import { diagnosticsToIssues, newStage, typecheckDiagnostics } from "./stage";

export type FrameStat = { frame: number; file: string; edgeFraction: number; borderFraction: number; bottomFraction: number; boxFraction: number; meanLuma: number; ms: number };

export type SceneCheckResult = {
  ok: boolean;
  /** First stage that failed (or "render" when everything passed). */
  stage: "static" | "types" | "render";
  issues: Issue[];
  warnings: Issue[];
  frames: FrameStat[];
  consoleErrors: string[];
  /** Mechanical API repairs applied before/while checking. When set, `fixedFile` holds the repaired source. */
  autofixes?: Fix[];
  fixedFile?: string;
};

export type SceneCheckArgs = {
  projectDir: string;
  /** Project-relative path of the candidate scene source. */
  file: string;
  durationInFrames: number;
  fps?: number;
  narration?: string;
  title?: string;
  /** Project-relative directory the stills are written to. */
  outDir: string;
  emit: Emit;
};

const WIDTH = 1080;
const HEIGHT = 1920;
const SCALE = 0.5;

/** Write the repaired source next to the candidate (`x.tsx` -> `x.fixed.tsx`); the original is never overwritten. Returns its project-relative path. */
function writeFixed(absOriginal: string, relOriginal: string, source: string): string {
  const abs = absOriginal.replace(/\.tsx$/, ".fixed.tsx");
  fs.writeFileSync(abs, source);
  return relOriginal.replace(/\.tsx$/, ".fixed.tsx");
}

export function sampleFrames(duration: number): number[] {
  const last = Math.max(0, duration - 1);
  const picks = [Math.min(2, last), Math.round(last * 0.25), Math.round(last * 0.5), Math.round(last * 0.75), Math.max(0, last - 2)];
  return [...new Set(picks)].sort((a, b) => a - b);
}

/** Pixel statistics that distinguish "something is drawn" from "only the smooth background gradient". */
/** Bottom 140px of the 1920px canvas is reserved for captions and phone edge margins. */
export const BOTTOM_ZONE = 140 / HEIGHT;

export function analysePng(buf: Buffer): { edgeFraction: number; borderFraction: number; bottomFraction: number; boxFraction: number; meanLuma: number; luma: Uint8Array; w: number; h: number } {
  const png = PNG.sync.read(buf);
  const { width: w, height: h, data } = png;
  const luma = new Uint8Array(w * h);
  let sum = 0;
  for (let i = 0; i < w * h; i++) {
    const l = Math.round(0.2126 * data[i * 4] + 0.7152 * data[i * 4 + 1] + 0.0722 * data[i * 4 + 2]);
    luma[i] = l;
    sum += l;
  }
  // "edge" = a pixel that differs strongly from the pixel 6px to its right or below. Smooth gradients have none.
  const D = 6, T = 24, ring = Math.round(20 * SCALE);
  let edges = 0, total = 0, borderEdges = 0, borderTotal = 0, bottomEdges = 0, bottomTotal = 0;
  let minX = w, maxX = -1, minY = h, maxY = -1;
  const bottomStart = Math.round(h * (1 - BOTTOM_ZONE));
  for (let y = 0; y < h - D; y += 2) {
    for (let x = 0; x < w - D; x += 2) {
      const i = y * w + x;
      const e = Math.abs(luma[i] - luma[i + D]) > T || Math.abs(luma[i] - luma[i + D * w]) > T;
      total++;
      if (e) {
        edges++;
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
      if (y >= bottomStart) {
        bottomTotal++;
        if (e) bottomEdges++;
      }
      if (x < ring || y < ring || x > w - ring - D || y > h - ring - D) {
        borderTotal++;
        if (e) borderEdges++;
      }
    }
  }
  return { edgeFraction: edges / total, borderFraction: borderTotal ? borderEdges / borderTotal : 0, bottomFraction: bottomTotal ? bottomEdges / bottomTotal : 0,
    boxFraction: maxX < 0 ? 0 : ((maxX - minX + D) * (maxY - minY + D)) / (w * h), meanLuma: sum / (w * h), luma, w, h };
}

/** Share of pixels whose brightness changed visibly between two frames. A single small moving element is ~0.05-0.1%; a dead scene is exactly 0. */
export function changedFraction(a: Uint8Array, b: Uint8Array): number {
  let changed = 0;
  for (let i = 0; i < a.length; i++) if (Math.abs(a[i] - b[i]) > 12) changed++;
  return changed / a.length;
}

/** A scene counts as animated when at least this share of pixels changes somewhere between sampled frames (about 100px at 540x960). */
export const MIN_CHANGED_FRACTION = 0.0002;
/** The visible content's bounding box should use at least this share of the canvas in the second half of the scene. */
export const MIN_CONTENT_BOX = 0.15;

/**
 * Check one candidate scene end to end: sandbox policy -> TypeScript -> real render of sample frames -> pixel sanity.
 * Returns structured, model-readable issues. Never throws for problems in the scene itself.
 */
export async function checkScene(a: SceneCheckArgs): Promise<SceneCheckResult> {
  const fps = a.fps ?? 30;
  const abs = path.resolve(a.projectDir, a.file);
  const empty = { frames: [], consoleErrors: [], warnings: [] as Issue[] };
  if (!abs.startsWith(path.resolve(a.projectDir) + path.sep) || !fs.existsSync(abs))
    return { ok: false, stage: "static", issues: [{ level: "error", path: a.file, message: "candidate file not found inside the project" }], ...empty };

  const source = fs.readFileSync(abs, "utf8");
  const staticIssues = checkSceneSource(source, a.file);
  if (staticIssues.length) return { ok: false, stage: "static", issues: staticIssues, ...empty };

  a.emit({ event: "progress", stage: "types", progress: 0.1, message: "Type-checking scene" });
  // Type-check; on failure repair the mechanical API slips the compiler pinpoints (up to 3 rounds) and check again.
  let renderFile = a.file;
  const applied: Fix[] = [];
  const stage = newStage("typecheck");
  try {
    const staged = path.join(stage.dir, a.file);
    fs.mkdirSync(path.dirname(staged), { recursive: true });
    let current = source;
    for (let round = 0; ; round++) {
      fs.writeFileSync(staged, current);
      const { sf, diags } = typecheckDiagnostics(staged);
      if (!sf) return { ok: false, stage: "types", issues: [{ level: "error", path: a.file, message: "could not load the scene file for type checking" }], ...empty };
      if (!diags.length) break;
      const repaired = round < 3 ? autofixScene(current, diags.map((d) => ({ code: d.code, start: d.start ?? 0, length: d.length ?? 0, message: ts.flattenDiagnosticMessageText(d.messageText, " ") }))) : null;
      if (repaired && repaired.fixes.length && repaired.source !== current) {
        applied.push(...repaired.fixes);
        current = repaired.source;
        continue;
      }
      // out of automatic repairs: report what is left, against the repaired source so line numbers match the file we hand back
      let fixedFile: string | undefined;
      if (applied.length) fixedFile = writeFixed(abs, a.file, current);
      return { ok: false, stage: "types", issues: diagnosticsToIssues(sf, diags, fixedFile ?? a.file), ...empty, autofixes: applied, fixedFile };
    }
    if (applied.length) {
      renderFile = writeFixed(abs, a.file, current);
      const policy = checkSceneSource(current, renderFile);
      if (policy.length) return { ok: false, stage: "static", issues: policy, ...empty, autofixes: applied, fixedFile: renderFile };
    }
  } finally {
    stage.cleanup();
  }

  const consoleErrors: string[] = [];
  const onLog = (log: BrowserLog) => {
    if (log.type === "error") consoleErrors.push(String(log.text).slice(0, 400));
  };
  const duration = Math.max(1, a.durationInFrames);
  const scene = {
    id: "check",
    type: "custom" as const,
    narration: a.narration ?? "",
    transition: { type: "none" as const, durationFrames: 0 },
    assets: [],
    params: { file: renderFile, title: a.title ?? "", props: {} },
  };
  const timeline: CompiledTimeline = {
    schemaVersion: 1, title: "scene check", fps, width: WIDTH, height: HEIGHT, durationInFrames: duration,
    scenes: [{ id: "check", type: "custom", from: 0, durationInFrames: duration, transition: { type: "none", durationFrames: 0 }, scene }],
    captions: [], narration: null, assets: [], customFiles: [renderFile],
  };

  const issues: Issue[] = [];
  const warnings: Issue[] = [];
  const frames: FrameStat[] = [];
  let prep: Awaited<ReturnType<typeof prepare>> | null = null;
  let browser: Awaited<ReturnType<typeof openBrowser>> | null = null;
  try {
    prep = await prepare(a.projectDir, timeline, a.emit, () => false, { onLog });
    browser = await openBrowser("chrome", { chromiumOptions: chromium() });
    const comp = await selectComposition({ serveUrl: prep.serveUrl, id: COMPOSITION_ID, inputProps: { timeline }, puppeteerInstance: browser, chromiumOptions: chromium(), onBrowserLog: onLog });
    const outAbs = path.resolve(a.projectDir, a.outDir);
    fs.mkdirSync(outAbs, { recursive: true });
    const picks = sampleFrames(duration);
    const lumas: Uint8Array[] = [];
    for (const [i, frame] of picks.entries()) {
      a.emit({ event: "progress", stage: "render", progress: 0.3 + (0.6 * i) / picks.length, message: `Rendering frame ${frame}` });
      const out = path.join(outAbs, `f${frame}.png`);
      const t0 = Date.now();
      try {
        await renderStill({
          composition: comp, serveUrl: prep.serveUrl, frame, output: out, inputProps: { timeline }, scale: SCALE,
          puppeteerInstance: browser, chromiumOptions: chromium(), onBrowserLog: onLog, timeoutInMilliseconds: 60_000,
        });
      } catch (e) {
        issues.push({ level: "error", path: `frame ${frame}`, message: `rendering threw: ${String((e as Error).message).split("\n")[0].slice(0, 400)}` });
        break;
      }
      const stat = analysePng(fs.readFileSync(out));
      lumas.push(stat.luma);
      frames.push({ frame, file: path.relative(path.resolve(a.projectDir), out).replace(/\\/g, "/"), edgeFraction: stat.edgeFraction, borderFraction: stat.borderFraction, bottomFraction: stat.bottomFraction, boxFraction: stat.boxFraction, meanLuma: stat.meanLuma, ms: Date.now() - t0 });
    }

    if (!issues.length) {
      const late = frames.filter((f) => f.frame >= duration * 0.4);
      if (late.length && late.every((f) => f.edgeFraction < 0.0015))
        issues.push({ level: "error", path: "render", message: "after the first half the frames show only the plain background: nothing is drawn. Make sure the scene renders visible shapes and text." });
      const diffs = lumas.slice(1).map((l, i) => changedFraction(lumas[i], l));
      if (diffs.length && Math.max(...diffs) < MIN_CHANGED_FRACTION)
        issues.push({ level: "error", path: "render", message: "the scene does not change over time (frames look identical). Animate elements with interpolate()/spring() from useCurrentFrame()." });
      if (late.length && late.every((f) => f.boxFraction < MIN_CONTENT_BOX))
        issues.push({ level: "error", path: "render", message: `the visible content fills only ${Math.round(Math.max(...late.map((f) => f.boxFraction)) * 100)}% of the frame (it must use at least ${Math.round(MIN_CONTENT_BOX * 100)}%). Make the text and shapes bigger and spread them over the frame, centred inside the safe area (64px side margins, nothing above y=120 or below y=1780).` });
      if (frames.some((f) => f.bottomFraction > 0.05))
        warnings.push({ level: "warning", path: "render", message: "content is drawn near the very bottom of the canvas (y > 1780), which may overlap bottom phone controls. Keep important text above y = 1780." });
      if (frames.some((f) => f.borderFraction > 0.06))
        warnings.push({ level: "warning", path: "render", message: "content touches the very edge of the frame; keep it inside the safe margins" });
      if (frames.some((f) => f.ms > 20_000)) warnings.push({ level: "warning", path: "render", message: "a frame took over 20s to render; reduce heavy 3D or large blur/filter use" });
    }
    if (consoleErrors.length) issues.push({ level: "error", path: "browser", message: `console errors while rendering: ${[...new Set(consoleErrors)].slice(0, 3).join(" | ")}` });
  } catch (e) {
    issues.push({ level: "error", path: "render", message: `could not render the scene: ${String((e as Error).message).split("\n")[0].slice(0, 400)}` });
  } finally {
    await browser?.close({ silent: true }).catch(() => {});
    prep?.cleanup();
  }
  return { ok: issues.length === 0, stage: "render", issues, warnings, frames, consoleErrors, autofixes: applied.length ? applied : undefined, fixedFile: applied.length ? renderFile : undefined };
}
