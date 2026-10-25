import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { bundle } from "@remotion/bundler";
import {
  ensureBrowser,
  makeCancelSignal,
  renderMedia,
  renderStill,
  selectComposition,
  type BrowserLog,
  type ChromiumOptions,
} from "@remotion/renderer";
import type { CompiledTimeline, Issue } from "../compiler";
import { COMPOSITION_ID } from "../constants";
import { resolveAsset } from "./project";
import { SRC_DIR, newStage, stageCustomScenes, sweepStaleStages, writeEntry } from "./stage";

export type Quality = "preview" | "production";
export type Emit = (e: Record<string, unknown>) => void;

export class Cancelled extends Error {
  constructor() {
    super("render cancelled");
  }
}

/** Thrown when scene code violates the sandbox policy; carries the issues so callers can report them precisely. */
export class SceneRejected extends Error {
  constructor(public issues: Issue[]) {
    super(`scene code rejected by the sandbox policy (${issues.length} issue${issues.length === 1 ? "" : "s"})`);
  }
}

export const QUALITY = {
  preview: { scale: 0.5, crf: 28 },
  production: { scale: 1, crf: 18 },
} as const;

export function defaultConcurrency(): number {
  // Conservative: each tab holds a WebGL context. Leave cores for the model runtime / UI.
  const fromEnv = Number(process.env.OPREL_VIDEO_CONCURRENCY);
  if (Number.isInteger(fromEnv) && fromEnv > 0) return fromEnv;
  return Math.max(1, Math.min(4, Math.floor(os.cpus().length / 4)));
}

function chromium(): ChromiumOptions {
  const gl = (process.env.OPREL_VIDEO_GL as ChromiumOptions["gl"]) || "angle";
  return { gl, disableWebSecurity: false };
}

/**
 * The render browser may run AI-written code. This policy removes every way for it to talk to the network or to
 * local services (including the Oprel daemon on another port): only same-origin loads from the render server are allowed.
 */
export const CSP =
  "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval' blob:; style-src 'self' 'unsafe-inline'; " +
  "img-src 'self' data: blob:; media-src 'self' data: blob:; font-src 'self' data:; connect-src 'self' data: blob:; " +
  "frame-src 'none'; object-src 'none'; worker-src 'self' blob:; base-uri 'self'; form-action 'none'";

export function injectCsp(bundleDir: string) {
  const index = path.join(bundleDir, "index.html");
  const html = fs.readFileSync(index, "utf8");
  if (!/<head[^>]*>/i.test(html)) throw new Error("bundle index.html has no <head>; cannot apply the sandbox policy");
  fs.writeFileSync(index, html.replace(/<head[^>]*>/i, (m) => `${m}<meta http-equiv="Content-Security-Policy" content="${CSP}">`));
}

export type Prepared = {
  serveUrl: string;
  stageDir: string;
  composition: Awaited<ReturnType<typeof selectComposition>>;
  cleanup: () => void;
};

export type PrepareOptions = {
  onLog?: (log: BrowserLog) => void;
};

/**
 * Bundle the composition. Only assets declared by the timeline are copied into the bundle's public dir, so a storyboard can
 * never expose other project files to the browser. Custom scene files are policy-checked again here, then bundled.
 */
export async function prepare(
  projectDir: string,
  tl: CompiledTimeline,
  emit: Emit,
  cancelled: () => boolean,
  opts: PrepareOptions = {},
): Promise<Prepared> {
  sweepStaleStages();
  const stage = newStage("render");
  const publicDir = path.join(stage.dir, "public");
  fs.mkdirSync(publicDir, { recursive: true });
  try {
    for (const rel of tl.assets) {
      const src = resolveAsset(projectDir, rel);
      if (!src) throw new Error(`asset escapes project: ${rel}`);
      const dst = path.join(publicDir, rel);
      fs.mkdirSync(path.dirname(dst), { recursive: true });
      fs.copyFileSync(src, dst);
    }
    const { files, issues } = stageCustomScenes(projectDir, stage.dir, tl.customFiles ?? []);
    if (issues.some((i) => i.level === "error")) throw new SceneRejected(issues);
    const entry = writeEntry(stage.dir, files);

    emit({ event: "progress", stage: "browser", progress: 0, message: "Checking headless browser" });
    await ensureBrowser();
    if (cancelled()) throw new Cancelled();

    emit({ event: "progress", stage: "bundle", progress: 0, message: "Bundling composition" });
    const bundleDir = await bundle({
      entryPoint: entry,
      publicDir,
      outDir: path.join(stage.dir, "bundle"),
      webpackOverride: (config) => ({
        ...config,
        resolve: { ...config.resolve, alias: { ...(config.resolve?.alias as object), "@oprel/kit": path.join(SRC_DIR, "kit.ts") } },
      }),
      onProgress: (p) => emit({ event: "progress", stage: "bundle", progress: p / 100 }),
    });
    injectCsp(bundleDir);
    if (cancelled()) throw new Cancelled();

    const composition = await selectComposition({
      serveUrl: bundleDir,
      id: COMPOSITION_ID,
      inputProps: { timeline: tl },
      chromiumOptions: chromium(),
      onBrowserLog: opts.onLog,
    });
    return { serveUrl: bundleDir, stageDir: stage.dir, composition, cleanup: stage.cleanup };
  } catch (e) {
    stage.cleanup();
    throw e;
  }
}

export type RenderArgs = {
  projectDir: string;
  timeline: CompiledTimeline;
  outFile: string;
  quality: Quality;
  frameRange?: [number, number];
  concurrency?: number;
  emit: Emit;
  cancelSignal: ReturnType<typeof makeCancelSignal>;
  isCancelled: () => boolean;
};

export async function renderVideo(a: RenderArgs): Promise<{ frames: number }> {
  const onLog = (log: BrowserLog) => {
    if (log.type === "error" || log.type === "warning") a.emit({ event: "log", level: log.type, message: String(log.text).slice(0, 500) });
  };
  const prep = await prepare(a.projectDir, a.timeline, a.emit, a.isCancelled, { onLog });
  try {
    const q = QUALITY[a.quality];
    fs.mkdirSync(path.dirname(a.outFile), { recursive: true });
    a.emit({ event: "progress", stage: "render", progress: 0, message: `Rendering ${a.quality}` });
    await renderMedia({
      composition: prep.composition,
      serveUrl: prep.serveUrl,
      codec: "h264",
      pixelFormat: "yuv420p",
      audioCodec: "aac",
      crf: q.crf,
      scale: q.scale,
      outputLocation: a.outFile,
      inputProps: { timeline: a.timeline },
      frameRange: a.frameRange,
      concurrency: a.concurrency ?? defaultConcurrency(),
      chromiumOptions: chromium(),
      cancelSignal: a.cancelSignal.cancelSignal,
      onBrowserLog: onLog,
      onProgress: (p) =>
        a.emit({ event: "progress", stage: "render", progress: p.progress, renderedFrames: p.renderedFrames, encodedFrames: p.encodedFrames }),
    });
    const frames = a.frameRange ? a.frameRange[1] - a.frameRange[0] + 1 : a.timeline.durationInFrames;
    return { frames };
  } catch (e) {
    if (a.isCancelled()) throw new Cancelled();
    throw e;
  } finally {
    prep.cleanup();
  }
}

export async function renderFrame(
  projectDir: string,
  timeline: CompiledTimeline,
  frame: number,
  outFile: string,
  emit: Emit,
): Promise<void> {
  const prep = await prepare(projectDir, timeline, emit, () => false);
  try {
    fs.mkdirSync(path.dirname(outFile), { recursive: true });
    await renderStill({
      composition: prep.composition,
      serveUrl: prep.serveUrl,
      frame,
      output: outFile,
      inputProps: { timeline },
      chromiumOptions: chromium(),
    });
  } finally {
    prep.cleanup();
  }
}

export { renderStill, chromium };
