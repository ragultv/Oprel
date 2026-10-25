import { z } from "zod";
import { FORMAT_DIMENSIONS, StoryboardSchema, type Caption, type Scene, type Storyboard } from "./schema";
import { isSceneImplemented } from "./sceneTypes";

export type Issue = { level: "error" | "warning"; path: string; message: string };

export type CompiledScene = {
  id: string;
  type: Scene["type"];
  from: number;
  durationInFrames: number;
  transition: { type: "none" | "fade"; durationFrames: number };
  scene: Scene;
};

export type CompiledCaption = { from: number; to: number; text: string; granularity: Caption["granularity"] };

export type CompiledTimeline = {
  schemaVersion: 1;
  title: string;
  fps: number;
  width: number;
  height: number;
  durationInFrames: number;
  scenes: CompiledScene[];
  captions: CompiledCaption[];
  narration: { file: string; durationMs?: number; volume: number } | null;
  assets: string[];
  /** Project-relative paths of AI/user-written scene components (never copied to the public dir). */
  customFiles: string[];
};

export type CompileResult =
  | { ok: true; timeline: CompiledTimeline; issues: Issue[] }
  | { ok: false; issues: Issue[] };

export type CompileOptions = {
  /** Return true if the project-relative asset exists. If omitted, asset existence is not checked. */
  assetExists?: (path: string) => boolean;
};

const msToFrame = (ms: number, fps: number) => Math.round((ms / 1000) * fps);

/** Pure + deterministic: same storyboard in, same timeline out. Frames are the canonical unit. */
export function compileStoryboard(raw: unknown, opts: CompileOptions = {}): CompileResult {
  const parsed = StoryboardSchema.safeParse(raw);
  if (!parsed.success) {
    return { ok: false, issues: parsed.error.issues.map(zodIssue) };
  }
  const sb: Storyboard = parsed.data;
  const issues: Issue[] = [];
  const fps = sb.format.fps;
  const { width, height } = FORMAT_DIMENSIONS[sb.format.preset];

  let cursor = 0;
  const scenes: CompiledScene[] = sb.scenes.map((s, i) => {
    const durationInFrames = s.durationFrames ?? Math.max(1, Math.round((s.durationSeconds as number) * fps));
    const compiled: CompiledScene = {
      id: s.id,
      type: s.type,
      from: cursor,
      durationInFrames,
      transition: { type: s.transition.type, durationFrames: s.transition.durationFrames },
      scene: s,
    };
    cursor += durationInFrames;

    if (s.transition.type === "none" && s.transition.durationFrames > 0)
      issues.push({ level: "warning", path: `scenes.${i}.transition`, message: "durationFrames ignored for type 'none'" });
    if (s.transition.type !== "none" && s.transition.durationFrames * 2 > durationInFrames)
      issues.push({
        level: "error",
        path: `scenes.${i}.transition`,
        message: `transition (${s.transition.durationFrames}f) exceeds half of scene (${durationInFrames}f)`,
      });
    if (!isSceneImplemented(s.type))
      issues.push({ level: "error", path: `scenes.${i}.type`, message: `scene type "${s.type}" has no registered component` });
    return compiled;
  });
  const durationInFrames = cursor;

  const captions: CompiledCaption[] = [];
  let prevEnd = -1;
  sb.captions.forEach((c, i) => {
    if (c.endMs <= c.startMs) issues.push({ level: "error", path: `captions.${i}`, message: "endMs must be > startMs" });
    if (c.startMs < prevEnd) issues.push({ level: "error", path: `captions.${i}`, message: "captions overlap or are out of order" });
    prevEnd = c.endMs;
    const from = msToFrame(c.startMs, fps);
    const to = Math.min(msToFrame(c.endMs, fps), durationInFrames);
    if (from >= durationInFrames) issues.push({ level: "error", path: `captions.${i}`, message: "caption starts after the video ends" });
    else captions.push({ from, to: Math.max(to, from + 1), text: c.text, granularity: c.granularity });
  });

  if (sb.narration?.durationMs !== undefined) {
    const narrFrames = msToFrame(sb.narration.durationMs, fps);
    if (narrFrames > durationInFrames)
      issues.push({
        level: "error",
        path: "narration",
        message: `narration (${narrFrames}f) is longer than the timeline (${durationInFrames}f); audio would be cut`,
      });
    else if (durationInFrames - narrFrames > fps * 6)
      issues.push({ level: "warning", path: "narration", message: "timeline runs >6s past the end of narration" });
  }
  if (sb.captions.length > 0 && !sb.narration)
    issues.push({ level: "warning", path: "captions", message: "captions present but no narration audio" });

  const assets = new Set<string>();
  const customFiles = new Set<string>();
  if (sb.narration) assets.add(sb.narration.file);
  sb.scenes.forEach((s) => {
    s.assets.forEach((a) => assets.add(a));
    if (s.type === "custom") customFiles.add(s.params.file);
  });
  if (opts.assetExists)
    for (const a of [...assets, ...customFiles]) if (!opts.assetExists(a)) issues.push({ level: "error", path: "assets", message: `missing file: ${a}` });

  if (issues.some((i) => i.level === "error")) return { ok: false, issues };

  return {
    ok: true,
    issues,
    timeline: {
      schemaVersion: 1,
      title: sb.title,
      fps,
      width,
      height,
      durationInFrames,
      scenes,
      captions,
      narration: sb.narration
        ? { file: sb.narration.file, durationMs: sb.narration.durationMs, volume: sb.narration.volume }
        : null,
      assets: [...assets].sort(),
      customFiles: [...customFiles].sort(),
    },
  };
}

function zodIssue(i: z.ZodIssue): Issue {
  return { level: "error", path: i.path.join("."), message: i.message };
}
