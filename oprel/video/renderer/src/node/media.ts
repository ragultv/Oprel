import { spawnSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";
import type { CompiledTimeline } from "../compiler";

const require = createRequire(import.meta.url);

/** ffmpeg/ffprobe shipped with Remotion's compositor (so no system install is required). OPREL_FFMPEG_DIR overrides. */
export function mediaBinary(name: "ffmpeg" | "ffprobe"): string {
  const exe = process.platform === "win32" ? `${name}.exe` : name;
  const dirs: string[] = [];
  if (process.env.OPREL_FFMPEG_DIR) dirs.push(process.env.OPREL_FFMPEG_DIR);
  const pkg = `@remotion/compositor-${process.platform}-${process.arch}${process.platform === "win32" ? "-msvc" : process.platform === "linux" ? "-gnu" : ""}`;
  try {
    dirs.push(path.dirname(require.resolve(`${pkg}/package.json`)));
  } catch {
    /* fall through */
  }
  for (const d of dirs) {
    const p = path.join(d, exe);
    if (fs.existsSync(p)) return p;
  }
  throw new Error(`${name} not found (looked in ${dirs.join(", ") || "no candidate dirs"})`);
}

function run(bin: string, args: string[]) {
  // No shell: args are passed as an array, never concatenated into a command line.
  return spawnSync(bin, args, { encoding: "utf8", maxBuffer: 64 * 1024 * 1024, windowsHide: true });
}

export type ProbeResult = {
  video?: { codec: string; width: number; height: number; fps: number; frames?: number; pixFmt?: string };
  audio?: { codec: string; sampleRate: number; channels: number };
  durationSec: number;
};

export function probe(file: string, countFrames = true): ProbeResult {
  const args = ["-v", "error", "-print_format", "json", "-show_streams", "-show_format"];
  if (countFrames) args.push("-count_frames");
  args.push(file);
  const r = run(mediaBinary("ffprobe"), args);
  if (r.error || r.status !== 0) throw new Error(`ffprobe failed: ${r.error?.message ?? r.stderr}`);
  const j = JSON.parse(r.stdout);
  const v = j.streams?.find((s: any) => s.codec_type === "video");
  const a = j.streams?.find((s: any) => s.codec_type === "audio");
  const [n, d] = String(v?.r_frame_rate ?? "0/1").split("/").map(Number);
  return {
    durationSec: Number(j.format?.duration ?? v?.duration ?? 0),
    video: v && {
      codec: v.codec_name,
      width: v.width,
      height: v.height,
      fps: d ? n / d : 0,
      frames: v.nb_read_frames ? Number(v.nb_read_frames) : v.nb_frames ? Number(v.nb_frames) : undefined,
      pixFmt: v.pix_fmt,
    },
    audio: a && { codec: a.codec_name, sampleRate: Number(a.sample_rate), channels: a.channels },
  };
}

export type Check = { name: string; ok: boolean; expected?: unknown; actual?: unknown; detail?: string };
export type ValidationReport = { file: string; ok: boolean; checks: Check[]; probe?: ProbeResult; createdAt: string };

export type ExpectOptions = { scale?: number; expectFrames?: number };

/** Validate a rendered MP4 against its compiled timeline: stream layout, codec, size, fps, frame count, decodability. */
export function validateOutput(file: string, tl: CompiledTimeline, opts: ExpectOptions = {}): ValidationReport {
  const checks: Check[] = [];
  const add = (c: Check) => checks.push(c);
  const report = (probeRes?: ProbeResult): ValidationReport => ({
    file,
    ok: checks.every((c) => c.ok),
    checks,
    probe: probeRes,
    createdAt: new Date().toISOString(),
  });

  const exists = fs.existsSync(file) && fs.statSync(file).size > 0;
  add({ name: "output exists and is non-empty", ok: exists });
  if (!exists) return report();

  let p: ProbeResult;
  try {
    p = probe(file);
  } catch (e) {
    add({ name: "ffprobe can read file", ok: false, detail: (e as Error).message });
    return report();
  }
  add({ name: "ffprobe can read file", ok: true });

  const scale = opts.scale ?? 1;
  const ew = Math.round(tl.width * scale);
  const eh = Math.round(tl.height * scale);
  const frames = opts.expectFrames ?? tl.durationInFrames;
  add({ name: "has video stream", ok: !!p.video });
  if (p.video) {
    add({ name: "video codec is h264", ok: p.video.codec === "h264", expected: "h264", actual: p.video.codec });
    add({ name: "resolution", ok: p.video.width === ew && p.video.height === eh, expected: `${ew}x${eh}`, actual: `${p.video.width}x${p.video.height}` });
    add({ name: "frame rate", ok: Math.abs(p.video.fps - tl.fps) < 0.01, expected: tl.fps, actual: p.video.fps });
    if (p.video.frames !== undefined)
      add({ name: "frame count", ok: p.video.frames === frames, expected: frames, actual: p.video.frames });
    const expDur = frames / tl.fps;
    add({ name: "duration within 1 frame", ok: Math.abs(p.durationSec - expDur) <= 1 / tl.fps + 0.05, expected: expDur, actual: p.durationSec });
  }
  if (tl.narration) {
    add({ name: "audio stream present (narration expected)", ok: !!p.audio });
    if (p.audio) add({ name: "audio codec is aac", ok: p.audio.codec === "aac", expected: "aac", actual: p.audio.codec });
  }

  // Remotion's bundled ffmpeg has no "null" encoder, so decode to tiny raw frames through the null muxer.
  const decodeErrors: string[] = [];
  const v = run(mediaBinary("ffmpeg"), ["-v", "error", "-i", file, "-an", "-vf", "scale=32:32", "-c:v", "rawvideo", "-f", "null", "-"]);
  if (v.status !== 0 || v.stderr.trim()) decodeErrors.push(`video: ${v.stderr.trim()}`);
  if (p.audio) {
    const a = run(mediaBinary("ffmpeg"), ["-v", "error", "-i", file, "-vn", "-c:a", "pcm_s16le", "-f", "null", "-"]);
    if (a.status !== 0 || a.stderr.trim()) decodeErrors.push(`audio: ${a.stderr.trim()}`);
  }
  add({ name: "full decode without errors", ok: decodeErrors.length === 0, detail: decodeErrors.join(" | ").slice(0, 500) || undefined });
  return report(p);
}
