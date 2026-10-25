/**
 * Oprel Video Studio render worker.
 *
 *   tsx bin/worker.ts <op> --project <dir> [options]
 *
 * ops: validate | compile | render | preview | still | inspect | check
 *
 * Protocol: one JSON object per stdout line ({"event":"progress"|"result"|"error", ...}).
 * Cancellation: write "cancel\n" to stdin. Exit code is nonzero on any failure.
 * The worker never loads models and never executes storyboard content; it only reads validated JSON.
 */
import fs from "node:fs";
import path from "node:path";
import readline from "node:readline";
import { parseArgs } from "node:util";
import { makeCancelSignal } from "@remotion/renderer";
import { loadAndCompile, policyIssues } from "../src/node/project";
import { mediaBinary, probe, validateOutput } from "../src/node/media";
import { Cancelled, QUALITY, renderFrame, renderVideo, type Quality } from "../src/node/render";
import type { CompiledTimeline, Issue } from "../src/compiler";
import { checkScene } from "../src/node/scenecheck";

const emit = (e: Record<string, unknown>) => process.stdout.write(JSON.stringify(e) + "\n");
const fail = (code: string, message: string, extra: Record<string, unknown> = {}): never => {
  emit({ event: "error", code, message, ...extra });
  process.exit(code === "cancelled" ? 130 : 1);
};

const { positionals, values } = parseArgs({
  allowPositionals: true,
  options: {
    project: { type: "string" },
    quality: { type: "string", default: "production" },
    out: { type: "string" },
    frames: { type: "string" },
    frame: { type: "string" },
    concurrency: { type: "string" },
    overwrite: { type: "boolean", default: false },
    file: { type: "string" },
    duration: { type: "string" },
    narration: { type: "string" },
    title: { type: "string" },
  },
});

const op = positionals[0];
if (!op) fail("usage", "missing operation");
const projectDir = values.project ? path.resolve(values.project) : "";
if (!projectDir || !fs.existsSync(projectDir) || !fs.statSync(projectDir).isDirectory())
  fail("project_not_found", `project directory not found: ${values.project}`);

const cancel = makeCancelSignal();
let cancelled = false;
readline.createInterface({ input: process.stdin }).on("line", (l) => {
  if (l.trim() === "cancel") {
    cancelled = true;
    cancel.cancel();
  }
});

function compileOrFail(): CompiledTimeline {
  const res = loadAndCompile(projectDir);
  const issues: Issue[] = res.issues;
  if (!res.ok) fail("invalid_project", "storyboard failed validation", { issues });
  else if (issues.length) emit({ event: "log", level: "warning", issues });
  return (res as Extract<typeof res, { ok: true }>).timeline;
}

function writeJson(rel: string, data: unknown) {
  const p = path.join(projectDir, rel);
  fs.mkdirSync(path.dirname(p), { recursive: true });
  fs.writeFileSync(p, JSON.stringify(data, null, 2));
  return p;
}

async function main() {
  switch (op) {
    case "validate": {
      const res = loadAndCompile(projectDir, values.file);
      if (!values.file) writeJson("output/storyboard-validation.json", { ok: res.ok, issues: res.issues, createdAt: new Date().toISOString() });
      if (!res.ok) fail("invalid_project", "storyboard failed validation", { issues: res.issues });
      const policy = res.ok ? policyIssues(projectDir, res.timeline) : [];
      if (policy.length) fail("invalid_project", "scene code failed the sandbox policy", { issues: policy });
      emit({ event: "result", op, ok: true, issues: res.issues });
      return;
    }
    case "compile": {
      const tl = compileOrFail();
      const p = writeJson("timeline.json", tl);
      emit({ event: "result", op, ok: true, timelinePath: p, durationInFrames: tl.durationInFrames, fps: tl.fps });
      return;
    }
    case "inspect": {
      const file = path.resolve(projectDir, values.file ?? "output/final.mp4");
      emit({ event: "result", op, ok: true, probe: probe(file) });
      return;
    }
    case "check-scene": {
      if (!values.file) return fail("bad_args", "--file is required");
      const duration = Number(values.duration ?? 150);
      if (!Number.isInteger(duration) || duration < 1 || duration > 7200) return fail("bad_args", "--duration must be 1..7200 frames");
      const res = await checkScene({
        projectDir, file: values.file, durationInFrames: duration, narration: values.narration, title: values.title,
        outDir: values.out ?? "agent/stills/check", emit,
      });
      emit({ event: "result", op, ...res });
      return;
    }
    case "still": {
      const tl = compileOrFail();
      const frame = Number(values.frame ?? 0);
      if (!Number.isInteger(frame) || frame < 0 || frame >= tl.durationInFrames) fail("bad_args", `frame out of range 0..${tl.durationInFrames - 1}`);
      const out = path.resolve(projectDir, values.out ?? `tmp/frame-${frame}.png`);
      await renderFrame(projectDir, tl, frame, out, emit);
      emit({ event: "result", op, ok: true, file: out, frame });
      return;
    }
    case "render":
    case "preview": {
      const quality = (op === "preview" ? "preview" : values.quality) as Quality;
      if (!(quality in QUALITY)) fail("bad_args", `quality must be preview|production`);
      const tl = compileOrFail();

      let frameRange: [number, number] | undefined;
      if (values.frames) {
        const m = /^(\d+)-(\d+)$/.exec(values.frames);
        if (!m) return fail("bad_args", "--frames must look like 0-89");
        frameRange = [Number(m[1]), Number(m[2])];
        if (frameRange[0] > frameRange[1] || frameRange[1] >= tl.durationInFrames) fail("bad_args", `frame range outside 0..${tl.durationInFrames - 1}`);
      }
      const isFull = !frameRange && quality === "production";
      const defaultOut = isFull ? "output/final.mp4" : `output/${quality}.mp4`;
      const finalOut = path.resolve(projectDir, values.out ?? defaultOut);
      if (!finalOut.startsWith(projectDir + path.sep)) fail("bad_args", "output must be inside the project directory");
      if (fs.existsSync(finalOut) && !values.overwrite) fail("output_exists", `${finalOut} exists; pass --overwrite to replace it`);

      const tmpOut = path.join(projectDir, "tmp", `render-${Date.now()}.mp4`);
      fs.mkdirSync(path.dirname(tmpOut), { recursive: true });
      const concurrency = values.concurrency ? Number(values.concurrency) : undefined;
      try {
        await renderVideo({
          projectDir, timeline: tl, outFile: tmpOut, quality, frameRange, concurrency,
          emit, cancelSignal: cancel, isCancelled: () => cancelled,
        });
      } catch (e) {
        fs.rmSync(tmpOut, { force: true });
        if (e instanceof Cancelled) fail("cancelled", "render cancelled");
        throw e;
      }

      emit({ event: "progress", stage: "validate", progress: 0, message: "Validating output" });
      const expectFrames = frameRange ? frameRange[1] - frameRange[0] + 1 : undefined;
      const report = validateOutput(tmpOut, tl, { scale: QUALITY[quality].scale, expectFrames });
      report.file = finalOut;
      const reportPath = writeJson(`output/${path.basename(finalOut, ".mp4")}.validation.json`, report);
      if (!report.ok) {
        fs.rmSync(tmpOut, { force: true });
        fail("validation_failed", "rendered output failed validation", { reportPath, checks: report.checks.filter((c) => !c.ok) });
      }
      fs.mkdirSync(path.dirname(finalOut), { recursive: true });
      fs.renameSync(tmpOut, finalOut);
      emit({ event: "result", op, ok: true, output: finalOut, reportPath, probe: report.probe, quality });
      return;
    }
    case "check": {
      // Environment self-check: binaries present, no render.
      const out: Record<string, unknown> = { node: process.version, ffmpeg: mediaBinary("ffmpeg"), ffprobe: mediaBinary("ffprobe") };
      emit({ event: "result", op, ok: true, ...out });
      return;
    }
    default:
      fail("usage", `unknown operation: ${op}`);
  }
}

main()
  .then(() => process.exit(0))
  .catch((e: Error) => fail("render_failed", e.message, { stack: String(e.stack).split("\n").slice(0, 8) }));
