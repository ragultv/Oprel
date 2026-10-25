import fs from "node:fs";
import path from "node:path";
import ts from "typescript";
import { afterAll, describe, expect, it } from "vitest";
import { autofixScene } from "../src/node/autofix";
import { newStage, typecheckDiagnostics } from "../src/node/stage";

const stages: ReturnType<typeof newStage>[] = [];
afterAll(() => stages.forEach((s) => s.cleanup()));

function diagnose(source: string) {
  const stage = newStage("af-test");
  stages.push(stage);
  const file = path.join(stage.dir, "scenes", "probe.tsx");
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, source);
  const { diags } = typecheckDiagnostics(file);
  return diags.map((d) => ({ code: d.code, start: d.start ?? 0, length: d.length ?? 0, message: ts.flattenDiagnosticMessageText(d.messageText, " ") }));
}

/** Repair until the compiler is happy (what scenecheck does, up to 3 rounds). */
function repair(source: string) {
  let current = source;
  const rules: string[] = [];
  for (let i = 0; i < 3; i++) {
    const d = diagnose(current);
    if (!d.length) break;
    const r = autofixScene(current, d);
    if (!r.fixes.length) break;
    rules.push(...r.fixes.map((f) => f.rule));
    current = r.source;
  }
  return { source: current, rules, remaining: diagnose(current).map((d) => d.message) };
}

const scene = (body: string, jsx = "<div />") =>
  `import React from "react";\nimport { AbsoluteFill, Backdrop, interpolate, spring, random, ease, useCurrentFrame, useVideoConfig, tokens } from "@oprel/kit";\n` +
  `export default function Scene() {\n  const frame = useCurrentFrame();\n  const { fps } = useVideoConfig();\n  ${body}\n  return ${jsx};\n}\n`;

describe("automatic repair of API slips seen in real model output", () => {
  it("interpolate(frame, [0, d]) with no output range", () => {
    const r = repair(scene("const p = interpolate(frame, [0, 240]); void p;"));
    expect(r.rules).toEqual(["interpolate-arity"]);
    expect(r.remaining).toEqual([]);
    expect(r.source).toContain('interpolate(frame, [0, 240], [0, 1], { extrapolateRight: "clamp" })');
  });

  it("spring({...})(100): a number called like a function", () => {
    const r = repair(scene("const s = spring({ frame: frame, fps, config: { damping: 200 } })(100); void s;"));
    expect(r.rules).toEqual(["number-called-as-function"]);
    expect(r.remaining).toEqual([]);
    expect(r.source).toContain("(spring({ frame: frame, fps, config: { damping: 200 } }) * (100))");
  });

  it("unknown prop on a kit component (<Backdrop bg=...>)", () => {
    const r = repair(scene("", '<Backdrop bg={tokens.bg.panel}><AbsoluteFill /></Backdrop>'));
    expect(r.rules).toEqual(["unknown-prop"]);
    expect(r.remaining).toEqual([]);
    expect(r.source).not.toContain("bg=");
  });

  it("textAlign on SVG text becomes textAnchor", () => {
    const r = repair(scene("", '<svg><text textAlign="center" x={1} y={2}>a</text></svg>'));
    expect(r.rules).toEqual(["svg-text-anchor"]);
    expect(r.remaining).toEqual([]);
    expect(r.source).toContain('textAnchor="middle"');
  });

  it("spring(frame, fps) positional call", () => {
    const r = repair(scene("const s = spring(frame, fps); void s;"));
    expect(r.rules).toEqual(["spring-signature"]);
    expect(r.remaining).toEqual([]);
    expect(r.source).toContain("spring({ frame: frame, fps: fps })");
  });

  it('random("seed", 0, 1) with a range', () => {
    const r = repair(scene('const q = random("seed", 0, 1); void q;'));
    expect(r.rules).toEqual(["random-range"]);
    expect(r.remaining).toEqual([]);
  });

  it(".value on a plain number", () => {
    const r = repair(scene("const v = spring({ frame, fps }).value; void v;"));
    expect(r.rules).toEqual(["no-dot-value"]);
    expect(r.remaining).toEqual([]);
  });

  it("several slips in one file are repaired together", () => {
    const r = repair(scene("const a = interpolate(frame, [0, 9]); const b = spring({ frame, fps })(5); void [a, b];", "<Backdrop bg=\"x\"><AbsoluteFill /></Backdrop>"));
    expect(r.rules.sort()).toEqual(["interpolate-arity", "number-called-as-function", "unknown-prop"]);
    expect(r.remaining).toEqual([]);
  });

  it("code the compiler accepts is never touched", () => {
    const src = scene("const p = interpolate(frame, [0, 10], [0, 1]); const e = ease(frame, 0, 20); void [p, e];");
    expect(diagnose(src)).toEqual([]);
    expect(autofixScene(src, []).source).toBe(src);
  });

  it("errors it cannot repair mechanically are left for the model", () => {
    const r = repair(scene("const x = notDefinedAnywhere(1); void x;"));
    expect(r.rules).toEqual([]);
    expect(r.remaining.join(" ")).toMatch(/Cannot find name 'notDefinedAnywhere'/);
  });
});
