import fs from "node:fs";
import path from "node:path";
import { afterAll, describe, expect, it } from "vitest";
import { KIT_EXPORTS, newStage, typecheckScene } from "../src/node/stage";

const stages: ReturnType<typeof newStage>[] = [];
afterAll(() => stages.forEach((s) => s.cleanup()));

/** Type-check a scene body exactly as the agent does (staged inside the renderer so imports resolve). */
function check(source: string) {
  const stage = newStage("tc-test");
  stages.push(stage);
  const rel = "scenes/probe.tsx";
  const abs = path.join(stage.dir, rel);
  fs.mkdirSync(path.dirname(abs), { recursive: true });
  fs.writeFileSync(abs, source);
  return typecheckScene(abs, rel).map((i) => i.message);
}

const scene = (imports: string, body: string) => `import React from "react";\n${imports}\nexport default function Scene() {\n  ${body}\n  return <div />;\n}\n`;

describe("@oprel/kit exports", () => {
  it("every documented name can be imported from the kit (keeps KIT_EXPORTS honest)", () => {
    const values = KIT_EXPORTS.filter((n) => n !== "SceneProps");
    expect(check(scene(`import { ${values.join(", ")}, type SceneProps } from "@oprel/kit";`, `void [${values.join(", ")}];`))).toEqual([]);
  });

  it("spring/interpolate can be imported from the kit (the mistake that repeated 4 times)", () => {
    expect(check(scene('import { spring, interpolate, useCurrentFrame, useVideoConfig } from "@oprel/kit";',
      "const f = useCurrentFrame(); const { fps } = useVideoConfig(); const a = spring({ frame: f, fps }); const b = interpolate(f, [0, 10], [0, 1]); void [a, b];"))).toEqual([]);
  });
});

describe("type error hints written for small models", () => {
  it("unknown kit export lists what the kit really exports", () => {
    const m = check(scene('import { Foo } from "@oprel/kit";', "void Foo;")).join(" | ");
    expect(m).toMatch(/TS2305/);
    expect(m).toMatch(/@oprel\/kit exports only:.*spring/);
  });
  it("missing import (Header) says where Header comes from", () => {
    const m = check(scene('import { Backdrop } from "@oprel/kit";', "void Backdrop; void Header;")).join(" | ");
    expect(m).toMatch(/Cannot find name 'Header'/);
    expect(m).toMatch(/Hint: "Header" is not defined: add it to an import.*@oprel\/kit/);
  });
  it("Vector3 from @remotion/three points at the three package", () => {
    const m = check(scene('import { Vector3 } from "@remotion/three";', "void Vector3;")).join(" | ");
    expect(m).toMatch(/Hint: .*"three"/);
  });
  it("wrong interpolate arity shows the signatures", () => {
    const m = check(scene('import { interpolate } from "@oprel/kit";', "void interpolate(1, [0, 1]);")).join(" | ");
    expect(m).toMatch(/Expected 3-4 arguments/);
    expect(m).toMatch(/Hint: wrong number of arguments\. Signatures: interpolate\(value, inputRange, outputRange/);
  });
  it("textAlign on SVG text suggests textAnchor", () => {
    const m = check(scene("", "void 0;").replace("<div />", '<svg><text textAlign="center" x={1} y={1}>a</text></svg>')).join(" | ");
    expect(m).toMatch(/Hint: SVG <text> has no textAlign; use textAnchor/);
  });
  it("using .value on spring() explains it returns a plain number", () => {
    const m = check(scene('import { spring } from "@oprel/kit";', "const v = spring({ frame: 1, fps: 30 }).value; void v;")).join(" | ");
    expect(m).toMatch(/Hint: spring\(\), interpolate\(\) and ease\(\) return plain numbers/);
  });
});
