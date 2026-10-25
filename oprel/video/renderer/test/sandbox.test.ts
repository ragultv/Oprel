import fs from "node:fs";
import { describe, expect, it } from "vitest";
import { checkSceneSource } from "../src/node/sandbox";

const GOOD = fs.readFileSync(new URL("../prompts/scene_example.tsx", import.meta.url), "utf8");
const wrap = (body: string, imports = 'import React from "react";\nimport { useCurrentFrame } from "remotion";') =>
  `${imports}\nexport default function Scene() {\n  const frame = useCurrentFrame();\n  ${body}\n  return <div>{frame}</div>;\n}\n`;
const messages = (src: string) => checkSceneSource(src).map((i) => i.message);

describe("scene sandbox policy", () => {
  it("accepts the documented example scene", () => {
    expect(checkSceneSource(GOOD, "scenes/ex.tsx")).toEqual([]);
  });

  it("requires a default export", () => {
    expect(messages('import React from "react";\nexport const Scene = () => <div/>;').join()).toMatch(/export default/);
  });

  it.each([
    ['fetch("https://evil.example/x")', /fetch/],
    ["new XMLHttpRequest()", /XMLHttpRequest/],
    ['new WebSocket("ws://x")', /WebSocket/],
    ['eval("1")', /eval/],
    ['new Function("return 1")', /Function/],
    ['(0, eval)("1")', /eval/],
    ["const x = window.innerWidth;", /window/],
    ["const x = document.body;", /document/],
    ["const x = globalThis;", /globalThis/],
    ["const t = Date.now();", /Date/],
    ["const t = new Date();", /Date/],
    ["const r = Math.random();", /Math\.random/],
    ["const t = performance.now();", /performance/],
    ["setTimeout(() => {}, 10);", /setTimeout/],
    ["requestAnimationFrame(() => {});", /requestAnimationFrame/],
    ['localStorage.setItem("a", "b");', /localStorage/],
    ["const p = process.env;", /process/],
    ['const m = require("fs");', /require/],
    ["const c = [].constructor.constructor;", /constructor/],
    ['const c = ({})["__proto__"];', /__proto__/],
    ['const u = "https://example.com/a.png";', /external URL/],
    ['const u = "//cdn.example.com/a.js";', /external URL/],
    ["const s = { transition: 'all 1s' };", /transition/],
    ["const s = { animation: 'spin 1s' };", /animation/],
  ])("rejects %s", (body, re) => {
    expect(messages(wrap(body as string)).join(" | ")).toMatch(re as RegExp);
  });

  it.each([
    'import fs from "fs";',
    'import x from "../../secret.json";',
    'import axios from "axios";',
    'import { Canvas } from "@react-three/fiber";',
    'import { readFile } from "node:fs/promises";',
  ])("rejects import: %s", (imp) => {
    expect(messages(wrap("", `import React from "react";\n${imp}`)).length).toBeGreaterThan(0);
  });

  it("rejects dynamic import and dangerous JSX", () => {
    expect(messages(wrap('import("fs");')).join()).toMatch(/dynamic import/);
    expect(messages(wrap("").replace("<div>{frame}</div>", "<iframe src='x' />")).join()).toMatch(/iframe/);
    expect(messages(wrap("").replace("<div>{frame}</div>", "<div dangerouslySetInnerHTML={{ __html: 'x' }} />")).join()).toMatch(/raw HTML/);
  });

  it("rejects rendering narration as subtitles in JSX to prevent duplicate captions", () => {
    const src = 'import React from "react";\nexport default function Scene({ narration }: { narration: string }) { return <div><p>{narration}</p></div>; }';
    expect(messages(src).join()).toMatch(/do not render the narration prop/);
  });

  it("allows SVG namespaces and ordinary local variables that share names with browser globals", () => {
    const src = wrap('const top = 10; const parent = { open: 1 }; const label = "http://www.w3.org/2000/svg"; void top; void parent; void label;');
    expect(checkSceneSource(src)).toEqual([]);
  });

  it.each(["animate", "animateTransform", "animateMotion", "set"])("rejects time-based SVG <%s> with the frame-based alternative", (tag) => {
    const src = wrap("").replace("<div>{frame}</div>", `<svg><circle r={5}><${tag} attributeName="cx" /></circle></svg>`);
    const m = messages(src).join(" ");
    expect(m).toMatch(new RegExp(`<${tag}> is time-based`));
    expect(m).toMatch(/interpolate\(frame/);
  });

  it("reports syntax errors with a line number", () => {
    const issues = checkSceneSource('import React from "react";\nexport default function Scene() { return <div> ; }');
    expect(issues.length).toBeGreaterThan(0);
    expect(issues[0].path).toMatch(/scene\.tsx:\d+/);
  });

  it("rejects oversized files", () => {
    expect(messages(wrap("// " + "x".repeat(70_000))).join()).toMatch(/larger than/);
  });
});
