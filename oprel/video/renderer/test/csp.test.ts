import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import { bundle } from "@remotion/bundler";
import { ensureBrowser, renderStill, selectComposition } from "@remotion/renderer";
import { afterAll, describe, expect, it } from "vitest";
import { injectCsp } from "../src/node/render";
import { CACHE_ROOT } from "../src/node/stage";

/**
 * Proves the render browser cannot reach local services: a hand-written composition (it deliberately bypasses the scene policy,
 * which would reject `fetch`) tries to fetch from a server on another local port.
 */
describe("render browser network isolation", () => {
  const dirs: string[] = [];
  afterAll(() => dirs.forEach((d) => fs.rmSync(d, { recursive: true, force: true })));

  async function attempt(withCsp: boolean): Promise<{ hits: number; logs: string[] }> {
    let hits = 0;
    const server = http.createServer((_req, res) => {
      hits++;
      res.setHeader("Access-Control-Allow-Origin", "*");
      res.end("pong");
    });
    await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
    const port = (server.address() as { port: number }).port;

    fs.mkdirSync(CACHE_ROOT, { recursive: true });
    const dir = fs.mkdtempSync(path.join(CACHE_ROOT, "csp-test-"));
    dirs.push(dir);
    fs.writeFileSync(
      path.join(dir, "entry.tsx"),
      `import React from "react";
import { Composition, registerRoot, delayRender, continueRender } from "remotion";
const Probe: React.FC = () => {
  const [h] = React.useState(() => delayRender("probe"));
  React.useEffect(() => {
    fetch("http://127.0.0.1:${port}/ping")
      .then(() => console.log("PROBE:leaked"))
      .catch(() => console.log("PROBE:blocked"))
      .finally(() => continueRender(h));
  }, [h]);
  return <div style={{ color: "white" }}>probe</div>;
};
registerRoot(() => <Composition id="Probe" component={Probe} width={200} height={200} fps={30} durationInFrames={5} />);
`,
    );
    const outDir = await bundle({ entryPoint: path.join(dir, "entry.tsx"), outDir: path.join(dir, "bundle") });
    if (withCsp) injectCsp(outDir);
    await ensureBrowser();
    const logs: string[] = [];
    const onBrowserLog = (l: { text: string }) => logs.push(l.text);
    const comp = await selectComposition({ serveUrl: outDir, id: "Probe", onBrowserLog });
    await renderStill({ composition: comp, serveUrl: outDir, frame: 0, output: path.join(dir, "out.png"), onBrowserLog });
    await new Promise<void>((r) => server.close(() => r()));
    return { hits, logs };
  }

  it("control: without the policy the page DOES reach another local port", async () => {
    const r = await attempt(false);
    expect(r.hits).toBeGreaterThan(0);
    expect(r.logs.join(" ")).toContain("PROBE:leaked");
  }, 240_000);

  it("with the policy the request never leaves the browser", async () => {
    const r = await attempt(true);
    expect(r.hits).toBe(0);
    expect(r.logs.join(" ")).toContain("PROBE:blocked");
  }, 240_000);
});
