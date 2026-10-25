import { describe, expect, it } from "vitest";
import { compileStoryboard } from "../src/compiler";
import { IMPLEMENTED_SCENE_TYPES } from "../src/sceneTypes";
import { heroPose, seededPoints } from "../src/lib/motion";

const hook = (id: string, extra: object = {}) => ({
  id,
  type: "hook",
  durationSeconds: 1,
  params: { headline: "Hi" },
  ...extra,
});
const base = (scenes: unknown[], more: object = {}) => ({ schemaVersion: 1, title: "t", scenes, ...more });

describe("compileStoryboard", () => {
  it("45s at 30fps is exactly 1350 frames with cumulative ranges", () => {
    const r = compileStoryboard(base([hook("a", { durationSeconds: 15 }), hook("b", { durationSeconds: 30 })]));
    expect(r.ok).toBe(true);
    if (!r.ok) return;
    expect(r.timeline.durationInFrames).toBe(1350);
    expect(r.timeline.scenes.map((s) => [s.from, s.durationInFrames])).toEqual([[0, 450], [450, 900]]);
    expect(r.timeline.width).toBe(1080);
    expect(r.timeline.height).toBe(1920);
  });

  it("is deterministic", () => {
    const sb = base([hook("a"), hook("b", { durationFrames: 17 })]);
    expect(JSON.stringify(compileStoryboard(sb))).toBe(JSON.stringify(compileStoryboard(sb)));
  });

  it("rejects duplicate ids, missing and double durations", () => {
    expect(compileStoryboard(base([hook("a"), hook("a")])).ok).toBe(false);
    expect(compileStoryboard(base([{ id: "a", type: "hook", params: { headline: "x" } }])).ok).toBe(false);
    expect(compileStoryboard(base([hook("a", { durationFrames: 10 })])).ok).toBe(false);
  });

  it("rejects unknown and not-yet-implemented scene types instead of rendering blank", () => {
    expect(compileStoryboard(base([{ id: "a", type: "token-flow", durationSeconds: 1, params: {} }])).ok).toBe(false);
  });

  it("rejects transitions longer than half the scene", () => {
    const r = compileStoryboard(base([hook("a", { durationFrames: 20, transition: { type: "fade", durationFrames: 15 } })]));
    expect(r.ok).toBe(false);
  });

  it("rejects unsafe asset paths", () => {
    for (const bad of ["../x.wav", "/etc/passwd", "C:/x.wav", "a\\b.wav", "file:///x", "assets//x.wav"]) {
      expect(compileStoryboard(base([hook("a")], { narration: { file: bad } })).ok, bad).toBe(false);
    }
  });

  it("reports missing assets via the injected resolver", () => {
    const sb = base([hook("a")], { narration: { file: "assets/n.wav" } });
    expect(compileStoryboard(sb, { assetExists: () => false }).ok).toBe(false);
    expect(compileStoryboard(sb, { assetExists: () => true }).ok).toBe(true);
  });

  it("flags narration longer than the timeline, and overlapping captions", () => {
    const long = base([hook("a", { durationSeconds: 1 })], { narration: { file: "assets/n.wav", durationMs: 5000 } });
    expect(compileStoryboard(long).ok).toBe(false);
    const caps = base([hook("a", { durationSeconds: 5 })], {
      captions: [
        { startMs: 0, endMs: 2000, text: "a" },
        { startMs: 1000, endMs: 3000, text: "b" },
      ],
    });
    expect(compileStoryboard(caps).ok).toBe(false);
  });

  it("converts captions to frames", () => {
    const r = compileStoryboard(base([hook("a", { durationSeconds: 5 })], { captions: [{ startMs: 1000, endMs: 2000, text: "x" }] }));
    expect(r.ok && r.timeline.captions[0]).toMatchObject({ from: 30, to: 60 });
  });
});

describe("new scene schemas", () => {
  const scene = (type: string, params: object) => ({ id: "s", type, durationSeconds: 2, params });
  it("rejects measured chart data without a source, accepts illustrative", () => {
    const bars = [{ label: "a", value: 0.5 }, { label: "b", value: 0.2 }];
    expect(compileStoryboard(base([scene("probability-chart", { headline: "h", bars, dataKind: "measured" })])).ok).toBe(false);
    expect(compileStoryboard(base([scene("probability-chart", { headline: "h", bars, dataKind: "measured", source: "paper" })])).ok).toBe(true);
    expect(compileStoryboard(base([scene("probability-chart", { headline: "h", bars })])).ok).toBe(true);
  });
  it("rejects chart values outside 0..1", () => {
    const bars = [{ label: "a", value: 1.5 }, { label: "b", value: 0.2 }];
    expect(compileStoryboard(base([scene("probability-chart", { headline: "h", bars })])).ok).toBe(false);
  });
  it("validates architecture diagram edges and grid cells", () => {
    const nodes = [{ id: "a", label: "A", col: 1, row: 0 }, { id: "b", label: "B", col: 1, row: 1 }];
    const ok = scene("architecture-diagram", { headline: "h", nodes, edges: [{ from: "a", to: "b" }] });
    expect(compileStoryboard(base([ok])).ok).toBe(true);
    expect(compileStoryboard(base([scene("architecture-diagram", { headline: "h", nodes, edges: [{ from: "a", to: "zzz" }] })])).ok).toBe(false);
    const clash = [{ id: "a", label: "A", col: 1, row: 0 }, { id: "b", label: "B", col: 1, row: 0 }];
    expect(compileStoryboard(base([scene("architecture-diagram", { headline: "h", nodes: clash, edges: [] })])).ok).toBe(false);
  });
  it("token-flow and neural-network apply defaults", () => {
    expect(compileStoryboard(base([scene("token-flow", { headline: "h", inputTokens: ["a"], outputTokens: ["b"] })])).ok).toBe(true);
    expect(compileStoryboard(base([scene("neural-network", { headline: "h" })])).ok).toBe(true);
  });
});

describe("scene registry", () => {
  it("every implemented scene type has a component registered", async () => {
    const { sceneRegistry } = await import("../src/scenes/registry");
    expect(Object.keys(sceneRegistry).sort()).toEqual([...IMPLEMENTED_SCENE_TYPES].sort());
  });
});

describe("deterministic scene maths", () => {
  it("heroPose is a pure function of its inputs", () => {
    for (const f of [0, 17, 63, 149]) expect(heroPose(f, 150, 30)).toEqual(heroPose(f, 150, 30));
  });
  it("scrubbing order does not matter", () => {
    const forward = [5, 40, 90].map((f) => heroPose(f, 150, 30));
    const backward = [90, 40, 5].map((f) => heroPose(f, 150, 30)).reverse();
    expect(forward).toEqual(backward);
  });
  it("seeded points are stable per seed and differ between seeds", () => {
    expect(seededPoints("a", 10, 2)).toEqual(seededPoints("a", 10, 2));
    expect(seededPoints("a", 10, 2)).not.toEqual(seededPoints("b", 10, 2));
  });
});
