import { describe, expect, it } from "vitest";
import { PNG } from "pngjs";
import { analysePng, changedFraction, MIN_CHANGED_FRACTION, MIN_CONTENT_BOX, sampleFrames } from "../src/node/scenecheck";

/** 540x960 image: smooth dark gradient, optional white rectangle. */
function image(rect?: { x: number; y: number; w: number; h: number }): Buffer {
  const png = new PNG({ width: 540, height: 960 });
  for (let y = 0; y < 960; y++)
    for (let x = 0; x < 540; x++) {
      const i = (y * 540 + x) * 4;
      const inRect = rect && x >= rect.x && x < rect.x + rect.w && y >= rect.y && y < rect.y + rect.h;
      const v = inRect ? 255 : 8 + Math.round((y / 960) * 10);
      png.data[i] = png.data[i + 1] = png.data[i + 2] = v;
      png.data[i + 3] = 255;
    }
  return PNG.sync.write(png);
}

describe("scene frame analysis", () => {
  it("a plain gradient has no edges anywhere", () => {
    const s = analysePng(image());
    expect(s.edgeFraction).toBeLessThan(0.0005);
    expect(s.bottomFraction).toBe(0);
  });
  it("a shape in the middle is detected but is not in the bottom zone", () => {
    const s = analysePng(image({ x: 100, y: 300, w: 300, h: 200 }));
    expect(s.edgeFraction).toBeGreaterThan(0.005);
    expect(s.bottomFraction).toBe(0);
  });
  it("a shape in the bottom zone (y > 1780 on the 1920 canvas) is flagged", () => {
    const s = analysePng(image({ x: 100, y: 900, w: 300, h: 50 }));
    expect(s.bottomFraction).toBeGreaterThan(0.005);
  });
  it("content running off the frame edge (striped text-like pattern) raises the border fraction", () => {
    const png = new PNG({ width: 540, height: 960 });
    for (let y = 0; y < 960; y++)
      for (let x = 0; x < 540; x++) {
        const i = (y * 540 + x) * 4;
        const stripe = x < 30 && y > 100 && y < 700 && Math.floor(y / 4) % 2 === 0;
        png.data[i] = png.data[i + 1] = png.data[i + 2] = stripe ? 255 : 10;
        png.data[i + 3] = 255;
      }
    expect(analysePng(PNG.sync.write(png)).borderFraction).toBeGreaterThan(0.06);
  });
  it("samples early, middle and late frames without duplicates", () => {
    expect(sampleFrames(300)).toEqual([2, 75, 150, 224, 297]);
    expect(sampleFrames(3)).toEqual([0, 1, 2]);
    expect(new Set(sampleFrames(1)).size).toBe(1);
  });
});

describe("motion and content-size checks (calibrated on real model output)", () => {
  const lumaOf = (png: Buffer) => analysePng(png).luma;

  it("identical frames have zero changed pixels (a dead scene)", () => {
    const a = lumaOf(image({ x: 100, y: 300, w: 200, h: 100 }));
    expect(changedFraction(a, a)).toBe(0);
    expect(changedFraction(a, a) < MIN_CHANGED_FRACTION).toBe(true);
  });

  it("a single small element moving counts as animation even though it barely moves the frame average", () => {
    const a = lumaOf(image({ x: 100, y: 300, w: 24, h: 24 }));
    const b = lumaOf(image({ x: 160, y: 300, w: 24, h: 24 }));
    const moved = changedFraction(a, b);
    expect(moved).toBeGreaterThanOrEqual(MIN_CHANGED_FRACTION);
    expect(moved).toBeLessThan(0.01); // tiny share of the frame, which the old mean-difference test (< 0.25) rejected
  });

  it("content crammed into a corner is under the minimum box; content spread over the frame is not", () => {
    expect(analysePng(image({ x: 20, y: 20, w: 90, h: 140 })).boxFraction).toBeLessThan(MIN_CONTENT_BOX);
    expect(analysePng(image({ x: 60, y: 200, w: 420, h: 480 })).boxFraction).toBeGreaterThan(MIN_CONTENT_BOX);
    expect(analysePng(image()).boxFraction).toBe(0);
  });
});
