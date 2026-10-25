/**
 * "@oprel/kit": the small, stable toolbox AI-written scenes may import besides react / remotion / three.
 * Everything here is deterministic and frame-driven. Keep this surface small: it is documented to the model verbatim.
 */
// The Remotion helpers every scene uses, re-exported so `import { spring, interpolate } from "@oprel/kit"` just works.
export { AbsoluteFill, Easing, Sequence, interpolate, interpolateColors, random, spring, useCurrentFrame, useVideoConfig } from "remotion";
export { tokens } from "./lib/design";
export { ease, reveal, clamp01, seededPoints } from "./lib/motion";
export { Backdrop, TitleBlock, FadeIn } from "./scenes/parts";
export { Header } from "./scenes/Header";
export type { Accent } from "./schema";

/** Props every scene component receives. */
export type SceneProps = {
  /** Length of this scene in frames. useCurrentFrame() runs from 0 to durationInFrames - 1 inside the scene. */
  durationInFrames: number;
  fps: number;
  /** Canvas size in px (1080 x 1920 for Shorts). */
  width: number;
  height: number;
  /** What the narrator says during this scene (may be empty). */
  narration: string;
  /** Optional short title of the scene. */
  title: string;
  /** Free-form string/number/boolean values the plan attached to this scene. */
  props: Record<string, string | number | boolean>;
};
