import React from "react";
import { Composition } from "remotion";
import { VideoComposition, type VideoProps } from "./Composition";

import { COMPOSITION_ID } from "./constants";
export { COMPOSITION_ID };

/** Placeholder props used only by tooling that needs defaults; real renders always pass a compiled timeline. */
const defaultTimeline: VideoProps["timeline"] = {
  schemaVersion: 1,
  title: "default",
  fps: 30,
  width: 1080,
  height: 1920,
  durationInFrames: 30,
  scenes: [],
  captions: [],
  narration: null,
  assets: [],
  customFiles: [],
};

export const Root: React.FC = () => (
  <Composition
    id={COMPOSITION_ID}
    component={VideoComposition}
    width={1080}
    height={1920}
    fps={30}
    durationInFrames={30}
    defaultProps={{ timeline: defaultTimeline }}
    // The compiled timeline is the single source of truth for size, fps and length.
    calculateMetadata={({ props }) => ({
      durationInFrames: props.timeline.durationInFrames,
      fps: props.timeline.fps,
      width: props.timeline.width,
      height: props.timeline.height,
    })}
  />
);
