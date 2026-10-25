# REMOTION MOTION GRAPHICS COMPACT SPEC
Write ONE React scene file for a 1080x1920 (9:16), 30 fps video. Reply with the whole file in one ```tsx block.

- `export default function Scene({ durationInFrames, narration, title }: SceneProps)`
- DO NOT draw boring application cards, forms, or web wireframe boxes. Create dynamic, broadcast-quality motion graphics: kinetic typography, dynamic SVG path vectors (`strokeDashoffset`), pulsing data nodes, animated charts, particle fields, floating metric readouts.
- Allowed imports: "react", "remotion" (AbsoluteFill, useCurrentFrame, useVideoConfig, interpolate, spring, random, Easing, Sequence, interpolateColors).
- Official Remotion Skills:
  - Animate ONLY from `useCurrentFrame()` and `useVideoConfig()`. Derive entrance/exit durations using `fps`.
  - Prefer transform shorthands `scale`, `translate`, `rotate` in style objects. Use `output: "perceptual-scale"` for scale interpolations.
  - Use `Easing.bezier(0.16, 1, 0.3, 1)` or `spring()` for natural physical momentum.
  - Safe area: Keep key elements between `y = 120` and `y = 1780` (top 120px and bottom 140px system margins). Side margin 64px. Elements can utilize the vertical height naturally.
- Signatures & Timing:
  - `const frame = useCurrentFrame(); const { fps, width, height } = useVideoConfig();`
  - `interpolate(frame, [0, 30], [0, 1], { extrapolateRight: "clamp", easing: Easing.bezier(0.16, 1, 0.3, 1) })`
  - `spring({ frame: frame - delay, fps, config: { damping: 16, mass: 0.8 } })` -> number 0..1
  - Maintain continuous movement across all frames; do not freeze early.
  - Vary visual metaphors according to the prompt (typography, network graphs, data paths, 3D meshes). Do NOT reproduce circular meters or cards.
  - DO NOT render the `narration` prop as subtitles/paragraphs in your JSX (the video player renders captions globally). Only render kinetic titles, diagrams, and motion graphics.

Example:
```tsx
import React from "react";
import { AbsoluteFill, spring, interpolate, useCurrentFrame, useVideoConfig, Easing } from "remotion";

export interface SceneProps {
  durationInFrames: number;
  fps: number;
  width: number;
  height: number;
  narration: string;
  title: string;
}

export default function Scene({ title, durationInFrames, narration }: SceneProps) {
  const frame = useCurrentFrame();
  const { fps, width, height } = useVideoConfig();

  // 1. Entrance animation
  const enter = spring({ frame: frame - 4, fps, config: { damping: 16 } });
  // 2. Continuous secondary animation across all frames
  const progress = interpolate(frame, [0, durationInFrames], [0, 1], { extrapolateRight: "clamp" });

  return (
    <AbsoluteFill style={{ backgroundColor: "#0b0f17", padding: 72, fontFamily: "system-ui, sans-serif", overflow: "hidden" }}>
      {/* Dynamic scene visual content tailored to title and narration */}
    </AbsoluteFill>
  );
}
```
