import React from "react";
import { AbsoluteFill, Audio, Sequence, staticFile, useCurrentFrame, useVideoConfig } from "remotion";
import type { CompiledTimeline } from "./compiler";
import { renderScene } from "./scenes/registry";
import { FadeIn } from "./scenes/parts";
import { ensureFonts, tokens } from "./lib/design";

export type VideoProps = { timeline: CompiledTimeline };

ensureFonts();

const Captions: React.FC<{ captions: CompiledTimeline["captions"] }> = ({ captions }) => {
  const frame = useCurrentFrame();
  const active = captions.find((c) => frame >= c.from && frame < c.to);
  if (!active || !active.text?.trim()) return null;
  return (
    <AbsoluteFill style={{ justifyContent: "flex-end", alignItems: "center", paddingBottom: tokens.safe.captionBottom, pointerEvents: "none" }}>
      <div
        style={{
          maxWidth: 860,
          padding: "14px 28px",
          borderRadius: 16,
          background: "rgba(7, 10, 18, 0.82)",
          border: "1px solid rgba(255, 255, 255, 0.12)",
          boxShadow: "0 10px 30px rgba(0, 0, 0, 0.6)",
          backdropFilter: "blur(10px)",
          color: tokens.text.primary,
          fontFamily: tokens.font.family,
          fontSize: 38,
          fontWeight: 700,
          textAlign: "center",
          lineHeight: 1.3,
          letterSpacing: "-0.01em",
        }}
      >
        {active.text}
      </div>
    </AbsoluteFill>
  );
};

export const VideoComposition: React.FC<VideoProps> = ({ timeline }) => {
  const { fps } = useVideoConfig();
  void fps;
  return (
    <AbsoluteFill style={{ background: tokens.bg.deep }}>
      {timeline.scenes.map((c) => (
        <Sequence key={c.id} from={c.from} durationInFrames={c.durationInFrames} name={c.id} layout="none">
          <AbsoluteFill>
            <FadeIn frames={c.transition.type === "fade" ? c.transition.durationFrames : 0}>{renderScene(c)}</FadeIn>
          </AbsoluteFill>
        </Sequence>
      ))}
      <Captions captions={timeline.captions} />
      {timeline.narration ? <Audio src={staticFile(timeline.narration.file)} volume={timeline.narration.volume} /> : null}
    </AbsoluteFill>
  );
};
