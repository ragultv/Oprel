import React from "react";
import { AbsoluteFill, useCurrentFrame, useVideoConfig } from "remotion";
import { ease, reveal } from "../lib/motion";
import { tokens } from "../lib/design";
import type { Accent } from "../schema";

/** Compact headline at the top of a diagram scene; leaves the middle of the frame for the visual. */
export const Header: React.FC<{ headline: string; subtitle?: string; accent?: Accent }> = ({ headline, subtitle, accent = "cyan" }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const p = reveal(frame, fps);
  return (
    <AbsoluteFill style={{ padding: `${tokens.safe.top}px ${tokens.safe.side}px 0`, alignItems: "flex-start", justifyContent: "flex-start" }}>
      <div style={{ opacity: p, transform: `translateY(${(1 - p) * 30}px)` }}>
        <div style={{ width: 80 * p, height: 6, borderRadius: 3, background: tokens.accent[accent].main, marginBottom: 22 }} />
        <div style={{ fontSize: 70, fontWeight: 800, lineHeight: 1.08, color: tokens.text.primary, letterSpacing: -1.5 }}>{headline}</div>
        {subtitle ? (
          <div style={{ marginTop: 18, fontSize: 36, lineHeight: 1.3, fontWeight: 500, color: tokens.text.secondary, opacity: ease(frame, 10, 30) }}>
            {subtitle}
          </div>
        ) : null}
      </div>
    </AbsoluteFill>
  );
};
