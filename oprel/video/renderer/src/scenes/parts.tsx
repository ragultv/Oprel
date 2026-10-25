import React from "react";
import { AbsoluteFill, useCurrentFrame, useVideoConfig } from "remotion";
import { ease, reveal } from "../lib/motion";
import { tokens } from "../lib/design";
import type { Accent } from "../schema";

export const Backdrop: React.FC<{ accent?: Accent; children?: React.ReactNode }> = ({ accent = "cyan", children }) => {
  const a = tokens.accent[accent];
  return (
    <AbsoluteFill
      style={{
        background: `radial-gradient(120% 70% at 50% 28%, ${tokens.bg.navy} 0%, ${tokens.bg.deep} 70%)`,
        fontFamily: tokens.font.family,
      }}
    >
      <AbsoluteFill style={{ background: `radial-gradient(50% 30% at 50% 30%, ${a.glow} 0%, transparent 70%)`, opacity: 0.35 }} />
      {children}
    </AbsoluteFill>
  );
};

/** Headline + subtitle block placed within the mobile safe area. */
export const TitleBlock: React.FC<{
  headline: string;
  subtitle?: string;
  accent?: Accent;
  position?: "top" | "center" | "lower";
  delay?: number;
}> = ({ headline, subtitle, accent = "cyan", position = "lower", delay = 0 }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const p = reveal(frame, fps, delay);
  const justify = position === "top" ? "flex-start" : position === "center" ? "center" : "flex-end";
  return (
    <AbsoluteFill
      style={{
        justifyContent: justify,
        alignItems: "flex-start",
        padding: `${tokens.safe.top}px ${tokens.safe.side}px ${position === "lower" ? tokens.safe.bottom : tokens.safe.top}px`,
      }}
    >
      <div style={{ opacity: p, transform: `translateY(${(1 - p) * 40}px)` }}>
        <div style={{ width: 96 * p, height: 6, borderRadius: 3, background: tokens.accent[accent].main, marginBottom: 28 }} />
        <div style={{ fontSize: 92, lineHeight: 1.05, fontWeight: 800, color: tokens.text.primary, letterSpacing: -2 }}>{headline}</div>
        {subtitle ? (
          <div
            style={{
              marginTop: 28,
              fontSize: 40,
              lineHeight: 1.3,
              fontWeight: 500,
              color: tokens.text.secondary,
              opacity: ease(frame, delay + 10, delay + 30),
            }}
          >
            {subtitle}
          </div>
        ) : null}
      </div>
    </AbsoluteFill>
  );
};

/** Fade-in from the background over the first `frames` frames of the enclosing Sequence. */
export const FadeIn: React.FC<{ frames: number; children: React.ReactNode }> = ({ frames, children }) => {
  const frame = useCurrentFrame();
  const o = frames > 0 ? ease(frame, 0, frames) : 1;
  return <AbsoluteFill style={{ opacity: o }}>{children}</AbsoluteFill>;
};
