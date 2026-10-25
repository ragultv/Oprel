import React from "react";
import { AbsoluteFill, useCurrentFrame, useVideoConfig } from "remotion";
import { Backdrop } from "./parts";
import { reveal } from "../lib/motion";
import { tokens } from "../lib/design";
import type { Scene } from "../schema";

export const ConclusionScene: React.FC<{ scene: Extract<Scene, { type: "conclusion" }> }> = ({ scene }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const { title, points, accent } = scene.params;
  const a = tokens.accent[accent];
  return (
    <Backdrop accent={accent}>
      <AbsoluteFill style={{ justifyContent: "center", padding: `${tokens.safe.top}px ${tokens.safe.side}px ${tokens.safe.bottom}px` }}>
        <div style={{ opacity: reveal(frame, fps), fontSize: 88, fontWeight: 800, color: tokens.text.primary, letterSpacing: -2, lineHeight: 1.05 }}>
          {title}
        </div>
        <div style={{ marginTop: 48, display: "flex", flexDirection: "column", gap: 24 }}>
          {points.map((pt, i) => {
            const p = reveal(frame, fps, 12 + i * 10);
            return (
              <div key={i} style={{ display: "flex", gap: 20, alignItems: "center", opacity: p, transform: `translateX(${(1 - p) * 40}px)` }}>
                <div style={{ width: 14, height: 14, borderRadius: 7, background: a.main, boxShadow: `0 0 16px ${a.glow}` }} />
                <div style={{ fontSize: 44, fontWeight: 600, color: tokens.text.secondary }}>{pt}</div>
              </div>
            );
          })}
        </div>
      </AbsoluteFill>
    </Backdrop>
  );
};
