import React from "react";
import { AbsoluteFill, useCurrentFrame, useVideoConfig } from "remotion";
import { Backdrop } from "./parts";
import { Header } from "./Header";
import { clamp01, ease, reveal } from "../lib/motion";
import { tokens } from "../lib/design";
import type { Scene } from "../schema";

const BOX = { x: 240, y: 840, w: 600, h: 240 };

/** Tokens drop into a stage box from above and different tokens leave from below. */
export const TokenFlowScene: React.FC<{ scene: Extract<Scene, { type: "token-flow" }> }> = ({ scene }) => {
  const rawFrame = useCurrentFrame();
  const { fps, width, height } = useVideoConfig();
  const { inputTokens, outputTokens, stageLabel, accent, headline, subtitle } = scene.params;
  const a = tokens.accent[accent];
  const gap = 10; // frames between consecutive tokens
  const travel = Math.round(fps * 0.9);
  const start = 14;

  const chip = (text: string, x: number, y: number, o: number, color: string, key: string, maxW: number) => {
    const w = Math.min(maxW, Math.max(130, text.length * 24 + 48));
    const fs = Math.min(38, (w - 28) / Math.max(1, text.length * 0.6));
    return (
      <g key={key} opacity={o}>
        <rect x={x - w / 2} y={y - 34} width={w} height={68} rx={16} fill={tokens.bg.panel} stroke={color} strokeWidth={3} />
        <text x={x} y={y + fs * 0.34} textAnchor="middle" fontSize={fs} fontWeight={700} fill={tokens.text.primary} fontFamily={tokens.font.family}>{text}</text>
      </g>
    );
  };

  const spacing = (n: number) => Math.min(280, 920 / n);
  const slotX = (i: number, n: number) => width / 2 + (i - (n - 1) / 2) * spacing(n);
  const inEnd = start + inputTokens.length * gap + travel;
  const outStart = inEnd - 6;
  // The token animation loops so the scene is never static for long scenes.
  const period = outStart + outputTokens.length * gap + travel + 40;
  const frame = rawFrame % period;
  const boxPulse = clamp01(1 - Math.abs(frame - inEnd) / 12);

  return (
    <Backdrop accent={accent}>
      <Header headline={headline} subtitle={subtitle} accent={accent} />
      <AbsoluteFill>
        <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`}>
          <rect x={BOX.x} y={BOX.y} width={BOX.w} height={BOX.h} rx={32} fill={tokens.bg.panel} stroke={a.main} strokeWidth={4 + boxPulse * 4}
            opacity={reveal(frame, fps)} style={{ filter: `drop-shadow(0 0 ${12 + boxPulse * 24}px ${a.glow})` }} />
          <text x={width / 2} y={BOX.y + BOX.h / 2 + 16} textAnchor="middle" fontSize={54} fontWeight={800} fill={tokens.text.primary}
            fontFamily={tokens.font.family} opacity={reveal(frame, fps)}>{stageLabel}</text>
          {inputTokens.map((tk, i) => {
            const p = ease(frame, start + i * gap, start + i * gap + travel);
            const y = 620 + (BOX.y + 40 - 620) * p;
            return chip(tk, slotX(i, inputTokens.length), y, frame < start + i * gap ? 0 : 1 - clamp01((p - 0.85) / 0.15), tokens.text.secondary, `i${i}`, spacing(inputTokens.length) - 14);
          })}
          {outputTokens.map((tk, i) => {
            const s = outStart + i * gap;
            const p = ease(frame, s, s + travel);
            const y = BOX.y + BOX.h - 40 + (1360 - (BOX.y + BOX.h - 40)) * p;
            return chip(tk, slotX(i, outputTokens.length), y, frame < s ? 0 : clamp01(p / 0.15), a.main, `o${i}`, spacing(outputTokens.length) - 14);
          })}
          <text x={width / 2} y={600} textAnchor="middle" fontSize={30} fill={tokens.text.secondary} fontFamily={tokens.font.family}>input</text>
          <text x={width / 2} y={1450} textAnchor="middle" fontSize={30} fill={tokens.text.secondary} fontFamily={tokens.font.family}
            opacity={ease(frame, outStart + 10, outStart + 30)}>output</text>
        </svg>
      </AbsoluteFill>
    </Backdrop>
  );
};
