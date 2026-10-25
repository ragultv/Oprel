import React from "react";
import { AbsoluteFill, useCurrentFrame, useVideoConfig } from "remotion";
import { Backdrop } from "./parts";
import { Header } from "./Header";
import { ease } from "../lib/motion";
import { tokens } from "../lib/design";
import type { Scene } from "../schema";

const L = 120, R = 960, BASE = 1300, MAXH = 600;

export const ProbabilityChartScene: React.FC<{ scene: Extract<Scene, { type: "probability-chart" }> }> = ({ scene }) => {
  const frame = useCurrentFrame();
  const { width, height } = useVideoConfig();
  const { bars, dataKind, source, accent, headline, subtitle } = scene.params;
  const a = tokens.accent[accent];
  const slot = (R - L) / bars.length;
  const bw = Math.min(150, slot * 0.62);
  const top = Math.max(...bars.map((b) => b.value));

  return (
    <Backdrop accent={accent}>
      <Header headline={headline} subtitle={subtitle} accent={accent} />
      <AbsoluteFill>
        <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`}>
          <line x1={L - 20} y1={BASE} x2={R + 20} y2={BASE} stroke={tokens.text.secondary} strokeWidth={2} opacity={0.5} />
          {bars.map((b, i) => {
            const p = ease(frame, 14 + i * 8, 44 + i * 8);
            const h = b.value * MAXH * p;
            const cx = L + slot * (i + 0.5);
            const hi = b.value === top;
            return (
              <g key={i}>
                <rect x={cx - bw / 2} y={BASE - h} width={bw} height={h} rx={12} fill={hi ? a.main : a.soft} opacity={hi ? 1 : 0.8}
                  style={hi ? { filter: `drop-shadow(0 0 14px ${a.glow})` } : undefined} />
                <text x={cx} y={BASE - h - 18} textAnchor="middle" fontSize={40} fontWeight={800} fill={tokens.text.primary}
                  fontFamily={tokens.font.family} opacity={p}>{(b.value * p).toFixed(2)}</text>
                <text x={cx} y={BASE + 52} textAnchor="middle" fontSize={30} fontWeight={600} fill={tokens.text.secondary}
                  fontFamily={tokens.font.family}>{b.label}</text>
              </g>
            );
          })}
          <text x={L - 20} y={BASE + 120} fontSize={28} fontWeight={700}
            fill={dataKind === "illustrative" ? tokens.accent.violet.main : tokens.text.secondary} fontFamily={tokens.font.family}>
            {dataKind === "illustrative" ? "ILLUSTRATIVE VALUES - NOT MEASURED RESULTS" : `Source: ${source}`}
          </text>
        </svg>
      </AbsoluteFill>
    </Backdrop>
  );
};
