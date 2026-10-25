import React from "react";
import { AbsoluteFill, useCurrentFrame, useVideoConfig } from "remotion";
import { Backdrop } from "./parts";
import { Header } from "./Header";
import { clamp01, ease } from "../lib/motion";
import { tokens } from "../lib/design";
import type { Scene } from "../schema";

const NODE_H = 96;
const colX = [200, 540, 880];
const rowY = (r: number) => 640 + r * 150;

/** Blocks reveal in array order; each edge draws after both ends exist, then a dot travels along it. */
export const ArchitectureDiagramScene: React.FC<{ scene: Extract<Scene, { type: "architecture-diagram" }> }> = ({ scene }) => {
  const frame = useCurrentFrame();
  const { width, height } = useVideoConfig();
  const { nodes, edges, accent, headline, subtitle } = scene.params;
  const a = tokens.accent[accent];
  // wider blocks when everything sits in one column; narrower when columns must fit side by side
  const NODE_W = new Set(nodes.map((n) => n.col)).size > 1 ? 300 : 420;
  const STEP = 14, FIRST = 16;
  const nodeAt = (i: number) => FIRST + i * STEP;
  const byId = new Map(nodes.map((n, i) => [n.id, { n, i }]));

  return (
    <Backdrop accent={accent}>
      <Header headline={headline} subtitle={subtitle} accent={accent} />
      <AbsoluteFill>
        <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`}>
          <defs>
            <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
              <path d="M0,0 L10,5 L0,10 z" fill={a.main} />
            </marker>
          </defs>
          {edges.map((e, k) => {
            const f = byId.get(e.from)!, t = byId.get(e.to)!;
            const x1 = colX[f.n.col], y1 = rowY(f.n.row), x2 = colX[t.n.col], y2 = rowY(t.n.row);
            const dx = x2 - x1, dy = y2 - y1, len = Math.hypot(dx, dy) || 1;
            // shorten the line so it starts and ends at the node borders
            const trim = Math.min(
              Math.abs(dx) > 0 ? NODE_W / 2 / (Math.abs(dx) / len) : Infinity,
              Math.abs(dy) > 0 ? NODE_H / 2 / (Math.abs(dy) / len) : Infinity,
            ) + 6;
            const sx = x1 + (dx / len) * trim, sy = y1 + (dy / len) * trim;
            const ex = x2 - (dx / len) * trim, ey = y2 - (dy / len) * trim;
            const begin = Math.max(nodeAt(f.i), nodeAt(t.i)) + 6;
            const p = ease(frame, begin, begin + 16);
            const flow = clamp01(((frame - begin - 16) % 54) / 40);
            return (
              <g key={k}>
                <line x1={sx} y1={sy} x2={sx + (ex - sx) * p} y2={sy + (ey - sy) * p} stroke={a.main} strokeWidth={4}
                  markerEnd={p > 0.98 ? "url(#arrow)" : undefined} opacity={0.9} />
                {frame > begin + 16 ? <circle cx={sx + (ex - sx) * flow} cy={sy + (ey - sy) * flow} r={8} fill="#fff" opacity={0.9} /> : null}
                {e.label ? (
                  <text x={(sx + ex) / 2 + 26} y={(sy + ey) / 2 + 10} fontSize={28} fontWeight={600} fill={tokens.text.secondary}
                    fontFamily={tokens.font.family} opacity={p}>{e.label}</text>
                ) : null}
              </g>
            );
          })}
          {nodes.map((n, i) => {
            const p = ease(frame, nodeAt(i), nodeAt(i) + 14);
            const x = colX[n.col], y = rowY(n.row);
            return (
              <g key={n.id} opacity={p} transform={`translate(0 ${(1 - p) * 20})`}>
                <rect x={x - NODE_W / 2} y={y - NODE_H / 2} width={NODE_W} height={NODE_H} rx={22} fill={tokens.bg.panel} stroke={a.main} strokeWidth={3} />
                <text x={x} y={n.detail ? y - 8 : y + 12} textAnchor="middle" fontSize={38} fontWeight={800} fill={tokens.text.primary}
                  fontFamily={tokens.font.family}>{n.label}</text>
                {n.detail ? (
                  <text x={x} y={y + 28} textAnchor="middle" fontSize={24} fill={tokens.text.secondary} fontFamily={tokens.font.family}>{n.detail}</text>
                ) : null}
              </g>
            );
          })}
        </svg>
      </AbsoluteFill>
    </Backdrop>
  );
};
