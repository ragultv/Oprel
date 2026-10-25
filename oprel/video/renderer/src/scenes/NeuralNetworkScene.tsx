import React from "react";
import { AbsoluteFill, random, useCurrentFrame, useVideoConfig } from "remotion";
import { Backdrop } from "./parts";
import { Header } from "./Header";
import { clamp01, ease } from "../lib/motion";
import { tokens } from "../lib/design";
import type { Scene } from "../schema";

const X0 = 170, X1 = 910, Y0 = 640, Y1 = 1320;

export const NeuralNetworkScene: React.FC<{ scene: Extract<Scene, { type: "neural-network" }> }> = ({ scene }) => {
  const frame = useCurrentFrame();
  const { fps, width, height } = useVideoConfig();
  const { layers, layerLabels, accent, headline, subtitle } = scene.params;
  const a = tokens.accent[accent];

  const pos = layers.map((n, li) => {
    const x = X0 + ((X1 - X0) * li) / (layers.length - 1);
    return Array.from({ length: n }, (_, ni) => ({ x, y: n === 1 ? (Y0 + Y1) / 2 : Y0 + ((Y1 - Y0) * ni) / (n - 1) }));
  });

  // Layers appear one after another; a pulse then travels along a seeded path through the layers.
  const appear = (li: number) => ease(frame, 12 + li * 8, 28 + li * 8);
  const pulseStart = 12 + layers.length * 8 + 14;
  const period = fps * 2.2;
  const cycle = Math.max(0, Math.floor((frame - pulseStart) / period));
  const t = ((frame - pulseStart) - cycle * period) / (fps * 1.6);
  const pathNodes = layers.map((n, li) => Math.floor(random(`${scene.id}-p${cycle}-${li}`) * n));
  const pos01 = clamp01(t) * (layers.length - 1);
  const hot = frame >= pulseStart ? Math.round(pos01) : -1;

  return (
    <Backdrop accent={accent}>
      <Header headline={headline} subtitle={subtitle} accent={accent} />
      <AbsoluteFill>
        <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`}>
          {pos.slice(0, -1).map((layer, li) =>
            layer.flatMap((p, ni) =>
              pos[li + 1].map((q, mi) => {
                const lit = frame >= pulseStart && pathNodes[li] === ni && pathNodes[li + 1] === mi && pos01 >= li;
                return (
                  <line key={`${li}-${ni}-${mi}`} x1={p.x} y1={p.y} x2={q.x} y2={q.y}
                    stroke={lit ? a.main : tokens.text.secondary} strokeWidth={lit ? 4 : 1.5}
                    opacity={Math.min(appear(li), appear(li + 1)) * (lit ? 1 : 0.22)} />
                );
              }),
            ),
          )}
          {pos.map((layer, li) =>
            layer.map((p, ni) => {
              const active = hot === li && pathNodes[li] === ni;
              return (
                <circle key={`${li}-${ni}`} cx={p.x} cy={p.y} r={active ? 26 : 20} fill={active ? a.main : tokens.bg.panel}
                  stroke={a.main} strokeWidth={3} opacity={appear(li)} style={active ? { filter: `drop-shadow(0 0 14px ${a.glow})` } : undefined} />
              );
            }),
          )}
          {layerLabels.map((lbl, li) =>
            pos[li] ? (
              <text key={li} x={pos[li][0].x} y={Y1 + 90} textAnchor="middle" fill={tokens.text.secondary} fontSize={32} fontWeight={600}
                fontFamily={tokens.font.family} opacity={appear(li)}>{lbl}</text>
            ) : null,
          )}
        </svg>
      </AbsoluteFill>
    </Backdrop>
  );
};
