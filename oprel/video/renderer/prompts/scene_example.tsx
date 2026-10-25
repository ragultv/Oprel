import React from "react";
import { AbsoluteFill, interpolate, spring, useCurrentFrame, useVideoConfig, Easing, random } from "remotion";

import { tokens } from "@oprel/kit";

export interface SceneProps {
  durationInFrames: number;
  fps: number;
  width: number;
  height: number;
  narration: string;
  title: string;
  props?: Record<string, any>;
}

export default function Scene({ durationInFrames, narration, title }: SceneProps) {
  const frame = useCurrentFrame();
  const { fps, width, height } = useVideoConfig();

  // Color tokens for this motion graphic
  const accent = tokens.accent.cyan;
  const primary = "#38bdf8"; // Electric cyan
  const secondary = "#a855f7"; // Royal violet
  const bg = "#07090e";

  // 1. Kinetic headline entrance (staggered spring + letter spacing)
  const titleSpring = spring({ frame: frame - 5, fps, config: { damping: 16, mass: 0.6 } });
  const titleY = interpolate(titleSpring, [0, 1], [60, 0]);
  const titleTracking = interpolate(titleSpring, [0, 1], [-2, 1]);

  // 2. Central Radar / Neural Core animation
  const coreScale = spring({ frame: frame - 15, fps, config: { damping: 14, mass: 0.8 } });
  const coreRotate = interpolate(frame, [0, 180], [0, 360]);
  const pulseGlow = Math.sin(frame / 12) * 12 + 28;

  // 3. Dynamic metric counter (0% -> 98.4%)
  const metricProgress = interpolate(frame, [25, 65], [0, 98.4], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });

  // 4. Data nodes for the network visualization
  const nodeCount = 5;
  const radius = 280;
  const centerX = width / 2;
  const centerY = 820;

  return (
    <AbsoluteFill style={{ backgroundColor: bg, overflow: "hidden", fontFamily: "system-ui, -apple-system, sans-serif" }}>
      {/* Background ambient radial gradients */}
      <div
        style={{
          position: "absolute",
          top: centerY - 450,
          left: centerX - 450,
          width: 900,
          height: 900,
          borderRadius: "50%",
          background: `radial-gradient(circle, ${secondary}15 0%, ${primary}10 40%, transparent 70%)`,
          filter: "blur(60px)",
          pointerEvents: "none",
        }}
      />

      {/* Floating ambient particles (deterministic seeded motion) */}
      {Array.from({ length: 18 }).map((_, i) => {
        const seedX = random(`px-${i}`);
        const seedY = random(`py-${i}`);
        const speed = 0.4 + random(`ps-${i}`) * 0.8;
        const pY = ((seedY * height + frame * speed) % (height - 300)) + 150;
        const pX = seedX * width;
        const opacity = Math.sin((frame + i * 20) / 15) * 0.3 + 0.4;
        return (
          <div
            key={i}
            style={{
              position: "absolute",
              left: pX,
              top: pY,
              width: 4 + (i % 3) * 2,
              height: 4 + (i % 3) * 2,
              borderRadius: "50%",
              backgroundColor: i % 2 === 0 ? primary : secondary,
              opacity,
              boxShadow: `0 0 10px ${primary}`,
            }}
          />
        );
      })}

      {/* Kinetic Headline Section */}
      <div
        style={{
          position: "absolute",
          top: 180,
          left: 72,
          right: 72,
          opacity: titleSpring,
          translate: `0px ${titleY}px`,
        }}
      >
        <div
          style={{
            display: "inline-flex",
            alignItems: "center",
            gap: 10,
            padding: "8px 20px",
            borderRadius: 9999,
            backgroundColor: `${primary}18`,
            border: `1px solid ${primary}40`,
            marginBottom: 20,
          }}
        >
          <div style={{ width: 8, height: 8, borderRadius: "50%", backgroundColor: primary, boxShadow: `0 0 8px ${primary}` }} />
          <span style={{ fontSize: 24, fontWeight: 700, color: primary, letterSpacing: "0.15em", textTransform: "uppercase" }}>
            Neural Core
          </span>
        </div>

        <h1
          style={{
            fontSize: 84,
            fontWeight: 800,
            color: "#f8fafc",
            lineHeight: 1.1,
            letterSpacing: `${titleTracking}px`,
            margin: 0,
            textShadow: `0 10px 40px rgba(0,0,0,0.8)`,
          }}
        >
          {title || "Dynamic Attention"}
        </h1>
      </div>

      {/* Center Motion Graphic: SVG Animated Vector Arcs & Nodes */}
      <svg
        style={{
          position: "absolute",
          left: centerX - 420,
          top: centerY - 420,
          width: 840,
          height: 840,
        }}
        viewBox="0 0 840 840"
      >
        {/* Outer Orbit Gauge */}
        <circle
          cx={420}
          cy={420}
          r={radius + 40}
          fill="none"
          stroke={`${primary}25`}
          strokeWidth={2}
          strokeDasharray="6 14"
        />

        {/* Dynamic Rotating HUD Arc */}
        <circle
          cx={420}
          cy={420}
          r={radius + 40}
          fill="none"
          stroke={primary}
          strokeWidth={4}
          strokeDasharray="180 800"
          strokeDashoffset={-frame * 3}
          strokeLinecap="round"
        />

        {/* Vector Connection Lines to Satellites */}
        {Array.from({ length: nodeCount }).map((_, i) => {
          const angle = (i * 2 * Math.PI) / nodeCount;
          const nodeX = 420 + Math.cos(angle) * radius;
          const nodeY = 420 + Math.sin(angle) * radius;
          const lineLength = radius;
          const drawProgress = interpolate(frame, [18 + i * 8, 42 + i * 8], [lineLength, 0], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
            easing: Easing.bezier(0.16, 1, 0.3, 1),
          });

          return (
            <g key={i}>
              <line
                x1={420}
                y1={420}
                x2={nodeX}
                y2={nodeY}
                stroke={`url(#lineGrad)`}
                strokeWidth={3}
                strokeDasharray={lineLength}
                strokeDashoffset={drawProgress}
                opacity={0.8}
              />
              {/* Satellite Node */}
              <circle
                cx={nodeX}
                cy={nodeY}
                r={16}
                fill={secondary}
                style={{
                  scale: interpolate(frame, [25 + i * 8, 45 + i * 8], [0, 1], {
                    extrapolateLeft: "clamp",
                    extrapolateRight: "clamp",
                    output: "perceptual-scale",
                  }),
                  transformOrigin: `${nodeX}px ${nodeY}px`,
                }}
              />
              <circle cx={nodeX} cy={nodeY} r={8} fill="#ffffff" />
            </g>
          );
        })}

        <defs>
          <linearGradient id="lineGrad" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor={primary} />
            <stop offset="100%" stopColor={secondary} />
          </linearGradient>
        </defs>
      </svg>

      {/* Central Pulsing Holographic Core with Metric Readout */}
      <div
        style={{
          position: "absolute",
          left: centerX - 140,
          top: centerY - 140,
          width: 280,
          height: 280,
          borderRadius: "50%",
          background: `radial-gradient(circle, #1e293b 0%, #0f172a 100%)`,
          border: `3px solid ${primary}`,
          boxShadow: `0 0 ${pulseGlow}px ${primary}88, inset 0 0 40px ${primary}44`,
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          scale: coreScale,
        }}
      >
        <span style={{ fontSize: 20, fontWeight: 700, color: primary, textTransform: "uppercase", letterSpacing: "0.1em" }}>
          Accuracy
        </span>
        <div style={{ fontSize: 64, fontWeight: 900, color: "#ffffff", letterSpacing: "-0.03em" }}>
          {metricProgress.toFixed(1)}%
        </div>
      </div>

      {/* Bottom Floating Telemetry Metric Badges */}
      <div
        style={{
          position: "absolute",
          bottom: 240,
          left: 72,
          right: 72,
          display: "flex",
          justifyContent: "space-between",
          gap: 20,
        }}
      >
        {[
          { label: "Vector Space", val: "768-D" },
          { label: "Latency", val: "12ms" },
          { label: "Attention Heads", val: "32" },
        ].map((item, idx) => {
          const badgeSpring = spring({ frame: frame - (40 + idx * 8), fps, config: { damping: 18 } });
          return (
            <div
              key={item.label}
              style={{
                flex: 1,
                padding: "20px 24px",
                borderRadius: 20,
                backgroundColor: "rgba(15, 23, 42, 0.75)",
                border: "1px solid rgba(56, 189, 248, 0.25)",
                backdropFilter: "blur(12px)",
                opacity: badgeSpring,
                translate: `0px ${interpolate(badgeSpring, [0, 1], [40, 0])}px`,
              }}
            >
              <div style={{ fontSize: 20, color: "#64748b", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.08em" }}>
                {item.label}
              </div>
              <div style={{ fontSize: 44, color: "#f8fafc", fontWeight: 800, marginTop: 6 }}>
                {item.val}
              </div>
            </div>
          );
        })}
      </div>
    </AbsoluteFill>
  );
}
