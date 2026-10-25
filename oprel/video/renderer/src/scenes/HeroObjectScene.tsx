import React from "react";
import { AbsoluteFill, useCurrentFrame, useVideoConfig } from "remotion";
import { ThreeCanvas } from "@remotion/three";
import { Backdrop, TitleBlock } from "./parts";
import { heroPose, seededPoints } from "../lib/motion";
import { tokens } from "../lib/design";
import type { Scene } from "../schema";

type Props = { scene: Extract<Scene, { type: "hero-object" }>; durationInFrames: number };

const Geometry: React.FC<{ shape: Props["scene"]["params"]["shape"] }> = ({ shape }) => {
  switch (shape) {
    case "torus-knot":
      return <torusKnotGeometry args={[0.8, 0.28, 160, 24]} />;
    case "sphere":
      return <sphereGeometry args={[1.1, 48, 48]} />;
    case "box":
      return <boxGeometry args={[1.5, 1.5, 1.5]} />;
    default:
      return <icosahedronGeometry args={[1.2, 1]} />;
  }
};

/**
 * One focal object. Every transform is computed from the current frame via heroPose();
 * no useFrame(), no clocks, no simulation state. Scrubbing to any frame is stable.
 */
export const HeroObjectScene: React.FC<Props> = ({ scene, durationInFrames }) => {
  const frame = useCurrentFrame();
  const { fps, width, height } = useVideoConfig();
  const { shape, headline, subtitle, accent } = scene.params;
  const color = tokens.accent[accent];
  const pose = heroPose(frame, durationInFrames, fps);
  const dust = React.useMemo(() => seededPoints(scene.id, 36, 2.6), [scene.id]);

  return (
    <Backdrop accent={accent}>
      <AbsoluteFill>
        <ThreeCanvas width={width} height={height} camera={{ position: pose.camera, fov: 40 }} style={{ background: "transparent" }}>
          <ambientLight intensity={0.35} />
          <directionalLight position={[3, 4, 5]} intensity={1.6} color={color.main} />
          <pointLight position={[-4, -2, 2]} intensity={25} color="#6d28d9" />
          <group scale={pose.scale} rotation={pose.rotation} position={[0, 1.5, 0]}>
            <mesh>
              <Geometry shape={shape} />
              <meshStandardMaterial color={tokens.bg.panel} metalness={0.7} roughness={0.25} emissive={color.soft} emissiveIntensity={pose.glow * 0.5} />
            </mesh>
            <mesh>
              <Geometry shape={shape} />
              <meshBasicMaterial color={color.main} wireframe transparent opacity={0.55} />
            </mesh>
          </group>
          {dust.map((p, i) => (
            <mesh key={i} position={[p[0], p[1] + 1.5, p[2] - 2]}>
              <sphereGeometry args={[0.014, 6, 6]} />
              <meshBasicMaterial color={color.main} transparent opacity={0.28} />
            </mesh>
          ))}
        </ThreeCanvas>
      </AbsoluteFill>
      <TitleBlock headline={headline} subtitle={subtitle} accent={accent} position="lower" delay={Math.round(fps * 0.6)} />
    </Backdrop>
  );
};
