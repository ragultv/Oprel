import React from "react";
import { useVideoConfig } from "remotion";
import type { CompiledScene } from "../compiler";
import type { SceneProps } from "../kit";

type SceneComponent = React.ComponentType<SceneProps>;
const registry = new Map<string, SceneComponent>();

/** Called by the generated bundle entry; the key is the project-relative file path (e.g. "scenes/intro.tsx"). */
export function registerCustomScenes(scenes: Record<string, SceneComponent>) {
  for (const [file, component] of Object.entries(scenes)) registry.set(file, component);
}

export const CustomScene: React.FC<{ compiled: CompiledScene }> = ({ compiled }) => {
  const { fps, width, height } = useVideoConfig();
  const scene = compiled.scene as Extract<CompiledScene["scene"], { type: "custom" }>;
  const Component = registry.get(scene.params.file);
  if (!Component) throw new Error(`custom scene "${scene.params.file}" was not bundled`);
  return (
    <Component
      durationInFrames={compiled.durationInFrames}
      fps={fps}
      width={width}
      height={height}
      narration={scene.narration ?? ""}
      title={scene.params.title ?? ""}
      props={scene.params.props}
    />
  );
};
