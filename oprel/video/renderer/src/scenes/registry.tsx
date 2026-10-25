import React from "react";
import type { CompiledScene } from "../compiler";
import { HookScene } from "./HookScene";
import { HeroObjectScene } from "./HeroObjectScene";
import { ConclusionScene } from "./ConclusionScene";
import { NeuralNetworkScene } from "./NeuralNetworkScene";
import { TokenFlowScene } from "./TokenFlowScene";
import { ArchitectureDiagramScene } from "./ArchitectureDiagramScene";
import { ProbabilityChartScene } from "./ProbabilityChartScene";
import { CustomScene } from "./custom";
import { IMPLEMENTED_SCENE_TYPES } from "../sceneTypes";

type Renderer = (c: CompiledScene) => React.ReactElement;

/** Maps validated scene types to components. Keys must match IMPLEMENTED_SCENE_TYPES (checked by tests). */
export const sceneRegistry: Record<(typeof IMPLEMENTED_SCENE_TYPES)[number], Renderer> = {
  hook: (c) => <HookScene scene={c.scene as any} />,
  "hero-object": (c) => <HeroObjectScene scene={c.scene as any} durationInFrames={c.durationInFrames} />,
  conclusion: (c) => <ConclusionScene scene={c.scene as any} />,
  "neural-network": (c) => <NeuralNetworkScene scene={c.scene as any} />,
  "token-flow": (c) => <TokenFlowScene scene={c.scene as any} />,
  "architecture-diagram": (c) => <ArchitectureDiagramScene scene={c.scene as any} />,
  "probability-chart": (c) => <ProbabilityChartScene scene={c.scene as any} />,
  custom: (c) => <CustomScene compiled={c} />,
};

export function renderScene(c: CompiledScene): React.ReactElement {
  const r = (sceneRegistry as Record<string, Renderer | undefined>)[c.type];
  if (!r) throw new Error(`No component registered for scene type "${c.type}"`);
  return r(c);
}
