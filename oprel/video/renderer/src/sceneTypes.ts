/** Scene types that have a component registered in scenes/registry.tsx. Kept separate so the compiler stays React-free. */
export const IMPLEMENTED_SCENE_TYPES = ["hook", "hero-object", "conclusion", "neural-network", "token-flow", "architecture-diagram", "probability-chart", "custom"] as const;
export const isSceneImplemented = (t: string) => (IMPLEMENTED_SCENE_TYPES as readonly string[]).includes(t);
