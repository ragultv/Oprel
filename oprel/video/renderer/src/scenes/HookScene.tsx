import React from "react";
import { Backdrop, TitleBlock } from "./parts";
import type { Scene } from "../schema";

export const HookScene: React.FC<{ scene: Extract<Scene, { type: "hook" }> }> = ({ scene }) => (
  <Backdrop accent={scene.params.accent}>
    <TitleBlock headline={scene.params.headline} subtitle={scene.params.subtitle} accent={scene.params.accent} position="center" />
  </Backdrop>
);
