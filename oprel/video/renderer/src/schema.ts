import { z } from "zod";

/** Current storyboard schema version. Bump + add a migration in migrate.ts on breaking changes. */
export const STORYBOARD_VERSION = 1 as const;

export const AccentSchema = z.enum(["cyan", "blue", "violet"]);
export type Accent = z.infer<typeof AccentSchema>;

/** Project-relative asset path. No absolute paths, no "..", no backslashes, no URL schemes. */
export const AssetPathSchema = z
  .string()
  .min(1)
  .max(260)
  .refine((p) => !/^[a-zA-Z][a-zA-Z0-9+.-]*:/.test(p), "URL schemes / drive letters are not allowed")
  .refine((p) => !p.startsWith("/") && !p.startsWith("\\"), "absolute paths are not allowed")
  .refine((p) => !p.includes("\\"), "use forward slashes")
  .refine((p) => !p.split("/").some((s) => s === ".." || s === "." || s === ""), "path traversal is not allowed");

export const TransitionSchema = z
  .object({
    type: z.enum(["none", "fade"]).default("none"),
    /** Frames at the start of the scene spent fading in from the background. Taken from the scene's own duration. */
    durationFrames: z.number().int().min(0).max(90).default(0),
  })
  .default({ type: "none", durationFrames: 0 });

const sceneBase = {
  id: z.string().regex(/^[a-z0-9][a-z0-9-_]{0,63}$/i, "scene id must be alphanumeric, '-' or '_'"),
  durationSeconds: z.number().positive().max(120).optional(),
  durationFrames: z.number().int().positive().max(120 * 120).optional(),
  /** Text spoken over this scene (fed to Audio Studio TTS). Scene length is later fitted to the real audio. */
  narration: z.string().max(700).optional(),
  narrationRef: z.string().optional(),
  transition: TransitionSchema,
  assets: z.array(AssetPathSchema).default([]),
};

export const HookSceneSchema = z.object({
  ...sceneBase,
  type: z.literal("hook"),
  params: z.object({
    headline: z.string().min(1).max(120),
    subtitle: z.string().max(200).optional(),
    accent: AccentSchema.default("cyan"),
  }),
});

export const HeroObjectSceneSchema = z.object({
  ...sceneBase,
  type: z.literal("hero-object"),
  params: z.object({
    shape: z.enum(["icosahedron", "torus-knot", "sphere", "box"]).default("icosahedron"),
    headline: z.string().min(1).max(120),
    subtitle: z.string().max(200).optional(),
    accent: AccentSchema.default("cyan"),
  }),
});

export const ConclusionSceneSchema = z.object({
  ...sceneBase,
  type: z.literal("conclusion"),
  params: z.object({
    title: z.string().min(1).max(120),
    points: z.array(z.string().min(1).max(120)).max(4).default([]),
    accent: AccentSchema.default("violet"),
  }),
});

/** A scene whose React component was written by the AI agent (or the user) and lives in the project as scenes/<name>.tsx. */
export const CustomSceneSchema = z.object({
  ...sceneBase,
  type: z.literal("custom"),
  params: z.object({
    file: AssetPathSchema.refine((p) => /^scenes\/[a-z0-9][a-z0-9_-]{0,63}\.tsx$/i.test(p), "custom scene files must be scenes/<name>.tsx"),
    title: z.string().max(120).optional(),
    props: z.record(z.string().max(40), z.union([z.string().max(300), z.number(), z.boolean()])).default({}),
  }),
});

export const NeuralNetworkSceneSchema = z.object({
  ...sceneBase,
  type: z.literal("neural-network"),
  params: z.object({
    headline: z.string().min(1).max(120),
    subtitle: z.string().max(200).optional(),
    /** Nodes per layer, e.g. [3,5,5,2]. */
    layers: z.array(z.number().int().min(1).max(8)).min(2).max(6).default([3, 5, 5, 2]),
    layerLabels: z.array(z.string().max(24)).max(6).default([]),
    accent: AccentSchema.default("blue"),
  }),
});

export const TokenFlowSceneSchema = z.object({
  ...sceneBase,
  type: z.literal("token-flow"),
  params: z.object({
    headline: z.string().min(1).max(120),
    subtitle: z.string().max(200).optional(),
    inputTokens: z.array(z.string().min(1).max(14)).min(1).max(8),
    outputTokens: z.array(z.string().min(1).max(14)).min(1).max(8),
    stageLabel: z.string().min(1).max(24).default("Model"),
    accent: AccentSchema.default("cyan"),
  }),
});

export const ArchitectureDiagramSceneSchema = z.object({
  ...sceneBase,
  type: z.literal("architecture-diagram"),
  params: z
    .object({
      headline: z.string().min(1).max(120),
      subtitle: z.string().max(200).optional(),
      /** Grid placement: col 0..2, row 0..5. Nodes reveal in array order. */
      nodes: z
        .array(
          z.object({
            id: z.string().regex(/^[a-z0-9_-]{1,32}$/i),
            label: z.string().min(1).max(28),
            detail: z.string().max(40).optional(),
            col: z.number().int().min(0).max(2),
            row: z.number().int().min(0).max(5),
          }),
        )
        .min(2)
        .max(8),
      edges: z.array(z.object({ from: z.string(), to: z.string(), label: z.string().max(20).optional() })).max(12),
      accent: AccentSchema.default("cyan"),
    })
    .superRefine((p, ctx) => {
      const ids = new Set(p.nodes.map((n) => n.id));
      if (ids.size !== p.nodes.length) ctx.addIssue({ code: "custom", message: "duplicate node id" });
      const cells = new Set<string>();
      p.nodes.forEach((n, i) => {
        const k = `${n.col},${n.row}`;
        if (cells.has(k)) ctx.addIssue({ code: "custom", path: ["nodes", i], message: `two nodes share cell ${k}` });
        cells.add(k);
      });
      p.edges.forEach((e, i) => {
        if (!ids.has(e.from) || !ids.has(e.to)) ctx.addIssue({ code: "custom", path: ["edges", i], message: "edge references unknown node" });
      });
    }),
});

export const ProbabilityChartSceneSchema = z.object({
  ...sceneBase,
  type: z.literal("probability-chart"),
  params: z
    .object({
      headline: z.string().min(1).max(120),
      subtitle: z.string().max(200).optional(),
      bars: z.array(z.object({ label: z.string().min(1).max(20), value: z.number().min(0).max(1) })).min(2).max(6),
      /** Illustrative values are labelled as such on screen. "measured" requires a source. */
      dataKind: z.enum(["illustrative", "measured"]).default("illustrative"),
      source: z.string().max(120).optional(),
      accent: AccentSchema.default("violet"),
    })
    .refine((p) => p.dataKind !== "measured" || !!p.source, { message: "measured data requires a source", path: ["source"] }),
});

/**
 * Scene types declared here are the ones with a schema AND a component.
 */
export const SceneSchema = z.discriminatedUnion("type", [
  HookSceneSchema,
  HeroObjectSceneSchema,
  ConclusionSceneSchema,
  NeuralNetworkSceneSchema,
  TokenFlowSceneSchema,
  ArchitectureDiagramSceneSchema,
  ProbabilityChartSceneSchema,
  CustomSceneSchema,
]);
export type Scene = z.infer<typeof SceneSchema>;
export type SceneType = Scene["type"];

export const CaptionSchema = z.object({
  startMs: z.number().int().min(0),
  endMs: z.number().int().min(1),
  text: z.string().min(1).max(300),
  /** "sentence" or "word": whatever the aligner really produced. Never fabricated. */
  granularity: z.enum(["sentence", "word"]).default("sentence"),
});
export type Caption = z.infer<typeof CaptionSchema>;

export const NarrationSchema = z.object({
  file: AssetPathSchema,
  /** Real, probed duration of the file. Authoritative if present. */
  durationMs: z.number().int().positive().optional(),
  volume: z.number().min(0).max(1).default(1),
});

export const FormatSchema = z
  .object({
    preset: z.enum(["shorts", "landscape"]).default("shorts"),
    fps: z.number().int().min(1).max(60).default(30),
  })
  .default({ preset: "shorts", fps: 30 });

export const FORMAT_DIMENSIONS = {
  shorts: { width: 1080, height: 1920 },
  landscape: { width: 1920, height: 1080 },
} as const;

export const StoryboardSchema = z
  .object({
    schemaVersion: z.literal(STORYBOARD_VERSION),
    title: z.string().min(1).max(200),
    format: FormatSchema,
    scenes: z.array(SceneSchema).min(1).max(60),
    narration: NarrationSchema.optional(),
    captions: z.array(CaptionSchema).default([]),
    /** How caption times were obtained. Only "estimated-from-audio-duration" is produced today (no aligner is available). */
    captionsSource: z.enum(["manual", "estimated-from-audio-duration"]).optional(),
  })
  .superRefine((sb, ctx) => {
    const seen = new Set<string>();
    sb.scenes.forEach((s, i) => {
      if (seen.has(s.id)) ctx.addIssue({ code: "custom", path: ["scenes", i, "id"], message: `duplicate scene id "${s.id}"` });
      seen.add(s.id);
      const has = (s.durationSeconds !== undefined ? 1 : 0) + (s.durationFrames !== undefined ? 1 : 0);
      if (has !== 1)
        ctx.addIssue({ code: "custom", path: ["scenes", i], message: "set exactly one of durationSeconds / durationFrames" });
    });
  });

export type Storyboard = z.infer<typeof StoryboardSchema>;
export type StoryboardInput = z.input<typeof StoryboardSchema>;
