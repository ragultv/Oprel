"use client"

import { useEffect, useMemo, useState } from "react"
import Link from "next/link"
import {
  AlertTriangle,
  Check,
  ChevronDown,
  Cloud,
  Cpu,
  ExternalLink,
  FileCode2,
  Film,
  Layers,
  Loader2,
  Play,
  ShieldCheck,
  Sliders,
  Sparkles,
  Zap,
} from "lucide-react"
import type { AIModel } from "@/services/data"
import type { AgentStartBody } from "@/services/api"
import { PROVIDER_PRESETS, type ProviderType } from "@/services/providers"
import { cn } from "@/services/utils"
import { card, inputCls, labelCls, primaryBtn } from "./shared"

const LENGTHS = [30, 45, 60, 90]
const BUDGETS = [1, 2, 4, 8, 12]

interface MotionStyle {
  id: string
  name: string
  icon: string
  description: string
  remotionDirective: string
}

const MOTION_STYLES: MotionStyle[] = [
  {
    id: "kinetic-type",
    name: "Kinetic Typography",
    icon: "⚡",
    description: "Fast staggered letter tracking, bold punchy titles, dynamic word highlights",
    remotionDirective: "Focus on kinetic typography: staggered spring letter reveals, dynamic tracking/letter-spacing interpolations, fluid scale jumps with perceptual-scale, and high-contrast bold titles.",
  },
  {
    id: "data-viz",
    name: "Animated Data & Gauges",
    icon: "📊",
    description: "Radial telemetry gauges, animated SVG line/bar charts, live counting numbers",
    remotionDirective: "Focus on animated data visualization: circular progress gauges with strokeDashoffset, dynamic SVG bezier curves, and live numeric metric counters interpolating with Easing.bezier.",
  },
  {
    id: "node-network",
    name: "Neural Node Network",
    icon: "🌐",
    description: "Pulsing holographic central core, interconnected satellite nodes, glowing paths",
    remotionDirective: "Focus on network & node motion graphics: a pulsating central core with glowing box-shadows, satellite nodes with connecting SVG vectors drawing smoothly, and floating ambient particles.",
  },
  {
    id: "wireframe-3d",
    name: "3D Geometric Mesh",
    icon: "🧊",
    description: "Three.js rotating 3D geometry, wireframe meshes, isometric visual depth",
    remotionDirective: "Focus on 3D spatial motion design: integrate @remotion/three ThreeCanvas with rotating wireframe geometries, dynamic ambient lighting, and floating layered planes.",
  },
  {
    id: "vector-flow",
    name: "Fluid Vector Flow",
    icon: "✨",
    description: "Morphing SVG shapes, glowing gradient arcs, sleek organic vector reveals",
    remotionDirective: "Focus on fluid vector motion graphics: morphing SVG path elements, glowing gradient stroke trails, and seamless staggered reveals using spring physics.",
  },
]

interface ColorAtmosphere {
  id: string
  name: string
  primary: string
  secondary: string
  bg: string
}

const COLOR_ATMOSPHERES: ColorAtmosphere[] = [
  { id: "amber-obsidian", name: "Claude Obsidian", primary: "#f59e0b", secondary: "#d97706", bg: "#101012" },
  { id: "cyber-cyan", name: "Electric Cyan", primary: "#38bdf8", secondary: "#818cf8", bg: "#07090e" },
  { id: "emerald-matrix", name: "Emerald Tech", primary: "#10b981", secondary: "#06b6d4", bg: "#06090a" },
  { id: "royal-violet", name: "Royal Violet", primary: "#a855f7", secondary: "#ec4899", bg: "#090611" },
  { id: "monochrome", name: "Minimalist Slate", primary: "#f8fafc", secondary: "#94a3b8", bg: "#0b0f17" },
]

const SCRIPT_TEMPLATE = `Title: Transformer Attention Engine
Logline: How neural networks process language tokens simultaneously.

Scene 1 (6s):
Narration: How do modern language models process complex sentences with superhuman speed?
Visual: Glowing kinetic typography headlines appearing with staggered spring tracking and radiant highlights.

Scene 2 (8s):
Narration: Instead of reading one word after another, self-attention links every token in parallel.
Visual: Pulsing central neural core with animated laser vector lines connecting interrelated words.

Scene 3 (6s):
Narration: Attention weights compute exact contextual relevance, delivering instant understanding.
Visual: Dynamic radial telemetry gauge rotating and filling up to 98% with ambient particle fields.`

type Props = {
  models: AIModel[]
  activeModelId: string | null
  ttsModels: { id: string; name: string; voices: { id: string; name: string }[] }[]
  rendererDown: boolean
  rendererMessage?: string | null
  busy: boolean
  error: string | null
  initialBrief?: string
  onStart: (body: AgentStartBody) => void
}

export function Composer({
  models,
  activeModelId,
  ttsModels,
  rendererDown,
  rendererMessage,
  busy,
  error,
  initialBrief = "",
  onStart,
}: Props) {
  const [brief, setBrief] = useState(initialBrief)
  const [promptMode, setPromptMode] = useState<"concept" | "script">("concept")
  const [modelId, setModelId] = useState("")
  const [modelDropdown, setModelDropdown] = useState(false)
  const [seconds, setSeconds] = useState(45)
  const [tts, setTts] = useState("")
  const [voice, setVoice] = useState("")
  const [reviewPlan, setReviewPlan] = useState(false)
  const [autoRender, setAutoRender] = useState(true)
  const [attempts, setAttempts] = useState(4)
  const [hours, setHours] = useState(4)
  const [advanced, setAdvanced] = useState(false)

  // Motion graphics direction state
  const [selectedMotionStyleId, setSelectedMotionStyleId] = useState("kinetic-type")
  const [selectedAtmosphereId, setSelectedAtmosphereId] = useState("amber-obsidian")
  const [customArtDirection, setCustomArtDirection] = useState("")

  // Split models into cloud (external providers) and local (on-device)
  const cloudModels = useMemo(
    () => models.filter((m) => m.category === "external"),
    [models]
  )
  const localModels = useMemo(
    () => models.filter((m) => m.category !== "external" && (m.status === "loaded" || m.status === "available" || m.downloaded)),
    [models]
  )
  const allUsableModels = useMemo(() => [...cloudModels, ...localModels], [cloudModels, localModels])
  const isLocalModel = (id: string) => localModels.some((m) => m.id === id)

  // Auto-detect custom scene durations if user entered a structured script
  const customTotalSeconds = useMemo(() => {
    const matches = Array.from(brief.matchAll(/Scene\s*\d+\s*(?:\(\s*(\d+)\s*s(?:ecs?)?\s*\)|:\s*(\d+)\s*s)/gi))
    let total = 0
    for (const m of matches) {
      const s = parseInt(m[1] || m[2], 10)
      if (!isNaN(s)) total += s
    }
    return total
  }, [brief])

  // Auto-select preferred model: cloud first, then local fallback
  useEffect(() => {
    if (allUsableModels.length === 0) {
      setModelId("")
      return
    }
    if (modelId && allUsableModels.some((m) => m.id === modelId)) return

    const preferred =
      cloudModels.find((m) => m.id === activeModelId) ||
      cloudModels.find((m) => m.id.toLowerCase().includes("claude") || m.name.toLowerCase().includes("claude")) ||
      cloudModels.find((m) => m.id.toLowerCase().includes("gpt-4o") || m.name.toLowerCase().includes("gpt-4")) ||
      cloudModels.find((m) => m.id.toLowerCase().includes("gemini")) ||
      cloudModels[0] ||
      localModels.find((m) => m.status === "loaded") ||
      localModels[0]

    if (preferred) setModelId(preferred.id)
  }, [allUsableModels, cloudModels, localModels, activeModelId, modelId])

  useEffect(() => {
    if (!tts && ttsModels[0]) {
      setTts(ttsModels[0].id)
      setVoice(ttsModels[0].voices?.[0]?.id ?? "")
    }
  }, [ttsModels, tts])

  const selectedModel = allUsableModels.find((m) => m.id === modelId)
  const selectedMotionStyle = MOTION_STYLES.find((s) => s.id === selectedMotionStyleId) || MOTION_STYLES[0]
  const selectedAtmosphere = COLOR_ATMOSPHERES.find((a) => a.id === selectedAtmosphereId) || COLOR_ATMOSPHERES[0]
  const voices = ttsModels.find((t) => t.id === tts)?.voices ?? []

  const effectiveTargetSeconds = customTotalSeconds > 10 ? customTotalSeconds : seconds

  const ready =
    brief.trim().length >= 10 &&
    !!modelId &&
    allUsableModels.length > 0 &&
    !rendererDown &&
    !busy

  const start = () => {
    if (!ready) return

    // Compose rich brief containing the user's idea and Remotion Agent Skills directives
    const motionDirective = `\n\n[Remotion Agent Skills & Motion Graphics Directives]
Primary Motion Archetype: ${selectedMotionStyle.name} (${selectedMotionStyle.description})
Color Atmosphere: ${selectedAtmosphere.name} (Background: ${selectedAtmosphere.bg}, Primary Accent: ${selectedAtmosphere.primary}, Secondary: ${selectedAtmosphere.secondary})
Remotion Skills Guidance: ${selectedMotionStyle.remotionDirective}${
      customArtDirection.trim() ? `\nUser Specific Art Direction: ${customArtDirection.trim()}` : ""
    }
CRITICAL MOTION RULE: Do NOT generate boring web application wireframes, static cards, or rectangular UI forms with numbered boxes. Build real broadcast-quality motion graphics with dynamic SVG curves, animated telemetry readouts, kinetic typography, and staggered spring momentum using official Remotion Agent Skills (remotion-markup). Elements can use vertical canvas height naturally between y = 120 and y = 1780.`

    const combinedBrief = `${brief.trim()}${motionDirective}`
    const usingLocal = isLocalModel(modelId)

    onStart({
      brief: combinedBrief,
      model_id: modelId,
      allow_code: true,
      allow_cloud: !usingLocal,
      target_seconds: effectiveTargetSeconds,
      tts_model: tts,
      voice,
      review_plan: reviewPlan,
      auto_render: autoRender,
      max_attempts: attempts,
      time_budget_hours: hours,
    })
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
      e.preventDefault()
      start()
    }
  }

  return (
    <div className="mx-auto flex w-full max-w-4xl flex-col gap-6 px-4 py-8 sm:px-8">
      {/* Claude Code Header */}
      <div className="space-y-3 text-center">
        <div className="inline-flex items-center gap-2 rounded-full border border-amber-500/30 bg-amber-500/10 px-3.5 py-1 font-mono text-[11px] font-medium tracking-wide text-amber-400 backdrop-blur-sm">
          <Sparkles size={12} className="animate-pulse" />
          CLAUDE CODE VIDEO ENGINE · REMOTION AGENT SKILLS
        </div>
        <h1 className="text-3xl font-bold tracking-tight sm:text-4xl text-foreground font-sans">
          Autonomous Motion Graphics Studio
        </h1>
        <p className="mx-auto max-w-2xl text-sm text-muted-foreground leading-relaxed">
          Driven by official Remotion Agent Skills (<code className="text-primary font-mono text-xs">remotion-markup</code> &amp; <code className="text-primary font-mono text-xs">remotion-render</code>). Generates broadcast-quality kinetic motion design, animated SVG telemetry, and particle graphs without cookie-cutter UI cards.
        </p>
      </div>

      {/* Model availability banner */}
      {allUsableModels.length === 0 && (
        <div className="rounded-2xl border border-amber-500/30 bg-amber-500/10 p-5 text-sm text-amber-200 backdrop-blur-sm">
          <div className="flex items-start gap-3">
            <AlertTriangle className="h-5 w-5 text-amber-400 shrink-0 mt-0.5" />
            <div className="space-y-2">
              <p className="font-semibold text-foreground">No Director Model Available</p>
              <p className="text-xs text-muted-foreground leading-relaxed">
                Connect a cloud provider (Anthropic Claude, Gemini, OpenAI) <strong>or</strong> download a local model (e.g. Qwen2.5-7B, Mistral) for on-device generation.
              </p>
              <div className="flex flex-wrap gap-2">
                <Link
                  href="/settings"
                  className="inline-flex items-center gap-1.5 rounded-lg bg-amber-500/20 px-3 py-1.5 text-xs font-semibold text-amber-300 hover:bg-amber-500/30 transition-colors"
                >
                  Configure Cloud Provider <ExternalLink size={12} />
                </Link>
                <Link
                  href="/models"
                  className="inline-flex items-center gap-1.5 rounded-lg bg-sky-500/20 px-3 py-1.5 text-xs font-semibold text-sky-300 hover:bg-sky-500/30 transition-colors"
                >
                  Download Local Model <Cpu size={12} />
                </Link>
              </div>
            </div>
          </div>
        </div>
      )}
      {/* Local model — SSD KV-cache status badge */}
      {allUsableModels.length > 0 && localModels.length > 0 && isLocalModel(modelId) && (
        <div className="rounded-2xl border border-emerald-500/20 bg-emerald-500/8 p-4 text-xs text-emerald-200 backdrop-blur-sm">
          <div className="flex items-start gap-2.5">
            <Cpu className="h-4 w-4 text-emerald-400 shrink-0 mt-0.5" />
            <p className="leading-relaxed">
              <span className="font-semibold text-foreground">On-device · SSD KV-Cache active.</span>{" "}
              Oprel extends local models to a <span className="font-mono text-emerald-300 font-semibold">131k-token</span> context via native SSD KV-cache (Direct I/O + Quest sparse retrieval). Full multi-scene scripts run entirely on-device with no data leaving your machine.
            </p>
          </div>
        </div>
      )}

      {/* Main Composer Studio Card */}
      <div className={cn(card, "space-y-6 p-6 sm:p-7 border-border/80 shadow-2xl bg-[#0f1117]/95 backdrop-blur-md rounded-2xl")}>
        {/* Top Bar: Model Selector (Chat Page Style) & Engine Status */}
        <div className="flex flex-wrap items-center justify-between gap-4 border-b border-border/50 pb-5">
          <div className="flex items-center gap-2">
            {modelId && isLocalModel(modelId)
              ? <Cpu size={16} className="text-sky-400" />
              : <Cloud size={16} className="text-primary" />}
            <span className="text-xs font-bold uppercase tracking-wider text-muted-foreground">
              {modelId && isLocalModel(modelId) ? "Director Model (On-Device)" : "Director Model (Cloud Inference)"}
            </span>
          </div>

          <div className="relative">
            <button
              type="button"
              onClick={() => setModelDropdown((v) => !v)}
              className="flex items-center gap-2.5 px-3.5 py-2 rounded-xl border border-border/80 bg-secondary/40 text-xs font-semibold text-foreground hover:bg-secondary transition-all shadow-sm"
            >
              {selectedModel ? (
                <>
                  <span
                    className="w-2.5 h-2.5 rounded-full shrink-0"
                    style={{
                      backgroundColor:
                        PROVIDER_PRESETS[selectedModel.architecture as ProviderType]?.color || "#38bdf8",
                    }}
                  />
                  <span>{selectedModel.name}</span>
                  <span className="text-[10px] text-muted-foreground opacity-80 uppercase tracking-wider font-mono">
                    {selectedModel.family || "Cloud"}
                  </span>
                </>
              ) : (
                <span className="text-muted-foreground">Select Cloud Model</span>
              )}
              <ChevronDown size={14} className={cn("text-muted-foreground transition-transform", modelDropdown && "rotate-180")} />
            </button>

            {modelDropdown && (
              <>
                <div className="fixed inset-0 z-40" onClick={() => setModelDropdown(false)} />
                <div className="absolute right-0 top-11 w-80 bg-[#16181f] border border-border/80 rounded-2xl shadow-2xl p-2 z-50 animate-fade-in-up backdrop-blur-xl">

                  {/* Cloud models section */}
                  {cloudModels.length > 0 && (
                    <>
                      <div className="px-3 py-2 text-[10px] font-bold text-muted-foreground uppercase tracking-wider border-b border-border/40 flex items-center justify-between">
                        <span className="flex items-center gap-1.5"><Cloud size={10} />Cloud Providers</span>
                        <span className="text-emerald-400 font-mono">128k+ Context</span>
                      </div>
                      <div className="py-1">
                        {cloudModels.map((m) => {
                          const isCurrent = m.id === modelId
                          const preset = PROVIDER_PRESETS[m.architecture as ProviderType]
                          const dotColor = preset?.color || "#38bdf8"
                          return (
                            <button
                              key={m.id}
                              type="button"
                              onClick={() => { setModelId(m.id); setModelDropdown(false) }}
                              className={cn(
                                "w-full text-left px-3 py-2.5 rounded-xl text-xs transition-all flex items-center justify-between gap-2 my-0.5",
                                isCurrent
                                  ? "bg-primary/15 text-primary font-bold border border-primary/25"
                                  : "text-muted-foreground hover:bg-secondary/60 hover:text-foreground"
                              )}
                            >
                              <div className="min-w-0 flex items-center gap-2">
                                <span className="w-2 h-2 rounded-full shrink-0" style={{ backgroundColor: dotColor }} />
                                <div className="truncate">
                                  <div className="font-medium text-foreground truncate">{m.name}</div>
                                  <div className="text-[10px] opacity-60 font-mono">{m.family || preset?.name}</div>
                                </div>
                              </div>
                              {isCurrent && <Check size={14} className="text-primary shrink-0" />}
                            </button>
                          )
                        })}
                      </div>
                    </>
                  )}

                  {/* Local models section */}
                  {localModels.length > 0 && (
                    <>
                      <div className={cn(
                        "px-3 py-2 text-[10px] font-bold text-muted-foreground uppercase tracking-wider flex items-center justify-between",
                        cloudModels.length > 0 ? "border-t border-border/40 mt-1" : "border-b border-border/40"
                      )}>
                        <span className="flex items-center gap-1.5"><Cpu size={10} />On-Device (Local)</span>
                        <span className="text-emerald-400 font-mono">131k SSD</span>
                      </div>
                      <div className="py-1">
                        {localModels.map((m) => {
                          const isCurrent = m.id === modelId
                          const isLoaded = m.status === "loaded"
                          return (
                            <button
                              key={m.id}
                              type="button"
                              onClick={() => { setModelId(m.id); setModelDropdown(false) }}
                              className={cn(
                                "w-full text-left px-3 py-2.5 rounded-xl text-xs transition-all flex items-center justify-between gap-2 my-0.5",
                                isCurrent
                                  ? "bg-sky-500/15 text-sky-300 font-bold border border-sky-500/25"
                                  : "text-muted-foreground hover:bg-secondary/60 hover:text-foreground"
                              )}
                            >
                              <div className="min-w-0 flex items-center gap-2">
                                <span className={cn("w-2 h-2 rounded-full shrink-0", isLoaded ? "bg-emerald-400" : "bg-zinc-500")} />
                                <div className="truncate">
                                  <div className="font-medium text-foreground truncate">{m.name}</div>
                                  <div className="text-[10px] opacity-60 font-mono">
                                    {m.parameters || m.family} · {isLoaded ? "loaded" : "local"}
                                  </div>
                                </div>
                              </div>
                              {isCurrent && <Check size={14} className="text-sky-400 shrink-0" />}
                            </button>
                          )
                        })}
                      </div>
                    </>
                  )}

                  {/* Empty state */}
                  {allUsableModels.length === 0 && (
                    <div className="px-4 py-6 text-center text-xs text-muted-foreground">
                      No models available.
                      <div className="mt-2 flex flex-col gap-1.5 items-center">
                        <Link href="/settings" className="text-primary underline">Add Cloud API Key</Link>
                        <Link href="/models" className="text-sky-400 underline">Download Local Model</Link>
                      </div>
                    </div>
                  )}
                </div>
              </>
            )}
          </div>
        </div>

        {/* Mode Switcher Tabs */}
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-border/40 pb-3">
          <div className="inline-flex rounded-xl bg-black/40 p-1 border border-border/60">
            <button
              type="button"
              onClick={() => {
                setPromptMode("concept")
                if (brief === SCRIPT_TEMPLATE) setBrief("")
              }}
              className={cn(
                "flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all",
                promptMode === "concept"
                  ? "bg-primary text-primary-foreground shadow-sm"
                  : "text-muted-foreground hover:text-foreground"
              )}
            >
              <Sparkles size={13} />
              Quick Concept Mode
            </button>
            <button
              type="button"
              onClick={() => setPromptMode("script")}
              className={cn(
                "flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all",
                promptMode === "script"
                  ? "bg-primary text-primary-foreground shadow-sm"
                  : "text-muted-foreground hover:text-foreground"
              )}
            >
              <FileCode2 size={13} />
              Full Script &amp; Scenes Mode
            </button>
          </div>

          {promptMode === "script" && (
            <button
              type="button"
              onClick={() => setBrief(SCRIPT_TEMPLATE)}
              className="text-xs text-primary font-mono hover:underline flex items-center gap-1"
            >
              Insert Scene Script Template
            </button>
          )}

          {customTotalSeconds > 0 && (
            <span className="text-[11px] font-mono font-semibold px-2.5 py-1 rounded-md bg-emerald-500/10 text-emerald-400 border border-emerald-500/25">
              Script Timing: {customTotalSeconds}s total
            </span>
          )}
        </div>

        {/* Prompt Input Box */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <label className={labelCls} htmlFor="brief">
              {promptMode === "concept" ? "Motion Graphic Concept Brief" : "Custom Scene-by-Scene Script with Timings"}
            </label>
            <span className="text-[11px] font-mono text-muted-foreground">
              {brief.length} / 4000
            </span>
          </div>
          <textarea
            id="brief"
            value={brief}
            onChange={(e) => setBrief(e.target.value)}
            onKeyDown={handleKeyDown}
            rows={promptMode === "script" ? 8 : 5}
            maxLength={4000}
            placeholder={
              promptMode === "concept"
                ? "Describe your motion graphic video. E.g. A high-energy 45-second explainer on how Transformer Attention works. Visualize neural attention weights as connecting laser paths, kinetic titles that scale with spring momentum, and glowing HUD data telemetry..."
                : `Title: Your Video Title\n\nScene 1 (6s):\nNarration: Exact words the narrator says...\nVisual: Motion graphics, kinetic text, glowing nodes...\n\nScene 2 (8s):\nNarration: Second scene spoken text...\nVisual: Animated SVG curves, 3D meshes, data charts...`
            }
            className={cn(
              inputCls,
              "resize-y leading-relaxed text-sm bg-black/50 border-border/70 focus:border-primary/80 focus:ring-1 focus:ring-primary/80 transition-all font-mono"
            )}
          />
          <div className="flex items-center justify-between text-[11px] text-muted-foreground">
            <span className="flex items-center gap-1.5">
              <Sparkles size={11} className="text-primary" />
              Press <kbd className="rounded border border-border px-1.5 py-0.5 text-[10px] font-mono bg-white/5">⌘ + Enter</kbd> to launch generation.
            </span>
            <span className="font-mono text-xs text-emerald-400">remotion-markup skill active</span>
          </div>
        </div>

        {/* Motion Graphics Archetype Selector */}
        <div className="space-y-3 rounded-2xl border border-border/60 bg-black/30 p-4">
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold uppercase tracking-wider text-foreground flex items-center gap-1.5">
              <Layers size={14} className="text-primary" />
              Motion Design Archetype
            </span>
            <span className="text-[11px] text-muted-foreground font-mono">
              Pure Remotion Skills
            </span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-2.5">
            {MOTION_STYLES.map((style) => {
              const active = selectedMotionStyleId === style.id
              return (
                <button
                  key={style.id}
                  type="button"
                  onClick={() => setSelectedMotionStyleId(style.id)}
                  className={cn(
                    "flex flex-col gap-1 p-3 rounded-xl border text-left transition-all",
                    active
                      ? "border-primary bg-primary/10 shadow-md shadow-primary/5 ring-1 ring-primary/40"
                      : "border-border/50 bg-secondary/20 hover:border-border hover:bg-secondary/40"
                  )}
                >
                  <div className="flex items-center gap-1.5">
                    <span className="text-base">{style.icon}</span>
                    <span className="text-xs font-bold text-foreground truncate">{style.name}</span>
                  </div>
                  <p className="text-[10px] text-muted-foreground line-clamp-2 leading-relaxed">
                    {style.description}
                  </p>
                </button>
              )
            })}
          </div>

          {/* Color Atmosphere Pills (Subtle, sleek, no form inputs) */}
          <div className="pt-2 border-t border-border/30 flex flex-wrap items-center justify-between gap-3">
            <span className="text-[11px] font-semibold text-muted-foreground uppercase tracking-wider">
              Visual Atmosphere:
            </span>
            <div className="flex flex-wrap gap-2">
              {COLOR_ATMOSPHERES.map((atm) => {
                const active = selectedAtmosphereId === atm.id
                return (
                  <button
                    key={atm.id}
                    type="button"
                    onClick={() => setSelectedAtmosphereId(atm.id)}
                    className={cn(
                      "flex items-center gap-2 px-3 py-1.5 rounded-lg border text-xs font-medium transition-all",
                      active
                        ? "border-primary bg-primary/15 text-foreground font-semibold"
                        : "border-border/40 bg-secondary/10 text-muted-foreground hover:bg-secondary/30 hover:text-foreground"
                    )}
                  >
                    <span
                      className="w-2.5 h-2.5 rounded-full shrink-0 border border-white/20"
                      style={{ backgroundColor: atm.primary }}
                    />
                    <span>{atm.name}</span>
                  </button>
                )
              })}
            </div>
          </div>

          {/* Optional Freeform Motion Directive */}
          <div className="pt-2">
            <input
              type="text"
              value={customArtDirection}
              onChange={(e) => setCustomArtDirection(e.target.value)}
              placeholder="Optional motion nuance: e.g. Smooth camera panning, glowing laser tracers, secondary subtle pulse..."
              className={cn(inputCls, "text-xs h-8 bg-black/40 border-border/50")}
            />
          </div>
        </div>

        {/* Video Duration & Narration Voice Controls */}
        <div className="grid gap-4 sm:grid-cols-2">
          <div>
            <div className="flex items-center justify-between">
              <span className={labelCls}>Target Duration</span>
              {customTotalSeconds > 0 && (
                <span className="text-[10px] text-primary font-mono font-medium">
                  Derived from script: {customTotalSeconds}s
                </span>
              )}
            </div>
            <div className="mt-2 inline-flex w-full rounded-xl border border-border/80 bg-black/40 p-1">
              {LENGTHS.map((l) => (
                <button
                  key={l}
                  type="button"
                  onClick={() => setSeconds(l)}
                  className={cn(
                    "flex-1 rounded-lg py-1.5 text-xs font-semibold transition-all",
                    (customTotalSeconds > 0 ? customTotalSeconds === l : seconds === l)
                      ? "bg-primary text-primary-foreground shadow-sm"
                      : "text-muted-foreground hover:text-foreground"
                  )}
                >
                  {l}s
                </button>
              ))}
            </div>
          </div>

          <div>
            <label className={labelCls} htmlFor="tts">
              Voice-over Synthesizer
            </label>
            {ttsModels.length === 0 ? (
              <p className="mt-2 text-xs text-muted-foreground">
                No speech synthesizer active. Video will be dynamically timed to text scripts.{" "}
                <Link href="/audio/" className="underline hover:text-foreground">
                  Setup Kokoro on Audio Page
                </Link>
                .
              </p>
            ) : (
              <div className="mt-2 flex gap-2">
                <select
                  id="tts"
                  value={tts}
                  onChange={(e) => {
                    setTts(e.target.value)
                    setVoice(ttsModels.find((t) => t.id === e.target.value)?.voices?.[0]?.id ?? "")
                  }}
                  className={cn(inputCls, "text-xs")}
                >
                  <option value="">No voice-over</option>
                  {ttsModels.map((t) => (
                    <option key={t.id} value={t.id}>
                      {t.name}
                    </option>
                  ))}
                </select>
                {tts && voices.length > 0 && (
                  <select
                    aria-label="Voice"
                    value={voice}
                    onChange={(e) => setVoice(e.target.value)}
                    className={cn(inputCls, "text-xs")}
                  >
                    {voices.map((v) => (
                      <option key={v.id} value={v.id}>
                        {v.name.split(" (")[0]}
                      </option>
                    ))}
                  </select>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Advanced Options Toggle */}
        <div>
          <button
            type="button"
            onClick={() => setAdvanced((v) => !v)}
            className="inline-flex items-center gap-1.5 text-xs font-semibold text-muted-foreground hover:text-foreground transition-colors"
          >
            <Sliders size={13} className={cn("transition-transform", advanced && "rotate-90 text-primary")} />
            {advanced ? "Hide Advanced Options" : "Advanced Engine Options"}
          </button>

          {advanced && (
            <div className="mt-3 grid gap-4 rounded-xl border border-border/60 bg-black/40 p-4 sm:grid-cols-2 animate-fade-in-up">
              <label className="flex items-start gap-2.5 text-xs cursor-pointer">
                <input
                  type="checkbox"
                  className="mt-0.5 accent-[var(--primary)]"
                  checked={reviewPlan}
                  onChange={(e) => setReviewPlan(e.target.checked)}
                />
                <span>
                  <b className="block text-foreground">Review Plan Before Coding</b>
                  <span className="text-muted-foreground">Pause after script generation to review and refine scenes before Remotion code is written.</span>
                </span>
              </label>

              <label className="flex items-start gap-2.5 text-xs cursor-pointer">
                <input
                  type="checkbox"
                  className="mt-0.5 accent-[var(--primary)]"
                  checked={autoRender}
                  onChange={(e) => setAutoRender(e.target.checked)}
                />
                <span>
                  <b className="block text-foreground">Auto-Render Final MP4</b>
                  <span className="text-muted-foreground">Automatically trigger the Remotion bundler when all scene checks pass.</span>
                </span>
              </label>

              <div>
                <label className={labelCls} htmlFor="attempts">
                  Fix Attempts Per Scene
                </label>
                <select
                  id="attempts"
                  value={attempts}
                  onChange={(e) => setAttempts(Number(e.target.value))}
                  className={cn(inputCls, "mt-1.5 text-xs")}
                >
                  {[2, 3, 4, 6, 8].map((n) => (
                    <option key={n} value={n}>
                      {n} attempts
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className={labelCls} htmlFor="budget">
                  Agent Time Budget
                </label>
                <select
                  id="budget"
                  value={hours}
                  onChange={(e) => setHours(Number(e.target.value))}
                  className={cn(inputCls, "mt-1.5 text-xs")}
                >
                  {BUDGETS.map((h) => (
                    <option key={h} value={h}>
                      {h} {h === 1 ? "hour" : "hours"}
                    </option>
                  ))}
                </select>
              </div>
            </div>
          )}
        </div>

        {/* Sandboxing & Quality Verification Banner */}
        <div className="flex items-start gap-3 rounded-xl border border-border/50 bg-black/30 p-3.5 text-xs text-muted-foreground">
          <ShieldCheck size={16} className="text-emerald-400 mt-0.5 shrink-0" />
          <p className="leading-relaxed">
            All scenes are validated against official Remotion skills via AST static verification, TypeScript compilation, and headless browser frame sampling before final multi-threaded composition.
          </p>
        </div>

        {rendererDown && (
          <p className="text-xs text-red-400 font-semibold">
            Rendering engine unavailable: {rendererMessage}
          </p>
        )}
        {error && (
          <div className="rounded-xl border border-red-500/30 bg-red-500/10 p-3.5 text-xs text-red-300">
            {error}
          </div>
        )}

        {/* Action Bar */}
        <div className="flex flex-wrap items-center justify-between gap-4 pt-2 border-t border-border/40">
          <div className="flex items-center gap-2 text-xs text-muted-foreground font-mono">
            <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
            <span>Remotion 4.0 Compositor Ready</span>
          </div>

          <button
            type="button"
            onClick={start}
            disabled={!ready}
            className={cn(
              primaryBtn,
              "px-6 py-2.5 rounded-xl font-bold flex items-center gap-2 shadow-lg transition-all",
              ready
                ? "bg-gradient-to-r from-amber-500 via-orange-500 to-amber-600 text-black hover:opacity-95 shadow-amber-500/20"
                : "opacity-40 cursor-not-allowed"
            )}
          >
            {busy ? <Loader2 size={16} className="animate-spin" /> : <Sparkles size={16} />}
            Generate Motion Graphics
          </button>
        </div>
      </div>
    </div>
  )
}
