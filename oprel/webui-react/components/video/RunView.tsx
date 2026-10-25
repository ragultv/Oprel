"use client"

import { useEffect, useMemo, useState } from "react"
import { AlertTriangle, Check, CheckCircle2, ChevronDown, Pencil, Play, Plus, RotateCcw, Save, Sparkles, Square, X } from "lucide-react"
import type { AgentEvent, AgentState } from "@/services/api"
import { cn } from "@/services/utils"
import { ActivityLog } from "./ActivityLog"
import { OutputCard } from "./OutputCard"
import { SceneCard, type SceneDraft } from "./SceneCard"
import { RuntimePanel } from "./RuntimePanel"
import { ACTIVE_RUN, RunChip, card, fmtElapsed, ghostBtn, labelCls, primaryBtn } from "./shared"

type Props = {
  projectId: string
  state: AgentState
  events: AgentEvent[]
  hasFinal: boolean
  busy: boolean
  onCancel: () => void
  onResume: () => void
  onApprove: () => void
  onRetry: (sceneId: string, feedback: string) => void
  onSavePlan: (title: string, scenes: { id?: string; title: string; narration: string; visual: string }[]) => Promise<void>
  onStartOver: () => void
  onRendered: () => void
}

type Step = { key: string; label: string; detail: string; tone: "done" | "active" | "todo" | "skip" }

export function RunView({ projectId, state, events, hasFinal, busy, onCancel, onResume, onApprove, onRetry, onSavePlan, onStartOver, onRendered }: Props) {
  const run = state.run!
  const active = ACTIVE_RUN.has(run.status)
  const [now, setNow] = useState(Date.now())
  const [draft, setDraft] = useState<SceneDraft[] | null>(null)
  const [draftTitle, setDraftTitle] = useState("")
  const [saveError, setSaveError] = useState<string | null>(null)
  const [showIdea, setShowIdea] = useState(false)

  useEffect(() => {
    if (!active) return
    const t = setInterval(() => setNow(Date.now()), 1000)
    return () => clearInterval(t)
  }, [active])

  const plan = run.plan
  const scenes = plan?.scenes ?? []
  const done = scenes.filter((s) => run.scenes[s.id]?.status === "done").length
  const needAttention = scenes.filter((s) => run.scenes[s.id]?.status === "failed").length
  const elapsed = ((run.finished_at ?? (active ? now / 1000 : run.updated_at)) - run.started_at)
  const version = Math.floor(run.updated_at)

  const steps: Step[] = useMemo(() => {
    const st = run.status
    const planDone = !!plan
    const narrSkipped = !run.config.tts_model
    return [
      { key: "plan", label: "Plan", detail: planDone ? `${scenes.length} scenes` : "writing the script", tone: planDone ? "done" : st === "planning" ? "active" : "todo" },
      { key: "narr", label: "Voice-over", detail: narrSkipped ? "skipped" : run.narrated ? "recorded" : "speech + timing", tone: narrSkipped ? "skip" : run.narrated ? "done" : st === "narrating" ? "active" : "todo" },
      { key: "scenes", label: "Scenes", detail: scenes.length ? `${done}/${scenes.length} ready` : "waiting", tone: scenes.length && done === scenes.length ? "done" : st === "writing" ? "active" : "todo" },
      { key: "render", label: "Final video", detail: st === "rendering" ? `${Math.round((run.render?.progress ?? 0) * 100)}%` : st === "completed" ? "ready" : "waiting", tone: st === "completed" ? "done" : st === "assembling" || st === "rendering" ? "active" : "todo" },
    ]
  }, [run, plan, scenes.length, done])

  const startEdit = () => {
    if (!plan) return
    setDraft(plan.scenes.map((s) => ({ id: s.id, title: s.title, narration: s.narration, visual: s.visual })))
    setDraftTitle(plan.title)
    setSaveError(null)
  }
  const saveEdit = async () => {
    if (!draft) return
    try {
      await onSavePlan(draftTitle, draft.map((d) => ({ id: d.id.startsWith("new-") ? undefined : d.id, title: d.title, narration: d.narration, visual: d.visual })))
      setDraft(null)
    } catch (e) { setSaveError((e as Error).message) }
  }

  const canRetry = !active && run.status !== "awaiting_review"
  const resumable = ["interrupted", "needs_attention", "cancelled", "failed"].includes(run.status)
  const rendering = run.status === "rendering" ? (run.render?.progress ?? 0) : null

  return (
    <div className="mx-auto w-full max-w-[1400px] space-y-5 px-5 py-6 sm:px-8">
      {/* Summary */}
      <div className={cn(card, "space-y-4 p-5")}>
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div className="min-w-0 flex-1">
            <div className="flex flex-wrap items-center gap-3">
              <h1 className="truncate text-xl font-bold tracking-tight">{plan?.title || "Planning your video…"}</h1>
              <RunChip status={run.status} />
            </div>
            {plan?.logline && <p className="mt-1.5 text-sm text-muted-foreground">{plan.logline}</p>}
            <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] text-muted-foreground">
              <span>Elapsed {fmtElapsed(elapsed)}</span>
              <span>Target {run.config.target_seconds}s</span>
              <span className="max-w-[260px] truncate">Model {run.config.model_id.split("::").pop()}</span>
              {run.config.tts_model && <span>Voice {run.config.tts_model}</span>}
              {run.context_tokens ? <span title="Context window of the model; small windows use shorter prompts">Context {run.context_tokens.toLocaleString()} tokens</span> : null}
              <button onClick={() => setShowIdea((v) => !v)} className="inline-flex items-center gap-1 font-semibold hover:text-foreground">
                Your idea <ChevronDown size={12} className={cn("transition", showIdea && "rotate-180")} />
              </button>
            </div>
            {showIdea && <p className="mt-2 rounded-xl border border-white/10 bg-black/20 p-3 text-xs leading-relaxed text-muted-foreground">{run.config.brief}</p>}
          </div>
          <div className="flex flex-wrap items-center gap-2">
            {active && <button className={ghostBtn} onClick={onCancel}><Square size={13} /> Stop</button>}
            {resumable && <button className={primaryBtn} onClick={onResume} disabled={busy}><Play size={15} /> Resume</button>}
            {run.status === "awaiting_review" && <button className={primaryBtn} onClick={onApprove} disabled={busy || !!draft}><Check size={15} /> Approve &amp; continue</button>}
            {!active && <button className={ghostBtn} onClick={onStartOver}><Sparkles size={13} /> New idea</button>}
          </div>
        </div>

        {/* Stepper */}
        <ol className="grid gap-2 sm:grid-cols-4">
          {steps.map((s, i) => (
            <li key={s.key} className={cn("flex items-center gap-3 rounded-xl border px-3 py-2.5",
              s.tone === "done" ? "border-emerald-500/20 bg-emerald-500/5" : s.tone === "active" ? "border-sky-500/30 bg-sky-500/5" : "border-white/10 bg-white/[0.02]")}>
              <span className={cn("flex h-6 w-6 shrink-0 items-center justify-center rounded-full text-[11px] font-bold",
                s.tone === "done" ? "bg-emerald-500/20 text-emerald-400" : s.tone === "active" ? "bg-sky-500/20 text-sky-400" : "bg-white/5 text-muted-foreground")}>
                {s.tone === "done" ? <CheckCircle2 size={14} /> : s.tone === "active" ? <span className="h-2 w-2 animate-pulse rounded-full bg-sky-400" /> : i + 1}
              </span>
              <div className="min-w-0">
                <div className="text-xs font-semibold">{s.label}</div>
                <div className="truncate text-[11px] text-muted-foreground">{s.detail}</div>
              </div>
            </li>
          ))}
        </ol>

        {active && run.message && <p className="text-xs text-muted-foreground">{run.message}</p>}
        {run.error && !active && (
          <div className="flex gap-2.5 rounded-xl border border-amber-500/25 bg-amber-500/10 p-3 text-xs text-amber-200">
            <AlertTriangle size={14} className="mt-0.5 shrink-0" />
            <div>
              <div className="font-semibold">{run.status === "failed" ? "The run stopped on an error" : "The run paused"}</div>
              <div className="mt-0.5 opacity-90">{run.error.message}</div>
            </div>
          </div>
        )}
        {run.error && !active && run.error.code === "small_context" && (
          <RuntimePanel modelId={run.config.model_id} />
        )}
        {run.status === "awaiting_review" && (
          <div className="rounded-xl border border-primary/25 bg-primary/5 p-3 text-xs">
            <b>Review the plan.</b> Edit any scene&apos;s narration or visual, add or remove scenes, then approve. No code is written until you do.
          </div>
        )}
      </div>

      <div className="grid gap-5 xl:grid-cols-[minmax(0,1fr)_380px]">
        {/* Scenes */}
        <section className="min-w-0 space-y-3">
          <div className="flex items-center justify-between">
            <h2 className={labelCls}>Scenes {scenes.length > 0 && `· ${done}/${scenes.length} ready${needAttention ? ` · ${needAttention} need attention` : ""}`}</h2>
            {plan && !active && !draft && <button className={ghostBtn} onClick={startEdit}><Pencil size={12} /> Edit plan</button>}
            {draft && (
              <div className="flex gap-2">
                <button className={ghostBtn} onClick={() => setDraft(null)}><X size={12} /> Cancel</button>
                <button className={primaryBtn} onClick={saveEdit}><Save size={14} /> Save plan</button>
              </div>
            )}
          </div>
          {saveError && <div className="rounded-xl border border-red-500/30 bg-red-500/10 p-3 text-xs text-red-300">{saveError}</div>}
          {draft && (
            <div className="rounded-xl border border-white/10 bg-black/20 p-3">
              <label className={labelCls} htmlFor="plan-title">Video title</label>
              <input id="plan-title" value={draftTitle} onChange={(e) => setDraftTitle(e.target.value)} className="mt-1.5 w-full rounded-xl border border-white/10 bg-white/5 px-3.5 py-2.5 text-sm outline-none focus:border-primary/50" />
              <p className="mt-2 text-[11px] text-muted-foreground">Changed scenes are rewritten. If you change narration, the voice-over is re-recorded for the whole video.</p>
            </div>
          )}

          {scenes.length === 0 && (
            <div className={cn(card, "p-8 text-center text-sm text-muted-foreground")}>
              {active ? "The director is writing the script…" : "No plan yet."}
            </div>
          )}
          {(draft ?? scenes).map((s, i) => {
            const base = scenes.find((x) => x.id === s.id) ?? { id: s.id, title: s.title, narration: s.narration, visual: s.visual, seconds: 0 }
            return (
              <SceneCard
                key={s.id}
                projectId={projectId} index={i} total={(draft ?? scenes).length} scene={base} state={run.scenes[s.id]} version={version}
                canRetry={canRetry} onRetry={onRetry}
                draft={draft ? {
                  value: draft[i],
                  onChange: (v) => setDraft(draft.map((d, k) => (k === i ? v : d))),
                  onRemove: () => setDraft(draft.filter((_, k) => k !== i)),
                  onMove: (dir) => { const j = i + dir; if (j < 0 || j >= draft.length) return; const n = [...draft]; [n[i], n[j]] = [n[j], n[i]]; setDraft(n) },
                } : undefined}
              />
            )
          })}
          {draft && draft.length < 14 && (
            <button className={cn(ghostBtn, "w-full border-dashed")} onClick={() => setDraft([...draft, { id: `new-${Date.now()}`, title: "New scene", narration: "", visual: "" }])}>
              <Plus size={14} /> Add a scene
            </button>
          )}
          {needAttention > 0 && !active && !draft && (
            <p className="flex items-center gap-1.5 text-[11px] text-muted-foreground"><RotateCcw size={11} /> Open a scene that needs attention to retry it with extra direction, or press Resume to give every unfinished scene a fresh round of attempts.</p>
          )}
        </section>

        {/* Output + activity */}
        <aside className="min-w-0 space-y-4">
          <OutputCard projectId={projectId} hasFinal={hasFinal} agentRenderProgress={rendering} canRender={!active && scenes.length > 0 && done === scenes.length} onRendered={onRendered} />
          <ActivityLog events={events} live={active} />
        </aside>
      </div>
    </div>
  )
}
