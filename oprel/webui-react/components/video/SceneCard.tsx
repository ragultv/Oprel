"use client"

import { useEffect, useState } from "react"
import { ArrowDown, ArrowUp, ChevronDown, Code2, Copy, RotateCcw, Trash2 } from "lucide-react"
import { VIDEO, type AgentPlanScene, type AgentSceneState } from "@/services/api"
import { cn } from "@/services/utils"
import { SceneChip, card, ghostBtn, inputCls, labelCls } from "./shared"

export type SceneDraft = Pick<AgentPlanScene, "id" | "title" | "narration" | "visual">

type Props = {
  projectId: string
  index: number
  total: number
  scene: AgentPlanScene
  state?: AgentSceneState
  /** Bumps when the run changes so thumbnails refresh. */
  version: number
  canRetry: boolean
  /** Present while the plan is being edited. */
  draft?: { value: SceneDraft; onChange: (v: SceneDraft) => void; onRemove: () => void; onMove: (dir: -1 | 1) => void }
  onRetry: (sceneId: string, feedback: string) => void
}

export function SceneCard({ projectId, index, total, scene, state, version, canRetry, draft, onRetry }: Props) {
  const [open, setOpen] = useState(false)
  const [code, setCode] = useState<string | null>(null)
  const [codeError, setCodeError] = useState<string | null>(null)
  const [showCode, setShowCode] = useState(false)
  const [feedback, setFeedback] = useState("")

  const status = state?.status ?? "pending"
  const attention = status === "failed"
  const stills = state?.stills ?? []
  const lastTry = state?.history?.[state.history.length - 1]

  useEffect(() => {
    if (attention) setOpen(true)
  }, [attention])

  useEffect(() => {
    if (!showCode || !state) return
    const rel = status === "done" ? state.file : lastTry?.file
    if (!rel) return
    setCode(null); setCodeError(null)
    VIDEO.fileText(projectId, rel).then(setCode).catch(() => setCodeError("Code is not available yet."))
  }, [showCode, status, state, lastTry?.file, projectId, version])

  const seconds = state ? state.duration_frames / 30 : scene.seconds

  return (
    <div className={cn(card, "overflow-hidden transition-colors", attention && "border-red-500/30", status === "done" && "border-emerald-500/15")}>
      <button onClick={() => setOpen((v) => !v)} className="flex w-full items-center gap-3 px-4 py-3 text-left" aria-expanded={open}>
        <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-white/5 text-xs font-bold text-muted-foreground">{index + 1}</span>
        <div className="min-w-0 flex-1">
          <div className="truncate text-sm font-semibold">{draft ? draft.value.title : scene.title}</div>
          <div className="truncate text-[11px] text-muted-foreground">
            {seconds ? `${seconds.toFixed(1)}s` : ""}{state && state.attempts > 0 ? ` · attempt ${state.attempts}` : ""}
          </div>
        </div>
        {stills.length > 0 && !open && (
          <div className="hidden gap-1 sm:flex">
            {stills.slice(-3).map((s) => (
              // eslint-disable-next-line @next/next/no-img-element
              <img key={s} src={VIDEO.fileUrl(projectId, s, version)} alt="" className="h-12 w-[27px] rounded-[3px] border border-white/10 object-cover" />
            ))}
          </div>
        )}
        {!draft && <SceneChip status={status} />}
        <ChevronDown size={16} className={cn("shrink-0 text-muted-foreground transition", open && "rotate-180")} />
      </button>

      {open && (
        <div className="space-y-4 border-t border-white/10 px-4 py-4">
          {draft ? (
            <div className="space-y-3">
              <div>
                <label className={labelCls}>Scene title</label>
                <input value={draft.value.title} onChange={(e) => draft.onChange({ ...draft.value, title: e.target.value })} className={cn(inputCls, "mt-1.5")} />
              </div>
              <div>
                <label className={labelCls}>Narration</label>
                <textarea value={draft.value.narration} onChange={(e) => draft.onChange({ ...draft.value, narration: e.target.value })} rows={3} className={cn(inputCls, "mt-1.5 resize-y")} />
              </div>
              <div>
                <label className={labelCls}>What the viewer sees</label>
                <textarea value={draft.value.visual} onChange={(e) => draft.onChange({ ...draft.value, visual: e.target.value })} rows={4} className={cn(inputCls, "mt-1.5 resize-y")} />
              </div>
              <div className="flex gap-2">
                <button className={ghostBtn} onClick={() => draft.onMove(-1)} disabled={index === 0}><ArrowUp size={13} /> Up</button>
                <button className={ghostBtn} onClick={() => draft.onMove(1)} disabled={index === total - 1}><ArrowDown size={13} /> Down</button>
                <button className={cn(ghostBtn, "ml-auto text-red-400")} onClick={draft.onRemove} disabled={total <= 1}><Trash2 size={13} /> Remove</button>
              </div>
            </div>
          ) : (
            <>
              <div className="grid gap-4 md:grid-cols-2">
                <div>
                  <div className={labelCls}>Narration</div>
                  <p className="mt-1.5 text-sm leading-relaxed text-foreground/90">{scene.narration}</p>
                </div>
                <div>
                  <div className={labelCls}>What the viewer sees</div>
                  <p className="mt-1.5 text-sm leading-relaxed text-muted-foreground">{scene.visual}</p>
                </div>
              </div>

              {stills.length > 0 && (
                <div>
                  <div className={labelCls}>Frames rendered by the checker</div>
                  <div className="mt-2 flex gap-2 overflow-x-auto pb-1">
                    {stills.map((s) => (
                      // eslint-disable-next-line @next/next/no-img-element
                      <img key={s} src={VIDEO.fileUrl(projectId, s, version)} alt="Sample frame of this scene" className="h-48 w-[108px] shrink-0 rounded-lg border border-white/10 object-cover" />
                    ))}
                  </div>
                </div>
              )}

              {state && state.last_issues.length > 0 && status !== "done" && (
                <div className="rounded-xl border border-red-500/25 bg-red-500/10 p-3">
                  <div className="text-xs font-semibold text-red-300">Last check found</div>
                  <ul className="mt-1.5 space-y-1 text-xs text-red-200/90">
                    {state.last_issues.slice(0, 5).map((i, k) => (
                      <li key={k}><span className="font-mono opacity-70">{i.path}</span> {i.message}</li>
                    ))}
                  </ul>
                </div>
              )}

              {state && state.history.length > 0 && (
                <div>
                  <div className={labelCls}>Attempts</div>
                  <ul className="mt-1.5 space-y-1 text-xs">
                    {state.history.map((h) => (
                      <li key={h.attempt} className="flex gap-2">
                        <span className={cn("w-16 shrink-0 font-semibold", h.ok ? "text-emerald-400" : "text-amber-400")}>#{h.attempt} {h.ok ? "passed" : "failed"}</span>
                        <span className="min-w-0 truncate text-muted-foreground">{h.ok ? "all checks passed" : `${h.stage}: ${h.issues[0]?.message ?? ""}`}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              <div className="flex flex-wrap items-center gap-2">
                {state && (state.history.length > 0 || status === "done") && (
                  <button className={ghostBtn} onClick={() => setShowCode((v) => !v)}><Code2 size={13} /> {showCode ? "Hide code" : "View code"}</button>
                )}
              </div>
              {showCode && (
                <div className="relative rounded-xl border border-white/10 bg-black/40">
                  {code && (
                    <button onClick={() => navigator.clipboard?.writeText(code)} className="absolute right-2 top-2 rounded-md border border-white/10 bg-white/5 p-1.5 text-muted-foreground hover:text-foreground" aria-label="Copy code"><Copy size={12} /></button>
                  )}
                  <pre className="max-h-96 overflow-auto p-4 font-mono text-[11px] leading-relaxed text-foreground/90">{codeError ?? code ?? "Loading…"}</pre>
                </div>
              )}

              {canRetry && state && (
                <div className="rounded-xl border border-white/10 bg-black/20 p-3">
                  <label className={labelCls} htmlFor={`fb-${scene.id}`}>{status === "failed" ? "Try again with direction" : "Rewrite this scene"}</label>
                  <div className="mt-2 flex gap-2">
                    <input id={`fb-${scene.id}`} value={feedback} onChange={(e) => setFeedback(e.target.value)} placeholder="Optional: e.g. make the labels bigger, slow the animation down"
                      className={inputCls} />
                    <button className={ghostBtn} onClick={() => { onRetry(scene.id, feedback); setFeedback("") }}><RotateCcw size={13} /> Rewrite</button>
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      )}
    </div>
  )
}
