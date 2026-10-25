"use client"

import { useCallback, useEffect, useRef, useState } from "react"
import { AlertTriangle, Clapperboard, Film, Plus, Trash2 } from "lucide-react"
import { VIDEO, type AgentEvent, type AgentStartBody, type AgentState, type VideoProject, type VideoStatus } from "@/services/api"
import { useApp } from "@/services/context"
import { cn } from "@/services/utils"
import { Composer } from "@/components/video/Composer"
import { RunView } from "@/components/video/RunView"
import { ACTIVE_RUN, RunChip, ghostBtn, timeAgo } from "@/components/video/shared"

const POLL_ACTIVE_MS = 1500
const POLL_IDLE_MS = 6000

export function VideoView() {
  const { models, activeModelId } = useApp()
  const [status, setStatus] = useState<VideoStatus | null>(null)
  const [projects, setProjects] = useState<VideoProject[]>([])
  const [ttsModels, setTtsModels] = useState<{ id: string; name: string; voices: { id: string; name: string }[] }[]>([])
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [agent, setAgent] = useState<AgentState | null>(null)
  const [events, setEvents] = useState<AgentEvent[]>([])
  const [composing, setComposing] = useState(true)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [hasFinal, setHasFinal] = useState(false)
  const [loaded, setLoaded] = useState(false)
  const lastSeq = useRef(0)
  const selectedRef = useRef<string | null>(null)
  selectedRef.current = selectedId

  const refreshList = useCallback(async () => {
    try { setProjects(await VIDEO.list()) } catch (e) { setError((e as Error).message) } finally { setLoaded(true) }
  }, [])

  const applyState = useCallback((id: string, s: AgentState, reset = false) => {
    if (selectedRef.current !== id) return
    setAgent(s)
    setEvents((prev) => {
      const base = reset ? [] : prev
      const seen = new Set(base.map((e) => e.seq))
      const merged = [...base, ...s.events.filter((e) => !seen.has(e.seq))]
      lastSeq.current = merged.length ? merged[merged.length - 1].seq : 0
      return merged
    })
  }, [])

  const select = useCallback(async (id: string) => {
    setSelectedId(id); selectedRef.current = id
    setError(null); setAgent(null); setEvents([]); lastSeq.current = 0
    try {
      const [p, s] = await Promise.all([VIDEO.get(id), VIDEO.agent(id, 0)])
      setHasFinal(!!p.hasFinalOutput)
      applyState(id, s, true)
      setComposing(!s.run)
    } catch (e) { setError((e as Error).message) }
  }, [applyState])

  // initial load
  useEffect(() => {
    VIDEO.status().then(setStatus).catch((e) => setStatus({ available: false, message: (e as Error).message }))
    VIDEO.ttsModels().then(setTtsModels).catch(() => {})
    refreshList()
    const linked = new URLSearchParams(window.location.search).get("project")
    if (linked) select(linked)
  }, [refreshList, select])

  // live updates for the open project
  useEffect(() => {
    if (!selectedId) return
    let stop = false
    let timer: ReturnType<typeof setTimeout>
    const tick = async () => {
      if (stop) return
      let delay = POLL_IDLE_MS
      try {
        const s = await VIDEO.agent(selectedId, lastSeq.current)
        if (stop) return
        applyState(selectedId, s)
        if (s.run && ACTIVE_RUN.has(s.run.status)) delay = POLL_ACTIVE_MS
        if (s.run && !ACTIVE_RUN.has(s.run.status)) {
          const p = await VIDEO.get(selectedId).catch(() => null)
          if (p && !stop) setHasFinal(!!p.hasFinalOutput)
        }
      } catch { /* transient: keep polling */ }
      if (!stop) timer = setTimeout(tick, delay)
    }
    timer = setTimeout(tick, POLL_ACTIVE_MS)
    return () => { stop = true; clearTimeout(timer) }
  }, [selectedId, applyState])

  // keep the project list fresh while something is running
  useEffect(() => {
    if (!agent?.run || !ACTIVE_RUN.has(agent.run.status)) { refreshList(); return }
    const t = setInterval(refreshList, 8000)
    return () => clearInterval(t)
  }, [agent?.run?.status, refreshList])

  const newVideo = () => { setSelectedId(null); selectedRef.current = null; setAgent(null); setEvents([]); setComposing(true); setError(null) }

  const start = async (body: AgentStartBody) => {
    setBusy(true); setError(null)
    try {
      let id = selectedId
      if (!id) {
        const firstLine = body.brief.split("\n")[0] || body.brief
        const title = firstLine.replace(/\s+/g, " ").slice(0, 60).trim() || "Untitled video"
        const p = await VIDEO.create(title)
        id = p.id
        setSelectedId(id); selectedRef.current = id
        setEvents([]); lastSeq.current = 0
      }
      const s = await VIDEO.agentStart(id, body)
      applyState(id, s, true)
      setComposing(false)
      refreshList()
    } catch (e) { setError((e as Error).message) } finally { setBusy(false) }
  }

  const act = async (fn: () => Promise<AgentState>) => {
    if (!selectedId) return
    setBusy(true); setError(null)
    try { applyState(selectedId, await fn()); refreshList() } catch (e) { setError((e as Error).message) } finally { setBusy(false) }
  }

  const remove = async (id: string) => {
    if (!confirm("Delete this video project, its scenes and renders? This cannot be undone.")) return
    try {
      await VIDEO.remove(id)
      if (selectedId === id) newVideo()
      refreshList()
    } catch (e) { setError((e as Error).message) }
  }

  const rendererDown = status !== null && !status.available
  const run = agent?.run ?? null

  return (
    <div className="relative flex h-full flex-col overflow-hidden bg-[#0a0a0a] text-foreground">
      <header className="sticky top-0 z-20 flex h-14 shrink-0 items-center justify-between border-b border-white/10 bg-background/95 px-5 backdrop-blur-sm sm:px-8">
        <div className="flex items-center gap-3">
          <div className="flex h-8 w-8 items-center justify-center rounded-xl border border-primary/20 bg-primary/10 text-primary"><Clapperboard size={18} /></div>
          <div>
            <div className="flex items-center gap-2">
              <span className="text-sm font-bold">Video Studio</span>
              <span title={rendererDown ? status?.message ?? "Rendering unavailable" : "Renderer ready"} className={cn("h-2 w-2 rounded-full", status === null ? "bg-white/20" : rendererDown ? "bg-red-500" : "animate-pulse bg-emerald-500")} />
            </div>
            <div className="text-[10px] font-medium text-muted-foreground">AI-directed motion graphics for YouTube Shorts</div>
          </div>
        </div>
        <button onClick={newVideo} className="inline-flex items-center gap-1.5 rounded-full bg-primary px-4 py-2 text-xs font-semibold text-primary-foreground transition hover:opacity-90">
          <Plus size={14} /> New video
        </button>
      </header>

      <div className="flex min-h-0 flex-1">
        <aside className="hidden w-64 shrink-0 flex-col border-r border-white/10 md:flex">
          <div className="px-4 pb-2 pt-4 text-[10px] font-semibold uppercase tracking-widest text-muted-foreground">Your videos</div>
          <div className="min-h-0 flex-1 space-y-1 overflow-y-auto px-2 pb-3">
            {loaded && projects.length === 0 && <p className="px-3 py-2 text-xs text-muted-foreground">Nothing here yet. Describe your first video to begin.</p>}
            {projects.map((p) => (
              <div key={p.id} role="button" tabIndex={0} onClick={() => select(p.id)} onKeyDown={(e) => { if (e.key === "Enter") select(p.id) }}
                className={cn("group flex cursor-pointer items-center gap-2.5 rounded-xl border px-3 py-2.5 transition-colors",
                  selectedId === p.id ? "border-white/10 bg-white/5" : "border-transparent hover:bg-white/[0.04]")}>
                <Film size={14} className="shrink-0 text-muted-foreground" />
                <div className="min-w-0 flex-1">
                  <div className="truncate text-xs font-semibold">{p.title}</div>
                  <div className="mt-0.5 flex items-center gap-1.5 text-[10px] text-muted-foreground">
                    {p.agentStatus ? <RunChip status={p.agentStatus} /> : <span>Not started</span>}
                  </div>
                  <div className="mt-1 text-[10px] text-muted-foreground/70">{timeAgo(p.updatedAt)}{p.hasFinalOutput ? " · video ready" : ""}</div>
                </div>
                <button onClick={(e) => { e.stopPropagation(); remove(p.id) }} aria-label={`Delete ${p.title}`} className="rounded-md p-1 text-muted-foreground opacity-0 transition hover:text-red-400 focus:opacity-100 group-hover:opacity-100">
                  <Trash2 size={13} />
                </button>
              </div>
            ))}
          </div>
        </aside>

        <main className="min-w-0 flex-1 overflow-y-auto">
          {rendererDown && (
            <div className="mx-5 mt-5 flex items-start gap-2.5 rounded-xl border border-amber-500/25 bg-amber-500/10 p-3 text-xs text-amber-200 sm:mx-8">
              <AlertTriangle size={14} className="mt-0.5 shrink-0" />
              <div>
                <b>Rendering is not set up on this machine.</b> {status?.message}
              </div>
            </div>
          )}
          {error && !composing && (
            <div className="mx-5 mt-5 flex items-center justify-between gap-3 rounded-xl border border-red-500/30 bg-red-500/10 p-3 text-xs text-red-300 sm:mx-8">
              <span>{error}</span>
              <button className={ghostBtn} onClick={() => setError(null)}>Dismiss</button>
            </div>
          )}

          {composing || !run ? (
            <Composer
              models={models} activeModelId={activeModelId} ttsModels={ttsModels} rendererDown={rendererDown} rendererMessage={status?.message}
              busy={busy} error={composing ? error : null} initialBrief={run?.config.brief ?? ""} onStart={start}
            />
          ) : (
            <RunView
              projectId={selectedId!} state={agent!} events={events} hasFinal={hasFinal} busy={busy}
              onCancel={() => selectedId && VIDEO.agentCancel(selectedId).catch((e) => setError(e.message))}
              onResume={() => act(() => VIDEO.agentResume(selectedId!))}
              onApprove={() => act(() => VIDEO.agentApprove(selectedId!))}
              onRetry={(sceneId, fb) => act(() => VIDEO.agentRetryScene(selectedId!, sceneId, fb))}
              onSavePlan={async (title, scenes) => {
                const s = await VIDEO.agentUpdatePlan(selectedId!, title, scenes as never)
                applyState(selectedId!, s)
              }}
              onStartOver={() => setComposing(true)}
              onRendered={() => { refreshList(); VIDEO.get(selectedId!).then((p) => setHasFinal(!!p.hasFinalOutput)).catch(() => {}) }}
            />
          )}
        </main>
      </div>
    </div>
  )
}
