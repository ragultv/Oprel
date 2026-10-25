"use client"

import { useEffect, useRef, useState } from "react"
import { AlertTriangle, Download, Film, Loader2, Play } from "lucide-react"
import { VIDEO, type VideoJob } from "@/services/api"
import { cn } from "@/services/utils"
import { card, ghostBtn, labelCls } from "./shared"

type Props = {
  projectId: string
  hasFinal: boolean
  /** Agent is rendering the final video itself (progress 0..1). */
  agentRenderProgress: number | null
  canRender: boolean
  /** Called after a manual render finishes so the project list refreshes. */
  onRendered: () => void
}

export function OutputCard({ projectId, hasFinal, agentRenderProgress, canRender, onRendered }: Props) {
  const [name, setName] = useState<"final.mp4" | "preview.mp4">(hasFinal ? "final.mp4" : "preview.mp4")
  const [bust, setBust] = useState(Date.now())
  const [job, setJob] = useState<VideoJob | null>(null)
  const [error, setError] = useState<string | null>(null)
  const timer = useRef<ReturnType<typeof setInterval> | null>(null)
  const [exists, setExists] = useState(hasFinal)

  useEffect(() => { setExists(hasFinal); if (hasFinal) { setName("final.mp4"); setBust(Date.now()) } }, [hasFinal])
  useEffect(() => () => { if (timer.current) clearInterval(timer.current) }, [])

  const rendering = job && (job.status === "queued" || job.status === "running")

  const render = async (quality: "preview" | "production") => {
    setError(null)
    try {
      const j = await VIDEO.render(projectId, quality, true)
      setJob(j)
      timer.current = setInterval(async () => {
        const cur = await VIDEO.job(j.id).catch(() => null)
        if (!cur) return
        setJob(cur)
        if (cur.status !== "queued" && cur.status !== "running") {
          if (timer.current) clearInterval(timer.current)
          if (cur.status === "completed") { setName(quality === "preview" ? "preview.mp4" : "final.mp4"); setExists(true); setBust(Date.now()); onRendered() }
          else setError(cur.error?.message ?? "Render failed")
        }
      }, 1000)
    } catch (e) { setError((e as Error).message) }
  }

  return (
    <div className={cn(card, "space-y-3 p-4")}>
      <div className="flex items-center justify-between">
        <span className={labelCls}>Video</span>
        {exists && (
          <div className="inline-flex rounded-full border border-white/10 bg-white/5 p-0.5 text-[11px]">
            {(["final.mp4", "preview.mp4"] as const).map((n) => (
              <button key={n} onClick={() => { setName(n); setBust(Date.now()) }} className={cn("rounded-full px-2.5 py-1 font-semibold", name === n ? "bg-white/10" : "text-muted-foreground")}>
                {n === "final.mp4" ? "HD" : "Preview"}
              </button>
            ))}
          </div>
        )}
      </div>

      {agentRenderProgress !== null ? (
        <div className="flex aspect-[9/16] max-h-[460px] w-full flex-col items-center justify-center gap-3 rounded-xl border border-white/10 bg-black/40">
          <Loader2 className="animate-spin text-sky-400" />
          <div className="text-sm font-semibold">Rendering {Math.round(agentRenderProgress * 100)}%</div>
          <div className="h-1.5 w-40 overflow-hidden rounded-full bg-white/10"><div className="h-full bg-primary transition-all" style={{ width: `${Math.round(agentRenderProgress * 100)}%` }} /></div>
        </div>
      ) : exists ? (
        <>
          <video key={name + bust} src={VIDEO.outputUrl(projectId, name, bust)} controls playsInline className="aspect-[9/16] max-h-[520px] w-full rounded-xl bg-black object-contain" />
          <a href={VIDEO.outputUrl(projectId, name, bust)} download className={cn(ghostBtn, "w-full")}><Download size={13} /> Download {name === "final.mp4" ? "HD video" : "preview"}</a>
        </>
      ) : (
        <div className="flex aspect-[9/16] max-h-[360px] w-full flex-col items-center justify-center gap-2 rounded-xl border border-dashed border-white/10 text-center text-xs text-muted-foreground">
          <Film size={22} /> The finished video appears here.
        </div>
      )}

      {rendering && job && (
        <div className="space-y-1.5">
          <div className="flex justify-between text-[11px] text-muted-foreground"><span>{job.stage}</span><span>{Math.round(job.progress * 100)}%</span></div>
          <div className="h-1.5 overflow-hidden rounded-full bg-white/10"><div className="h-full bg-primary transition-all" style={{ width: `${Math.round(job.progress * 100)}%` }} /></div>
          <button className={ghostBtn} onClick={() => VIDEO.cancel(job.id)}>Cancel render</button>
        </div>
      )}
      {error && <div className="flex gap-2 rounded-xl border border-red-500/25 bg-red-500/10 p-2.5 text-xs text-red-300"><AlertTriangle size={13} className="mt-0.5 shrink-0" />{error}</div>}

      {canRender && !rendering && (
        <div className="flex gap-2">
          <button className={cn(ghostBtn, "flex-1")} onClick={() => render("preview")}><Play size={13} /> Quick preview</button>
          <button className={cn(ghostBtn, "flex-1")} onClick={() => render("production")}><Film size={13} /> Re-render HD</button>
        </div>
      )}
    </div>
  )
}
