"use client"

import { useEffect, useRef, useState } from "react"
import type { AgentEvent } from "@/services/api"
import { cn } from "@/services/utils"
import { card, fmtClock, labelCls } from "./shared"

const dot: Record<AgentEvent["level"], string> = { info: "bg-white/30", warn: "bg-amber-400", error: "bg-red-400" }

export function ActivityLog({ events, live }: { events: AgentEvent[]; live: boolean }) {
  const box = useRef<HTMLDivElement>(null)
  const [stick, setStick] = useState(true)
  const [onlyProblems, setOnlyProblems] = useState(false)
  const shown = onlyProblems ? events.filter((e) => e.level !== "info") : events

  useEffect(() => {
    if (stick && box.current) box.current.scrollTop = box.current.scrollHeight
  }, [shown.length, stick])

  return (
    <div className={cn(card, "flex min-h-0 flex-col")}>
      <div className="flex items-center justify-between border-b border-white/10 px-4 py-3">
        <div className="flex items-center gap-2">
          <span className={labelCls}>Activity</span>
          {live && <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-sky-400" />}
        </div>
        <label className="flex items-center gap-1.5 text-[11px] text-muted-foreground">
          <input type="checkbox" className="accent-[var(--primary)]" checked={onlyProblems} onChange={(e) => setOnlyProblems(e.target.checked)} /> Problems only
        </label>
      </div>
      <div
        ref={box}
        onScroll={(e) => { const el = e.currentTarget; setStick(el.scrollHeight - el.scrollTop - el.clientHeight < 40) }}
        className="max-h-[420px] min-h-[160px] space-y-2.5 overflow-y-auto px-4 py-3"
        role="log" aria-live="polite"
      >
        {shown.length === 0 && <p className="text-xs text-muted-foreground">Nothing yet.</p>}
        {shown.map((e) => (
          <div key={e.seq} className="flex gap-2.5 text-xs">
            <span className={cn("mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full", dot[e.level])} />
            <div className="min-w-0 flex-1">
              <div className={cn("leading-relaxed", e.level === "error" ? "text-red-300" : e.level === "warn" ? "text-amber-200/90" : "text-foreground/85")}>{e.message}</div>
              <div className="font-mono text-[10px] text-muted-foreground/70">{fmtClock(e.ts)}</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
