"use client"

import { AlertTriangle, CheckCircle2, Clock, Loader2, PauseCircle, XCircle } from "lucide-react"
import type { AgentStatus, SceneStatus } from "@/services/api"
import { cn } from "@/services/utils"

export const labelCls = "text-[10px] font-semibold uppercase tracking-widest text-muted-foreground"
export const inputCls =
  "w-full rounded-xl border border-white/10 bg-white/5 px-3.5 py-2.5 text-sm outline-none transition-colors placeholder:text-muted-foreground/60 focus:border-primary/50 disabled:opacity-50"
export const primaryBtn =
  "inline-flex items-center justify-center gap-2 rounded-full bg-primary px-5 py-2.5 text-sm font-semibold text-primary-foreground shadow-lg shadow-primary/20 transition hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-40 disabled:shadow-none"
export const ghostBtn =
  "inline-flex items-center justify-center gap-1.5 rounded-full border border-white/10 bg-white/5 px-4 py-2 text-xs font-semibold transition hover:bg-white/10 disabled:cursor-not-allowed disabled:opacity-40"
export const card = "rounded-2xl border border-white/10 bg-white/[0.025]"

type Tone = "neutral" | "active" | "good" | "warn" | "bad"
const toneCls: Record<Tone, string> = {
  neutral: "border-white/10 bg-white/5 text-muted-foreground",
  active: "border-sky-500/30 bg-sky-500/10 text-sky-400",
  good: "border-emerald-500/25 bg-emerald-500/10 text-emerald-400",
  warn: "border-amber-500/25 bg-amber-500/10 text-amber-400",
  bad: "border-red-500/25 bg-red-500/10 text-red-400",
}

const RUN_META: Record<AgentStatus, { label: string; tone: Tone; spin?: boolean }> = {
  planning: { label: "Planning", tone: "active", spin: true },
  awaiting_review: { label: "Review plan", tone: "warn" },
  narrating: { label: "Recording narration", tone: "active", spin: true },
  writing: { label: "Writing scenes", tone: "active", spin: true },
  assembling: { label: "Assembling", tone: "active", spin: true },
  rendering: { label: "Rendering", tone: "active", spin: true },
  completed: { label: "Complete", tone: "good" },
  needs_attention: { label: "Needs attention", tone: "warn" },
  failed: { label: "Failed", tone: "bad" },
  cancelled: { label: "Stopped", tone: "neutral" },
  interrupted: { label: "Interrupted", tone: "warn" },
}

const SCENE_META: Record<SceneStatus, { label: string; tone: Tone; spin?: boolean }> = {
  pending: { label: "Waiting", tone: "neutral" },
  writing: { label: "Writing", tone: "active", spin: true },
  checking: { label: "Checking", tone: "active", spin: true },
  fixing: { label: "Fixing", tone: "warn", spin: true },
  done: { label: "Ready", tone: "good" },
  failed: { label: "Needs attention", tone: "bad" },
}

function Chip({ tone, spin, label, icon }: { tone: Tone; spin?: boolean; label: string; icon?: "ok" | "warn" | "stop" | "wait" }) {
  const Icon = spin ? Loader2 : icon === "ok" ? CheckCircle2 : icon === "warn" ? AlertTriangle : icon === "stop" ? XCircle : tone === "neutral" ? PauseCircle : Clock
  return (
    <span className={cn("inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[11px] font-semibold", toneCls[tone])}>
      <Icon size={12} className={spin ? "animate-spin" : undefined} />
      {label}
    </span>
  )
}

export function RunChip({ status }: { status: AgentStatus }) {
  const m = RUN_META[status]
  return <Chip tone={m.tone} spin={m.spin} label={m.label} icon={m.tone === "good" ? "ok" : m.tone === "bad" ? "stop" : m.tone === "warn" ? "warn" : undefined} />
}

export function SceneChip({ status }: { status: SceneStatus }) {
  const m = SCENE_META[status]
  return <Chip tone={m.tone} spin={m.spin} label={m.label} icon={m.tone === "good" ? "ok" : m.tone === "bad" ? "stop" : undefined} />
}

export const ACTIVE_RUN = new Set<AgentStatus>(["planning", "narrating", "writing", "assembling", "rendering"])

export function fmtElapsed(seconds: number): string {
  const s = Math.max(0, Math.floor(seconds))
  const h = Math.floor(s / 3600)
  const m = Math.floor((s % 3600) / 60)
  const sec = s % 60
  return h > 0 ? `${h}h ${String(m).padStart(2, "0")}m` : m > 0 ? `${m}m ${String(sec).padStart(2, "0")}s` : `${sec}s`
}

export function timeAgo(tsSeconds: number): string {
  const d = Date.now() / 1000 - tsSeconds
  if (d < 60) return "just now"
  if (d < 3600) return `${Math.floor(d / 60)}m ago`
  if (d < 86400) return `${Math.floor(d / 3600)}h ago`
  return `${Math.floor(d / 86400)}d ago`
}

export function fmtClock(tsSeconds: number): string {
  return new Date(tsSeconds * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })
}

/** Parameter count in billions parsed from strings like "7B" / "1.2B"; null when unknown (cloud models etc.). */
export function paramsBillions(text: string | undefined): number | null {
  const m = /(?:^|[^a-z\d.])(\d+(?:\.\d+)?)\s*b(?![a-z])/i.exec(text ?? "")
  return m ? parseFloat(m[1]) : null
}
