"use client"

import { useEffect, useRef, useState } from "react"
import { Globe, Pencil, ShieldCheck, Eye } from "lucide-react"
import { MCP, type PendingApproval } from "@/services/mcp"
import { Highlighted, pretty } from "@/components/ToolCallCard"
import { Button } from "@/components/ui/button"
import { cn } from "@/services/utils"

function humanize(name: string) {
  const s = name.replace(/[_-]+/g, " ").trim()
  return s.charAt(0).toUpperCase() + s.slice(1)
}

function Chip({ icon, children, tone }: { icon: React.ReactNode; children: React.ReactNode; tone: "neutral" | "warn" }) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs",
        tone === "warn" ? "bg-amber-500/10 text-amber-300" : "bg-secondary text-muted-foreground"
      )}
    >
      {icon}
      {children}
    </span>
  )
}

/**
 * Asks before the assistant runs a connector tool. Sits above the composer instead of covering the
 * chat, so you can still read the conversation while you decide. No answer in time counts as "deny".
 */
export function ApprovalPrompt() {
  const [pending, setPending] = useState<PendingApproval[]>([])
  const [now, setNow] = useState(() => Date.now() / 1000)
  const [showDetails, setShowDetails] = useState<Record<string, boolean>>({})
  const cardRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    let stopped = false
    const poll = async () => {
      try {
        const list = await MCP.approvals()
        if (!stopped) setPending(list)
      } catch {
        if (!stopped) setPending([])
      }
    }
    poll()
    const timer = setInterval(poll, 1500)
    return () => {
      stopped = true
      clearInterval(timer)
    }
  }, [])

  const current = pending[0]

  useEffect(() => {
    if (!current) return
    const tick = setInterval(() => setNow(Date.now() / 1000), 500)
    return () => clearInterval(tick)
  }, [current?.id])

  // Move keyboard focus to the prompt when it appears; Escape declines.
  useEffect(() => {
    if (!current) return
    cardRef.current?.focus()
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") answer("deny")
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [current?.id])

  if (!current) return null

  async function answer(decision: "once" | "always" | "deny") {
    if (!current) return
    const id = current.id
    setPending((p) => p.filter((a) => a.id !== id))
    try {
      await MCP.decide(id, decision)
    } catch {
      /* expired or already answered */
    }
  }

  const total = current.expires_in || 120
  const left = Math.max(0, total - (now - current.created_at))
  const pct = Math.min(100, (left / total) * 100)
  const mins = Math.floor(left / 60)
  const secs = Math.floor(left % 60)

  const request = pretty(current.arguments)
  const noInput = request.trim() === "{}"
  const open = showDetails[current.id] ?? (!noInput && current.arguments.length <= 400)
  const online = current.network !== "none"

  return (
    <div className="pointer-events-none fixed inset-x-0 bottom-0 z-[100] flex justify-center p-4 sm:p-6">
      <div
        ref={cardRef}
        tabIndex={-1}
        role="alertdialog"
        aria-labelledby="approval-title"
        aria-describedby="approval-desc"
        className="pointer-events-auto w-full max-w-xl overflow-hidden rounded-2xl border border-border/80 bg-[#161616] shadow-2xl shadow-black/60 outline-none animate-in fade-in slide-in-from-bottom-4 duration-200"
      >
        <div className="h-0.5 w-full bg-secondary" aria-hidden>
          <div className="h-full bg-primary/70 transition-[width] duration-500 ease-linear" style={{ width: `${pct}%` }} />
        </div>

        <div className="space-y-4 p-5">
          <div className="flex items-start gap-3">
            <div
              aria-hidden
              className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg border border-border bg-secondary text-sm font-semibold"
            >
              {current.connector_name.charAt(0).toUpperCase()}
            </div>
            <div className="min-w-0 flex-1">
              <p className="text-xs text-muted-foreground">{current.connector_name} wants to</p>
              <h2 id="approval-title" className="truncate text-lg font-semibold leading-tight">
                {humanize(current.tool)}
              </h2>
            </div>
            {pending.length > 1 && (
              <span className="shrink-0 rounded-full bg-secondary px-2 py-0.5 text-xs text-muted-foreground">
                1 of {pending.length}
              </span>
            )}
          </div>

          <div id="approval-desc" className="flex flex-wrap gap-2">
            {current.read_only ? (
              <Chip icon={<Eye size={12} />} tone="neutral">Only reads data</Chip>
            ) : (
              <Chip icon={<Pencil size={12} />} tone="neutral">Can change data</Chip>
            )}
            {online ? (
              <Chip icon={<Globe size={12} />} tone="warn">Sends data over the internet</Chip>
            ) : (
              <Chip icon={<ShieldCheck size={12} />} tone="neutral">Stays on this computer</Chip>
            )}
          </div>

          {noInput ? (
            <p className="text-sm text-muted-foreground">This action needs no input.</p>
          ) : (
            <div>
              <button
                onClick={() => setShowDetails((d) => ({ ...d, [current.id]: !open }))}
                aria-expanded={open}
                className="text-sm text-muted-foreground underline-offset-4 hover:text-foreground hover:underline"
              >
                {open ? "Hide details" : "Show details"}
              </button>
              {open && (
                <pre className="mt-2 max-h-56 overflow-auto whitespace-pre-wrap break-words rounded-xl bg-[#1d1d1d] px-4 py-3 font-mono text-[13px] leading-relaxed">
                  <Highlighted text={request} />
                </pre>
              )}
            </div>
          )}

          <div className="flex flex-wrap items-center justify-between gap-3 pt-1">
            <span className="text-xs text-muted-foreground" aria-live="off">
              Declines automatically in {mins}:{String(secs).padStart(2, "0")}
            </span>
            <div className="flex flex-wrap items-center gap-2">
              <Button variant="ghost" onClick={() => answer("deny")}>
                Deny
              </Button>
              <Button
                variant="secondary"
                onClick={() => answer("always")}
                title={`Do not ask again for ${humanize(current.tool)}. You can change this in Settings, Connectors.`}
              >
                Always allow
              </Button>
              <Button onClick={() => answer("once")}>Allow once</Button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
