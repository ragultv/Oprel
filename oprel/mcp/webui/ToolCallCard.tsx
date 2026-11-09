"use client"

import { useState } from "react"
import { Ban, ChevronDown, ChevronRight, Hand, TriangleAlert, Wrench } from "lucide-react"
import { cn } from "@/services/utils"

export interface ToolCall {
  connector: string
  tool: string
  title: string
  request: string
  result: string
  status: "ok" | "error" | "denied" | "blocked" | "invalid"
}

const TAG_OPEN = "<toolcall>"
const TAG_CLOSE = "</toolcall>"

export type Segment =
  | { type: "text"; text: string }
  | { type: "tool"; call: ToolCall }
  | { type: "tools"; calls: ToolCall[] }

function decode(b64: string): ToolCall | null {
  try {
    const c = JSON.parse(atob(b64.trim()))
    if (c && typeof c.tool === "string") return c as ToolCall
  } catch {
    /* ignore a malformed card */
  }
  return null
}

/** Split a reply into text and tool calls, in order. An unfinished card is hidden until complete. */
export function splitToolCalls(content: string): Segment[] {
  const out: Segment[] = []
  let rest = content
  while (rest.length) {
    const start = rest.indexOf(TAG_OPEN)
    if (start === -1) {
      out.push({ type: "text", text: rest })
      break
    }
    if (start > 0) out.push({ type: "text", text: rest.slice(0, start) })
    const end = rest.indexOf(TAG_CLOSE, start + TAG_OPEN.length)
    if (end === -1) break // still streaming
    const call = decode(rest.slice(start + TAG_OPEN.length, end))
    if (call) out.push({ type: "tool", call })
    rest = rest.slice(end + TAG_CLOSE.length)
  }
  return out
}

/** Merge tool calls that follow each other (blank text between them is ignored) into one group. */
export function groupToolCalls(segments: Segment[]): Segment[] {
  const out: Segment[] = []
  for (const seg of segments) {
    const last = out[out.length - 1]
    if (seg.type === "tool") {
      if (last && last.type === "tools") last.calls.push(seg.call)
      else out.push({ type: "tools", calls: [seg.call] })
    } else if (seg.type === "text" && !seg.text.trim() && last && last.type === "tools") {
      continue
    } else {
      out.push(seg)
    }
  }
  return out
}

function parse(text: string): unknown {
  try {
    return JSON.parse(text)
  } catch {
    return undefined
  }
}

export function pretty(text: string): string {
  const v = parse(text)
  return v === undefined ? text : JSON.stringify(v, null, 2)
}

/** One short line describing the request, e.g. "San Francisco 37.77". */
function summarize(call: ToolCall): string {
  const v = parse(call.request)
  if (!v || typeof v !== "object") return ""
  const parts: string[] = []
  const walk = (x: unknown) => {
    if (parts.length >= 2) return
    if (typeof x === "string" || typeof x === "number") parts.push(String(x))
    else if (Array.isArray(x)) x.forEach(walk)
    else if (x && typeof x === "object") Object.values(x).forEach(walk)
  }
  walk(v)
  const line = parts.join(" · ")
  return line.length > 48 ? line.slice(0, 47) + "…" : line
}

const STATUS_TEXT: Record<ToolCall["status"], string> = {
  ok: "Done",
  error: "Failed",
  denied: "You declined",
  blocked: "Blocked",
  invalid: "Invalid request",
}

// Syntax colours for JSON (dark theme). Works on partial JSON too, so a truncated result still colours.
const TOKEN_RE = /("(?:\\.|[^"\\])*")(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?/g

export function Highlighted({ text }: { text: string }) {
  const nodes: React.ReactNode[] = []
  let last = 0
  let i = 0
  for (const m of text.matchAll(TOKEN_RE)) {
    const at = m.index ?? 0
    if (at > last) nodes.push(text.slice(last, at))
    const [token, str, colon, literal] = m
    let color = "#b5cea8" // numbers
    if (str) color = colon ? "#9cdcfe" : "#ce9178" // keys, then string values
    else if (literal) color = "#c586c0" // true / false / null
    nodes.push(
      <span key={i++} style={{ color }}>
        {str && colon ? str : token}
      </span>
    )
    if (str && colon) nodes.push(colon)
    last = at + token.length
  }
  if (last < text.length) nodes.push(text.slice(last))
  return <>{nodes}</>
}

function CodeBlock({ label, text }: { label: string; text: string }) {
  const isJson = parse(text) !== undefined || /^\s*[\[{]/.test(text)
  return (
    <div className="space-y-1.5">
      <p className="text-xs text-muted-foreground">{label}</p>
      <pre className="max-h-72 overflow-auto whitespace-pre-wrap break-words rounded-xl bg-[#1b1b1b] px-4 py-3 font-mono text-[13px] leading-relaxed text-foreground/90">
        {isJson ? <Highlighted text={text} /> : text}
      </pre>
    </div>
  )
}

function StatusIcon({ status }: { status: ToolCall["status"] }) {
  const common = { size: 15, "aria-label": STATUS_TEXT[status] } as const
  if (status === "denied") return <Hand {...common} className="text-muted-foreground" />
  if (status === "blocked") return <Ban {...common} className="text-muted-foreground" />
  if (status === "error") return <TriangleAlert {...common} className="text-red-400" />
  if (status === "invalid") return <TriangleAlert {...common} className="text-amber-400" />
  return <Wrench {...common} className="text-muted-foreground" />
}

function ToolRow({ call }: { call: ToolCall }) {
  const [open, setOpen] = useState(false)
  const request = pretty(call.request)
  const noInput = request.trim() === "{}"
  const summary = summarize(call)

  return (
    <li>
      <button
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        title={STATUS_TEXT[call.status]}
        className="flex w-full items-center gap-3 rounded-lg py-1.5 text-left text-sm hover:text-foreground"
      >
        <StatusIcon status={call.status} />
        <span className="font-mono text-[15px] text-foreground">{call.tool}</span>
        {summary && <span className="min-w-0 flex-1 truncate text-muted-foreground">{summary}</span>}
        {!summary && <span className="flex-1" />}
        {call.status !== "ok" && <span className="shrink-0 text-xs text-muted-foreground">{STATUS_TEXT[call.status]}</span>}
        {open ? <ChevronDown size={15} className="shrink-0 text-muted-foreground" /> : <ChevronRight size={15} className="shrink-0 text-muted-foreground" />}
      </button>
      {open && (
        <div className="ml-7 space-y-4 pb-3 pt-1">
          <CodeBlock label="Parameters" text={noInput ? "No parameters" : request} />
          <CodeBlock label="Result" text={pretty(call.result)} />
        </div>
      )}
    </li>
  )
}

/** Tool calls of one step in a reply: "Used Memory" for one call, "3 tool calls" for several. */
export function ToolCallGroup({ calls }: { calls: ToolCall[] }) {
  const [open, setOpen] = useState(false)
  const servers = Array.from(new Set(calls.map((c) => c.connector)))
  const label = calls.length === 1 ? `Used ${servers[0]}` : `${calls.length} tool calls`
  const failed = calls.some((c) => c.status !== "ok")

  return (
    <div className="my-2">
      <button
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        className="flex items-center gap-1.5 text-[15px] text-muted-foreground transition-colors hover:text-foreground"
      >
        <span>{label}</span>
        {calls.length > 1 && <span className="text-muted-foreground/70">&middot; {servers.join(", ")}</span>}
        {failed && !open && <TriangleAlert size={13} className="text-amber-400" aria-label="Some calls did not complete" />}
        {open ? <ChevronDown size={15} /> : <ChevronRight size={15} />}
      </button>
      {open && (
        <ul className={cn("mt-1 space-y-0.5", calls.length > 1 ? "ml-1" : "ml-1")}>
          {calls.map((c, i) => (
            <ToolRow key={i} call={c} />
          ))}
        </ul>
      )}
    </div>
  )
}
