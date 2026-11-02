'use client'

import React, { useMemo, useState } from 'react'
import {
  AlertTriangle, Check, ChevronDown, ChevronRight, ExternalLink, Loader2, Pencil, ShieldQuestion, X, Wrench,
} from 'lucide-react'
import type { ToolEvent } from '@/services/integrations'
import { cn } from '@/services/utils'

export type ApprovalDecisionHandler = (
  approvalId: string,
  decision: 'approve' | 'reject',
  opts?: { arguments?: Record<string, any>; reason?: string },
) => Promise<void>

interface CallView {
  callId: string
  tool: string
  title: string
  kind: 'read' | 'write'
  arguments: Record<string, any>
  approval?: Extract<ToolEvent, { type: 'approval_required' }>
  call?: Extract<ToolEvent, { type: 'tool_call' }>
  result?: Extract<ToolEvent, { type: 'tool_result' }>
}

function groupCalls(events: ToolEvent[]): { calls: CallView[]; errors: string[] } {
  const byId = new Map<string, CallView>()
  const errors: string[] = []
  for (const ev of events) {
    if (ev.type === 'error') { errors.push(ev.message); continue }
    if (ev.type === 'done') continue
    const view = byId.get(ev.call_id) || {
      callId: ev.call_id,
      tool: ev.tool,
      title: 'title' in ev ? ev.title : ev.tool,
      kind: 'kind' in ev ? ev.kind : 'read',
      arguments: 'arguments' in ev ? ev.arguments : {},
    }
    if (ev.type === 'approval_required') view.approval = ev
    if (ev.type === 'tool_call') { view.call = ev; view.arguments = ev.arguments; view.title = ev.title; view.kind = ev.kind }
    if (ev.type === 'tool_result') view.result = ev
    byId.set(ev.call_id, view)
  }
  return { calls: Array.from(byId.values()), errors }
}

/** Pull a link + label out of a tool result for display (works for any integration returning {url}). */
function resultLink(data: Record<string, any> | undefined): { url: string; label: string } | null {
  if (!data) return null
  for (const value of Object.values(data)) {
    if (value && typeof value === 'object' && typeof value.url === 'string' && /^https:\/\//.test(value.url)) {
      const id = value.id ?? value.work_item_id
      const label = [value.type, id ? `#${id}` : null, value.title].filter(Boolean).join(' ')
      return { url: value.url, label: label || 'Open' }
    }
  }
  return null
}

function argSummary(args: Record<string, any>): string {
  const parts: string[] = []
  if (args.work_item_type) parts.push(String(args.work_item_type))
  if (args.id) parts.push(`#${args.id}`)
  if (args.title) parts.push(`“${args.title}”`)
  if (args.text && !args.title) parts.push(`“${String(args.text).slice(0, 60)}”`)
  return parts.join(' ')
}

const LONG_FIELDS = new Set(['description', 'acceptance_criteria', 'text'])

function ArgumentsTable({ args }: { args: Record<string, any> }) {
  const entries = Object.entries(args)
  if (!entries.length) return <p className="text-[11px] text-muted-foreground">No arguments</p>
  return (
    <div className="space-y-2">
      {entries.map(([key, value]) => (
        <div key={key} className={cn(LONG_FIELDS.has(key) ? 'block' : 'flex gap-2 items-baseline')}>
          <span className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground shrink-0 min-w-[90px] inline-block">
            {key.replace(/_/g, ' ')}
          </span>
          <span className={cn('text-xs text-foreground break-words', LONG_FIELDS.has(key) && 'block mt-1 whitespace-pre-wrap bg-background/60 border border-border/50 rounded-md p-2 font-mono text-[11px] max-h-60 overflow-auto')}>
            {Array.isArray(value) ? value.join(', ') : typeof value === 'object' ? JSON.stringify(value) : String(value)}
          </span>
        </div>
      ))}
    </div>
  )
}

function EditForm({
  args, onSubmit, onCancel, busy,
}: {
  args: Record<string, any>
  onSubmit: (args: Record<string, any>) => void
  onCancel: () => void
  busy: boolean
}) {
  const [draft, setDraft] = useState<Record<string, any>>(() => ({ ...args }))
  return (
    <div className="space-y-2.5">
      {Object.entries(draft).map(([key, value]) => {
        const isList = Array.isArray(value)
        const isLong = LONG_FIELDS.has(key)
        const display = isList ? value.join(', ') : String(value ?? '')
        const update = (raw: string) => setDraft(d => ({
          ...d,
          [key]: isList ? raw.split(',').map(s => s.trim()).filter(Boolean) : typeof args[key] === 'number' ? Number(raw) : raw,
        }))
        return (
          <div key={key} className="space-y-1">
            <label className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">{key.replace(/_/g, ' ')}</label>
            {isLong ? (
              <textarea value={display} onChange={e => update(e.target.value)} rows={6}
                className="w-full px-2.5 py-2 text-xs rounded-md bg-background border border-border focus:border-primary/60 focus:outline-none font-mono" />
            ) : (
              <input value={display} onChange={e => update(e.target.value)}
                type={typeof args[key] === 'number' ? 'number' : 'text'}
                className="w-full px-2.5 py-1.5 text-xs rounded-md bg-background border border-border focus:border-primary/60 focus:outline-none" />
            )}
          </div>
        )
      })}
      <div className="flex gap-2 pt-1">
        <button onClick={() => onSubmit(draft)} disabled={busy}
          className="flex items-center gap-1.5 px-3 py-1.5 text-xs rounded-md bg-primary text-primary-foreground font-semibold hover:bg-primary/90 disabled:opacity-50">
          {busy ? <Loader2 size={12} className="animate-spin" /> : <Check size={12} />} Approve with changes
        </button>
        <button onClick={onCancel} disabled={busy} className="px-3 py-1.5 text-xs rounded-md text-muted-foreground hover:text-foreground hover:bg-secondary">
          Cancel
        </button>
      </div>
    </div>
  )
}

function ApprovalCard({ view, onDecision }: { view: CallView; onDecision?: ApprovalDecisionHandler }) {
  const approval = view.approval!
  const [busy, setBusy] = useState<'approve' | 'reject' | null>(null)
  const [editing, setEditing] = useState(false)
  const [rejecting, setRejecting] = useState(false)
  const [reason, setReason] = useState('')
  const [error, setError] = useState<string | null>(null)

  const decide = async (decision: 'approve' | 'reject', opts?: { arguments?: Record<string, any>; reason?: string }) => {
    if (!onDecision) return
    setBusy(decision)
    setError(null)
    try {
      await onDecision(approval.approval_id, decision, opts)
    } catch (err: any) {
      setError(err.message || 'Failed')
    } finally {
      setBusy(null)
    }
  }

  return (
    <div className="rounded-xl border border-amber-500/30 bg-amber-500/5 overflow-hidden">
      <div className="flex items-center gap-2 px-3 py-2 border-b border-amber-500/20">
        <ShieldQuestion size={14} className="text-amber-500" />
        <span className="text-xs font-semibold text-foreground">Approval needed: {view.title}</span>
        {approval.reason === 'write_limit' && (
          <span className="text-[10px] text-amber-500">(auto-approval limit reached)</span>
        )}
      </div>
      <div className="p-3 space-y-3">
        {editing ? (
          <EditForm args={approval.arguments} busy={busy !== null}
            onSubmit={args => decide('approve', { arguments: args })} onCancel={() => setEditing(false)} />
        ) : (
          <>
            <ArgumentsTable args={approval.arguments} />
            {rejecting && (
              <input value={reason} onChange={e => setReason(e.target.value)} autoFocus
                placeholder="Optional: tell the model why (e.g. wrong project)"
                className="w-full px-2.5 py-1.5 text-xs rounded-md bg-background border border-border focus:border-primary/60 focus:outline-none" />
            )}
            <div className="flex flex-wrap gap-2">
              {!rejecting && (
                <>
                  <button onClick={() => decide('approve')} disabled={busy !== null || !onDecision}
                    className="flex items-center gap-1.5 px-3 py-1.5 text-xs rounded-md bg-primary text-primary-foreground font-semibold hover:bg-primary/90 disabled:opacity-50">
                    {busy === 'approve' ? <Loader2 size={12} className="animate-spin" /> : <Check size={12} />} Approve
                  </button>
                  <button onClick={() => setEditing(true)} disabled={busy !== null || !onDecision}
                    className="flex items-center gap-1.5 px-3 py-1.5 text-xs rounded-md border border-border text-foreground hover:bg-secondary disabled:opacity-50">
                    <Pencil size={12} /> Edit
                  </button>
                </>
              )}
              <button
                onClick={() => (rejecting ? decide('reject', { reason: reason.trim() || undefined }) : setRejecting(true))}
                disabled={busy !== null || !onDecision}
                className="flex items-center gap-1.5 px-3 py-1.5 text-xs rounded-md border border-rose-500/30 text-rose-500 hover:bg-rose-500/10 disabled:opacity-50">
                {busy === 'reject' ? <Loader2 size={12} className="animate-spin" /> : <X size={12} />} {rejecting ? 'Confirm reject' : 'Reject'}
              </button>
              {rejecting && (
                <button onClick={() => setRejecting(false)} className="px-3 py-1.5 text-xs rounded-md text-muted-foreground hover:text-foreground">
                  Back
                </button>
              )}
            </div>
          </>
        )}
        {error && <p className="text-[11px] text-rose-500">{error}</p>}
      </div>
    </div>
  )
}

function CallCard({ view }: { view: CallView }) {
  const [open, setOpen] = useState(false)
  const result = view.result
  const running = !result
  const rejected = result?.rejected
  const link = result?.ok ? resultLink(result.data) : null
  const errorText = result && !result.ok && !rejected ? String(result.data?.error || 'Failed') : null

  return (
    <div className="rounded-lg border border-border/60 bg-secondary/30 text-xs">
      <button onClick={() => setOpen(o => !o)} className="w-full flex items-center gap-2 px-3 py-2 text-left">
        {running ? <Loader2 size={13} className="animate-spin text-primary shrink-0" />
          : rejected ? <X size={13} className="text-muted-foreground shrink-0" />
          : result.ok ? <Check size={13} className="text-emerald-500 shrink-0" />
          : <AlertTriangle size={13} className="text-rose-500 shrink-0" />}
        <span className="font-medium text-foreground shrink-0">{view.title}</span>
        <span className="text-muted-foreground truncate flex-1">
          {rejected ? 'Rejected' : argSummary(view.arguments)}
        </span>
        {view.call?.approval === 'manual' && !rejected && (
          <span className="text-[9px] uppercase tracking-wider text-muted-foreground shrink-0">approved</span>
        )}
        {view.call?.approval === 'auto' && view.kind === 'write' && (
          <span className="text-[9px] uppercase tracking-wider text-amber-500 shrink-0">auto</span>
        )}
        {open ? <ChevronDown size={12} className="text-muted-foreground shrink-0" /> : <ChevronRight size={12} className="text-muted-foreground shrink-0" />}
      </button>
      {link && (
        <a href={link.url} target="_blank" rel="noreferrer noopener"
          className="flex items-center gap-1.5 mx-3 mb-2 text-primary hover:underline">
          <ExternalLink size={11} /> {link.label}
        </a>
      )}
      {errorText && <p className="mx-3 mb-2 text-rose-500">{errorText}</p>}
      {open && (
        <div className="px-3 pb-3 pt-1 border-t border-border/50 space-y-2">
          <ArgumentsTable args={view.arguments} />
          {result && (
            <pre className="text-[10px] font-mono bg-background/60 border border-border/50 rounded-md p-2 overflow-auto max-h-48">
              {JSON.stringify(result.data, null, 2)}
            </pre>
          )}
        </div>
      )}
    </div>
  )
}

export default function ToolActivity({
  events, onDecision,
}: {
  events: ToolEvent[]
  onDecision?: ApprovalDecisionHandler
}) {
  const { calls, errors } = useMemo(() => groupCalls(events), [events])
  if (!calls.length && !errors.length) return null
  return (
    <div className="space-y-2 mb-3">
      <div className="flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
        <Wrench size={11} /> Tools
      </div>
      {calls.map(view =>
        view.approval && !view.call && !view.result
          ? <ApprovalCard key={view.callId} view={view} onDecision={onDecision} />
          : <CallCard key={view.callId} view={view} />,
      )}
      {errors.map((e, i) => (
        <div key={i} className="flex items-start gap-2 text-xs text-rose-500 bg-rose-500/10 border border-rose-500/20 rounded-lg px-3 py-2">
          <AlertTriangle size={13} className="shrink-0 mt-px" /> <span>{e}</span>
        </div>
      ))}
    </div>
  )
}
