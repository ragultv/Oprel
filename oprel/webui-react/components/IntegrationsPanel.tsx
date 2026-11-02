'use client'

import React, { useCallback, useEffect, useState } from 'react'
import * as Icons from 'lucide-react'
import {
  Check, ExternalLink, KeyRound, Loader2, Lock, Plug, RefreshCw, ShieldCheck, Trash2, X, AlertTriangle,
} from 'lucide-react'
import {
  IntegrationsAPI, type CatalogEntry, type Connection, type ToolPolicy,
} from '@/services/integrations'
import { cn } from '@/services/utils'

const ICONS: Record<string, React.ElementType> = {
  'azure-devops': (Icons as any).SquareKanban || (Icons as any).KanbanSquare || Plug,
}

function IntegrationIcon({ name, className }: { name: string; className?: string }) {
  const Icon = ICONS[name] || Plug
  return <Icon className={className} size={18} />
}

const STATUS_STYLES: Record<Connection['status'], { label: string; cls: string }> = {
  connected: { label: 'Connected', cls: 'text-emerald-500 bg-emerald-500/10 border-emerald-500/20' },
  error: { label: 'Error', cls: 'text-amber-500 bg-amber-500/10 border-amber-500/20' },
  expired: { label: 'Token expired', cls: 'text-rose-500 bg-rose-500/10 border-rose-500/20' },
  unverified: { label: 'Unverified', cls: 'text-muted-foreground bg-secondary border-border' },
}

// ─── Connect form (rendered from the integration's auth_fields) ───────────────

function ConnectForm({
  entry, connection, onDone, onCancel,
}: {
  entry: CatalogEntry
  connection?: Connection
  onDone: (c: Connection) => void
  onCancel?: () => void
}) {
  const [values, setValues] = useState<Record<string, string>>(() => {
    const initial: Record<string, string> = {}
    entry.auth_fields.forEach(f => {
      initial[f.key] = f.type === 'secret' ? '' : String(connection?.config?.[f.key] ?? '')
    })
    return initial
  })
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const isUpdate = !!connection

  const submit = async (e: React.FormEvent) => {
    e.preventDefault()
    setBusy(true)
    setError(null)
    try {
      // Omitting an empty secret on update keeps the stored one server-side.
      const payload = Object.fromEntries(Object.entries(values).filter(([, v]) => v.trim() !== ''))
      const result = await IntegrationsAPI.connect(entry.id, payload)
      onDone(result)
    } catch (err: any) {
      setError(err.message || 'Connection failed')
    } finally {
      // Never keep the secret around in component state after submitting.
      setValues(prev => {
        const next = { ...prev }
        entry.auth_fields.filter(f => f.type === 'secret').forEach(f => { next[f.key] = '' })
        return next
      })
      setBusy(false)
    }
  }

  return (
    <form onSubmit={submit} className="space-y-4" autoComplete="off">
      {entry.auth_fields.map(field => (
        <div key={field.key} className="space-y-1.5">
          <label className="text-xs font-semibold text-muted-foreground tracking-widest uppercase flex items-center gap-1.5">
            {field.type === 'secret' && <KeyRound size={11} />}
            {field.label}
            {field.required && !(isUpdate && field.type === 'secret') && <span className="text-rose-500">*</span>}
          </label>
          <input
            type={field.type === 'secret' ? 'password' : 'text'}
            name={`oprel-${entry.id}-${field.key}`}
            autoComplete={field.type === 'secret' ? 'new-password' : 'off'}
            spellCheck={false}
            value={values[field.key] ?? ''}
            onChange={e => setValues(v => ({ ...v, [field.key]: e.target.value }))}
            placeholder={
              field.type === 'secret' && isUpdate && connection?.has_secret
                ? `Stored (${connection.secret_hint || 'hidden'}) — leave blank to keep`
                : field.placeholder
            }
            required={field.required && !(isUpdate && field.type === 'secret')}
            className="w-full px-3 py-2 text-sm rounded-lg bg-secondary/60 border border-border focus:border-primary/60 focus:outline-none font-mono"
          />
          {(field.help || field.help_url) && (
            <p className="text-[11px] text-muted-foreground leading-relaxed">
              {field.help}{' '}
              {field.help_url && (
                <a href={field.help_url} target="_blank" rel="noreferrer noopener" className="text-primary hover:underline inline-flex items-center gap-0.5">
                  How to create one <ExternalLink size={10} />
                </a>
              )}
            </p>
          )}
        </div>
      ))}

      <div className="flex items-start gap-2 text-[11px] text-muted-foreground bg-secondary/40 border border-border/50 rounded-lg px-3 py-2.5">
        <ShieldCheck size={14} className="text-emerald-500 shrink-0 mt-px" />
        <span>
          {entry.secure_storage_available
            ? <>Credentials are verified first, then stored in your OS keychain (<span className="font-mono">{entry.secure_storage}</span>). They are never sent back to the browser or shown to the model.</>
            : <span className="text-amber-500">No secure credential store is available on this machine, so connecting is disabled.</span>}
        </span>
      </div>

      {error && (
        <div className="flex items-start gap-2 text-xs text-rose-500 bg-rose-500/10 border border-rose-500/20 rounded-lg px-3 py-2">
          <AlertTriangle size={13} className="shrink-0 mt-px" />
          <span>{error}</span>
        </div>
      )}

      <div className="flex items-center gap-2">
        <button
          type="submit"
          disabled={busy || !entry.secure_storage_available}
          className="flex items-center gap-1.5 px-4 py-2 text-sm rounded-lg bg-primary text-primary-foreground hover:bg-primary/90 disabled:opacity-50 transition-all font-semibold"
        >
          {busy ? <Loader2 size={13} className="animate-spin" /> : <Lock size={13} />}
          {busy ? 'Verifying…' : isUpdate ? 'Save & verify' : 'Connect'}
        </button>
        {onCancel && (
          <button type="button" onClick={onCancel} className="px-3 py-2 text-sm rounded-lg text-muted-foreground hover:text-foreground hover:bg-secondary transition-all">
            Cancel
          </button>
        )}
      </div>
    </form>
  )
}

// ─── Tool policy row ──────────────────────────────────────────────────────────

function ToolRow({
  tool, onChange,
}: {
  tool: ToolPolicy
  onChange: (update: { enabled?: boolean; approval_mode?: 'manual' | 'auto' }) => Promise<void>
}) {
  const [busy, setBusy] = useState(false)

  const apply = async (update: { enabled?: boolean; approval_mode?: 'manual' | 'auto' }) => {
    if (update.approval_mode === 'auto' && tool.kind === 'write') {
      const ok = confirm(
        `Run "${tool.title}" automatically, without asking?\n\n` +
        'The model will be able to change data in this app on its own. Content it reads from the app ' +
        '(e.g. work item descriptions written by others) could try to trick it into actions you did not ask for.\n\n' +
        'A limit of 10 automatic changes per message still applies.',
      )
      if (!ok) return
    }
    setBusy(true)
    try { await onChange(update) } finally { setBusy(false) }
  }

  return (
    <div className={cn('flex items-center gap-3 px-3 py-2.5 rounded-lg border border-border/50', !tool.enabled && 'opacity-60')}>
      <button
        onClick={() => apply({ enabled: !tool.enabled })}
        disabled={busy}
        role="switch"
        aria-checked={tool.enabled}
        title={tool.enabled ? 'Disable tool' : 'Enable tool'}
        className={cn('w-8 h-[18px] rounded-full relative transition-all shrink-0', tool.enabled ? 'bg-primary' : 'bg-secondary border border-border')}
      >
        <span className={cn('absolute top-[2px] w-3.5 h-3.5 rounded-full bg-white transition-all', tool.enabled ? 'left-[15px]' : 'left-[2px]')} />
      </button>
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium text-foreground truncate">{tool.title}</span>
          <span className={cn(
            'px-1.5 py-px rounded text-[9px] font-bold uppercase tracking-wider',
            tool.kind === 'write' ? 'text-amber-500 bg-amber-500/10' : 'text-sky-500 bg-sky-500/10',
          )}>
            {tool.kind === 'write' ? 'Changes data' : 'Read only'}
          </span>
        </div>
        <p className="text-[11px] text-muted-foreground truncate">{tool.description}</p>
      </div>
      <div className="flex items-center bg-secondary/60 border border-border rounded-full p-0.5 gap-0.5 shrink-0">
        {(['manual', 'auto'] as const).map(mode => (
          <button
            key={mode}
            onClick={() => tool.approval_mode !== mode && apply({ approval_mode: mode })}
            disabled={busy || !tool.enabled}
            className={cn(
              'px-2.5 py-0.5 rounded-full text-[10px] font-bold transition-all',
              tool.approval_mode === mode
                ? mode === 'auto' && tool.kind === 'write' ? 'bg-amber-500 text-black' : 'bg-primary text-primary-foreground'
                : 'text-muted-foreground hover:text-foreground',
            )}
            title={mode === 'manual' ? 'Ask me before running' : 'Run without asking'}
          >
            {mode === 'manual' ? 'Ask first' : 'Auto'}
          </button>
        ))}
      </div>
    </div>
  )
}

// ─── Integration card ─────────────────────────────────────────────────────────

function IntegrationCard({
  entry, connection, onChanged, onRemoved,
}: {
  entry: CatalogEntry
  connection?: Connection
  onChanged: (c: Connection) => void
  onRemoved: () => void
}) {
  const [editing, setEditing] = useState(false)
  const [testing, setTesting] = useState(false)
  const [testMsg, setTestMsg] = useState<{ ok: boolean; message: string } | null>(null)

  const test = async () => {
    if (!connection) return
    setTesting(true)
    setTestMsg(null)
    try {
      const r = await IntegrationsAPI.test(connection.id)
      setTestMsg({ ok: r.ok, message: r.message })
      onChanged(r.connection)
    } catch (err: any) {
      setTestMsg({ ok: false, message: err.message })
    } finally {
      setTesting(false)
    }
  }

  const disconnect = async () => {
    if (!connection) return
    if (!confirm(`Disconnect ${entry.name}? The stored token will be deleted from your keychain.`)) return
    await IntegrationsAPI.disconnect(connection.id)
    onRemoved()
  }

  const toggleEnabled = async () => {
    if (!connection) return
    onChanged(await IntegrationsAPI.setEnabled(connection.id, !connection.enabled))
  }

  const status = connection ? STATUS_STYLES[connection.status] || STATUS_STYLES.unverified : null

  return (
    <div className="rounded-xl border border-border bg-secondary/20 overflow-hidden">
      <div className="flex items-start gap-3 p-4">
        <div className="w-10 h-10 rounded-lg bg-primary/10 border border-primary/20 flex items-center justify-center text-primary shrink-0">
          <IntegrationIcon name={entry.icon} />
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <h3 className="text-sm font-bold text-foreground">{entry.name}</h3>
            {status && (
              <span className={cn('px-1.5 py-0.5 rounded border text-[10px] font-bold uppercase tracking-wider', status.cls)}>
                {status.label}
              </span>
            )}
            {connection && !connection.enabled && (
              <span className="px-1.5 py-0.5 rounded border border-border text-[10px] font-bold uppercase tracking-wider text-muted-foreground">
                Paused
              </span>
            )}
          </div>
          <p className="text-xs text-muted-foreground mt-0.5">{entry.description}</p>
          {connection && (
            <div className="flex flex-wrap gap-x-4 gap-y-1 mt-2 text-[11px] text-muted-foreground font-mono">
              {connection.config.organization && <span>org: <span className="text-foreground">{connection.config.organization}</span></span>}
              {connection.config.project && <span>project: <span className="text-foreground">{connection.config.project}</span></span>}
              {connection.has_secret && <span className="flex items-center gap-1"><Lock size={10} /> token {connection.secret_hint}</span>}
            </div>
          )}
          {connection?.status_message && connection.status !== 'connected' && (
            <p className="text-[11px] text-amber-500 mt-1.5">{connection.status_message}</p>
          )}
        </div>
        {connection && !editing && (
          <div className="flex items-center gap-1 shrink-0">
            <button onClick={test} disabled={testing} title="Test connection" className="p-2 rounded-lg text-muted-foreground hover:text-foreground hover:bg-secondary transition-all">
              {testing ? <Loader2 size={14} className="animate-spin" /> : <RefreshCw size={14} />}
            </button>
            <button onClick={() => setEditing(true)} title="Edit / replace token" className="p-2 rounded-lg text-muted-foreground hover:text-foreground hover:bg-secondary transition-all">
              <KeyRound size={14} />
            </button>
            <button onClick={toggleEnabled} title={connection.enabled ? 'Pause integration' : 'Resume integration'} className="p-2 rounded-lg text-muted-foreground hover:text-foreground hover:bg-secondary transition-all">
              {connection.enabled ? <X size={14} /> : <Check size={14} />}
            </button>
            <button onClick={disconnect} title="Disconnect" className="p-2 rounded-lg text-muted-foreground hover:text-rose-500 hover:bg-rose-500/10 transition-all">
              <Trash2 size={14} />
            </button>
          </div>
        )}
      </div>

      {testMsg && (
        <div className={cn('mx-4 mb-3 text-xs rounded-lg px-3 py-2 border', testMsg.ok ? 'text-emerald-500 bg-emerald-500/10 border-emerald-500/20' : 'text-rose-500 bg-rose-500/10 border-rose-500/20')}>
          {testMsg.message}
        </div>
      )}

      {(!connection || editing) && (
        <div className="px-4 pb-4 pt-1 border-t border-border/50">
          <div className="pt-4">
            <ConnectForm
              entry={entry}
              connection={connection}
              onDone={c => { setEditing(false); onChanged(c) }}
              onCancel={connection ? () => setEditing(false) : undefined}
            />
          </div>
        </div>
      )}

      {connection && !editing && (
        <div className="px-4 pb-4 pt-3 border-t border-border/50 space-y-2">
          <p className="text-[10px] font-semibold text-muted-foreground tracking-widest uppercase">Tools available to models</p>
          {connection.tools.map(tool => (
            <ToolRow
              key={tool.name}
              tool={tool}
              onChange={async update => onChanged(await IntegrationsAPI.updateTools(connection.id, [{ name: tool.name, ...update }]))}
            />
          ))}
        </div>
      )}
    </div>
  )
}

// ─── Panel ────────────────────────────────────────────────────────────────────

export default function IntegrationsPanel() {
  const [catalog, setCatalog] = useState<CatalogEntry[]>([])
  const [connections, setConnections] = useState<Connection[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const load = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const [cat, conns] = await Promise.all([IntegrationsAPI.catalog(), IntegrationsAPI.list()])
      setCatalog(cat)
      setConnections(conns)
    } catch (err: any) {
      setError(err.message || 'Failed to load integrations')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { load() }, [load])

  const upsert = (c: Connection) =>
    setConnections(prev => [...prev.filter(p => p.id !== c.id && p.integration_type !== c.integration_type), c])

  return (
    <>
      <div>
        <h1 className="text-xl font-bold text-foreground">Integrations</h1>
        <p className="text-sm text-muted-foreground mt-0.5">
          Connect apps so models can act on your behalf: create tasks, update work items and more.
          Turn tools on per chat with the <span className="font-semibold text-foreground">Tools</span> toggle.
        </p>
      </div>

      {loading && (
        <div className="flex items-center gap-2 text-sm text-muted-foreground py-8 justify-center">
          <Loader2 size={14} className="animate-spin" /> Loading integrations…
        </div>
      )}

      {error && (
        <div className="flex items-center justify-between gap-2 text-xs text-rose-500 bg-rose-500/10 border border-rose-500/20 rounded-lg px-3 py-2">
          <span>{error}</span>
          <button onClick={load} className="underline">Retry</button>
        </div>
      )}

      {!loading && !error && (
        <div className="space-y-4">
          {catalog.map(entry => (
            <IntegrationCard
              key={entry.id}
              entry={entry}
              connection={connections.find(c => c.integration_type === entry.id)}
              onChanged={upsert}
              onRemoved={() => setConnections(prev => prev.filter(c => c.integration_type !== entry.id))}
            />
          ))}
          <p className="text-xs text-muted-foreground bg-secondary/40 rounded-lg px-4 py-3 border border-border/50">
            <strong className="text-foreground">Approvals:</strong> tools that change data ask before running by default.
            Read-only tools run automatically. Every tool run is recorded in the audit log.
          </p>
        </div>
      )}
    </>
  )
}
