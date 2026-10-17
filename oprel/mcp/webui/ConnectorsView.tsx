"use client"

import { useCallback, useEffect, useMemo, useState } from "react"
import { Ban, Check, ChevronDown, ChevronRight, Hand, Plus, Search, ShieldCheck } from "lucide-react"
import {
  MCP,
  type CallLog,
  type CatalogEntry,
  type Connector,
  type ConnectorDetail,
  type ConnectorTool,
  type PermissionMode,
} from "@/services/mcp"
import { Button } from "@/components/ui/button"
import { cn } from "@/services/utils"

const MODE_LABEL: Record<PermissionMode, string> = {
  allow: "Always allow",
  ask: "Needs approval",
  deny: "Blocked",
}

function networkLabel(network: string) {
  return network === "none" ? "Works only on this computer" : "Uses the internet. Data can leave this computer."
}

function humanize(name: string) {
  const s = name.replace(/[_-]+/g, " ").trim()
  return s.charAt(0).toUpperCase() + s.slice(1)
}

function Avatar({ name }: { name: string }) {
  return (
    <div
      aria-hidden
      className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg border border-border bg-secondary text-sm font-semibold"
    >
      {name.charAt(0).toUpperCase()}
    </div>
  )
}

type View = { kind: "list" } | { kind: "catalog"; id: string } | { kind: "connector"; id: string }

export function ConnectorsView({ embedded = false }: { embedded?: boolean }) {
  const [tab, setTab] = useState<"yours" | "discover">("yours")
  const [query, setQuery] = useState("")
  const [view, setView] = useState<View>({ kind: "list" })
  const [catalog, setCatalog] = useState<CatalogEntry[]>([])
  const [connectors, setConnectors] = useState<Connector[]>([])
  const [logs, setLogs] = useState<CallLog[]>([])
  const [loaded, setLoaded] = useState(false)
  const [error, setError] = useState("")

  const refresh = useCallback(async () => {
    try {
      const [c, k, l] = await Promise.all([MCP.catalog(), MCP.connectors(), MCP.logs(30)])
      setCatalog(c)
      setConnectors(k)
      setLogs(l)
      return k
    } catch (e: any) {
      setError(e?.message ?? "Could not load connectors.")
      return null
    }
  }, [])

  useEffect(() => {
    refresh().then((k) => {
      setLoaded(true)
      if (k && k.length === 0) setTab("discover")
    })
  }, [refresh])

  const q = query.trim().toLowerCase()
  const shownCatalog = useMemo(
    () => catalog.filter((c) => !q || c.name.toLowerCase().includes(q) || c.description.toLowerCase().includes(q)),
    [catalog, q]
  )
  const shownConnectors = useMemo(
    () => connectors.filter((c) => !q || c.name.toLowerCase().includes(q)),
    [connectors, q]
  )

  if (view.kind === "catalog") {
    const entry = catalog.find((c) => c.id === view.id)
    if (entry) {
      return (
        <CatalogDetail
          embedded={embedded}
          entry={entry}
          onBack={() => setView({ kind: "list" })}
          onInstalled={async (id) => {
            await refresh()
            setView({ kind: "connector", id })
          }}
        />
      )
    }
  }

  if (view.kind === "connector") {
    return (
      <ConnectorDetailView
        embedded={embedded}
        id={view.id}
        logs={logs.filter((l) => l.connector_id === view.id)}
        onBack={() => {
          refresh()
          setView({ kind: "list" })
        }}
        onChanged={refresh}
      />
    )
  }

  return (
    <div className={cn(embedded ? "" : "h-full overflow-y-auto bg-[#0a0a0a] p-6")}>
      <div className={cn("space-y-6", embedded ? "" : "mx-auto max-w-3xl")}>
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div role="tablist" aria-label="Connectors" className="flex rounded-lg border border-border p-0.5">
            {(["yours", "discover"] as const).map((t) => (
              <button
                key={t}
                role="tab"
                aria-selected={tab === t}
                onClick={() => setTab(t)}
                className={cn(
                  "rounded-md px-3 py-1.5 text-sm font-medium transition-colors",
                  tab === t ? "bg-secondary text-foreground" : "text-muted-foreground hover:text-foreground"
                )}
              >
                {t === "yours" ? "Yours" : "Discover"}
              </button>
            ))}
          </div>
          <label className="relative block min-w-[200px] flex-1 sm:max-w-xs">
            <span className="sr-only">Search connectors</span>
            <Search size={14} className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground" />
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search"
              className="w-full rounded-lg border border-border bg-background py-2 pl-8 pr-3 text-sm"
            />
          </label>
        </div>

        {error && (
          <p role="alert" className="text-sm text-red-400">
            {error}
          </p>
        )}

        {tab === "yours" && (
          <section aria-label="Your connectors" className="space-y-3">
            {loaded && shownConnectors.length === 0 && (
              <p className="text-sm text-muted-foreground">
                {connectors.length === 0 ? "You have not connected anything yet. Open Discover to add a connector." : "No connectors match your search."}
              </p>
            )}
            {shownConnectors.map((c) => (
              <button
                key={c.id}
                onClick={() => setView({ kind: "connector", id: c.id })}
                className="flex w-full items-center gap-3 rounded-xl border border-border p-4 text-left transition-colors hover:bg-secondary/40"
              >
                <Avatar name={c.name} />
                <div className="min-w-0 flex-1">
                  <p className="font-medium">{c.name}</p>
                  <p className="truncate text-sm text-muted-foreground">{c.description || networkLabel(c.network)}</p>
                </div>
                <span
                  className={cn(
                    "shrink-0 rounded-full px-2 py-0.5 text-xs",
                    c.status === "connected" && c.enabled ? "bg-green-500/15 text-green-400" : "bg-amber-500/15 text-amber-400"
                  )}
                >
                  {!c.enabled ? "Off" : c.status === "connected" ? "Connected" : c.status}
                </span>
                <ChevronRight size={16} className="text-muted-foreground" />
              </button>
            ))}

            {logs.length > 0 && !q && (
              <div className="space-y-2 pt-4">
                <h2 className="text-sm font-medium">Recent activity</h2>
                <ul className="space-y-1 text-xs text-muted-foreground">
                  {logs.slice(0, 10).map((l) => (
                    <li key={l.id}>
                      {new Date(l.ts).toLocaleString()} &middot; {humanize(l.tool)} &middot; {l.decision}
                      {l.ok === 0 ? " · failed" : ""}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </section>
        )}

        {tab === "discover" && (
          <section aria-label="Discover connectors" className="space-y-3">
            <h2 className="text-sm font-medium text-muted-foreground">Default connectors</h2>
            {shownCatalog.length === 0 && <p className="text-sm text-muted-foreground">No connectors match your search.</p>}
            <div className="grid gap-3 sm:grid-cols-2">
              {shownCatalog.map((entry) => (
                <button
                  key={entry.id}
                  onClick={() =>
                    entry.installed_id ? setView({ kind: "connector", id: entry.installed_id }) : setView({ kind: "catalog", id: entry.id })
                  }
                  className="flex items-start gap-3 rounded-xl border border-border p-4 text-left transition-colors hover:bg-secondary/40"
                >
                  <Avatar name={entry.name} />
                  <div className="min-w-0 flex-1">
                    <p className="font-medium">{entry.name}</p>
                    <p className="text-sm text-muted-foreground">{entry.description}</p>
                    <p className="mt-1 text-xs text-muted-foreground">{entry.tools.length} tools</p>
                  </div>
                  <span
                    aria-label={entry.installed_id ? "Connected" : "Not connected"}
                    className={cn(
                      "flex h-7 w-7 shrink-0 items-center justify-center rounded-md border",
                      entry.installed_id ? "border-green-500/40 bg-green-500/10 text-green-400" : "border-border text-muted-foreground"
                    )}
                  >
                    {entry.installed_id ? <Check size={14} /> : <Plus size={14} />}
                  </span>
                </button>
              ))}
            </div>
          </section>
        )}
      </div>
    </div>
  )
}

function BackLink({ onBack, label }: { onBack: () => void; label: string }) {
  return (
    <button onClick={onBack} className="mb-4 flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground">
      <ChevronRight size={14} className="rotate-180" />
      {label}
    </button>
  )
}

function Chips({ items, title }: { items: { name: string; description?: string }[]; title: string }) {
  return (
    <section className="space-y-2" aria-label={title}>
      <h2 className="text-lg font-medium">{title}</h2>
      <ul className="flex flex-wrap gap-2">
        {items.map((i) => (
          <li
            key={i.name}
            title={i.description || undefined}
            className="rounded-md bg-secondary px-2.5 py-1 font-mono text-xs text-foreground/90"
          >
            {i.name}
          </li>
        ))}
      </ul>
    </section>
  )
}

/** A connector that is not connected yet: what it is, what it can do, and a Connect button. */
function CatalogDetail({
  entry,
  onBack,
  onInstalled,
  embedded,
}: {
  entry: CatalogEntry
  onBack: () => void
  onInstalled: (id: string) => void
  embedded: boolean
}) {
  const [folder, setFolder] = useState("")
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState("")
  const needsFolder = !!entry.needs_paths
  const canConnect = !busy && (!needsFolder || folder.trim().length > 0)

  const connect = async () => {
    setBusy(true)
    setError("")
    try {
      const c = await MCP.install(entry.id, needsFolder ? [folder.trim()] : [])
      onInstalled(c.id)
    } catch (e: any) {
      setError(e?.message ?? "Could not connect.")
      setBusy(false)
    }
  }

  return (
    <div className={cn(embedded ? "" : "h-full overflow-y-auto bg-[#0a0a0a] p-6")}>
      <div className={cn("space-y-6", embedded ? "" : "mx-auto max-w-3xl")}>
        <BackLink onBack={onBack} label="All connectors" />
        <header className="flex items-start justify-between gap-4 rounded-xl border border-border p-5">
          <div className="flex items-center gap-4">
            <Avatar name={entry.name} />
            <div>
              <h1 className="text-xl font-semibold">{entry.name}</h1>
              <p className="text-sm text-muted-foreground">{entry.description}</p>
            </div>
          </div>
          <Button onClick={connect} disabled={!canConnect}>
            {busy ? "Connecting..." : "Connect"}
          </Button>
        </header>

        {needsFolder && (
          <label className="block text-sm text-muted-foreground">
            Folder it may use
            <input
              value={folder}
              onChange={(e) => setFolder(e.target.value)}
              placeholder="C:\Users\you\Documents\Notes"
              className="mt-1 w-full rounded-md border border-border bg-background px-3 py-2 text-sm text-foreground"
            />
            <span className="mt-1 block text-xs">The assistant can only reach files inside this folder.</span>
          </label>
        )}

        {error && (
          <p role="alert" className="text-sm text-red-400">
            {error}
          </p>
        )}

        <p className="flex items-center gap-2 text-sm text-muted-foreground">
          <ShieldCheck size={16} />
          {networkLabel(entry.network)} Every action asks for your permission unless you allow it.
        </p>

        {entry.tools.length > 0 && <Chips title="Tools" items={entry.tools} />}
        {entry.prompts.length > 0 && <Chips title="Prompts" items={entry.prompts} />}
        {entry.checked_version && (
          <p className="text-xs text-muted-foreground">
            Tool list checked against version {entry.checked_version}. After connecting, the live list from the server is shown.
          </p>
        )}
      </div>
    </div>
  )
}

function ModeButtons({
  mode,
  onChange,
  label,
}: {
  mode: PermissionMode
  onChange: (m: PermissionMode) => void
  label: string
}) {
  const options: { m: PermissionMode; icon: React.ReactNode; text: string }[] = [
    { m: "allow", icon: <Check size={14} />, text: "Always allow" },
    { m: "ask", icon: <Hand size={14} />, text: "Needs approval" },
    { m: "deny", icon: <Ban size={14} />, text: "Blocked" },
  ]
  return (
    <div role="radiogroup" aria-label={label} className="flex rounded-lg border border-border p-0.5">
      {options.map((o) => (
        <button
          key={o.m}
          role="radio"
          aria-checked={mode === o.m}
          aria-label={`${label}: ${o.text}`}
          title={o.text}
          onClick={() => onChange(o.m)}
          className={cn(
            "flex h-7 w-8 items-center justify-center rounded-md transition-colors",
            mode === o.m ? "bg-secondary text-foreground" : "text-muted-foreground hover:text-foreground"
          )}
        >
          {o.icon}
        </button>
      ))}
    </div>
  )
}

function groupMode(tools: ConnectorTool[]): PermissionMode | "mixed" {
  const modes = new Set(tools.map((t) => t.mode))
  return modes.size === 1 ? (tools[0].mode as PermissionMode) : "mixed"
}

/** A connected connector: about, prompts and per-tool permissions grouped like read-only vs write/delete. */
function ConnectorDetailView({
  id,
  logs,
  onBack,
  onChanged,
  embedded,
}: {
  id: string
  logs: CallLog[]
  onBack: () => void
  onChanged: () => void
  embedded: boolean
}) {
  const [detail, setDetail] = useState<ConnectorDetail | null>(null)
  const [open, setOpen] = useState<{ read: boolean; write: boolean }>({ read: false, write: true })
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState("")

  const load = useCallback(async () => {
    try {
      setDetail(await MCP.detail(id))
    } catch (e: any) {
      setError(e?.message ?? "Could not load this connector.")
    }
  }, [id])

  useEffect(() => {
    load()
  }, [load])

  const act = async (fn: () => Promise<unknown>, after?: () => void) => {
    setBusy(true)
    setError("")
    try {
      await fn()
      after ? after() : await load()
      onChanged()
    } catch (e: any) {
      setError(e?.message ?? "Something went wrong.")
    }
    setBusy(false)
  }

  const setMode = (target: { tool: string } | { group: "read" | "write" }, mode: PermissionMode) =>
    act(async () => {
      const res = await MCP.setPermission(id, target, mode)
      setDetail((d) => (d ? { ...d, tools: res.tools } : d))
    }, () => undefined)

  if (!detail) {
    return (
      <div className={cn(embedded ? "" : "p-6")}>
        <BackLink onBack={onBack} label="All connectors" />
        <p className={cn("text-sm", error ? "text-red-400" : "text-muted-foreground")} role={error ? "alert" : "status"}>
          {error || "Loading..."}
        </p>
      </div>
    )
  }

  const readTools = detail.tools.filter((t) => t.read_only)
  const writeTools = detail.tools.filter((t) => !t.read_only)
  const running = detail.enabled && detail.status === "connected"

  const renderGroup = (kind: "read" | "write", tools: ConnectorTool[], title: string) => {
    if (tools.length === 0) return null
    const mode = groupMode(tools)
    const expanded = open[kind]
    return (
      <div key={kind} className="border-b border-border last:border-b-0">
        <div className="flex items-center justify-between gap-3 py-3">
          <button
            onClick={() => setOpen((o) => ({ ...o, [kind]: !o[kind] }))}
            aria-expanded={expanded}
            className="flex items-center gap-2 text-sm font-medium"
          >
            {expanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
            {title}
            <span className="rounded-full bg-secondary px-2 py-0.5 text-xs text-muted-foreground">{tools.length}</span>
          </button>
          <label className="flex items-center gap-2 text-xs text-muted-foreground">
            <span className="sr-only">{title} permission</span>
            <select
              value={mode}
              disabled={busy}
              onChange={(e) => setMode({ group: kind }, e.target.value as PermissionMode)}
              className="rounded-lg border border-border bg-background px-2 py-1.5 text-sm text-foreground"
            >
              {mode === "mixed" && <option value="mixed" disabled>Custom</option>}
              {(Object.keys(MODE_LABEL) as PermissionMode[]).map((m) => (
                <option key={m} value={m}>
                  {MODE_LABEL[m]}
                </option>
              ))}
            </select>
          </label>
        </div>
        {expanded && (
          <ul className="pb-2">
            {tools.map((t) => (
              <li key={t.name} className="flex items-center justify-between gap-3 py-2 pl-6">
                <div className="min-w-0">
                  <p className="text-sm">{humanize(t.name)}</p>
                  {t.description && <p className="truncate text-xs text-muted-foreground" title={t.description}>{t.description}</p>}
                </div>
                <ModeButtons mode={t.mode} label={humanize(t.name)} onChange={(m) => setMode({ tool: t.name }, m)} />
              </li>
            ))}
          </ul>
        )}
      </div>
    )
  }

  return (
    <div className={cn(embedded ? "" : "h-full overflow-y-auto bg-[#0a0a0a] p-6")}>
      <div className={cn("space-y-6", embedded ? "" : "mx-auto max-w-3xl")}>
        <BackLink onBack={onBack} label="All connectors" />
        <header className="flex flex-wrap items-start justify-between gap-4">
          <div className="flex items-center gap-4">
            <Avatar name={detail.name} />
            <div>
              <h1 className="text-xl font-semibold">{detail.name}</h1>
              <p className="text-xs text-muted-foreground">
                v{detail.version ?? "?"} &middot;{" "}
                <span className={running ? "text-green-400" : "text-amber-400"}>
                  {!detail.enabled ? "Off" : detail.status === "connected" ? "Connected" : detail.status}
                </span>
              </p>
            </div>
          </div>
          <div className="flex gap-2">
            <Button
              variant="secondary"
              disabled={busy}
              onClick={() => act(() => MCP.patch(id, { enabled: !detail.enabled }))}
            >
              {detail.enabled ? "Turn off" : "Turn on"}
            </Button>
            <Button
              variant="ghost"
              disabled={busy}
              onClick={() => {
                if (window.confirm(`Disconnect ${detail.name}? Its saved permissions will be removed.`)) {
                  act(() => MCP.remove(id), onBack)
                }
              }}
            >
              Disconnect
            </Button>
          </div>
        </header>

        {error && (
          <p role="alert" className="text-sm text-red-400">
            {error}
          </p>
        )}

        <section className="space-y-2" aria-label="About">
          <p className="text-sm text-muted-foreground">{detail.description}</p>
          <p className="flex items-center gap-2 text-sm text-muted-foreground">
            <ShieldCheck size={16} />
            {networkLabel(detail.network)}
          </p>
        </section>

        {detail.prompts.length > 0 && <Chips title="Prompts" items={detail.prompts} />}

        <section aria-label="Tool permissions" className="space-y-1">
          <h2 className="text-lg font-medium">Tool permissions</h2>
          <p className="text-sm text-muted-foreground">Choose when the assistant is allowed to use these tools.</p>
          {detail.tools.length === 0 ? (
            <p className="pt-2 text-sm text-muted-foreground">
              {running ? "This connector did not report any tools." : "Turn the connector on to load its tools."}
            </p>
          ) : (
            <div className="pt-2">
              {renderGroup("read", readTools, "Read-only tools")}
              {renderGroup("write", writeTools, "Write/delete tools")}
            </div>
          )}
        </section>

        {logs.length > 0 && (
          <section aria-label="Recent activity" className="space-y-2">
            <h2 className="text-lg font-medium">Recent activity</h2>
            <ul className="space-y-1 text-xs text-muted-foreground">
              {logs.slice(0, 10).map((l) => (
                <li key={l.id}>
                  {new Date(l.ts).toLocaleString()} &middot; {humanize(l.tool)} &middot; {l.decision}
                  {l.ok === 0 ? " · failed" : ""}
                </li>
              ))}
            </ul>
          </section>
        )}
      </div>
    </div>
  )
}
