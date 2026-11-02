"use client"

import React, { useState, useMemo, useEffect } from "react"
import { HardDrive, Cpu, Zap, Activity, Database, Radio, FlaskConical } from "lucide-react"
import { cn } from "@/services/utils"

// Shape of GET /api/v1/system/kv-cache-stats (all values measured by the backend).
interface KVStats {
  status: "active" | "inactive"
  message?: string
  model_id?: string
  kv_cache_type?: string
  block_tokens?: number
  retrieval_top_k?: number | null
  max_context?: number | null
  active_context?: number | null
  session_id?: string | null
  total_tokens?: number
  active_tokens?: number
  evicted_tokens?: number
  ssd_blocks?: number
  active_restored_blocks?: number[]
  last_prefill_tokens?: number
  last_restore_latency_ms?: number
  last_read_ms?: number
  last_set_data_ms?: number
  ssd_bytes_written?: number
  ssd_bytes_read?: number
  archive_validation_failures?: number
  session_block_file_bytes?: number
  bytes_per_token?: number | null
  tokens_fit_in_free_space?: number | null
  ssd_dir?: string
  ssd_dir_bytes?: number
  disk_free_bytes?: number
  disk_total_bytes?: number
}

const POLL_MS = 1500
const MAX_RENDERED_BLOCKS = 2048
// Used by the simulator only when no model has reported a measured size yet:
// Qwen3-8B with a Q4_0 KV cache (36 layers x 8 KV heads x 128 dims x K+V x 4.5 bits).
const FALLBACK_BYTES_PER_TOKEN = 41472

function fmtBytes(n?: number | null): string {
  if (!n || n <= 0) return "0 B"
  const units = ["B", "KB", "MB", "GB", "TB"]
  let i = 0
  let v = n
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024
    i++
  }
  return `${v.toFixed(v >= 100 || i === 0 ? 0 : 1)} ${units[i]}`
}

function fmtTokens(n?: number | null): string {
  if (n === null || n === undefined) return "—"
  return n.toLocaleString()
}

function StatTile({
  icon,
  label,
  value,
  sub,
  valueClass = "text-white",
}: {
  icon: React.ReactNode
  label: string
  value: string
  sub?: string
  valueClass?: string
}) {
  return (
    <div className="bg-[#1a1a1a] border border-white/5 rounded-xl p-4 flex flex-col justify-between">
      <div className="text-[10px] uppercase font-bold text-muted-foreground tracking-wider mb-1 flex items-center gap-1.5">
        {icon} {label}
      </div>
      <div className={cn("text-2xl font-bold font-mono", valueClass)}>{value}</div>
      {sub && <div className="text-[10px] text-muted-foreground mt-1">{sub}</div>}
    </div>
  )
}

export function KVCacheExplorer() {
  const [mode, setMode] = useState<"live" | "sim">("live")
  const [stats, setStats] = useState<KVStats | null>(null)
  const [fetchError, setFetchError] = useState(false)

  // Live telemetry: poll while the live view is open.
  useEffect(() => {
    if (mode !== "live") return
    let cancelled = false
    const load = () =>
      fetch("/api/v1/system/kv-cache-stats")
        .then((res) => (res.ok ? res.json() : Promise.reject(res.status)))
        .then((data: KVStats) => {
          if (!cancelled) {
            setStats(data)
            setFetchError(false)
          }
        })
        .catch(() => {
          if (!cancelled) setFetchError(true)
        })
    load()
    const id = setInterval(load, POLL_MS)
    return () => {
      cancelled = true
      clearInterval(id)
    }
  }, [mode])

  return (
    <div className="bg-[#141414] border border-border/80 rounded-2xl p-6 shadow-2xl space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-5 border-b border-white/5">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span
              className={cn(
                "w-2.5 h-2.5 rounded-full",
                mode === "live" && stats?.status === "active" ? "bg-emerald-400 animate-pulse" : "bg-muted-foreground/50"
              )}
            />
            <h2 className="text-base font-bold text-white tracking-tight flex items-center gap-2">
              KV Cache & SSD Offload
              {stats?.kv_cache_type && (
                <span className="text-[10px] uppercase font-mono px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 font-bold">
                  {stats.kv_cache_type}
                </span>
              )}
            </h2>
          </div>
          <p className="text-xs text-muted-foreground">
            {mode === "live"
              ? "Live state of the native SSD backend, refreshed every 1.5 s."
              : "What-if simulator. Numbers below are estimates, not measurements."}
          </p>
        </div>

        <div className="flex items-center bg-[#1e1e1e] border border-border/80 rounded-xl p-1 gap-1">
          {(
            [
              { key: "live", label: "Live Backend", icon: <Radio size={12} /> },
              { key: "sim", label: "What-If Simulator", icon: <FlaskConical size={12} /> },
            ] as const
          ).map((m) => (
            <button
              key={m.key}
              onClick={() => setMode(m.key)}
              className={cn(
                "px-3 py-1 rounded-lg text-xs font-medium transition-all flex items-center gap-1.5",
                mode === m.key ? "bg-primary text-primary-foreground shadow-sm font-bold" : "text-muted-foreground hover:text-white"
              )}
            >
              {m.icon}
              {m.label}
            </button>
          ))}
        </div>
      </div>

      {mode === "live" ? (
        <LiveView stats={stats} fetchError={fetchError} />
      ) : (
        <SimulatorView bytesPerToken={stats?.bytes_per_token ?? null} defaultTopK={stats?.retrieval_top_k ?? 2} />
      )}
    </div>
  )
}

function LiveView({ stats, fetchError }: { stats: KVStats | null; fetchError: boolean }) {
  if (fetchError && !stats) {
    return <div className="text-xs text-red-400">Could not reach /api/v1/system/kv-cache-stats.</div>
  }
  if (!stats) {
    return <div className="text-xs text-muted-foreground">Loading…</div>
  }
  if (stats.status !== "active") {
    return (
      <div className="bg-[#1a1a1a] border border-white/5 rounded-xl p-4 text-xs text-muted-foreground space-y-1">
        <div className="text-white font-semibold">SSD backend not active</div>
        <div>{stats.message ?? "No model is loaded with the SSD KV-cache backend."}</div>
        <div>
          Enable it with <code className="text-white">"use_ssd_wrapper": true</code> in runtime_settings.json, then load a model.
          Disk free on the cache drive: <span className="text-white font-mono">{fmtBytes(stats.disk_free_bytes)}</span>.
        </div>
      </div>
    )
  }

  const blockTokens = stats.block_tokens ?? 512
  const ssdBlocks = stats.ssd_blocks ?? 0
  const restored = new Set(stats.active_restored_blocks ?? [])
  const activeTokens = stats.active_tokens ?? 0
  // Restored blocks sit inside the live window; the rest of it is the recent tail.
  const tailTokens = Math.max(0, activeTokens - restored.size * blockTokens)
  const tailBlocks = Math.ceil(tailTokens / blockTokens)
  const totalBlocks = ssdBlocks + tailBlocks
  const rendered = Math.min(totalBlocks, MAX_RENDERED_BLOCKS)
  const activeCtx = stats.active_context ?? 0
  const vramPct = activeCtx ? Math.min(100, (activeTokens / activeCtx) * 100) : 0

  return (
    <>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3.5">
        <StatTile
          icon={<Database size={12} className="text-primary" />}
          label="Context (total)"
          value={fmtTokens(stats.total_tokens)}
          sub={`of ${fmtTokens(stats.max_context)} max · session ${stats.session_id ?? "—"}`}
        />
        <StatTile
          icon={<Zap size={12} className="text-emerald-400" />}
          label="Live in VRAM"
          value={fmtTokens(activeTokens)}
          valueClass="text-emerald-400"
          sub={`of ${fmtTokens(activeCtx)}-token window (${vramPct.toFixed(0)}%)`}
        />
        <StatTile
          icon={<HardDrive size={12} className="text-cyan-400" />}
          label="On SSD"
          value={fmtTokens(stats.evicted_tokens)}
          valueClass="text-cyan-400"
          sub={`${ssdBlocks} blocks · ${fmtBytes(stats.session_block_file_bytes)} · ${
            stats.bytes_per_token ? `${fmtBytes(stats.bytes_per_token)}/token` : "size per token not measured yet"
          }`}
        />
        <StatTile
          icon={<Activity size={12} className="text-purple-400" />}
          label="Last restore"
          value={stats.last_restore_latency_ms ? `${stats.last_restore_latency_ms.toFixed(0)} ms` : "—"}
          valueClass="text-purple-400"
          sub={
            stats.last_restore_latency_ms
              ? `read ${(stats.last_read_ms ?? 0).toFixed(1)} ms · inject ${(stats.last_set_data_ms ?? 0).toFixed(1)} ms`
              : "no block restored yet"
          }
        />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-3.5 text-[11px]">
        <div className="bg-[#1a1a1a] border border-white/5 rounded-xl px-4 py-3 flex justify-between">
          <span className="text-muted-foreground">Disk free ({stats.ssd_dir})</span>
          <span className="font-mono text-white">{fmtBytes(stats.disk_free_bytes)}</span>
        </div>
        <div className="bg-[#1a1a1a] border border-white/5 rounded-xl px-4 py-3 flex justify-between">
          <span className="text-muted-foreground">Tokens that fit in free space</span>
          <span className="font-mono text-white">{fmtTokens(stats.tokens_fit_in_free_space)}</span>
        </div>
        <div className="bg-[#1a1a1a] border border-white/5 rounded-xl px-4 py-3 flex justify-between">
          <span className="text-muted-foreground">Written / read (this load)</span>
          <span className="font-mono text-white">
            {fmtBytes(stats.ssd_bytes_written)} / {fmtBytes(stats.ssd_bytes_read)}
          </span>
        </div>
      </div>

      <div>
        <div className="flex flex-wrap items-center justify-between gap-2 mb-3 text-xs">
          <div className="flex items-center gap-2">
            <span className="font-semibold text-white">Blocks</span>
            <span className="text-muted-foreground font-mono">
              ({totalBlocks} · ~{blockTokens} tokens each · top-k {stats.retrieval_top_k ?? "—"})
            </span>
          </div>
          <div className="flex items-center gap-3 text-[11px] text-muted-foreground">
            <span className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded bg-slate-600" /> Cold on NVMe SSD
            </span>
            <span className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded bg-cyan-400 animate-pulse" /> Paged from SSD into VRAM
            </span>
            <span className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded bg-emerald-500" /> Active live window
            </span>
          </div>
        </div>

        <div className="bg-[#0f0f0f] border border-white/10 rounded-xl p-3.5 max-h-56 overflow-y-auto">
          {totalBlocks === 0 ? (
            <div className="text-xs text-muted-foreground">Empty context. Send a message to fill it.</div>
          ) : (
            <div className="grid grid-cols-16 sm:grid-cols-24 md:grid-cols-32 gap-1.5">
              {Array.from({ length: rendered }).map((_, b) => {
                const onSsd = b < ssdBlocks
                const isRestored = onSsd && restored.has(b)
                const bg = isRestored
                  ? "bg-cyan-400 text-black ring-2 ring-cyan-300 animate-pulse"
                  : onSsd
                  ? "bg-slate-700/80 text-slate-200 border-slate-500/30"
                  : "bg-emerald-600/80 text-white border-emerald-400/40"
                const label = onSsd ? `SSD block #${b}` : `Live window chunk ${b - ssdBlocks + 1}/${tailBlocks}`
                const status = isRestored ? "Paged from SSD into VRAM" : onSsd ? "Cold on NVMe SSD (0 VRAM)" : "Active live window (VRAM)"
                return (
                  <div
                    key={b}
                    className={cn(
                      "aspect-square rounded text-[8px] font-mono flex items-center justify-center border border-transparent transition-colors",
                      bg
                    )}
                    title={`${label} | ${status}`}
                  >
                    {onSsd ? b : ""}
                  </div>
                )
              })}
            </div>
          )}
          {totalBlocks > rendered && (
            <div className="text-[10px] text-muted-foreground mt-2">Showing the first {rendered} of {totalBlocks} blocks.</div>
          )}
        </div>
      </div>

      {(stats.archive_validation_failures ?? 0) > 0 && (
        <div className="text-[11px] text-amber-400">
          {stats.archive_validation_failures} archived block(s) failed validation on restore and were skipped.
        </div>
      )}
    </>
  )
}

function SimulatorView({ bytesPerToken, defaultTopK }: { bytesPerToken: number | null; defaultTopK: number }) {
  const blockTokens = 512
  const [contextTokens, setContextTokens] = useState<number>(32768)
  const [activeWindow, setActiveWindow] = useState<number>(2560)
  const [topK, setTopK] = useState<number>(Math.max(1, Math.min(8, defaultTopK)))
  const perToken = bytesPerToken ?? FALLBACK_BYTES_PER_TOKEN

  const liveTokens = Math.min(contextTokens, activeWindow)
  const ssdTokens = Math.max(0, contextTokens - liveTokens)
  const ssdBlocks = Math.ceil(ssdTokens / blockTokens)
  const liveBlocks = Math.ceil(liveTokens / blockTokens)
  const k = Math.min(topK, ssdBlocks)

  // Deterministic spread of k "retrieved" blocks across the archive, for illustration only.
  const retrieved = useMemo(() => {
    const set = new Set<number>()
    for (let i = 0; i < k; i++) set.add(Math.floor(((i + 0.5) * ssdBlocks) / k))
    return set
  }, [k, ssdBlocks])

  const totalBlocks = ssdBlocks + liveBlocks
  const rendered = Math.min(totalBlocks, MAX_RENDERED_BLOCKS)

  return (
    <>
      <div className="bg-[#1a1a1a] border border-white/5 rounded-xl p-4 flex flex-wrap items-center gap-4 text-xs">
        <div className="flex items-center gap-1 bg-[#1e1e1e] border border-border/80 rounded-xl p-1">
          {[8192, 16384, 32768, 65536, 131072].map((ctx) => (
            <button
              key={ctx}
              onClick={() => setContextTokens(ctx)}
              className={cn(
                "px-2.5 py-1 rounded-lg text-xs font-mono font-medium transition-all",
                contextTokens === ctx ? "bg-primary text-primary-foreground font-bold" : "text-muted-foreground hover:text-white"
              )}
            >
              {ctx / 1024}k
            </button>
          ))}
        </div>
        <label className="flex items-center gap-1.5 text-muted-foreground">
          VRAM window:
          <select
            value={activeWindow}
            onChange={(e) => setActiveWindow(parseInt(e.target.value))}
            className="bg-[#242424] border border-white/10 text-white rounded-lg px-2 py-1 font-mono text-xs outline-none"
          >
            {[2560, 4096, 8192].map((w) => (
              <option key={w} value={w}>
                {w.toLocaleString()} tokens
              </option>
            ))}
          </select>
        </label>
        <label className="flex items-center gap-1.5 text-muted-foreground">
          Top-k restored:
          <select
            value={topK}
            onChange={(e) => setTopK(parseInt(e.target.value))}
            className="bg-[#242424] border border-white/10 text-white rounded-lg px-2 py-1 font-mono text-xs outline-none"
          >
            {[1, 2, 4, 8].map((v) => (
              <option key={v} value={v}>
                {v} blocks
              </option>
            ))}
          </select>
        </label>
        <span className="text-[10px] text-muted-foreground">
          {bytesPerToken ? `${fmtBytes(perToken)}/token measured from the loaded model` : `${fmtBytes(perToken)}/token assumed (Qwen3-8B, Q4_0)`}
        </span>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3.5">
        <StatTile icon={<Database size={12} className="text-primary" />} label="Context" value={contextTokens.toLocaleString()} sub="tokens" />
        <StatTile
          icon={<Zap size={12} className="text-emerald-400" />}
          label="KV in VRAM"
          value={fmtBytes((liveTokens + k * blockTokens) * perToken)}
          valueClass="text-emerald-400"
          sub={`${liveTokens.toLocaleString()} live + ${k} restored blocks`}
        />
        <StatTile
          icon={<HardDrive size={12} className="text-cyan-400" />}
          label="KV on SSD"
          value={fmtBytes(ssdTokens * perToken)}
          valueClass="text-cyan-400"
          sub={`${ssdBlocks} blocks of ${blockTokens} tokens`}
        />
        <StatTile
          icon={<Cpu size={12} className="text-purple-400" />}
          label="Read per query"
          value={fmtBytes(k * blockTokens * perToken)}
          valueClass="text-purple-400"
          sub="restore costs ~0.2–0.8 s per block (measured)"
        />
      </div>

      <div className="bg-[#0f0f0f] border border-white/10 rounded-xl p-3.5 max-h-56 overflow-y-auto">
        <div className="grid grid-cols-16 sm:grid-cols-24 md:grid-cols-32 gap-1.5">
          {Array.from({ length: rendered }).map((_, b) => {
            const onSsd = b < ssdBlocks
            const isRetrieved = onSsd && retrieved.has(b)
            return (
              <div
                key={b}
                className={cn(
                  "aspect-square rounded text-[8px] font-mono flex items-center justify-center",
                  isRetrieved ? "bg-cyan-400 text-black" : onSsd ? "bg-slate-700/80 text-slate-200" : "bg-emerald-600/80 text-white"
                )}
                title={onSsd ? `Simulated SSD block #${b}${isRetrieved ? " (restored)" : ""}` : "Simulated live window"}
              >
                {onSsd ? b : ""}
              </div>
            )
          })}
        </div>
      </div>
      <p className="text-[11px] text-muted-foreground">
        The model only attends to the live window plus the restored blocks; the rest of the SSD archive is invisible to a
        query unless the ranker selects it.
      </p>
    </>
  )
}
