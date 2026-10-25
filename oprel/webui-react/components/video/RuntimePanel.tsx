"use client"

import { useEffect, useState } from "react"
import { Check, Cpu, Loader2, RefreshCw } from "lucide-react"
import { RUNTIME, type RuntimeSettings } from "@/services/api"
import { cn } from "@/services/utils"
import { ghostBtn, inputCls, labelCls } from "./shared"

const CONTEXTS = [4096, 6144, 8192, 12288, 16384, 32768]
const KV_CHOICES: { value: string; label: string; hint: string }[] = [
  { value: "auto", label: "Automatic", hint: "Oprel decides (16-bit for vision models)" },
  { value: "q4_0", label: "4-bit", hint: "smallest: 1/4 the memory of 16-bit" },
  { value: "q8_0", label: "8-bit", hint: "half the memory, almost no quality loss" },
  { value: "f16", label: "16-bit", hint: "largest, best quality" },
]

/** Context window + KV cache type for the local model. Saved settings apply the next time the model loads. */
export function RuntimePanel({ modelId, onSaved, className }: { modelId: string; onSaved?: () => void; className?: string }) {
  const [settings, setSettings] = useState<RuntimeSettings | null>(null)
  const [ctx, setCtx] = useState<number | null>(null)
  const [kv, setKv] = useState("auto")
  const [busy, setBusy] = useState(false)
  const [note, setNote] = useState<{ ok: boolean; text: string } | null>(null)

  useEffect(() => {
    let live = true
    RUNTIME.get(modelId).then((s) => {
      if (!live) return
      setSettings(s); setCtx(s.ctx_size); setKv(s.kv_cache_type || "auto")
    }).catch(() => {})
    return () => { live = false }
  }, [modelId])

  const dirty = !!settings && (ctx !== settings.ctx_size || kv !== (settings.kv_cache_type || "auto"))

  const save = async () => {
    setBusy(true); setNote(null)
    try {
      const r = await RUNTIME.save({ ctx_size: ctx, kv_cache_type: kv === "auto" ? null : kv, reload_model_id: modelId })
      setSettings((s) => (s ? { ...s, ctx_size: r.ctx_size, kv_cache_type: r.kv_cache_type, loaded_context: r.unloaded ? null : s.loaded_context } : s))
      setNote({ ok: true, text: r.unloaded ? "Saved. The model was unloaded and reloads with these settings on its next use." : "Saved. The model will use these settings when it loads." })
      onSaved?.()
    } catch (e) { setNote({ ok: false, text: (e as Error).message }) } finally { setBusy(false) }
  }

  return (
    <div className={cn("space-y-3 rounded-xl border border-white/10 bg-black/20 p-4", className)}>
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-2 text-xs font-semibold"><Cpu size={13} /> Model memory</div>
        {settings?.loaded_context ? (
          <span className="text-[11px] text-muted-foreground">Running now with {settings.loaded_context.toLocaleString()} tokens</span>
        ) : (
          <span className="text-[11px] text-muted-foreground">Model not loaded</span>
        )}
      </div>
      <div className="grid gap-3 sm:grid-cols-2">
        <div>
          <label className={labelCls} htmlFor="rt-ctx">Context window</label>
          <select id="rt-ctx" value={ctx ?? ""} onChange={(e) => setCtx(e.target.value ? Number(e.target.value) : null)} className={cn(inputCls, "mt-1.5")}>
            <option value="">Automatic (based on your GPU)</option>
            {CONTEXTS.map((c) => <option key={c} value={c}>{c.toLocaleString()} tokens{c === 8192 ? " (recommended)" : ""}</option>)}
          </select>
        </div>
        <div>
          <label className={labelCls} htmlFor="rt-kv">KV cache precision</label>
          <select id="rt-kv" value={kv} onChange={(e) => setKv(e.target.value)} className={cn(inputCls, "mt-1.5")}>
            {KV_CHOICES.filter((c) => c.value === "auto" || !settings || settings.kv_cache_types.includes(c.value)).map((c) => (
              <option key={c.value} value={c.value}>{c.label} · {c.hint}</option>
            ))}
          </select>
        </div>
      </div>
      <p className="text-[11px] leading-relaxed text-muted-foreground">
        Video Studio needs about 8k tokens. A larger window uses more GPU memory; 4-bit KV cache keeps that small, at a small cost in accuracy on long prompts.
        If memory runs short, Oprel keeps fewer layers on the GPU, which is slower but works.
      </p>
      <div className="flex flex-wrap items-center gap-3">
        <button onClick={save} disabled={busy || !dirty} className={ghostBtn}>
          {busy ? <Loader2 size={13} className="animate-spin" /> : <RefreshCw size={13} />} Save &amp; reload model
        </button>
        {note && (
          <span className={cn("flex items-center gap-1.5 text-[11px]", note.ok ? "text-emerald-400" : "text-red-400")}>
            {note.ok && <Check size={12} />} {note.text}
          </span>
        )}
      </div>
    </div>
  )
}
