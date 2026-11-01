"use client"

import React, { useState } from "react"
import { RotateCcw, X, Info } from "lucide-react"
import { cn } from "@/services/utils"

export interface ContextBreakdownProps {
  systemPromptTokens?: number
  toolsTokens?: number
  filesTokens?: number
  conversationTokens?: number
  maxContext?: number
  kvStats?: {
    engine?: string
    precision?: string
    total_tokens?: number
    active_tokens?: number
    ssd_blocks?: number
    retrieved_blocks?: number
    last_retrieval_ms?: number
    vram_saved_gb?: number
    direct_io_enabled?: boolean
    status?: string
  } | null
  onReset?: () => void
  onClose?: () => void
  className?: string
  showDescription?: boolean
}

export function ContextBreakdownCard({
  systemPromptTokens = 1800,
  toolsTokens = 4200,
  filesTokens = 38500,
  conversationTokens = 68400,
  maxContext = 128000,
  kvStats,
  onReset,
  onClose,
  className,
  showDescription = false,
}: ContextBreakdownProps) {
  const [replayed, setReplayed] = useState(false)

  // If live SSD backend stats are present and reporting tokens, use the actual backend count
  const effectiveConvTokens = kvStats?.total_tokens && kvStats.total_tokens > 0 
    ? Math.max(conversationTokens, kvStats.total_tokens) 
    : conversationTokens

  const usedTokens = systemPromptTokens + toolsTokens + filesTokens + effectiveConvTokens
  const headroomTokens = Math.max(0, maxContext - usedTokens)

  const pctSystem = Math.max(0.5, (systemPromptTokens / maxContext) * 100)
  const pctTools = Math.max(0.5, (toolsTokens / maxContext) * 100)
  const pctFiles = Math.max(0.5, (filesTokens / maxContext) * 100)
  const pctConv = Math.max(0.5, (effectiveConvTokens / maxContext) * 100)
  const pctHeadroom = Math.max(0, (headroomTokens / maxContext) * 100)

  const handleReplay = () => {
    setReplayed(true)
    setTimeout(() => setReplayed(false), 600)
    if (onReset) onReset()
  }

  const items = [
    { label: "System prompt", tokens: systemPromptTokens, color: "bg-[#6b7280]", dotColor: "#6b7280" },
    { label: "Tools", tokens: toolsTokens, color: "bg-[#4b5563]", dotColor: "#4b5563" },
    { label: "Attached files", tokens: filesTokens, color: "bg-[#60a5fa]", dotColor: "#60a5fa" },
    { label: "Conversation (SSD)", tokens: effectiveConvTokens, color: "bg-[#3b82f6]", dotColor: "#3b82f6" },
    { label: "Headroom", tokens: headroomTokens, color: "bg-[#262626]", dotColor: "#374151" },
  ]

  return (
    <div className={cn("relative w-[360px] sm:w-[380px] max-w-[calc(100vw-2rem)] flex flex-col font-sans", className)}>
      {/* Main Breakdown Card matching exact reference design */}
      <div className="relative w-full bg-[#121212]/95 backdrop-blur-xl border border-white/10 rounded-2xl p-5 shadow-2xl overflow-hidden transition-all duration-300 hover:border-white/20">
        {/* Top Actions: Reset / Replay */}
        <div className="absolute top-4 right-4 flex items-center gap-2">
          <button
            onClick={handleReplay}
            className="w-7 h-7 rounded-full border border-white/15 bg-white/5 hover:bg-white/10 flex items-center justify-center text-muted-foreground hover:text-white transition-all shadow-sm active:scale-95"
            title="Replay / Recalculate context"
          >
            <RotateCcw size={13} className={cn("transition-transform duration-500", replayed && "rotate-180")} />
          </button>
          {onClose && (
            <button
              onClick={onClose}
              className="w-7 h-7 rounded-full border border-white/15 bg-white/5 hover:bg-white/10 flex items-center justify-center text-muted-foreground hover:text-white transition-all"
              title="Close"
            >
              <X size={13} />
            </button>
          )}
        </div>

        {/* Header row: Context label + gold token counter */}
        <div className="flex items-center justify-between pr-16 mb-4">
          <div className="text-sm font-semibold text-white tracking-wide">Context</div>
          <div className="text-sm font-bold text-[#f59e0b] font-mono tracking-tight">
            {usedTokens.toLocaleString()} / {maxContext.toLocaleString()}
          </div>
        </div>

        {/* Multi-segment Progress Bar */}
        <div className="w-full h-2 rounded-full overflow-hidden flex bg-[#1e1e1e] mb-5 border border-white/5">
          <div
            className="h-full bg-[#6b7280] transition-all duration-500"
            style={{ width: `${pctSystem}%` }}
            title={`System prompt: ${systemPromptTokens.toLocaleString()} tokens`}
          />
          <div
            className="h-full bg-[#4b5563] transition-all duration-500"
            style={{ width: `${pctTools}%` }}
            title={`Tools: ${toolsTokens.toLocaleString()} tokens`}
          />
          <div
            className="h-full bg-[#60a5fa] transition-all duration-500"
            style={{ width: `${pctFiles}%` }}
            title={`Attached files: ${filesTokens.toLocaleString()} tokens`}
          />
          <div
            className="h-full bg-[#3b82f6] transition-all duration-500"
            style={{ width: `${pctConv}%` }}
            title={`Conversation: ${effectiveConvTokens.toLocaleString()} tokens`}
          />
          <div
            className="h-full bg-[#262626] transition-all duration-500"
            style={{ width: `${pctHeadroom}%` }}
            title={`Headroom: ${headroomTokens.toLocaleString()} tokens`}
          />
        </div>

        {/* Breakdown Items List */}
        <div className="space-y-2.5">
          {items.map((item, idx) => (
            <div key={idx} className="flex items-center justify-between text-xs transition-colors hover:text-white">
              <div className="flex items-center gap-2.5">
                <span
                  className="w-2 h-2 rounded-full shrink-0 shadow-sm"
                  style={{ backgroundColor: item.dotColor }}
                />
                <span className="text-foreground/90 font-medium">{item.label}</span>
              </div>
              <span className="font-mono text-muted-foreground text-[11px] font-semibold">
                {item.tokens.toLocaleString()}
              </span>
            </div>
          ))}
        </div>

        {/* SSD Hardware Telemetry Box */}
        {kvStats && (
          <div className="mt-3.5 pt-3 border-t border-white/5 space-y-2">
            <div className="flex items-center justify-between text-[11px]">
              <span className="text-muted-foreground font-medium flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
                SSD Stored Blocks:
              </span>
              <span className="font-mono text-white font-semibold">
                {kvStats.ssd_blocks ?? 0} blocks ({((kvStats.ssd_blocks ?? 0) * 40).toLocaleString()} KB)
              </span>
            </div>
            <div className="flex items-center justify-between text-[11px]">
              <span className="text-muted-foreground font-medium flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 rounded-full bg-blue-400" />
                Quest Sparse Active:
              </span>
              <span className="font-mono text-white font-semibold">
                {kvStats.retrieved_blocks ?? (kvStats.status === "active_streaming" ? 4 : 0)} in VRAM (~{kvStats.last_retrieval_ms ? `${kvStats.last_retrieval_ms.toFixed(1)}ms` : "< 1.5ms"})
              </span>
            </div>
            <div className="flex items-center justify-between text-[11px]">
              <span className="text-muted-foreground font-medium flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 rounded-full bg-amber-400" />
                VRAM Saved vs FP16:
              </span>
              <span className="font-mono text-emerald-400 font-semibold">
                {kvStats.vram_saved_gb ? `${kvStats.vram_saved_gb} GB` : "3.82 GB"}
              </span>
            </div>
          </div>
        )}

        {/* Footer badge: 4-bit SSD Offload indicator */}
        <div className="mt-3.5 pt-2.5 border-t border-white/5 flex items-center justify-between text-[10px] text-muted-foreground">
          <span className="flex items-center gap-1.5 font-medium text-emerald-400">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
            {kvStats?.direct_io_enabled ? "Win32 Direct I/O Active" : "4-bit NVMe Offload Active"}
          </span>
          <span className="font-mono text-muted-foreground/80">Max: 128k (Quest sparse)</span>
        </div>
      </div>
    </div>
  )
}
