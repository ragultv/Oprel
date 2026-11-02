"use client"

import React, { useState } from "react"
import {
  X,
  Zap,
  Lock,
  ExternalLink,
  Loader2,
  CheckCircle2,
  AlertCircle,
  ShieldCheck,
  Download,
  KeyRound,
  Eye,
  EyeOff,
} from "lucide-react"
import { HF, type HfAuthStatus } from "@/services/api"
import { cn } from "@/services/utils"

interface HfConnectModalProps {
  isOpen: boolean
  onClose: () => void
  onConnected?: (status: HfAuthStatus) => void
  onSkip?: () => void
  variant?: "welcome" | "download_nudge" | "general"
}

export function HfConnectModal({
  isOpen,
  onClose,
  onConnected,
  onSkip,
  variant = "general",
}: HfConnectModalProps) {
  const [token, setToken] = useState("")
  const [showToken, setShowToken] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState<HfAuthStatus | null>(null)

  if (!isOpen) return null

  const handleConnect = async (e: React.FormEvent) => {
    e.preventDefault()
    const cleanToken = token.trim()
    if (!cleanToken) {
      setError("Please paste your Hugging Face access token")
      return
    }

    setLoading(true)
    setError(null)

    try {
      const res = await HF.connectToken(cleanToken)
      setSuccess(res)
      setTimeout(() => {
        onConnected?.(res)
        onClose()
      }, 1000)
    } catch (err: any) {
      setError(err?.message || "Failed to verify token with Hugging Face")
    } finally {
      setLoading(false)
    }
  }

  const handleSkip = () => {
    onSkip?.()
    onClose()
  }

  const isDownloadNudge = variant === "download_nudge"
  const isWelcome = variant === "welcome"

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-md p-4 animate-fade-in">
      <div className="relative w-full max-w-lg overflow-hidden rounded-2xl border border-border bg-[#0d0d0d] shadow-2xl shadow-primary/10 flex flex-col">
        {/* Glow Header Accent */}
        <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-amber-500 via-primary to-emerald-500" />

        {/* Close Button */}
        <button
          onClick={handleSkip}
          className="absolute top-4 right-4 p-2 rounded-xl text-muted-foreground hover:text-foreground hover:bg-secondary/60 transition-colors"
          aria-label="Close"
        >
          <X size={18} />
        </button>

        {/* Modal Content */}
        <div className="p-6 md:p-8 space-y-6">
          {/* Header Title with HF Emoji */}
          <div className="flex items-start gap-4">
            <div className="flex h-14 w-14 shrink-0 items-center justify-center rounded-2xl border border-amber-500/30 bg-amber-500/10 text-3xl shadow-inner shadow-amber-500/10">
              🤗
            </div>
            <div className="space-y-1">
              <h2 className="text-xl font-bold tracking-tight text-foreground">
                {isDownloadNudge
                  ? "Download Faster with Hugging Face"
                  : isWelcome
                  ? "Welcome to Oprel Studio"
                  : "Connect Hugging Face Account"}
              </h2>
              <p className="text-sm text-muted-foreground leading-relaxed">
                {isDownloadNudge
                  ? "Sign in with your Hugging Face token for accelerated multi-stream downloads and access to gated models."
                  : "Connect your personal HF token to unlock high-speed downloads, gated models, and higher rate limits."}
              </p>
            </div>
          </div>

          {/* Benefits Feature Grid */}
          <div className="grid grid-cols-1 gap-2.5 rounded-xl border border-border/60 bg-secondary/20 p-3.5 text-xs text-muted-foreground">
            <div className="flex items-center gap-2.5">
              <Zap size={14} className="text-amber-400 shrink-0" />
              <span>
                <strong className="text-foreground">Accelerated Downloads:</strong> Multi-threaded chunked transfers via fast transfer.
              </span>
            </div>
            <div className="flex items-center gap-2.5">
              <ShieldCheck size={14} className="text-emerald-400 shrink-0" />
              <span>
                <strong className="text-foreground">Gated Models Access:</strong> Seamlessly download Llama 3, Gemma, and Mistral GGUFs.
              </span>
            </div>
            <div className="flex items-center gap-2.5">
              <Lock size={14} className="text-cyan-400 shrink-0" />
              <span>
                <strong className="text-foreground">10× API Rate Limits:</strong> Avoid IP throttles and download stalls on shared networks.
              </span>
            </div>
          </div>

          {/* Token Form */}
          {!success ? (
            <form onSubmit={handleConnect} className="space-y-4">
              <div className="space-y-2">
                <div className="flex items-center justify-between text-xs">
                  <label className="font-semibold text-foreground flex items-center gap-1.5">
                    <KeyRound size={13} className="text-primary" /> Hugging Face Access Token
                  </label>
                  <a
                    href="https://huggingface.co/settings/tokens/new?token_name=Oprel+Studio&scopes=read"
                    target="_blank"
                    rel="noreferrer"
                    className="inline-flex items-center gap-1 text-primary hover:underline hover:text-primary/80 transition-colors"
                  >
                    <span>Create Token</span>
                    <ExternalLink size={11} />
                  </a>
                </div>

                <div className="relative">
                  <input
                    type={showToken ? "text" : "password"}
                    value={token}
                    onChange={(e) => {
                      setToken(e.target.value)
                      if (error) setError(null)
                    }}
                    placeholder="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
                    className="w-full rounded-xl border border-border bg-secondary/40 px-3.5 py-2.5 pr-10 text-sm font-mono text-foreground placeholder:text-muted-foreground/50 focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary transition-all"
                    disabled={loading}
                    autoFocus
                  />
                  <button
                    type="button"
                    onClick={() => setShowToken(!showToken)}
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground transition-colors"
                    tabIndex={-1}
                  >
                    {showToken ? <EyeOff size={15} /> : <Eye size={15} />}
                  </button>
                </div>
              </div>

              {error && (
                <div className="flex items-start gap-2 rounded-xl border border-destructive/30 bg-destructive/10 p-3 text-xs text-destructive">
                  <AlertCircle size={15} className="shrink-0 mt-0.5" />
                  <span>{error}</span>
                </div>
              )}

              {/* Actions */}
              <div className="flex items-center gap-3 pt-2">
                <button
                  type="submit"
                  disabled={loading || !token.trim()}
                  className={cn(
                    "flex-1 flex items-center justify-center gap-2 rounded-xl px-4 py-2.5 text-sm font-semibold transition-all",
                    loading || !token.trim()
                      ? "bg-secondary text-muted-foreground cursor-not-allowed"
                      : "bg-primary text-primary-foreground hover:bg-primary/90 shadow-md shadow-primary/20"
                  )}
                >
                  {loading && <Loader2 size={16} className="animate-spin" />}
                  <span>{loading ? "Verifying Token…" : "Connect Account"}</span>
                </button>

                <button
                  type="button"
                  onClick={handleSkip}
                  disabled={loading}
                  className="rounded-xl border border-border bg-secondary/30 px-4 py-2.5 text-sm font-medium text-muted-foreground hover:text-foreground hover:bg-secondary/60 transition-all"
                >
                  {isDownloadNudge ? "Download Anonymously" : "Skip for Now"}
                </button>
              </div>
            </form>
          ) : (
            <div className="flex flex-col items-center justify-center p-6 space-y-3 text-center">
              <div className="h-12 w-12 rounded-full bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center text-emerald-400">
                <CheckCircle2 size={28} />
              </div>
              <h3 className="text-base font-bold text-foreground">
                Connected as @{success.username}!
              </h3>
              <p className="text-xs text-muted-foreground">
                Your Hugging Face credentials are saved. Local model downloads will now use high-speed transfer.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
