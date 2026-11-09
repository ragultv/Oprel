"use client"

import { useEffect, useState } from "react"
import { useRouter } from "next/navigation"
import { HardDrive, Check, AlertCircle } from "lucide-react"
import { API } from "@/services/api"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"

export const ONBOARDED_KEY = "oprel_onboarded"

type Onboarding = Awaited<ReturnType<typeof API.getOnboarding>>
type Step = "scan" | "ready" | "downloading" | "error"

export function markOnboarded() {
  try {
    localStorage.setItem(ONBOARDED_KEY, "1")
  } catch {
    /* private mode: the welcome screen will simply show again */
  }
}

function friendlyError(message: string): string {
  if (/network|fetch|failed to start/i.test(message)) {
    return "The download could not start. Check your internet connection and try again."
  }
  return message || "Something went wrong. You can try again or skip this step."
}

export function WelcomeView() {
  const router = useRouter()
  const [step, setStep] = useState<Step>("scan")
  const [info, setInfo] = useState<Onboarding | null>(null)
  const [progress, setProgress] = useState(0)
  const [error, setError] = useState("")

  // Storage selection state
  const [storagePath, setStoragePath] = useState("")
  const [freeSpace, setFreeSpace] = useState<number | undefined>(undefined)
  const [pathSaved, setPathSaved] = useState(false)
  const [isSavingPath, setIsSavingPath] = useState(false)

  useEffect(() => {
    API.getOnboarding()
      .then((data) => {
        setInfo(data)
        if (data.storage?.storage_path) {
          setStoragePath(data.storage.storage_path)
          setFreeSpace(data.storage.free_space_gb)
        }
        setStep("ready")
      })
      .catch((e) => {
        setError(friendlyError(String(e?.message ?? e)))
        setStep("error")
      })
  }, [])

  const saveStorageSelection = async () => {
    if (!storagePath.trim()) return
    setIsSavingPath(true)
    try {
      const res = await API.setStorage(storagePath.trim(), false)
      if (res.info) {
        setStoragePath(res.info.storage_path)
        setFreeSpace(res.info.free_space_gb)
      }
      setPathSaved(true)
      setTimeout(() => setPathSaved(false), 2500)
    } catch (err: any) {
      setError(err?.message || "Could not save storage location")
    } finally {
      setIsSavingPath(false)
    }
  }

  const finish = async () => {
    if (storagePath.trim() && storagePath !== info?.storage?.storage_path) {
      try {
        await API.setStorage(storagePath.trim(), false)
      } catch {
        /* best effort before navigating */
      }
    }
    markOnboarded()
    router.replace("/new-chat")
  }

  const startDownload = async () => {
    if (!info) return
    if (storagePath.trim() && storagePath !== info?.storage?.storage_path) {
      try {
        await API.setStorage(storagePath.trim(), false)
      } catch (e: any) {
        setError(e?.message || "Could not update storage path")
        return
      }
    }
    setStep("downloading")
    setError("")
    try {
      const { alias, quantization } = info.recommended
      const job = await API.pullModel(alias, quantization)
      API.streamDownloadProgress(
        job.download_id,
        (p) => setProgress(Math.min(100, Math.round(p.progress ?? 0))),
        () => {
          setProgress(100)
          finish()
        },
        (msg) => {
          setError(friendlyError(msg))
          setStep("error")
        }
      )
    } catch (e: any) {
      setError(friendlyError(String(e?.message ?? e)))
      setStep("error")
    }
  }

  const hw = info?.hardware

  return (
    <main className="min-h-full flex items-center justify-center bg-[#0e0e0e] text-foreground p-6">
      <div className="w-full max-w-lg space-y-6">
        <header className="space-y-2">
          <h1 className="text-2xl font-semibold">Welcome to Oprel</h1>
          <p className="text-sm text-muted-foreground">
            Oprel runs AI models on this computer. Your chats and documents stay on this device. The
            only time Oprel uses the internet is to download models you choose, and to talk to cloud
            providers if you add one in Settings.
          </p>
        </header>

        {step === "scan" && <p role="status">Checking your computer...</p>}

        {info && step !== "scan" && (
          <>
            {/* Storage Path Choice Card */}
            <section className="rounded-lg border border-border p-4 space-y-3 bg-[#141414]" aria-live="polite">
              <div className="flex items-center justify-between">
                <h2 className="font-medium flex items-center gap-2 text-sm text-neutral-200">
                  <HardDrive className="w-4 h-4 text-emerald-400" />
                  Model & Data Storage Location
                </h2>
                {freeSpace !== undefined && (
                  <span className="text-xs text-emerald-400 bg-emerald-500/10 border border-emerald-500/20 px-2 py-0.5 rounded font-mono">
                    {freeSpace} GB free
                  </span>
                )}
              </div>
              <p className="text-xs text-muted-foreground">
                Choose the drive and directory where models, binaries, and local data will be stored.
              </p>
              <div className="flex gap-2">
                <input
                  type="text"
                  value={storagePath}
                  onChange={(e) => {
                    setStoragePath(e.target.value)
                    setPathSaved(false)
                  }}
                  placeholder="e.g. D:/oprel"
                  className="flex-1 bg-[#1a1a1a] border border-border/80 rounded px-3 py-1.5 text-sm font-mono text-neutral-200 focus:outline-none focus:ring-1 focus:ring-emerald-500/50"
                />
                <Button
                  size="sm"
                  variant="outline"
                  onClick={saveStorageSelection}
                  disabled={isSavingPath || !storagePath.trim()}
                  className="text-xs shrink-0"
                >
                  {pathSaved ? (
                    <span className="flex items-center gap-1 text-emerald-400">
                      <Check className="w-3.5 h-3.5" /> Saved
                    </span>
                  ) : isSavingPath ? (
                    "Saving..."
                  ) : (
                    "Set Path"
                  )}
                </Button>
              </div>
            </section>

            {hw && (
              <section className="rounded-lg border border-border p-4 space-y-3" aria-live="polite">
                <h2 className="font-medium">Your computer</h2>
                <ul className="text-sm text-muted-foreground space-y-1">
                  <li>Memory: {hw.ram_total_gb} GB</li>
                  <li>
                    Graphics card:{" "}
                    {hw.gpu_name ? `${hw.gpu_name}${hw.vram_total_gb ? ` (${hw.vram_total_gb} GB)` : ""}` : "none found (the processor will be used)"}
                  </li>
                </ul>
                <div className="border-t border-border pt-3 space-y-1">
                  <h2 className="font-medium">Recommended first model</h2>
                  <p className="text-sm">
                    {info.recommended.alias} ({info.recommended.quantization}), about {info.recommended.approx_size_gb} GB
                  </p>
                  <p className="text-sm text-muted-foreground">{info.recommended.reason}</p>
                </div>
              </section>
            )}
          </>
        )}

        {step === "downloading" && (
          <div className="space-y-2">
            <Progress value={progress} aria-label="Download progress" />
            <p className="text-sm text-muted-foreground" role="status">
              Downloading... {progress}%. You can leave this page; the download continues.
            </p>
          </div>
        )}

        {error && (
          <p className="text-sm text-red-400 flex items-center gap-1.5" role="alert">
            <AlertCircle className="w-4 h-4 shrink-0" />
            {error}
          </p>
        )}

        <div className="flex gap-3">
          {(step === "ready" || step === "error") && info && (
            <Button onClick={startDownload}>
              {step === "error" ? "Try again" : "Download and start"}
            </Button>
          )}
          <Button variant="ghost" onClick={finish}>
            Skip for now
          </Button>
        </div>
      </div>
    </main>
  )
}
