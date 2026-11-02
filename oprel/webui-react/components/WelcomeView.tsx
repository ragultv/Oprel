"use client"

import { useEffect, useState } from "react"
import { useRouter } from "next/navigation"
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

  useEffect(() => {
    API.getOnboarding()
      .then((data) => {
        setInfo(data)
        setStep("ready")
      })
      .catch((e) => {
        setError(friendlyError(String(e?.message ?? e)))
        setStep("error")
      })
  }, [])

  const finish = () => {
    markOnboarded()
    router.replace("/new-chat")
  }

  const startDownload = async () => {
    if (!info) return
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

        {info && hw && step !== "scan" && (
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

        {step === "downloading" && (
          <div className="space-y-2">
            <Progress value={progress} aria-label="Download progress" />
            <p className="text-sm text-muted-foreground" role="status">
              Downloading... {progress}%. You can leave this page; the download continues.
            </p>
          </div>
        )}

        {error && (
          <p className="text-sm text-red-400" role="alert">
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
