"use client"

import { useEffect, useRef, useState } from "react"
import { Download, Image as ImageIcon, RefreshCcw, Sparkles, Trash2, Wand2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { API } from "@/services/api"
import type { ImageGenerationJob, ImageModel } from "@/services/api"
import { cn } from "@/services/utils"

type GalleryItem = {
  id: string
  model: string
  prompt: string
  negativePrompt: string
  size: string
  steps: number
  cfgScale: number
  seed: number
  imageUrl: string
  createdAt: number
}

const sizePresets = ["512x512", "1024x1024", "1024x768", "768x1024", "1536x864", "864x1536"]
const samplerPresets = ["euler", "euler_a", "dpmpp_2m", "dpmpp_2m_sde", "heun"]
const IMAGE_STATE_STORAGE_KEY = "oprel_image_generation_state_v1"

type PersistedActiveJob = {
  id: string
  model: string
  prompt: string
  negativePrompt: string
  size: string
  steps: number
  cfgScale: number
  seed: number
  sampler: string
  responseFormat: "url" | "b64_json"
  startedAt: number
}

type PersistedImageState = {
  form: {
    selectedModel: string
    prompt: string
    negativePrompt: string
    size: string
    steps: number
    cfgScale: number
    seed: number
    sampler: string
    responseFormat: "url" | "b64_json"
  }
  gallery: GalleryItem[]
  activeJob: PersistedActiveJob | null
}

function toImageUrl(data: { url?: string; b64_json?: string }) {
  if (data.url) return data.url
  if (data.b64_json) return `data:image/png;base64,${data.b64_json}`
  return ""
}

function readPersistedImageState(): PersistedImageState | null {
  if (typeof window === "undefined") return null
  const raw = window.localStorage.getItem(IMAGE_STATE_STORAGE_KEY)
  if (!raw) return null

  try {
    const parsed = JSON.parse(raw) as Partial<PersistedImageState>
    if (!parsed.form) return null

    return {
      form: {
        selectedModel: parsed.form.selectedModel || "",
        prompt: parsed.form.prompt || "",
        negativePrompt: parsed.form.negativePrompt || "",
        size: parsed.form.size || "512x512",
        steps: Number.isFinite(parsed.form.steps) ? parsed.form.steps : 10,
        cfgScale: Number.isFinite(parsed.form.cfgScale) ? parsed.form.cfgScale : 7,
        seed: Number.isFinite(parsed.form.seed) ? parsed.form.seed : -1,
        sampler: parsed.form.sampler || "euler",
        responseFormat: parsed.form.responseFormat === "b64_json" ? "b64_json" : "url",
      },
      gallery: Array.isArray(parsed.gallery) ? (parsed.gallery as GalleryItem[]) : [],
      activeJob: parsed.activeJob || null,
    }
  } catch {
    return null
  }
}

function writePersistedImageState(state: PersistedImageState) {
  if (typeof window === "undefined") return
  window.localStorage.setItem(IMAGE_STATE_STORAGE_KEY, JSON.stringify(state))
}

export function ImageGenerationView() {
  const [selectedModel, setSelectedModel] = useState("")
  const [imageModels, setImageModels] = useState<ImageModel[]>([])
  const [modelLoading, setModelLoading] = useState(true)
  const [prompt, setPrompt] = useState("A cinematic portrait of a fox explorer, dramatic lighting, high detail")
  const [negativePrompt, setNegativePrompt] = useState("blurry, low quality, watermark, extra limbs")
  const [size, setSize] = useState("512x512")
  const [steps, setSteps] = useState(10)
  const [cfgScale, setCfgScale] = useState(7)
  const [seed, setSeed] = useState(-1)
  const [sampler, setSampler] = useState("euler")
  const [responseFormat, setResponseFormat] = useState<"url" | "b64_json">("url")
  const [loading, setLoading] = useState(false)
  const [generationProgress, setGenerationProgress] = useState<{ value: number; message: string } | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [gallery, setGallery] = useState<GalleryItem[]>([])
  const streamCleanupRef = useRef<(() => void) | null>(null)
  const resumeAttemptedRef = useRef(false)

  const clearActiveJob = () => {
    if (typeof window === "undefined") return
    const saved = readPersistedImageState()
    if (!saved) return
    writePersistedImageState({ ...saved, activeJob: null })
  }

  const persistCurrentState = (activeJob: PersistedActiveJob | null = null, nextGallery = gallery) => {
    if (typeof window === "undefined") return
    writePersistedImageState({
      form: {
        selectedModel,
        prompt,
        negativePrompt,
        size,
        steps,
        cfgScale,
        seed,
        sampler,
        responseFormat,
      },
      gallery: nextGallery,
      activeJob,
    })
  }

  const stopStreaming = () => {
    if (streamCleanupRef.current) {
      streamCleanupRef.current()
      streamCleanupRef.current = null
    }
  }

  const finalizeJob = (imageUrl: string, jobModel: string, jobPrompt: string, jobNegativePrompt: string, jobSize: string, jobSteps: number, jobCfgScale: number, jobSeed: number, jobResponseFormat: "url" | "b64_json") => {
    stopStreaming()
    setGenerationProgress(null)
    setLoading(false)
    clearActiveJob()
    setGallery((current) => {
      const nextGallery = [
        {
          id: `${Date.now()}`,
          model: jobModel,
          prompt: jobPrompt,
          negativePrompt: jobNegativePrompt,
          size: jobSize,
          steps: jobSteps,
          cfgScale: jobCfgScale,
          seed: jobSeed,
          imageUrl,
          createdAt: Date.now(),
        },
        ...current,
      ]
      persistCurrentState(null, nextGallery)
      return nextGallery
    })
  }

  const startOrResumeStreaming = (job: ImageGenerationJob, meta: PersistedActiveJob) => {
    stopStreaming()
    setLoading(job.status !== "completed")
    setGenerationProgress({ value: Math.max(1, job.progress || 1), message: job.message || "Generating" })

    if (job.status === "completed") {
      const first = job.result?.data?.[0]
      const imageUrl = first ? toImageUrl(first) : ""
      if (imageUrl) {
        finalizeJob(imageUrl, meta.model, meta.prompt, meta.negativePrompt, meta.size, meta.steps, meta.cfgScale, meta.seed, meta.responseFormat)
      } else {
        setError("Image generation completed, but no image data was returned.")
        setGenerationProgress(null)
        setLoading(false)
        clearActiveJob()
      }
      return
    }

    const cleanup = API.streamImageGenerationProgress(
      job.id,
      (nextJob) => {
        setGenerationProgress({ value: Math.max(1, nextJob.progress || 1), message: nextJob.message || "Generating" })
      },
      (completedJob) => {
        const first = completedJob.result?.data?.[0]
        const imageUrl = first ? toImageUrl(first) : ""
        if (!imageUrl) {
          setError("Image generation completed, but no image data was returned.")
          setGenerationProgress(null)
          setLoading(false)
          clearActiveJob()
          return
        }
        finalizeJob(imageUrl, meta.model, meta.prompt, meta.negativePrompt, meta.size, meta.steps, meta.cfgScale, meta.seed, meta.responseFormat)
      },
      (streamError) => {
        setError(streamError)
        setGenerationProgress(null)
        setLoading(false)
      }
    )

    streamCleanupRef.current = cleanup
  }

  useEffect(() => {
    let active = true

    API.fetchImageModels()
      .then((models) => {
        if (!active) return

        const downloadedModels = models.filter((model) => model.downloaded && model.backend === "stable-diffusion.cpp")
        setImageModels(downloadedModels)

        if (downloadedModels.length > 0) {
          setSelectedModel((current) => {
            if (current && downloadedModels.some((model) => (model.local_path || model.repo_id) === current)) {
              return current
            }
            return downloadedModels[0].local_path || downloadedModels[0].repo_id
          })
        }
      })
      .catch((err) => setError(err instanceof Error ? err.message : "Failed to fetch image models"))
      .finally(() => {
        if (active) setModelLoading(false)
      })

    return () => {
      active = false
    }
  }, [])

  useEffect(() => {
    const savedState = readPersistedImageState()
    if (!savedState) {
      resumeAttemptedRef.current = true
      return
    }

    setSelectedModel(savedState.form.selectedModel)
    setPrompt(savedState.form.prompt)
    setNegativePrompt(savedState.form.negativePrompt)
    setSize(savedState.form.size)
    setSteps(savedState.form.steps)
    setCfgScale(savedState.form.cfgScale)
    setSeed(savedState.form.seed)
    setSampler(savedState.form.sampler)
    setResponseFormat(savedState.form.responseFormat)
    setGallery(savedState.gallery || [])

    if (savedState.activeJob && !resumeAttemptedRef.current) {
      resumeAttemptedRef.current = true
      API.getImageGenerationJob(savedState.activeJob.id)
        .then((job) => {
          if (job.status === "error") {
            setError(job.error || "Image generation failed")
            setLoading(false)
            setGenerationProgress(null)
            clearActiveJob()
            return
          }
          startOrResumeStreaming(job, savedState.activeJob!)
        })
        .catch(() => {
          setLoading(false)
          setGenerationProgress(null)
          clearActiveJob()
        })
    }
  }, [])

  useEffect(() => {
    persistCurrentState(null, gallery)
  }, [selectedModel, prompt, negativePrompt, size, steps, cfgScale, seed, sampler, responseFormat, gallery])

  useEffect(() => {
    return () => {
      stopStreaming()
    }
  }, [])

  const selectedModelLabel = (() => {
    const selected = imageModels.find((model) => (model.local_path || model.repo_id) === selectedModel)
    if (!selected) return selectedModel
    const supportLabel = selected.supported === false ? " · unsupported" : ""
    return `${selected.id}${selected.quantization ? ` · ${selected.quantization}` : ""}${supportLabel} · ${selected.repo_id}`
  })()

  const handleGenerate = async () => {
    const trimmedPrompt = prompt.trim()
    const trimmedModel = selectedModel.trim()
    if (!trimmedPrompt || !trimmedModel || loading) return

    setLoading(true)
    setGenerationProgress({ value: 1, message: "Queued" })
    setError(null)

    const activeJob: PersistedActiveJob = {
      id: "",
      model: trimmedModel,
      prompt: trimmedPrompt,
      negativePrompt: negativePrompt.trim(),
      size,
      steps,
      cfgScale,
      seed,
      sampler,
      responseFormat,
      startedAt: Date.now(),
    }

    try {
      const started = await API.startImageGeneration({
        model: trimmedModel,
        prompt: trimmedPrompt,
        responseFormat,
        size,
        negativePrompt: negativePrompt.trim() || undefined,
        steps,
        cfgScale,
        seed,
        sampler,
      })

      activeJob.id = started.id
      persistCurrentState(activeJob, gallery)

      const startedJob = await API.getImageGenerationJob(started.id)
      startOrResumeStreaming(startedJob, activeJob)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Image generation failed")
      clearActiveJob()
    } finally {
      // Streaming keeps the progress state alive; do not clear it here.
    }
  }

  const handleDeleteImage = (id: string) => {
    setGallery((current) => {
      const nextGallery = current.filter((item) => item.id !== id)
      persistCurrentState(null, nextGallery)
      return nextGallery
    })
  }

  const latest = gallery[0]

  return (
    <div className="h-full overflow-y-auto bg-[radial-gradient(circle_at_top_right,rgba(255,82,82,0.16),transparent_38%),radial-gradient(circle_at_bottom_left,rgba(255,255,255,0.06),transparent_30%),linear-gradient(180deg,#0e0e0e_0%,#111111_40%,#0b0b0b_100%)] text-foreground">
      <div className="mx-auto max-w-7xl px-6 py-6 lg:px-8 lg:py-8 space-y-6">
        <section className="relative overflow-hidden rounded-3xl border border-white/10 bg-[#141414]/85 backdrop-blur-xl shadow-2xl shadow-black/30">
          <div className="absolute inset-0 bg-[linear-gradient(135deg,rgba(255,255,255,0.04),transparent_40%,rgba(255,82,82,0.08))]" />
          <div className="relative grid gap-6 p-6 xl:grid-cols-[1.15fr_0.85fr] xl:p-8">
            <div className="space-y-5">
              <div className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.22em] text-muted-foreground">
                <Sparkles size={12} />
                Stable Diffusion.cpp
              </div>
              <div className="space-y-3">
                <h1 className="max-w-2xl text-3xl font-black tracking-tight text-balance sm:text-4xl lg:text-5xl">
                  Generate images from text without leaving Oprel Studio.
                </h1>
                <p className="max-w-2xl text-sm leading-6 text-muted-foreground sm:text-base">
                  This view drives the same stable-diffusion.cpp backend used by the CLI and OpenAI-compatible API,
                  so you can iterate on prompts, sizes, and sampling settings from one place.
                </p>
              </div>
              <div className="grid gap-3 sm:grid-cols-3">
                <div className="rounded-2xl border border-white/10 bg-black/20 p-4">
                  <div className="text-[10px] font-bold uppercase tracking-[0.24em] text-muted-foreground">Model</div>
                  <div className="mt-1 text-sm font-semibold text-foreground">{selectedModelLabel || "Select downloaded model"}</div>
                </div>
                <div className="rounded-2xl border border-white/10 bg-black/20 p-4">
                  <div className="text-[10px] font-bold uppercase tracking-[0.24em] text-muted-foreground">Canvas</div>
                  <div className="mt-1 text-sm font-semibold text-foreground">{size}</div>
                </div>
                <div className="rounded-2xl border border-white/10 bg-black/20 p-4">
                  <div className="text-[10px] font-bold uppercase tracking-[0.24em] text-muted-foreground">History</div>
                  <div className="mt-1 text-sm font-semibold text-foreground">{gallery.length} render{gallery.length === 1 ? "" : "s"}</div>
                </div>
              </div>
            </div>

            <div className="rounded-2xl border border-white/10 bg-black/25 p-4">
              <div className="mb-3 flex items-center justify-between">
                <div>
                  <div className="text-[10px] font-bold uppercase tracking-[0.24em] text-muted-foreground">Live Preview</div>
                  <div className="text-sm font-semibold text-foreground">Latest output</div>
                </div>
                {latest && (
                  <button
                    onClick={() => window.open(latest.imageUrl, "_blank", "noopener,noreferrer")}
                    className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-3 py-1.5 text-[11px] font-semibold text-foreground hover:bg-white/10"
                  >
                    <Download size={12} />
                    Open
                  </button>
                )}
              </div>
              <div className="relative aspect-square overflow-hidden rounded-2xl border border-white/10 bg-linear-to-br from-white/10 to-black/30">
                {latest ? (
                  <img src={latest.imageUrl} alt={latest.prompt} className="h-full w-full object-cover" />
                ) : (
                  <div className="flex h-full flex-col items-center justify-center gap-3 text-center text-muted-foreground">
                    <div className="flex h-16 w-16 items-center justify-center rounded-2xl border border-white/10 bg-white/5 text-foreground">
                      <ImageIcon size={28} />
                    </div>
                    <div>
                      <div className="text-sm font-semibold text-foreground">No renders yet</div>
                      <div className="text-xs text-muted-foreground">Generate a first image to populate the gallery.</div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </section>

        <section className="grid gap-6 xl:grid-cols-[0.95fr_1.05fr]">
          <div className="space-y-6 rounded-3xl border border-white/10 bg-[#141414]/90 p-6 shadow-xl shadow-black/20">
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-2xl bg-primary/15 text-primary">
                <Wand2 size={18} />
              </div>
              <div>
                <div className="text-sm font-semibold text-foreground">Prompt Studio</div>
                <div className="text-xs text-muted-foreground">Tune the render request before it hits stable-diffusion.cpp.</div>
              </div>
            </div>

            <div className="space-y-2">
              <label className="text-[11px] font-bold uppercase tracking-[0.24em] text-muted-foreground">Image Model</label>
              <select
                value={selectedModel}
                onChange={(event) => setSelectedModel(event.target.value)}
                disabled={modelLoading || imageModels.length === 0}
                className="h-10 w-full rounded-md border border-input bg-black/20 px-3 text-sm text-foreground outline-none transition-colors focus:border-ring disabled:cursor-not-allowed disabled:opacity-60"
              >
                <option value="" disabled>
                  {modelLoading
                    ? "Loading downloaded models..."
                    : imageModels.length > 0
                      ? "Choose a downloaded GGUF model"
                      : "No downloaded image models found"}
                </option>
                {imageModels.map((model) => {
                  const value = model.local_path || model.repo_id
                  const quantLabel = model.quantization ? ` · ${model.quantization}` : ""
                  const supportLabel = model.supported === false ? " · unsupported" : ""
                  return (
                    <option key={value} value={value}>
                      {model.id}{quantLabel}{supportLabel} · {model.repo_id}
                    </option>
                  )
                })}
              </select>
              {!modelLoading && imageModels.length === 0 && (
                <p className="text-xs text-muted-foreground">
                  Download a stable-diffusion.cpp image model from the Models page first.
                </p>
              )}
            </div>

            <div className="space-y-2">
              <label className="text-[11px] font-bold uppercase tracking-[0.24em] text-muted-foreground">Prompt</label>
              <Textarea
                value={prompt}
                onChange={(event) => setPrompt(event.target.value)}
                rows={6}
                className="min-h-37.5 bg-black/20 text-sm leading-6"
                placeholder="Describe the image you want to generate..."
              />
            </div>

            <div className="space-y-2">
              <label className="text-[11px] font-bold uppercase tracking-[0.24em] text-muted-foreground">Negative Prompt</label>
              <Textarea
                value={negativePrompt}
                onChange={(event) => setNegativePrompt(event.target.value)}
                rows={3}
                className="min-h-23 bg-black/20 text-sm leading-6"
                placeholder="What should the model avoid?"
              />
            </div>

            <div className="grid gap-4 sm:grid-cols-2">
              <div className="space-y-2">
                <label className="text-[11px] font-bold uppercase tracking-[0.24em] text-muted-foreground">Canvas Size</label>
                <select
                  value={size}
                  onChange={(event) => setSize(event.target.value)}
                  className="h-10 w-full rounded-md border border-input bg-black/20 px-3 text-sm text-foreground outline-none transition-colors focus:border-ring"
                >
                  {sizePresets.map((preset) => (
                    <option key={preset} value={preset}>
                      {preset}
                    </option>
                  ))}
                </select>
              </div>

              <div className="space-y-2">
                <label className="text-[11px] font-bold uppercase tracking-[0.24em] text-muted-foreground">Steps</label>
                <Input type="number" min={1} max={80} value={steps} onChange={(event) => setSteps(Number(event.target.value) || 0)} className="bg-black/20" />
              </div>

              <div className="space-y-2">
                <label className="text-[11px] font-bold uppercase tracking-[0.24em] text-muted-foreground">CFG Scale</label>
                <Input type="number" min={0} step={0.1} value={cfgScale} onChange={(event) => setCfgScale(Number(event.target.value) || 0)} className="bg-black/20" />
              </div>

              <div className="space-y-2">
                <label className="text-[11px] font-bold uppercase tracking-[0.24em] text-muted-foreground">Seed</label>
                <Input
                  type="number"
                  value={seed}
                  onChange={(event) => {
                    const parsed = Number(event.target.value)
                    setSeed(Number.isFinite(parsed) ? parsed : -1)
                  }}
                  className="bg-black/20"
                />
              </div>

              <div className="space-y-2">
                <label className="text-[11px] font-bold uppercase tracking-[0.24em] text-muted-foreground">Sampler</label>
                <select
                  value={sampler}
                  onChange={(event) => setSampler(event.target.value)}
                  className="h-10 w-full rounded-md border border-input bg-black/20 px-3 text-sm text-foreground outline-none transition-colors focus:border-ring"
                >
                  {samplerPresets.map((preset) => (
                    <option key={preset} value={preset}>
                      {preset}
                    </option>
                  ))}
                </select>
              </div>

              <div className="space-y-2 sm:col-span-2">
                <label className="text-[11px] font-bold uppercase tracking-[0.24em] text-muted-foreground">Response Format</label>
                <div className="grid grid-cols-2 gap-2">
                  {(["url", "b64_json"] as const).map((format) => (
                    <button
                      key={format}
                      onClick={() => setResponseFormat(format)}
                      className={cn(
                        "rounded-xl border px-3 py-2 text-sm font-semibold transition-all",
                        responseFormat === format
                          ? "border-primary/40 bg-primary/10 text-foreground"
                          : "border-white/10 bg-black/20 text-muted-foreground hover:bg-white/5 hover:text-foreground"
                      )}
                    >
                      {format}
                    </button>
                  ))}
                </div>
              </div>
            </div>

            {error && (
              <div className="rounded-2xl border border-red-500/20 bg-red-500/10 p-4 text-sm text-red-200">
                {error}
              </div>
            )}

            {generationProgress && (
              <div className="rounded-2xl border border-white/10 bg-black/20 p-4">
                <div className="mb-2 flex items-center justify-between text-xs text-muted-foreground">
                  <span>{generationProgress.message}</span>
                  <span>{Math.round(generationProgress.value)}%</span>
                </div>
                <div className="h-1.5 overflow-hidden rounded-full bg-white/10">
                  <div className="h-full bg-primary transition-all" style={{ width: `${generationProgress.value}%` }} />
                </div>
              </div>
            )}

            <div className="flex flex-wrap items-center gap-3">
              <Button onClick={handleGenerate} disabled={loading || !prompt.trim() || !selectedModel.trim()} className="min-w-40">
                {loading ? (
                  <>
                    <RefreshCcw size={16} className="animate-spin" />
                    Generating
                  </>
                ) : (
                  <>
                    <Sparkles size={16} />
                    Generate Image
                  </>
                )}
              </Button>
              <Button
                type="button"
                variant="outline"
                onClick={() => {
                  setPrompt("")
                  setNegativePrompt("")
                  setError(null)
                }}
              >
                Reset Prompt
              </Button>
            </div>
          </div>

          <div className="space-y-6 rounded-3xl border border-white/10 bg-[#141414]/90 p-6 shadow-xl shadow-black/20">
            <div className="flex items-center justify-between gap-3">
              <div>
                <div className="text-sm font-semibold text-foreground">Gallery</div>
                <div className="text-xs text-muted-foreground">Your recent generations stay here until you refresh the page.</div>
              </div>
              <div className="rounded-full border border-white/10 bg-white/5 px-3 py-1 text-[11px] font-semibold text-muted-foreground">
                {gallery.length} item{gallery.length === 1 ? "" : "s"}
              </div>
            </div>

            {gallery.length === 0 ? (
              <div className="flex min-h-90 items-center justify-center rounded-3xl border border-dashed border-white/10 bg-black/15 p-8 text-center text-muted-foreground">
                <div className="max-w-sm space-y-3">
                  <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-2xl border border-white/10 bg-white/5 text-foreground">
                    <ImageIcon size={26} />
                  </div>
                  <div className="text-sm font-semibold text-foreground">Nothing rendered yet</div>
                  <p className="text-sm leading-6 text-muted-foreground">
                    Enter a prompt and generate your first image to populate this panel.
                  </p>
                </div>
              </div>
            ) : (
              <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-2">
                {gallery.map((item) => (
                  <article key={item.id} className="group overflow-hidden rounded-3xl border border-white/10 bg-black/25">
                    <div className="relative aspect-square overflow-hidden bg-black">
                      <img src={item.imageUrl} alt={item.prompt} className="h-full w-full object-cover transition-transform duration-500 group-hover:scale-[1.03]" />
                      <div className="absolute inset-x-0 bottom-0 bg-linear-to-t from-black via-black/50 to-transparent p-4">
                        <div className="space-y-2">
                          <div className="flex items-center justify-between gap-2 text-[11px] text-white/75">
                            <span>{item.model}</span>
                            <span>{new Date(item.createdAt).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</span>
                          </div>
                          <p className="line-clamp-3 text-sm font-medium leading-6 text-white">{item.prompt}</p>
                        </div>
                      </div>
                    </div>
                    <div className="space-y-3 p-4">
                      <div className="grid grid-cols-2 gap-2 text-[11px] text-muted-foreground">
                        <div className="rounded-xl border border-white/10 bg-white/5 px-3 py-2">{item.size}</div>
                        <div className="rounded-xl border border-white/10 bg-white/5 px-3 py-2">{item.steps} steps</div>
                        <div className="rounded-xl border border-white/10 bg-white/5 px-3 py-2">CFG {item.cfgScale}</div>
                        <div className="rounded-xl border border-white/10 bg-white/5 px-3 py-2">Seed {item.seed}</div>
                      </div>
                      <div className="flex gap-2">
                        <Button
                          variant="outline"
                          size="sm"
                          className="flex-1"
                          onClick={() => window.open(item.imageUrl, "_blank", "noopener,noreferrer")}
                        >
                          <Download size={14} />
                          Open
                        </Button>
                        <Button
                          variant="outline"
                          size="sm"
                          className="border-red-500/30 text-red-200 hover:bg-red-500/15 hover:text-red-100"
                          onClick={() => handleDeleteImage(item.id)}
                        >
                          <Trash2 size={14} />
                          Delete
                        </Button>
                      </div>
                    </div>
                  </article>
                ))}
              </div>
            )}
          </div>
        </section>
      </div>
    </div>
  )
}