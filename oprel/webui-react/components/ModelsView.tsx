"use client"

import { useState, useEffect } from "react"
import { useRouter } from "next/navigation"
import {
  Search,
  Play,
  CheckCircle2,
  Layers,
  HardDrive,
  Zap,
  MemoryStick,
  FileCode,
  Cpu,
  Box,
  MessageSquarePlus,
  Trash2,
} from "lucide-react"
import { cn } from "@/services/utils"
import { useApp } from "@/services/context"
import { useDownloads } from "@/services/downloadContext"
import { useToast } from "@/hooks/use-toast"
import { API } from "@/services/api"
import type { AIModel } from "@/services/data"
import { QuantizationSelector } from "./QuantizationSelector"

function CompatBadge({ compat }: { compat: AIModel["compatibility"] }) {
  const map = {
    compatible: { label: "Compatible", class: "bg-green-500/10 text-green-400 border-green-500/20" },
    hybrid: { label: "Partial", class: "bg-amber-500/10 text-amber-400 border-amber-500/20" },
    incompatible: { label: "Incompatible", class: "bg-destructive/10 text-destructive border-destructive/20" },
  }
  const { label, class: cls } = map[compat]
  return (
    <span className={cn("px-2 py-0.5 rounded-md text-[10px] font-bold border", cls)}>
      {label}
    </span>
  )
}

function StatusBadge({ status }: { status: AIModel["status"] }) {
  if (status === "loaded")
    return (
      <span className="flex items-center gap-1 text-[10px] font-bold text-green-400">
        <span className="w-1.5 h-1.5 rounded-full bg-green-400 pulse-dot" /> LOADED
      </span>
    )
  if (status === "downloading")
    return <span className="text-[10px] font-bold text-amber-400">DOWNLOADING</span>
  if (status === "available")
    return <span className="text-[10px] font-bold text-blue-400 truncate max-w-15">READY</span>
  return <span className="text-[10px] font-bold text-muted-foreground uppercase opacity-50">REGISTRY</span>
}

function ModelListItem({
  model,
  isSelected,
  onSelect,
}: {
  model: AIModel
  isSelected: boolean
  onSelect: () => void
}) {
  return (
    <button
      onClick={onSelect}
      className={cn(
        "w-full text-left p-3 rounded-lg transition-all border",
        isSelected
          ? "bg-secondary border-primary/30 text-foreground"
          : "border-transparent hover:bg-secondary/50 text-muted-foreground hover:text-foreground"
      )}
    >
      <div className="flex items-start justify-between gap-2">
        <div className="flex-1 min-w-0">
          <div className="text-sm font-semibold text-foreground truncate">{model.name}</div>
          <div className="flex items-center gap-2 mt-1">
            <span className="text-[10px] text-muted-foreground">{model.size}</span>
            <span className="text-muted-foreground/30">·</span>
            <span className="text-[10px] text-muted-foreground font-mono">{model.quantization}</span>
          </div>
        </div>
        <div className="flex flex-col items-end gap-1 shrink-0">
          <StatusBadge status={model.status} />
          {model.category === "external" && <CompatBadge compat={model.compatibility} />}
        </div>
      </div>
    </button>
  )
}

function StatCard({ icon, label, value }: { icon: React.ReactNode; label: string; value: string }) {
  return (
    <div className="bg-[#0f0f0f] border border-border rounded-lg p-3 flex items-center gap-3">
      <div className="w-8 h-8 rounded-md bg-secondary flex items-center justify-center shrink-0 text-muted-foreground">
        {icon}
      </div>
      <div>
        <div className="text-[10px] text-muted-foreground font-medium uppercase tracking-wider">{label}</div>
        <div className="text-sm font-bold text-foreground">{value}</div>
      </div>
    </div>
  )
}

function ModelDetailPanel({ model }: { model: AIModel }) {
  const { setActiveModelId, refreshModels, setIsModelLoading, createConversation } = useApp()
  const { addDownload, updateDownload, setDialogOpen } = useDownloads()
  const router = useRouter()
  const { toast } = useToast()
  const [loading, setLoading] = useState(false)
  const [qualityLevel, setQualityLevel] = useState<string>("")
  const [selectedQuantization, setSelectedQuantization] = useState<string>("")
  const [selectedSize, setSelectedSize] = useState<number>(0)
  const [localQuants, setLocalQuants] = useState<string[]>([])
  const [deletingQuant, setDeletingQuant] = useState<string | null>(null)
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null)
  const isImageModel = model.category === "text-to-image"

  // Fetch (or re-fetch) local quantizations
  const refreshLocalQuants = async () => {
    try {
      const actualModelId = model.modelRepoId || model.id
      const result = await API.fetchLocalQuantizations(actualModelId)
      setLocalQuants(result.local_quantizations)
    } catch (error) {
      console.warn("Failed to fetch local quantizations:", error)
      setLocalQuants([])
    }
  }

  // Fetch local quantizations when model changes
  useEffect(() => {
    refreshLocalQuants()
  }, [model.id, model.modelRepoId])

  const handleDownload = async () => {
    if (!selectedQuantization) {
      toast({
        title: "No Quantization Selected",
        description: "Please select a quantization level before downloading",
        variant: "destructive",
      })
      return
    }

    setLoading(true)
    
    try {
      const actualModelId = model.modelRepoId || model.id
      // Start download and get download_id
      const response = await API.pullModel(actualModelId, selectedQuantization)
      const downloadId = response.download_id
      
      // Add to download manager with initial state
      const displayId = `${actualModelId}-${selectedQuantization}`
      addDownload({
        modelId: displayId,
        modelName: model.name,
        quantization: selectedQuantization,
        progress: 0,
        downloaded: 0,
        total: selectedSize * 1024 * 1024 * 1024, // Convert GB to bytes
        speed: 0,
        timeLeft: "Calculating...",
        status: "ongoing",
      })
      
      // Open the download dialog
      setDialogOpen(true)
      
      // Show success toast
      toast({
        title: "Download Started",
        description: `${model.name} (${selectedQuantization}) is now downloading`,
      })
      
      // Stream progress updates via SSE
      const cleanup = API.streamDownloadProgress(
        downloadId,
        (progress) => {
          // Update download manager with real progress
          const formatTime = (seconds: number) => {
            if (seconds < 60) return `${Math.ceil(seconds)}s`
            const mins = Math.floor(seconds / 60)
            const secs = Math.ceil(seconds % 60)
            return `${mins}:${secs.toString().padStart(2, '0')}`
          }
          
          updateDownload(displayId, {
            progress: progress.progress,
            downloaded: progress.downloaded,
            total: progress.total,
            speed: progress.speed,
            timeLeft: formatTime(progress.eta),
            status: progress.status === "completed" ? "completed" : "ongoing",
          })
        },
        () => {
          // Download completed — refresh local quants immediately so LOCAL badge appears
          updateDownload(displayId, {
            status: "completed",
            progress: 100,
            timeLeft: "0s",
          })
          toast({
            title: "Download Complete",
            description: `${model.name} (${selectedQuantization}) is ready to use`,
          })
          refreshLocalQuants()   // ← update LOCAL badge without full refresh
          refreshModels()
          setLoading(false)
        },
        (error) => {
          // Download failed or SSE connection error
          if (error.includes("restart the server")) {
            // SSE endpoint not available - server needs restart
            toast({
              title: "Server Restart Required",
              description: "Download is running in background. Restart server for real-time progress: pkill -f oprel.server.daemon && oprel start",
              variant: "default",
              duration: 10000,
            })
            // Keep the download in "ongoing" state since it's actually downloading
            updateDownload(displayId, {
              status: "ongoing",
              timeLeft: "Calculating...",
            })
            setLoading(false)
          } else {
            // Actual download error
            updateDownload(displayId, {
              status: "error",
              error: error,
            })
            toast({
              title: "Download Failed",
              description: error,
              variant: "destructive",
            })
            setLoading(false)
          }
        }
      )
      
      // Store cleanup function for component unmount
      return () => cleanup()
      
    } catch (error) {
      console.error("Failed to start download:", error)
      toast({
        title: "Failed to Start Download",
        description: error instanceof Error ? error.message : "Unknown error",
        variant: "destructive",
      })
      setLoading(false)
    }
  }

  const handleImageDownload = async () => {
    setLoading(true)
    try {
      const actualModelId = model.modelRepoId || model.id
      const quantToDownload = selectedQuantization || model.quantization || "Q8_0"
      const response = await API.pullImageModel(actualModelId, quantToDownload)
      const displayId = `${actualModelId}-${quantToDownload}`

      addDownload({
        modelId: displayId,
        modelName: model.name,
        quantization: quantToDownload,
        progress: 0,
        downloaded: 0,
        total: 0,
        speed: 0,
        timeLeft: "Calculating...",
        status: "ongoing",
      })
      setDialogOpen(true)

      API.streamDownloadProgress(
        response.download_id,
        (progress) => {
          const formatTime = (seconds: number) => {
            if (seconds < 60) return `${Math.ceil(seconds)}s`
            const mins = Math.floor(seconds / 60)
            const secs = Math.ceil(seconds % 60)
            return `${mins}:${secs.toString().padStart(2, '0')}`
          }
          updateDownload(displayId, {
            progress: progress.progress,
            downloaded: progress.downloaded,
            total: progress.total,
            speed: progress.speed,
            timeLeft: formatTime(progress.eta),
            status: progress.status === "completed" ? "completed" : "ongoing",
          })
        },
        () => {
          updateDownload(displayId, { status: "completed", progress: 100, timeLeft: "0s" })
          toast({ title: "Download Complete", description: `${model.name} is ready` })
          refreshModels()
          setLoading(false)
        },
        (streamError) => {
          updateDownload(displayId, { status: "error", error: streamError })
          toast({ title: "Download Failed", description: streamError, variant: "destructive" })
          setLoading(false)
        }
      )
    } catch (error) {
      toast({
        title: "Failed to Start Download",
        description: error instanceof Error ? error.message : "Unknown error",
        variant: "destructive",
      })
      setLoading(false)
    }
  }

  const handleLoad = async () => {
    setLoading(true)
    setIsModelLoading(true)
    try {
      const actualModelId = model.modelRepoId || model.id
      // If local quantizations exist, use the first one (or selected one)
      const quantToLoad = localQuants.length > 0 ? localQuants[0] : selectedQuantization
      
      if (quantToLoad) {
        await API.loadModel(actualModelId, { quantization: quantToLoad })
      } else {
        await API.loadModel(actualModelId)
      }
      
      await refreshModels()
      setActiveModelId(model.id)
      
      toast({
        title: "Model Loaded",
        description: `${model.name} is ready for inference`,
      })
    } catch (error) {
      console.error("Failed to load model:", error)
      toast({
        title: "Failed to Load Model",
        description: error instanceof Error ? error.message : "Unknown error",
        variant: "destructive",
      })
    } finally {
      setLoading(false)
      setIsModelLoading(false)
    }
  }

  const handleDeleteQuant = async (quant: string) => {
    setDeletingQuant(quant)
    try {
      const actualModelId = model.modelRepoId || model.id
      await API.deleteModelQuant(actualModelId, quant)
      toast({ title: "Deleted", description: `${model.name} (${quant}) removed from disk` })
      await refreshLocalQuants()
      await refreshModels()
    } catch (error) {
      toast({
        title: "Delete Failed",
        description: error instanceof Error ? error.message : "Unknown error",
        variant: "destructive",
      })
    } finally {
      setDeletingQuant(null)
      setConfirmDelete(null)
    }
  }

  // Determine button state based on selected quantization vs. what's locally available.
  //
  // Priority (evaluated top to bottom):
  //   Download  — a specific quant is chosen that is NOT on disk
  //   Chat      — model is loaded in RAM AND (no quant chosen, OR chosen quant IS on disk)
  //   Load      — chosen quant IS on disk, but model isn't loaded yet
  //
  const isDownloaded = model.downloaded || model.status === 'available' || model.status === 'loaded'
  const isSelectedQuantAvailable = localQuants.includes(selectedQuantization)

  // ① A quant is explicitly selected and it is NOT present locally → Download
  const shouldShowDownload = !!selectedQuantization && !isSelectedQuantAvailable

  // ② Model is loaded in VRAM AND (no quant selection override, OR the chosen quant IS local)
  //    This is intentionally checked BEFORE shouldShowLoad so Chat wins when already loaded.
  const shouldShowChat =
    model.status === "loaded" && (!selectedQuantization || isSelectedQuantAvailable)

  // ③ A local quant is chosen but the model isn't loaded (or a different quant is loaded)
  const shouldShowLoad =
    !shouldShowDownload && !shouldShowChat && isDownloaded && isSelectedQuantAvailable

  return (
    <>
    <div className="flex flex-col h-full animate-fade-in-up">
      {/* Hero */}
      <div className="p-7 border-b border-border">
        <div className="flex items-start gap-5">
          <div className="w-14 h-14 rounded-xl bg-secondary border border-border flex items-center justify-center shrink-0">
            <Box size={28} className="text-primary" />
          </div>
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 flex-wrap">
              <h1 className="text-xl font-bold text-foreground leading-tight">{model.name}</h1>
              {qualityLevel && (
                <span className={cn(
                  "px-2 py-0.5 rounded-md text-[10px] font-bold border",
                  qualityLevel === "Excellent" && "bg-green-500/10 text-green-400 border-green-500/20",
                  qualityLevel === "Good" && "bg-blue-500/10 text-blue-400 border-blue-500/20",
                  qualityLevel === "Fair" && "bg-amber-500/10 text-amber-400 border-amber-500/20",
                  qualityLevel === "Low" && "bg-red-500/10 text-red-400 border-red-500/20"
                )}>
                  {qualityLevel}
                </span>
              )}
            </div>
            <div className="flex items-center gap-2 mt-1.5 flex-wrap">
              <span className="text-xs text-muted-foreground">{model.publisher}</span>
              <span className="text-border">·</span>
              {model.category === "external" && <CompatBadge compat={model.compatibility} />}
              <StatusBadge status={model.status} />
            </div>
          </div>

          <div className="flex items-center gap-2 shrink-0">
            {isImageModel ? (
              isDownloaded ? (
                <button
                  onClick={() => router.push('/images')}
                  className="flex items-center gap-2 px-4 py-2 rounded-lg bg-primary text-primary-foreground text-sm font-semibold hover:bg-primary/90 transition-all"
                >
                  <MessageSquarePlus size={15} /> Open Image Studio
                </button>
              ) : (
                <button
                  onClick={handleImageDownload}
                  disabled={loading}
                  className="flex items-center gap-2 px-4 py-2 rounded-lg bg-green-600 text-white text-sm font-semibold hover:bg-green-700 transition-all disabled:opacity-60"
                >
                  {loading ? (
                    <>
                      <div className="w-3 h-3 rounded-full border-2 border-white/30 border-t-white animate-spin" />
                      Downloading...
                    </>
                  ) : (
                    <>
                      <Play size={14} /> Download GGUF
                    </>
                  )}
                </button>
              )
            ) : shouldShowChat ? (
              <button
                onClick={() => {
                  const newId = createConversation();
                    router.push(`/chat?conversationId=${newId}`);
                }}
                className="flex items-center gap-2 px-4 py-2 rounded-lg bg-primary text-primary-foreground text-sm font-semibold hover:bg-primary/90 transition-all"
              >
                <MessageSquarePlus size={15} /> Chat
              </button>
            ) : shouldShowLoad ? (
              <button
                onClick={handleLoad}
                disabled={loading}
                className="flex items-center gap-2 px-4 py-2 rounded-lg bg-primary text-primary-foreground text-sm font-semibold hover:bg-primary/90 transition-all disabled:opacity-60"
              >
                {loading ? (
                  <>
                    <div className="w-3 h-3 rounded-full border-2 border-white/30 border-t-white animate-spin" />
                    Loading...
                  </>
                ) : (
                  <>
                    <Play size={14} /> Load Model
                  </>
                )}
              </button>
            ) : (
              <button
                onClick={handleDownload}
                disabled={loading || !selectedQuantization}
                className="flex items-center gap-2 px-4 py-2 rounded-lg bg-green-600 text-white text-sm font-semibold hover:bg-green-700 transition-all disabled:opacity-60"
              >
                {loading ? (
                  <>
                    <div className="w-3 h-3 rounded-full border-2 border-white/30 border-t-white animate-spin" />
                    Downloading...
                  </>
                ) : (
                  <>
                    <Play size={14} /> Download {selectedQuantization ? `(${selectedQuantization})` : ""}
                  </>
                )}
              </button>
            )}
          </div>
        </div>

        <p className="mt-4 text-sm text-muted-foreground leading-relaxed">{model.description}</p>

        {/* Tags */}
        <div className="flex flex-wrap gap-1.5 mt-4">
          {model.tags.map((tag) => (
            <span
              key={tag}
              className="px-2 py-0.5 rounded-md bg-secondary text-[10px] font-bold text-muted-foreground uppercase tracking-wide border border-border"
            >
              {tag}
            </span>
          ))}
        </div>
      </div>

      {/* Stats grid */}
      <div className="p-6 border-b border-border">
        <h3 className="text-[10px] font-bold text-muted-foreground uppercase tracking-widest mb-4">Specifications</h3>
        
        {/* Parameters and Quantization Selector */}
        <div className="grid grid-cols-2 xl:grid-cols-3 gap-3">
          <QuantizationSelector 
            modelId={model.modelRepoId || model.id}
            onQuantizationChange={(quant, size, parameters) => {
              setSelectedQuantization(quant)
              setSelectedSize(size)
              console.log(`Selected ${quant}: ${size} GB (${parameters})`)
            }}
            onQualityChange={(quality) => {
              setQualityLevel(quality)
            }}
          />
          <StatCard icon={<Cpu size={15} />} label="Context Length" value={`${(model.contextLength / 1000).toFixed(0)}K tokens`} />
          <StatCard icon={<MemoryStick size={15} />} label="RAM Required" value={`${model.ramRequired || 0} GB`} />
          <StatCard icon={<Zap size={15} />} label="Inference Speed" value={model.speed ?? "—"} />
          <StatCard icon={<FileCode size={15} />} label="Architecture" value={(model.architecture || 'llama.cpp').split("For")[0]} />
        </div>
      </div>

      {/* Details */}
      <div className="p-6 grid grid-cols-2 gap-6">
        <div>
          <h3 className="text-[10px] font-bold text-muted-foreground uppercase tracking-widest mb-3">Model Info</h3>
          <div className="space-y-2.5">
            {[
              { label: "Publisher", value: model.publisher },
              { label: "Architecture", value: model.architecture },
              { label: "Quantization", value: model.quantization },
              { label: "License", value: model.license },
            ].map(({ label, value }) => (
              <div key={label} className="flex items-center justify-between py-2 border-b border-border/50">
                <span className="text-xs text-muted-foreground">{label}</span>
                <span className="text-xs font-semibold text-foreground">{value}</span>
              </div>
            ))}
          </div>
        </div>

        <div>
          <h3 className="text-[10px] font-bold text-muted-foreground uppercase tracking-widest mb-3">Performance</h3>
          <div className="space-y-3">
            <div>
              <div className="flex justify-between text-xs mb-1.5">
                <span className="text-muted-foreground">RAM Usage</span>
                <span className="font-semibold text-foreground">{model.ramRequired} GB</span>
              </div>
              <div className="h-1.5 rounded-full bg-secondary overflow-hidden">
                <div
                  className="h-full rounded-full bg-primary transition-all"
                  style={{ width: `${Math.min((model.ramRequired / 32) * 100, 100)}%` }}
                />
              </div>
            </div>
            {model.vramRequired !== undefined && (
              <div>
                <div className="flex justify-between text-xs mb-1.5">
                  <span className="text-muted-foreground">VRAM</span>
                  <span className="font-semibold text-foreground">{model.vramRequired} GB</span>
                </div>
                <div className="h-1.5 rounded-full bg-secondary overflow-hidden">
                  <div
                    className="h-full rounded-full bg-amber-500 transition-all"
                    style={{ width: `${Math.min((model.vramRequired / 24) * 100, 100)}%` }}
                  />
                </div>
              </div>
            )}
            <div>
              <div className="flex justify-between text-xs mb-1.5">
                <span className="text-muted-foreground">Context</span>
                <span className="font-semibold text-foreground">{(model.contextLength / 1000).toFixed(0)}K</span>
              </div>
              <div className="h-1.5 rounded-full bg-secondary overflow-hidden">
                <div
                  className="h-full rounded-full bg-green-500 transition-all"
                  style={{ width: `${Math.min((model.contextLength / 131072) * 100, 100)}%` }}
                />
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Local Copies — each downloaded quant with trash button */}
      {localQuants.length > 0 && (
        <div className="p-6 border-t border-border">
          <h3 className="text-[10px] font-bold text-muted-foreground uppercase tracking-widest mb-3">Local Copies</h3>
          <div className="space-y-1.5">
            {localQuants.map((quant) => (
              <div key={quant} className="flex items-center justify-between px-3 py-2 rounded-lg bg-[#0f0f0f] border border-border">
                <div className="flex items-center gap-3">
                  <span className="px-1.5 py-0.5 rounded text-[9px] font-bold bg-green-500/10 text-green-400 border border-green-500/20">LOCAL</span>
                  <span className="text-xs font-mono font-bold text-foreground">{quant}</span>
                </div>
                <button
                  onClick={() => setConfirmDelete(quant)}
                  disabled={deletingQuant === quant}
                  className="p-1.5 rounded-md text-muted-foreground hover:text-destructive hover:bg-destructive/10 transition-all disabled:opacity-40"
                  title={`Delete ${quant} from disk`}
                >
                  {deletingQuant === quant
                    ? <div className="w-3 h-3 rounded-full border-2 border-muted-foreground/30 border-t-muted-foreground animate-spin" />
                    : <Trash2 size={13} />}
                </button>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>

    {/* Confirm Delete Dialog */}
    {confirmDelete && (
      <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm">
        <div className="bg-[#1e1e1e] border border-border rounded-xl p-6 w-90 shadow-2xl animate-fade-in-up">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-9 h-9 rounded-lg bg-destructive/10 flex items-center justify-center">
              <Trash2 size={16} className="text-destructive" />
            </div>
            <div>
              <h3 className="text-sm font-bold text-foreground">Delete {confirmDelete}?</h3>
              <p className="text-xs text-muted-foreground">{model.name} · This will free disk space</p>
            </div>
          </div>
          <div className="flex gap-2 justify-end mt-5">
            <button
              onClick={() => setConfirmDelete(null)}
              className="px-4 py-2 text-xs font-semibold rounded-lg bg-secondary text-foreground hover:bg-secondary/80 transition-all"
            >
              Cancel
            </button>
            <button
              onClick={() => confirmDelete && handleDeleteQuant(confirmDelete)}
              disabled={!!deletingQuant}
              className="px-4 py-2 text-xs font-semibold rounded-lg bg-destructive text-white hover:bg-destructive/90 transition-all flex items-center gap-1.5 disabled:opacity-60"
            >
              <Trash2 size={12} /> Delete
            </button>
          </div>
        </div>
      </div>
    )}
  </>
  )
}

export function ModelsView() {
  const { models, localModels, selectedModelDetail, setSelectedModelDetail } = useApp()
  const [search, setSearch] = useState("")
  const [filterStatus, setFilterStatus] = useState<"all" | "loaded" | "available">("all")

  const filtered = (filterStatus === "available" ? localModels : models).filter((m) => {
    const matchSearch =
      m.name.toLowerCase().includes(search.toLowerCase()) ||
      m.family.toLowerCase().includes(search.toLowerCase())

    if (filterStatus === "loaded") return m.status === "loaded" && matchSearch
    return matchSearch
  })

  const categories = [...new Set(models.map((m) => m.category || "General LLM"))]

  const categoryLabels: Record<string, string> = {
    "text-generation": "Text LLMs",
    "vision": "Vision models",
    "coding": "Coding specialists",
    "reasoning": "Reasoning / Thinking",
    "text-to-image": "Image Generation",
    "external": "Cloud Providers",
  }

  return (
    <div className="flex h-full animate-fade-in-up">
      {/* Left panel */}
      <div className="w-[320px] shrink-0 flex flex-col border-r border-border bg-background">
        <div className="p-4 border-b border-border">
          <h2 className="text-base font-bold text-foreground mb-3">Model Library</h2>

          {/* Search */}
          <div className="relative mb-3">
            <Search size={13} className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground" />
            <input
              type="text"
              placeholder="Search models..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="w-full bg-[#1e1e1e] border border-border rounded-lg py-2 pl-8 pr-3 text-xs text-foreground placeholder:text-muted-foreground focus:border-primary/50 transition-all"
            />
          </div>

          {/* Filter tabs */}
          <div className="flex gap-1 bg-secondary rounded-lg p-0.5">
            {(["all", "loaded", "available"] as const).map((f) => (
              <button
                key={f}
                onClick={() => setFilterStatus(f)}
                className={cn(
                  "flex-1 py-1 text-[10px] font-bold rounded-md capitalize transition-all",
                  filterStatus === f
                    ? "bg-[#1e1e1e] text-foreground shadow-sm"
                    : "text-muted-foreground hover:text-foreground"
                )}
              >
                {f}
              </button>
            ))}
          </div>
        </div>

        <div className="flex-1 overflow-y-auto p-2 space-y-3">
          {categories.map((category) => {
            const categoryModels = filtered.filter((m) => m.category === category || (!m.category && category === "General LLM"))
            if (!categoryModels.length) return null
            return (
              <div key={category}>
                <div className="px-2 py-1 text-[10px] font-bold text-muted-foreground uppercase tracking-widest">
                  {categoryLabels[category] || category}
                </div>
                <div className="space-y-0.5">
                  {categoryModels.map((m) => (
                    <ModelListItem
                      key={m.id}
                      model={m}
                      isSelected={selectedModelDetail?.id === m.id}
                      onSelect={() => setSelectedModelDetail(m)}
                    />
                  ))}
                </div>
              </div>
            )
          })}
          {!filtered.length && (
            <div className="text-center py-10 text-xs text-muted-foreground">No models found</div>
          )}
        </div>
      </div>

      {/* Right: detail */}
      <div className="flex-1 overflow-y-auto bg-[#121212]">
        {selectedModelDetail ? (
          <ModelDetailPanel model={selectedModelDetail} />
        ) : (
          <div className="flex flex-col items-center justify-center h-full gap-3 text-muted-foreground">
            <Box size={40} className="opacity-20" />
            <p className="text-sm">Select a model to view details</p>
          </div>
        )}
      </div>
    </div>
  )
}
