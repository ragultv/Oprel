"use client"

import { createContext, useContext, useState, useCallback, useEffect, useRef, ReactNode } from "react"
import { API } from "@/services/api"
import { useToast } from "@/hooks/use-toast"
import { useApp } from "@/services/context"

export interface DownloadProgress {
  modelId: string
  modelName: string
  quantization: string
  progress: number // 0-100
  downloaded: number // bytes
  total: number // bytes
  speed: number // bytes/sec
  timeLeft: string // formatted time
  status: "ongoing" | "completed" | "paused" | "error"
  error?: string
  downloadId?: string
}

interface DownloadContextType {
  downloads: DownloadProgress[]
  addDownload: (download: DownloadProgress) => void
  updateDownload: (modelId: string, updates: Partial<DownloadProgress>) => void
  removeDownload: (modelId: string) => void
  pauseDownload: (modelId: string) => void
  resumeDownload: (modelId: string) => void
  cancelDownload: (modelId: string) => void
  getOngoingCount: () => number
  dialogOpen: boolean
  setDialogOpen: (open: boolean) => void
}

const DownloadContext = createContext<DownloadContextType | undefined>(undefined)

export function DownloadProvider({ children }: { children: ReactNode }) {
  const [downloads, setDownloads] = useState<DownloadProgress[]>([])
  const [dialogOpen, setDialogOpen] = useState(false)
  const activeStreams = useRef<Record<string, () => void>>({})
  const { toast } = useToast()

  let refreshModelsGlobal: () => Promise<void> = async () => {};
  try {
    const app = useApp();
    if (app) {
      refreshModelsGlobal = app.refreshModels;
    }
  } catch (e) {
    console.warn("DownloadProvider: AppContext not available");
  }

  const formatTime = (seconds: number) => {
    if (seconds < 60) return `${Math.ceil(seconds)}s`
    const mins = Math.floor(seconds / 60)
    const secs = Math.ceil(seconds % 60)
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  const startStreaming = useCallback((modelId: string, downloadId: string, modelName: string, quantization: string) => {
    if (activeStreams.current[modelId]) return;

    console.log(`Starting SSE progress stream for ${modelId} (${downloadId})`);
    const cleanup = API.streamDownloadProgress(
      downloadId,
      (progress) => {
        setDownloads((prev) =>
          prev.map((d) =>
            d.modelId === modelId
              ? {
                  ...d,
                  progress: progress.progress,
                  downloaded: progress.downloaded,
                  total: progress.total,
                  speed: progress.speed,
                  timeLeft: formatTime(progress.eta),
                  status: "ongoing",
                }
              : d
          )
        )
      },
      () => {
        setDownloads((prev) =>
          prev.map((d) =>
            d.modelId === modelId
              ? {
                  ...d,
                  status: "completed",
                  progress: 100,
                  timeLeft: "0s",
                }
              : d
          )
        )
        toast({
          title: "Download Complete",
          description: `${modelName} (${quantization}) is ready to use`,
        })
        refreshModelsGlobal();
        
        if (activeStreams.current[modelId]) {
          activeStreams.current[modelId]()
          delete activeStreams.current[modelId]
        }
      },
      (error) => {
        const isRestartRequired = error.includes("restart the server")
        setDownloads((prev) =>
          prev.map((d) =>
            d.modelId === modelId
              ? {
                  ...d,
                  status: isRestartRequired ? "ongoing" : "error",
                  error: error,
                }
              : d
          )
        )
        if (isRestartRequired) {
          toast({
            title: "Server Restart Required",
            description: "Download is running in background. Restart server for real-time progress: pkill -f oprel.server.daemon && oprel start",
            variant: "default",
            duration: 10000,
          })
        } else {
          toast({
            title: "Download Failed",
            description: error,
            variant: "destructive",
          })
          if (activeStreams.current[modelId]) {
            activeStreams.current[modelId]()
            delete activeStreams.current[modelId]
          }
        }
      }
    )

    activeStreams.current[modelId] = cleanup
  }, [toast, refreshModelsGlobal])

  // Load ongoing downloads on mount
  useEffect(() => {
    async function loadOngoingDownloads() {
      try {
        const response = await fetch('/api/downloads')
        if (response.ok) {
          const data = await response.json()
          const ongoingDownloads = data.downloads
            .filter((d: any) => d.status === 'downloading' || d.status === 'paused')
            .map((d: any) => ({
              modelId: `${d.model_id}-${d.quantization}`,
              modelName: d.model_id.split('/').pop() || d.model_id,
              quantization: d.quantization,
              progress: d.progress,
              downloaded: d.downloaded,
              total: d.total,
              speed: d.speed,
              timeLeft: formatTime(d.eta),
              status: d.status === 'paused' ? ('paused' as const) : ('ongoing' as const),
              downloadId: d.download_id,
            }))
          
          if (ongoingDownloads.length > 0) {
            setDownloads(ongoingDownloads)
            ongoingDownloads.forEach((d: any) => {
              if (d.status === 'ongoing' && d.downloadId) {
                startStreaming(d.modelId, d.downloadId, d.modelName, d.quantization)
              }
            })
          }
        }
      } catch (error) {
        console.warn('Failed to load ongoing downloads:', error)
      }
    }
    
    loadOngoingDownloads()
  }, [startStreaming])

  useEffect(() => {
    return () => {
      Object.values(activeStreams.current).forEach((cleanup) => cleanup())
    }
  }, [])

  const addDownload = useCallback((download: DownloadProgress) => {
    setDownloads((prev) => {
      const exists = prev.find((d) => d.modelId === download.modelId)
      if (exists) {
        return prev.map((d) => (d.modelId === download.modelId ? download : d))
      }
      return [...prev, download]
    })
    if (download.status === "ongoing" && download.downloadId) {
      startStreaming(download.modelId, download.downloadId, download.modelName, download.quantization)
    }
  }, [startStreaming])

  const updateDownload = useCallback((modelId: string, updates: Partial<DownloadProgress>) => {
    setDownloads((prev) =>
      prev.map((d) => (d.modelId === modelId ? { ...d, ...updates } : d))
    )
  }, [])

  const removeDownload = useCallback((modelId: string) => {
    setDownloads((prev) => prev.filter((d) => d.modelId !== modelId))
  }, [])

  const pauseDownload = useCallback(async (modelId: string) => {
    setDownloads((prev) => {
      const download = prev.find((d) => d.modelId === modelId)
      if (download && download.downloadId) {
        API.pauseDownload(download.downloadId).catch(console.error)
      }
      if (activeStreams.current[modelId]) {
        activeStreams.current[modelId]()
        delete activeStreams.current[modelId]
      }
      return prev.map((d) => (d.modelId === modelId ? { ...d, status: "paused" as const, speed: 0 } : d))
    })
  }, [])

  const resumeDownload = useCallback(async (modelId: string) => {
    setDownloads((prev) => {
      const download = prev.find((d) => d.modelId === modelId)
      if (download && download.downloadId) {
        API.resumeDownload(download.downloadId)
          .then(() => {
            startStreaming(modelId, download.downloadId!, download.modelName, download.quantization)
          })
          .catch(console.error)
      }
      return prev.map((d) => (d.modelId === modelId ? { ...d, status: "ongoing" as const } : d))
    })
  }, [startStreaming])

  const cancelDownload = useCallback(async (modelId: string) => {
    setDownloads((prev) => {
      const download = prev.find((d) => d.modelId === modelId)
      if (download && download.downloadId) {
        API.cancelDownload(download.downloadId).catch(console.error)
      }
      if (activeStreams.current[modelId]) {
        activeStreams.current[modelId]()
        delete activeStreams.current[modelId]
      }
      return prev.filter((d) => d.modelId !== modelId)
    })
  }, [])

  const getOngoingCount = useCallback(() => {
    return downloads.filter((d) => d.status === "ongoing").length
  }, [downloads])

  return (
    <DownloadContext.Provider
      value={{
        downloads,
        addDownload,
        updateDownload,
        removeDownload,
        pauseDownload,
        resumeDownload,
        cancelDownload,
        getOngoingCount,
        dialogOpen,
        setDialogOpen,
      }}
    >
      {children}
    </DownloadContext.Provider>
  )
}

export function useDownloads() {
  const context = useContext(DownloadContext)
  if (!context) {
    throw new Error("useDownloads must be used within DownloadProvider")
  }
  return context
}
