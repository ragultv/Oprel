"use client"

import { createContext, useContext, useState, useCallback, useEffect, ReactNode } from "react"

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

  // Load ongoing downloads on mount
  useEffect(() => {
    async function loadOngoingDownloads() {
      try {
        const response = await fetch('/api/downloads')
        if (response.ok) {
          const data = await response.json()
          // Convert backend format to frontend format
          const ongoingDownloads = data.downloads
            .filter((d: any) => d.status === 'downloading')
            .map((d: any) => ({
              modelId: `${d.model_id}-${d.quantization}`,
              modelName: d.model_id.split('/').pop() || d.model_id,
              quantization: d.quantization,
              progress: d.progress,
              downloaded: d.downloaded,
              total: d.total,
              speed: d.speed,
              timeLeft: formatTime(d.eta),
              status: 'ongoing' as const,
            }))
          
          if (ongoingDownloads.length > 0) {
            setDownloads(ongoingDownloads)
          }
        }
      } catch (error) {
        console.warn('Failed to load ongoing downloads:', error)
      }
    }
    
    loadOngoingDownloads()
  }, [])

  const formatTime = (seconds: number) => {
    if (seconds < 60) return `${Math.ceil(seconds)}s`
    const mins = Math.floor(seconds / 60)
    const secs = Math.ceil(seconds % 60)
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  const addDownload = useCallback((download: DownloadProgress) => {
    setDownloads((prev) => {
      // Check if already exists
      const exists = prev.find((d) => d.modelId === download.modelId)
      if (exists) {
        return prev.map((d) => (d.modelId === download.modelId ? download : d))
      }
      return [...prev, download]
    })
  }, [])

  const updateDownload = useCallback((modelId: string, updates: Partial<DownloadProgress>) => {
    setDownloads((prev) =>
      prev.map((d) => (d.modelId === modelId ? { ...d, ...updates } : d))
    )
  }, [])

  const removeDownload = useCallback((modelId: string) => {
    setDownloads((prev) => prev.filter((d) => d.modelId !== modelId))
  }, [])

  const pauseDownload = useCallback((modelId: string) => {
    updateDownload(modelId, { status: "paused" })
  }, [updateDownload])

  const resumeDownload = useCallback((modelId: string) => {
    updateDownload(modelId, { status: "ongoing" })
  }, [updateDownload])

  const cancelDownload = useCallback((modelId: string) => {
    removeDownload(modelId)
  }, [removeDownload])

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
