"use client"

import { useState, useEffect } from "react"
import { X, Download, Pause, Play, XCircle, Clock, HardDrive, Gauge, CheckCircle2, History } from "lucide-react"
import { cn } from "@/services/utils"
import { useDownloads } from "@/services/downloadContext"
import { API } from "@/services/api"

interface DownloadLog {
  id: string
  model_id: string
  model_name: string
  quantization: string
  status: "completed" | "error" | "downloading"
  size_bytes: number
  duration_seconds: number
  error?: string
  started_at: string
  completed_at?: string
}

export function DownloadDialog() {
  const { downloads, dialogOpen, setDialogOpen, pauseDownload, resumeDownload, cancelDownload } = useDownloads()
  const [filter, setFilter] = useState("")
  const [tab, setTab] = useState<"active" | "history">("active")
  const [logs, setLogs] = useState<DownloadLog[]>([])
  const [logsLoading, setLogsLoading] = useState(false)

  useEffect(() => {
    if (dialogOpen && tab === "history") {
      setLogsLoading(true)
      API.fetchDownloadLogs().then(setLogs).catch(console.warn).finally(() => setLogsLoading(false))
    }
  }, [dialogOpen, tab])

  if (!dialogOpen) return null

  const ongoing = downloads.filter((d) => d.status === "ongoing" || d.status === "paused")
  const completed = downloads.filter((d) => d.status === "completed")

  const filteredOngoing = ongoing.filter((d) =>
    d.modelName.toLowerCase().includes(filter.toLowerCase())
  )
  const filteredCompleted = completed.filter((d) =>
    d.modelName.toLowerCase().includes(filter.toLowerCase())
  )

  const formatBytes = (bytes: number) => {
    if (!bytes || bytes === 0) return "—"
    const k = 1024
    const sizes = ["B", "KB", "MB", "GB"]
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return `${(bytes / Math.pow(k, i)).toFixed(2)} ${sizes[i]}`
  }

  const formatSpeed = (bytesPerSec: number) => `${formatBytes(bytesPerSec)}/s`

  const formatDate = (iso: string) => {
    const d = new Date(iso)
    return d.toLocaleDateString(undefined, { month: "short", day: "numeric" }) + " " +
      d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" })
  }

  const formatDuration = (secs: number) => {
    if (!secs) return "—"
    if (secs < 60) return `${secs.toFixed(0)}s`
    return `${Math.floor(secs / 60)}m ${(secs % 60).toFixed(0)}s`
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
      <div className="bg-[#0a0a0a] border border-border rounded-xl w-[520px] max-h-[680px] shadow-2xl flex flex-col animate-fade-in-up">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-border">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-primary/10 flex items-center justify-center">
              <Download size={16} className="text-primary" />
            </div>
            <h2 className="text-sm font-bold text-foreground">Downloads</h2>
          </div>
          <button
            onClick={() => setDialogOpen(false)}
            className="p-1.5 rounded-lg text-muted-foreground hover:text-foreground hover:bg-secondary transition-all"
          >
            <X size={16} />
          </button>
        </div>

        {/* Tabs */}
        <div className="flex border-b border-border">
          {(["active", "history"] as const).map((t) => (
            <button
              key={t}
              onClick={() => setTab(t)}
              className={cn(
                "flex-1 py-2.5 text-xs font-bold capitalize transition-all flex items-center justify-center gap-1.5",
                tab === t
                  ? "text-foreground border-b-2 border-primary"
                  : "text-muted-foreground hover:text-foreground"
              )}
            >
              {t === "history" ? <History size={13} /> : <Download size={13} />}
              {t === "active" ? `Active (${ongoing.length})` : "History"}
            </button>
          ))}
        </div>

        {/* Filter */}
        <div className="px-4 pt-3 pb-0">
          <input
            type="text"
            placeholder="Filter..."
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            className="w-full bg-[#0f0f0f] border border-border rounded-lg px-3 py-2 text-xs text-foreground placeholder:text-muted-foreground focus:border-primary/50 transition-all"
          />
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {tab === "active" ? (
            <>
              {/* Ongoing */}
              {filteredOngoing.length > 0 && (
                <div>
                  <h3 className="text-[10px] font-bold text-muted-foreground uppercase tracking-wider mb-2">Ongoing</h3>
                  <div className="space-y-2">
                    {filteredOngoing.map((download) => (
                      <div key={download.modelId} className="bg-[#0f0f0f] border border-border rounded-lg p-3">
                        <div className="flex items-start justify-between mb-2">
                          <div>
                            <div className="text-xs font-semibold text-foreground">{download.modelName}</div>
                            <div className="text-[10px] text-muted-foreground">{download.quantization}</div>
                          </div>
                          <div className="flex items-center gap-1.5">
                            {download.status === "paused" ? (
                              <button onClick={() => resumeDownload(download.modelId)} className="p-1.5 rounded-lg bg-secondary hover:bg-secondary/80 text-foreground transition-all">
                                <Play size={14} />
                              </button>
                            ) : (
                              <button onClick={() => pauseDownload(download.modelId)} className="p-1.5 rounded-lg bg-secondary hover:bg-secondary/80 text-foreground transition-all">
                                <Pause size={14} />
                              </button>
                            )}
                            <button onClick={() => cancelDownload(download.modelId)} className="p-1.5 rounded-lg bg-destructive/10 hover:bg-destructive/20 text-destructive transition-all">
                              <XCircle size={14} />
                            </button>
                          </div>
                        </div>
                        <div className="w-full h-1.5 bg-secondary rounded-full overflow-hidden mb-2">
                          <div
                            className={cn("h-full transition-all duration-300", download.status === "paused" ? "bg-yellow-500" : "bg-primary")}
                            style={{ width: `${download.progress}%` }}
                          />
                        </div>
                        <div className="flex items-center gap-3 text-[10px]">
                          <div className="flex items-center gap-1 text-muted-foreground">
                            <HardDrive size={11} className="text-primary" />
                            <span className="font-mono">{formatBytes(download.downloaded)} / {formatBytes(download.total)}</span>
                            <span className="text-foreground font-semibold">({download.progress.toFixed(1)}%)</span>
                          </div>
                          {download.status === "ongoing" && (
                            <>
                              <div className="flex items-center gap-1 text-muted-foreground">
                                <Gauge size={11} className="text-blue-500" />
                                <span className="font-mono">{formatSpeed(download.speed)}</span>
                              </div>
                              <div className="flex items-center gap-1 text-muted-foreground">
                                <Clock size={11} className="text-green-500" />
                                <span>{download.timeLeft} left</span>
                              </div>
                            </>
                          )}
                          {download.status === "paused" && <span className="text-yellow-500 font-semibold">Paused</span>}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Completed this session */}
              {filteredCompleted.length > 0 && (
                <div>
                  <h3 className="text-[10px] font-bold text-muted-foreground uppercase tracking-wider mb-2">Completed</h3>
                  <div className="space-y-1.5">
                    {filteredCompleted.map((download) => (
                      <div key={download.modelId} className="bg-[#0f0f0f] border border-border rounded-lg p-2.5 flex items-center justify-between">
                        <div className="flex items-center gap-2.5">
                          <div className="w-7 h-7 rounded-lg bg-green-500/10 flex items-center justify-center">
                            <CheckCircle2 size={14} className="text-green-500" />
                          </div>
                          <div>
                            <div className="text-xs font-semibold text-foreground">{download.modelName}</div>
                            <div className="text-[10px] text-muted-foreground">{download.quantization} · {formatBytes(download.total)}</div>
                          </div>
                        </div>
                        <button onClick={() => cancelDownload(download.modelId)} className="p-1 rounded-lg text-muted-foreground hover:text-foreground hover:bg-secondary transition-all">
                          <X size={13} />
                        </button>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {filteredOngoing.length === 0 && filteredCompleted.length === 0 && (
                <div className="text-center py-10">
                  <Download size={24} className="text-muted-foreground mx-auto mb-2 opacity-30" />
                  <p className="text-xs text-muted-foreground">{filter ? "No downloads match your filter" : "No active downloads"}</p>
                </div>
              )}
            </>
          ) : (
            /* History Tab */
            <>
              {logsLoading ? (
                <div className="flex items-center justify-center py-10">
                  <div className="w-6 h-6 rounded-full border-2 border-primary/30 border-t-primary animate-spin" />
                </div>
              ) : logs.filter(l => l.model_name?.toLowerCase().includes(filter.toLowerCase())).length === 0 ? (
                <div className="text-center py-10">
                  <History size={24} className="text-muted-foreground mx-auto mb-2 opacity-30" />
                  <p className="text-xs text-muted-foreground">No download history yet</p>
                </div>
              ) : (
                <div className="space-y-1.5">
                  {logs
                    .filter(l => l.model_name?.toLowerCase().includes(filter.toLowerCase()))
                    .map((log) => (
                      <div key={log.id} className="bg-[#0f0f0f] border border-border rounded-lg p-2.5 flex items-center justify-between">
                        <div className="flex items-center gap-2.5">
                          <div className={cn(
                            "w-7 h-7 rounded-lg flex items-center justify-center",
                            log.status === "completed" ? "bg-green-500/10" : "bg-destructive/10"
                          )}>
                            {log.status === "completed"
                              ? <CheckCircle2 size={14} className="text-green-500" />
                              : <XCircle size={14} className="text-destructive" />}
                          </div>
                          <div>
                            <div className="text-xs font-semibold text-foreground">{log.model_name || log.model_id}</div>
                            <div className="text-[10px] text-muted-foreground">
                              {log.quantization}
                              {log.size_bytes ? ` · ${formatBytes(log.size_bytes)}` : ""}
                              {log.duration_seconds ? ` · ${formatDuration(log.duration_seconds)}` : ""}
                            </div>
                          </div>
                        </div>
                        <div className="text-[10px] text-muted-foreground text-right">
                          {log.started_at ? formatDate(log.started_at) : ""}
                        </div>
                      </div>
                    ))}
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  )
}
