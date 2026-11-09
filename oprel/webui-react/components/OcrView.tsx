"use client"

import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { useRouter } from "next/navigation"
import {
  CheckCircle2,
  ClipboardCopy,
  Download,
  FileJson,
  FileText,
  Loader2,
  ScanText,
  Trash2,
  Upload,
  X,
  Zap,
  ArrowLeft,
  Image as ImageIcon,
  Plus,
  Layers,
  ChevronDown,
  ZoomIn,
  ZoomOut,
  RotateCcw,
  RefreshCw,
  PanelRightClose,
  PanelRightOpen,
  Copy,
  Check,
  SlidersHorizontal,
  History as HistoryIcon,
  Table as TableIcon,
  AlignLeft,
  Eye,
  FileCode,
} from "lucide-react"
import { API, OCR, type OcrJob, type OcrResult, type OcrStatus } from "@/services/api"
import { cn } from "@/services/utils"

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

type OcrResultWithNorm = OcrResult & {
  bbox_norm?: { left: number; top: number; width: number; height: number }
}

// ─────────────────────────────────────────────────────────────────────────────
// Confidence colour helpers
// ─────────────────────────────────────────────────────────────────────────────

function confBadge(conf: number) {
  if (conf >= 0.9) return "bg-emerald-500/15 text-emerald-400 border-emerald-500/20"
  if (conf >= 0.7) return "bg-amber-500/15 text-amber-400 border-amber-500/20"
  return "bg-red-500/15 text-red-400 border-red-500/20"
}

function confBorder(conf: number) {
  if (conf >= 0.9) return "rgba(34,197,94,0.7)"
  if (conf >= 0.7) return "rgba(245,158,11,0.7)"
  return "rgba(239,68,68,0.7)"
}

function confBg(conf: number) {
  if (conf >= 0.9) return "rgba(34,197,94,0.15)"
  if (conf >= 0.7) return "rgba(245,158,11,0.15)"
  return "rgba(239,68,68,0.15)"
}

function formatDate(iso: string) {
  try {
    const d = new Date(iso)
    if (isNaN(d.getTime())) return ""
    return (
      d.toLocaleDateString([], { month: "short", day: "numeric" }) +
      " " +
      d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
    )
  } catch {
    return ""
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Detect if a set of results looks like a table
// ─────────────────────────────────────────────────────────────────────────────

function groupIntoTableRows(results: OcrResultWithNorm[]): OcrResultWithNorm[][] | null {
  if (results.length < 4) return null
  const sorted = [...results].sort((a, b) => {
    const at = a.bbox_norm?.top ?? 0
    const bt = b.bbox_norm?.top ?? 0
    return at - bt
  })
  const rows: OcrResultWithNorm[][] = []
  let currentRow: OcrResultWithNorm[] = [sorted[0]]
  let rowTop = sorted[0].bbox_norm?.top ?? 0

  for (let i = 1; i < sorted.length; i++) {
    const top = sorted[i].bbox_norm?.top ?? 0
    if (top - rowTop < 0.028) {
      currentRow.push(sorted[i])
    } else {
      rows.push([...currentRow].sort((a, b) => (a.bbox_norm?.left ?? 0) - (b.bbox_norm?.left ?? 0)))
      currentRow = [sorted[i]]
      rowTop = top
    }
  }
  if (currentRow.length) {
    rows.push([...currentRow].sort((a, b) => (a.bbox_norm?.left ?? 0) - (b.bbox_norm?.left ?? 0)))
  }

  const multiColRows = rows.filter(r => r.length >= 2).length
  if (multiColRows < rows.length * 0.5) return null
  return rows
}

// ─────────────────────────────────────────────────────────────────────────────
// Setup Screen
// ─────────────────────────────────────────────────────────────────────────────

function SetupScreen({ onReady }: { onReady: () => void }) {
  const router = useRouter()

  return (
    <div className="h-full overflow-y-auto bg-[linear-gradient(180deg,#0e0e0e_0%,#111_50%,#0b0b0b_100%)] text-foreground flex items-center justify-center p-6">
      <div className="mx-auto max-w-2xl px-6 py-12 flex flex-col items-center text-center space-y-8">
        <div className="flex h-20 w-20 items-center justify-center rounded-3xl border border-primary/30 bg-primary/10 shadow-lg shadow-primary/10">
          <ScanText size={36} className="text-primary" />
        </div>

        <div className="space-y-3">
          <h1 className="text-4xl font-black tracking-tight">OCR — Text Extraction</h1>
          <p className="text-muted-foreground text-base leading-7 max-w-lg">
            Extract text from images, screenshots, invoices, and documents using <strong className="text-foreground">dedicated OCR models</strong>.
            Install an OCR model from your local model library to begin.
          </p>
        </div>

        <div className="grid grid-cols-2 gap-3 w-full text-left text-sm">
          {["Local GGUF OCR Model", "Markdown & Table Extraction", "Confidence scoring", "Export TXT / MD / JSON", "Persistent history", "GPU auto-detected"].map(f => (
            <div key={f} className="flex items-center gap-2 rounded-xl border border-border bg-secondary/30 px-4 py-3 text-muted-foreground">
              <CheckCircle2 size={14} className="text-primary shrink-0" />
              {f}
            </div>
          ))}
        </div>

        <div className="flex flex-wrap items-center justify-center gap-3">
          <button
            onClick={() => router.push("/models?category=OCR")}
            className="inline-flex items-center gap-2 rounded-2xl bg-primary px-7 py-4 text-base font-semibold text-primary-foreground hover:bg-primary/90 transition-all shadow-lg shadow-primary/20"
          >
            <Layers size={18} />
            <span>Browse OCR Models</span>
          </button>
        </div>
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// Enhanced Results Panel with Modern Tabs (Markdown, Text, Table, Lines, JSON)
// ─────────────────────────────────────────────────────────────────────────────

function ResultsPanel({
  results,
  selectedIdx,
  onHover,
  onSelect,
  minConfidence,
  job,
  autoDetectTable,
}: {
  results: OcrResultWithNorm[]
  selectedIdx: number | null
  onHover: (i: number | null) => void
  onSelect: (i: number) => void
  minConfidence: number
  job: OcrJob
  autoDetectTable: boolean
}) {
  const rowRefs = useRef<(HTMLDivElement | null)[]>([])
  const [copied, setCopied] = useState(false)
  const [resultTab, setResultTab] = useState<"text" | "markdown" | "table" | "lines" | "json">("text")

  useEffect(() => {
    if (selectedIdx != null && rowRefs.current[selectedIdx]) {
      rowRefs.current[selectedIdx]!.scrollIntoView({ behavior: "smooth", block: "nearest" })
    }
  }, [selectedIdx])

  const filtered = useMemo(() => {
    return results.filter(r => r.confidence >= minConfidence / 100)
  }, [results, minConfidence])

  const tableRows = useMemo(() => {
    if (!autoDetectTable) return null
    return groupIntoTableRows(filtered)
  }, [filtered, autoDetectTable])

  // Automatically default to table tab if table is detected on first load
  useEffect(() => {
    if (tableRows && tableRows.length > 0) {
      setResultTab("table")
    }
  }, [tableRows])

  const markdownContent = useMemo(() => {
    if (tableRows) {
      let md = `| ${tableRows[0].map(c => c.text).join(" | ")} |\n`
      md += `| ${tableRows[0].map(() => "---").join(" | ")} |\n`
      for (let i = 1; i < tableRows.length; i++) {
        md += `| ${tableRows[i].map(c => c.text).join(" | ")} |\n`
      }
      return md
    }
    return `# ${job.filename}\n\n${job.full_text}`
  }, [tableRows, job])

  const handleCopyCurrent = () => {
    let textToCopy = job.full_text
    if (resultTab === "markdown") textToCopy = markdownContent
    else if (resultTab === "json") textToCopy = JSON.stringify(results, null, 2)
    else if (resultTab === "table" && tableRows) {
      textToCopy = tableRows.map(r => r.map(c => c.text).join("\t")).join("\n")
    }

    navigator.clipboard.writeText(textToCopy).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    })
  }

  const triggerDownload = (blob: Blob, name: string) => {
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = name
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="flex flex-col h-full min-h-0 bg-[#141414] rounded-2xl border border-white/10 overflow-hidden shadow-2xl">
      {/* Header & Tab Bar */}
      <div className="flex-none px-4 py-3 border-b border-white/10 bg-black/30 flex flex-wrap items-center justify-between gap-3">
        {/* Output Tabs Switcher */}
        <div className="inline-flex items-center p-1 rounded-xl bg-black/50 border border-white/10 gap-1">
          <button
            onClick={() => setResultTab("text")}
            className={cn(
              "px-3 py-1 rounded-lg text-xs font-semibold transition-all flex items-center gap-1.5",
              resultTab === "text"
                ? "bg-primary text-white shadow-sm font-bold"
                : "text-muted-foreground hover:text-foreground"
            )}
          >
            <AlignLeft size={13} />
            <span>Text</span>
          </button>
          <button
            onClick={() => setResultTab("markdown")}
            className={cn(
              "px-3 py-1 rounded-lg text-xs font-semibold transition-all flex items-center gap-1.5",
              resultTab === "markdown"
                ? "bg-primary text-white shadow-sm font-bold"
                : "text-muted-foreground hover:text-foreground"
            )}
          >
            <FileCode size={13} />
            <span>Markdown</span>
          </button>
          {tableRows && (
            <button
              onClick={() => setResultTab("table")}
              className={cn(
                "px-3 py-1 rounded-lg text-xs font-semibold transition-all flex items-center gap-1.5",
                resultTab === "table"
                  ? "bg-primary text-white shadow-sm font-bold"
                  : "text-muted-foreground hover:text-foreground"
              )}
            >
              <TableIcon size={13} />
              <span>Table</span>
            </button>
          )}
          <button
            onClick={() => setResultTab("lines")}
            className={cn(
              "px-3 py-1 rounded-lg text-xs font-semibold transition-all flex items-center gap-1.5",
              resultTab === "lines"
                ? "bg-primary text-white shadow-sm font-bold"
                : "text-muted-foreground hover:text-foreground"
            )}
          >
            <Eye size={13} />
            <span>Lines ({filtered.length})</span>
          </button>
          <button
            onClick={() => setResultTab("json")}
            className={cn(
              "px-3 py-1 rounded-lg text-xs font-semibold transition-all flex items-center gap-1.5",
              resultTab === "json"
                ? "bg-primary text-white shadow-sm font-bold"
                : "text-muted-foreground hover:text-foreground"
            )}
          >
            <FileJson size={13} />
            <span>JSON</span>
          </button>
        </div>

        {/* Action Suite: Copy & Downloads */}
        <div className="flex items-center gap-1.5">
          <button
            onClick={handleCopyCurrent}
            className="inline-flex items-center gap-1.5 rounded-xl border border-white/10 bg-white/5 hover:bg-white/10 px-3 py-1.5 text-xs font-semibold text-foreground transition-all"
            title="Copy content in active view"
          >
            {copied ? <Check size={13} className="text-emerald-400" /> : <Copy size={13} />}
            <span>{copied ? "Copied" : "Copy"}</span>
          </button>
          <button
            onClick={() => triggerDownload(new Blob([job.full_text], { type: "text/plain" }), `${job.filename}.txt`)}
            className="inline-flex items-center gap-1 rounded-xl border border-white/10 bg-white/5 hover:bg-white/10 px-2.5 py-1.5 text-[11px] font-mono font-semibold text-foreground transition-all"
            title="Download TXT"
          >
            TXT
          </button>
          <button
            onClick={() => triggerDownload(new Blob([markdownContent], { type: "text/markdown" }), `${job.filename}.md`)}
            className="inline-flex items-center gap-1 rounded-xl border border-white/10 bg-white/5 hover:bg-white/10 px-2.5 py-1.5 text-[11px] font-mono font-semibold text-foreground transition-all"
            title="Download Markdown"
          >
            MD
          </button>
          <button
            onClick={() => triggerDownload(new Blob([JSON.stringify(results, null, 2)], { type: "application/json" }), `${job.filename}.json`)}
            className="inline-flex items-center gap-1 rounded-xl border border-white/10 bg-white/5 hover:bg-white/10 px-2.5 py-1.5 text-[11px] font-mono font-semibold text-foreground transition-all"
            title="Download JSON"
          >
            JSON
          </button>
        </div>
      </div>

      {/* Stats Bar */}
      <div className="px-4 py-2 border-b border-white/5 bg-black/10 flex items-center justify-between text-[11px] text-muted-foreground font-medium">
        <div className="flex items-center gap-2">
          <span>{job.word_count} words extracted</span>
          <span>&bull;</span>
          <span>{filtered.length} grounded lines</span>
          {minConfidence > 0 && (
            <>
              <span>&bull;</span>
              <span className="text-amber-400 font-mono">Filtered &ge; {minConfidence}% conf</span>
            </>
          )}
        </div>
        <span className="font-mono text-muted-foreground/60">{job.filename}</span>
      </div>

      {/* Scrollable Tab Content View */}
      <div className="flex-1 overflow-y-auto p-4">
        {filtered.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-center py-12 text-sm text-muted-foreground space-y-2">
            <ScanText size={28} className="opacity-40" />
            <p>No text detected above {minConfidence}% confidence threshold.</p>
          </div>
        ) : resultTab === "text" ? (
          /* Clean Monospace Plain Text View */
          <div className="p-4 rounded-2xl bg-black/40 border border-white/5">
            <pre className="text-xs text-foreground font-mono whitespace-pre-wrap leading-relaxed select-text">
              {job.full_text}
            </pre>
          </div>
        ) : resultTab === "markdown" ? (
          /* Markdown Preview */
          <div className="p-4 rounded-2xl bg-black/40 border border-white/5">
            <pre className="text-xs text-foreground font-mono whitespace-pre-wrap leading-relaxed select-text">
              {markdownContent}
            </pre>
          </div>
        ) : resultTab === "table" && tableRows ? (
          /* Formatted Table View */
          <div className="overflow-x-auto rounded-2xl border border-white/10 bg-black/30">
            <table className="w-full text-xs border-collapse">
              <tbody>
                {tableRows.map((row, rIdx) => (
                  <tr key={rIdx} className={cn("border-b border-white/10 transition-colors", rIdx === 0 && "bg-white/5 font-semibold")}>
                    {row.map((cell, cIdx) => {
                      const origIdx = results.indexOf(cell)
                      const isSelected = origIdx === selectedIdx
                      return (
                        <td
                          key={cIdx}
                          ref={el => { rowRefs.current[origIdx] = el }}
                          onClick={() => onSelect(origIdx)}
                          onMouseEnter={() => onHover(origIdx)}
                          onMouseLeave={() => onHover(null)}
                          className={cn(
                            "px-3 py-2.5 text-left border border-white/5 cursor-pointer transition-all align-top leading-5",
                            isSelected
                              ? "ring-1 ring-primary/60 bg-primary/10"
                              : "hover:bg-white/5"
                          )}
                        >
                          <span className={cn("text-foreground", rIdx === 0 && "text-primary font-bold")}>{cell.text}</span>
                          <span className={cn("ml-2 text-[9px] font-semibold px-1 py-0.5 rounded border font-mono", confBadge(cell.confidence))}>
                            {(cell.confidence * 100).toFixed(0)}%
                          </span>
                        </td>
                      )
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : resultTab === "json" ? (
          /* Formatted JSON View */
          <div className="p-4 rounded-2xl bg-black/40 border border-white/5 overflow-x-auto">
            <pre className="text-xs text-muted-foreground font-mono whitespace-pre select-text">
              {JSON.stringify(results, null, 2)}
            </pre>
          </div>
        ) : (
          /* Line by line with bounding box sync */
          <div className="space-y-1.5">
            {filtered.map((r) => {
              const origIdx = results.indexOf(r)
              const isSelected = origIdx === selectedIdx
              return (
                <div
                  key={origIdx}
                  ref={el => { rowRefs.current[origIdx] = el }}
                  id={`ocr_result_${origIdx}`}
                  onClick={() => onSelect(origIdx)}
                  onMouseEnter={() => onHover(origIdx)}
                  onMouseLeave={() => onHover(null)}
                  style={{
                    borderLeftWidth: isSelected ? "3px" : "2px",
                    borderLeftStyle: "solid",
                    borderLeftColor: isSelected ? "var(--primary)" : confBorder(r.confidence),
                  }}
                  className={cn(
                    "flex items-start justify-between gap-3 rounded-xl p-3 cursor-pointer text-xs transition-all",
                    isSelected ? "bg-primary/15 ring-1 ring-primary/40 text-foreground" : "bg-black/30 hover:bg-white/5 border border-white/5"
                  )}
                >
                  <span className="text-foreground leading-relaxed flex-1 select-text">{r.text}</span>
                  <div className="flex items-center gap-1.5 shrink-0 mt-0.5">
                    {r.engine && (
                      <span className="text-[9px] font-semibold uppercase tracking-wider px-1.5 py-0.5 rounded bg-primary/10 text-primary border border-primary/20">
                        {r.engine}
                      </span>
                    )}
                    <span className={cn("text-[10px] font-semibold font-mono rounded-full px-2 py-0.5 border", confBadge(r.confidence))}>
                      {(r.confidence * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>
              )
            })}
          </div>
        )}
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// Image Pane — with Bounding Box Overlay synchronized to results panel
// ─────────────────────────────────────────────────────────────────────────────

function ImagePane({
  src,
  results,
  selectedIdx,
  hoveredIdx,
  minConfidence,
  showBoundingBoxes,
  onBboxClick,
  onBboxHover,
}: {
  src: string
  results: OcrResultWithNorm[]
  selectedIdx: number | null
  hoveredIdx: number | null
  minConfidence: number
  showBoundingBoxes: boolean
  onBboxClick: (i: number) => void
  onBboxHover: (i: number | null) => void
}) {
  const [naturalSize, setNaturalSize] = useState<{ w: number; h: number } | null>(null)
  const [zoom, setZoom] = useState<number>(100)
  const [viewMode, setViewMode] = useState<"width" | "fit" | "actual">("width")
  const bboxRefs = useRef<{ [key: number]: HTMLDivElement | null }>({})
  const scrollContainerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (selectedIdx !== null && bboxRefs.current[selectedIdx]) {
      bboxRefs.current[selectedIdx]?.scrollIntoView({
        behavior: "smooth",
        block: "nearest",
        inline: "nearest"
      })
    }
  }, [selectedIdx])

  const handleImageLoad = (e: React.SyntheticEvent<HTMLImageElement>) => {
    const { naturalWidth, naturalHeight } = e.currentTarget
    setNaturalSize({ w: naturalWidth, h: naturalHeight })
    if (naturalHeight / naturalWidth > 1.15) {
      setViewMode("width")
    } else {
      setViewMode("fit")
    }
  }

  const zoomIn = () => setZoom(prev => Math.min(prev + 25, 300))
  const zoomOut = () => setZoom(prev => Math.max(prev - 25, 50))
  const resetZoom = () => setZoom(100)

  const getWrapperStyle = (): React.CSSProperties => {
    if (viewMode === "fit") {
      return {
        width: "fit-content",
        height: "fit-content",
        maxWidth: "100%",
        maxHeight: "100%",
        transform: zoom !== 100 ? `scale(${zoom / 100})` : undefined,
        transformOrigin: "top center",
      }
    }
    if (viewMode === "actual" && naturalSize) {
      const w = naturalSize.w * (zoom / 100)
      return {
        width: `${w}px`,
        maxWidth: "none",
        flexShrink: 0,
      }
    }
    return {
      width: zoom === 100 ? "100%" : `${zoom}%`,
      maxWidth: zoom <= 100 ? "100%" : undefined,
      flexShrink: 0,
    }
  }

  const getImageClass = () => {
    if (viewMode === "fit") {
      return "block max-w-full max-h-[640px] w-auto h-auto rounded-lg select-none pointer-events-none"
    }
    return "block w-full h-auto rounded-lg select-none pointer-events-none"
  }

  return (
    <div className="rounded-2xl border border-white/10 bg-[#141414] p-4 flex flex-col gap-3 h-full min-h-0 shadow-2xl">
      {/* Image View Toolbar */}
      <div className="flex items-center justify-between flex-none">
        <div className="flex items-center gap-2">
          <span className="text-[11px] font-bold uppercase tracking-[0.15em] text-muted-foreground">Original Document</span>
          {naturalSize && (
            <span className="text-[10px] text-muted-foreground/60 font-mono">
              {naturalSize.w} &times; {naturalSize.h}px
            </span>
          )}
        </div>

        <div className="flex items-center gap-1 bg-black/40 p-1 rounded-xl border border-white/10">
          <button
            onClick={() => { setViewMode("width"); setZoom(100) }}
            className={cn(
              "px-2.5 py-1 rounded-lg text-[10px] font-semibold transition-all",
              viewMode === "width" ? "bg-primary text-white shadow" : "text-muted-foreground hover:text-foreground"
            )}
            title="Fit to pane width"
          >
            Fit Width
          </button>
          <button
            onClick={() => { setViewMode("actual"); setZoom(100) }}
            className={cn(
              "px-2.5 py-1 rounded-lg text-[10px] font-semibold transition-all",
              viewMode === "actual" ? "bg-primary text-white shadow" : "text-muted-foreground hover:text-foreground"
            )}
            title="View at 100% resolution"
          >
            100%
          </button>
          <div className="w-[1px] h-3 bg-white/10 mx-0.5" />
          <button
            onClick={zoomOut}
            disabled={zoom <= 50}
            className="p-1 rounded-md text-muted-foreground hover:text-foreground disabled:opacity-30 transition-colors"
            title="Zoom out"
          >
            <ZoomOut size={12} />
          </button>
          <span className="text-[10px] font-mono px-1 min-w-[34px] text-center text-muted-foreground">
            {zoom}%
          </span>
          <button
            onClick={zoomIn}
            disabled={zoom >= 300}
            className="p-1 rounded-md text-muted-foreground hover:text-foreground disabled:opacity-30 transition-colors"
            title="Zoom in"
          >
            <ZoomIn size={12} />
          </button>
          {zoom !== 100 && (
            <button
              onClick={resetZoom}
              className="p-1 rounded-md text-muted-foreground hover:text-foreground transition-colors"
              title="Reset zoom"
            >
              <RotateCcw size={11} />
            </button>
          )}
        </div>
      </div>

      {/* Scrollable image viewport */}
      <div
        ref={scrollContainerRef}
        className="flex-1 overflow-auto rounded-xl border border-white/5 bg-black/50 p-3 flex justify-center items-start min-h-0 relative select-none"
      >
        <div
          className="relative inline-block transition-all duration-150"
          style={getWrapperStyle()}
        >
          <img
            src={src}
            alt="Uploaded Document"
            onLoad={handleImageLoad}
            className={getImageClass()}
          />

          {/* Bbox overlays — strictly anchored to image bitmap space */}
          {showBoundingBoxes && results.map((r, i) => {
            if (!r.bbox_norm) return null
            if (r.grounded === false) return null
            if (r.confidence < minConfidence / 100) return null
            const { left, top, width, height } = r.bbox_norm
            const isSelected = i === selectedIdx
            const isHovered = i === hoveredIdx
            const active = isSelected || isHovered

            return (
              <div
                key={i}
                ref={el => { bboxRefs.current[i] = el }}
                onClick={() => onBboxClick(i)}
                onMouseEnter={() => onBboxHover(i)}
                onMouseLeave={() => onBboxHover(null)}
                className="absolute cursor-pointer transition-all duration-150 z-10"
                style={{
                  top: `${top * 100}%`,
                  left: `${left * 100}%`,
                  width: `${width * 100}%`,
                  height: `${height * 100}%`,
                  outline: `2px solid ${active ? "var(--primary)" : confBorder(r.confidence)}`,
                  backgroundColor: active ? "rgba(238,70,71,0.2)" : confBg(r.confidence),
                  opacity: active ? 1 : 0.6,
                }}
              >
                {isHovered && (
                  <div
                    className="absolute -top-6 left-0 whitespace-nowrap rounded-t-md px-2 py-0.5 text-[10px] font-semibold text-white z-20 pointer-events-none shadow"
                    style={{ backgroundColor: "var(--primary)" }}
                  >
                    {r.engine ? `[${r.engine.toUpperCase()}] ` : ""}
                    {r.text.length > 35 ? r.text.slice(0, 35) + "…" : r.text}
                  </div>
                )}
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// Resizable Split Pane
// ─────────────────────────────────────────────────────────────────────────────

function ResizerPane({ leftChild, rightChild }: { leftChild: React.ReactNode, rightChild: React.ReactNode }) {
  const [leftWidth, setLeftWidth] = useState(50)
  const [isResizing, setIsResizing] = useState(false)
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!isResizing) return
    const handleMouseMove = (e: MouseEvent) => {
      if (!containerRef.current) return
      const rect = containerRef.current.getBoundingClientRect()
      let newWidth = ((e.clientX - rect.left) / rect.width) * 100
      if (newWidth < 25) newWidth = 25
      if (newWidth > 75) newWidth = 75
      setLeftWidth(newWidth)
    }
    const handleMouseUp = () => setIsResizing(false)
    window.addEventListener("mousemove", handleMouseMove)
    window.addEventListener("mouseup", handleMouseUp)
    return () => {
      window.removeEventListener("mousemove", handleMouseMove)
      window.removeEventListener("mouseup", handleMouseUp)
    }
  }, [isResizing])

  return (
    <div ref={containerRef} className="flex w-full items-stretch min-h-0 select-none relative flex-1 h-full">
      <div style={{ width: `calc(${leftWidth}% - 8px)` }} className="flex-none flex flex-col min-w-0 h-full">
        {leftChild}
      </div>
      <div 
        className="w-4 flex items-center justify-center cursor-col-resize group px-1 flex-none z-10"
        onMouseDown={(e) => { e.preventDefault(); setIsResizing(true) }}
      >
        <div className="w-1 h-12 rounded-full bg-white/10 group-hover:bg-primary transition-colors" />
      </div>
      <div style={{ width: `calc(${100 - leftWidth}% - 8px)` }} className="flex-none relative flex flex-col min-w-0 h-full">
        <div className="absolute inset-0 flex flex-col h-full overflow-hidden min-w-0">
          {rightChild}
        </div>
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// Main Extraction Screen with Unified Top Header & Right Sidebar
// ─────────────────────────────────────────────────────────────────────────────

function ExtractionScreen({
  history,
  status,
  onHistoryChange,
  onStatusChange,
  onReloadStatus,
}: {
  history: OcrJob[]
  status: OcrStatus | null
  onHistoryChange: () => void
  onStatusChange?: (newStatus: OcrStatus) => void
  onReloadStatus: () => void
}) {
  const router = useRouter()
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [extracting, setExtracting] = useState(false)
  const [clearing, setClearing] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [job, setJob] = useState<(OcrJob & { img_width?: number; img_height?: number }) | null>(null)
  const [selectedIdx, setSelectedIdx] = useState<number | null>(null)
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null)
  const [minConfidence, setMinConfidence] = useState(0)
  const [isDragging, setIsDragging] = useState(false)
  const [isModelDropdownOpen, setIsModelDropdownOpen] = useState(false)
  const [isSwitchingModel, setIsSwitchingModel] = useState(false)

  // Sidebar state (matching Audio Studio and Image Studio)
  const [sidebarTab, setSidebarTab] = useState<"settings" | "history">("settings")
  const [isSidebarOpen, setIsSidebarOpen] = useState(true)
  const [daysFilter, setDaysFilter] = useState<number>(30)
  const [isDaysDropdownOpen, setIsDaysDropdownOpen] = useState(false)
  const [showBoundingBoxes, setShowBoundingBoxes] = useState<boolean>(true)
  const [autoDetectTable, setAutoDetectTable] = useState<boolean>(true)

  const fileInputRef = useRef<HTMLInputElement>(null)

  const filteredHistory = useMemo(() => {
    if (daysFilter === 0) return history
    const cutoff = Date.now() - daysFilter * 24 * 60 * 60 * 1000
    return history.filter((h) => {
      const time = new Date(h.created_at).getTime()
      return isNaN(time) || time >= cutoff
    })
  }, [history, daysFilter])

  const handleClearAllHistory = async () => {
    if (!window.confirm("Are you sure you want to clear all OCR extraction history?")) return
    try {
      setClearing(true)
      await OCR.clearHistory()
      onHistoryChange()
    } catch (e: any) {
      console.error("Failed to clear OCR history:", e)
    } finally {
      setClearing(false)
    }
  }

  const handleFile = (f: File) => {
    setFile(f)
    setJob(null)
    setError(null)
    setSelectedIdx(null)
    setHoveredIdx(null)
    setPreview(URL.createObjectURL(f))
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
    const f = e.dataTransfer.files[0]
    if (f) handleFile(f)
  }

  const handleExtract = async () => {
    if (!file) return
    setExtracting(true)
    setError(null)
    setSelectedIdx(null)
    try {
      const activeModelId = status?.active_model_id || status?.model_id
      const result = await OCR.extract(file, activeModelId)
      setJob(result as any)
      onHistoryChange()
    } catch (e: any) {
      setError(e?.message || "Extraction failed")
    } finally {
      setExtracting(false)
    }
  }

  const handleResetWorkspace = () => {
    setFile(null)
    setJob(null)
    setPreview(null)
    setSelectedIdx(null)
    setHoveredIdx(null)
    setError(null)
  }

  const handleBboxClick = useCallback((i: number) => {
    setSelectedIdx(prev => prev === i ? null : i)
  }, [])

  const results: OcrResultWithNorm[] = (job?.results as OcrResultWithNorm[]) ?? []
  const activeModelDisplay = status?.active_model_name || status?.model_name || "RapidOCR (Production DBNet)"
  const activeModelObj = status?.models?.find(m => m.id === (status?.active_model_id || status?.model_id))

  return (
    <div className="flex flex-col h-full bg-[#0a0a0a] relative overflow-hidden text-foreground select-none font-sans">
      {/* ── Studio Header (Full width across top, identical to Audio/Image Studio) ── */}
      <header className="h-14 border-b border-white/10 flex items-center justify-between px-5 sm:px-8 shrink-0 bg-background/95 backdrop-blur-sm sticky top-0 z-20">
        <div className="flex items-center gap-3 min-w-0">
          <div className="w-8 h-8 rounded-xl bg-primary/10 text-primary border border-primary/20 flex items-center justify-center shrink-0">
            <ScanText size={18} />
          </div>
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <span className="text-sm font-bold text-foreground">OCR Studio</span>
              <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
            </div>
            <div className="text-[10px] text-muted-foreground font-medium flex items-center gap-1.5">
              <span>Vision Document &amp; Text Extraction</span>
            </div>
          </div>
        </div>

        {/* Right Actions & Sidebar Toggle */}
        <div className="flex items-center gap-2 sm:gap-3">
          <button
            onClick={() => fileInputRef.current?.click()}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-primary hover:bg-primary/90 text-primary-foreground text-xs font-semibold transition-all shadow-md shadow-primary/20"
            title="Upload image or document"
          >
            <Plus size={14} />
            <span className="hidden sm:inline">Add Document</span>
          </button>
          <input
            ref={fileInputRef}
            type="file"
            accept="image/png,image/jpeg,image/jpg,image/webp,image/bmp,image/tiff"
            className="hidden"
            onChange={(e) => {
              const f = e.target.files?.[0]
              if (f) handleFile(f)
            }}
          />

          <button
            onClick={onReloadStatus}
            className="p-2 rounded-lg hover:bg-white/10 text-muted-foreground hover:text-foreground transition-all"
            title="Refresh models and extraction history"
          >
            <RefreshCw size={15} />
          </button>

          <button
            onClick={() => setIsSidebarOpen((v) => !v)}
            className={cn(
              "p-2 rounded-lg text-muted-foreground hover:text-foreground hover:bg-white/10 transition-all border border-white/5",
              isSidebarOpen && "bg-white/10 text-foreground"
            )}
            title={isSidebarOpen ? "Collapse right panel" : "Expand right panel"}
          >
            {isSidebarOpen ? <PanelRightClose size={15} /> : <PanelRightOpen size={15} />}
          </button>
        </div>
      </header>

      {/* ── Main Content Area (Under the header, side-by-side with right sidebar) ── */}
      <div className="flex-1 flex flex-col lg:flex-row overflow-hidden w-full relative min-h-0">
        {/* Left Area: Document Canvas & Results Viewer */}
        <div className="flex-1 h-full min-w-0 flex flex-col p-4 overflow-hidden relative">
          {!preview && !job ? (
            /* ── Empty State / Dropzone ── */
            <div
              onDragOver={e => { e.preventDefault(); setIsDragging(true) }}
              onDragLeave={() => setIsDragging(false)}
              onDrop={handleDrop}
              className={cn(
                "flex-1 flex flex-col items-center justify-center p-8 rounded-3xl border-2 border-dashed transition-all text-center space-y-6",
                isDragging ? "border-primary/60 bg-primary/10" : "border-white/10 bg-[#121212]/80 hover:border-white/20"
              )}
            >
              <div className="w-16 h-16 rounded-3xl bg-primary/10 text-primary border border-primary/20 flex items-center justify-center shadow-lg shadow-primary/10">
                <ScanText size={32} />
              </div>
              <div className="space-y-2 max-w-md">
                <h3 className="text-xl font-bold text-foreground">Upload document to extract</h3>
                <p className="text-xs text-muted-foreground leading-relaxed">
                  Drag and drop any invoice, form, receipt, screenshot, or document. Grounded bounding boxes, tables, and full markdown are automatically parsed.
                </p>
              </div>
              <button
                onClick={() => fileInputRef.current?.click()}
                className="px-6 py-3 rounded-2xl bg-primary hover:bg-primary/90 text-white text-xs font-bold transition-all shadow-lg shadow-primary/25 active:scale-95 flex items-center gap-2"
              >
                <Upload size={14} />
                <span>Browse Files</span>
              </button>
              <div className="text-[11px] text-muted-foreground/60 font-mono">
                Supports PNG, JPG, WEBP, BMP, TIFF up to 25MB
              </div>
            </div>
          ) : (
            /* ── Active Document Workspace ── */
            <div className="flex-1 flex flex-col min-h-0 space-y-3">
              {/* Ready to Extract Banner (if file selected but not yet run) */}
              {file && !job && (
                <div className="flex items-center justify-between p-3.5 rounded-2xl border border-white/10 bg-[#141414] shadow-lg shrink-0">
                  <div className="flex items-center gap-3 min-w-0">
                    <div className="h-10 w-10 overflow-hidden rounded-xl border border-white/10 bg-black/40 shrink-0">
                      <img src={preview || ""} alt="Document" className="h-full w-full object-cover" />
                    </div>
                    <div className="min-w-0">
                      <div className="text-xs font-bold text-foreground truncate">{file.name}</div>
                      <div className="text-[10px] text-muted-foreground font-mono mt-0.5">
                        {(file.size / 1024).toFixed(1)} KB &bull; Ready to extract
                      </div>
                    </div>
                  </div>

                  <div className="flex items-center gap-2">
                    <button
                      onClick={handleResetWorkspace}
                      className="px-3 py-1.5 rounded-xl border border-white/10 text-xs font-semibold text-muted-foreground hover:text-foreground hover:bg-white/5 transition-all"
                    >
                      Cancel
                    </button>
                    <button
                      onClick={handleExtract}
                      disabled={extracting}
                      className="px-5 py-2 rounded-xl bg-primary hover:bg-primary/90 text-white text-xs font-bold transition-all shadow-md shadow-primary/20 flex items-center gap-2 active:scale-95 disabled:opacity-50"
                    >
                      {extracting ? <Loader2 size={13} className="animate-spin text-white" /> : <Zap size={13} />}
                      <span>{extracting ? "Extracting with Model..." : "Extract Text"}</span>
                    </button>
                  </div>
                </div>
              )}

              {/* Extraction Error Alert */}
              {error && (
                <div className="p-3 rounded-2xl bg-destructive/10 border border-destructive/20 text-xs text-destructive font-medium shrink-0">
                  {error}
                </div>
              )}

              {/* Synchronized Split Pane: Document Image Canvas & Results */}
              <div className="flex-1 min-h-0">
                <ResizerPane
                  leftChild={
                    <ImagePane
                      src={preview || ""}
                      results={results}
                      selectedIdx={selectedIdx}
                      hoveredIdx={hoveredIdx}
                      minConfidence={minConfidence}
                      showBoundingBoxes={showBoundingBoxes}
                      onBboxClick={handleBboxClick}
                      onBboxHover={setHoveredIdx}
                    />
                  }
                  rightChild={
                    job ? (
                      <ResultsPanel
                        results={results}
                        selectedIdx={selectedIdx}
                        onHover={setHoveredIdx}
                        onSelect={setSelectedIdx}
                        minConfidence={minConfidence}
                        job={job}
                        autoDetectTable={autoDetectTable}
                      />
                    ) : extracting ? (
                      <div className="flex-1 flex flex-col items-center justify-center gap-3 text-xs text-muted-foreground p-8 text-center h-full bg-[#141414] rounded-2xl border border-white/10">
                        <Loader2 size={24} className="animate-spin text-primary" />
                        <p className="font-semibold text-foreground">Extracting text &amp; layout structures...</p>
                        <p className="text-[11px] text-muted-foreground/60 max-w-xs">
                          Detecting text lines, bounding boxes, and tabular structures with local vision pipeline.
                        </p>
                      </div>
                    ) : (
                      <div className="flex-1 flex flex-col items-center justify-center text-xs text-muted-foreground p-8 text-center h-full bg-[#141414] rounded-2xl border border-white/10">
                        <p>Click &quot;Extract Text&quot; above to process this document.</p>
                      </div>
                    )
                  }
                />
              </div>
            </div>
          )}
        </div>

        {/* ── Right Sidebar: Docked Top-to-Bottom UNDER THE HEADER ── */}
        {isSidebarOpen && (
          <aside className="w-full lg:w-[340px] xl:w-[380px] shrink-0 h-full border-l border-white/10 bg-[#121212] flex flex-col overflow-hidden animate-in slide-in-from-right-4 duration-200">
            {/* Top Tabs Switcher: Settings | History */}
            <div className="p-3.5 border-b border-white/10 bg-black/20 shrink-0">
              <div className="p-1 rounded-full bg-black/40 border border-white/10 flex items-center">
                <button
                  onClick={() => setSidebarTab("settings")}
                  className={cn(
                    "flex-1 py-1.5 rounded-full text-xs font-semibold text-center transition-all",
                    sidebarTab === "settings"
                      ? "bg-primary text-white shadow-md shadow-primary/20 font-bold"
                      : "text-muted-foreground hover:text-foreground"
                  )}
                >
                  Settings
                </button>
                <button
                  onClick={() => setSidebarTab("history")}
                  className={cn(
                    "flex-1 py-1.5 rounded-full text-xs font-semibold text-center transition-all",
                    sidebarTab === "history"
                      ? "bg-primary text-white shadow-md shadow-primary/20 font-bold"
                      : "text-muted-foreground hover:text-foreground"
                  )}
                >
                  History {history.length > 0 && `(${history.length})`}
                </button>
              </div>
            </div>

            {/* Scrollable Sidebar Body */}
            <div className="flex-1 overflow-y-auto p-4 space-y-4">
              {sidebarTab === "settings" ? (
                /* ── Settings View ── */
                <div className="space-y-4 animate-in fade-in duration-150">
                  {/* Model Selection Dropdown */}
                  <div className="space-y-1.5">
                    <label className="text-xs font-medium text-muted-foreground block">
                      OCR Model
                    </label>
                    <div className="relative">
                      <button
                        type="button"
                        onClick={() => setIsModelDropdownOpen(!isModelDropdownOpen)}
                        disabled={isSwitchingModel}
                        className="w-full flex items-center justify-between px-3.5 py-2.5 rounded-2xl bg-black/30 border border-white/10 hover:border-white/20 text-xs font-semibold text-foreground transition-all shadow-inner disabled:opacity-50"
                      >
                        <div className="flex items-center gap-2 truncate">
                          <span className="truncate">{activeModelDisplay}</span>
                          <span className="px-2 py-0.5 rounded-full text-[10px] font-mono bg-white/10 text-muted-foreground font-semibold">
                            Fast RTF
                          </span>
                        </div>
                        <ChevronDown
                          size={14}
                          className={cn("text-muted-foreground transition-transform shrink-0 ml-2", isModelDropdownOpen && "rotate-180")}
                        />
                      </button>

                      {isModelDropdownOpen && (
                        <>
                          <div className="fixed inset-0 z-30" onClick={() => setIsModelDropdownOpen(false)} />
                          <div className="absolute top-full left-0 right-0 mt-1.5 z-40 rounded-2xl bg-[#1c1c1c] border border-white/15 p-2 shadow-2xl space-y-1 max-h-56 overflow-y-auto">
                            {(status?.models || []).map((m) => {
                              const isActive = m.id === (status?.active_model_id || status?.model_id)
                              const isReady = m.ready
                              const isInstalled = m.installed

                              return (
                                <button
                                  key={m.id}
                                  type="button"
                                  onClick={async () => {
                                    setIsModelDropdownOpen(false)
                                    if (isReady && !isActive) {
                                      setIsSwitchingModel(true)
                                      try {
                                        const newStatus = await OCR.selectModel(m.id)
                                        onStatusChange?.(newStatus)
                                        if (typeof window !== "undefined") {
                                          localStorage.setItem("oprel_ocr_model_id", m.id)
                                        }
                                      } catch (err) {
                                        console.error("Failed to select OCR model:", err)
                                      } finally {
                                        setIsSwitchingModel(false)
                                      }
                                    } else if (!isInstalled) {
                                      router.push("/models?category=OCR")
                                    }
                                  }}
                                  className={cn(
                                    "w-full flex items-center justify-between px-3 py-2 rounded-xl text-xs font-medium transition-all text-left",
                                    isActive
                                      ? "bg-white/10 text-white font-semibold"
                                      : "text-muted-foreground hover:bg-white/5 hover:text-foreground"
                                  )}
                                >
                                  <div className="truncate pr-2">
                                    <div className="truncate text-foreground font-semibold">{m.name}</div>
                                    <div className="text-[10px] text-muted-foreground font-mono">{m.quant || "Q8_0"} &bull; Local</div>
                                  </div>
                                  {isActive ? (
                                    <span className="px-1.5 py-0.5 rounded bg-primary/20 text-primary text-[10px] font-mono font-bold">
                                      ACTIVE
                                    </span>
                                  ) : isReady ? (
                                    <span className="px-1.5 py-0.5 rounded bg-emerald-500/10 text-emerald-400 text-[10px] font-mono">
                                      READY
                                    </span>
                                  ) : (
                                    <span className="px-1.5 py-0.5 rounded bg-white/5 text-muted-foreground text-[10px] font-mono">
                                      INSTALL
                                    </span>
                                  )}
                                </button>
                              )
                            })}
                          </div>
                        </>
                      )}
                    </div>
                  </div>

                  {/* Model Highlight Card (matching Kokoro-82M and SD.CPP) */}
                  <div className="p-3.5 rounded-2xl bg-gradient-to-br from-white/[0.06] to-white/[0.02] border border-white/10 space-y-1">
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-bold text-foreground truncate">
                        {activeModelDisplay}
                      </span>
                      <span className="text-[10px] font-mono text-primary font-bold shrink-0">Active Engine</span>
                    </div>
                    <p className="text-[11px] text-muted-foreground leading-relaxed">
                      {activeModelObj?.description || "Production DBNet + SVTR pixel-grounded text detection & recognition engine with local acceleration."}
                    </p>
                  </div>

                  {/* Extraction Controls Section Header */}
                  <div className="space-y-3 pt-2">
                    <label className="text-xs font-semibold text-muted-foreground block uppercase tracking-wider text-[10px]">
                      Extraction Controls
                    </label>

                    {/* Confidence Threshold Slider */}
                    <div className="flex items-center justify-between gap-3 p-2.5 rounded-2xl bg-black/20 border border-white/5">
                      <span className="text-xs font-medium text-muted-foreground">Min Confidence</span>
                      <div className="flex items-center gap-3 flex-1 max-w-[160px]">
                        <input
                          type="range"
                          min="0"
                          max="95"
                          step="5"
                          value={minConfidence}
                          onChange={(e) => setMinConfidence(parseInt(e.target.value, 10))}
                          className="w-full h-1.5 bg-white/10 rounded-lg appearance-none cursor-pointer accent-primary"
                        />
                        <span className="text-xs font-mono font-semibold text-foreground min-w-[28px] text-right">
                          {minConfidence}%
                        </span>
                      </div>
                    </div>

                    {/* Bounding Box Overlays Toggle */}
                    <div className="flex items-center justify-between gap-2 p-2.5 rounded-2xl bg-black/20 border border-white/5">
                      <span className="text-xs font-medium text-muted-foreground truncate">Bounding Box Overlays</span>
                      <div className="inline-flex items-center p-0.5 rounded-xl bg-black/50 border border-white/10 shrink-0">
                        <button
                          type="button"
                          onClick={() => setShowBoundingBoxes(false)}
                          className={cn(
                            "px-2.5 py-1 rounded-lg text-[11px] font-semibold transition-all",
                            !showBoundingBoxes ? "bg-white/20 text-white font-bold shadow-sm" : "text-muted-foreground hover:text-foreground"
                          )}
                        >
                          Off
                        </button>
                        <button
                          type="button"
                          onClick={() => setShowBoundingBoxes(true)}
                          className={cn(
                            "px-2.5 py-1 rounded-lg text-[11px] font-semibold transition-all",
                            showBoundingBoxes ? "bg-primary text-white font-bold shadow-sm" : "text-muted-foreground hover:text-foreground"
                          )}
                        >
                          On
                        </button>
                      </div>
                    </div>

                    {/* Table Structuring Toggle */}
                    <div className="flex items-center justify-between gap-2 p-2.5 rounded-2xl bg-black/20 border border-white/5">
                      <span className="text-xs font-medium text-muted-foreground truncate">Table Auto-Detection</span>
                      <div className="inline-flex items-center p-0.5 rounded-xl bg-black/50 border border-white/10 shrink-0">
                        <button
                          type="button"
                          onClick={() => setAutoDetectTable(false)}
                          className={cn(
                            "px-2.5 py-1 rounded-lg text-[11px] font-semibold transition-all",
                            !autoDetectTable ? "bg-white/20 text-white font-bold shadow-sm" : "text-muted-foreground hover:text-foreground"
                          )}
                        >
                          Off
                        </button>
                        <button
                          type="button"
                          onClick={() => setAutoDetectTable(true)}
                          className={cn(
                            "px-2.5 py-1 rounded-lg text-[11px] font-semibold transition-all",
                            autoDetectTable ? "bg-primary text-white font-bold shadow-sm" : "text-muted-foreground hover:text-foreground"
                          )}
                        >
                          On
                        </button>
                      </div>
                    </div>
                  </div>
                </div>
              ) : (
                /* ── History View ── */
                <div className="space-y-3 animate-in fade-in duration-150">
                  {/* Filter Row: Days dropdown + Refresh button */}
                  <div className="flex items-center justify-between gap-2">
                    <div className="relative flex-1">
                      <button
                        type="button"
                        onClick={() => setIsDaysDropdownOpen(!isDaysDropdownOpen)}
                        className="w-full px-3 py-1.5 rounded-full bg-black/30 border border-white/10 text-xs font-medium text-foreground flex items-center justify-between hover:border-white/20 transition-all"
                      >
                        <span>
                          {daysFilter === 7
                            ? "Last 7 days"
                            : daysFilter === 30
                            ? "Last 30 days"
                            : daysFilter === 90
                            ? "Last 90 days"
                            : "All time"}
                        </span>
                        <ChevronDown size={12} className={cn("text-muted-foreground transition-transform", isDaysDropdownOpen && "rotate-180")} />
                      </button>

                      {isDaysDropdownOpen && (
                        <>
                          <div className="fixed inset-0 z-30" onClick={() => setIsDaysDropdownOpen(false)} />
                          <div className="absolute top-full left-0 right-0 mt-1 z-40 rounded-xl bg-[#1c1c1c] border border-white/15 p-1 shadow-2xl space-y-0.5">
                            {[
                              { label: "Last 7 days", value: 7 },
                              { label: "Last 30 days", value: 30 },
                              { label: "Last 90 days", value: 90 },
                              { label: "All time", value: 0 },
                            ].map((opt) => (
                              <button
                                key={opt.value}
                                type="button"
                                onClick={() => {
                                  setDaysFilter(opt.value)
                                  setIsDaysDropdownOpen(false)
                                }}
                                className={cn(
                                  "w-full flex items-center justify-between px-3 py-1.5 rounded-lg text-xs font-medium transition-all text-left",
                                  daysFilter === opt.value
                                    ? "bg-white/10 text-white font-semibold"
                                    : "text-muted-foreground hover:bg-white/5 hover:text-foreground"
                                )}
                              >
                                <span>{opt.label}</span>
                                {daysFilter === opt.value && <Check size={12} className="text-primary" />}
                              </button>
                            ))}
                          </div>
                        </>
                      )}
                    </div>

                    <button
                      type="button"
                      onClick={onHistoryChange}
                      className="p-2 rounded-full bg-white/5 hover:bg-white/10 text-muted-foreground hover:text-foreground transition-all"
                      title="Refresh history"
                    >
                      <RefreshCw size={13} />
                    </button>
                  </div>

                  {/* Extract New Document Button */}
                  <button
                    type="button"
                    onClick={() => {
                      handleResetWorkspace()
                      fileInputRef.current?.click()
                    }}
                    className="w-full flex items-center justify-center gap-2 py-2.5 px-4 rounded-2xl bg-primary text-white text-xs font-bold hover:bg-primary/90 transition-all shadow-md shadow-primary/20"
                  >
                    <Plus size={15} />
                    <span>Extract New Document</span>
                  </button>

                  {/* Header Actions */}
                  <div className="flex items-center justify-between text-[11px] text-muted-foreground font-semibold pt-1">
                    <span>
                      {filteredHistory.length > 0
                        ? `${filteredHistory.length} extraction${filteredHistory.length > 1 ? "s" : ""}`
                        : "No extractions in range"}
                    </span>
                    {history.length > 0 && (
                      <button
                        type="button"
                        onClick={handleClearAllHistory}
                        disabled={clearing}
                        className="text-muted-foreground hover:text-rose-400 transition-colors"
                      >
                        {clearing ? "Clearing..." : "Clear All"}
                      </button>
                    )}
                  </div>

                  {/* History Cards List */}
                  {filteredHistory.length === 0 ? (
                    <div className="text-center py-12 text-xs text-muted-foreground space-y-2">
                      <HistoryIcon size={26} className="mx-auto opacity-40" />
                      <p>No extractions found in this time range.</p>
                    </div>
                  ) : (
                    <div className="space-y-2.5">
                      {filteredHistory.map((h) => {
                        const isActive = job?.id === h.id
                        return (
                          <div
                            key={h.id}
                            onClick={() => {
                              setJob({ ...h } as any)
                              setPreview(h.image_data)
                              setFile(null)
                              setSelectedIdx(null)
                              setHoveredIdx(null)
                            }}
                            className={cn(
                              "p-3.5 rounded-2xl bg-black/30 border transition-all space-y-2 cursor-pointer group",
                              isActive
                                ? "border-primary/50 shadow-lg shadow-primary/5 bg-primary/[0.03]"
                                : "border-white/5 hover:border-white/15"
                            )}
                          >
                            {/* Avatar + Title + Timestamp */}
                            <div className="flex items-center justify-between text-xs">
                              <div className="flex items-center gap-2 truncate min-w-0">
                                <div className="h-5 w-5 rounded-md overflow-hidden bg-black/50 border border-white/10 shrink-0">
                                  <img src={h.image_data} alt={h.filename} className="h-full w-full object-cover" />
                                </div>
                                <span className="font-semibold text-foreground truncate">{h.filename}</span>
                              </div>
                              <span className="text-[10px] font-mono text-muted-foreground shrink-0 ml-2">
                                {formatDate(h.created_at)}
                              </span>
                            </div>

                            {/* Extracted Text Preview */}
                            <p className="text-xs text-muted-foreground/90 line-clamp-2 leading-relaxed font-mono">
                              {h.full_text.slice(0, 80)}
                            </p>

                            {/* Controls Bar: Word count badge + Actions */}
                            <div className="flex items-center justify-between pt-2 border-t border-white/5">
                              <div className="flex items-center gap-1.5">
                                <span className="text-[10px] px-2 py-0.5 rounded-full bg-white/5 border border-white/10 text-muted-foreground font-mono">
                                  {h.word_count} words
                                </span>
                              </div>

                              <div className="flex items-center gap-1.5 text-muted-foreground">
                                <button
                                  type="button"
                                  onClick={(e) => {
                                    e.stopPropagation()
                                    navigator.clipboard.writeText(h.full_text)
                                  }}
                                  className="p-1.5 rounded-lg hover:bg-white/10 hover:text-foreground transition-all"
                                  title="Copy text"
                                >
                                  <Copy size={13} />
                                </button>
                                <button
                                  type="button"
                                  onClick={async (e) => {
                                    e.stopPropagation()
                                    await OCR.deleteJob(h.id)
                                    onHistoryChange()
                                    if (job?.id === h.id) {
                                      handleResetWorkspace()
                                    }
                                  }}
                                  className="p-1.5 rounded-lg hover:bg-rose-500/20 hover:text-rose-400 transition-all"
                                  title="Delete from history"
                                >
                                  <Trash2 size={13} />
                                </button>
                              </div>
                            </div>
                          </div>
                        )
                      })}
                    </div>
                  )}
                </div>
              )}
            </div>
          </aside>
        )}
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// Root
// ─────────────────────────────────────────────────────────────────────────────

const INITIAL_OCR_STATUS: OcrStatus = {
  ready: true,
  installed: true,
  active_model_id: "rapidocr",
  active_model_name: "RapidOCR (Production DBNet)",
  model_id: "rapidocr",
  model_name: "RapidOCR (Production DBNet)",
  model_dir: "",
  size_mb: 15,
  gpu: false,
  models: [
    {
      id: "rapidocr",
      name: "RapidOCR (Production DBNet)",
      repo_id: "rapidocr-onnxruntime",
      architecture: "onnx",
      params: "15M",
      quant: "INT8/FP32",
      description: "Production DBNet + SVTR pixel-grounded text detection & recognition (Default)",
      installed: true,
      has_mmproj: true,
      ready: true,
      size_mb: 15,
    },
    {
      id: "auto",
      name: "Auto (Hybrid DBNet + VLM)",
      repo_id: "auto-hybrid",
      architecture: "hybrid",
      params: "Hybrid",
      quant: "Optimal",
      description: "Pixel-grounded RapidOCR detection with optional VLM layout structuring",
      installed: true,
      has_mmproj: true,
      ready: true,
      size_mb: 15,
    },
  ],
}

export function OcrView() {
  const [status, setStatus] = useState<OcrStatus>(INITIAL_OCR_STATUS)
  const [history, setHistory] = useState<OcrJob[]>([])

  const loadOcrData = async () => {
    try {
      const [s, h] = await Promise.all([
        OCR.fetchStatus().catch(() => null),
        OCR.fetchHistory().catch(() => []),
      ])

      if (h) setHistory(h)

      if (s) {
        const savedId = typeof window !== "undefined" ? localStorage.getItem("oprel_ocr_model_id") : null
        if (savedId && s.active_model_id !== savedId && s.models?.some(m => m.id === savedId && m.ready)) {
          try {
            const updated = await OCR.selectModel(savedId)
            setStatus(updated)
            return
          } catch {}
        }
        setStatus(s)
      }
    } catch (e) {
      console.warn("Background OCR sync:", e)
    }
  }

  useEffect(() => {
    loadOcrData()
  }, [])

  const hasAnyModel = status?.ready || status?.installed || (status?.models && status.models.some(m => m.ready || m.installed))

  if (!hasAnyModel) {
    return <SetupScreen onReady={loadOcrData} />
  }

  return (
    <ExtractionScreen
      history={history}
      status={status}
      onHistoryChange={() => { OCR.fetchHistory().then(setHistory).catch(() => {}) }}
      onStatusChange={setStatus}
      onReloadStatus={loadOcrData}
    />
  )
}
