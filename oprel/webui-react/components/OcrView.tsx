"use client"

import { useCallback, useEffect, useRef, useState } from "react"
import {
  CheckCircle2, ClipboardCopy, Download, FileJson, FileText,
  Loader2, ScanText, Trash2, Upload, X, Zap,
} from "lucide-react"
import { OCR, type OcrJob, type OcrResult, type OcrStatus } from "@/services/api"
import { cn } from "@/services/utils"

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

type OcrResultWithNorm = OcrResult & {
  bbox_norm?: { left: number; top: number; width: number; height: number }
}

type SetupStep = { step: string; message: string }

// ─────────────────────────────────────────────────────────────────────────────
// Confidence colour helpers (no blue)
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

// ─────────────────────────────────────────────────────────────────────────────
// Detect if a set of results looks like a table
// (≥3 lines all sharing similar horizontal bands → render as table rows)
// ─────────────────────────────────────────────────────────────────────────────

function groupIntoTableRows(results: OcrResultWithNorm[]): OcrResultWithNorm[][] | null {
  if (results.length < 4) return null
  // Sort by top
  const sorted = [...results].sort((a, b) => {
    const at = a.bbox_norm?.top ?? 0
    const bt = b.bbox_norm?.top ?? 0
    return at - bt
  })
  // Group by rows: items within 0.025 vertical tolerance go into the same row
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

  // Only treat as table if most rows have ≥ 2 columns
  const multiColRows = rows.filter(r => r.length >= 2).length
  if (multiColRows < rows.length * 0.5) return null
  return rows
}

// ─────────────────────────────────────────────────────────────────────────────
// Setup Screen
// ─────────────────────────────────────────────────────────────────────────────

function SetupScreen({ onReady }: { onReady: () => void }) {
  const [downloading, setDownloading] = useState(false)
  const [steps, setSteps] = useState<SetupStep[]>([])
  const [error, setError] = useState<string | null>(null)
  const [done, setDone] = useState(false)
  const cleanupRef = useRef<(() => void) | null>(null)

  const handleDownload = async () => {
    setDownloading(true)
    setSteps([])
    setError(null)
    try {
      await OCR.startSetup()
      cleanupRef.current = OCR.streamSetupProgress(
        (step, message) => setSteps(prev => [...prev, { step, message }]),
        (err) => {
          setDownloading(false)
          if (err) { setError(err) } else { setDone(true); setTimeout(onReady, 1500) }
        },
      )
    } catch (e: any) {
      setError(e?.message || "Failed to start download")
      setDownloading(false)
    }
  }

  useEffect(() => () => { cleanupRef.current?.() }, [])

  return (
    <div className="h-full overflow-y-auto bg-[linear-gradient(180deg,#0e0e0e_0%,#111_50%,#0b0b0b_100%)] text-foreground">
      <div className="mx-auto max-w-2xl px-6 py-20 flex flex-col items-center text-center space-y-8">
        {/* Icon */}
        <div className="flex h-20 w-20 items-center justify-center rounded-3xl border border-primary/30 bg-primary/10 shadow-lg shadow-primary/10">
          <ScanText size={36} className="text-primary" />
        </div>

        {/* Heading */}
        <div className="space-y-3">
          <h1 className="text-4xl font-black tracking-tight">OCR — Text Extraction</h1>
          <p className="text-muted-foreground text-base leading-7 max-w-lg">
            Extract text from images, screenshots, invoices, and documents using <strong className="text-foreground">PaddleOCR</strong>.
            Models are downloaded once and stored locally.
          </p>
        </div>

        {/* Feature bullets */}
        <div className="grid grid-cols-2 gap-3 w-full text-left text-sm">
          {["Bounding box overlay", "Confidence scoring", "Table detection", "Export TXT / MD / JSON", "Persistent history", "GPU auto-detected"].map(f => (
            <div key={f} className="flex items-center gap-2 rounded-xl border border-border bg-secondary/30 px-4 py-3 text-muted-foreground">
              <CheckCircle2 size={14} className="text-primary shrink-0" />
              {f}
            </div>
          ))}
        </div>

        {/* Download button */}
        {!done && (
          <button
            onClick={handleDownload}
            disabled={downloading}
            className={cn(
              "inline-flex items-center gap-3 rounded-2xl px-8 py-4 text-base font-bold transition-all",
              downloading
                ? "bg-secondary/50 border border-border text-muted-foreground cursor-not-allowed"
                : "bg-primary hover:bg-primary/90 text-primary-foreground shadow-lg shadow-primary/25"
            )}
          >
            {downloading ? <Loader2 size={20} className="animate-spin" /> : <Download size={20} />}
            {downloading ? "Downloading…" : "Download OCR Models (~30MB MB)"}
          </button>
        )}

        {/* Progress */}
        {steps.length > 0 && (
          <div className="w-full rounded-2xl border border-border bg-secondary/20 p-5 text-left space-y-2">
            {steps.map((s, i) => (
              <div key={i} className={cn("flex items-start gap-3 text-sm", i === steps.length - 1 && !done ? "text-foreground" : "text-muted-foreground")}>
                {i === steps.length - 1 && !done
                  ? <Loader2 size={14} className="animate-spin mt-0.5 text-primary shrink-0" />
                  : <CheckCircle2 size={14} className="mt-0.5 text-emerald-400 shrink-0" />
                }
                {s.message}
              </div>
            ))}
            {done && <div className="flex items-center gap-2 text-sm text-emerald-400 font-semibold mt-2"><CheckCircle2 size={14} />Setup complete — opening OCR…</div>}
          </div>
        )}

        {error && (
          <div className="w-full rounded-2xl border border-red-500/20 bg-red-500/10 p-4 text-sm text-red-300 text-left">
            <strong>Error:</strong> {error}
          </div>
        )}
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// Results Panel — doc-ai-ui style rows with bbox sync
// ─────────────────────────────────────────────────────────────────────────────

function ResultsPanel({
  results,
  selectedIdx,
  onHover,
  onSelect,
  minConfidence,
  onMinConfidenceChange,
  job,
}: {
  results: OcrResultWithNorm[]
  selectedIdx: number | null
  onHover: (i: number | null) => void
  onSelect: (i: number) => void
  minConfidence: number
  onMinConfidenceChange: (v: number) => void
  job: OcrJob
}) {
  const rowRefs = useRef<(HTMLDivElement | null)[]>([])
  const [copied, setCopied] = useState(false)

  // Scroll the selected row into view when selectedIdx changes
  useEffect(() => {
    if (selectedIdx != null && rowRefs.current[selectedIdx]) {
      rowRefs.current[selectedIdx]!.scrollIntoView({ behavior: "smooth", block: "nearest" })
    }
  }, [selectedIdx])

  const filtered = results.filter((_, i) => results[i].confidence >= minConfidence / 100)

  // Detect table layout
  const tableRows = groupIntoTableRows(filtered)

  const handleCopy = () => {
    navigator.clipboard.writeText(job.full_text).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    })
  }

  const triggerDownload = (blob: Blob, name: string) => {
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url; a.download = name; a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="flex flex-col h-full min-h-0">
      {/* Header */}
      <div className="flex-none px-4 py-3 border-b border-border flex items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <span className="text-[11px] font-bold uppercase tracking-[0.2em] text-muted-foreground">Results</span>
          <span className="rounded-full border border-border bg-secondary/40 px-2.5 py-0.5 text-[11px] font-semibold text-muted-foreground">
            {filtered.length} lines · {job.word_count} words
          </span>
        </div>
        {/* Export */}
        <div className="flex items-center gap-1.5">
          <button onClick={handleCopy} className="inline-flex items-center gap-1.5 rounded-lg border border-border bg-secondary/30 px-2.5 py-1.5 text-[11px] font-semibold text-foreground hover:bg-secondary/60 transition-all">
            {copied ? <CheckCircle2 size={12} className="text-emerald-400" /> : <ClipboardCopy size={12} />}
            {copied ? "Copied" : "Copy"}
          </button>
          <button onClick={() => triggerDownload(new Blob([job.full_text], { type: "text/plain" }), `${job.filename}.txt`)} className="inline-flex items-center gap-1 rounded-lg border border-border bg-secondary/30 px-2 py-1.5 text-[11px] font-semibold text-foreground hover:bg-secondary/60 transition-all">
            <FileText size={11} /> TXT
          </button>
          <button onClick={() => triggerDownload(new Blob([`# ${job.filename}\n\n${job.full_text}`], { type: "text/markdown" }), `${job.filename}.md`)} className="inline-flex items-center gap-1 rounded-lg border border-border bg-secondary/30 px-2 py-1.5 text-[11px] font-semibold text-foreground hover:bg-secondary/60 transition-all">
            <FileText size={11} /> MD
          </button>
          <button onClick={() => triggerDownload(new Blob([JSON.stringify(results, null, 2)], { type: "application/json" }), `${job.filename}.json`)} className="inline-flex items-center gap-1 rounded-lg border border-border bg-secondary/30 px-2 py-1.5 text-[11px] font-semibold text-foreground hover:bg-secondary/60 transition-all">
            <FileJson size={11} /> JSON
          </button>
        </div>
      </div>

      {/* Confidence slider */}
      <div className="flex-none px-4 py-2.5 border-b border-border flex items-center gap-3">
        <span className="text-[11px] text-muted-foreground whitespace-nowrap">Min confidence</span>
        <input type="range" min={0} max={99} value={minConfidence} onChange={e => onMinConfidenceChange(Number(e.target.value))} className="flex-1" />
        <span className="text-[11px] font-semibold text-foreground w-8 text-right">{minConfidence}%</span>
      </div>

      {/* Content — table or line list */}
      <div className="flex-1 overflow-y-auto">
        {filtered.length === 0 && (
          <div className="flex items-center justify-center h-full text-sm text-muted-foreground py-8">
            No results above {minConfidence}% confidence
          </div>
        )}

        {tableRows ? (
          /* ── Table view ── */
          <div className="p-3 overflow-x-auto">
            <table className="w-full text-xs border-collapse">
              <tbody>
                {tableRows.map((row, rIdx) => (
                  <tr key={rIdx} className={cn("border-b border-border/50 transition-colors", rIdx === 0 && "bg-secondary/30 font-semibold")}>
                    {row.map((cell, cIdx) => {
                      const origIdx = results.indexOf(cell)
                      const isSelected = origIdx === selectedIdx
                      return (
                        <td
                          key={cIdx}
                          ref={el => { rowRefs.current[origIdx] = el }}
                          onClick={() => { onSelect(origIdx) }}
                          onMouseEnter={() => onHover(origIdx)}
                          onMouseLeave={() => onHover(null)}
                          className={cn(
                            "px-3 py-2 text-left border border-border/30 cursor-pointer transition-all align-top leading-5",
                            isSelected
                              ? "ring-1 ring-primary/60 bg-primary/10"
                              : "hover:bg-secondary/40"
                          )}
                          style={isSelected ? { borderColor: "rgba(238,70,71,0.4)" } : {}}
                        >
                          <span className={cn("text-foreground", rIdx === 0 && "text-primary")}>{cell.text}</span>
                          <span className={cn("ml-1.5 text-[9px] font-semibold px-1 py-0.5 rounded border", confBadge(cell.confidence))}>
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
        ) : (
          /* ── Line list view ── */
          <div className="p-2 space-y-0.5">
            {filtered.map((r, i) => {
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
                    transition: "background-color 0.15s, border-color 0.15s",
                  }}
                  className={cn(
                    "flex items-start justify-between gap-2 rounded-lg px-3 py-2 cursor-pointer text-sm mx-1",
                    isSelected ? "bg-primary/10 ring-1 ring-primary/30" : "hover:bg-secondary/40"
                  )}
                >
                  <span className="text-foreground leading-5 flex-1">{r.text}</span>
                  <span className={cn("shrink-0 text-[10px] font-semibold rounded-full px-2 py-0.5 border mt-0.5", confBadge(r.confidence))}>
                    {(r.confidence * 100).toFixed(0)}%
                  </span>
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
// Image Pane — with bbox overlay synchronized to results panel
// ─────────────────────────────────────────────────────────────────────────────

function ImagePane({
  src,
  results,
  selectedIdx,
  hoveredIdx,
  minConfidence,
  onBboxClick,
  onBboxHover,
}: {
  src: string
  results: OcrResultWithNorm[]
  selectedIdx: number | null
  hoveredIdx: number | null
  minConfidence: number
  onBboxClick: (i: number) => void
  onBboxHover: (i: number | null) => void
}) {
  return (
    <div className="rounded-3xl border border-border bg-[#141414]/90 p-4 flex flex-col gap-3 h-full">
      <div className="text-[11px] font-bold uppercase tracking-[0.2em] text-muted-foreground flex-none">Image</div>
      <div className="flex-1 flex items-center justify-center overflow-hidden rounded-2xl border border-border bg-black/30 p-2">
        <div className="relative inline-block max-w-full max-h-full">
          <img src={src} alt="Uploaded" className="max-w-full max-h-full object-contain block rounded-lg" />

          {/* Bbox overlays — normalized coordinates */}
          {results.map((r, i) => {
            if (!r.bbox_norm) return null
            if (r.confidence < minConfidence / 100) return null
            const { left, top, width, height } = r.bbox_norm
            const isSelected = i === selectedIdx
            const isHovered = i === hoveredIdx
            const active = isSelected || isHovered

            return (
              <div
                key={i}
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
                  backgroundColor: active ? "rgba(238,70,71,0.18)" : confBg(r.confidence),
                  opacity: active ? 1 : 0.55,
                }}
              >
                {/* Tooltip on hover */}
                {isHovered && (
                  <div
                    className="absolute -top-6 left-0 whitespace-nowrap rounded-t-md px-2 py-0.5 text-[10px] font-semibold text-primary-foreground z-20 pointer-events-none"
                    style={{ backgroundColor: "var(--primary)" }}
                  >
                    {r.text.length > 40 ? r.text.slice(0, 40) + "…" : r.text}
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
      if (newWidth < 20) newWidth = 20
      if (newWidth > 80) newWidth = 80
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
    <div ref={containerRef} className="flex w-full items-stretch min-h-0 select-none relative h-[720px]">
      <div style={{ width: `calc(${leftWidth}% - 8px)` }} className="flex-none flex flex-col min-w-0">
        {leftChild}
      </div>
      <div 
        className="w-4 flex items-center justify-center cursor-col-resize group px-1 flex-none z-10"
        onMouseDown={(e) => { e.preventDefault(); setIsResizing(true) }}
      >
        <div className="w-1 h-12 rounded-full bg-border/50 group-hover:bg-primary transition-colors" />
      </div>
      <div style={{ width: `calc(${100 - leftWidth}% - 8px)` }} className="flex-none relative flex flex-col min-w-0">
        <div className="absolute inset-0 flex flex-col h-full overflow-hidden min-w-0">
          {rightChild}
        </div>
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// Extraction Screen
// ─────────────────────────────────────────────────────────────────────────────

function ExtractionScreen({ history, onHistoryChange }: { history: OcrJob[]; onHistoryChange: () => void }) {
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [extracting, setExtracting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [job, setJob] = useState<(OcrJob & { img_width?: number; img_height?: number }) | null>(null)
  const [selectedIdx, setSelectedIdx] = useState<number | null>(null)
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null)
  const [minConfidence, setMinConfidence] = useState(0)
  const [isDragging, setIsDragging] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleFile = (f: File) => {
    setFile(f); setJob(null); setError(null); setSelectedIdx(null); setHoveredIdx(null)
    setPreview(URL.createObjectURL(f))
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault(); setIsDragging(false)
    const f = e.dataTransfer.files[0]
    if (f) handleFile(f)
  }

  const handleExtract = async () => {
    if (!file) return
    setExtracting(true); setError(null); setSelectedIdx(null)
    try {
      const result = await OCR.extract(file)
      setJob(result as any)
      onHistoryChange()
    } catch (e: any) {
      setError(e?.message || "Extraction failed")
    } finally {
      setExtracting(false)
    }
  }

  // Clicking a bbox selects that result row
  const handleBboxClick = useCallback((i: number) => {
    setSelectedIdx(prev => prev === i ? null : i)
  }, [])

  // Clicking a result row selects that bbox
  const handleResultSelect = useCallback((i: number) => {
    setSelectedIdx(prev => prev === i ? null : i)
  }, [])

  const results: OcrResultWithNorm[] = (job?.results as OcrResultWithNorm[]) ?? []

  return (
    <div className="h-full overflow-y-auto bg-[linear-gradient(180deg,#0e0e0e_0%,#111_50%,#0b0b0b_100%)] text-foreground">
      <div className="mx-auto max-w-7xl px-6 py-6 lg:px-8 lg:py-8 space-y-6">

        {/* Header */}
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-2xl bg-primary/15 text-primary">
            <ScanText size={18} />
          </div>
          <div>
            <div className="text-sm font-semibold text-foreground">OCR — Text Extraction</div>
            <div className="text-xs text-muted-foreground">Upload an image to extract text with PaddleOCR</div>
          </div>
        </div>

        {/* Upload zone */}
        <div
          onDragOver={e => { e.preventDefault(); setIsDragging(true) }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          className={cn(
            "relative cursor-pointer rounded-2xl border-2 border-dashed p-8 text-center transition-all",
            isDragging ? "border-primary/60 bg-primary/8" :
            file ? "border-primary/30 bg-primary/5 hover:bg-primary/8" :
            "border-border hover:border-border/80 hover:bg-secondary/20"
          )}
        >
          <input ref={fileInputRef} type="file" accept="image/png,image/jpeg,image/jpg,image/webp,image/bmp,image/tiff" className="hidden"
            onChange={e => { const f = e.target.files?.[0]; if (f) handleFile(f) }} />
          <div className="flex flex-col items-center gap-3">
            <Upload size={28} className={file ? "text-primary" : "text-muted-foreground"} />
            {file ? (
              <div>
                <div className="text-sm font-semibold text-foreground">{file.name}</div>
                <div className="text-xs text-muted-foreground mt-0.5">{(file.size / 1024).toFixed(1)} KB — click to change</div>
              </div>
            ) : (
              <div>
                <div className="text-sm font-semibold text-foreground">Drop image here or click to upload</div>
                <div className="text-xs text-muted-foreground mt-0.5">PNG, JPG, WEBP, BMP, TIFF · max 20 MB</div>
              </div>
            )}
          </div>
        </div>

        {file && (
          <button
            onClick={handleExtract}
            disabled={extracting}
            className={cn(
              "inline-flex items-center gap-2 rounded-xl px-6 py-3 text-sm font-bold transition-all",
              extracting ? "bg-secondary/50 border border-border text-muted-foreground cursor-not-allowed"
                : "bg-primary hover:bg-primary/90 text-primary-foreground shadow-lg shadow-primary/20"
            )}
          >
            {extracting ? <Loader2 size={16} className="animate-spin" /> : <Zap size={16} />}
            {extracting ? "Extracting…" : "Extract Text"}
          </button>
        )}

        {error && (
          <div className="rounded-2xl border border-red-500/20 bg-red-500/10 p-4 text-sm text-red-300">{error}</div>
        )}

        {/* Split view */}
        {preview && (
          <ResizerPane
            leftChild={
              <ImagePane
                src={preview}
                results={results}
                selectedIdx={selectedIdx}
                hoveredIdx={hoveredIdx}
                minConfidence={minConfidence}
                onBboxClick={handleBboxClick}
                onBboxHover={setHoveredIdx}
              />
            }
            rightChild={
              <div className="rounded-3xl border border-border bg-[#141414]/90 overflow-hidden flex flex-col h-full shadow-xl">
                {job ? (
                  <ResultsPanel
                    results={results}
                    selectedIdx={selectedIdx}
                    onHover={setHoveredIdx}
                    onSelect={handleResultSelect}
                    minConfidence={minConfidence}
                    onMinConfidenceChange={setMinConfidence}
                    job={job}
                  />
                ) : extracting ? (
                  <div className="flex-1 flex flex-col items-center justify-center gap-4 text-sm text-muted-foreground p-8 text-center h-full">
                    <Loader2 size={24} className="animate-spin text-primary" />
                    Running PaddleOCR...
                  </div>
                ) : (
                  <div className="flex-1 flex items-center justify-center text-sm text-muted-foreground p-8 text-center h-full">
                    Click "Extract Text" to run OCR
                  </div>
                )}
              </div>
            }
          />
        )}

        {/* History */}
        {history.length > 0 && (
          <div className="rounded-3xl border border-border bg-[#141414]/90 p-6 space-y-4">
            <div className="flex items-center justify-between">
              <div className="text-sm font-semibold text-foreground">Recent Extractions</div>
              <div className="rounded-full border border-border bg-secondary/40 px-3 py-1 text-[11px] font-semibold text-muted-foreground">
                {history.length} item{history.length === 1 ? "" : "s"}
              </div>
            </div>
            <div className="space-y-3">
              {history.map(h => (
                <div key={h.id} className="group flex items-center gap-4 rounded-2xl border border-border/60 bg-black/20 p-4 hover:border-border transition-all">
                  <div className="shrink-0 h-14 w-14 overflow-hidden rounded-xl border border-border/50 bg-black/30">
                    <img src={h.image_data} alt={h.filename} className="h-full w-full object-cover" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="text-sm font-semibold text-foreground truncate">{h.filename}</div>
                    <div className="text-xs text-muted-foreground mt-0.5">
                      {h.word_count} words · {new Date(h.created_at).toLocaleString([], { dateStyle: "short", timeStyle: "short" })}
                    </div>
                    <div className="text-xs text-muted-foreground mt-1 line-clamp-1 opacity-60">{h.full_text.slice(0, 80)}</div>
                  </div>
                  <div className="flex items-center gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                    <button
                      onClick={() => {
                        setJob({ ...h } as any); setPreview(h.image_data)
                        setFile(null); setSelectedIdx(null)
                        window.scrollTo({ top: 0, behavior: "smooth" })
                      }}
                      className="rounded-lg border border-border bg-secondary/40 px-3 py-1.5 text-xs font-semibold text-foreground hover:bg-secondary/70"
                    >
                      View
                    </button>
                    <button
                      onClick={async () => { await OCR.deleteJob(h.id); onHistoryChange() }}
                      className="rounded-lg border border-red-500/20 bg-red-500/8 p-1.5 text-red-400 hover:bg-red-500/15 transition-all"
                    >
                      <Trash2 size={13} />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// Root
// ─────────────────────────────────────────────────────────────────────────────

export function OcrView() {
  const [status, setStatus] = useState<OcrStatus | null>(null)
  const [loading, setLoading] = useState(true)
  const [history, setHistory] = useState<OcrJob[]>([])

  const fetchStatus = async () => {
    try { setStatus(await OCR.fetchStatus()) }
    catch { setStatus({ ready: false, model_dir: "", size_mb: 0, gpu: false, installed: false }) }
    finally { setLoading(false) }
  }

  const fetchHistory = async () => {
    try { setHistory(await OCR.fetchHistory()) } catch {}
  }

  useEffect(() => { fetchStatus(); fetchHistory() }, [])

  if (loading) {
    return (
      <div className="h-full flex items-center justify-center bg-[#0e0e0e] text-muted-foreground">
        <Loader2 size={24} className="animate-spin" />
      </div>
    )
  }

  if (!status?.ready) {
    return <SetupScreen onReady={() => { fetchStatus(); fetchHistory() }} />
  }

  return <ExtractionScreen history={history} onHistoryChange={fetchHistory} />
}
