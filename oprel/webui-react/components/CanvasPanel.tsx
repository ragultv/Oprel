"use client"

import { useEffect, useRef, useState, useCallback } from "react"
import {
  X,
  Bold,
  Italic,
  List,
  ListOrdered,
  Undo2,
  Redo2,
  Printer,
  Download,
  ChevronDown,
  Pencil,
  Check,
} from "lucide-react"
import { cn } from "@/services/utils"

// ─── Types ────────────────────────────────────────────────────────────────────

export interface CanvasDocument {
  id: string
  title: string
  content: string          // innerHTML / rich HTML string
  createdAt: Date
  updatedAt: Date
}

interface CanvasPanelProps {
  document: CanvasDocument
  onClose: () => void
  onChange: (doc: Partial<CanvasDocument>) => void
  isStreaming?: boolean
}

// ─── Text Style Options ───────────────────────────────────────────────────────

const TEXT_STYLES = [
  { label: "Normal text", tag: "p",  className: "text-sm leading-relaxed" },
  { label: "Heading 1",   tag: "h1", className: "text-2xl font-bold leading-tight" },
  { label: "Heading 2",   tag: "h2", className: "text-xl font-semibold leading-snug" },
  { label: "Heading 3",   tag: "h3", className: "text-base font-semibold leading-snug" },
]

// ─── Helpers ─────────────────────────────────────────────────────────────────

function execCmd(command: string, value?: string) {
  document.execCommand(command, false, value ?? undefined)
}

function getBlockTag(): string {
  const sel = window.getSelection()
  if (!sel || !sel.rangeCount) return "p"
  let node: Node | null = sel.getRangeAt(0).startContainer
  while (node && node.nodeType !== Node.ELEMENT_NODE) node = node.parentNode
  const el = node as HTMLElement | null
  if (!el) return "p"
  const tag = el.tagName?.toLowerCase() || "p"
  return ["h1", "h2", "h3"].includes(tag) ? tag : "p"
}

function applyBlockFormat(tag: string) {
  document.execCommand("formatBlock", false, `<${tag}>`)
}

// ─── Export helpers ───────────────────────────────────────────────────────────

function exportPDF(title: string, content: string) {
  const iframe = window.document.createElement("iframe")
  iframe.style.cssText = "position:fixed;top:-9999px;left:-9999px;width:794px;height:1123px;border:none;"
  window.document.body.appendChild(iframe)

  const doc = iframe.contentDocument!
  doc.open()
  doc.write(`<!DOCTYPE html><html><head>
  <meta charset="utf-8">
  <title>${title}</title>
  <style>
    @page { margin: 20mm 25mm; }
    body { font-family: 'Georgia', serif; font-size: 12pt; line-height: 1.6; color: #111; background: #fff; padding: 0; margin: 0; }
    h1 { font-size: 22pt; font-weight: 700; margin: 16pt 0 8pt; }
    h2 { font-size: 17pt; font-weight: 600; margin: 14pt 0 6pt; }
    h3 { font-size: 14pt; font-weight: 600; margin: 10pt 0 4pt; }
    p  { margin: 6pt 0; }
    ul, ol { margin: 6pt 0 6pt 20pt; }
    li { margin: 2pt 0; }
    b, strong { font-weight: 700; }
    i, em { font-style: italic; }
  </style>
</head><body>${content}</body></html>`)
  doc.close()

  setTimeout(() => {
    iframe.contentWindow!.print()
    setTimeout(() => window.document.body.removeChild(iframe), 1000)
  }, 300)
}

function exportHTML(title: string, content: string) {
  const html = `<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>${title}</title>
<style>
  body { font-family: Calibri, Arial, sans-serif; font-size: 11pt; line-height: 1.5; max-width: 740px; margin: 40px auto; color: #222; }
  h1 { font-size: 20pt; font-weight: 700; margin: 20pt 0 8pt; }
  h2 { font-size: 16pt; font-weight: 600; margin: 16pt 0 6pt; }
  h3 { font-size: 13pt; font-weight: 600; margin: 12pt 0 4pt; }
  p  { margin: 6pt 0; }
  ul, ol { margin: 6pt 0 6pt 20pt; }
  li { margin: 2pt 0; }
</style>
</head>
<body>${content}</body>
</html>`

  const blob = new Blob([html], { type: "text/html;charset=utf-8" })
  const url = URL.createObjectURL(blob)
  const a = window.document.createElement("a")
  a.href = url
  a.download = `${title.replace(/[^a-z0-9]/gi, "_") || "canvas"}.html`
  a.click()
  URL.revokeObjectURL(url)
}

// ─── Toolbar Button ───────────────────────────────────────────────────────────

function ToolbarBtn({
  onClick,
  active,
  title,
  children,
}: {
  onClick: () => void
  active?: boolean
  title: string
  children: React.ReactNode
}) {
  return (
    <button
      onMouseDown={(e) => { e.preventDefault(); onClick() }}
      title={title}
      className={cn(
        "w-7 h-7 flex items-center justify-center rounded-md transition-all text-[13px]",
        active
          ? "bg-primary/15 text-primary"
          : "text-muted-foreground hover:bg-secondary hover:text-foreground"
      )}
    >
      {children}
    </button>
  )
}

// ─── Main Component ───────────────────────────────────────────────────────────

export function CanvasPanel({ document: doc, onClose, onChange, isStreaming }: CanvasPanelProps) {
  const editorRef = useRef<HTMLDivElement>(null)
  const [blockTag, setBlockTag] = useState("p")
  const [styleDropdown, setStyleDropdown] = useState(false)
  const [editingTitle, setEditingTitle] = useState(false)
  const [titleVal, setTitleVal] = useState(doc.title)
  const titleInputRef = useRef<HTMLInputElement>(null)

  // Sync incoming AI-streamed content into editor
  useEffect(() => {
    if (!editorRef.current) return
    // Only sync if the content actually differs (avoids cursor jump during user editing)
    if (editorRef.current.innerHTML !== doc.content) {
      editorRef.current.innerHTML = doc.content
      // Auto-scroll to bottom during streaming
      if (isStreaming) {
        editorRef.current.scrollTop = editorRef.current.scrollHeight
      }
    }
  }, [doc.content, isStreaming])

  // Sync title
  useEffect(() => {
    setTitleVal(doc.title)
  }, [doc.title])

  const handleEditorInput = useCallback(() => {
    if (!editorRef.current) return
    onChange({ content: editorRef.current.innerHTML, updatedAt: new Date() })
  }, [onChange])

  const handleSelectionChange = useCallback(() => {
    setBlockTag(getBlockTag())
  }, [])

  useEffect(() => {
    document.addEventListener("selectionchange", handleSelectionChange)
    return () => document.removeEventListener("selectionchange", handleSelectionChange)
  }, [handleSelectionChange])

  const applyStyle = (tag: string) => {
    applyBlockFormat(tag)
    setBlockTag(tag)
    setStyleDropdown(false)
    editorRef.current?.focus()
  }

  const currentStyleLabel = TEXT_STYLES.find(s => s.tag === blockTag)?.label ?? "Normal text"

  const saveTitle = () => {
    setEditingTitle(false)
    if (titleVal.trim()) onChange({ title: titleVal.trim(), updatedAt: new Date() })
    else setTitleVal(doc.title)
  }

  return (
    <div className="flex flex-col h-full bg-[#141414] border-l border-border/40 overflow-hidden">
      {/* ── Top bar ── */}
      <div className="h-12 flex items-center justify-between px-4 border-b border-border/40 shrink-0 gap-3">
        {/* Title */}
        <div className="flex items-center gap-2 min-w-0 flex-1">
          {editingTitle ? (
            <input
              ref={titleInputRef}
              value={titleVal}
              onChange={e => setTitleVal(e.target.value)}
              onBlur={saveTitle}
              onKeyDown={e => {
                if (e.key === "Enter") saveTitle()
                if (e.key === "Escape") { setEditingTitle(false); setTitleVal(doc.title) }
              }}
              autoFocus
              className="text-sm font-semibold bg-secondary border border-primary/40 rounded-md px-2 py-0.5 text-foreground outline-none flex-1 min-w-0"
            />
          ) : (
            <button
              className="flex items-center gap-1.5 group min-w-0"
              onClick={() => setEditingTitle(true)}
              title="Click to rename"
            >
              <span className="text-sm font-semibold text-foreground truncate max-w-[200px]">{titleVal}</span>
              <Pencil size={11} className="shrink-0 opacity-0 group-hover:opacity-50 transition-opacity" />
            </button>
          )}
          {isStreaming && (
            <span className="text-[10px] text-primary animate-pulse font-semibold px-1.5 py-0.5 bg-primary/10 rounded-full border border-primary/20 shrink-0">
              Writing…
            </span>
          )}
        </div>

        {/* Actions */}
        <div className="flex items-center gap-1 shrink-0">
          <button
            onMouseDown={(e) => { e.preventDefault(); exportPDF(doc.title, editorRef.current?.innerHTML ?? doc.content) }}
            title="Export PDF"
            className="p-1.5 rounded-md text-muted-foreground hover:text-foreground hover:bg-secondary transition-all"
          >
            <Printer size={14} />
          </button>
          <button
            onMouseDown={(e) => { e.preventDefault(); exportHTML(doc.title, editorRef.current?.innerHTML ?? doc.content) }}
            title="Export HTML (opens in Word / Docs)"
            className="p-1.5 rounded-md text-muted-foreground hover:text-foreground hover:bg-secondary transition-all"
          >
            <Download size={14} />
          </button>
          <div className="w-px h-4 bg-border mx-1" />
          <button
            onClick={onClose}
            title="Close canvas"
            className="p-1.5 rounded-md text-muted-foreground hover:text-foreground hover:bg-secondary transition-all"
          >
            <X size={14} />
          </button>
        </div>
      </div>

      {/* ── Formatting toolbar ── */}
      <div className="flex items-center gap-0.5 px-3 py-2 border-b border-border/30 shrink-0 bg-[#171717]/50 flex-wrap">
        {/* Block style selector */}
        <div className="relative">
          <button
            onMouseDown={(e) => { e.preventDefault(); setStyleDropdown(v => !v) }}
            className="flex items-center gap-1 px-2 py-1 rounded-md text-[11px] text-muted-foreground hover:text-foreground hover:bg-secondary transition-all font-medium border border-transparent hover:border-border/40 min-w-[110px] justify-between"
          >
            <span>{currentStyleLabel}</span>
            <ChevronDown size={11} className={cn("transition-transform", styleDropdown && "rotate-180")} />
          </button>
          {styleDropdown && (
            <>
              <div className="fixed inset-0 z-40" onMouseDown={() => setStyleDropdown(false)} />
              <div className="absolute top-full left-0 mt-1 w-48 bg-[#1a1a1a] border border-border rounded-xl shadow-2xl py-1.5 z-50 animate-in fade-in slide-in-from-top-1 duration-150">
                {TEXT_STYLES.map(s => (
                  <button
                    key={s.tag}
                    onMouseDown={(e) => { e.preventDefault(); applyStyle(s.tag) }}
                    className={cn(
                      "w-full text-left px-3 py-2 transition-all flex items-center justify-between",
                      s.className,
                      blockTag === s.tag ? "text-primary bg-primary/10" : "text-foreground/80 hover:bg-secondary hover:text-foreground"
                    )}
                  >
                    {s.label}
                    {blockTag === s.tag && <Check size={12} className="text-primary" />}
                  </button>
                ))}
              </div>
            </>
          )}
        </div>

        <div className="w-px h-4 bg-border/50 mx-1" />

        <ToolbarBtn onClick={() => execCmd("bold")} title="Bold (Ctrl+B)">
          <Bold size={13} />
        </ToolbarBtn>
        <ToolbarBtn onClick={() => execCmd("italic")} title="Italic (Ctrl+I)">
          <Italic size={13} />
        </ToolbarBtn>

        <div className="w-px h-4 bg-border/50 mx-1" />

        <ToolbarBtn onClick={() => execCmd("insertUnorderedList")} title="Bullet list">
          <List size={13} />
        </ToolbarBtn>
        <ToolbarBtn onClick={() => execCmd("insertOrderedList")} title="Numbered list">
          <ListOrdered size={13} />
        </ToolbarBtn>

        <div className="w-px h-4 bg-border/50 mx-1" />

        <ToolbarBtn onClick={() => execCmd("undo")} title="Undo (Ctrl+Z)">
          <Undo2 size={13} />
        </ToolbarBtn>
        <ToolbarBtn onClick={() => execCmd("redo")} title="Redo (Ctrl+Y)">
          <Redo2 size={13} />
        </ToolbarBtn>
      </div>

      {/* ── Editor body ── */}
      <div
        ref={editorRef}
        contentEditable={!isStreaming}
        suppressContentEditableWarning
        onInput={handleEditorInput}
        spellCheck
        className={cn(
          "flex-1 overflow-y-auto px-10 py-8 text-sm text-foreground leading-relaxed outline-none",
          "canvas-editor",
          isStreaming && "pointer-events-none select-none"
        )}
        style={{ minHeight: 0 }}
      />

      {/* ── Canvas editor styles injected globally ── */}
      <style>{`
        .canvas-editor h1 { font-size: 1.6rem; font-weight: 700; margin: 1.25rem 0 0.6rem; line-height: 1.25; }
        .canvas-editor h2 { font-size: 1.25rem; font-weight: 600; margin: 1rem 0 0.5rem; line-height: 1.3; }
        .canvas-editor h3 { font-size: 1.05rem; font-weight: 600; margin: 0.8rem 0 0.4rem; line-height: 1.35; }
        .canvas-editor p  { margin: 0.4rem 0; }
        .canvas-editor ul { list-style: disc;   padding-left: 1.4rem; margin: 0.4rem 0; }
        .canvas-editor ol { list-style: decimal; padding-left: 1.4rem; margin: 0.4rem 0; }
        .canvas-editor li { margin: 0.15rem 0; }
        .canvas-editor b, .canvas-editor strong { font-weight: 700; }
        .canvas-editor i, .canvas-editor em { font-style: italic; }
        .canvas-editor [contenteditable]:focus { outline: none; }
        .canvas-editor::selection, .canvas-editor *::selection { background: rgba(99,102,241,0.25); }

        /* ── Diff highlight: <ins> tags from AI surgical updates ── */
        @keyframes canvas-ins-fade {
          0%   { background-color: rgba(34,197,94,0.30); }
          60%  { background-color: rgba(34,197,94,0.18); }
          100% { background-color: transparent; }
        }
        .canvas-editor ins {
          text-decoration: none;
          border-radius: 3px;
          padding: 0 1px;
          animation: canvas-ins-fade 5s ease-out forwards;
          background-color: transparent;
        }
      `}</style>
    </div>
  )
}
