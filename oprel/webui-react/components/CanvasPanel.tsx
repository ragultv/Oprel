"use client"

import { useEffect, useRef, useState, useCallback, useMemo } from "react"
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
  Table as TableIcon,
  Maximize2,
  Minimize2,
  Share2,
  Plus,
  Eye,
  Type,
  MoveHorizontal,
  Trash2,
  Copy,
  SlidersHorizontal,
  Layers,
  Sparkles,
} from "lucide-react"
import { cn } from "@/services/utils"

// ─── Document Types ──────────────────────────────────────────────────────────

export interface CanvasTab {
  id: string
  title: string
  content: string
}

export interface CanvasDocument {
  id: string
  title: string
  author?: string
  date?: string
  activeTabId?: string
  tabs?: CanvasTab[]
  content: string // Active tab content (HTML)
  createdAt: Date
  updatedAt: Date
}

interface CanvasPanelProps {
  document: CanvasDocument
  onClose: () => void
  onChange: (doc: Partial<CanvasDocument>) => void
  isStreaming?: boolean
}

// ─── Typography & Block Styles ───────────────────────────────────────────────

const TEXT_STYLES = [
  { label: "Normal text", tag: "p", className: "text-sm leading-relaxed" },
  { label: "Title", tag: "h1", className: "text-2xl font-serif font-bold leading-tight" },
  { label: "Heading 1", tag: "h2", className: "text-xl font-bold leading-snug" },
  { label: "Heading 2", tag: "h3", className: "text-base font-semibold leading-snug" },
  { label: "Heading 3", tag: "h4", className: "text-sm font-semibold leading-snug" },
  { label: "Quote", tag: "blockquote", className: "text-sm italic pl-3 border-l-2 border-primary/50 text-muted-foreground" },
]

// ─── Helpers ──────────────────────────────────────────────────────────────────

export function cleanCanvasHtml(raw: string): string {
  if (!raw) return ""
  let cleaned = raw.trim()
  cleaned = cleaned.replace(/^```(?:html|xml)?\s*\n?/i, "")
  cleaned = cleaned.replace(/\n?```\s*$/i, "")
  return cleaned.trim()
}

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
  return ["h1", "h2", "h3", "h4", "blockquote"].includes(tag) ? tag : "p"
}

function applyBlockFormat(tag: string) {
  if (tag === "blockquote") {
    execCmd("formatBlock", "<blockquote>")
  } else {
    execCmd("formatBlock", `<${tag}>`)
  }
}

// ─── Export Helpers ──────────────────────────────────────────────────────────

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
    body { font-family: 'Georgia', serif; font-size: 11pt; line-height: 1.6; color: #111; background: #fff; padding: 0; margin: 0; }
    h1 { font-size: 22pt; font-weight: 700; margin: 16pt 0 8pt; font-family: 'Georgia', serif; }
    h2 { font-size: 16pt; font-weight: 600; margin: 14pt 0 6pt; }
    h3 { font-size: 13pt; font-weight: 600; margin: 10pt 0 4pt; }
    p  { margin: 6pt 0; }
    ul, ol { margin: 6pt 0 6pt 20pt; }
    li { margin: 2pt 0; }
    table { width: 100%; border-collapse: collapse; margin: 12pt 0; font-size: 10pt; }
    th, td { border: 1px solid #ccc; padding: 6pt 8pt; text-align: left; }
    th { background: #f4f4f4; font-weight: 600; }
    b, strong { font-weight: 700; }
    i, em { font-style: italic; }
  </style>
  </head><body>
    <h1>${title}</h1>
    ${content}
  </body></html>`)
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
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; font-size: 11pt; line-height: 1.6; max-width: 800px; margin: 40px auto; color: #222; }
  h1 { font-size: 24pt; font-weight: 700; margin: 20pt 0 8pt; font-family: Georgia, serif; }
  h2 { font-size: 16pt; font-weight: 600; margin: 16pt 0 6pt; }
  h3 { font-size: 13pt; font-weight: 600; margin: 12pt 0 4pt; }
  p  { margin: 6pt 0; }
  ul, ol { margin: 6pt 0 6pt 20pt; }
  li { margin: 2pt 0; }
  table { width: 100%; border-collapse: collapse; margin: 16pt 0; }
  th, td { border: 1px solid #ddd; padding: 8pt 10pt; text-align: left; }
  th { background: #f8f9fa; font-weight: 600; }
</style>
</head>
<body>
  <h1>${title}</h1>
  ${content}
</body>
</html>`

  const blob = new Blob([html], { type: "text/html;charset=utf-8" })
  const url = URL.createObjectURL(blob)
  const a = window.document.createElement("a")
  a.href = url
  a.download = `${title.replace(/[^a-z0-9]/gi, "_") || "canvas"}.html`
  a.click()
  URL.revokeObjectURL(url)
}

// ─── Toolbar Button Component ────────────────────────────────────────────────

function ToolbarBtn({
  onClick,
  active,
  title,
  children,
  className,
}: {
  onClick: () => void
  active?: boolean
  title: string
  children: React.ReactNode
  className?: string
}) {
  return (
    <button
      type="button"
      onMouseDown={(e) => {
        e.preventDefault()
        onClick()
      }}
      title={title}
      className={cn(
        "h-7 px-1.5 flex items-center justify-center rounded-lg transition-all text-xs font-medium gap-1",
        active
          ? "bg-primary/20 text-primary border border-primary/30"
          : "text-muted-foreground hover:bg-white/5 hover:text-foreground",
        className
      )}
    >
      {children}
    </button>
  )
}

// ─── Main CanvasPanel Component ──────────────────────────────────────────────

export function CanvasPanel({ document: doc, onClose, onChange, isStreaming }: CanvasPanelProps) {
  const editorRef = useRef<HTMLDivElement>(null)
  const isUserTypingRef = useRef(false)

  // Typography state
  const [blockTag, setBlockTag] = useState("p")
  const [fontFamily, setFontFamily] = useState<"serif" | "sans">("serif")
  const [styleDropdown, setStyleDropdown] = useState(false)
  const [fontDropdown, setFontDropdown] = useState(false)

  // Mode state: Editing vs Viewing (clean presentation)
  const [canvasMode, setCanvasMode] = useState<"editing" | "viewing">("editing")
  const [modeDropdown, setModeDropdown] = useState(false)

  // Width layout: Compact reading column (760px) vs Full width (100%)
  const [layoutWidth, setLayoutWidth] = useState<"reading" | "full">("reading")

  // Fullscreen state
  const [isFullscreen, setIsFullscreen] = useState(false)

  // Tabs management
  const [tabs, setTabs] = useState<CanvasTab[]>(() => {
    if (doc.tabs && doc.tabs.length > 0) return doc.tabs
    return [{ id: "tab-1", title: "Tab 1", content: doc.content || "" }]
  })
  const [activeTabId, setActiveTabId] = useState<string>(() => {
    return doc.activeTabId || tabs[0]?.id || "tab-1"
  })
  const [tabDropdown, setTabDropdown] = useState(false)

  // Title editing state
  const [editingTitle, setEditingTitle] = useState(false)
  const [titleVal, setTitleVal] = useState(doc.title || "Untitled Document")
  const titleInputRef = useRef<HTMLInputElement>(null)

  // Word count and reading stats calculation
  const stats = useMemo(() => {
    const raw = (editorRef.current?.innerText || doc.content || "").replace(/<[^>]+>/g, " ")
    const words = raw.trim() ? raw.trim().split(/\s+/).length : 0
    const chars = raw.length
    const mins = Math.max(1, Math.ceil(words / 200))
    return { words, chars, mins }
  }, [doc.content])

  // Sync incoming content safely without overwriting user cursor
  useEffect(() => {
    if (!editorRef.current) return
    const currentTab = tabs.find((t) => t.id === activeTabId)
    const targetContent = cleanCanvasHtml(currentTab?.content ?? doc.content)

    // Only inject if not currently actively focused/typing or if streaming
    if (!isUserTypingRef.current || isStreaming) {
      if (editorRef.current.innerHTML !== targetContent) {
        editorRef.current.innerHTML = targetContent
        if (isStreaming) {
          editorRef.current.scrollTop = editorRef.current.scrollHeight
        }
      }
    }
  }, [doc.content, activeTabId, tabs, isStreaming])

  // Sync title from doc prop
  useEffect(() => {
    setTitleVal(doc.title || "Untitled Document")
  }, [doc.title])

  // Editor Input Callback
  const handleEditorInput = useCallback(() => {
    if (!editorRef.current) return
    isUserTypingRef.current = true
    const updatedHtml = editorRef.current.innerHTML

    // Update active tab in local state
    setTabs((prev) =>
      prev.map((t) => (t.id === activeTabId ? { ...t, content: updatedHtml } : t))
    )

    // Notify parent
    onChange({
      content: updatedHtml,
      tabs: tabs.map((t) => (t.id === activeTabId ? { ...t, content: updatedHtml } : t)),
      updatedAt: new Date(),
    })

    // Reset typing flag after debounce
    setTimeout(() => {
      isUserTypingRef.current = false
    }, 400)
  }, [activeTabId, tabs, onChange])

  // Selection change to reflect active block tag (h1, h2, p, etc.)
  const handleSelectionChange = useCallback(() => {
    setBlockTag(getBlockTag())
  }, [])

  useEffect(() => {
    window.document.addEventListener("selectionchange", handleSelectionChange)
    return () => window.document.removeEventListener("selectionchange", handleSelectionChange)
  }, [handleSelectionChange])

  // Apply block style
  const applyStyle = (tag: string) => {
    applyBlockFormat(tag)
    setBlockTag(tag)
    setStyleDropdown(false)
    editorRef.current?.focus()
  }

  const currentStyleLabel = TEXT_STYLES.find((s) => s.tag === blockTag)?.label ?? "Normal text"

  // Title save
  const saveTitle = () => {
    setEditingTitle(false)
    if (titleVal.trim()) {
      onChange({ title: titleVal.trim(), updatedAt: new Date() })
    } else {
      setTitleVal(doc.title || "Untitled Document")
    }
  }

  // Tab switching
  const handleSelectTab = (tabId: string) => {
    if (!editorRef.current) return
    // Save current content to current tab first
    const currentHtml = editorRef.current.innerHTML
    const updated = tabs.map((t) => (t.id === activeTabId ? { ...t, content: currentHtml } : t))
    setTabs(updated)

    // Switch active tab
    setActiveTabId(tabId)
    setTabDropdown(false)

    const targetTab = updated.find((t) => t.id === tabId)
    if (targetTab) {
      editorRef.current.innerHTML = targetTab.content
      onChange({
        activeTabId: tabId,
        content: targetTab.content,
        tabs: updated,
        updatedAt: new Date(),
      })
    }
  }

  // Add new tab
  const handleAddTab = () => {
    const newId = `tab-${Date.now()}`
    const newTabNum = tabs.length + 1
    const newTab: CanvasTab = {
      id: newId,
      title: `Tab ${newTabNum}`,
      content: `<p>Start writing section ${newTabNum}...</p>`,
    }
    const updated = [...tabs, newTab]
    setTabs(updated)
    handleSelectTab(newId)
  }

  // Insert Rich Interactive Table
  const insertTable = () => {
    if (!editorRef.current) return
    const tableHtml = `
      <div class="canvas-table-wrapper my-6 overflow-x-auto rounded-xl border border-white/10 bg-black/40 shadow-inner">
        <table class="w-full text-left text-xs border-collapse font-sans">
          <thead>
            <tr class="bg-white/5 border-b border-white/10 text-foreground font-semibold">
              <th class="px-3.5 py-2.5 border-r border-white/10">Feature / Tier</th>
              <th class="px-3.5 py-2.5 border-r border-white/10">Specification</th>
              <th class="px-3.5 py-2.5 border-r border-white/10">Memory / VRAM</th>
              <th class="px-3.5 py-2.5">Status</th>
            </tr>
          </thead>
          <tbody class="divide-y divide-white/5 text-muted-foreground">
            <tr class="hover:bg-white/5 transition-colors">
              <td class="px-3.5 py-2.5 border-r border-white/5 text-foreground font-medium">Core Engine</td>
              <td class="px-3.5 py-2.5 border-r border-white/5">Unified multi-stage pipeline</td>
              <td class="px-3.5 py-2.5 border-r border-white/5">4 GB - 8 GB</td>
              <td class="px-3.5 py-2.5 text-emerald-400 font-semibold">Active</td>
            </tr>
            <tr class="hover:bg-white/5 transition-colors">
              <td class="px-3.5 py-2.5 border-r border-white/5 text-foreground font-medium">KV Cache</td>
              <td class="px-3.5 py-2.5 border-r border-white/5">Paged Q4 / Q8 Quantization</td>
              <td class="px-3.5 py-2.5 border-r border-white/5">~1.2 GB (128k)</td>
              <td class="px-3.5 py-2.5 text-sky-400 font-semibold">Optimized</td>
            </tr>
            <tr class="hover:bg-white/5 transition-colors">
              <td class="px-3.5 py-2.5 border-r border-white/5 text-foreground font-medium">Speculative Decoding</td>
              <td class="px-3.5 py-2.5 border-r border-white/5">Multi-token prediction head</td>
              <td class="px-3.5 py-2.5 border-r border-white/5">+150 MB</td>
              <td class="px-3.5 py-2.5 text-amber-400 font-semibold">Planned</td>
            </tr>
          </tbody>
        </table>
      </div>
      <p><br></p>
    `
    execCmd("insertHTML", tableHtml)
    handleEditorInput()
  }

  // Insert Architecture Diagram Node
  const insertArchitectureDiagram = () => {
    if (!editorRef.current) return
    const diagramHtml = `
      <div class="my-6 p-5 rounded-2xl border border-white/10 bg-[#161616] shadow-xl font-sans">
        <div class="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-3">System Architecture Pipeline</div>
        <div class="space-y-3">
          <div class="p-3 rounded-xl bg-black/50 border border-white/10 text-xs text-foreground">
            <span class="font-bold text-primary">1. HTTP Gateway:</span> Single-flight gate, tokenizer, and SSE streaming router
          </div>
          <div class="flex justify-center text-muted-foreground/60 text-xs">↓ (one request at a time; 503 if busy)</div>
          <div class="p-3 rounded-xl bg-blue-500/10 border border-blue-500/20 text-xs text-blue-200">
            <span class="font-bold text-blue-400">2. Engine Thread:</span> Owns GPU session, Paged KV Cache, and chunked prefill
          </div>
          <div class="flex justify-center text-muted-foreground/60 text-xs">↓ (calls backend for loaded model)</div>
          <div class="grid grid-cols-2 gap-3">
            <div class="p-2.5 rounded-xl bg-white/5 border border-white/10 text-[11px] text-muted-foreground">
              <span class="font-semibold text-foreground">Memory Engine:</span> Paged KV pool, BF16/FP8/Q4
            </div>
            <div class="p-2.5 rounded-xl bg-white/5 border border-white/10 text-[11px] text-muted-foreground">
              <span class="font-semibold text-foreground">Compute Engine:</span> CUDA graph replay & fused kernels
            </div>
          </div>
        </div>
      </div>
      <p><br></p>
    `
    execCmd("insertHTML", diagramHtml)
    handleEditorInput()
  }

  // Keyboard navigation inside tables: Tab / Shift+Tab moves between cells
  const handleKeyDown = (e: React.KeyboardEvent<HTMLDivElement>) => {
    if (e.key === "Tab") {
      const sel = window.getSelection()
      if (!sel || !sel.anchorNode) return
      let el: HTMLElement | null = sel.anchorNode as HTMLElement
      while (el && el.tagName !== "TD" && el.tagName !== "TH") {
        el = el.parentElement
      }
      if (el) {
        e.preventDefault()
        const table = el.closest("table")
        if (table) {
          const cells = Array.from(table.querySelectorAll("th, td")) as HTMLElement[]
          const currentIndex = cells.indexOf(el)
          if (currentIndex !== -1) {
            const nextIndex = e.shiftKey ? currentIndex - 1 : currentIndex + 1
            if (nextIndex >= 0 && nextIndex < cells.length) {
              cells[nextIndex].focus()
              const range = window.document.createRange()
              range.selectNodeContents(cells[nextIndex])
              sel.removeAllRanges()
              sel.addRange(range)
            }
          }
        }
      }
    }
  }

  // Copy document markdown/text
  const handleShare = () => {
    const rawText = editorRef.current?.innerText || doc.content
    navigator.clipboard.writeText(`# ${titleVal}\n\n${rawText}`)
  }

  const activeTab = tabs.find((t) => t.id === activeTabId) || tabs[0]

  return (
    <div
      className={cn(
        "flex flex-col h-full bg-[#121212] border-l border-white/10 overflow-hidden text-foreground transition-all",
        isFullscreen && "fixed inset-0 z-50 bg-[#121212]"
      )}
    >
      {/* ── Row 1: Document Top Navigation ── */}
      <div className="h-12 flex items-center justify-between px-4 border-b border-white/10 shrink-0 gap-3 bg-[#161616]">
        {/* Document Title Selector */}
        <div className="flex items-center gap-2 min-w-0 flex-1">
          <div className="w-6 h-6 rounded-lg bg-primary/10 border border-primary/20 flex items-center justify-center shrink-0 text-primary">
            <FileText size={13} />
          </div>

          {editingTitle ? (
            <input
              ref={titleInputRef}
              value={titleVal}
              onChange={(e) => setTitleVal(e.target.value)}
              onBlur={saveTitle}
              onKeyDown={(e) => {
                if (e.key === "Enter") saveTitle()
                if (e.key === "Escape") {
                  setEditingTitle(false)
                  setTitleVal(doc.title)
                }
              }}
              autoFocus
              className="text-xs font-semibold bg-black/60 border border-primary/50 rounded-lg px-2.5 py-1 text-foreground outline-none flex-1 min-w-0"
            />
          ) : (
            <button
              type="button"
              className="flex items-center gap-1.5 group min-w-0 text-left"
              onClick={() => setEditingTitle(true)}
              title="Click to rename document"
            >
              <span className="text-xs font-semibold text-foreground truncate max-w-[280px]">
                {titleVal}
              </span>
              <Pencil size={11} className="shrink-0 opacity-0 group-hover:opacity-60 transition-opacity text-muted-foreground" />
            </button>
          )}

          {isStreaming && (
            <span className="text-[10px] text-primary animate-pulse font-semibold px-2 py-0.5 bg-primary/10 rounded-full border border-primary/20 shrink-0">
              Generating…
            </span>
          )}
        </div>

        {/* Top Right Workspace Actions */}
        <div className="flex items-center gap-1 shrink-0">
          <button
            type="button"
            onClick={handleShare}
            title="Copy Document Markdown"
            className="flex items-center gap-1 px-2.5 py-1 rounded-lg text-xs font-medium bg-white/5 hover:bg-white/10 text-muted-foreground hover:text-foreground transition-all"
          >
            <Share2 size={12} />
            <span className="hidden sm:inline">Share</span>
          </button>

          <button
            type="button"
            onClick={() => exportPDF(titleVal, editorRef.current?.innerHTML ?? doc.content)}
            title="Export / Print PDF"
            className="p-1.5 rounded-lg text-muted-foreground hover:text-foreground hover:bg-white/5 transition-all"
          >
            <Printer size={13} />
          </button>

          <button
            type="button"
            onClick={() => exportHTML(titleVal, editorRef.current?.innerHTML ?? doc.content)}
            title="Export HTML"
            className="p-1.5 rounded-lg text-muted-foreground hover:text-foreground hover:bg-white/5 transition-all"
          >
            <Download size={13} />
          </button>

          <button
            type="button"
            onClick={() => setIsFullscreen((prev) => !prev)}
            title={isFullscreen ? "Exit Fullscreen" : "Expand Fullscreen"}
            className="p-1.5 rounded-lg text-muted-foreground hover:text-foreground hover:bg-white/5 transition-all"
          >
            {isFullscreen ? <Minimize2 size={13} /> : <Maximize2 size={13} />}
          </button>

          <div className="w-px h-4 bg-white/10 mx-1" />

          <button
            type="button"
            onClick={onClose}
            title="Close canvas"
            className="p-1.5 rounded-lg text-muted-foreground hover:text-foreground hover:bg-white/5 transition-all"
          >
            <X size={14} />
          </button>
        </div>
      </div>

      {/* ── Row 2: Subheader Interactive Toolbar ── */}
      <div className="h-10 flex items-center justify-between px-3 border-b border-white/10 shrink-0 bg-[#141414] gap-2 overflow-x-auto select-none">
        {/* Left Toolbar Cluster */}
        <div className="flex items-center gap-1 shrink-0">
          {/* Tab 1 ▾ Dropdown */}
          <div className="relative">
            <button
              type="button"
              onClick={() => setTabDropdown((v) => !v)}
              className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-semibold bg-white/5 hover:bg-white/10 text-foreground transition-all border border-white/10"
            >
              <span>{activeTab?.title || "Tab 1"}</span>
              <ChevronDown size={11} className={cn("transition-transform text-muted-foreground", tabDropdown && "rotate-180")} />
            </button>

            {tabDropdown && (
              <>
                <div className="fixed inset-0 z-40" onClick={() => setTabDropdown(false)} />
                <div className="absolute top-full left-0 mt-1 w-44 bg-[#1a1a1a] border border-white/10 rounded-xl shadow-2xl py-1.5 z-50 animate-in fade-in slide-in-from-top-1 duration-150">
                  <div className="px-2.5 py-1 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
                    Document Sections
                  </div>
                  {tabs.map((tab) => (
                    <button
                      key={tab.id}
                      type="button"
                      onClick={() => handleSelectTab(tab.id)}
                      className={cn(
                        "w-full text-left px-3 py-1.5 text-xs transition-all flex items-center justify-between",
                        tab.id === activeTabId
                          ? "bg-primary/15 text-primary font-semibold"
                          : "text-foreground/80 hover:bg-white/5 hover:text-foreground"
                      )}
                    >
                      <span>{tab.title}</span>
                      {tab.id === activeTabId && <Check size={12} className="text-primary" />}
                    </button>
                  ))}
                  <div className="h-px bg-white/10 my-1" />
                  <button
                    type="button"
                    onClick={() => {
                      handleAddTab()
                      setTabDropdown(false)
                    }}
                    className="w-full text-left px-3 py-1.5 text-xs text-primary hover:bg-primary/10 transition-colors flex items-center gap-1.5 font-medium"
                  >
                    <Plus size={12} />
                    <span>Add New Tab</span>
                  </button>
                </div>
              </>
            )}
          </div>

          <div className="w-px h-4 bg-white/10 mx-1" />

          {/* TT: Typography Selector */}
          <div className="relative">
            <ToolbarBtn onClick={() => setStyleDropdown((v) => !v)} title="Format Style">
              <Type size={13} />
              <span className="hidden sm:inline text-[11px] font-medium">{currentStyleLabel}</span>
              <ChevronDown size={10} className="text-muted-foreground" />
            </ToolbarBtn>

            {styleDropdown && (
              <>
                <div className="fixed inset-0 z-40" onClick={() => setStyleDropdown(false)} />
                <div className="absolute top-full left-0 mt-1 w-48 bg-[#1a1a1a] border border-white/10 rounded-xl shadow-2xl py-1.5 z-50 animate-in fade-in slide-in-from-top-1 duration-150">
                  {TEXT_STYLES.map((s) => (
                    <button
                      key={s.tag}
                      type="button"
                      onClick={() => applyStyle(s.tag)}
                      className={cn(
                        "w-full text-left px-3 py-1.5 transition-all flex items-center justify-between",
                        s.className,
                        blockTag === s.tag
                          ? "bg-primary/15 text-primary font-semibold"
                          : "text-foreground/80 hover:bg-white/5 hover:text-foreground"
                      )}
                    >
                      <span>{s.label}</span>
                      {blockTag === s.tag && <Check size={12} className="text-primary" />}
                    </button>
                  ))}
                  <div className="h-px bg-white/10 my-1" />
                  <div className="px-3 py-1 flex items-center justify-between">
                    <span className="text-[11px] text-muted-foreground">Serif Font</span>
                    <button
                      type="button"
                      onClick={() => setFontFamily((f) => (f === "serif" ? "sans" : "serif"))}
                      className={cn(
                        "px-2 py-0.5 rounded text-[10px] font-bold border transition-colors",
                        fontFamily === "serif"
                          ? "bg-primary/20 text-primary border-primary/40"
                          : "bg-white/5 text-muted-foreground border-white/10"
                      )}
                    >
                      {fontFamily === "serif" ? "Serif" : "Sans"}
                    </button>
                  </div>
                </div>
              </>
            )}
          </div>

          {/* ⟷ Reading Width Toggle */}
          <ToolbarBtn
            onClick={() => setLayoutWidth((w) => (w === "reading" ? "full" : "reading"))}
            title={layoutWidth === "reading" ? "Switch to Full Width" : "Switch to Reading Width (Centered)"}
            active={layoutWidth === "full"}
          >
            <MoveHorizontal size={13} />
            <span className="hidden md:inline text-[10px] uppercase font-bold tracking-wider">
              {layoutWidth === "reading" ? "Reading" : "Full"}
            </span>
          </ToolbarBtn>

          {/* Insert Table Icon */}
          <ToolbarBtn onClick={insertTable} title="Insert Rich Interactive Table">
            <TableIcon size={13} />
            <span className="hidden lg:inline text-[11px]">Table</span>
          </ToolbarBtn>

          {/* Insert Architecture / Diagram Block */}
          <ToolbarBtn onClick={insertArchitectureDiagram} title="Insert Architecture Flow Block">
            <Layers size={13} />
            <span className="hidden lg:inline text-[11px]">Architecture</span>
          </ToolbarBtn>

          <div className="w-px h-4 bg-white/10 mx-1" />

          {/* Inline Text Formatting */}
          <ToolbarBtn onClick={() => execCmd("bold")} title="Bold (Ctrl+B)">
            <Bold size={12} />
          </ToolbarBtn>
          <ToolbarBtn onClick={() => execCmd("italic")} title="Italic (Ctrl+I)">
            <Italic size={12} />
          </ToolbarBtn>

          <div className="w-px h-4 bg-white/10 mx-1" />

          <ToolbarBtn onClick={() => execCmd("insertUnorderedList")} title="Bullet List">
            <List size={12} />
          </ToolbarBtn>
          <ToolbarBtn onClick={() => execCmd("insertOrderedList")} title="Numbered List">
            <ListOrdered size={12} />
          </ToolbarBtn>

          <div className="w-px h-4 bg-white/10 mx-1" />

          <ToolbarBtn onClick={() => execCmd("undo")} title="Undo (Ctrl+Z)">
            <Undo2 size={12} />
          </ToolbarBtn>
          <ToolbarBtn onClick={() => execCmd("redo")} title="Redo (Ctrl+Y)">
            <Redo2 size={12} />
          </ToolbarBtn>
        </div>

        {/* Right Mode Switcher: Editing ▾ vs Viewing */}
        <div className="relative shrink-0">
          <button
            type="button"
            onClick={() => setModeDropdown((v) => !v)}
            className={cn(
              "flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-semibold transition-all border",
              canvasMode === "editing"
                ? "bg-primary/10 text-primary border-primary/30"
                : "bg-white/5 text-muted-foreground border-white/10 hover:text-foreground"
            )}
          >
            {canvasMode === "editing" ? <Pencil size={11} /> : <Eye size={11} />}
            <span className="capitalize">{canvasMode}</span>
            <ChevronDown size={10} className="text-muted-foreground" />
          </button>

          {modeDropdown && (
            <>
              <div className="fixed inset-0 z-40" onClick={() => setModeDropdown(false)} />
              <div className="absolute top-full right-0 mt-1 w-36 bg-[#1a1a1a] border border-white/10 rounded-xl shadow-2xl py-1 z-50 animate-in fade-in slide-in-from-top-1 duration-150">
                <button
                  type="button"
                  onClick={() => {
                    setCanvasMode("editing")
                    setModeDropdown(false)
                  }}
                  className={cn(
                    "w-full text-left px-3 py-1.5 text-xs flex items-center justify-between",
                    canvasMode === "editing" ? "bg-primary/15 text-primary font-semibold" : "text-foreground hover:bg-white/5"
                  )}
                >
                  <span className="flex items-center gap-1.5">
                    <Pencil size={11} /> Editing
                  </span>
                  {canvasMode === "editing" && <Check size={11} />}
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setCanvasMode("viewing")
                    setModeDropdown(false)
                  }}
                  className={cn(
                    "w-full text-left px-3 py-1.5 text-xs flex items-center justify-between",
                    canvasMode === "viewing" ? "bg-primary/15 text-primary font-semibold" : "text-foreground hover:bg-white/5"
                  )}
                >
                  <span className="flex items-center gap-1.5">
                    <Eye size={11} /> Viewing
                  </span>
                  {canvasMode === "viewing" && <Check size={11} />}
                </button>
              </div>
            </>
          )}
        </div>
      </div>

      {/* ── Document Scroll Canvas ── */}
      <div className="flex-1 overflow-y-auto bg-[#0f0f0f] px-4 md:px-8 py-8 md:py-10">
        <div
          className={cn(
            "transition-all duration-200",
            layoutWidth === "reading" ? "max-w-[760px] mx-auto" : "w-full px-2 md:px-6"
          )}
        >
          {/* Document Hero Header */}
          <div className="mb-8 pb-6 border-b border-white/10">
            <h1
              className={cn(
                "text-2xl md:text-3xl font-bold tracking-tight text-foreground mb-3 leading-tight",
                fontFamily === "serif" ? "font-serif" : "font-sans"
              )}
            >
              {titleVal}
            </h1>

            {/* Metadata Badges Bar */}
            <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground font-sans">
              <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full bg-white/5 border border-white/10 text-foreground/80">
                📅 {doc.date || new Date().toLocaleDateString("en-US", { day: "numeric", month: "short", year: "numeric" })}
              </span>
              <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full bg-white/5 border border-white/10 text-foreground/80">
                👤 {doc.author || "@ragul"}
              </span>
              <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full bg-white/5 border border-white/10 text-muted-foreground">
                ⏱️ {stats.mins} min read • {stats.words} words
              </span>
            </div>
          </div>

          {/* Living Document Content Area */}
          <div
            ref={editorRef}
            contentEditable={canvasMode === "editing" && !isStreaming}
            suppressContentEditableWarning
            onInput={handleEditorInput}
            onKeyDown={handleKeyDown}
            spellCheck={canvasMode === "editing"}
            className={cn(
              "text-sm leading-relaxed text-foreground/90 outline-none select-text",
              fontFamily === "serif" ? "font-serif" : "font-sans",
              "oprel-canvas-document",
              canvasMode === "viewing" && "cursor-default",
              isStreaming && "pointer-events-none select-none"
            )}
            style={{ minHeight: "60vh" }}
          />
        </div>
      </div>

      {/* ── Scoped Styling for Living Document, Tables & Architecture Nodes ── */}
      <style>{`
        .oprel-canvas-document h1 {
          font-size: 1.75rem;
          font-weight: 700;
          margin: 1.5rem 0 0.75rem;
          line-height: 1.25;
          color: #ffffff;
        }
        .oprel-canvas-document h2 {
          font-size: 1.35rem;
          font-weight: 600;
          margin: 1.35rem 0 0.6rem;
          line-height: 1.3;
          color: #f3f4f6;
        }
        .oprel-canvas-document h3 {
          font-size: 1.1rem;
          font-weight: 600;
          margin: 1rem 0 0.45rem;
          line-height: 1.35;
          color: #e5e7eb;
        }
        .oprel-canvas-document p {
          margin: 0.65rem 0;
          color: rgba(255, 255, 255, 0.88);
          line-height: 1.7;
        }
        .oprel-canvas-document ul {
          list-style: disc;
          padding-left: 1.5rem;
          margin: 0.65rem 0;
        }
        .oprel-canvas-document ol {
          list-style: decimal;
          padding-left: 1.5rem;
          margin: 0.65rem 0;
        }
        .oprel-canvas-document li {
          margin: 0.35rem 0;
          line-height: 1.65;
        }
        .oprel-canvas-document b, .oprel-canvas-document strong {
          font-weight: 700;
          color: #ffffff;
        }
        .oprel-canvas-document i, .oprel-canvas-document em {
          font-style: italic;
        }
        .oprel-canvas-document blockquote {
          border-left: 3px solid rgba(230, 115, 90, 0.6);
          padding-left: 1rem;
          margin: 1rem 0;
          font-style: italic;
          color: rgba(255, 255, 255, 0.65);
        }

        /* ── Table Styling matching Claude Canvas ── */
        .oprel-canvas-document table {
          width: 100%;
          border-collapse: collapse;
          margin: 1.25rem 0;
          font-family: ui-sans-serif, system-ui, sans-serif;
          font-size: 0.825rem;
          background: rgba(0, 0, 0, 0.35);
          border-radius: 0.75rem;
          overflow: hidden;
          border: 1px solid rgba(255, 255, 255, 0.12);
        }
        .oprel-canvas-document th, .oprel-canvas-document td {
          padding: 0.65rem 0.85rem;
          text-align: left;
          border: 1px solid rgba(255, 255, 255, 0.08);
          line-height: 1.5;
        }
        .oprel-canvas-document th {
          background: rgba(255, 255, 255, 0.06);
          font-weight: 600;
          color: #ffffff;
          letter-spacing: 0.01em;
        }
        .oprel-canvas-document tr:hover td {
          background: rgba(255, 255, 255, 0.03);
        }
        .oprel-canvas-document td[contenteditable]:focus, .oprel-canvas-document th[contenteditable]:focus {
          outline: 1px solid rgba(230, 115, 90, 0.6);
          background: rgba(230, 115, 90, 0.05);
        }
      `}</style>
    </div>
  )
}
