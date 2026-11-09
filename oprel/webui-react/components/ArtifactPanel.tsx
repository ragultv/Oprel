'use client'

import { useState, useEffect, useRef } from 'react'
import {
  Eye,
  Code2,
  Download,
  X,
  GitBranch,
  Globe,
  AlertCircle,
  Loader2,
  FileText,
  ZoomIn,
  ZoomOut,
  RotateCcw,
  Maximize2,
} from 'lucide-react'
import { cn } from '@/services/utils'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism'

// ─── Types ───────────────────────────────────────────────────────────────────

export interface Artifact {
  type: 'mermaid' | 'html' | 'text'
  code: string
  title?: string
  chunk?: string
}

// ─── Mermaid Preview ─────────────────────────────────────────────────────────

function MermaidPreview({ code }: { code: string }) {
  const [svg, setSvg] = useState<string>('')
  const [error, setError] = useState<string>('')
  const [loading, setLoading] = useState(true)

  // Pan & Zoom state
  const [zoom, setZoom] = useState<number>(1.0)
  const [pan, setPan] = useState<{ x: number; y: number }>({ x: 0, y: 0 })
  const [isDragging, setIsDragging] = useState(false)
  const dragStartRef = useRef<{ x: number; y: number }>({ x: 0, y: 0 })
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    setError('')
    setSvg('')

    const cleanCode = (code || '')
      .replace(/^```(?:mermaid)?\s*\n?/i, '')
      .replace(/\n?```\s*$/i, '')
      .trim()

    if (!cleanCode) {
      setLoading(false)
      setError('Empty Mermaid diagram code')
      return
    }

    const uniqueId = `mermaid-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`

    import('mermaid').then(({ default: mermaid }) => {
      mermaid.initialize({
        startOnLoad: false,
        theme: 'dark',
        securityLevel: 'loose',
        fontFamily: 'Inter, system-ui, sans-serif',
        themeVariables: {
          darkMode: true,
          background: '#121212',
          primaryColor: '#3b82f6',
          primaryTextColor: '#f8fafc',
          primaryBorderColor: '#60a5fa',
          lineColor: '#94a3b8',
          secondaryColor: '#1e293b',
          tertiaryColor: '#0f172a',
          fontSize: '14px',
        },
      })
      return mermaid.render(uniqueId, cleanCode)
    }).then(({ svg: rendered }) => {
      if (!cancelled) {
        // Extract natural viewBox width to ensure the diagram is never squeezed or 0px
        const vbMatch = rendered.match(/viewBox=["']\s*([\d\.-]+)\s+([\d\.-]+)\s+([\d\.-]+)\s+([\d\.-]+)\s*["']/i)
        const naturalWidth = vbMatch ? Math.max(300, Math.round(parseFloat(vbMatch[3]))) : 650

        // Ensure SVG has explicit natural width so Chromium flexbox cannot collapse it to 0px
        let cleanSvg = rendered
          .replace(/style="[^"]*"/i, '')
          .replace(/<svg\b/i, `<svg style="width:${naturalWidth}px;min-width:${naturalWidth}px;max-width:none;height:auto;display:block;" `)

        setSvg(cleanSvg)
        setLoading(false)
        setZoom(1.0)
        setPan({ x: 0, y: 0 })
      }
    }).catch((err: Error) => {
      if (!cancelled) {
        console.error('Mermaid render error:', err)
        setError(err.message || 'Failed to render diagram')
        setLoading(false)
      }
    })

    return () => { cancelled = true }
  }, [code])

  const handleMouseDown = (e: React.MouseEvent) => {
    if (e.button !== 0) return
    setIsDragging(true)
    dragStartRef.current = { x: e.clientX - pan.x, y: e.clientY - pan.y }
  }

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDragging) return
    setPan({
      x: e.clientX - dragStartRef.current.x,
      y: e.clientY - dragStartRef.current.y,
    })
  }

  const handleMouseUp = () => {
    setIsDragging(false)
  }

  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault()
    const zoomFactor = e.deltaY < 0 ? 0.15 : -0.15
    setZoom((prev) => Math.min(4.0, Math.max(0.3, +(prev + zoomFactor).toFixed(2))))
  }

  const zoomIn = () => setZoom((prev) => Math.min(4.0, +(prev + 0.25).toFixed(2)))
  const zoomOut = () => setZoom((prev) => Math.max(0.3, +(prev - 0.25).toFixed(2)))
  const resetZoom = () => {
    setZoom(1.0)
    setPan({ x: 0, y: 0 })
  }
  const fitZoom = () => {
    setZoom(1.4)
    setPan({ x: 0, y: 0 })
  }

  if (loading) return (
    <div className="flex items-center justify-center h-full min-h-[200px]">
      <Loader2 size={22} className="text-primary animate-spin" />
    </div>
  )

  if (error) return (
    <div className="flex flex-col items-center justify-center min-h-[200px] gap-3 px-8 py-8 text-center">
      <AlertCircle size={32} className="text-destructive/80" />
      <p className="text-destructive font-semibold text-sm">Failed to render Mermaid diagram</p>
      <p className="text-muted-foreground text-[11px] font-mono bg-destructive/5 border border-destructive/10 px-3 py-2 rounded-lg max-w-sm break-all">{error}</p>
    </div>
  )

  return (
    <div className="relative w-full h-full bg-[#0d0d0d] select-none flex flex-col">
      {/* Scrollable / pannable viewport */}
      <div
        ref={containerRef}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onWheel={handleWheel}
        className={cn(
          "w-full flex-1 overflow-auto bg-[#0d0d0d] flex items-center justify-center p-8",
          isDragging ? "cursor-grabbing" : "cursor-grab"
        )}
      >
        {/* Centered transform container for diagram */}
        <div
          style={{
            transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
            transformOrigin: 'center center',
            transition: isDragging ? 'none' : 'transform 0.12s ease-out',
            minWidth: 'min-content',
            minHeight: 'min-content',
            display: 'inline-flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
          dangerouslySetInnerHTML={{ __html: svg }}
        />
      </div>

      {/* Floating HUD Controls */}
      <div className="absolute bottom-4 left-1/2 -translate-x-1/2 z-20 flex items-center gap-1 px-3 py-1.5 rounded-full bg-[#181818]/90 border border-white/10 backdrop-blur-md shadow-2xl text-xs text-muted-foreground select-none">
        <button
          type="button"
          onClick={zoomOut}
          title="Zoom out"
          className="p-1 rounded-md hover:bg-white/10 hover:text-foreground transition-colors"
        >
          <ZoomOut size={14} />
        </button>

        <button
          type="button"
          onClick={resetZoom}
          title="Click to reset to 100%"
          className="px-2 py-0.5 rounded font-mono text-[11px] font-semibold hover:bg-white/10 hover:text-foreground transition-colors text-foreground"
        >
          {Math.round(zoom * 100)}%
        </button>

        <button
          type="button"
          onClick={zoomIn}
          title="Zoom in"
          className="p-1 rounded-md hover:bg-white/10 hover:text-foreground transition-colors"
        >
          <ZoomIn size={14} />
        </button>

        <div className="w-px h-3.5 bg-white/15 mx-1" />

        <button
          type="button"
          onClick={fitZoom}
          title="Fit diagram (140%)"
          className="p-1 rounded-md hover:bg-white/10 hover:text-foreground transition-colors"
        >
          <Maximize2 size={13} />
        </button>

        <button
          type="button"
          onClick={resetZoom}
          title="Reset zoom & position"
          className="p-1 rounded-md hover:bg-white/10 hover:text-foreground transition-colors"
        >
          <RotateCcw size={13} />
        </button>
      </div>
    </div>
  )
}

// ─── HTML Preview ────────────────────────────────────────────────────────────

function HtmlPreview({ code }: { code: string }) {
  return (
    <iframe
      srcDoc={code}
      title="HTML Preview"
      sandbox="allow-scripts allow-same-origin allow-forms allow-popups"
      className="w-full h-full border-none bg-white"
    />
  )
}

// ─── ArtifactCard ────────────────────────────────────────────────────────────

export function ArtifactCard({ artifact, onClick }: { artifact: Artifact; onClick: () => void }) {
  const isMermaid = artifact.type === 'mermaid'

  const handleDownload = (e?: React.MouseEvent) => {
    e?.stopPropagation()
    const ext = isMermaid ? 'mmd' : 'html'
    const blob = new Blob([artifact.code], { type: isMermaid ? 'text/plain' : 'text/html' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${isMermaid ? 'diagram' : 'preview'}.${ext}`
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div
      onClick={onClick}
      className={cn(
        'flex items-center gap-3 px-3 py-2.5 my-2 rounded-xl cursor-pointer',
        'bg-[#1a1a1a] border border-white/[0.08] shadow-sm',
        'hover:bg-[#222] hover:border-primary/20 transition-all max-w-[300px]'
      )}
    >
      <div className={cn(
        'w-10 h-10 rounded-xl flex items-center justify-center shrink-0 border',
        isMermaid ? 'bg-primary/10 border-primary/20' : 
        artifact.type === 'text' ? 'bg-blue-500/10 border-blue-500/20' : 'bg-emerald-500/10 border-emerald-500/20'
      )}>
        {isMermaid ? <GitBranch size={17} className="text-primary" /> : 
         artifact.type === 'text' ? <FileText size={17} className="text-blue-400" /> : <Globe size={17} className="text-emerald-400" />}
      </div>
      <div className="flex-1 min-w-0">
        <div className="text-[12px] font-bold text-foreground truncate">{artifact.title || (isMermaid ? 'MERMAID Document' : artifact.type === 'text' ? 'TEXT Document' : 'HTML Document')}</div>
        <div className="text-[10px] text-muted-foreground font-semibold tracking-widest uppercase mt-0.5">
          {isMermaid ? 'Mermaid Diagram' : artifact.type === 'text' ? 'Text File' : 'HTML Preview'}
        </div>
      </div>
      <button
        onClick={(e) => { e.stopPropagation(); handleDownload(); }}
        className="w-8 h-8 rounded-lg flex items-center justify-center shrink-0 border bg-[#252525] border-white/[0.06] text-muted-foreground hover:bg-primary/10 hover:border-primary/20 hover:text-primary transition-all"
        title="Download"
      >
        <Download size={13} />
      </button>
    </div>
  )
}

// ─── ArtifactPanel ───────────────────────────────────────────────────────────

export function ArtifactPanel({
  artifact,
  onClose,
  panelRef,
}: {
  artifact: Artifact
  onClose: () => void
  panelRef?: React.RefObject<HTMLDivElement>
}) {
  const [tab, setTab] = useState<'preview' | 'code'>('preview')
  const isMermaid = artifact.type === 'mermaid'

  const handleDownload = () => {
    const ext = isMermaid ? 'mmd' : 'html'
    const blob = new Blob([artifact.code], { type: isMermaid ? 'text/plain' : 'text/html' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${isMermaid ? 'diagram' : 'preview'}.${ext}`
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="flex flex-col w-full h-full bg-[#0d0d0d]">

        {/* Header */}
        <div className="h-14 border-b border-border/60 flex items-center justify-between px-5 shrink-0 bg-[#111]">
          <div className="flex items-center gap-2">
            {artifact.type === 'text' ? (
              <FileText size={14} className="text-orange-400/80" />
            ) : isMermaid ? (
              <GitBranch size={13} className="text-primary" />
            ) : (
              <Globe size={13} className="text-emerald-400" />
            )}
            <span className="text-[12px] font-bold text-foreground truncate max-w-[200px]">
              {artifact.type === 'text' ? (artifact.title || 'Source Document') : isMermaid ? 'MERMAID Document' : 'HTML Document'}
            </span>
          </div>

          <div className="flex items-center gap-0.5">
            {artifact.type !== 'text' && (
              <>
                <button
                  onClick={() => setTab('preview')}
                  className={cn('w-7 h-7 rounded-md flex items-center justify-center transition-all',
                    tab === 'preview' ? 'bg-secondary text-foreground' : 'text-muted-foreground hover:text-foreground hover:bg-secondary/50')}
                  title="Preview"
                ><Eye size={13} /></button>

                <button
                  onClick={() => setTab('code')}
                  className={cn('w-7 h-7 rounded-md flex items-center justify-center transition-all',
                    tab === 'code' ? 'bg-secondary text-foreground' : 'text-muted-foreground hover:text-foreground hover:bg-secondary/50')}
                  title="Source Code"
                ><Code2 size={13} /></button>

                <div className="w-px h-4 bg-border/60 mx-1" />

                <button
                  onClick={handleDownload}
                  className="w-7 h-7 rounded-md flex items-center justify-center text-muted-foreground hover:text-foreground hover:bg-secondary/50 transition-all"
                  title="Download"
                ><Download size={13} /></button>
              </>
            )}
            <button
              onClick={onClose}
              className="w-7 h-7 rounded-md flex items-center justify-center text-muted-foreground hover:text-foreground hover:bg-secondary/50 transition-all ml-0.5"
              title="Close"
            ><X size={13} /></button>
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-hidden">
          {tab === 'preview' ? (
            <div className="w-full h-full bg-[#0d0d0d]">
              {isMermaid ? <MermaidPreview code={artifact.code} /> : 
               artifact.type === 'text' ? (
                 <div className="h-full p-6 overflow-auto bg-[#0a0a0a]">
                   <div className="text-[13.5px] text-foreground/70 whitespace-pre-wrap font-sans leading-[1.7]">
                     {artifact.chunk && artifact.code.includes(artifact.chunk) ? (
                       <>
                         {artifact.code.split(artifact.chunk)[0]}
                         <span className="bg-orange-500/20 text-orange-200/90 rounded px-1 py-0.5 shadow-[0_0_0_1px_rgba(249,115,22,0.2)]">
                           {artifact.chunk}
                         </span>
                         {artifact.code.split(artifact.chunk)[1]}
                       </>
                     ) : (
                       artifact.code
                     )}
                   </div>
                 </div>
               ) : <HtmlPreview code={artifact.code} />}
            </div>
          ) : (
            <div className="h-full overflow-auto">
              <SyntaxHighlighter
                language={isMermaid ? 'text' : artifact.type === 'text' ? 'markdown' : 'html'}
                style={vscDarkPlus}
                customStyle={{ margin: 0, minHeight: '100%', background: '#0a0a0a', fontSize: '12px', padding: '16px', lineHeight: '1.6' }}
                showLineNumbers
                lineNumberStyle={{ color: '#444', fontSize: '11px' }}
              >
                {artifact.code}
              </SyntaxHighlighter>
            </div>
          )}
        </div>
      </div>
  )
}
