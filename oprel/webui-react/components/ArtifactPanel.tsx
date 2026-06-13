'use client'

import { useState, useEffect, useRef } from 'react'
import { Eye, Code2, Download, X, GitBranch, Globe, AlertCircle, Loader2 } from 'lucide-react'
import { cn } from '@/services/utils'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism'

// ─── Types ───────────────────────────────────────────────────────────────────

export interface Artifact {
  type: 'mermaid' | 'html'
  code: string
}

// ─── Mermaid Preview ─────────────────────────────────────────────────────────

function MermaidPreview({ code }: { code: string }) {
  const [svg, setSvg] = useState<string>('')
  const [error, setError] = useState<string>('')
  const [loading, setLoading] = useState(true)
  const idRef = useRef(`mermaid-${Math.random().toString(36).slice(2)}`)

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    setError('')
    setSvg('')

    import('mermaid').then(({ default: mermaid }) => {
      mermaid.initialize({
        startOnLoad: false,
        theme: 'dark',
        securityLevel: 'loose',
        fontFamily: 'Inter, system-ui, sans-serif',
      })
      return mermaid.render(idRef.current, code)
    }).then(({ svg: rendered }) => {
      if (!cancelled) { setSvg(rendered); setLoading(false) }
    }).catch((err: Error) => {
      if (!cancelled) { setError(err.message || 'Failed to render diagram'); setLoading(false) }
    })

    return () => { cancelled = true }
  }, [code])

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
    // Outer scrollable area — fills the panel
    <div className="overflow-auto w-full h-full flex items-start justify-center p-8 bg-[#0d0d0d]">
      <div
        style={{ transform: 'scale(1.0)', transformOrigin: 'top center', minWidth: 'min-content' }}
        className="[&_svg]:block [&_svg]:mx-auto [&_svg]:max-w-none [&_svg]:h-auto"
        dangerouslySetInnerHTML={{ __html: svg }}
      />
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

  const handleDownload = (e: React.MouseEvent) => {
    e.stopPropagation()
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
        isMermaid ? 'bg-primary/10 border-primary/20' : 'bg-emerald-500/10 border-emerald-500/20'
      )}>
        {isMermaid ? <GitBranch size={17} className="text-primary" /> : <Globe size={17} className="text-emerald-400" />}
      </div>
      <div className="flex-1 min-w-0">
        <div className="text-[12px] font-bold text-foreground">{isMermaid ? 'MERMAID Document' : 'HTML Document'}</div>
        <div className="text-[10px] text-muted-foreground font-semibold tracking-widest uppercase mt-0.5">
          {isMermaid ? 'Mermaid Diagram' : 'HTML Preview'}
        </div>
      </div>
      <button
        onClick={handleDownload}
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
  panelRef: React.RefObject<HTMLDivElement>
}) {
  const [tab, setTab] = useState<'preview' | 'code'>('preview')
  const isMermaid = artifact.type === 'mermaid'

  // ── Drag-to-resize: manipulate DOM directly — zero React re-renders = zero lag
  const MIN_W = 300
  const handleMouseDown = (e: React.MouseEvent) => {
    e.preventDefault()
    const startX = e.clientX
    const startW = panelRef.current?.offsetWidth ?? 480

    const onMove = (ev: MouseEvent) => {
      if (!panelRef.current) return
      const delta = startX - ev.clientX   // drag left → wider
      const newW = Math.max(MIN_W, startW + delta)
      panelRef.current.style.width = `${newW}px`
    }
    const onUp = () => {
      document.removeEventListener('mousemove', onMove)
      document.removeEventListener('mouseup', onUp)
      document.body.style.cursor = ''
      document.body.style.userSelect = ''
    }
    document.body.style.cursor = 'col-resize'
    document.body.style.userSelect = 'none'
    document.addEventListener('mousemove', onMove)
    document.addEventListener('mouseup', onUp)
  }

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
    <div className="flex h-full">
      {/* ── Drag handle (left edge) ──────────────────────────── */}
      <div
        onMouseDown={handleMouseDown}
        className="w-1 shrink-0 cursor-col-resize hover:bg-primary/50 active:bg-primary transition-colors"
        title="Drag to resize"
      />

      {/* ── Panel body ───────────────────────────────────────── */}
      <div className="flex flex-col flex-1 min-w-0 bg-[#0d0d0d]">

        {/* Header */}
        <div className="h-12 border-b border-border/60 flex items-center justify-between px-3 shrink-0 bg-[#111]">
          <div className="flex items-center gap-2">
            {isMermaid ? <GitBranch size={13} className="text-primary" /> : <Globe size={13} className="text-emerald-400" />}
            <span className="text-[12px] font-bold text-foreground">
              {isMermaid ? 'MERMAID Document' : 'HTML Document'}
            </span>
          </div>

          <div className="flex items-center gap-0.5">
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
              {isMermaid ? <MermaidPreview code={artifact.code} /> : <HtmlPreview code={artifact.code} />}
            </div>
          ) : (
            <div className="h-full overflow-auto">
              <SyntaxHighlighter
                language={isMermaid ? 'text' : 'html'}
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
    </div>
  )
}
