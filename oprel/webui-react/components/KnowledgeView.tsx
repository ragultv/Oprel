"use client"

import { useState, useEffect, useRef, useMemo } from "react"
import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"
import remarkBreaks from "remark-breaks"
import { 
  FileText, 
  Upload, 
  Trash2, 
  Search, 
  CheckCircle2, 
  AlertCircle,
  Clock,
  HardDrive,
  RefreshCcw,
  FileUp,
  Database,
  Layers,
  ArrowRight,
  ExternalLink,
  Zap,
  Eye,
  X,
  PanelRightClose,
  PanelRightOpen,
  ChevronDown,
  Check,
  Copy,
  Plus,
  SlidersHorizontal,
  BookOpen,
  Maximize2,
  Minimize2,
  Download,
  FileSpreadsheet,
  Code2,
  FileCode,
  Table as TableIcon
} from "lucide-react"
import { API } from "@/services/api"
import { cn } from "@/services/utils"
import { useToast } from "@/components/ui/use-toast"

interface IndexedFile {
  id: string
  filename: string
  size_bytes: number
  indexed_at: string
  chunks: number
}

interface SearchChunk {
  text: string
  score?: number
  hybrid_score?: number
  metadata: {
    filename: string
    [key: string]: any
  }
}

function formatBytes(bytes: number) {
  if (!bytes || bytes === 0) return "0 B"
  const k = 1024
  const sizes = ["B", "KB", "MB", "GB"]
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + " " + sizes[i]
}

function formatDate(iso?: string) {
  if (!iso) return ""
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

type FileCategory = "markdown" | "csv" | "json" | "code" | "pdf" | "docx" | "text"

function getFileCategory(filename?: string | null): { category: FileCategory; label: string; ext: string } {
  if (!filename) return { category: "text", label: "Plain Text", ext: "txt" }
  const ext = filename.split(".").pop()?.toLowerCase() || ""
  if (["md", "markdown"].includes(ext)) return { category: "markdown", label: "Markdown Document", ext }
  if (["csv", "tsv"].includes(ext)) return { category: "csv", label: "Tabular Dataset", ext }
  if (["json"].includes(ext)) return { category: "json", label: "JSON Data", ext }
  if (["py", "ts", "tsx", "js", "jsx", "html", "css", "scss", "sh", "bash", "sql", "yaml", "yml", "rs", "go", "c", "cpp", "h", "java", "toml"].includes(ext)) {
    return { category: "code", label: `${ext.toUpperCase()} Source`, ext }
  }
  if (ext === "pdf") return { category: "pdf", label: "PDF Document", ext }
  if (["docx", "doc"].includes(ext)) return { category: "docx", label: "Word Document", ext }
  return { category: "text", label: "Plain Text", ext: ext || "txt" }
}

export function KnowledgeView() {
  const [activeTab, setActiveTab] = useState<"files" | "search">("files")
  const [files, setFiles] = useState<IndexedFile[]>([])
  const [searchResults, setSearchResults] = useState<SearchChunk[]>([])
  const [loading, setLoading] = useState(true)
  const [uploading, setUploading] = useState(false)
  const [searchQuery, setSearchQuery] = useState("")
  const [isSearching, setIsSearching] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const { toast } = useToast()

  // Sidebar state (matching Audio, Image, and OCR Studios)
  const [sidebarTab, setSidebarTab] = useState<"settings" | "index">("settings")
  const [isSidebarOpen, setIsSidebarOpen] = useState(true)
  const [daysFilter, setDaysFilter] = useState<number>(30)
  const [isDaysDropdownOpen, setIsDaysDropdownOpen] = useState(false)

  // Retrieval Tuning Controls (Settings Tab)
  const [topK, setTopK] = useState<number>(5)
  const [minScore, setMinScore] = useState<number>(0.20)
  const [hybridFusion, setHybridFusion] = useState<boolean>(true)

  // Preview & Modal states
  const [previewContent, setPreviewContent] = useState<string | null>(null)
  const [previewFilename, setPreviewFilename] = useState<string | null>(null)
  const [previewLoading, setPreviewLoading] = useState(false)
  const [previewTab, setPreviewTab] = useState<"formatted" | "raw">("formatted")
  const [previewSearch, setPreviewSearch] = useState("")
  const [previewCopied, setPreviewCopied] = useState(false)
  const [previewFullscreen, setPreviewFullscreen] = useState(false)
  const [fileToDelete, setFileToDelete] = useState<IndexedFile | null>(null)
  const [isDeleting, setIsDeleting] = useState(false)
  const [showDeleteAllModal, setShowDeleteAllModal] = useState(false)
  const [isDeletingAll, setIsDeletingAll] = useState(false)
  const [dragging, setDragging] = useState(false)

  const totalChunks = useMemo(() => {
    return files.reduce((acc, f) => acc + (f.chunks || 0), 0)
  }, [files])

  const totalSizeBytes = useMemo(() => {
    return files.reduce((acc, f) => acc + (f.size_bytes || 0), 0)
  }, [files])

  const filteredHistoryFiles = useMemo(() => {
    if (daysFilter === 0) return files
    const cutoff = Date.now() - daysFilter * 24 * 60 * 60 * 1000
    return files.filter((f) => {
      const time = new Date(f.indexed_at || Date.now()).getTime()
      return isNaN(time) || time >= cutoff
    })
  }, [files, daysFilter])

  const filteredInventoryFiles = useMemo(() => {
    if (!searchQuery.trim() || activeTab !== "files") return files
    return files.filter(f => 
      f.filename.toLowerCase().includes(searchQuery.toLowerCase())
    )
  }, [files, searchQuery, activeTab])

  const refreshFiles = async () => {
    try {
      setLoading(true)
      const data = await API.fetchDocuments()
      setFiles(data)
    } catch (err) {
      console.error("Failed to fetch documents:", err)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    refreshFiles()
  }, [])

  const handleConfirmDelete = async () => {
    if (!fileToDelete) return
    setIsDeleting(true)
    try {
      await API.deleteDocument(fileToDelete.filename)
      toast({
        title: "Document Deleted",
        description: `Successfully removed ${fileToDelete.filename} from the knowledge base.`,
      })
      setFileToDelete(null)
      refreshFiles()
    } catch (err: any) {
      toast({
        title: "Delete Failed",
        description: err.message || "Could not delete document.",
        variant: "destructive",
      })
    } finally {
      setIsDeleting(false)
    }
  }

  const handleConfirmDeleteAll = async () => {
    setIsDeletingAll(true)
    try {
      await API.deleteAllDocuments()
      toast({
        title: "Knowledge Base Cleared",
        description: "All documents and vectors have been successfully removed.",
      })
      setShowDeleteAllModal(false)
      refreshFiles()
    } catch (err: any) {
      toast({
        title: "Clear Failed",
        description: err.message || "Could not clear knowledge base.",
        variant: "destructive",
      })
    } finally {
      setIsDeletingAll(false)
    }
  }

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    await uploadFiles(Array.from(e.target.files || []))
  }

  const handleDrop = async (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    setDragging(false)
    if (uploading) return
    await uploadFiles(Array.from(e.dataTransfer.files || []))
  }

  const uploadFiles = async (selectedFiles: File[]) => {
    if (!selectedFiles.length) return

    setUploading(true)
    let successCount = 0
    let failCount = 0

    for (const file of selectedFiles) {
      const forbidden = [".mp3", ".mp4", ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"]
      const ext = file.name.slice(file.name.lastIndexOf(".")).toLowerCase()
      if (forbidden.includes(ext)) {
        console.warn(`Skipping forbidden file type: ${file.name}`)
        failCount++
        continue
      }

      try {
        await API.uploadDocument(file)
        successCount++
      } catch (err) {
        console.error(`Failed to upload ${file.name}:`, err)
        failCount++
      }
    }

    if (successCount > 0) {
      toast({
        title: "Indexing Complete",
        description: `Successfully indexed ${successCount} file(s) into the RAG environment.`,
      })
      refreshFiles()
    }

    if (failCount > 0) {
      toast({
        title: "Upload Failed",
        description: `Failed to index ${failCount} file(s). Check local logs.`,
        variant: "destructive",
      })
    }

    setUploading(false)
    if (fileInputRef.current) fileInputRef.current.value = ""
  }

  const handleKnowledgeSearch = async () => {
    if (!searchQuery.trim()) return
    setIsSearching(true)
    try {
      const results = await API.searchKnowledge(searchQuery, topK)
      setSearchResults(results)
    } catch (err: any) {
      toast({ title: "Search Failed", description: err.message, variant: "destructive" })
    } finally {
      setIsSearching(false)
    }
  }

  const handlePreviewDocument = async (filename: string) => {
    setPreviewFilename(filename)
    setPreviewLoading(true)
    setPreviewTab("formatted")
    setPreviewSearch("")
    setPreviewCopied(false)
    setPreviewFullscreen(false)
    try {
      const res = await API.previewDocument(filename)
      setPreviewContent(res.content)
    } catch (e: any) {
      toast({ title: "Preview Failed", description: e.message, variant: "destructive" })
      setPreviewFilename(null)
    } finally {
      setPreviewLoading(false)
    }
  }

  const handleCopyPreview = () => {
    if (!previewContent) return
    navigator.clipboard.writeText(previewContent)
    setPreviewCopied(true)
    toast({ title: "Copied to Clipboard", description: "Document content copied successfully." })
    setTimeout(() => setPreviewCopied(false), 2000)
  }

  return (
    <div
      className="flex flex-col h-full bg-[#0a0a0a] relative overflow-hidden text-foreground select-none font-sans"
      onDragOver={(e) => {
        e.preventDefault()
        setDragging(true)
      }}
      onDragLeave={(e) => {
        if (e.currentTarget === e.target) setDragging(false)
      }}
      onDrop={handleDrop}
    >
      {dragging && (
        <div
          className="absolute inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm border-2 border-dashed border-primary pointer-events-none"
          role="status"
        >
          <div className="flex flex-col items-center gap-3 p-8 rounded-3xl bg-[#141414] border border-white/10 shadow-2xl">
            <FileUp size={36} className="text-primary animate-bounce" />
            <p className="text-sm font-bold text-foreground">Drop documents to vectorize into Knowledge Base</p>
            <p className="text-xs text-muted-foreground">Supports PDF, Markdown, TXT, DOCX, Code, JSON, YAML</p>
          </div>
        </div>
      )}

      {/* ── Studio Header (Full width across top, identical to Audio/Image/OCR Studios) ── */}
      <header className="h-14 border-b border-white/10 flex items-center justify-between px-5 sm:px-8 shrink-0 bg-background/95 backdrop-blur-sm sticky top-0 z-20">
        <div className="flex items-center gap-3 min-w-0">
          <div className="w-8 h-8 rounded-xl bg-primary/10 text-primary border border-primary/20 flex items-center justify-center shrink-0">
            <Database size={18} />
          </div>
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <span className="text-sm font-bold text-foreground">Knowledge Studio</span>
              <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
            </div>
            <div className="text-[10px] text-muted-foreground font-medium flex items-center gap-1.5">
              <span>Local RAG Vector Store &amp; Semantic Retrieval</span>
            </div>
          </div>
        </div>

        {/* Right Actions & Sidebar Toggle */}
        <div className="flex items-center gap-2 sm:gap-3">
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={uploading}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-primary hover:bg-primary/90 text-primary-foreground text-xs font-semibold transition-all shadow-md shadow-primary/20 disabled:opacity-50"
            title="Add documents to knowledge base"
          >
            <Plus size={14} />
            <span className="hidden sm:inline">{uploading ? "Indexing..." : "Ingest Document"}</span>
          </button>
          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept=".pdf,.docx,.txt,.md,.py,.js,.ts,.tsx,.json,.yaml,.yml"
            className="hidden"
            onChange={handleUpload}
          />

          <button
            onClick={refreshFiles}
            disabled={loading}
            className="p-2 rounded-lg hover:bg-white/10 text-muted-foreground hover:text-foreground transition-all"
            title="Refresh knowledge store"
          >
            <RefreshCcw size={15} className={cn(loading && "animate-spin text-primary")} />
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
        {/* Left Workspace: Document Inventory & Neural Search Playground */}
        <div className="flex-1 h-full min-w-0 flex flex-col p-4 md:p-6 overflow-hidden relative space-y-4">
          
          {/* Top Control Bar: View Switcher Tabs & Search Filter */}
          <div className="flex flex-col sm:flex-row gap-3 items-stretch sm:items-center justify-between shrink-0">
            {/* View Switcher: Document Inventory | Neural Search */}
            <div className="inline-flex items-center p-1 rounded-2xl bg-black/40 border border-white/10 gap-1 self-start">
              <button
                onClick={() => setActiveTab("files")}
                className={cn(
                  "px-4 py-2 rounded-xl text-xs font-semibold transition-all flex items-center gap-2",
                  activeTab === "files"
                    ? "bg-primary text-white shadow-md shadow-primary/20 font-bold"
                    : "text-muted-foreground hover:text-foreground"
                )}
              >
                <FileText size={14} />
                <span>Document Inventory</span>
                <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-white/10 font-mono">
                  {files.length}
                </span>
              </button>

              <button
                onClick={() => setActiveTab("search")}
                className={cn(
                  "px-4 py-2 rounded-xl text-xs font-semibold transition-all flex items-center gap-2",
                  activeTab === "search"
                    ? "bg-primary text-white shadow-md shadow-primary/20 font-bold"
                    : "text-muted-foreground hover:text-foreground"
                )}
              >
                <Search size={14} />
                <span>Neural Retrieval</span>
              </button>
            </div>

            {/* Search Input Bar */}
            <div className="relative flex-1 max-w-md">
              <Search size={14} className="absolute left-3.5 top-1/2 -translate-y-1/2 text-muted-foreground" />
              <input
                type="text"
                placeholder={activeTab === "files" ? "Filter documents by name..." : "Query knowledge base vectors..."}
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && activeTab === "search" && handleKnowledgeSearch()}
                className="w-full bg-black/30 border border-white/10 rounded-2xl py-2 pl-9 pr-10 text-xs text-foreground placeholder:text-muted-foreground/60 focus:outline-none focus:border-primary/50 transition-all"
              />
              {activeTab === "search" && searchQuery && (
                <button
                  onClick={handleKnowledgeSearch}
                  disabled={isSearching}
                  className="absolute right-2 top-1/2 -translate-y-1/2 p-1.5 text-primary hover:bg-primary/10 rounded-lg transition-all"
                  title="Run vector search"
                >
                  {isSearching ? <RefreshCcw size={13} className="animate-spin" /> : <ArrowRight size={13} />}
                </button>
              )}
            </div>
          </div>

          {/* Main Body View */}
          <div className="flex-1 overflow-y-auto min-h-0 space-y-4">
            {activeTab === "files" ? (
              /* ── Document Inventory View ── */
              <div className="space-y-4">
                {/* Drag and Drop Card */}
                <div
                  onClick={() => fileInputRef.current?.click()}
                  className="p-5 rounded-2xl border border-dashed border-white/10 hover:border-primary/40 bg-black/20 hover:bg-white/[0.02] transition-all cursor-pointer flex flex-col sm:flex-row items-center justify-between gap-4 text-center sm:text-left"
                >
                  <div className="flex items-center gap-3.5">
                    <div className="w-10 h-10 rounded-xl bg-primary/10 text-primary border border-primary/20 flex items-center justify-center shrink-0">
                      <FileUp size={18} />
                    </div>
                    <div>
                      <div className="text-xs font-bold text-foreground">Drag &amp; drop files here to vectorize</div>
                      <div className="text-[11px] text-muted-foreground mt-0.5">
                        PDF, Markdown, Code, DOCX, TXT, JSON, YAML automatically chunked and indexed.
                      </div>
                    </div>
                  </div>
                  <span className="px-3 py-1.5 rounded-xl bg-white/5 border border-white/10 text-xs font-semibold text-foreground hover:bg-white/10 shrink-0">
                    Browse Files
                  </span>
                </div>

                {/* Documents Table */}
                <div className="rounded-2xl border border-white/10 bg-[#141414] overflow-hidden shadow-2xl">
                  <table className="w-full text-left text-xs border-collapse">
                    <thead>
                      <tr className="border-b border-white/10 bg-black/40 text-[11px] font-semibold text-muted-foreground uppercase tracking-wider">
                        <th className="px-5 py-3.5">Document Name</th>
                        <th className="px-5 py-3.5">Chunks</th>
                        <th className="px-5 py-3.5 hidden sm:table-cell">File Size</th>
                        <th className="px-5 py-3.5 hidden md:table-cell">Indexed Date</th>
                        <th className="px-5 py-3.5 text-right w-20">Actions</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-white/5">
                      {loading ? (
                        Array.from({ length: 4 }).map((_, i) => (
                          <tr key={i} className="animate-pulse">
                            <td className="px-5 py-4"><div className="h-4 bg-white/5 rounded w-48" /></td>
                            <td className="px-5 py-4"><div className="h-4 bg-white/5 rounded w-16" /></td>
                            <td className="px-5 py-4 hidden sm:table-cell"><div className="h-4 bg-white/5 rounded w-16" /></td>
                            <td className="px-5 py-4 hidden md:table-cell"><div className="h-4 bg-white/5 rounded w-24" /></td>
                            <td className="px-5 py-4 text-right"></td>
                          </tr>
                        ))
                      ) : filteredInventoryFiles.length === 0 ? (
                        <tr>
                          <td colSpan={5} className="px-5 py-16 text-center">
                            <div className="flex flex-col items-center gap-2 text-muted-foreground opacity-50">
                              <BookOpen size={36} />
                              <div className="text-xs font-semibold">
                                {searchQuery ? "No matching documents found" : "No documents indexed yet"}
                              </div>
                              <p className="text-[11px] max-w-xs">
                                Upload documents to populate your local vector knowledge store.
                              </p>
                            </div>
                          </td>
                        </tr>
                      ) : (
                        filteredInventoryFiles.map((file) => (
                          <tr key={file.id} className="group hover:bg-white/[0.03] transition-colors">
                            <td className="px-5 py-3.5">
                              <div className="flex items-center gap-3">
                                <FileText size={15} className="text-muted-foreground group-hover:text-primary transition-colors shrink-0" />
                                <span className="font-semibold text-foreground truncate max-w-[280px]">
                                  {file.filename}
                                </span>
                              </div>
                            </td>
                            <td className="px-5 py-3.5">
                              <span className="px-2 py-0.5 rounded-full bg-primary/10 text-[10px] font-mono font-bold text-primary border border-primary/20">
                                {file.chunks} Chunks
                              </span>
                            </td>
                            <td className="px-5 py-3.5 hidden sm:table-cell text-muted-foreground font-mono">
                              {formatBytes(file.size_bytes)}
                            </td>
                            <td className="px-5 py-3.5 hidden md:table-cell text-muted-foreground font-mono">
                              {formatDate(file.indexed_at)}
                            </td>
                            <td className="px-5 py-3.5 text-right">
                              <div className="flex items-center justify-end gap-1 text-muted-foreground">
                                <button
                                  onClick={() => handlePreviewDocument(file.filename)}
                                  className="p-1.5 rounded-lg hover:bg-white/10 hover:text-foreground transition-all"
                                  title="Preview Document"
                                >
                                  {previewLoading && previewFilename === file.filename ? (
                                    <RefreshCcw size={13} className="animate-spin text-primary" />
                                  ) : (
                                    <Eye size={13} />
                                  )}
                                </button>
                                <button
                                  onClick={() => setFileToDelete(file)}
                                  className="p-1.5 rounded-lg hover:bg-rose-500/20 hover:text-rose-400 transition-all"
                                  title="Delete Document"
                                >
                                  <Trash2 size={13} />
                                </button>
                              </div>
                            </td>
                          </tr>
                        ))
                      )}
                    </tbody>
                  </table>
                </div>
              </div>
            ) : (
              /* ── Neural Search Playground View ── */
              <div className="space-y-4">
                {isSearching ? (
                  <div className="flex flex-col items-center justify-center py-20 gap-3 text-center">
                    <RefreshCcw size={26} className="text-primary animate-spin" />
                    <p className="text-xs font-bold text-foreground">Retrieving most relevant chunks...</p>
                    <p className="text-[11px] text-muted-foreground">Computing dense embeddings &amp; BM25 hybrid ranking.</p>
                  </div>
                ) : searchResults.length === 0 ? (
                  <div className="flex flex-col items-center justify-center py-20 border border-dashed border-white/10 rounded-3xl bg-black/20 text-center space-y-3">
                    <Search size={32} className="text-muted-foreground/30" />
                    <div className="space-y-1">
                      <p className="text-xs font-bold text-foreground">Semantic Search Playground</p>
                      <p className="text-[11px] text-muted-foreground max-w-sm">
                        Enter any question, phrase, or keyword above to preview which document chunks your LLM will retrieve during chat.
                      </p>
                    </div>
                  </div>
                ) : (
                  <div className="space-y-3">
                    <div className="flex items-center justify-between text-xs text-muted-foreground px-1">
                      <span>Found {searchResults.length} matching chunk{searchResults.length > 1 ? "s" : ""}</span>
                      <span className="font-mono text-[10px]">Top-{topK} results</span>
                    </div>

                    {searchResults.map((result, i) => {
                      const rawScore = result.score || (result as any).hybrid_score || 0
                      const displayScore = rawScore > 1 ? "1.00" : rawScore.toFixed(3)
                      const filename = result.metadata?.filename || "Unknown Source"

                      return (
                        <div
                          key={i}
                          className="p-4 rounded-2xl bg-[#141414] border border-white/10 hover:border-white/20 transition-all space-y-3 shadow-md"
                        >
                          <div className="flex items-center justify-between">
                            <div className="flex items-center gap-2 min-w-0">
                              <span className="px-2 py-0.5 rounded-full bg-primary/10 text-primary text-[10px] font-bold uppercase tracking-wider font-mono">
                                Chunk {i + 1}
                              </span>
                              <span className="text-xs font-semibold text-foreground truncate flex items-center gap-1.5">
                                <FileText size={12} className="text-muted-foreground shrink-0" />
                                {filename}
                              </span>
                            </div>

                            <div className="text-[10px] font-mono text-muted-foreground shrink-0">
                              Similarity: <span className="text-primary font-bold">{displayScore}</span>
                            </div>
                          </div>

                          <div className="p-3.5 rounded-xl bg-black/40 border border-white/5 text-xs text-foreground/90 font-mono leading-relaxed whitespace-pre-wrap select-text">
                            &ldquo;{result.text}&rdquo;
                          </div>

                          <div className="flex justify-end pt-1">
                            <button
                              onClick={() => handlePreviewDocument(filename)}
                              className="inline-flex items-center gap-1 text-[11px] font-semibold text-muted-foreground hover:text-primary transition-colors"
                            >
                              <span>View document context</span>
                              <ExternalLink size={11} />
                            </button>
                          </div>
                        </div>
                      )
                    })}
                  </div>
                )}
              </div>
            )}
          </div>
        </div>

        {/* ── Right Sidebar: Docked Top-to-Bottom UNDER THE HEADER ── */}
        {isSidebarOpen && (
          <aside className="w-full lg:w-[340px] xl:w-[380px] shrink-0 h-full border-l border-white/10 bg-[#121212] flex flex-col overflow-hidden animate-in slide-in-from-right-4 duration-200">
            {/* Top Tabs Switcher: Settings | Index */}
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
                  onClick={() => setSidebarTab("index")}
                  className={cn(
                    "flex-1 py-1.5 rounded-full text-xs font-semibold text-center transition-all",
                    sidebarTab === "index"
                      ? "bg-primary text-white shadow-md shadow-primary/20 font-bold"
                      : "text-muted-foreground hover:text-foreground"
                  )}
                >
                  Index {files.length > 0 && `(${files.length})`}
                </button>
              </div>
            </div>

            {/* Scrollable Sidebar Body */}
            <div className="flex-1 overflow-y-auto p-4 space-y-4">
              {sidebarTab === "settings" ? (
                /* ── Settings View ── */
                <div className="space-y-4 animate-in fade-in duration-150">
                  {/* Embedding Model Selection */}
                  <div className="space-y-1.5">
                    <label className="text-xs font-medium text-muted-foreground block">
                      Embedding Model
                    </label>
                    <div className="w-full flex items-center justify-between px-3.5 py-2.5 rounded-2xl bg-black/30 border border-white/10 text-xs font-semibold text-foreground shadow-inner">
                      <div className="flex items-center gap-2 truncate">
                        <span className="truncate">all-MiniLM-L6-v2 (Local)</span>
                        <span className="px-2 py-0.5 rounded-full text-[10px] font-mono bg-white/10 text-muted-foreground font-semibold">
                          384-dim
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* Model Highlight Card (matching Audio/Image/OCR Studios) */}
                  <div className="p-3.5 rounded-2xl bg-gradient-to-br from-white/[0.06] to-white/[0.02] border border-white/10 space-y-1">
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-bold text-foreground truncate">
                        Hybrid Vector + BM25
                      </span>
                      <span className="text-[10px] font-mono text-primary font-bold shrink-0">Active Engine</span>
                    </div>
                    <p className="text-[11px] text-muted-foreground leading-relaxed">
                      Dual-retrieval pipeline combining dense semantic embeddings with sparse BM25 lexical ranking for high-precision document context retrieval.
                    </p>
                  </div>

                  {/* Retrieval Controls Section Header */}
                  <div className="space-y-3 pt-2">
                    <label className="text-xs font-semibold text-muted-foreground block uppercase tracking-wider text-[10px]">
                      Retrieval Controls
                    </label>

                    {/* Top-K Chunks Slider */}
                    <div className="flex items-center justify-between gap-3 p-2.5 rounded-2xl bg-black/20 border border-white/5">
                      <span className="text-xs font-medium text-muted-foreground">Top-K Chunks</span>
                      <div className="flex items-center gap-3 flex-1 max-w-[160px]">
                        <input
                          type="range"
                          min="1"
                          max="20"
                          step="1"
                          value={topK}
                          onChange={(e) => setTopK(parseInt(e.target.value, 10))}
                          className="w-full h-1.5 bg-white/10 rounded-lg appearance-none cursor-pointer accent-primary"
                        />
                        <span className="text-xs font-mono font-semibold text-foreground min-w-[28px] text-right">
                          {topK}
                        </span>
                      </div>
                    </div>

                    {/* Min Similarity Cutoff Slider */}
                    <div className="flex items-center justify-between gap-3 p-2.5 rounded-2xl bg-black/20 border border-white/5">
                      <span className="text-xs font-medium text-muted-foreground">Min Similarity</span>
                      <div className="flex items-center gap-3 flex-1 max-w-[160px]">
                        <input
                          type="range"
                          min="0.0"
                          max="0.8"
                          step="0.05"
                          value={minScore}
                          onChange={(e) => setMinScore(parseFloat(e.target.value))}
                          className="w-full h-1.5 bg-white/10 rounded-lg appearance-none cursor-pointer accent-primary"
                        />
                        <span className="text-xs font-mono font-semibold text-foreground min-w-[28px] text-right">
                          {minScore.toFixed(2)}
                        </span>
                      </div>
                    </div>

                    {/* BM25 Keyword Fusion Toggle */}
                    <div className="flex items-center justify-between gap-2 p-2.5 rounded-2xl bg-black/20 border border-white/5">
                      <span className="text-xs font-medium text-muted-foreground truncate">BM25 Keyword Fusion</span>
                      <div className="inline-flex items-center p-0.5 rounded-xl bg-black/50 border border-white/10 shrink-0">
                        <button
                          type="button"
                          onClick={() => setHybridFusion(false)}
                          className={cn(
                            "px-2.5 py-1 rounded-lg text-[11px] font-semibold transition-all",
                            !hybridFusion ? "bg-white/20 text-white font-bold shadow-sm" : "text-muted-foreground hover:text-foreground"
                          )}
                        >
                          Off
                        </button>
                        <button
                          type="button"
                          onClick={() => setHybridFusion(true)}
                          className={cn(
                            "px-2.5 py-1 rounded-lg text-[11px] font-semibold transition-all",
                            hybridFusion ? "bg-primary text-white font-bold shadow-sm" : "text-muted-foreground hover:text-foreground"
                          )}
                        >
                          On
                        </button>
                      </div>
                    </div>

                    {/* Chunking Parameters Display */}
                    <div className="flex items-center justify-between gap-2 p-2.5 rounded-2xl bg-black/20 border border-white/5">
                      <span className="text-xs font-medium text-muted-foreground truncate">Chunk Window</span>
                      <span className="text-[10px] font-mono text-muted-foreground px-2 py-0.5 rounded-lg bg-white/5 border border-white/10 font-semibold">
                        512 tokens &bull; 64 overlap
                      </span>
                    </div>
                  </div>
                </div>
              ) : (
                /* ── Index / Stats View ── */
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
                      onClick={refreshFiles}
                      className="p-2 rounded-full bg-white/5 hover:bg-white/10 text-muted-foreground hover:text-foreground transition-all"
                      title="Refresh inventory"
                    >
                      <RefreshCcw size={13} className={cn(loading && "animate-spin text-primary")} />
                    </button>
                  </div>

                  {/* Add New File Button */}
                  <button
                    type="button"
                    onClick={() => fileInputRef.current?.click()}
                    className="w-full flex items-center justify-center gap-2 py-2.5 px-4 rounded-2xl bg-primary text-white text-xs font-bold hover:bg-primary/90 transition-all shadow-md shadow-primary/20"
                  >
                    <Plus size={15} />
                    <span>Ingest New Document</span>
                  </button>

                  {/* Storage & Chunks Metrics Box */}
                  <div className="grid grid-cols-2 gap-2 p-3 rounded-2xl bg-black/30 border border-white/5 text-center">
                    <div>
                      <div className="text-[10px] text-muted-foreground font-mono uppercase">Total Vectors</div>
                      <div className="text-sm font-bold text-foreground mt-0.5">{totalChunks}</div>
                    </div>
                    <div>
                      <div className="text-[10px] text-muted-foreground font-mono uppercase">Total Storage</div>
                      <div className="text-sm font-bold text-foreground mt-0.5">{formatBytes(totalSizeBytes)}</div>
                    </div>
                  </div>

                  {/* Header Actions */}
                  <div className="flex items-center justify-between text-[11px] text-muted-foreground font-semibold pt-1">
                    <span>
                      {filteredHistoryFiles.length > 0
                        ? `${filteredHistoryFiles.length} document${filteredHistoryFiles.length > 1 ? "s" : ""}`
                        : "No files in range"}
                    </span>
                    {files.length > 0 && (
                      <button
                        type="button"
                        onClick={() => setShowDeleteAllModal(true)}
                        className="text-muted-foreground hover:text-rose-400 transition-colors"
                      >
                        Clear All
                      </button>
                    )}
                  </div>

                  {/* Files List in History */}
                  {filteredHistoryFiles.length === 0 ? (
                    <div className="text-center py-12 text-xs text-muted-foreground space-y-2">
                      <BookOpen size={26} className="mx-auto opacity-40" />
                      <p>No documents found in this range.</p>
                    </div>
                  ) : (
                    <div className="space-y-2.5">
                      {filteredHistoryFiles.map((file) => (
                        <div
                          key={file.id}
                          className="p-3.5 rounded-2xl bg-black/30 border border-white/5 hover:border-white/15 transition-all space-y-2 group"
                        >
                          <div className="flex items-center justify-between text-xs">
                            <div className="flex items-center gap-2 truncate min-w-0">
                              <div className="w-5 h-5 rounded-md bg-gradient-to-tr from-sky-500 to-indigo-500 text-white flex items-center justify-center text-[9px] font-bold shrink-0">
                                {file.filename.charAt(0).toUpperCase()}
                              </div>
                              <span className="font-semibold text-foreground truncate">{file.filename}</span>
                            </div>
                            <span className="text-[10px] font-mono text-muted-foreground shrink-0 ml-2">
                              {formatDate(file.indexed_at)}
                            </span>
                          </div>

                          <div className="flex items-center justify-between pt-1 border-t border-white/5">
                            <span className="text-[10px] px-2 py-0.5 rounded-full bg-white/5 border border-white/10 text-muted-foreground font-mono">
                              {file.chunks} chunks &bull; {formatBytes(file.size_bytes)}
                            </span>

                            <div className="flex items-center gap-1.5 text-muted-foreground">
                              <button
                                type="button"
                                onClick={() => handlePreviewDocument(file.filename)}
                                className="p-1.5 rounded-lg hover:bg-white/10 hover:text-foreground transition-all"
                                title="Preview document"
                              >
                                <Eye size={13} />
                              </button>
                              <button
                                type="button"
                                onClick={() => setFileToDelete(file)}
                                className="p-1.5 rounded-lg hover:bg-rose-500/20 hover:text-rose-400 transition-all"
                                title="Delete document"
                              >
                                <Trash2 size={13} />
                              </button>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>
          </aside>
        )}
      </div>

      {/* ── Document Preview Modal ── */}
      {(previewContent !== null || previewLoading) && previewFilename && (() => {
        const fileMeta = getFileCategory(previewFilename)
        const rawContent = previewContent || ""
        const wordCount = rawContent.trim() ? rawContent.trim().split(/\s+/).filter(Boolean).length : 0
        const charCount = rawContent.length
        const allLines = rawContent.split(/\r?\n/)
        const lineCount = rawContent ? allLines.length : 0
        const estReadingMinutes = Math.max(1, Math.ceil(wordCount / 200))
        const searchLower = previewSearch.trim().toLowerCase()

        // Filter lines for code / raw / text
        const filteredLines = searchLower
          ? allLines
              .map((line, idx) => ({ line, idx: idx + 1 }))
              .filter((item) => item.line.toLowerCase().includes(searchLower))
          : allLines.map((line, idx) => ({ line, idx: idx + 1 }))

        // Parse CSV if applicable
        const parsedCsv = (() => {
          if (fileMeta.category !== "csv" || !previewContent) return null
          const rowsList = previewContent.trim().split(/\r?\n/).filter((l) => l.trim().length > 0)
          if (rowsList.length === 0) return null
          const sep = previewFilename.toLowerCase().endsWith(".tsv") ? "\t" : ","
          const parseRow = (line: string) => {
            const row: string[] = []
            let cur = ""
            let inQuotes = false
            for (let i = 0; i < line.length; i++) {
              const c = line[i]
              if (c === '"') inQuotes = !inQuotes
              else if (c === sep && !inQuotes) {
                row.push(cur.trim())
                cur = ""
              } else {
                cur += c
              }
            }
            row.push(cur.trim())
            return row
          }
          const headers = parseRow(rowsList[0])
          let rows = rowsList.slice(1).map(parseRow)
          if (searchLower) {
            rows = rows.filter((r) => r.some((cell) => cell.toLowerCase().includes(searchLower)))
          }
          return { headers, rows }
        })()

        const canToggleFormatted = ["markdown", "csv", "json", "pdf"].includes(fileMeta.category)

        return (
          <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-md flex items-center justify-center p-2 sm:p-4 md:p-6 animate-in fade-in duration-200">
            <div
              className={cn(
                "flex flex-col bg-[#141414] border border-white/15 shadow-2xl overflow-hidden transition-all duration-200",
                previewFullscreen
                  ? "w-full h-full rounded-none"
                  : "w-full max-w-5xl max-h-[90vh] rounded-3xl"
              )}
            >
              {/* Modal Header */}
              <div className="flex items-center justify-between px-5 py-3.5 border-b border-white/10 bg-black/40 gap-3 shrink-0">
                <div className="flex items-center gap-3 min-w-0">
                  <div
                    className={cn(
                      "w-9 h-9 rounded-xl flex items-center justify-center shrink-0 border",
                      fileMeta.category === "pdf" && "bg-rose-500/10 border-rose-500/20 text-rose-400",
                      fileMeta.category === "docx" && "bg-sky-500/10 border-sky-500/20 text-sky-400",
                      fileMeta.category === "markdown" && "bg-purple-500/10 border-purple-500/20 text-purple-400",
                      fileMeta.category === "csv" && "bg-emerald-500/10 border-emerald-500/20 text-emerald-400",
                      fileMeta.category === "json" && "bg-amber-500/10 border-amber-500/20 text-amber-400",
                      fileMeta.category === "code" && "bg-primary/10 border-primary/20 text-primary",
                      fileMeta.category === "text" && "bg-white/5 border-white/10 text-muted-foreground"
                    )}
                  >
                    {fileMeta.category === "pdf" && <FileText size={18} />}
                    {fileMeta.category === "docx" && <FileText size={18} />}
                    {fileMeta.category === "markdown" && <FileCode size={18} />}
                    {fileMeta.category === "csv" && <FileSpreadsheet size={18} />}
                    {fileMeta.category === "json" && <FileCode size={18} />}
                    {fileMeta.category === "code" && <Code2 size={18} />}
                    {fileMeta.category === "text" && <FileText size={18} />}
                  </div>
                  <div className="min-w-0">
                    <div className="flex items-center gap-2">
                      <h3 className="font-bold text-sm text-foreground truncate max-w-[200px] sm:max-w-xs md:max-w-md">
                        {previewFilename}
                      </h3>
                      <span
                        className={cn(
                          "px-2 py-0.5 rounded-full text-[10px] font-semibold border hidden sm:inline-block",
                          fileMeta.category === "pdf" && "bg-rose-500/10 border-rose-500/20 text-rose-400",
                          fileMeta.category === "docx" && "bg-sky-500/10 border-sky-500/20 text-sky-400",
                          fileMeta.category === "markdown" && "bg-purple-500/10 border-purple-500/20 text-purple-400",
                          fileMeta.category === "csv" && "bg-emerald-500/10 border-emerald-500/20 text-emerald-400",
                          fileMeta.category === "json" && "bg-amber-500/10 border-amber-500/20 text-amber-400",
                          fileMeta.category === "code" && "bg-primary/10 border-primary/20 text-primary",
                          fileMeta.category === "text" && "bg-white/5 border-white/10 text-muted-foreground"
                        )}
                      >
                        {fileMeta.label}
                      </span>
                    </div>
                  </div>
                </div>

                {/* Center Formatted / Raw Mode Pill */}
                {canToggleFormatted && (
                  <div className="hidden md:inline-flex items-center p-0.5 rounded-full bg-black/40 border border-white/10 shadow-inner">
                    <button
                      type="button"
                      onClick={() => setPreviewTab("formatted")}
                      className={cn(
                        "px-3 py-1 rounded-full text-xs font-semibold transition-all",
                        previewTab === "formatted"
                          ? "bg-primary text-white shadow-md shadow-primary/20"
                          : "text-muted-foreground hover:text-foreground"
                      )}
                    >
                      {fileMeta.category === "csv"
                        ? "Data Grid"
                        : fileMeta.category === "markdown"
                        ? "Formatted Document"
                        : fileMeta.category === "pdf"
                        ? "Visual PDF"
                        : "Pretty Print"}
                    </button>
                    <button
                      type="button"
                      onClick={() => setPreviewTab("raw")}
                      className={cn(
                        "px-3 py-1 rounded-full text-xs font-semibold transition-all",
                        previewTab === "raw"
                          ? "bg-primary text-white shadow-md shadow-primary/20"
                          : "text-muted-foreground hover:text-foreground"
                      )}
                    >
                      {fileMeta.category === "pdf" ? "Extracted Text" : "Raw Source"}
                    </button>
                  </div>
                )}

                {/* Right Tool Actions */}
                <div className="flex items-center gap-1.5 sm:gap-2">
                  {/* Inline Search Box */}
                  <div className="relative items-center hidden sm:flex">
                    <Search size={13} className="absolute left-2.5 text-muted-foreground pointer-events-none" />
                    <input
                      type="text"
                      value={previewSearch}
                      onChange={(e) => setPreviewSearch(e.target.value)}
                      placeholder="Find in file..."
                      className="pl-7 pr-7 py-1.5 rounded-xl bg-black/50 border border-white/10 text-xs text-foreground placeholder:text-muted-foreground/60 focus:outline-none focus:border-primary/50 w-28 md:w-40 transition-all"
                    />
                    {previewSearch && (
                      <button
                        type="button"
                        onClick={() => setPreviewSearch("")}
                        className="absolute right-2 text-muted-foreground hover:text-foreground"
                      >
                        <X size={12} />
                      </button>
                    )}
                  </div>

                  {/* Copy Text Button */}
                  <button
                    type="button"
                    onClick={handleCopyPreview}
                    className="p-2 rounded-xl text-muted-foreground hover:bg-white/10 hover:text-foreground transition-all"
                    title="Copy document text"
                  >
                    {previewCopied ? <Check size={15} className="text-emerald-400" /> : <Copy size={15} />}
                  </button>

                  {/* Download Raw File */}
                  <a
                    href={API.getDocumentRawUrl(previewFilename)}
                    download={previewFilename}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="p-2 rounded-xl text-muted-foreground hover:bg-white/10 hover:text-foreground transition-all"
                    title={`Download original ${previewFilename}`}
                  >
                    <Download size={15} />
                  </a>

                  {/* Toggle Fullscreen */}
                  <button
                    type="button"
                    onClick={() => setPreviewFullscreen(!previewFullscreen)}
                    className="p-2 rounded-xl text-muted-foreground hover:bg-white/10 hover:text-foreground transition-all hidden sm:inline-flex"
                    title={previewFullscreen ? "Exit full screen" : "Full screen"}
                  >
                    {previewFullscreen ? <Minimize2 size={15} /> : <Maximize2 size={15} />}
                  </button>

                  {/* Close Modal */}
                  <button
                    type="button"
                    onClick={() => {
                      setPreviewContent(null)
                      setPreviewFilename(null)
                    }}
                    className="p-2 rounded-xl text-muted-foreground hover:bg-white/10 hover:text-foreground transition-all"
                    title="Close preview"
                  >
                    <X size={16} />
                  </button>
                </div>
              </div>

              {/* Stats Bar */}
              <div className="flex items-center justify-between px-5 py-2 border-b border-white/5 bg-black/20 text-[11px] text-muted-foreground shrink-0 font-mono">
                <div className="flex items-center gap-3">
                  <span>{wordCount.toLocaleString()} words</span>
                  <span>&bull;</span>
                  <span>{charCount.toLocaleString()} characters</span>
                  <span>&bull;</span>
                  <span>{lineCount.toLocaleString()} lines</span>
                  <span>&bull;</span>
                  <span>{estReadingMinutes} min read</span>
                </div>
                {previewSearch && (
                  <div className="text-primary font-semibold flex items-center gap-1.5">
                    <span>Matches:</span>
                    <span className="px-1.5 py-0.5 rounded bg-primary/20 text-foreground font-mono">
                      {fileMeta.category === "csv" && parsedCsv
                        ? `${parsedCsv.rows.length} rows`
                        : `${filteredLines.length} lines`}
                    </span>
                  </div>
                )}
              </div>

              {/* Modal Body Container */}
              <div className="flex-1 overflow-y-auto p-4 sm:p-6 bg-black/30">
                {previewLoading ? (
                  <div className="flex flex-col items-center justify-center py-28 gap-3">
                    <RefreshCcw size={28} className="animate-spin text-primary" />
                    <p className="text-xs text-muted-foreground">Extracting document content from knowledge base...</p>
                  </div>
                ) : !rawContent.trim() ? (
                  <div className="flex flex-col items-center justify-center py-28 gap-2 text-center text-muted-foreground">
                    <FileText size={32} className="opacity-40 mb-2" />
                    <p className="text-sm font-semibold text-foreground">Document is empty</p>
                    <p className="text-xs">No extractable text content found for this file.</p>
                  </div>
                ) : (
                  <>
                    {/* PDF or DOCX banner */}
                    {(fileMeta.category === "pdf" || fileMeta.category === "docx") && (
                      <div className="flex flex-wrap items-center justify-between p-3.5 mb-5 rounded-2xl bg-white/5 border border-white/10 gap-3">
                        <div className="flex items-center gap-2.5">
                          <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                          <span className="text-xs text-foreground font-medium">
                            Displaying text extracted by local ingestion parser
                          </span>
                        </div>
                        <a
                          href={API.getDocumentRawUrl(previewFilename)}
                          download={previewFilename}
                          className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-primary text-white text-xs font-semibold hover:bg-primary/90 transition-all shadow-md shadow-primary/20"
                        >
                          <Download size={13} />
                          <span>Download Original ({fileMeta.ext.toUpperCase()})</span>
                        </a>
                      </div>
                    )}

                    {/* Format-Specific Viewers */}
                    {fileMeta.category === "markdown" && previewTab === "formatted" ? (
                      <div className="oprel-markdown max-w-4xl mx-auto p-6 md:p-8 bg-black/25 rounded-2xl border border-white/5 shadow-inner">
                        <ReactMarkdown remarkPlugins={[remarkGfm, remarkBreaks]}>
                          {rawContent}
                        </ReactMarkdown>
                      </div>
                    ) : fileMeta.category === "csv" && previewTab === "formatted" && parsedCsv ? (
                      <div className="w-full overflow-x-auto rounded-2xl border border-white/10 bg-black/25 shadow-inner">
                        <table className="w-full text-left text-xs border-collapse">
                          <thead className="bg-[#181818] border-b border-white/10 sticky top-0 z-10 text-foreground font-semibold">
                            <tr>
                              <th className="px-4 py-2.5 w-12 text-center text-muted-foreground font-mono">#</th>
                              {parsedCsv.headers.map((h, i) => (
                                <th key={i} className="px-4 py-2.5 whitespace-nowrap">
                                  {h}
                                </th>
                              ))}
                            </tr>
                          </thead>
                          <tbody className="divide-y divide-white/5 text-muted-foreground font-mono">
                            {parsedCsv.rows.length === 0 ? (
                              <tr>
                                <td
                                  colSpan={parsedCsv.headers.length + 1}
                                  className="px-4 py-8 text-center text-muted-foreground"
                                >
                                  No rows match your search filter.
                                </td>
                              </tr>
                            ) : (
                              parsedCsv.rows.map((row, rIdx) => (
                                <tr key={rIdx} className="hover:bg-white/5 transition-colors">
                                  <td className="px-4 py-2 text-center text-muted-foreground/60 select-none">
                                    {rIdx + 1}
                                  </td>
                                  {row.map((cell, cIdx) => (
                                    <td
                                      key={cIdx}
                                      className={cn(
                                        "px-4 py-2 whitespace-nowrap text-foreground/80",
                                        searchLower && cell.toLowerCase().includes(searchLower) && "text-primary font-bold bg-primary/10"
                                      )}
                                    >
                                      {cell}
                                    </td>
                                  ))}
                                </tr>
                              ))
                            )}
                          </tbody>
                        </table>
                      </div>
                    ) : fileMeta.category === "pdf" && previewTab === "formatted" ? (
                      <div className="w-full h-[72vh] rounded-2xl overflow-hidden border border-white/10 bg-[#1e1e1e] shadow-2xl relative flex flex-col">
                        <iframe
                          src={API.getDocumentRawUrl(previewFilename)}
                          className="w-full h-full border-0 rounded-2xl bg-[#525659]"
                          title={previewFilename}
                        />
                      </div>
                    ) : fileMeta.category === "pdf" || fileMeta.category === "docx" ? (
                      <div className="max-w-3xl mx-auto p-6 md:p-8 bg-black/20 rounded-2xl border border-white/5 font-sans text-sm md:text-base leading-relaxed text-foreground/90 whitespace-pre-wrap select-text">
                        {rawContent}
                      </div>
                    ) : (
                      /* Code / JSON / Raw / Text view with line-number gutter */
                      <div className="rounded-2xl border border-white/10 bg-black/40 overflow-hidden font-mono text-xs shadow-inner">
                        <div className="overflow-x-auto p-4 flex">
                          <div className="select-none text-muted-foreground/40 text-right pr-4 border-r border-white/10 space-y-0.5">
                            {filteredLines.map((item) => (
                              <div key={item.idx}>{item.idx}</div>
                            ))}
                          </div>
                          <div className="pl-4 space-y-0.5 text-muted-foreground/90 flex-1 whitespace-pre">
                            {filteredLines.map((item) => (
                              <div
                                key={item.idx}
                                className={cn(
                                  "hover:text-foreground transition-colors",
                                  searchLower && item.line.toLowerCase().includes(searchLower) && "bg-primary/20 text-foreground px-1 -mx-1 rounded"
                                )}
                              >
                                {item.line || " "}
                              </div>
                            ))}
                          </div>
                        </div>
                      </div>
                    )}
                  </>
                )}
              </div>
            </div>
          </div>
        )
      })()}

      {/* ── Delete Single File Modal ── */}
      {fileToDelete && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm animate-in fade-in duration-150 p-4">
          <div className="bg-[#181818] border border-white/15 rounded-3xl p-6 w-full max-w-sm shadow-2xl">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-10 h-10 rounded-2xl bg-destructive/10 flex items-center justify-center shrink-0 border border-destructive/20 text-destructive">
                <Trash2 size={18} />
              </div>
              <div className="min-w-0">
                <h3 className="text-sm font-bold text-foreground">Delete Document</h3>
                <p className="text-xs text-muted-foreground truncate">{fileToDelete.filename}</p>
              </div>
            </div>
            <p className="text-xs text-muted-foreground mb-6 leading-relaxed">
              Are you sure you want to delete <span className="text-foreground font-semibold">&ldquo;{fileToDelete.filename}&rdquo;</span>? It will be permanently removed from the vector database.
            </p>
            <div className="flex gap-2 justify-end">
              <button
                onClick={() => setFileToDelete(null)}
                disabled={isDeleting}
                className="px-4 py-2 text-xs font-semibold rounded-xl bg-white/5 hover:bg-white/10 text-foreground transition-all"
              >
                Cancel
              </button>
              <button
                onClick={handleConfirmDelete}
                disabled={isDeleting}
                className="px-4 py-2 text-xs font-semibold rounded-xl bg-destructive text-white hover:bg-destructive/90 transition-all flex items-center gap-1.5 shadow-md shadow-destructive/20"
              >
                {isDeleting ? <RefreshCcw size={12} className="animate-spin" /> : <Trash2 size={12} />}
                <span>Delete</span>
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ── Delete All Documents Modal ── */}
      {showDeleteAllModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm animate-in fade-in duration-150 p-4">
          <div className="bg-[#181818] border border-white/15 rounded-3xl p-6 w-full max-w-sm shadow-2xl">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-10 h-10 rounded-2xl bg-destructive/10 flex items-center justify-center shrink-0 border border-destructive/20 text-destructive">
                <Trash2 size={18} />
              </div>
              <div className="min-w-0">
                <h3 className="text-sm font-bold text-foreground">Clear Knowledge Base</h3>
                <p className="text-xs text-muted-foreground">Erase all indexed documents</p>
              </div>
            </div>
            <p className="text-xs text-muted-foreground mb-6 leading-relaxed">
              Are you sure you want to delete all <span className="text-foreground font-semibold">{files.length} indexed documents</span>? All text chunks, full-text indexes, and vector embeddings will be permanently erased.
            </p>
            <div className="flex gap-2 justify-end">
              <button
                onClick={() => setShowDeleteAllModal(false)}
                disabled={isDeletingAll}
                className="px-4 py-2 text-xs font-semibold rounded-xl bg-white/5 hover:bg-white/10 text-foreground transition-all"
              >
                Cancel
              </button>
              <button
                onClick={handleConfirmDeleteAll}
                disabled={isDeletingAll}
                className="px-4 py-2 text-xs font-semibold rounded-xl bg-destructive text-white hover:bg-destructive/90 transition-all flex items-center gap-1.5 shadow-md shadow-destructive/20"
              >
                {isDeletingAll ? <RefreshCcw size={12} className="animate-spin" /> : <Trash2 size={12} />}
                <span>Delete All</span>
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
