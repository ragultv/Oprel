"use client"

import { useState, useEffect, useRef } from "react"
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
  Zap
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

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFiles = Array.from(e.target.files || [])
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
      const results = await API.searchKnowledge(searchQuery, 10)
      setSearchResults(results)
    } catch (err: any) {
      toast({ title: "Search Failed", description: err.message, variant: "destructive" })
    } finally {
      setIsSearching(false)
    }
  }

  const filteredFiles = files.filter(f => 
    f.filename.toLowerCase().includes(searchQuery.toLowerCase())
  )

  return (
    <div className="flex flex-col h-full bg-[#0a0a0a] relative overflow-hidden">
      <div className="absolute inset-x-0  pointer-events-none" />
      {/* Refined Header */}
      <header className="relative px-6 py-5 border-b border-border/80 bg-[#0f0f0f]/90 backdrop-blur flex items-center justify-between shrink-0">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-primary/10 border border-primary/20 flex items-center justify-center shadow-[0_0_0_1px_rgba(238,70,71,0.08)]">
            <Database size={20} className="text-primary" />
          </div>
          <div>
            <h1 className="text-base font-bold text-foreground">Knowledge Base</h1>
            <div className="flex items-center gap-1.5">
              <span className="flex h-1.5 w-1.5 rounded-full bg-primary" />
              <p className="text-[11px] text-muted-foreground font-medium uppercase tracking-wider">Local Vector Store</p>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <button 
            onClick={refreshFiles}
            className="p-2 rounded-lg hover:bg-secondary text-muted-foreground transition-all"
            title="Refresh documents"
          >
            <RefreshCcw size={16} className={cn(loading && "animate-spin")} />
          </button>
        </div>
      </header>

      <div className="relative flex-1 overflow-y-auto p-6 md:p-10">
        <div className="max-w-5xl mx-auto space-y-8">
          
          {/* Accent Stats Cards */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div className="p-5 rounded-2xl bg-secondary/20 border border-border/80 group shadow-sm">
              <div className="text-[10px] uppercase font-bold text-muted-foreground tracking-widest mb-1 flex items-center gap-2">
                <FileText size={12} className="text-primary/60" />
                Indexed Files
              </div>
              <div className="text-2xl font-bold text-foreground">{files.length}</div>
            </div>
            <div className="p-5 rounded-2xl bg-secondary/20 border border-border/80 shadow-sm">
              <div className="text-[10px] uppercase font-bold text-muted-foreground tracking-widest mb-1 flex items-center gap-2">
                <CheckCircle2 size={12} className="text-primary/60" />
                Index Status
              </div>
              <div className="text-2xl font-bold text-foreground">{loading ? "Updating..." : "Ready"}</div>
            </div>
            <div className="p-5 rounded-2xl bg-secondary/20 border border-border/80 shadow-sm">
              <div className="text-[10px] uppercase font-bold text-muted-foreground tracking-widest mb-1 flex items-center gap-2">
                <Layers size={12} className="text-primary/60" />
                Current Engine
              </div>
              <div className="text-2xl font-bold text-foreground">Hybrid</div>
            </div>
          </div>

          <div className="space-y-6">
            {/* View Selection & Actions */}
            <div className="flex flex-col md:flex-row gap-4 items-center justify-between">
              <div className="flex p-1 bg-secondary/30 border border-border/80 rounded-xl">
                <button 
                  onClick={() => setActiveTab("files")}
                  className={cn(
                    "px-4 py-1.5 rounded-lg text-xs font-bold transition-all",
                    activeTab === "files" ? "bg-primary text-primary-foreground shadow-sm" : "text-muted-foreground hover:text-foreground"
                  )}
                >
                  Document Inventory
                </button>
                <button 
                  onClick={() => setActiveTab("search")}
                  className={cn(
                    "px-4 py-1.5 rounded-lg text-xs font-bold transition-all",
                    activeTab === "search" ? "bg-primary text-primary-foreground shadow-sm" : "text-muted-foreground hover:text-foreground"
                  )}
                >
                  Neural Search
                </button>
              </div>

              <div className="flex items-center gap-3 w-full md:w-auto">
                 <div className="relative flex-1 md:w-80">
                    <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground" />
                    <input 
                      type="text" 
                      placeholder={activeTab === "files" ? "Filter files..." : "Search for knowledge chunks..."}
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      onKeyDown={(e) => e.key === 'Enter' && activeTab === 'search' && handleKnowledgeSearch()}
                      className="w-full bg-secondary/30 border border-border rounded-xl py-2 pl-9 pr-4 text-xs text-foreground placeholder:text-muted-foreground focus:outline-none focus:border-primary/30 transition-all"
                    />
                    {activeTab === 'search' && searchQuery && (
                      <button 
                        onClick={handleKnowledgeSearch}
                        disabled={isSearching}
                        className="absolute right-2 top-1/2 -translate-y-1/2 p-1.5 text-primary hover:bg-primary/10 rounded-lg transition-all"
                      >
                         {isSearching ? <RefreshCcw size={12} className="animate-spin" /> : <ArrowRight size={12} />}
                      </button>
                    )}
                 </div>
                 
                 {activeTab === "files" && (
                   <>
                    <button 
                      onClick={() => fileInputRef.current?.click()}
                      disabled={uploading}
                      className="flex items-center gap-2 px-4 py-2 rounded-xl bg-primary hover:bg-primary/90 text-primary-foreground font-bold text-xs transition-all disabled:opacity-50 shrink-0 shadow-lg shadow-primary/10"
                    >
                      <FileUp size={14} />
                      {uploading ? "Indexing..." : "Add File"}
                    </button>
                    <input 
                      type="file" 
                      ref={fileInputRef} 
                      className="hidden" 
                      multiple 
                      accept=".pdf,.docx,.txt,.md,.py,.js,.ts,.tsx,.json,.yaml,.yml"
                      onChange={handleUpload}
                    />
                   </>
                 )}
              </div>
            </div>

            {/* Content Area */}
            <div className="min-h-[300px]">
              {activeTab === "files" ? (
                <div className="bg-background border border-border rounded-2xl overflow-hidden">
                  <table className="w-full text-left text-sm border-collapse">
                    <thead>
                      <tr className="border-b border-border bg-secondary/20">
                        <th className="px-6 py-4 font-bold text-muted-foreground text-[11px] uppercase tracking-wider">Document Name</th>
                        <th className="px-6 py-4 font-bold text-muted-foreground text-[11px] uppercase tracking-wider">Chunks</th>
                        <th className="px-6 py-4 font-bold text-muted-foreground text-[11px] uppercase tracking-wider hidden sm:table-cell">Added</th>
                        <th className="px-6 py-4 text-right w-16"></th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-border">
                      {loading ? (
                        Array.from({ length: 3 }).map((_, i) => (
                          <tr key={i} className="animate-pulse">
                            <td className="px-6 py-6"><div className="h-4 bg-secondary rounded w-48" /></td>
                            <td className="px-6 py-6"><div className="h-4 bg-secondary rounded w-12" /></td>
                            <td className="px-6 py-6 hidden sm:table-cell"><div className="h-4 bg-secondary rounded w-24" /></td>
                            <td className="px-6 whitespace-nowrap"></td>
                          </tr>
                        ))
                      ) : filteredFiles.length === 0 ? (
                        <tr>
                          <td colSpan={4} className="px-6 py-16 text-center">
                            <div className="flex flex-col items-center gap-2 opacity-30">
                              <Database size={40} />
                              <div className="text-xs font-bold">The index is empty</div>
                            </div>
                          </td>
                        </tr>
                      ) : (
                        filteredFiles.map(file => (
                          <tr key={file.id} className="group hover:bg-secondary/20 transition-all">
                            <td className="px-6 py-4">
                              <div className="flex items-center gap-3">
                                <FileText size={16} className="text-muted-foreground group-hover:text-primary transition-colors" />
                                <span className="font-semibold text-foreground truncate max-w-[300px]">{file.filename}</span>
                              </div>
                            </td>
                            <td className="px-6 py-4">
                              <span className="px-2 py-0.5 rounded-lg bg-primary/10 text-[10px] font-bold text-primary border border-primary/20">
                                {file.chunks} Chunks
                              </span>
                            </td>
                            <td className="px-6 py-4 hidden sm:table-cell">
                              <div className="text-xs text-muted-foreground font-medium">
                                {new Date(file.indexed_at || Date.now()).toLocaleDateString()}
                              </div>
                            </td>
                            <td className="px-6 py-4 text-right">
                              <button className="p-1.5 rounded-lg text-muted-foreground hover:bg-red-500/10 hover:text-red-400 transition-all">
                                <Trash2 size={14} />
                              </button>
                            </td>
                          </tr>
                        ))
                      )}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div className="space-y-4 pb-20">
                  {isSearching ? (
                    <div className="flex flex-col items-center justify-center py-20 gap-4">
                       <RefreshCcw size={24} className="text-primary animate-spin" />
                       <p className="text-[11px] font-bold uppercase tracking-widest text-primary/70">Searching Index...</p>
                    </div>
                  ) : searchResults.length === 0 ? (
                    <div className="flex flex-col items-center justify-center py-20 border border-border border-dashed rounded-3xl bg-secondary/10">
                       <Search size={32} className="text-muted-foreground/20 mb-3" />
                       <p className="text-xs font-bold text-muted-foreground/60 tracking-wider uppercase">Search across vectors</p>
                    </div>
                  ) : (
                    <div className="grid grid-cols-1 gap-4">
                       {searchResults.map((result, i) => {
                         const rawScore = result.score || (result as any).hybrid_score || 0
                         // Normalize if it looks like RRF (very small) or similarity (up to 1)
                         // For this display, we'll just show as percentage if 0-1
                         const displayScore = rawScore > 1 ? "1.0" : rawScore.toFixed(3)

                         return (
                          <div key={i} className="p-6 rounded-[1.5rem] bg-secondary/20 border border-border hover:border-primary/20 transition-all">
                              <div className="flex items-center justify-between mb-4">
                                <div className="flex items-center gap-3">
                                    <div className="px-2.5 py-1 rounded bg-primary/10 text-[10px] font-bold text-primary uppercase tracking-widest">
                                      Chunk {i + 1}
                                    </div>
                                    <span className="text-[11px] font-bold text-muted-foreground flex items-center gap-1.5">
                                      <FileText size={12} className="opacity-50" />
                                      {result.metadata?.filename || "Unknown Source"}
                                    </span>
                                </div>
                                <div className="text-[10px] font-bold text-muted-foreground uppercase">
                                  Similarity Score: <span className="text-primary font-black">{displayScore}</span>
                                </div>
                              </div>
                              <div className="text-sm leading-relaxed text-foreground/80 font-medium bg-black/10 p-5 rounded-xl border border-border/50 italic">
                                "{result.text}"
                              </div>
                              <div className="flex justify-end mt-4">
                                <button className="flex items-center gap-1 text-[11px] font-bold text-muted-foreground hover:text-primary transition-colors">
                                    Show full context <ExternalLink size={10} />
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

          {/* Local Security Tip */}
          <div className="p-6 rounded-2xl border border-border bg-secondary/15 flex gap-5 items-start shadow-sm">
            <div className="p-3 rounded-xl bg-primary/10 shrink-0 border border-primary/20">
              <Zap className="text-primary" size={18} />
            </div>
            <div>
              <h3 className="text-xs font-bold text-foreground mb-1 uppercase tracking-wider">Local Neural Privacy</h3>
              <p className="text-[11px] text-muted-foreground leading-relaxed font-medium">
                Your knowledge chunks and vectors are indexed and stored **locally** on your device. Oprel never uploads your private documents to external clouds for indexing. 
              </p>
            </div>
          </div>

        </div>
      </div>
    </div>
  )
}
