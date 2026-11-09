"use client"

import { useEffect, useRef, useState, useCallback, useMemo } from "react"
import {
  ChevronDown,
  Download,
  Image as ImageIcon,
  Sparkles,
  Trash2,
  Maximize2,
  Copy,
  Check,
  RotateCcw,
  SlidersHorizontal,
  X,
  ArrowUp,
  Dice5,
  Loader2,
  Info,
  Plus,
  History,
  MessageSquare,
  PanelRightClose,
  PanelRightOpen,
  RefreshCw,
} from "lucide-react"
import { API } from "@/services/api"
import type { ImageGenerationJob, ImageModel, ImageChat, ImageChatMessage } from "@/services/api"
import { cn } from "@/services/utils"

const ASPECT_RATIO_PRESETS = [
  { label: "1:1 Square", size: "1024x1024", fallbackSize: "512x512", aspect: "aspect-square" },
  { label: "16:9 Landscape", size: "1536x864", fallbackSize: "1024x768", aspect: "aspect-video" },
  { label: "9:16 Portrait", size: "864x1536", fallbackSize: "768x1024", aspect: "aspect-[9/16]" },
  { label: "4:3 Classic", size: "1024x768", fallbackSize: "512x384", aspect: "aspect-[4/3]" },
]

const SAMPLER_PRESETS = ["euler", "euler_a", "dpmpp_2m", "dpmpp_2m_sde", "heun"]

const INSPIRING_PROMPTS = [
  {
    title: "Cyberpunk Tokyo",
    prompt: "A cinematic photo of a cyberpunk street in rain-slicked Tokyo, neon signs reflecting in puddles, dramatic shadows, 8k masterpiece",
    tag: "Cinematic",
  },
  {
    title: "Glass Hummingbird",
    prompt: "Macro close-up of an intricate glass hummingbird suspended in mid-air, catching iridescent golden hour sunlight, hyperrealistic",
    tag: "Macro",
  },
  {
    title: "Synthwave Sunset",
    prompt: "Retro-futuristic chrome sports car speeding down a desert highway towards a giant wireframe sunset, synthwave 1980s aesthetic",
    tag: "Retro 3D",
  },
  {
    title: "Space Explorer",
    prompt: "An adorable fluffy red panda astronaut floating outside a lunar base, detailed spacesuit, Earth glowing in the cosmic background",
    tag: "Whimsical",
  },
]

const SURPRISE_PROMPTS = [
  "Bioluminescent jellyfish forest beneath an alien ocean, ethereal soft cyan glow, dreamlike 8k",
  "A cozy ancient library carved into a giant redwood tree, glowing lanterns, floating dust motes, studio ghibli vibe",
  "Steampunk mechanical owl with brass clockwork gears and luminous amber glass eyes, perched on an antique telescope",
  "Futuristic solarpunk city with lush vertical gardens, crystal sky bridges, and solar airships in golden morning light",
  "A mystical samurai meditating beneath a luminous pink cherry blossom tree, mist drifting across stone lanterns, octane render",
  "A crystal dragon sleeping on a hoard of glowing gemstones inside a volcanic cavern, cinematic volumetric lighting",
]

function toImageUrl(data: { url?: string; b64_json?: string }) {
  if (data.url) return data.url
  if (data.b64_json) return `data:image/png;base64,${data.b64_json}`
  return ""
}

export function ImageGenerationView() {
  // ── Multi-Session State ───────────────────────────────────────────────────
  const [chats, setChats] = useState<ImageChat[]>([])
  const [activeChatId, setActiveChatId] = useState<string | null>(null)
  const [messages, setMessages] = useState<ImageChatMessage[]>([])
  const [isLoadingChats, setIsLoadingChats] = useState(true)
  const [isLoadingMessages, setIsLoadingMessages] = useState(false)

  // ── Tuning Parameters State ───────────────────────────────────────────────
  const [selectedModel, setSelectedModel] = useState("")
  const [imageModels, setImageModels] = useState<ImageModel[]>([])
  const [modelLoading, setModelLoading] = useState(true)
  const [prompt, setPrompt] = useState("")
  const [negativePrompt, setNegativePrompt] = useState("blurry, low quality, watermark, extra limbs")
  const [size, setSize] = useState("1024x1024")
  const [steps, setSteps] = useState(15)
  const [cfgScale, setCfgScale] = useState(7)
  const [seed, setSeed] = useState(-1)
  const [sampler, setSampler] = useState("euler")
  const [responseFormat, setResponseFormat] = useState<"url" | "b64_json">("url")

  // ── Generation / Progress State ───────────────────────────────────────────
  const [loading, setLoading] = useState(false)
  const [generationProgress, setGenerationProgress] = useState<{ value: number; message: string } | null>(null)
  const [activeJobPrompt, setActiveJobPrompt] = useState("")
  const [error, setError] = useState<string | null>(null)
  const [copiedId, setCopiedId] = useState<string | null>(null)

  // ── Right Sidebar & Modal States ──────────────────────────────────────────
  const [sidebarTab, setSidebarTab] = useState<"settings" | "history">("settings")
  const [isSidebarOpen, setIsSidebarOpen] = useState(true)
  const [isModelDropdownOpen, setIsModelDropdownOpen] = useState(false)
  const [activeLightboxItem, setActiveLightboxItem] = useState<ImageChatMessage | null>(null)
  const [showMetadataId, setShowMetadataId] = useState<string | null>(null)
  const [daysFilter, setDaysFilter] = useState<number>(30)
  const [isDaysDropdownOpen, setIsDaysDropdownOpen] = useState(false)

  const streamCleanupRef = useRef<(() => void) | null>(null)
  const feedEndRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  const formatDate = (iso?: string) => {
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

  const filteredChats = useMemo(() => {
    if (daysFilter === 0) return chats
    const cutoff = Date.now() - daysFilter * 24 * 60 * 60 * 1000
    return chats.filter((c) => {
      const ts = c.updated_at || c.created_at
      if (!ts) return true
      const time = new Date(ts).getTime()
      return isNaN(time) || time >= cutoff
    })
  }, [chats, daysFilter])

  // Auto-scroll feed to latest bottom item
  const scrollToBottom = useCallback((smooth = true) => {
    if (feedEndRef.current) {
      feedEndRef.current.scrollIntoView({ behavior: smooth ? "smooth" : "auto" })
    }
  }, [])

  const stopStreaming = () => {
    if (streamCleanupRef.current) {
      streamCleanupRef.current()
      streamCleanupRef.current = null
    }
  }

  // ── Load Image Models on Mount ────────────────────────────────────────────
  useEffect(() => {
    let active = true

    API.fetchImageModels()
      .then((models) => {
        if (!active) return
        const downloadedModels = models.filter(
          (m) => m.downloaded && m.backend === "stable-diffusion.cpp" && m.supported !== false
        )
        setImageModels(downloadedModels)
        if (downloadedModels.length > 0) {
          setSelectedModel((current) => {
            if (current && downloadedModels.some((m) => (m.local_path || m.repo_id) === current)) {
              return current
            }
            return downloadedModels[0].local_path || downloadedModels[0].repo_id
          })
        }
      })
      .catch((err) => setError(err instanceof Error ? err.message : "Failed to fetch image models"))
      .finally(() => {
        if (active) setModelLoading(false)
      })

    return () => {
      active = false
    }
  }, [])

  // ── Load All Image Chats from DB on Mount ─────────────────────────────────
  const loadChatsList = useCallback(async (selectFirst = false) => {
    try {
      setIsLoadingChats(true)
      const list = await API.fetchImageChats()
      setChats(list)
      if (selectFirst && list.length > 0) {
        setActiveChatId(list[0].id)
      } else if (list.length === 0) {
        // Automatically start fresh session
        handleNewChat()
      }
    } catch (err) {
      console.error("Failed to load image chats:", err)
    } finally {
      setIsLoadingChats(false)
    }
  }, [])

  useEffect(() => {
    loadChatsList(true)
  }, [loadChatsList])

  // ── Load Messages When Active Chat Changes ────────────────────────────────
  useEffect(() => {
    if (!activeChatId) {
      setMessages([])
      return
    }

    let active = true
    setIsLoadingMessages(true)

    API.getImageChat(activeChatId)
      .then((chat) => {
        if (active && chat) {
          setMessages(chat.messages || [])
          setTimeout(() => scrollToBottom(false), 50)
        }
      })
      .catch((err) => {
        console.error(`Failed to fetch chat ${activeChatId}:`, err)
        setMessages([])
      })
      .finally(() => {
        if (active) setIsLoadingMessages(false)
      })

    return () => {
      active = false
    }
  }, [activeChatId, scrollToBottom])

  useEffect(() => {
    return () => {
      stopStreaming()
    }
  }, [])

  // ── Create New Chat Session ───────────────────────────────────────────────
  const handleNewChat = async () => {
    stopStreaming()
    setLoading(false)
    setGenerationProgress(null)
    setError(null)
    try {
      const newChat = await API.createImageChat("New Image Chat")
      setChats((prev) => [newChat, ...prev.filter((c) => c.id !== newChat.id)])
      setActiveChatId(newChat.id)
      setMessages([])
      setPrompt("")
      if (textareaRef.current) {
        textareaRef.current.focus()
      }
    } catch (err) {
      console.error("Failed to create new image chat:", err)
    }
  }

  // ── Finalize Completed Render into DB ─────────────────────────────────────
  const finalizeJob = async (
    imageUrl: string,
    jobModel: string,
    jobPrompt: string,
    jobNegativePrompt: string,
    jobSize: string,
    jobSteps: number,
    jobCfgScale: number,
    jobSeed: number,
    jobSampler: string
  ) => {
    stopStreaming()
    setGenerationProgress(null)
    setLoading(false)

    let targetChatId = activeChatId
    if (!targetChatId) {
      const newChat = await API.createImageChat(jobPrompt.slice(0, 40) || "Image Chat")
      targetChatId = newChat.id
      setActiveChatId(newChat.id)
      setChats((prev) => [newChat, ...prev])
    }

    try {
      const savedMsg = await API.saveImageMessage(targetChatId, {
        prompt: jobPrompt,
        image_url: imageUrl,
        model: jobModel,
        size: jobSize,
        negative_prompt: jobNegativePrompt,
        steps: jobSteps,
        cfg_scale: jobCfgScale,
        seed: jobSeed,
        sampler: jobSampler,
      })

      setMessages((prev) => [...prev, savedMsg])
      // Refresh chats list to update titles & counts
      loadChatsList(false)
      setTimeout(() => scrollToBottom(true), 120)
    } catch (err) {
      console.error("Failed to save image message to DB:", err)
    }
  }

  // ── Start / Stream Generation ─────────────────────────────────────────────
  const handleGenerate = async (customPrompt?: string) => {
    const targetPrompt = (customPrompt || prompt).trim()
    const trimmedModel = selectedModel.trim()
    if (!targetPrompt || !trimmedModel || loading) return

    setLoading(true)
    setActiveJobPrompt(targetPrompt)
    setGenerationProgress({ value: 1, message: "Queued" })
    setError(null)
    setPrompt("") // Clear input bar

    if (textareaRef.current) {
      textareaRef.current.style.height = "auto"
    }

    // Ensure we have an active chat session
    let targetChatId = activeChatId
    if (!targetChatId) {
      try {
        const newChat = await API.createImageChat(targetPrompt.slice(0, 40) || "Image Chat")
        targetChatId = newChat.id
        setActiveChatId(newChat.id)
        setChats((prev) => [newChat, ...prev])
      } catch (err) {
        console.error("Failed to auto-create image chat:", err)
      }
    }

    try {
      const started = await API.startImageGeneration({
        model: trimmedModel,
        prompt: targetPrompt,
        responseFormat,
        size,
        negativePrompt: negativePrompt.trim() || undefined,
        steps,
        cfgScale,
        seed,
        sampler,
      })

      setTimeout(() => scrollToBottom(true), 80)

      const cleanup = API.streamImageGenerationProgress(
        started.id,
        (nextJob) => {
          setGenerationProgress({
            value: Math.max(1, nextJob.progress || 1),
            message: nextJob.message || "Generating",
          })
        },
        (completedJob) => {
          const first = completedJob.result?.data?.[0]
          const imageUrl = first ? toImageUrl(first) : ""
          if (!imageUrl) {
            setError("Image generation completed, but no image data was returned.")
            setGenerationProgress(null)
            setLoading(false)
            return
          }
          finalizeJob(imageUrl, trimmedModel, targetPrompt, negativePrompt.trim(), size, steps, cfgScale, seed, sampler)
        },
        (streamError) => {
          setError(streamError)
          setGenerationProgress(null)
          setLoading(false)
        }
      )

      streamCleanupRef.current = cleanup
    } catch (err) {
      setError(err instanceof Error ? err.message : "Image generation failed")
      setLoading(false)
      setGenerationProgress(null)
    }
  }

  // ── Actions ───────────────────────────────────────────────────────────────
  const handleDeleteChat = async (chatId: string, e?: React.MouseEvent) => {
    e?.stopPropagation()
    if (!confirm("Delete this image chat session?")) return
    try {
      await API.deleteImageChat(chatId)
      setChats((prev) => prev.filter((c) => c.id !== chatId))
      if (activeChatId === chatId) {
        const remaining = chats.filter((c) => c.id !== chatId)
        if (remaining.length > 0) {
          setActiveChatId(remaining[0].id)
        } else {
          handleNewChat()
        }
      }
    } catch (err) {
      console.error("Failed to delete image chat:", err)
    }
  }

  const handleClearAllHistory = async () => {
    if (!confirm("Clear all image chat history from the database?")) return
    try {
      await API.clearImageChats()
      setChats([])
      setMessages([])
      handleNewChat()
    } catch (err) {
      console.error("Failed to clear image history:", err)
    }
  }

  const handleCopyPrompt = (id: string, text: string, e?: React.MouseEvent) => {
    e?.stopPropagation()
    navigator.clipboard.writeText(text)
    setCopiedId(id)
    setTimeout(() => setCopiedId(null), 1500)
  }

  const handleIteratePrompt = (targetPrompt: string, e?: React.MouseEvent) => {
    e?.stopPropagation()
    setPrompt(targetPrompt)
    if (textareaRef.current) {
      textareaRef.current.focus()
    }
  }

  const handleSurpriseMe = () => {
    const random = SURPRISE_PROMPTS[Math.floor(Math.random() * SURPRISE_PROMPTS.length)]
    setPrompt(random)
    if (textareaRef.current) {
      textareaRef.current.focus()
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleGenerate()
    }
  }

  const selectedModelObj = imageModels.find((m) => (m.local_path || m.repo_id) === selectedModel)
  const selectedModelName = selectedModelObj ? selectedModelObj.id : "Choose Image Model"
  const activeChat = chats.find((c) => c.id === activeChatId)

  return (
    <div className="flex flex-col h-full bg-[#0a0a0a] relative overflow-hidden text-foreground select-none font-sans">
      {/* Studio Header (Full width across top, identical to Audio Studio) */}
      <header className="h-14 border-b border-white/10 flex items-center justify-between px-5 sm:px-8 shrink-0 bg-background/95 backdrop-blur-sm sticky top-0 z-20">
        <div className="flex items-center gap-3 min-w-0">
          <div className="w-8 h-8 rounded-xl bg-primary/10 text-primary border border-primary/20 flex items-center justify-center shrink-0">
            <ImageIcon size={18} />
          </div>
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <span className="text-sm font-bold text-foreground">Image Studio</span>
              <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
            </div>
          </div>
        </div>


        {/* Right Actions & Sidebar Toggle */}
        <div className="flex items-center gap-2 sm:gap-3">
          <button
            onClick={() => {
              loadChatsList(false)
              setModelLoading(true)
              API.fetchImageModels()
                .then((models) => {
                  setImageModels(models.filter((m) => m.downloaded && m.backend === "stable-diffusion.cpp"))
                })
                .finally(() => setModelLoading(false))
            }}
            disabled={isLoadingChats}
            className="p-2 rounded-lg hover:bg-white/10 text-muted-foreground hover:text-foreground transition-all"
            title="Refresh models and chats"
          >
            <RefreshCw size={15} className={cn(isLoadingChats && "animate-spin text-primary")} />
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

      {/* Main Content Area (Under the header, side-by-side with right sidebar) */}
      <div className="flex-1 flex flex-col lg:flex-row overflow-hidden w-full relative min-h-0">
        {/* Left Canvas: Scrollable Feed + Bottom Floating Dock */}
        <div className="flex-1 h-full min-w-0 flex flex-col overflow-hidden relative">
          {/* Scrollable Conversation Feed */}
          <main className="flex-1 overflow-y-auto px-4 py-6 md:px-8 space-y-6 pb-40">
          {/* Empty State / Hero view when active session has no messages */}
          {messages.length === 0 && !loading && (
            <div className="max-w-2xl mx-auto pt-8 pb-6 text-center space-y-8 animate-in fade-in duration-300">
              <div className="space-y-3">
                <h2 className="text-3xl font-extrabold tracking-tight text-foreground sm:text-4xl">
                  What will you imagine today?
                </h2>
                <p className="text-sm text-muted-foreground max-w-md mx-auto leading-relaxed">
                  Type any prompt below to generate an image. Your renders and settings are automatically organized into this session.
                </p>
              </div>

              {/* Inspiring Prompt Cards Grid */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-left">
                {INSPIRING_PROMPTS.map((item, idx) => (
                  <button
                    key={idx}
                    type="button"
                    onClick={() => {
                      setPrompt(item.prompt)
                      if (textareaRef.current) textareaRef.current.focus()
                    }}
                    className="group p-4 rounded-2xl bg-white/[0.03] border border-white/5 hover:border-primary/40 hover:bg-white/[0.06] transition-all flex flex-col justify-between gap-3 text-left shadow-sm"
                  >
                    <div className="space-y-1">
                      <span className="text-[10px] font-bold uppercase tracking-wider text-primary font-mono">
                        {item.tag}
                      </span>
                      <h3 className="text-xs font-semibold text-foreground group-hover:text-primary transition-colors">
                        {item.title}
                      </h3>
                      <p className="text-[11px] text-muted-foreground line-clamp-2 leading-relaxed">
                        {item.prompt}
                      </p>
                    </div>
                    <div className="text-[10px] text-muted-foreground/60 flex items-center gap-1 group-hover:text-foreground transition-colors font-medium">
                      <span>Use prompt</span>
                    </div>
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Chronological Message Turns (Latest at Bottom) */}
          <div className="max-w-3xl mx-auto space-y-8">
            {messages.map((item) => {
              const isShowingInfo = showMetadataId === item.id
              return (
                <div key={item.id} className="space-y-3 animate-in fade-in slide-in-from-bottom-3 duration-200">
                  {/* 1. User Prompt Bubble (Right aligned) */}
                  <div className="flex justify-end">
                    <div className="max-w-[85%] rounded-2xl bg-[#1e1e1e] border border-white/10 px-4 py-2.5 text-xs text-foreground shadow-md space-y-1.5">
                      <p className="leading-relaxed select-text font-normal">{item.prompt}</p>
                      <div className="flex items-center gap-2 pt-1 text-[10px] font-mono text-muted-foreground border-t border-white/5">
                        <span>{item.size}</span>
                        <span>&bull;</span>
                        <span>{item.steps} steps</span>
                        <span>&bull;</span>
                        <span>CFG {item.cfg_scale}</span>
                      </div>
                    </div>
                  </div>

                  {/* 2. Assistant Render Output Card (Left aligned) */}
                  <div className="flex justify-start">
                    <div className="max-w-[85%] w-full rounded-3xl bg-[#141414] border border-white/10 overflow-hidden shadow-2xl relative group">
                      {/* Image Preview Container */}
                      <div
                        onClick={() => setActiveLightboxItem(item)}
                        className="cursor-zoom-in relative overflow-hidden bg-black/40 flex items-center justify-center min-h-[320px]"
                      >
                        <img
                          src={item.image_url}
                          alt={item.prompt}
                          className="w-full h-auto object-cover transition-transform duration-300 group-hover:scale-[1.01]"
                          loading="lazy"
                        />

                        {/* Hover action overlay */}
                        <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-black/20 opacity-0 group-hover:opacity-100 transition-opacity p-3 flex flex-col justify-between pointer-events-none">
                          <div className="flex justify-end gap-1.5 pointer-events-auto">
                            <button
                              type="button"
                              onClick={(e) => handleCopyPrompt(item.id, item.prompt, e)}
                              title="Copy prompt"
                              className="p-2 rounded-xl bg-black/60 backdrop-blur-md text-white hover:bg-black/90 transition-all border border-white/10"
                            >
                              {copiedId === item.id ? <Check size={14} className="text-emerald-400" /> : <Copy size={14} />}
                            </button>
                            <button
                              type="button"
                              onClick={(e) => handleIteratePrompt(item.prompt, e)}
                              title="Tweak / Re-roll prompt"
                              className="p-2 rounded-xl bg-black/60 backdrop-blur-md text-white hover:bg-black/90 transition-all border border-white/10"
                            >
                              <RotateCcw size={14} />
                            </button>
                            <button
                              type="button"
                              onClick={(e) => {
                                e.stopPropagation()
                                window.open(item.image_url, "_blank", "noopener,noreferrer")
                              }}
                              title="Download full resolution"
                              className="p-2 rounded-xl bg-black/60 backdrop-blur-md text-white hover:bg-black/90 transition-all border border-white/10"
                            >
                              <Download size={14} />
                            </button>
                          </div>

                          <div className="pointer-events-auto flex items-center justify-between">
                            <span className="text-[11px] text-white/90 font-medium truncate max-w-[80%] drop-shadow">
                              {item.prompt}
                            </span>
                            <button
                              type="button"
                              onClick={() => setActiveLightboxItem(item)}
                              className="inline-flex items-center gap-1 px-2.5 py-1 rounded-lg bg-white/20 backdrop-blur-md text-[11px] font-semibold text-white hover:bg-white/30 transition-all border border-white/15"
                            >
                              <Maximize2 size={12} />
                              <span>Expand</span>
                            </button>
                          </div>
                        </div>
                      </div>

                      {/* Card Footer */}
                      <div className="px-4 py-2.5 bg-[#161616] border-t border-white/5 flex items-center justify-between text-xs text-muted-foreground">
                        <div className="flex items-center gap-2 truncate">
                          <span className="font-semibold text-foreground truncate">{item.model.split(/[/\\]/).pop()}</span>
                          <span>&bull;</span>
                          <span className="font-mono text-[11px]">{item.size}</span>
                        </div>

                        <button
                          type="button"
                          onClick={() => setShowMetadataId(isShowingInfo ? null : item.id)}
                          className="p-1 rounded-md hover:bg-white/5 hover:text-foreground transition-colors"
                          title="Toggle parameters"
                        >
                          <Info size={14} className={cn("transition-colors", isShowingInfo && "text-primary")} />
                        </button>
                      </div>

                      {/* Metadata collapse */}
                      {isShowingInfo && (
                        <div className="px-4 py-3 bg-[#111] border-t border-white/5 text-[11px] font-mono space-y-1.5 text-muted-foreground animate-in fade-in duration-150">
                          <div><strong className="text-foreground">Prompt:</strong> {item.prompt}</div>
                          {item.negative_prompt && (
                            <div><strong className="text-foreground">Negative:</strong> {item.negative_prompt}</div>
                          )}
                          <div className="flex flex-wrap gap-x-4 gap-y-1 pt-1 text-[10px]">
                            <span>Steps: <strong className="text-foreground">{item.steps}</strong></span>
                            <span>CFG: <strong className="text-foreground">{item.cfg_scale}</strong></span>
                            <span>Sampler: <strong className="text-foreground">{item.sampler || "euler"}</strong></span>
                            <span>Seed: <strong className="text-foreground">{item.seed === -1 ? "random" : item.seed}</strong></span>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              )
            })}

            {/* Active Shimmering Card while Generating */}
            {loading && (
              <div className="space-y-3 animate-in fade-in duration-200">
                <div className="flex justify-end">
                  <div className="max-w-[85%] rounded-2xl bg-[#1e1e1e] border border-white/10 px-4 py-2.5 text-xs text-foreground shadow-md space-y-1.5">
                    <p className="leading-relaxed select-text font-normal">{activeJobPrompt}</p>
                    <div className="flex items-center gap-2 pt-1 text-[10px] font-mono text-muted-foreground border-t border-white/5">
                      <span>{size}</span>
                      <span>&bull;</span>
                      <span>{steps} steps</span>
                    </div>
                  </div>
                </div>

                <div className="flex justify-start">
                  <div className="max-w-[85%] w-full rounded-3xl bg-[#141414] border border-primary/30 p-8 shadow-2xl relative overflow-hidden flex flex-col items-center justify-center min-h-[340px] text-center space-y-4">
                    <div className="absolute inset-0 bg-gradient-to-tr from-primary/10 via-transparent to-orange-500/10 animate-pulse" />
                    <div className="relative w-16 h-16 rounded-2xl bg-primary/10 border border-primary/30 flex items-center justify-center text-primary shadow-lg shadow-primary/20">
                      <Loader2 size={28} className="animate-spin" />
                    </div>
                    <div className="relative space-y-1.5 max-w-xs">
                      <h4 className="text-sm font-bold text-foreground">
                        {generationProgress?.message || "Crafting your image"}...
                      </h4>
                      <p className="text-xs text-muted-foreground">Running diffusion steps locally via GPU offload.</p>
                    </div>

                    <div className="relative w-48 bg-white/10 h-2 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-gradient-to-r from-primary to-orange-400 rounded-full transition-all duration-300"
                        style={{ width: `${generationProgress?.value || 15}%` }}
                      />
                    </div>

                    <div className="relative text-[11px] font-mono text-muted-foreground">
                      {generationProgress?.value ? `${generationProgress.value}% complete` : "Processing"}
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Error Banner */}
            {error && (
              <div className="p-4 rounded-2xl bg-destructive/10 border border-destructive/20 text-destructive text-xs flex items-center justify-between gap-3 animate-in fade-in duration-200">
                <div className="flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-destructive shrink-0" />
                  <span className="font-semibold">{error}</span>
                </div>
                <button
                  type="button"
                  onClick={() => setError(null)}
                  className="text-xs text-muted-foreground hover:text-foreground font-semibold px-2 py-1 rounded"
                >
                  Dismiss
                </button>
              </div>
            )}

            <div ref={feedEndRef} />
          </div>
        </main>

        {/* ── Bottom Floating Prompt Dock (Gemini / Grok style) ───────────── */}
        <div className="absolute bottom-6 left-1/2 -translate-x-1/2 w-full max-w-3xl px-4 z-30">
          <div className="rounded-3xl border border-white/15 bg-[#141414]/90 backdrop-blur-2xl shadow-2xl p-2.5 space-y-2">
            {/* Quick Pills Bar */}
            <div className="flex items-center justify-between gap-2 px-1 text-xs">
              <div className="flex items-center gap-1 overflow-x-auto no-scrollbar py-0.5">
                {ASPECT_RATIO_PRESETS.map((preset) => {
                  const isActive = size === preset.size || size === preset.fallbackSize
                  return (
                    <button
                      key={preset.label}
                      type="button"
                      onClick={() => setSize(preset.size)}
                      className={cn(
                        "px-2.5 py-1 rounded-full text-[11px] font-semibold transition-all shrink-0 border",
                        isActive
                          ? "bg-white/15 text-white border-white/30 shadow-sm"
                          : "bg-transparent text-muted-foreground border-transparent hover:bg-white/5 hover:text-foreground"
                      )}
                    >
                      {preset.label}
                    </button>
                  )
                })}
              </div>

              <div className="flex items-center gap-1 shrink-0">
                <button
                  type="button"
                  onClick={handleSurpriseMe}
                  title="Surprise me (Random prompt)"
                  className="flex items-center gap-1 px-2.5 py-1 rounded-full text-[11px] font-semibold text-muted-foreground hover:text-foreground hover:bg-white/10 transition-colors"
                >
                  <Dice5 size={13} className="text-orange-400" />
                  <span className="hidden sm:inline">Inspire</span>
                </button>

                <button
                  type="button"
                  onClick={() => {
                    setIsSidebarOpen(true)
                    setSidebarTab("settings")
                  }}
                  title="Settings in sidebar"
                  className={cn(
                    "flex items-center gap-1 px-2.5 py-1 rounded-full text-[11px] font-semibold transition-colors border",
                    isSidebarOpen && sidebarTab === "settings"
                      ? "bg-primary/20 text-primary border-primary/30"
                      : "text-muted-foreground hover:text-foreground hover:bg-white/10 border-transparent"
                  )}
                >
                  <SlidersHorizontal size={13} />
                  <span className="hidden sm:inline">Settings</span>
                </button>
              </div>
            </div>

            {/* Prompt Text Input Row */}
            <div className="flex items-end gap-2 px-2 pb-1">
              <textarea
                ref={textareaRef}
                rows={1}
                value={prompt}
                onChange={(e) => {
                  setPrompt(e.target.value)
                  e.target.style.height = "auto"
                  e.target.style.height = `${Math.min(120, e.target.scrollHeight)}px`
                }}
                onKeyDown={handleKeyDown}
                placeholder="Describe the image you want to imagine... (Press Enter to generate)"
                className="flex-1 bg-transparent border-none text-sm text-foreground placeholder:text-muted-foreground/60 focus:outline-none resize-none max-h-32 leading-relaxed py-1.5"
              />

              <button
                type="button"
                onClick={() => handleGenerate()}
                disabled={!prompt.trim() || !selectedModel || loading}
                className={cn(
                  "w-9 h-9 rounded-2xl flex items-center justify-center transition-all shrink-0 shadow-md",
                  prompt.trim() && selectedModel && !loading
                    ? "bg-primary text-white hover:bg-primary/90 shadow-primary/25 cursor-pointer scale-100"
                    : "bg-white/10 text-muted-foreground cursor-not-allowed opacity-50"
                )}
              >
                {loading ? <Loader2 size={16} className="animate-spin text-white" /> : <ArrowUp size={16} />}
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* ── Right Sidebar: Settings & History Tabs (Audio Studio style) ─────── */}
      {isSidebarOpen && (
        <aside className="w-full lg:w-[340px] xl:w-[380px] shrink-0 border-l border-white/10 bg-[#121212] flex flex-col h-full z-20 animate-in slide-in-from-right-4 duration-200 overflow-hidden">
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
                History {chats.length > 0 && `(${chats.length})`}
              </button>
            </div>
          </div>

          {/* Scrollable Sidebar Body */}
          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {sidebarTab === "settings" ? (
              /* Settings View */
              <div className="space-y-4">
                {/* Model Selection */}
                <div className="space-y-1.5">
                  <label className="text-xs font-medium text-muted-foreground block">
                    Model
                  </label>
                  <div className="relative">
                    <button
                      type="button"
                      onClick={() => !modelLoading && setIsModelDropdownOpen((v) => !v)}
                      disabled={modelLoading}
                      className="w-full flex items-center justify-between px-3.5 py-2.5 rounded-2xl bg-black/30 border border-white/10 hover:border-white/20 text-xs font-semibold text-foreground transition-all shadow-inner disabled:opacity-50"
                    >
                      <div className="flex items-center gap-2 truncate">
                        <span className="truncate">{selectedModelName}</span>
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
                          {imageModels.length === 0 ? (
                            <div className="text-xs text-muted-foreground p-3 text-center">
                              No image models downloaded.
                            </div>
                          ) : (
                            imageModels.map((m) => {
                              const val = m.local_path || m.repo_id
                              const isSelected = val === selectedModel
                              return (
                                <button
                                  key={val}
                                  type="button"
                                  onClick={() => {
                                    setSelectedModel(val)
                                    setIsModelDropdownOpen(false)
                                  }}
                                  className={cn(
                                    "w-full flex items-center justify-between px-3 py-2 rounded-xl text-xs font-medium transition-all text-left",
                                    isSelected
                                      ? "bg-white/10 text-white font-semibold"
                                      : "text-muted-foreground hover:bg-white/5 hover:text-foreground"
                                  )}
                                >
                                  <div className="truncate pr-2">
                                    <div className="truncate text-foreground font-semibold">{m.id}</div>
                                    <div className="text-[10px] text-muted-foreground font-mono">{m.quantization || "SD.CPP"} · Local</div>
                                  </div>
                                  {isSelected && <Check size={13} className="text-primary shrink-0" />}
                                </button>
                              )
                            })
                          )}
                        </div>
                      </>
                    )}
                  </div>
                </div>


                {/* Generation Controls Header */}
                <div className="space-y-3 pt-2">
                  <label className="text-xs font-semibold text-muted-foreground block uppercase tracking-wider text-[10px]">
                    Image Controls
                  </label>

                  {/* Inference Steps Slider */}
                  <div className="flex items-center justify-between gap-3 p-2.5 rounded-2xl bg-black/20 border border-white/5">
                    <span className="text-xs font-medium text-muted-foreground">Inference Steps</span>
                    <div className="flex items-center gap-3 flex-1 max-w-[160px]">
                      <input
                        type="range"
                        min="5"
                        max="40"
                        step="1"
                        value={steps}
                        onChange={(e) => setSteps(parseInt(e.target.value, 10))}
                        className="w-full h-1.5 bg-white/10 rounded-lg appearance-none cursor-pointer accent-primary"
                      />
                      <span className="text-xs font-mono font-semibold text-foreground min-w-[28px] text-right">
                        {steps}
                      </span>
                    </div>
                  </div>

                  {/* CFG Guidance Scale Slider */}
                  <div className="flex items-center justify-between gap-3 p-2.5 rounded-2xl bg-black/20 border border-white/5">
                    <span className="text-xs font-medium text-muted-foreground">CFG Guidance</span>
                    <div className="flex items-center gap-3 flex-1 max-w-[160px]">
                      <input
                        type="range"
                        min="1"
                        max="15"
                        step="0.5"
                        value={cfgScale}
                        onChange={(e) => setCfgScale(parseFloat(e.target.value))}
                        className="w-full h-1.5 bg-white/10 rounded-lg appearance-none cursor-pointer accent-primary"
                      />
                      <span className="text-xs font-mono font-semibold text-foreground min-w-[28px] text-right">
                        {cfgScale.toFixed(1)}
                      </span>
                    </div>
                  </div>

                  {/* Sampling Method */}
                  <div className="flex items-center justify-between gap-3 p-2.5 rounded-2xl bg-black/20 border border-white/5">
                    <span className="text-xs font-medium text-muted-foreground">Sampler</span>
                    <select
                      value={sampler}
                      onChange={(e) => setSampler(e.target.value)}
                      className="bg-black/50 border border-white/10 rounded-xl px-2.5 py-1 text-xs text-foreground font-mono outline-none focus:border-primary/50 cursor-pointer"
                    >
                      {SAMPLER_PRESETS.map((s) => (
                        <option key={s} value={s} className="bg-[#1c1c1c] text-foreground">
                          {s}
                        </option>
                      ))}
                    </select>
                  </div>

                  {/* Seed Input */}
                  <div className="flex items-center justify-between gap-3 p-2.5 rounded-2xl bg-black/20 border border-white/5">
                    <div className="space-y-0.5">
                      <span className="text-xs font-medium text-muted-foreground block">Seed</span>
                      <span className="text-[10px] text-muted-foreground/60 block">-1 = Random</span>
                    </div>
                    <input
                      type="number"
                      value={seed}
                      onChange={(e) => setSeed(parseInt(e.target.value, 10) || -1)}
                      className="w-24 bg-black/50 border border-white/10 rounded-xl px-2.5 py-1 text-xs text-foreground font-mono text-right outline-none focus:border-primary/50"
                    />
                  </div>
                </div>

                {/* Aspect Ratio */}
                <div className="space-y-1.5 pt-1">
                  <span className="text-xs font-medium text-muted-foreground block">
                    Aspect Ratio
                  </span>
                  <div className="grid grid-cols-2 gap-1.5">
                    {ASPECT_RATIO_PRESETS.map((p) => {
                      const isActive = size === p.size || size === p.fallbackSize
                      return (
                        <button
                          key={p.label}
                          type="button"
                          onClick={() => setSize(p.size)}
                          className={cn(
                            "py-2 px-3 rounded-xl text-xs font-medium border text-center transition-all",
                            isActive
                              ? "bg-white/15 border-white/30 text-white font-semibold shadow-sm"
                              : "bg-black/20 border-white/5 text-muted-foreground hover:bg-white/5 hover:text-foreground"
                          )}
                        >
                          <div>{p.label}</div>
                          <div className="text-[10px] text-muted-foreground/70 font-mono mt-0.5">{p.size}</div>
                        </button>
                      )
                    })}
                  </div>
                </div>

                {/* Negative Prompt */}
                <div className="space-y-1.5 pt-1">
                  <span className="text-xs font-medium text-muted-foreground block">
                    Negative Prompt
                  </span>
                  <textarea
                    rows={2}
                    value={negativePrompt}
                    onChange={(e) => setNegativePrompt(e.target.value)}
                    placeholder="e.g. blurry, low quality, extra limbs"
                    className="w-full bg-black/30 border border-white/10 rounded-2xl p-3 text-xs text-foreground placeholder:text-muted-foreground/60 outline-none focus:border-primary/50 resize-none leading-relaxed"
                  />
                </div>
              </div>
            ) : (
              /* History View (Matching Audio Studio) */
              <div className="space-y-3">
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
                    onClick={() => loadChatsList(false)}
                    disabled={isLoadingChats}
                    className="p-2 rounded-full bg-white/5 hover:bg-white/10 text-muted-foreground hover:text-foreground transition-all"
                    title="Refresh history"
                  >
                    <RefreshCw size={13} className={cn(isLoadingChats && "animate-spin text-primary")} />
                  </button>
                </div>

                {/* New Chat Button */}
                <button
                  type="button"
                  onClick={handleNewChat}
                  className="w-full flex items-center justify-center gap-2 py-2.5 px-4 rounded-2xl bg-primary text-white text-xs font-bold hover:bg-primary/90 transition-all shadow-md shadow-primary/20"
                >
                  <Plus size={15} />
                  <span>New Image Chat</span>
                </button>

                {/* Header Actions */}
                <div className="flex items-center justify-between text-[11px] text-muted-foreground font-semibold pt-1">
                  <span>
                    {filteredChats.length > 0
                      ? `${filteredChats.length} chat session${filteredChats.length > 1 ? "s" : ""}`
                      : "No chats in range"}
                  </span>
                  {chats.length > 0 && (
                    <button
                      type="button"
                      onClick={handleClearAllHistory}
                      className="text-muted-foreground hover:text-rose-400 transition-colors"
                    >
                      Clear All
                    </button>
                  )}
                </div>

                {/* Chats List */}
                {isLoadingChats ? (
                  <div className="text-center py-12 text-xs text-muted-foreground space-y-2">
                    <Loader2 size={22} className="mx-auto animate-spin text-primary" />
                    <p>Loading image sessions...</p>
                  </div>
                ) : filteredChats.length === 0 ? (
                  <div className="text-center py-12 text-xs text-muted-foreground space-y-2">
                    <History size={26} className="mx-auto opacity-40" />
                    <p>No image chats found in this time range.</p>
                  </div>
                ) : (
                  <div className="space-y-3">
                    {filteredChats.map((c) => {
                      const isActive = c.id === activeChatId
                      return (
                        <div
                          key={c.id}
                          onClick={() => setActiveChatId(c.id)}
                          className={cn(
                            "p-3.5 rounded-2xl bg-black/30 border transition-all space-y-2 cursor-pointer group",
                            isActive
                              ? "border-primary/50 shadow-lg shadow-primary/5 bg-primary/[0.03]"
                              : "border-white/5 hover:border-white/15"
                          )}
                        >
                          {/* Avatar + Title + Badge + Timestamp */}
                          <div className="flex items-center justify-between text-xs">
                            <div className="flex items-center gap-2 truncate min-w-0">
                              <div className="w-5 h-5 rounded-full bg-gradient-to-tr from-rose-500 to-orange-500 text-white flex items-center justify-center text-[9px] font-bold shrink-0">
                                {c.title ? c.title.charAt(0).toUpperCase() : "I"}
                              </div>
                              <span className="font-semibold text-foreground truncate">{c.title}</span>
                            </div>
                            <span className="text-[10px] font-mono text-muted-foreground shrink-0 ml-2">
                              {formatDate(c.updated_at || c.created_at)}
                            </span>
                          </div>

                          {/* Latest Prompt preview */}
                          {c.latest_prompt && (
                            <p className="text-xs text-muted-foreground/90 line-clamp-2 leading-relaxed font-mono">
                              {c.latest_prompt}
                            </p>
                          )}

                          {/* Controls Bar: Render count badge + Actions (Copy prompt, Delete) */}
                          <div className="flex items-center justify-between pt-2 border-t border-white/5">
                            <div className="flex items-center gap-2">
                              <span className="text-[10px] px-2 py-0.5 rounded-full bg-white/5 border border-white/10 text-muted-foreground font-mono">
                                {c.message_count ? `${c.message_count} render${c.message_count > 1 ? "s" : ""}` : "Empty"}
                              </span>
                            </div>

                            <div className="flex items-center gap-1.5 text-muted-foreground">
                              {c.latest_prompt && (
                                <button
                                  type="button"
                                  onClick={(e) => {
                                    e.stopPropagation()
                                    navigator.clipboard.writeText(c.latest_prompt || "")
                                  }}
                                  className="p-1.5 rounded-lg hover:bg-white/10 hover:text-foreground transition-all"
                                  title="Copy Prompt"
                                >
                                  <Copy size={13} />
                                </button>
                              )}
                              <button
                                type="button"
                                onClick={(e) => handleDeleteChat(c.id, e)}
                                className="p-1.5 rounded-lg hover:bg-rose-500/20 hover:text-rose-400 transition-all"
                                title="Delete chat"
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

      {/* ── Fullscreen Lightbox Modal ────────────────────────────────────── */}
      {activeLightboxItem && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/90 backdrop-blur-xl animate-in fade-in duration-200 p-4 md:p-8"
          onClick={() => setActiveLightboxItem(null)}
        >
          <div
            className="relative max-w-5xl max-h-full flex flex-col items-center justify-center space-y-4"
            onClick={(e) => e.stopPropagation()}
          >
            <button
              type="button"
              onClick={() => setActiveLightboxItem(null)}
              className="absolute -top-12 right-0 p-2 rounded-full bg-white/10 hover:bg-white/20 text-white transition-colors"
            >
              <X size={18} />
            </button>

            <div className="max-h-[75vh] overflow-hidden rounded-2xl shadow-2xl border border-white/10 bg-black flex items-center justify-center">
              <img
                src={activeLightboxItem.image_url}
                alt={activeLightboxItem.prompt}
                className="max-h-[75vh] w-auto object-contain"
              />
            </div>

            <div className="w-full max-w-2xl bg-[#161616]/90 border border-white/10 rounded-2xl p-4 shadow-2xl flex items-center justify-between gap-4">
              <div className="min-w-0 flex-1 space-y-1">
                <p className="text-xs font-semibold text-foreground truncate">{activeLightboxItem.prompt}</p>
                <div className="flex items-center gap-2 text-[10px] text-muted-foreground font-mono">
                  <span>{activeLightboxItem.size}</span>
                  <span>&bull;</span>
                  <span>{activeLightboxItem.steps} steps</span>
                  <span>&bull;</span>
                  <span>CFG {activeLightboxItem.cfg_scale}</span>
                  <span>&bull;</span>
                  <span>Seed: {activeLightboxItem.seed}</span>
                </div>
              </div>

              <div className="flex items-center gap-2 shrink-0">
                <button
                  type="button"
                  onClick={() => handleCopyPrompt(activeLightboxItem.id, activeLightboxItem.prompt)}
                  className="px-3 py-1.5 rounded-xl bg-white/10 hover:bg-white/20 text-xs font-semibold text-white transition-colors flex items-center gap-1.5"
                >
                  {copiedId === activeLightboxItem.id ? <Check size={12} className="text-emerald-400" /> : <Copy size={12} />}
                  <span>Copy</span>
                </button>
                <button
                  type="button"
                  onClick={() => window.open(activeLightboxItem.image_url, "_blank", "noopener,noreferrer")}
                  className="px-3 py-1.5 rounded-xl bg-primary text-white hover:bg-primary/90 text-xs font-semibold transition-colors flex items-center gap-1.5 shadow-md shadow-primary/20"
                >
                  <Download size={12} />
                  <span>Download</span>
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
