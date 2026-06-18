"use client"

import { useState, useEffect, useRef } from "react"
import {
  MessageSquarePlus,
  Box,
  BarChart2,
  Settings,
  Search,
  Trash2,
  ChevronRight,
  Cpu,
  Bot,
  Download,
  Database,
  Image,
  RefreshCw,
  ScanText,
  Users,
} from "lucide-react"
import { usePathname, useRouter } from "next/navigation"
import { cn } from "@/services/utils"
import { useApp } from "@/services/context"
import { useDownloads } from "@/services/downloadContext"
import { API } from "@/services/api"
import type { Conversation } from "@/services/data"

function groupConversations(convs: Conversation[]) {
  const now = new Date()
  const today: Conversation[] = []
  const yesterday: Conversation[] = []
  const older: Conversation[] = []

  // Sort by last-message time (updatedAt preferred, fallback createdAt) — newest first
  const sorted = [...convs].sort((a, b) => {
    const aTime = (a.updatedAt || a.createdAt).getTime()
    const bTime = (b.updatedAt || b.createdAt).getTime()
    return bTime - aTime
  })

  sorted.forEach((c) => {
    const refTime = (c.updatedAt || c.createdAt).getTime()
    const diff = now.getTime() - refTime
    const dayMs = 86400000
    if (diff < dayMs) today.push(c)
    else if (diff < dayMs * 2) yesterday.push(c)
    else older.push(c)
  })

  return { today, yesterday, older }
}

function BackgroundTasksIndicator({ tasks }: { tasks: Array<{ id: string; label: string }> }) {
  const [visible, setVisible] = useState(false);
  const [activeTasks, setActiveTasks] = useState(tasks);
  const timerRef = useRef<NodeJS.Timeout | null>(null);
  const mountTimeRef = useRef<number | null>(null);

  useEffect(() => {
    if (tasks.length > 0) {
      if (timerRef.current) clearTimeout(timerRef.current);
      setActiveTasks(tasks);
      setVisible(true);
      if (!mountTimeRef.current) {
        mountTimeRef.current = Date.now();
      }
    } else {
      // Delay hiding to prevent flicker
      const elapsed = mountTimeRef.current ? Date.now() - mountTimeRef.current : 0;
      const minDuration = 1000; // Keep it visible for at least 1 second
      const delay = Math.max(0, minDuration - elapsed);

      timerRef.current = setTimeout(() => {
        setVisible(false);
        mountTimeRef.current = null;
      }, delay);
    }

    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [tasks]);

  if (!visible) return null;

  return (
    <div className="px-3 py-2 rounded-xl bg-[#222] border border-border/80 text-[10px] text-muted-foreground flex items-center gap-2.5 mb-2 shadow-inner select-none animate-in fade-in slide-in-from-bottom-2 duration-200">
      <RefreshCw size={11} className="animate-spin text-primary shrink-0" />
      <div className="flex-1 min-w-0">
        <span className="font-bold text-foreground block leading-tight">
          Background Tasks ({activeTasks.length})
        </span>
        <span className="text-[9px] text-muted-foreground truncate block mt-0.5">
          {activeTasks[activeTasks.length - 1]?.label || "Processing"}...
        </span>
      </div>
    </div>
  );
}

export function Sidebar() {
  const {
    conversations,
    activeConversationId,
    setActiveConversationId,
    createConversation,
    deleteConversation,
    settingsOpen,
    setSettingsOpen,
    activeModelId,
    models,
    backgroundTasks,
  } = useApp()

  const { getOngoingCount, setDialogOpen } = useDownloads()

  const pathname = usePathname()
  const router = useRouter()
  const isChatRoute = pathname.startsWith("/chat") || pathname === "/new-chat" || pathname === "/"
  const canUpdateChatUrlInPlace = pathname.startsWith("/chat")
  const isModelsRoute = pathname.startsWith("/models")
  const isImagesRoute = pathname.startsWith("/images")
  const isOcrRoute = pathname.startsWith("/ocr")
  const isKnowledgeRoute = pathname.startsWith("/knowledge")
  const isDevRoute = pathname.startsWith("/dev")
  const isGroupsRoute = pathname.startsWith("/groups")

  const [search, setSearch] = useState("")
  const [deleteId, setDeleteId] = useState<string | null>(null)

  const activeModel = models.find((m) => m.id === activeModelId)
  const filtered = conversations.filter((c) =>
    c.title.toLowerCase().includes(search.toLowerCase())
  )
  const groups = groupConversations(filtered)

  const openConversation = (conversationId: string) => {
    setActiveConversationId(conversationId)

    if (canUpdateChatUrlInPlace) {
      const basePath = window.location.pathname.startsWith('/gui') ? '/gui' : ''
      window.history.pushState(null, "", `${basePath}/chat?conversationId=${conversationId}`)
      return
    }

    router.push(`/chat?conversationId=${conversationId}`)
  }

  const startConversation = async () => {
    const newId = await createConversation()
    setActiveConversationId(newId)

    if (canUpdateChatUrlInPlace) {
      const basePath = window.location.pathname.startsWith('/gui') ? '/gui' : ''
      window.history.pushState(null, "", `${basePath}/chat?conversationId=${newId}`)
      return
    }

    router.push(`/chat?conversationId=${newId}`)
  }

  function ConvGroup({ label, items }: { label: string; items: Conversation[] }) {
    if (!items.length) return null
    return (
      <div>
        <div className="px-3 mb-1 text-[10px] font-bold uppercase tracking-widest text-muted-foreground">
          {label}
        </div>
        <div className="space-y-0.5">
          {items.map((conv) => (
            <div
              key={conv.id}
              onClick={() => openConversation(conv.id)}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault()
                  openConversation(conv.id)
                }
              }}
              role="button"
              tabIndex={0}
              className={cn(
                "group flex w-full items-center gap-2 px-3 py-2 rounded-lg transition-all text-sm text-left",
                activeConversationId === conv.id && pathname.startsWith("/chat")
                  ? "bg-secondary text-foreground"
                  : "text-muted-foreground hover:bg-secondary/60 hover:text-foreground"
              )}
            >
              <MessageSquarePlus size={13} className="shrink-0 opacity-60" />
              <span className="flex-1 truncate text-xs">{conv.title}</span>
              <button
                onClick={(e) => {
                  e.preventDefault()
                  e.stopPropagation()
                  setDeleteId(conv.id)
                }}
                className="opacity-0 group-hover:opacity-100 p-0.5 rounded hover:text-destructive transition-all"
              >
                <Trash2 size={12} />
              </button>
            </div>
          ))}
        </div>
      </div>
    )
  }

  const [user, setUser] = useState<{ name: string; role: string; initials: string }>({
    name: "User",
    role: "Developer",
    initials: "U",
  })

  useEffect(() => {
    API.fetchUser()
      .then((data) => {
        const initials = data.name
          .split(" ")
          .map((n) => n[0])
          .join("")
          .toUpperCase()
        setUser({ ...data, initials })
      })
      .catch((err) => console.error("Failed to fetch user:", err))
  }, [])

  return (
    <>
      <aside className="w-65 shrink-0 flex flex-col h-full bg-[#171717] border-r border-border overflow-hidden transition-all duration-300">
        {/* Logo */}
        <div className="p-4 pb-3 flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg overflow-hidden shrink-0">
            <img src="/gui/logo1.png" alt="Oprel" className="w-full h-full object-cover" />
          </div>
          <span className="font-bold text-sm tracking-tight text-foreground">OPREL STUDIO</span>
        </div>

        {/* New Chat Button */}
        <div className="px-3 pb-3 space-y-2">
          <button
            onClick={startConversation}
            className="w-full flex items-center justify-between px-4 py-2.5 rounded-lg bg-primary text-primary-foreground text-sm font-semibold hover:bg-primary/90 transition-all"
          >
            <span>New Chat</span>
            <MessageSquarePlus size={16} />
          </button>

          {/* Nav buttons — 2 per row */}
          <div className="grid grid-cols-2 gap-2">
            <button
              onClick={() => router.push("/models")}
              className={cn(
                "flex flex-col items-center gap-1.5 py-3 rounded-lg border border-border text-xs font-semibold transition-all text-center",
                isModelsRoute
                  ? "bg-secondary text-foreground border-border"
                  : "text-muted-foreground hover:bg-secondary/60 hover:text-foreground"
              )}
            >
              <Box size={18} />
              <span className="text-[10px]">Models</span>
            </button>
            <button
              onClick={() => router.push("/images")}
              className={cn(
                "flex flex-col items-center gap-1.5 py-3 rounded-lg border border-border text-xs font-semibold transition-all text-center",
                isImagesRoute
                  ? "bg-secondary text-foreground border-border"
                  : "text-muted-foreground hover:bg-secondary/60 hover:text-foreground"
              )}
            >
              <Image size={18} />
              <span className="text-[10px]">Images</span>
            </button>
            <button
              onClick={() => router.push("/ocr")}
              className={cn(
                "flex flex-col items-center gap-1.5 py-3 rounded-lg border border-border text-xs font-semibold transition-all text-center",
                isOcrRoute
                  ? "bg-secondary text-foreground border-border"
                  : "text-muted-foreground hover:bg-secondary/60 hover:text-foreground"
              )}
            >
              <ScanText size={18} />
              <span className="text-[10px]">OCR</span>
            </button>
            <button
              onClick={() => router.push("/knowledge")}
              className={cn(
                "flex flex-col items-center gap-1.5 py-3 rounded-lg border border-border text-xs font-semibold transition-all text-center",
                isKnowledgeRoute
                  ? "bg-secondary text-foreground border-border"
                  : "text-muted-foreground hover:bg-secondary/60 hover:text-foreground"
              )}
            >
              <Database size={18} />
              <span className="text-[10px]">Knowledge</span>
            </button>
            <button
              onClick={() => router.push("/groups")}
              className={cn(
                "flex flex-col items-center gap-1.5 py-3 rounded-lg border border-border text-xs font-semibold transition-all text-center",
                isGroupsRoute
                  ? "bg-secondary text-foreground border-border"
                  : "text-muted-foreground hover:bg-secondary/60 hover:text-foreground"
              )}
            >
              <Users size={18} />
              <span className="text-[10px]">Groups</span>
            </button>
            <button
              onClick={() => router.push("/dev")}
              className={cn(
                "flex flex-col items-center gap-1.5 py-3 rounded-lg border border-border text-xs font-semibold transition-all text-center",
                isDevRoute
                  ? "bg-secondary text-foreground border-border"
                  : "text-muted-foreground hover:bg-secondary/60 hover:text-foreground"
              )}
            >
              <BarChart2 size={18} />
              <span className="text-[10px]">Dev</span>
            </button>
          </div>
        </div>

        {/* Conversation List */}
        <div className="flex-1 overflow-y-auto px-2 py-2 space-y-4">
          <ConvGroup label="Today" items={groups.today} />
          <ConvGroup label="Yesterday" items={groups.yesterday} />
          <ConvGroup label="Older" items={groups.older} />
          {filtered.length === 0 && (
            <div className="text-center py-8 text-muted-foreground text-xs">No chats found</div>
          )}
        </div>

        {/* Footer */}
        <div className="border-t border-border p-3 space-y-3">
          {/* Search */}
          <div className="relative">
            <Search size={13} className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground" />
            <input
              type="text"
              placeholder="Search chats..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="w-full bg-[#1e1e1e] border border-border rounded-lg py-2 pl-8 pr-3 text-xs text-foreground placeholder:text-muted-foreground focus:border-primary/50 transition-all"
            />
          </div>

          {/* Active model pill */}
          {/* <div className="flex items-center gap-2 px-2 py-2 rounded-lg bg-secondary/50">
            <div className="w-7 h-7 rounded-md bg-primary/10 border border-primary/20 flex items-center justify-center shrink-0">
              <Cpu size={13} className="text-primary" />
            </div>
            <div className="flex-1 min-w-0">
              <div className="text-[11px] font-semibold text-foreground truncate">
                {activeModel?.name.split(" ").slice(0, 3).join(" ")}
              </div>
              <div className="text-[10px] text-muted-foreground">
                {activeModel?.size} · {activeModel?.quantization}
              </div>
            </div>
            <div className="w-2 h-2 rounded-full bg-green-500 pulse-dot shrink-0" />
          </div> */}

          {/* Background Tasks */}
          <BackgroundTasksIndicator tasks={backgroundTasks} />

          {/* User / Settings */}
          <div className="flex items-center gap-2 px-1">
            <div className="w-7 h-7 rounded-lg bg-primary/20 flex items-center justify-center text-[10px] font-bold text-primary shrink-0">
              {user.initials}
            </div>
            <div className="flex-1 min-w-0">
              <div className="text-[11px] font-semibold text-foreground">{user.name}</div>
              <div className="text-[10px] text-muted-foreground">{user.role}</div>
            </div>
            <button
              onClick={() => setDialogOpen(true)}
              className="relative p-1.5 rounded-md text-muted-foreground hover:text-foreground hover:bg-secondary transition-all"
            >
              <Download size={15} />
              {getOngoingCount() > 0 && (
                <span className="absolute -top-0.5 -right-0.5 w-4 h-4 bg-primary text-[9px] font-bold text-primary-foreground rounded-full flex items-center justify-center border border-[#171717]">
                  {getOngoingCount()}
                </span>
              )}
            </button>
            <button
              onClick={() => setSettingsOpen(true)}
              className="p-1.5 rounded-md text-muted-foreground hover:text-foreground hover:bg-secondary transition-all"
            >
              <Settings size={15} />
            </button>
          </div>
        </div>
      </aside>

      {/* Delete confirm dialog */}
      {deleteId && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
          <div className="bg-[#1e1e1e] border border-border rounded-xl p-6 w-90 shadow-2xl animate-fade-in-up">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-9 h-9 rounded-lg bg-destructive/10 flex items-center justify-center">
                <Trash2 size={16} className="text-destructive" />
              </div>
              <div>
                <h3 className="text-sm font-bold text-foreground">Delete Chat</h3>
                <p className="text-xs text-muted-foreground">This action cannot be undone</p>
              </div>
            </div>
            <div className="flex gap-2 justify-end mt-5">
              <button
                onClick={() => setDeleteId(null)}
                className="px-4 py-2 text-xs font-semibold rounded-lg bg-secondary text-foreground hover:bg-secondary/80 transition-all"
              >
                Cancel
              </button>
              <button
                onClick={() => {
                  deleteConversation(deleteId)
                  setDeleteId(null)
                }}
                className="px-4 py-2 text-xs font-semibold rounded-lg bg-destructive text-white hover:bg-destructive/90 transition-all flex items-center gap-1.5"
              >
                <Trash2 size={12} /> Delete
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  )
}
