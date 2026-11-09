"use client"

import { useState, useEffect, type ReactNode } from "react"
import { usePathname } from "next/navigation"
import { Sidebar } from "@/components/Sidebar"
import { SettingsModal } from "@/components/SettingsModal"
import { HfConnectModal } from "@/components/HfConnectModal"
import { HF } from "@/services/api"
import { cn } from "@/services/utils"

export function AppShell({ children }: { children: ReactNode }) {
  const [showHfModal, setShowHfModal] = useState(false)
  const [hfVariant, setHfVariant] = useState<"welcome" | "download_nudge" | "expired" | "missing">("missing")
  const pathname = usePathname()
  const onWelcome = pathname?.startsWith("/welcome") ?? false

  // ── Resizable navigation sidebar ───────────────────────────────────────
  const [sidebarWidth, setSidebarWidth] = useState<number>(() => {
    if (typeof window !== "undefined") {
      const saved = localStorage.getItem("oprel_sidebar_width")
      if (saved) {
        const parsed = parseInt(saved, 10)
        if (!isNaN(parsed) && parsed >= 200 && parsed <= 500) return parsed
      }
    }
    return 260
  })

  const [isResizingSidebar, setIsResizingSidebar] = useState(false)

  const handleStartResize = (e: React.MouseEvent) => {
    e.preventDefault()
    setIsResizingSidebar(true)

    const startX = e.clientX
    const initialWidth = sidebarWidth

    const handleMouseMove = (moveEvent: MouseEvent) => {
      const delta = moveEvent.clientX - startX
      const newWidth = Math.min(500, Math.max(200, Math.round(initialWidth + delta)))
      setSidebarWidth(newWidth)
      try {
        localStorage.setItem("oprel_sidebar_width", String(newWidth))
      } catch {}
    }

    const handleMouseUp = () => {
      setIsResizingSidebar(false)
      window.removeEventListener("mousemove", handleMouseMove)
      window.removeEventListener("mouseup", handleMouseUp)
    }

    window.addEventListener("mousemove", handleMouseMove)
    window.addEventListener("mouseup", handleMouseUp)
  }

  useEffect(() => {
    // Every time user enters Oprel, check Hugging Face token
    HF.getStatus()
      .then((status) => {
        if (status.status === "valid" && status.connected) {
          // Token is present and verified: leave it alone!
          setShowHfModal(false)
          return
        }
        if (status.status === "expired") {
          // Token expired: prompt user with expired status
          setHfVariant("expired")
          setShowHfModal(true)
        } else if (status.status === "missing" || !status.connected) {
          // Token missing: prompt user to add token
          setHfVariant("missing")
          setShowHfModal(true)
        }
      })
      .catch(() => {})
  }, [])

  const handleDismissHf = () => {
    setShowHfModal(false)
  }

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-background">
      {/* Global drag shield overlay while resizing sidebar */}
      {isResizingSidebar && (
        <div className="fixed inset-0 z-[9999] cursor-col-resize select-none bg-transparent" />
      )}

      <Sidebar width={sidebarWidth} />

      {/* Vertical Resizer Divider */}
      <div
        onMouseDown={handleStartResize}
        onDoubleClick={() => {
          setSidebarWidth(260)
          try {
            localStorage.setItem("oprel_sidebar_width", "260")
          } catch {}
        }}
        className="w-2 shrink-0 cursor-col-resize select-none relative group/sidebar-resizer flex items-center justify-center transition-colors hover:bg-primary/20 z-10"
        title="Drag to resize sidebar (Double click to reset to 260px)"
      >
        <div
          className={cn(
            "w-[1.5px] h-full transition-colors",
            isResizingSidebar
              ? "bg-primary shadow-[0_0_8px_rgba(var(--primary),0.6)]"
              : "bg-border/60 group-hover/sidebar-resizer:bg-primary/70"
          )}
        />
        <div
          className={cn(
            "absolute w-1.5 h-7 rounded-full transition-colors flex flex-col items-center justify-center gap-1",
            isResizingSidebar
              ? "bg-primary"
              : "bg-muted-foreground/30 group-hover/sidebar-resizer:bg-primary"
          )}
        >
          <div className="w-0.5 h-0.5 rounded-full bg-white/70" />
          <div className="w-0.5 h-0.5 rounded-full bg-white/70" />
          <div className="w-0.5 h-0.5 rounded-full bg-white/70" />
        </div>
      </div>

      <main className="flex-1 min-w-0 overflow-hidden relative flex flex-col">
        {children}
      </main>
      <SettingsModal />
      <HfConnectModal
        isOpen={showHfModal && !onWelcome}
        onClose={handleDismissHf}
        onSkip={handleDismissHf}
        onConnected={handleDismissHf}
        variant={hfVariant}
      />
    </div>
  )
}
