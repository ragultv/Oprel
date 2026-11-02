"use client"

import { useState, useEffect, type ReactNode } from "react"
import { usePathname } from "next/navigation"
import { Sidebar } from "@/components/Sidebar"
import { SettingsModal } from "@/components/SettingsModal"
import { HfConnectModal } from "@/components/HfConnectModal"
import { HF } from "@/services/api"

export function AppShell({ children }: { children: ReactNode }) {
  const [showHfWelcome, setShowHfWelcome] = useState(false)
  const pathname = usePathname()
  const onWelcome = pathname?.startsWith("/welcome") ?? false

  useEffect(() => {
    // Only check if user has not seen or skipped the welcome prompt
    const seen = localStorage.getItem("oprel_hf_welcome_seen")
    if (!seen) {
      HF.getStatus()
        .then((status) => {
          if (!status.connected) {
            setShowHfWelcome(true)
          }
        })
        .catch(() => {})
    }
  }, [])

  const handleDismissWelcome = () => {
    localStorage.setItem("oprel_hf_welcome_seen", "1")
    setShowHfWelcome(false)
  }

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-background">
      <Sidebar />
      <main className="flex-1 min-w-0 overflow-hidden relative flex flex-col">
        {children}
      </main>
      <SettingsModal />
      <HfConnectModal
        isOpen={showHfWelcome && !onWelcome}
        onClose={handleDismissWelcome}
        onSkip={handleDismissWelcome}
        onConnected={handleDismissWelcome}
        variant="welcome"
      />
    </div>
  )
}