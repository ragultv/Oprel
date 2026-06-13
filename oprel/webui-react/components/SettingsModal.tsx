"use client"

import { useEffect } from "react"
import { useRouter } from "next/navigation"
import { useApp } from "@/services/context"

/**
 * SettingsModal — now a navigation shim.
 * When settingsOpen is true (triggered by the Settings button in any sidebar/header),
 * we navigate to the full /settings page instead of showing a dialog.
 */
export function SettingsModal() {
  const { settingsOpen, setSettingsOpen } = useApp()
  const router = useRouter()

  useEffect(() => {
    if (settingsOpen) {
      setSettingsOpen(false)
      router.push("/settings")
    }
  }, [settingsOpen, setSettingsOpen, router])

  // Nothing to render — navigation is handled above
  return null
}
