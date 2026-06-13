"use client"

import type { ReactNode } from "react"
import { Sidebar } from "@/components/Sidebar"
import { SettingsModal } from "@/components/SettingsModal"

export function AppShell({ children }: { children: ReactNode }) {
  return (
    <div className="flex h-screen w-screen overflow-hidden bg-background">
      <Sidebar/>
      <main className="flex-1 min-w-0 overflow-hidden relative flex flex-col">
        {children}
      </main>
      <SettingsModal />
    </div>
  )
}