"use client"

import type { ReactNode } from "react"
import { usePathname } from "next/navigation"
import { Sidebar } from "@/components/Sidebar"
import { SettingsModal } from "@/components/SettingsModal"

export function AppShell({ children }: { children: ReactNode }) {
  const pathname = usePathname();
  const hideSidebar = pathname === '/connectors';

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-background">
      {!hideSidebar && <Sidebar/>}
      <main className="flex-1 min-w-0 overflow-hidden relative flex flex-col">
        {children}
      </main>
      <SettingsModal />
    </div>
  )
}