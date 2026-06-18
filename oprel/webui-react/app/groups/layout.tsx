import { GroupSidebar } from "@/components/GroupSidebar"
import type { ReactNode } from "react"
import { Suspense } from "react"

export default function GroupsLayout({ children }: { children: ReactNode }) {
  return (
    <div className="flex h-full w-full overflow-hidden bg-background">
      <Suspense fallback={<div className="w-64 border-r border-border bg-[#171717]"></div>}>
        <GroupSidebar />
      </Suspense>
      <div className="flex-1 min-w-0 flex flex-col relative h-full">
        {children}
      </div>
    </div>
  )
}
