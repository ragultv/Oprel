"use client"

import { useSearchParams } from "next/navigation"
import { GroupChatView } from "@/components/GroupChatView"
import { Users } from "lucide-react"
import { Suspense } from "react"

function GroupsPageContent() {
  const searchParams = useSearchParams()
  const groupId = searchParams.get("groupId")

  if (!groupId) {
    return (
      <div className="h-full flex flex-col items-center justify-center text-muted-foreground p-8 animate-in fade-in duration-500">
        <div className="w-16 h-16 rounded-2xl bg-secondary/50 flex items-center justify-center mb-6 border border-border">
          <Users size={32} className="opacity-50" />
        </div>
        <h2 className="text-xl font-semibold text-foreground tracking-tight mb-2">AI Groups Workspace</h2>
        <p className="text-sm max-w-md text-center opacity-80 leading-relaxed">
          Create a group and add multiple AI agents to collaborate, debate, and solve complex problems together.
        </p>
      </div>
    )
  }

  return <GroupChatView groupId={groupId} />
}

export default function GroupsPage() {
  return (
    <Suspense fallback={<div className="flex-1 flex items-center justify-center text-muted-foreground">Loading...</div>}>
      <GroupsPageContent />
    </Suspense>
  )
}
