"use client"

import { useEffect } from "react"
import { useSearchParams } from "next/navigation"
import { ChatView } from "@/components/ChatView"
import { useApp } from "@/services/context"

export function ChatRouteClient() {
  const searchParams = useSearchParams()
  const { activeConversationId, setActiveConversationId } = useApp()
  const conversationId = searchParams.get("conversationId")

  useEffect(() => {
    if (conversationId && conversationId !== activeConversationId) {
      setActiveConversationId(conversationId)
      return
    }

    if (!conversationId && activeConversationId) {
      setActiveConversationId(null)
    }
  }, [conversationId, activeConversationId, setActiveConversationId])

  return (
    <ChatView />
  )
}
