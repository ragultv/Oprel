"use client"

import { useEffect, useRef } from "react"
import { useSearchParams } from "next/navigation"
import { ChatView } from "@/components/ChatView"
import { useApp } from "@/services/context"

export function ChatRouteClient() {
  const searchParams = useSearchParams()
  const { activeConversationId, setActiveConversationId } = useApp()
  const urlConvId = searchParams.get("conversationId")
  const initialSyncDone = useRef(false)

  // 1. Initial URL synchronization on mount
  useEffect(() => {
    if (!initialSyncDone.current) {
      initialSyncDone.current = true
      if (urlConvId && urlConvId !== activeConversationId) {
        setActiveConversationId(urlConvId)
      }
    }
  }, [urlConvId, activeConversationId, setActiveConversationId])

  // 2. Handle browser Back/Forward navigation
  useEffect(() => {
    const handlePopState = () => {
      const params = new URLSearchParams(window.location.search)
      const convId = params.get("conversationId")
      setActiveConversationId(convId || null)
    }

    window.addEventListener("popstate", handlePopState)
    return () => window.removeEventListener("popstate", handlePopState)
  }, [setActiveConversationId])

  return (
    <ChatView />
  )
}
