"use client"

import { useEffect, useRef } from "react"
import { useRouter } from "next/navigation"
import { ChatView } from "@/components/ChatView"
import { useApp } from "@/services/context"

export default function NewChatPage() {
  const router = useRouter()
  const didInit = useRef(false)
  const { createConversation } = useApp()

  useEffect(() => {
    if (didInit.current) return
    didInit.current = true
    const newId = createConversation()
    router.replace(`/chat?conversationId=${newId}`)
  }, [createConversation, router])

  return <ChatView />
}