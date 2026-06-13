import { Suspense } from "react"
import { ChatRouteClient } from "@/components/ChatRouteClient"

export default function ChatPage() {
  return (
    <Suspense fallback={null}>
      <ChatRouteClient />
    </Suspense>
  )
}
