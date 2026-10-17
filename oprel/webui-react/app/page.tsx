"use client"

import { useEffect } from "react"
import { useRouter } from "next/navigation"
import { API } from "@/services/api"
import { ONBOARDED_KEY } from "@/components/WelcomeView"

export default function HomePage() {
  const router = useRouter()

  useEffect(() => {
    let cancelled = false
    const go = (path: string) => {
      if (!cancelled) router.replace(path)
    }

    let seen = false
    try {
      seen = localStorage.getItem(ONBOARDED_KEY) === "1"
    } catch {
      /* storage blocked: rely on the server's answer */
    }
    if (seen) {
      go("/new-chat")
      return
    }

    // The server knows whether this install has any chats, models, providers or an HF account.
    // Anyone who has used the app before goes straight to chat; if we cannot tell, assume they have.
    API.getOnboarding()
      .then((info) => go(info.first_run ? "/welcome" : "/new-chat"))
      .catch(() => go("/new-chat"))

    return () => {
      cancelled = true
    }
  }, [router])

  return null
}
