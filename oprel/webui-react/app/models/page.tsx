"use client"

import { Suspense } from "react"
import { ModelsView } from "@/components/ModelsView"

export default function ModelsPage() {
  return (
    <Suspense fallback={<div className="h-full flex items-center justify-center bg-[#0e0e0e] text-muted-foreground">Loading models...</div>}>
      <ModelsView />
    </Suspense>
  )
}
