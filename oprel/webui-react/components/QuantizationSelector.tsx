"use client"

import { useState, useEffect } from "react"
import { ChevronDown, HardDrive, Layers } from "lucide-react"
import { cn } from "@/services/utils"
import { API, type ModelDetailedInfo } from "@/services/api"

interface QuantizationSelectorProps {
  modelId: string
  onQuantizationChange?: (quant: string, size: number, parameters: string) => void
  onQualityChange?: (quality: string) => void
}

export function QuantizationSelector({ modelId, onQuantizationChange, onQualityChange }: QuantizationSelectorProps) {
  const [modelInfo, setModelInfo] = useState<ModelDetailedInfo | null>(null)
  const [selectedQuant, setSelectedQuant] = useState<string>("")
  const [isOpen, setIsOpen] = useState(false)
  const [loading, setLoading] = useState(false)
  const [localQuants, setLocalQuants] = useState<string[]>([])

  useEffect(() => {
    async function fetchInfo() {
      if (!modelId) return
      
      setLoading(true)
      try {
        const [info, localInfo] = await Promise.all([
          API.fetchModelInfo(modelId),
          API.fetchLocalQuantizations(modelId).catch(() => ({
            model_id: modelId,
            repo_id: modelId,
            local_quantizations: [],
            has_local: false
          }))
        ])
        
        setModelInfo(info)
        setLocalQuants(localInfo.local_quantizations)
        
        // Prefer local quantization if available
        let defaultQuant = info.default_quantization
        if (localInfo.local_quantizations.length > 0) {
          // Use first local quantization
          defaultQuant = localInfo.local_quantizations[0]
        }
        
        if (defaultQuant) {
          setSelectedQuant(defaultQuant)
          const size = info.sizes[defaultQuant] || 0
          onQuantizationChange?.(defaultQuant, size, info.parameters)
          onQualityChange?.(getQualityLabel(defaultQuant))
        } else if (info.quantizations.length > 0) {
          // Fallback to first quantization
          const firstQuant = info.quantizations[0]
          setSelectedQuant(firstQuant)
          const size = info.sizes[firstQuant] || 0
          onQuantizationChange?.(firstQuant, size, info.parameters)
          onQualityChange?.(getQualityLabel(firstQuant))
        }
      } catch (error) {
        console.error("Failed to fetch model info:", error)
      } finally {
        setLoading(false)
      }
    }

    fetchInfo()
  }, [modelId])

  const handleQuantChange = (quant: string) => {
    setSelectedQuant(quant)
    setIsOpen(false)
    const size = modelInfo?.sizes[quant] || 0
    onQuantizationChange?.(quant, size, modelInfo?.parameters || "Unknown")
    onQualityChange?.(getQualityLabel(quant))
  }

  if (loading) {
    return (
      <>
        <div className="bg-[#0f0f0f] border border-border rounded-lg p-3 flex items-center gap-3">
          <div className="w-8 h-8 rounded-md bg-secondary flex items-center justify-center shrink-0">
            <Layers size={15} className="text-muted-foreground" />
          </div>
          <div className="flex-1">
            <div className="text-[10px] text-muted-foreground font-medium uppercase tracking-wider">Parameters</div>
            <div className="h-4 w-16 bg-secondary animate-pulse rounded mt-1" />
          </div>
        </div>
        <div className="bg-[#0f0f0f] border border-border rounded-lg p-3 flex items-center gap-3">
          <div className="w-8 h-8 rounded-md bg-secondary flex items-center justify-center shrink-0">
            <HardDrive size={15} className="text-muted-foreground" />
          </div>
          <div className="flex-1">
            <div className="h-4 w-24 bg-secondary animate-pulse rounded" />
          </div>
        </div>
      </>
    )
  }

  if (!modelInfo) {
    return (
      <>
        <div className="bg-[#0f0f0f] border border-border rounded-lg p-3 flex items-center gap-3">
          <div className="w-8 h-8 rounded-md bg-secondary flex items-center justify-center shrink-0 text-muted-foreground">
            <Layers size={15} />
          </div>
          <div>
            <div className="text-[10px] text-muted-foreground font-medium uppercase tracking-wider">Parameters</div>
            <div className="text-sm font-bold text-foreground">Unknown</div>
          </div>
        </div>
        <div className="bg-[#0f0f0f] border border-border rounded-lg p-3 flex items-center gap-3">
          <div className="w-8 h-8 rounded-md bg-secondary flex items-center justify-center shrink-0 text-muted-foreground">
            <HardDrive size={15} />
          </div>
          <div>
            <div className="text-[10px] text-muted-foreground font-medium uppercase tracking-wider">File Size</div>
            <div className="text-sm font-bold text-foreground">—</div>
          </div>
        </div>
      </>
    )
  }

  const currentSize = selectedQuant ? modelInfo.sizes[selectedQuant] : 0

  return (
    <>
      {/* Parameters Card */}
      <div className="bg-[#0f0f0f] border border-border rounded-lg p-3 flex items-center gap-3">
        <div className="w-8 h-8 rounded-md bg-secondary flex items-center justify-center shrink-0 text-muted-foreground">
          <Layers size={15} />
        </div>
        <div>
          <div className="text-[10px] text-muted-foreground font-medium uppercase tracking-wider">Parameters</div>
          <div className="text-sm font-bold text-foreground">{modelInfo.parameters}</div>
        </div>
      </div>

      {/* File Size + Quantization Selector Card */}
      <div className="bg-[#0f0f0f] border border-border rounded-lg p-3 flex items-center gap-3 relative">
        <div className="w-8 h-8 rounded-md bg-secondary flex items-center justify-center shrink-0 text-muted-foreground">
          <HardDrive size={15} />
        </div>
        <div className="flex-1 flex items-center justify-between gap-3">
          {/* Left: File Size */}
          <div>
            <div className="text-[10px] text-muted-foreground font-medium uppercase tracking-wider">File Size</div>
            <div className="text-sm font-bold text-foreground">
              {currentSize > 0 ? `${currentSize.toFixed(2)} GB` : "—"}
            </div>
          </div>

          {/* Right: Quantization Dropdown */}
          {modelInfo.quantizations.length > 0 && (
            <div className="relative">
              <button
                onClick={() => setIsOpen(!isOpen)}
                className="flex items-center gap-2 px-3 py-1.5 rounded-md bg-secondary hover:bg-secondary/70 transition-all border border-border"
              >
                <span className="text-xs font-mono font-bold text-foreground">{selectedQuant || "Select"}</span>
                <ChevronDown
                  size={14}
                  className={cn(
                    "text-muted-foreground transition-transform",
                    isOpen && "rotate-180"
                  )}
                />
              </button>

              {/* Dropdown Menu */}
              {isOpen && (
                <div className="absolute top-full right-0 mt-2 bg-[#0f0f0f] border border-border rounded-lg shadow-xl z-50 min-w-[200px] max-h-64 overflow-y-auto">
                  {modelInfo.quantizations.map((quant) => {
                    const size = modelInfo.sizes[quant] || 0
                    const isSelected = quant === selectedQuant
                    const isLocal = localQuants.includes(quant)

                    return (
                      <button
                        key={quant}
                        onClick={() => handleQuantChange(quant)}
                        className={cn(
                          "w-full p-2.5 flex items-center justify-between hover:bg-secondary/50 transition-all border-b border-border/50 last:border-0 text-left",
                          isSelected && "bg-secondary"
                        )}
                      >
                        <div className="flex flex-col gap-0.5">
                          <div className="flex items-center gap-2">
                            <div className="text-xs font-mono font-bold text-foreground">{quant}</div>
                            {isLocal && (
                              <span className="px-1.5 py-0.5 rounded text-[9px] font-bold bg-green-500/10 text-green-400 border border-green-500/20">
                                LOCAL
                              </span>
                            )}
                          </div>
                          <div className="text-[10px] text-muted-foreground">
                            {getQuantDescription(quant)}
                          </div>
                        </div>
                        <div className="text-xs font-bold text-foreground ml-3">
                          {size.toFixed(1)} GB
                        </div>
                      </button>
                    )
                  })}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </>
  )
}

function getQuantDescription(quant: string): string {
  const descriptions: Record<string, string> = {
    "F32": "Full precision",
    "F16": "Half precision",
    "Q8_0": "Very high quality",
    "Q6_K": "High quality",
    "Q5_K_M": "Good quality",
    "Q5_K_S": "Good quality (small)",
    "Q4_K_M": "Balanced",
    "Q4_K_S": "Balanced (small)",
    "Q4_0": "Fast inference",
    "Q3_K_M": "Low quality",
    "Q2_K": "Very low quality",
  }
  return descriptions[quant] || "Standard"
}

function getQualityPercentage(quant: string): number {
  const percentages: Record<string, number> = {
    "F32": 100,
    "F16": 95,
    "Q8_0": 90,
    "Q6_K": 80,
    "Q5_K_M": 70,
    "Q5_K_S": 68,
    "Q4_K_M": 60,
    "Q4_K_S": 58,
    "Q4_0": 55,
    "Q3_K_M": 40,
    "Q2_K": 25,
  }
  return percentages[quant] || 50
}

function getQualityLabel(quant: string): string {
  const percentage = getQualityPercentage(quant)
  if (percentage >= 85) return "Excellent"
  if (percentage >= 70) return "Good"
  if (percentage >= 55) return "Fair"
  return "Low"
}

function getQualityColor(quant: string): string {
  const percentage = getQualityPercentage(quant)
  if (percentage >= 85) return "bg-green-500"
  if (percentage >= 70) return "bg-blue-500"
  if (percentage >= 55) return "bg-amber-500"
  return "bg-red-500"
}
