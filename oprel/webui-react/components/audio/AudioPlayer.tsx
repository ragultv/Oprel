"use client"

import React, { useState, useRef, useEffect } from "react"
import { Play, Pause, Download } from "lucide-react"
import { cn } from "@/services/utils"

interface AudioPlayerProps {
  audioUrl: string
  onDownload?: () => void
  duration?: number
  title?: string
  subtitle?: string
  className?: string
}

// Waveform bar heights matching realistic speech pattern in user reference
const WAVEFORM_BARS = [
  10, 14, 22, 16, 28, 22, 12, 18, 26, 30, 24, 16, 12, 22, 28, 26,
  18, 14, 24, 30, 22, 12, 18, 26, 32, 22, 16, 10, 18, 24, 20, 14,
  22, 28, 18, 12, 16, 24, 20, 14, 8,
]

export function AudioPlayer({ audioUrl, onDownload, duration, title, subtitle, className }: AudioPlayerProps) {
  const audioRef = useRef<HTMLAudioElement | null>(null)
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [totalDuration, setTotalDuration] = useState(duration || 0)

  useEffect(() => {
    const audio = audioRef.current
    if (!audio) return

    const handleLoadedMetadata = () => {
      if (audio.duration && !isNaN(audio.duration)) {
        setTotalDuration(audio.duration)
      }
    }

    const handleTimeUpdate = () => {
      setCurrentTime(audio.currentTime)
    }

    const handleEnded = () => {
      setIsPlaying(false)
      setCurrentTime(0)
    }

    audio.addEventListener("loadedmetadata", handleLoadedMetadata)
    audio.addEventListener("timeupdate", handleTimeUpdate)
    audio.addEventListener("ended", handleEnded)

    return () => {
      audio.removeEventListener("loadedmetadata", handleLoadedMetadata)
      audio.removeEventListener("timeupdate", handleTimeUpdate)
      audio.removeEventListener("ended", handleEnded)
    }
  }, [audioUrl])

  const togglePlay = () => {
    if (!audioRef.current) return
    if (isPlaying) {
      audioRef.current.pause()
      setIsPlaying(false)
    } else {
      audioRef.current.play().then(() => setIsPlaying(true)).catch(console.error)
    }
  }

  const handleSeekIndex = (index: number) => {
    if (!audioRef.current || totalDuration <= 0) return
    const pct = index / WAVEFORM_BARS.length
    const newTime = pct * totalDuration
    audioRef.current.currentTime = newTime
    setCurrentTime(newTime)
  }

  const formatTime = (seconds: number) => {
    if (isNaN(seconds) || seconds <= 0) return "0:00"
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    return `${mins}:${secs < 10 ? "0" : ""}${secs}`
  }

  const progressPercent = totalDuration > 0 ? (currentTime / totalDuration) * 100 : 0

  return (
    <div
      className={cn(
        "inline-flex items-center gap-3 sm:gap-4 px-3 sm:px-4 py-2 sm:py-2.5 rounded-full bg-[#1c1c1c]/90 border border-white/10 shadow-xl backdrop-blur-md transition-all",
        className
      )}
    >
      <audio ref={audioRef} src={audioUrl} preload="auto" autoPlay />

      {/* Circular Play / Pause Button (Matching Reference Image) */}
      <button
        onClick={togglePlay}
        className="w-8 h-8 sm:w-9 sm:h-9 rounded-full bg-white text-black flex items-center justify-center hover:bg-white/90 transition-all shrink-0 shadow-md active:scale-95"
        title={isPlaying ? "Pause" : "Play"}
      >
        {isPlaying ? (
          <Pause size={14} className="fill-current text-black" />
        ) : (
          <Play size={14} className="fill-current text-black ml-0.5" />
        )}
      </button>

      {/* Waveform Bars (Matching Reference Image) */}
      <div
        className="flex items-center gap-[2px] sm:gap-[3px] h-8 cursor-pointer select-none px-1"
        title="Click to seek"
      >
        {WAVEFORM_BARS.map((height, i) => {
          const barProgress = (i / WAVEFORM_BARS.length) * 100
          const isPlayed = barProgress <= progressPercent
          return (
            <div
              key={i}
              onClick={() => handleSeekIndex(i)}
              style={{ height: `${height}px` }}
              className={cn(
                "w-[2px] sm:w-[2.5px] rounded-full transition-colors duration-150 hover:opacity-80",
                isPlayed ? "bg-white" : "bg-white/20"
              )}
            />
          )
        })}
      </div>

      {/* Duration Timer (Matching Reference Image: 0:00 / 0:09) */}
      <div className="text-[11px] sm:text-xs font-mono text-muted-foreground whitespace-nowrap select-none">
        <span className="text-foreground font-semibold">{formatTime(currentTime)}</span>
        <span className="opacity-60"> / </span>
        <span>{formatTime(totalDuration)}</span>
      </div>

      {/* Download Icon Button (Matching Reference Image) */}
      {onDownload && (
        <button
          onClick={onDownload}
          className="p-1.5 rounded-full hover:bg-white/10 text-muted-foreground hover:text-foreground transition-all shrink-0 active:scale-95"
          title="Download WAV audio"
        >
          <Download size={15} />
        </button>
      )}
    </div>
  )
}

