"use client"

import React, { useState, useEffect, useRef } from "react"
import {
  Mic,
  Volume2,
  Sparkles,
  Download,
  Play,
  Pause,
  RotateCcw,
  Check,
  ChevronDown,
  Upload,
  Copy,
  Loader2,
  ArrowRight,
  ShieldCheck,
  Cpu,
  Layers,
  FileAudio,
  Zap,
  Radio,
  Clock,
  Trash2,
  RefreshCw,
  Sliders,
  AudioLines,
  Settings,
  History as HistoryIcon,
  FileText,
  CheckCheck,
} from "lucide-react"
import { AudioAPI, SpeechModel, AudioHistoryItem } from "@/services/api"
import { AudioPlayer } from "@/components/audio/AudioPlayer"
import { useToast } from "@/components/ui/use-toast"
import { cn } from "@/services/utils"

interface AudioGenerationHistoryItem {
  id: string
  modelId: string
  modelName: string
  voiceId: string
  voiceName: string
  text: string
  audioUrl: string
  timestamp: number
  speed: number
}

const DEFAULT_SCRIPT =
  'She said, "I\'m fine. Really. Go on without me."'

const INITIAL_SPEECH_MODELS: SpeechModel[] = [
  {
    id: "kokoro-82m",
    name: "Kokoro-82M",
    type: "tts",
    engine: "kokoro",
    description: "Ultra-fast speech synthesis with CUDA GPU acceleration and 100,000 char capacity",
    tags_supported: false,
    acceleration: "CUDA GPU",
    is_downloaded: true,
    size: "350 MB",
    hf_repo: "hexgrad/Kokoro-82M",
    voices: [
      { id: "af_heart", name: "Heart (American Female)", gender: "female", avatar: "female" },
      { id: "af_sarah", name: "Sarah (American Female)", gender: "female", avatar: "female" },
      { id: "af_bella", name: "Bella (American Female)", gender: "female", avatar: "female" },
      { id: "am_adam", name: "Adam (American Male)", gender: "male", avatar: "male" },
      { id: "am_michael", name: "Michael (American Male)", gender: "male", avatar: "male" },
      { id: "bf_emma", name: "Emma (British Female)", gender: "female", avatar: "female" },
      { id: "bm_george", name: "George (British Male)", gender: "male", avatar: "male" },
    ],
  },
]

export default function AudioStudioPage() {
  const { toast } = useToast()

  // Navigation tabs (TTS, STT, Cloning)
  const [activeTab, setActiveTab] = useState<"tts" | "stt" | "cloning">("tts")

  // Sidebar Tab: Settings | History
  const [sidebarTab, setSidebarTab] = useState<"settings" | "history">("settings")

  // Speech Models & Engine (Pre-seeded for instant 0ms mount)
  const [models, setModels] = useState<SpeechModel[]>(INITIAL_SPEECH_MODELS)
  const [isLoadingModels, setIsLoadingModels] = useState(false)
  const [selectedEngine, setSelectedEngine] = useState<string>("kokoro-82m")
  const [downloadingJob, setDownloadingJob] = useState<{ id: string; modelId: string; progress: number; message: string } | null>(null)

  // Dropdown states
  const [isVoiceDropdownOpen, setIsVoiceDropdownOpen] = useState(false)
  const [isModelDropdownOpen, setIsModelDropdownOpen] = useState(false)

  // Voice & Controls State
  const [selectedVoice, setSelectedVoice] = useState<string>("af_heart")
  const [volume, setVolume] = useState<number>(100)
  const [speechSpeed, setSpeechSpeed] = useState<number>(1.0)
  const [loudnessNorm, setLoudnessNorm] = useState<boolean>(true)
  const [textNorm, setTextNorm] = useState<boolean>(true)

  // TTS Input
  const [inputText, setInputText] = useState(DEFAULT_SCRIPT)

  // TTS Generation & Playback
  const [isGenerating, setIsGenerating] = useState(false)
  const [generatedAudioUrl, setGeneratedAudioUrl] = useState<string | null>(null)
  const [activeSpeechJob, setActiveSpeechJob] = useState<{
    id: string
    progress: number
    message: string
    current_chunk: number
    total_chunks: number
  } | null>(null)

  // Persistent Database Audio History State
  const [historyItems, setHistoryItems] = useState<AudioHistoryItem[]>([])
  const [isLoadingHistory, setIsLoadingHistory] = useState(false)
  const [daysFilter, setDaysFilter] = useState<number>(30)
  const [isDaysDropdownOpen, setIsDaysDropdownOpen] = useState(false)
  const [playingHistoryId, setPlayingHistoryId] = useState<string | null>(null)
  const [historySpeeds, setHistorySpeeds] = useState<Record<string, number>>({})
  const historyAudioRef = useRef<HTMLAudioElement | null>(null)

  // Load history from SQLite database
  const loadDbHistory = async (days: number = daysFilter) => {
    setIsLoadingHistory(true)
    try {
      const items = await AudioAPI.getHistory(days)
      setHistoryItems(items || [])
    } catch (err) {
      console.warn("Failed to load audio history:", err)
    } finally {
      setIsLoadingHistory(false)
    }
  }

  useEffect(() => {
    loadDbHistory(daysFilter)
  }, [daysFilter])

  // In-history audio playback toggle
  const handleTogglePlayHistory = (item: AudioHistoryItem) => {
    if (!historyAudioRef.current) return
    if (playingHistoryId === item.id) {
      historyAudioRef.current.pause()
      setPlayingHistoryId(null)
    } else {
      const audioUrl = AudioAPI.getHistoryAudioUrl(item.id)
      historyAudioRef.current.src = audioUrl
      const speed = historySpeeds[item.id] || item.speed || 1.0
      historyAudioRef.current.playbackRate = speed
      historyAudioRef.current
        .play()
        .then(() => setPlayingHistoryId(item.id))
        .catch((err) => {
          console.error("Playback error:", err)
          setPlayingHistoryId(null)
        })
    }
  }

  // In-history speed adjustment (1x, 1.25x, 1.5x, 2x)
  const cycleHistorySpeed = (itemId: string, currentItemSpeed: number) => {
    const current = historySpeeds[itemId] || currentItemSpeed || 1.0
    const speeds = [1.0, 1.25, 1.5, 2.0]
    const nextIdx = (speeds.indexOf(current) + 1) % speeds.length
    const nextSpeed = speeds[nextIdx] !== -1 ? speeds[nextIdx] : 1.0
    setHistorySpeeds((prev) => ({ ...prev, [itemId]: nextSpeed }))
    if (playingHistoryId === itemId && historyAudioRef.current) {
      historyAudioRef.current.playbackRate = nextSpeed
    }
  }

  // In-history download WAV
  const handleDownloadHistoryItem = (item: AudioHistoryItem) => {
    const url = AudioAPI.getHistoryAudioUrl(item.id)
    const a = document.createElement("a")
    a.href = url
    a.download = `speech_${item.id}.wav`
    a.click()
  }

  // In-history delete single item
  const handleDeleteHistoryItem = async (id: string) => {
    try {
      if (playingHistoryId === id && historyAudioRef.current) {
        historyAudioRef.current.pause()
        setPlayingHistoryId(null)
      }
      await AudioAPI.deleteHistoryItem(id)
      setHistoryItems((prev) => prev.filter((item) => item.id !== id))
      toast({ title: "Deleted", description: "Audio item removed from history." })
    } catch (err: any) {
      toast({ title: "Delete Failed", description: err.message, variant: "destructive" })
    }
  }

  // Clear all audio history
  const handleClearAllHistory = async () => {
    try {
      if (historyAudioRef.current) {
        historyAudioRef.current.pause()
        setPlayingHistoryId(null)
      }
      await AudioAPI.clearHistory()
      setHistoryItems([])
      toast({ title: "Cleared", description: "All audio history cleared." })
    } catch (err: any) {
      toast({ title: "Clear Failed", description: err.message, variant: "destructive" })
    }
  }

  const formatDate = (iso: string) => {
    try {
      const d = new Date(iso)
      if (isNaN(d.getTime())) return ""
      return (
        d.toLocaleDateString([], { month: "short", day: "numeric" }) +
        " " +
        d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
      )
    } catch {
      return ""
    }
  }

  // STT State (Whisper-Only & Option A Theme)
  const [sttSubTab, setSttSubTab] = useState<"live" | "upload">("live")

  // Whisper Model Readiness State
  const [whisperReady, setWhisperReady] = useState<"connecting" | "ready">("connecting")
  const whisperPollRef = useRef<NodeJS.Timeout | null>(null)

  // Live STT State
  const [liveState, setLiveState] = useState<"idle" | "recording" | "transcribing" | "finished">("idle")
  const [liveTranscript, setLiveTranscript] = useState<string>("")
  const [liveRecordingSeconds, setLiveRecordingSeconds] = useState(0)
  const [copiedLive, setCopiedLive] = useState(false)
  const liveAudioChunksRef = useRef<Blob[]>([])
  const liveMediaRecorderRef = useRef<MediaRecorder | null>(null)
  const liveStreamRef = useRef<MediaStream | null>(null)
  const liveTimerRef = useRef<NodeJS.Timeout | null>(null)
  const liveChunkIntervalRef = useRef<NodeJS.Timeout | null>(null)
  const isTranscribingChunkRef = useRef<boolean>(false)

  // Upload STT State
  const [uploadFile, setUploadFile] = useState<File | null>(null)
  const [uploadAudioUrl, setUploadAudioUrl] = useState<string | null>(null)
  const [isUploadProcessing, setIsUploadProcessing] = useState(false)
  const [uploadSeconds, setUploadSeconds] = useState(0)
  const [uploadTranscript, setUploadTranscript] = useState<string | null>(null)
  const [isDraggingFile, setIsDraggingFile] = useState(false)
  const [copiedUpload, setCopiedUpload] = useState(false)
  const uploadTimerRef = useRef<NodeJS.Timeout | null>(null)
  const [isUploadAudioPlaying, setIsUploadAudioPlaying] = useState(false)
  const uploadAudioRef = useRef<HTMLAudioElement | null>(null)

  const textareaRef = useRef<HTMLTextAreaElement | null>(null)

  // Parallel background sync of models
  const loadModels = async () => {
    try {
      const data = await AudioAPI.getModels().catch(() => null)

      if (data && data.length > 0) {
        setModels(data)
        const kokoro = data.find((m) => m.id === "kokoro-82m")
        if (kokoro) {
          setSelectedEngine("kokoro-82m")
          setSelectedVoice((prev) => prev && prev !== "default" ? prev : "af_heart")
        }
      }
    } catch (err) {
      console.warn("Background audio models check:", err)
    }
  }

  useEffect(() => {
    loadModels()
  }, [])

  // Poll download progress
  useEffect(() => {
    if (!downloadingJob) return

    const interval = setInterval(async () => {
      try {
        const job = await AudioAPI.getDownloadJob(downloadingJob.id)
        if (job.status === "completed") {
          clearInterval(interval)
          setDownloadingJob(null)
          toast({ title: "Model Downloaded", description: `${job.message || "Model ready for synthesis."}` })
          const updated = await AudioAPI.getModels()
          setModels(updated)
        } else if (job.status === "failed") {
          clearInterval(interval)
          setDownloadingJob(null)
          toast({ title: "Download Failed", description: job.error || "Failed to download model", variant: "destructive" })
        } else {
          setDownloadingJob({
            id: job.job_id,
            modelId: downloadingJob.modelId,
            progress: job.progress || 0.5,
            message: job.message || "Downloading...",
          })
        }
      } catch (err) {
        console.error("Error polling download job:", err)
      }
    }, 1500)

    return () => clearInterval(interval)
  }, [downloadingJob, toast])

  const currentModel = models.find((m) => m.id === selectedEngine)
  const isCurrentModelDownloaded = currentModel?.is_downloaded ?? false

  // Poll speech generation job progress
  const activeSpeechJobId = activeSpeechJob?.id
  useEffect(() => {
    if (!activeSpeechJobId) return

    const interval = setInterval(async () => {
      try {
        const job = await AudioAPI.getSpeechJob(activeSpeechJobId)
        if (job.status === "completed") {
          clearInterval(interval)
          const audioBlob = await AudioAPI.getSpeechJobResultBlob(activeSpeechJobId)
          const url = URL.createObjectURL(audioBlob)
          setGeneratedAudioUrl(url)

          loadDbHistory(daysFilter)
          setActiveSpeechJob(null)
          setIsGenerating(false)
          toast({ title: "Speech Generated", description: "Audio ready for playback and download." })
        } else if (job.status === "failed") {
          clearInterval(interval)
          setActiveSpeechJob(null)
          setIsGenerating(false)
          toast({
            title: "Synthesis Failed",
            description: job.error || "Failed to synthesize speech.",
            variant: "destructive",
          })
        } else if (job.status === "cancelled") {
          clearInterval(interval)
          setActiveSpeechJob(null)
          setIsGenerating(false)
          toast({ title: "Cancelled", description: "Speech synthesis was cancelled." })
        } else {
          setActiveSpeechJob((prev) =>
            prev && prev.id === activeSpeechJobId
              ? {
                  ...prev,
                  progress: job.progress || 0.1,
                  message: job.message || "Synthesizing on CUDA GPU...",
                  current_chunk: job.current_chunk || 0,
                  total_chunks: job.total_chunks || 1,
                }
              : prev
          )
        }
      } catch (err: any) {
        console.error("Error polling speech job:", err)
      }
    }, 600)

    return () => clearInterval(interval)
  }, [activeSpeechJobId, daysFilter, toast])

  const handleCancelSpeechJob = async () => {
    if (!activeSpeechJob) return
    try {
      await AudioAPI.cancelSpeechJob(activeSpeechJob.id)
      setActiveSpeechJob(null)
      setIsGenerating(false)
      toast({ title: "Job Cancelled", description: "Speech synthesis stopped." })
    } catch {
      setActiveSpeechJob(null)
      setIsGenerating(false)
    }
  }

  const handleDownloadModel = async (modelId: string) => {
    try {
      const res = await AudioAPI.downloadModel(modelId)
      setDownloadingJob({
        id: res.job_id,
        modelId,
        progress: 0.05,
        message: "Starting download from Hugging Face...",
      })
      toast({ title: "Download Started", description: `Downloading ${modelId} into speech directory...` })
    } catch (err: any) {
      toast({ title: "Download Error", description: err.message, variant: "destructive" })
    }
  }

  // Handle TTS Generation
  const handleGenerate = async () => {
    if (!inputText.trim()) {
      toast({ title: "Empty Text", description: "Please enter text to synthesize.", variant: "destructive" })
      return
    }

    const maxAllowedChars = selectedEngine === "kokoro-82m" ? 100000 : 3000
    if (inputText.length > maxAllowedChars) {
      toast({
        title: "Character Limit Exceeded",
        description: `This engine supports up to ${maxAllowedChars.toLocaleString()} characters (current: ${inputText.length.toLocaleString()}).`,
        variant: "destructive",
      })
      return
    }

    if (!isCurrentModelDownloaded) {
      toast({ title: "Model Not Installed", description: "Please download this model before generating speech.", variant: "destructive" })
      return
    }

    setIsGenerating(true)
    try {
      const res = await AudioAPI.startSpeechJob(selectedEngine, inputText, selectedVoice, speechSpeed)
      setActiveSpeechJob({
        id: res.job_id,
        progress: 0.05,
        message: "Starting synthesis job on CUDA GPU...",
        current_chunk: 0,
        total_chunks: 1,
      })
    } catch (err: any) {
      setIsGenerating(false)
      toast({
        title: "Synthesis Error",
        description: err.message || "Failed to start speech job.",
        variant: "destructive",
      })
    }
  }

  // Whisper Model Warmup Effect — triggers when user visits STT tab
  useEffect(() => {
    if (activeTab !== "stt") return

    // Already ready
    if (whisperReady === "ready") return

    // Trigger background model load on backend
    AudioAPI.whisperWarmup().catch(() => {})

    // Poll status every 1.5 seconds until ready
    if (whisperPollRef.current) clearInterval(whisperPollRef.current)
    whisperPollRef.current = setInterval(async () => {
      try {
        const res = await AudioAPI.whisperStatus()
        if (res.ready) {
          setWhisperReady("ready")
          if (whisperPollRef.current) clearInterval(whisperPollRef.current)
          whisperPollRef.current = null
        }
      } catch {
        // ignore transient errors, keep polling
      }
    }, 1500)

    return () => {
      if (whisperPollRef.current) clearInterval(whisperPollRef.current)
    }
  }, [activeTab])

  // STT Cleanup Effect
  useEffect(() => {
    return () => {
      if (liveChunkIntervalRef.current) clearInterval(liveChunkIntervalRef.current)
      if (liveTimerRef.current) clearInterval(liveTimerRef.current)
      if (uploadTimerRef.current) clearInterval(uploadTimerRef.current)
      if (whisperPollRef.current) clearInterval(whisperPollRef.current)
      if (liveStreamRef.current) {
        liveStreamRef.current.getTracks().forEach((track) => track.stop())
      }
      if (uploadAudioUrl) {
        URL.revokeObjectURL(uploadAudioUrl)
      }
    }
  }, [uploadAudioUrl])

  // Live STT Handlers (Faster-Whisper Realtime)
  const startLiveRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          sampleRate: 16000,
          echoCancellation: true,
          noiseSuppression: true,
        },
      })
      liveStreamRef.current = stream
      liveAudioChunksRef.current = []

      let mimeType = "audio/webm"
      if (typeof MediaRecorder !== "undefined") {
        if (MediaRecorder.isTypeSupported("audio/webm;codecs=opus")) {
          mimeType = "audio/webm;codecs=opus"
        } else if (MediaRecorder.isTypeSupported("audio/ogg;codecs=opus")) {
          mimeType = "audio/ogg;codecs=opus"
        } else if (MediaRecorder.isTypeSupported("audio/wav")) {
          mimeType = "audio/wav"
        }
      }

      const recorder = new MediaRecorder(stream, { mimeType })
      liveMediaRecorderRef.current = recorder

      recorder.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) {
          liveAudioChunksRef.current.push(e.data)
        }
      }

      recorder.onstop = () => {
        if (liveStreamRef.current) {
          liveStreamRef.current.getTracks().forEach((track) => track.stop())
          liveStreamRef.current = null
        }
      }

      recorder.start(1000)
      setLiveState("recording")
      setLiveTranscript("")
      setLiveRecordingSeconds(0)

      if (liveTimerRef.current) clearInterval(liveTimerRef.current)
      liveTimerRef.current = setInterval(() => {
        setLiveRecordingSeconds((prev) => prev + 1)
      }, 1000)

      // Periodic live Whisper recognition chunks
      isTranscribingChunkRef.current = false
      if (liveChunkIntervalRef.current) clearInterval(liveChunkIntervalRef.current)
      liveChunkIntervalRef.current = setInterval(async () => {
        if (liveAudioChunksRef.current.length === 0 || isTranscribingChunkRef.current) return

        try {
          isTranscribingChunkRef.current = true
          const currentBlob = new Blob(liveAudioChunksRef.current, { type: mimeType })
          if (currentBlob.size > 1200) {
            const partialText = await AudioAPI.transcribeAudio(currentBlob, true)
            if (partialText && partialText.trim()) {
              setLiveTranscript(partialText.trim())
            }
          }
        } catch (err) {
          console.warn("Live STT chunk recognition:", err)
        } finally {
          isTranscribingChunkRef.current = false
        }
      }, 2500)
    } catch (err: any) {
      toast({
        title: "Microphone Access Denied",
        description: err.message || "Please allow microphone access in your browser to transcribe speech.",
        variant: "destructive",
      })
    }
  }

  const finishLiveRecording = async () => {
    if (liveChunkIntervalRef.current) {
      clearInterval(liveChunkIntervalRef.current)
      liveChunkIntervalRef.current = null
    }
    if (liveTimerRef.current) {
      clearInterval(liveTimerRef.current)
      liveTimerRef.current = null
    }

    const recorder = liveMediaRecorderRef.current
    if (!recorder || recorder.state === "inactive") {
      setLiveState("finished")
      return
    }

    setLiveState("transcribing")

    await new Promise<void>((resolve) => {
      recorder.onstop = () => {
        if (liveStreamRef.current) {
          liveStreamRef.current.getTracks().forEach((track) => track.stop())
          liveStreamRef.current = null
        }
        resolve()
      }
      try {
        recorder.stop()
      } catch {
        resolve()
      }
    })

    try {
      const mimeType = recorder.mimeType || "audio/webm"
      const fullBlob = new Blob(liveAudioChunksRef.current, { type: mimeType })

      if (fullBlob.size > 500) {
        const finalText = await AudioAPI.transcribeAudio(fullBlob, false)
        if (finalText && finalText.trim()) {
          setLiveTranscript(finalText.trim())
        } else if (!liveTranscript) {
          setLiveTranscript("No speech detected.")
        }
      } else if (!liveTranscript) {
        setLiveTranscript("Audio recording was too short.")
      }
      setLiveState("finished")
      toast({ title: "Transcription Finished", description: "Audio transcribed with local Whisper model." })
    } catch (err: any) {
      setLiveState("finished")
      toast({
        title: "Transcription Notice",
        description: err.message || "Failed to finalize audio transcription.",
        variant: "destructive",
      })
    }
  }

  const resetLiveRecording = () => {
    if (liveChunkIntervalRef.current) clearInterval(liveChunkIntervalRef.current)
    if (liveTimerRef.current) clearInterval(liveTimerRef.current)
    if (liveStreamRef.current) {
      liveStreamRef.current.getTracks().forEach((track) => track.stop())
    }
    setLiveState("idle")
    setLiveTranscript("")
    setLiveRecordingSeconds(0)
  }

  // Upload STT Handlers
  const handleUploadAudioFile = async (file: File) => {
    if (!file) return

    setUploadFile(file)
    setUploadTranscript(null)
    setIsUploadProcessing(true)
    setUploadSeconds(0)

    if (uploadAudioUrl) {
      URL.revokeObjectURL(uploadAudioUrl)
    }
    const audioUrl = URL.createObjectURL(file)
    setUploadAudioUrl(audioUrl)

    if (uploadTimerRef.current) clearInterval(uploadTimerRef.current)
    uploadTimerRef.current = setInterval(() => {
      setUploadSeconds((prev) => Number((prev + 0.1).toFixed(1)))
    }, 100)

    try {
      const text = await AudioAPI.transcribeAudio(file, false)
      setUploadTranscript(text || "No speech detected in audio file.")
      toast({ title: "Transcription Complete", description: "File processed successfully with Whisper." })
    } catch (err: any) {
      toast({
        title: "Transcription Failed",
        description: err.message || "Failed to transcribe audio file.",
        variant: "destructive",
      })
    } finally {
      setIsUploadProcessing(false)
      if (uploadTimerRef.current) clearInterval(uploadTimerRef.current)
    }
  }

  const handleDropFile = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    setIsDraggingFile(false)
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleUploadAudioFile(e.dataTransfer.files[0])
    }
  }

  const downloadTextFile = (text: string, filename: string) => {
    const blob = new Blob([text], { type: "text/plain;charset=utf-8" })
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = filename
    a.click()
    URL.revokeObjectURL(url)
  }

  const formatSeconds = (sec: number) => {
    const m = Math.floor(sec / 60)
    const s = sec % 60
    return `${m}:${s < 10 ? "0" : ""}${s}`
  }

  const currentVoiceObj = currentModel?.voices.find((v) => v.id === selectedVoice)



  return (
    <div className="flex flex-col h-full bg-[#0a0a0a] relative overflow-hidden text-foreground">
      {/* Studio Header */}
      <header className="h-14 border-b border-white/10 flex items-center justify-between px-5 sm:px-8 shrink-0 bg-background/95 backdrop-blur-sm sticky top-0 z-20">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-xl bg-primary/10 text-primary border border-primary/20 flex items-center justify-center">
            <AudioLines size={18} />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="text-sm font-bold text-foreground">Audio Studio</span>
              <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
            </div>
            <div className="text-[10px] text-muted-foreground font-medium flex items-center gap-1.5">
              <span>Text-to-Speech &amp; Speech-to-Text</span>
            </div>
          </div>
        </div>

        {/* Center Pill Tabs */}
        <div className="hidden md:inline-flex items-center p-1 rounded-full bg-white/5 border border-white/10 shadow-inner gap-1">
          <button
            onClick={() => setActiveTab("tts")}
            className={cn(
              "px-4 py-1.5 rounded-full text-xs font-semibold transition-all",
              activeTab === "tts"
                ? "bg-primary text-white shadow-md shadow-primary/20"
                : "text-muted-foreground hover:text-foreground"
            )}
          >
            Text To Speech
          </button>
          <button
            onClick={() => setActiveTab("stt")}
            className={cn(
              "px-4 py-1.5 rounded-full text-xs font-semibold transition-all",
              activeTab === "stt"
                ? "bg-primary text-white shadow-md shadow-primary/20"
                : "text-muted-foreground hover:text-foreground"
            )}
          >
            Speech To Text
          </button>
          <button
            onClick={() => setActiveTab("cloning")}
            className={cn(
              "px-4 py-1.5 rounded-full text-xs font-semibold transition-all",
              activeTab === "cloning"
                ? "bg-primary text-white shadow-md shadow-primary/20"
                : "text-muted-foreground hover:text-foreground"
            )}
          >
            Voice Cloning
          </button>
        </div>

        {/* Right Accelerators & Actions */}
        <div className="flex items-center gap-2 sm:gap-3">
          <button
            onClick={loadModels}
            disabled={isLoadingModels}
            className="p-2 rounded-lg hover:bg-white/10 text-muted-foreground hover:text-foreground transition-all"
            title="Refresh speech models status"
          >
            <RefreshCw size={15} className={cn(isLoadingModels && "animate-spin text-primary")} />
          </button>
        </div>
      </header>

      {/* Main Content Area */}
      {activeTab === "tts" ? (
        <div className="flex-1 flex flex-col lg:flex-row overflow-hidden w-full relative">
          
          {/* Left Canvas: Full Height & Width Text Container */}
          <div className="flex-1 h-full min-w-0 flex flex-col p-3 sm:p-2 lg:p-2 overflow-hidden">
            <div className="flex-1 min-h-0 flex flex-col rounded-md border border-white/10 bg-[#141414]/90 p-2 overflow-hidden shadow-2xl relative">
              
              {/* Top Bar inside Canvas */}
              <div className="flex items-center justify-between px-2 py-1.5 border-b border-white/5 bg-black/20 shrink-0 rounded-t-sm">
                <div className="flex items-center gap-2">
                  <AudioLines size={14} className="text-primary" />
                  <span className="text-xs font-semibold text-foreground">Speech Script</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-[10px] font-mono text-muted-foreground px-2 py-0.5 rounded bg-white/5 border border-white/5">
                    {currentModel?.name || selectedEngine}
                  </span>
                </div>
              </div>

              {/* Native Text Editor Container */}
              <div className="relative flex-1 min-h-0 w-full overflow-hidden bg-black/10">
                <textarea
                  ref={textareaRef}
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value)}
                  placeholder="Enter your own text or dialogue..."
                  className="w-full h-full p-5 font-mono text-sm sm:text-base leading-relaxed whitespace-pre-wrap break-words bg-transparent text-foreground caret-white resize-none focus:outline-none selection:bg-primary/30"
                />
              </div>

              {/* Real-time Job Progress Banner (if active) */}
              {activeSpeechJob && (
                <div className="p-3.5 mx-4 mb-2 rounded-2xl bg-black/60 border border-primary/25 space-y-2 animate-in fade-in slide-in-from-bottom-2 shrink-0">
                  <div className="flex items-center justify-between text-xs">
                    <span className="font-semibold text-foreground flex items-center gap-2 truncate">
                      <Loader2 size={13} className="animate-spin text-primary shrink-0" />
                      <span className="truncate">{activeSpeechJob.message || "Synthesizing speech on CUDA GPU..."}</span>
                    </span>
                    <span className="font-mono text-xs font-bold text-primary shrink-0 ml-2">
                      {Math.round((activeSpeechJob.progress || 0) * 100)}%
                    </span>
                  </div>
                  <div className="w-full h-2 rounded-full bg-white/10 overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-primary via-emerald-400 to-primary transition-all duration-300"
                      style={{ width: `${Math.max(5, (activeSpeechJob.progress || 0) * 100)}%` }}
                    />
                  </div>
                  <div className="flex items-center justify-between text-[11px] text-muted-foreground">
                    <span>
                      {activeSpeechJob.total_chunks > 1
                        ? `Chunk ${activeSpeechJob.current_chunk} of ${activeSpeechJob.total_chunks}`
                        : "Batch synthesis"}
                    </span>
                    <button
                      onClick={handleCancelSpeechJob}
                      className="text-muted-foreground hover:text-rose-400 font-medium transition-colors"
                    >
                      Cancel
                    </button>
                  </div>
                </div>
              )}

              {/* Bottom Controls Bar: Engine Info, Counter, Generate Speech */}
              <div className="flex items-center justify-between p-3.5 sm:p-4 border-t border-white/5 bg-black/20 shrink-0">
                {/* Left: Engine Info Indicator */}
                <div className="relative">
                  {selectedEngine === "fish-s1-mini" ? (
                    <span className="text-[11px] text-muted-foreground italic flex items-center gap-1.5">
                      <Sparkles size={12} className="text-primary" />
                      <span>Fish Audio S1-mini (CUDA GPU &bull; up to 3,000 chars)</span>
                    </span>
                  ) : (
                    <span className="text-[11px] text-muted-foreground italic flex items-center gap-1.5">
                      <Zap size={12} className="text-emerald-400" />
                      <span>Kokoro-82M (CUDA GPU &bull; up to 100,000 chars)</span>
                    </span>
                  )}
                </div>

                {/* Right: Character Count + Generate Speech Button */}
                <div className="flex items-center gap-3 sm:gap-4">
                  {(() => {
                    const maxChars = selectedEngine === "kokoro-82m" ? 100000 : 3000
                    const isOverLimit = inputText.length > maxChars
                    return (
                      <>
                        <span className={cn(
                          "text-xs font-semibold",
                          isOverLimit ? "text-rose-400 font-bold" : "text-muted-foreground"
                        )}>
                          <strong className={cn(
                            "font-bold",
                            isOverLimit ? "text-rose-400" : "text-foreground"
                          )}>
                            {inputText.length.toLocaleString()}
                          </strong>
                          {` / ${maxChars.toLocaleString()} characters`}
                        </span>

                        <button
                          onClick={handleGenerate}
                          disabled={isGenerating || !isCurrentModelDownloaded || !inputText.trim() || isOverLimit}
                          className={cn(
                            "px-6 py-2.5 rounded-full text-xs font-bold transition-all shadow-md flex items-center gap-2",
                            isGenerating || !isCurrentModelDownloaded || !inputText.trim() || isOverLimit
                              ? "bg-white/10 text-muted-foreground cursor-not-allowed"
                              : "bg-primary hover:bg-primary/90 text-white active:scale-95 shadow-md shadow-primary/25"
                          )}
                        >
                          {isGenerating ? (
                            <>
                              <Loader2 size={13} className="animate-spin" />
                              <span>
                                {activeSpeechJob
                                  ? `Generating ${Math.round((activeSpeechJob.progress || 0) * 100)}%...`
                                  : "Generating..."}
                              </span>
                            </>
                          ) : (
                            <span>Generate Speech</span>
                          )}
                        </button>
                      </>
                    )
                  })()}
                </div>
              </div>

            </div>

            {/* Generated Audio Player Row (Below the text container) */}
            {generatedAudioUrl && (
              <div className="pt-3 flex justify-center shrink-0 animate-in fade-in duration-200">
                <AudioPlayer
                  audioUrl={generatedAudioUrl}
                  title={currentModel?.name}
                  subtitle={currentVoiceObj?.name}
                  onDownload={() => {
                    const a = document.createElement("a")
                    a.href = generatedAudioUrl
                    a.download = `speech_${selectedEngine}_${Date.now()}.wav`
                    a.click()
                  }}
                />
              </div>
            )}
          </div>

          {/* Right Sidebar: Docked Top-to-Bottom (Flush on right edge) */}
          <div className="w-full lg:w-[340px] xl:w-[380px] shrink-0 h-full border-l border-white/10 bg-[#121212] flex flex-col overflow-hidden">
            
            {/* Top Tabs Switcher: Settings | History */}
            <div className="p-3.5 border-b border-white/10 bg-black/20 shrink-0">
              <div className="p-1 rounded-full bg-black/40 border border-white/10 flex items-center">
                <button
                  onClick={() => setSidebarTab("settings")}
                  className={cn(
                    "flex-1 py-1.5 rounded-full text-xs font-semibold text-center transition-all",
                    sidebarTab === "settings"
                      ? "bg-primary text-white shadow-md shadow-primary/20 font-bold"
                      : "text-muted-foreground hover:text-foreground"
                  )}
                >
                  Settings
                </button>
                <button
                  onClick={() => setSidebarTab("history")}
                  className={cn(
                    "flex-1 py-1.5 rounded-full text-xs font-semibold text-center transition-all",
                    sidebarTab === "history"
                      ? "bg-primary text-white shadow-md shadow-primary/20 font-bold"
                      : "text-muted-foreground hover:text-foreground"
                  )}
                >
                  History {historyItems.length > 0 && `(${historyItems.length})`}
                </button>
              </div>
            </div>

            {/* Scrollable Sidebar Body */}
            <div className="flex-1 overflow-y-auto p-4 space-y-4">
              {sidebarTab === "settings" ? (
                /* Settings View */
                <div className="space-y-4">
                  {/* Voice Selection */}
                  <div className="space-y-1.5">
                    <label className="text-xs font-medium text-muted-foreground block">
                      Voice
                    </label>
                    <div className="relative">
                      <button
                        type="button"
                        onClick={() => setIsVoiceDropdownOpen(!isVoiceDropdownOpen)}
                        className="w-full flex items-center justify-between px-3.5 py-2.5 rounded-2xl bg-black/30 border border-white/10 hover:border-white/20 text-xs font-semibold text-foreground transition-all shadow-inner"
                      >
                        <div className="flex items-center gap-2.5 min-w-0">
                          <div className="w-6 h-6 rounded-full bg-gradient-to-tr from-purple-500 to-indigo-500 text-white flex items-center justify-center text-[10px] font-bold shadow-sm shrink-0">
                            {currentVoiceObj?.name?.charAt(0) || "V"}
                          </div>
                          <span className="truncate">{currentVoiceObj?.name || "Select voice"}</span>
                        </div>
                        <ChevronDown
                          size={14}
                          className={cn("text-muted-foreground transition-transform shrink-0 ml-2", isVoiceDropdownOpen && "rotate-180")}
                        />
                      </button>

                      {isVoiceDropdownOpen && (
                        <>
                          <div className="fixed inset-0 z-30" onClick={() => setIsVoiceDropdownOpen(false)} />
                          <div className="absolute top-full left-0 right-0 mt-1.5 z-40 rounded-2xl bg-[#1c1c1c] border border-white/15 p-2 shadow-2xl space-y-1 max-h-56 overflow-y-auto">
                            {currentModel?.voices && currentModel.voices.length > 0 ? (
                              currentModel.voices.map((v) => (
                                <button
                                  key={v.id}
                                  onClick={() => {
                                    setSelectedVoice(v.id)
                                    setIsVoiceDropdownOpen(false)
                                  }}
                                  className={cn(
                                    "w-full flex items-center justify-between px-3 py-2 rounded-xl text-xs font-medium transition-all text-left",
                                    selectedVoice === v.id
                                      ? "bg-white/10 text-white font-semibold"
                                      : "text-muted-foreground hover:bg-white/5 hover:text-foreground"
                                  )}
                                >
                                  <div className="flex items-center gap-2 truncate">
                                    <div className="w-5 h-5 rounded-full bg-white/10 flex items-center justify-center text-[9px]">
                                      {v.name.charAt(0)}
                                    </div>
                                    <span className="truncate">{v.name}</span>
                                  </div>
                                  {selectedVoice === v.id && <Check size={13} className="text-primary shrink-0" />}
                                </button>
                              ))
                            ) : (
                              <div className="text-xs text-muted-foreground p-2 text-center">
                                Default voice active
                              </div>
                            )}
                          </div>
                        </>
                      )}
                    </div>
                  </div>

                  {/* Model Selection */}
                  <div className="space-y-1.5">
                    <label className="text-xs font-medium text-muted-foreground block">
                      Model
                    </label>
                    <div className="relative">
                      <button
                        type="button"
                        onClick={() => setIsModelDropdownOpen(!isModelDropdownOpen)}
                        className="w-full flex items-center justify-between px-3.5 py-2.5 rounded-2xl bg-black/30 border border-white/10 hover:border-white/20 text-xs font-semibold text-foreground transition-all shadow-inner"
                      >
                        <div className="flex items-center gap-2 truncate">
                          <span className="truncate">{currentModel?.name || selectedEngine}</span>
                          <span className="px-2 py-0.5 rounded-full text-[10px] font-mono bg-white/10 text-muted-foreground font-semibold">
                            {selectedEngine === "fish-s1-mini" ? "Stable" : "Fast RTF"}
                          </span>
                        </div>
                        <ChevronDown
                          size={14}
                          className={cn("text-muted-foreground transition-transform shrink-0 ml-2", isModelDropdownOpen && "rotate-180")}
                        />
                      </button>

                      {isModelDropdownOpen && (
                        <>
                          <div className="fixed inset-0 z-30" onClick={() => setIsModelDropdownOpen(false)} />
                          <div className="absolute top-full left-0 right-0 mt-1.5 z-40 rounded-2xl bg-[#1c1c1c] border border-white/15 p-2 shadow-2xl space-y-1">
                            <button
                              onClick={() => {
                                setSelectedEngine("kokoro-82m")
                                setSelectedVoice("af_heart")
                                setIsModelDropdownOpen(false)
                              }}
                              className={cn(
                                "w-full flex items-center justify-between px-3 py-2 rounded-xl text-xs font-medium transition-all text-left",
                                selectedEngine === "kokoro-82m"
                                  ? "bg-white/10 text-white font-semibold"
                                  : "text-muted-foreground hover:bg-white/5 hover:text-foreground"
                              )}
                            >
                              <div>
                                <div className="font-semibold text-foreground">Kokoro-82M</div>
                                <div className="text-[10px] text-muted-foreground">CUDA GPU / CPU · 100,000 Chars</div>
                              </div>
                              <span className="px-1.5 py-0.5 rounded bg-emerald-500/10 text-emerald-400 text-[10px] font-mono">
                                READY
                              </span>
                            </button>
                          </div>
                        </>
                      )}
                    </div>

                    {/* Download alert if model is missing */}
                    {!isCurrentModelDownloaded && (
                      <div className="p-3 rounded-2xl bg-amber-500/10 border border-amber-500/20 space-y-2 mt-2">
                        <div className="flex items-center justify-between text-xs font-semibold text-amber-300">
                          <span>Model Download Required</span>
                          <span className="text-[10px] font-mono">{currentModel?.size || "350 MB"}</span>
                        </div>
                        {downloadingJob?.modelId === selectedEngine ? (
                          <div className="space-y-1">
                            <div className="flex items-center justify-between text-[10px] text-amber-300 font-semibold">
                              <span className="truncate pr-2">{downloadingJob.message}</span>
                              <span className="font-mono">{Math.round(downloadingJob.progress * 100)}%</span>
                            </div>
                            <div className="w-full bg-amber-950/60 h-1.5 rounded-full overflow-hidden border border-amber-500/30">
                              <div
                                className="bg-amber-500 h-full transition-all duration-300"
                                style={{ width: `${Math.max(5, downloadingJob.progress * 100)}%` }}
                              />
                            </div>
                          </div>
                        ) : (
                          <button
                            onClick={() => handleDownloadModel(selectedEngine)}
                            className="w-full py-1.5 px-3 rounded-xl bg-amber-500 hover:bg-amber-600 text-black text-xs font-bold flex items-center justify-center gap-1.5 transition-all shadow-md active:scale-95"
                          >
                            <Download size={13} />
                            <span>Download Model</span>
                          </button>
                        )}
                      </div>
                    )}
                  </div>

                  {/* Model Highlight Card */}
                  <div className="p-3.5 rounded-2xl bg-gradient-to-br from-white/[0.06] to-white/[0.02] border border-white/10 space-y-1">
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-bold text-foreground">
                        Kokoro-82M
                      </span>
                      <span className="text-[10px] font-mono text-primary font-bold">Active Engine</span>
                    </div>
                    <p className="text-[11px] text-muted-foreground leading-relaxed">
                      High-performance speech engine optimized for ultra-fast, natural speech synthesis and long-form documents up to 100k characters.
                    </p>
                  </div>

                  {/* Audio Controls (Matching Image 2) */}
                  <div className="space-y-3 pt-2">
                    <label className="text-xs font-semibold text-muted-foreground block uppercase tracking-wider text-[10px]">
                      Audio Controls
                    </label>

                    {/* Volume Slider */}
                    <div className="flex items-center justify-between gap-3 p-2.5 rounded-2xl bg-black/20 border border-white/5">
                      <span className="text-xs font-medium text-muted-foreground">Volume</span>
                      <div className="flex items-center gap-3 flex-1 max-w-[160px]">
                        <input
                          type="range"
                          min="0"
                          max="100"
                          value={volume}
                          onChange={(e) => setVolume(parseInt(e.target.value))}
                          className="w-full h-1.5 bg-white/10 rounded-lg appearance-none cursor-pointer accent-primary"
                        />
                        <span className="text-xs font-mono font-semibold text-foreground min-w-[28px] text-right">
                          {volume}
                        </span>
                      </div>
                    </div>

                    {/* Speed Slider */}
                    <div className="flex items-center justify-between gap-3 p-2.5 rounded-2xl bg-black/20 border border-white/5">
                      <span className="text-xs font-medium text-muted-foreground">Speed</span>
                      <div className="flex items-center gap-3 flex-1 max-w-[160px]">
                        <input
                          type="range"
                          min="0.5"
                          max="2.0"
                          step="0.1"
                          value={speechSpeed}
                          onChange={(e) => setSpeechSpeed(parseFloat(e.target.value))}
                          className="w-full h-1.5 bg-white/10 rounded-lg appearance-none cursor-pointer accent-primary"
                        />
                        <span className="text-xs font-mono font-semibold text-foreground min-w-[28px] text-right">
                          {speechSpeed.toFixed(1)}x
                        </span>
                      </div>
                    </div>

                    {/* Loudness Normalization Toggle */}
                    <div className="flex items-center justify-between gap-2 p-2.5 rounded-2xl bg-black/20 border border-white/5">
                      <span className="text-xs font-medium text-muted-foreground truncate">Loudness Normalization</span>
                      <div className="inline-flex items-center p-0.5 rounded-xl bg-black/50 border border-white/10 shrink-0">
                        <button
                          onClick={() => setLoudnessNorm(false)}
                          className={cn(
                            "px-2.5 py-1 rounded-lg text-[11px] font-semibold transition-all",
                            !loudnessNorm ? "bg-white/20 text-white font-bold shadow-sm" : "text-muted-foreground hover:text-foreground"
                          )}
                        >
                          Off
                        </button>
                        <button
                          onClick={() => setLoudnessNorm(true)}
                          className={cn(
                            "px-2.5 py-1 rounded-lg text-[11px] font-semibold transition-all",
                            loudnessNorm ? "bg-primary text-white font-bold shadow-sm" : "text-muted-foreground hover:text-foreground"
                          )}
                        >
                          On
                        </button>
                      </div>
                    </div>

                    {/* Text Normalization Toggle */}
                    <div className="flex items-center justify-between gap-2 p-2.5 rounded-2xl bg-black/20 border border-white/5">
                      <span className="text-xs font-medium text-muted-foreground truncate">Text Normalization</span>
                      <div className="inline-flex items-center p-0.5 rounded-xl bg-black/50 border border-white/10 shrink-0">
                        <button
                          onClick={() => setTextNorm(false)}
                          className={cn(
                            "px-2.5 py-1 rounded-lg text-[11px] font-semibold transition-all",
                            !textNorm ? "bg-white/20 text-white font-bold shadow-sm" : "text-muted-foreground hover:text-foreground"
                          )}
                        >
                          Off
                        </button>
                        <button
                          onClick={() => setTextNorm(true)}
                          className={cn(
                            "px-2.5 py-1 rounded-lg text-[11px] font-semibold transition-all",
                            textNorm ? "bg-primary text-white font-bold shadow-sm" : "text-muted-foreground hover:text-foreground"
                          )}
                        >
                          On
                        </button>
                      </div>
                    </div>
                  </div>

                </div>
              ) : (
                /* History View (SQLite Database-backed) */
                <div className="space-y-3">
                  {/* History Filter Row */}
                  <div className="flex items-center justify-between gap-2">
                    <div className="relative flex-1">
                      <button
                        type="button"
                        onClick={() => setIsDaysDropdownOpen(!isDaysDropdownOpen)}
                        className="w-full px-3 py-1.5 rounded-full bg-black/30 border border-white/10 text-xs font-medium text-foreground flex items-center justify-between hover:border-white/20 transition-all"
                      >
                        <span>
                          {daysFilter === 7
                            ? "Last 7 days"
                            : daysFilter === 30
                            ? "Last 30 days"
                            : daysFilter === 90
                            ? "Last 90 days"
                            : "All time"}
                        </span>
                        <ChevronDown size={12} className={cn("text-muted-foreground transition-transform", isDaysDropdownOpen && "rotate-180")} />
                      </button>

                      {isDaysDropdownOpen && (
                        <>
                          <div className="fixed inset-0 z-30" onClick={() => setIsDaysDropdownOpen(false)} />
                          <div className="absolute top-full left-0 right-0 mt-1 z-40 rounded-xl bg-[#1c1c1c] border border-white/15 p-1 shadow-2xl space-y-0.5">
                            {[
                              { label: "Last 7 days", value: 7 },
                              { label: "Last 30 days", value: 30 },
                              { label: "Last 90 days", value: 90 },
                              { label: "All time", value: 0 },
                            ].map((opt) => (
                              <button
                                key={opt.value}
                                onClick={() => {
                                  setDaysFilter(opt.value)
                                  setIsDaysDropdownOpen(false)
                                }}
                                className={cn(
                                  "w-full flex items-center justify-between px-3 py-1.5 rounded-lg text-xs font-medium transition-all text-left",
                                  daysFilter === opt.value
                                    ? "bg-white/10 text-white font-semibold"
                                    : "text-muted-foreground hover:bg-white/5 hover:text-foreground"
                                )}
                              >
                                <span>{opt.label}</span>
                                {daysFilter === opt.value && <Check size={12} className="text-primary" />}
                              </button>
                            ))}
                          </div>
                        </>
                      )}
                    </div>

                    <button
                      onClick={() => loadDbHistory(daysFilter)}
                      disabled={isLoadingHistory}
                      className="p-2 rounded-full bg-white/5 hover:bg-white/10 text-muted-foreground hover:text-foreground transition-all"
                      title="Refresh history"
                    >
                      <RefreshCw size={13} className={cn(isLoadingHistory && "animate-spin text-primary")} />
                    </button>
                  </div>

                  <div className="flex items-center justify-between text-[11px] text-muted-foreground font-semibold pt-1">
                    <span>
                      {historyItems.length > 0
                        ? `${historyItems.length} generation${historyItems.length > 1 ? "s" : ""}`
                        : "No generations"}
                    </span>
                    {historyItems.length > 0 && (
                      <button
                        onClick={handleClearAllHistory}
                        className="text-muted-foreground hover:text-rose-400 transition-colors"
                      >
                        Clear All
                      </button>
                    )}
                  </div>

                  {isLoadingHistory ? (
                    <div className="text-center py-12 text-xs text-muted-foreground space-y-2">
                      <Loader2 size={22} className="mx-auto animate-spin text-primary" />
                      <p>Loading audio history...</p>
                    </div>
                  ) : historyItems.length === 0 ? (
                    <div className="text-center py-12 text-xs text-muted-foreground space-y-2">
                      <HistoryIcon size={26} className="mx-auto opacity-40" />
                      <p>No audio generated in this time range.</p>
                    </div>
                  ) : (
                    <div className="space-y-3">
                      {historyItems.map((item) => {
                        const isThisPlaying = playingHistoryId === item.id
                        const currentSpeed = historySpeeds[item.id] || item.speed || 1.0

                        return (
                          <div
                            key={item.id}
                            className={cn(
                              "p-3.5 rounded-2xl bg-black/30 border transition-all space-y-2",
                              isThisPlaying
                                ? "border-primary/50 shadow-lg shadow-primary/5 bg-primary/[0.03]"
                                : "border-white/5 hover:border-white/15"
                            )}
                          >
                            {/* Avatar + Voice + Model badge + Timestamp */}
                            <div className="flex items-center justify-between text-xs">
                              <div className="flex items-center gap-2 truncate">
                                <div className="w-5 h-5 rounded-full bg-gradient-to-tr from-purple-500 to-indigo-500 text-white flex items-center justify-center text-[9px] font-bold shrink-0">
                                  {item.voice_name ? item.voice_name.charAt(0) : "V"}
                                </div>
                                <span className="font-semibold text-foreground truncate">{item.voice_name || "Voice"}</span>
                                <span className="text-[10px] px-1.5 py-0.5 rounded bg-white/5 text-muted-foreground font-mono">
                                  {item.model_name || item.model_id}
                                </span>
                              </div>
                              <span className="text-[10px] font-mono text-muted-foreground shrink-0 ml-2">
                                {formatDate(item.created_at)}
                              </span>
                            </div>

                            {/* Spoken Text Preview */}
                            <p className="text-xs text-muted-foreground/90 line-clamp-2 leading-relaxed font-mono">
                              {item.text}
                            </p>

                            {/* Player Controls Bar */}
                            <div className="flex items-center justify-between pt-2 border-t border-white/5">
                              {/* Play/Pause + Speed Adjuster */}
                              <div className="flex items-center gap-2">
                                <button
                                  onClick={() => handleTogglePlayHistory(item)}
                                  className={cn(
                                    "w-7 h-7 rounded-full flex items-center justify-center transition-all shadow-sm active:scale-95",
                                    isThisPlaying
                                      ? "bg-primary text-white shadow-primary/30"
                                      : "bg-white/10 hover:bg-white/20 text-foreground"
                                  )}
                                  title={isThisPlaying ? "Pause playback" : "Play audio in history"}
                                >
                                  {isThisPlaying ? (
                                    <Pause size={12} className="fill-current text-white" />
                                  ) : (
                                    <Play size={11} className="fill-current text-foreground ml-0.5" />
                                  )}
                                </button>

                                {/* Interactive Speed Adjustment Pill */}
                                <button
                                  onClick={() => cycleHistorySpeed(item.id, item.speed)}
                                  className="px-2 py-0.5 rounded-full bg-white/5 hover:bg-white/10 border border-white/10 text-[10px] font-mono font-semibold text-muted-foreground hover:text-foreground transition-all"
                                  title="Click to cycle playback speed (1x, 1.25x, 1.5x, 2x)"
                                >
                                  {currentSpeed}x
                                </button>

                                {item.duration_seconds > 0 && (
                                  <span className="text-[10px] font-mono text-muted-foreground/70">
                                    {Math.round(item.duration_seconds)}s
                                  </span>
                                )}
                              </div>

                              {/* Actions: Download, Copy, Delete */}
                              <div className="flex items-center gap-1.5 text-muted-foreground">
                                <button
                                  onClick={() => handleDownloadHistoryItem(item)}
                                  className="p-1.5 rounded-lg hover:bg-white/10 hover:text-foreground transition-all"
                                  title="Download WAV"
                                >
                                  <Download size={13} />
                                </button>
                                <button
                                  onClick={() => {
                                    navigator.clipboard.writeText(item.text)
                                    toast({ title: "Copied", description: "Script copied to clipboard." })
                                  }}
                                  className="p-1.5 rounded-lg hover:bg-white/10 hover:text-foreground transition-all"
                                  title="Copy Text"
                                >
                                  <Copy size={13} />
                                </button>
                                <button
                                  onClick={() => handleDeleteHistoryItem(item.id)}
                                  className="p-1.5 rounded-lg hover:bg-rose-500/20 hover:text-rose-400 transition-all"
                                  title="Delete from history"
                                >
                                  <Trash2 size={13} />
                                </button>
                              </div>
                            </div>
                          </div>
                        )
                      })}
                    </div>
                  )}
                </div>
              )}
            </div>

          </div>

        </div>
      ) : activeTab === "stt" ? (
        /* Speech To Text (Whisper) Tab - Fits page nicely without scroll */
        <div className="flex-1 min-h-0 h-full p-2.5 sm:p-3 overflow-hidden flex flex-col w-full">
          {/* Ambient Organic Blurred Glow Mesh Backdrop Container — Fills viewport */}
          <div className="relative w-full h-full flex-1 min-h-0 rounded-2xl overflow-hidden border border-white/10 shadow-2xl bg-[#0a0a0d] flex flex-col">
            {/* Organic Colorful Glowing Blobs */}
            <div className="absolute -top-32 -left-32 w-[500px] h-[500px] rounded-full bg-gradient-to-br from-amber-500/20 via-orange-600/15 to-rose-600/10 blur-[120px] pointer-events-none" />
            <div className="absolute -bottom-32 -right-32 w-[500px] h-[500px] rounded-full bg-gradient-to-tl from-teal-500/20 via-emerald-600/15 to-cyan-500/10 blur-[120px] pointer-events-none" />
            <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[400px] rounded-full bg-gradient-to-tr from-purple-700/20 via-indigo-600/15 to-pink-500/10 blur-[140px] pointer-events-none" />
            {/* Analog subtle noise grain overlay */}
            <div className="absolute inset-0 bg-[radial-gradient(#ffffff08_1px,transparent_1px)] [background-size:16px_16px] pointer-events-none" />

            {/* CONTAINER HEADER — tabs right-aligned */}
            <div className="relative z-10 flex items-center justify-between px-5 sm:px-7 py-3 border-b border-white/8 shrink-0">
              <div className="flex items-center gap-2.5">
                <div className="w-2 h-2 rounded-full bg-white/50" />
                <span className="text-xs sm:text-sm text-zinc-300 font-semibold tracking-wide">Speech to Text</span>
              </div>

              {/* Sub-Tab Pills */}
              <div className="flex items-center gap-1 p-1 rounded-full bg-zinc-900/80 border border-white/10 backdrop-blur-xl shadow-lg">
                <button
                  onClick={() => setSttSubTab("live")}
                  className={cn(
                    "flex items-center gap-1.5 px-4 py-1.5 rounded-full text-xs font-semibold transition-all duration-200",
                    sttSubTab === "live"
                      ? "bg-white text-zinc-950 shadow-md"
                      : "text-zinc-400 hover:text-white"
                  )}
                >
                  <Mic size={12} />
                  <span>Live Recording</span>
                </button>
                <button
                  onClick={() => setSttSubTab("upload")}
                  className={cn(
                    "flex items-center gap-1.5 px-4 py-1.5 rounded-full text-xs font-semibold transition-all duration-200",
                    sttSubTab === "upload"
                      ? "bg-white text-zinc-950 shadow-md"
                      : "text-zinc-400 hover:text-white"
                  )}
                >
                  <Upload size={12} />
                  <span>Upload Audio</span>
                </button>
              </div>
            </div>

            {/* CONTENT AREA — spacious inner container, dark application theme */}
            <div className="relative z-10 flex-1 min-h-0 flex items-center justify-center p-3 sm:p-5 overflow-hidden">

              {/* SUB-TAB 1: LIVE RECORDING */}
              {sttSubTab === "live" && (
                <div className="w-full max-w-4xl lg:max-w-5xl h-[88%] min-h-[360px] max-h-[600px] rounded-2xl bg-[#131316]/95 backdrop-blur-2xl border border-white/12 shadow-[0_25px_60px_-15px_rgba(0,0,0,0.8)] p-6 sm:p-8 flex flex-col justify-between transition-all duration-300">

                  {/* CONNECTING STATE — dark theme */}
                  {liveState === "idle" && whisperReady === "connecting" && (
                    <>
                      <div className="flex-1 flex items-center justify-center py-10">
                        <p className="text-base sm:text-lg text-zinc-400 font-normal">Loading Whisper engine...</p>
                      </div>
                      <div className="flex items-center justify-between pt-4 border-t border-white/10">
                        <div className="flex items-center gap-2.5">
                          <span className="w-2.5 h-2.5 rounded-full border-2 border-zinc-500 inline-block animate-pulse"></span>
                          <span className="text-xs sm:text-sm text-zinc-400 font-medium">Connecting...</span>
                        </div>
                        <button
                          disabled
                          className="px-6 py-2 rounded-full text-xs font-semibold bg-white/10 text-zinc-500 cursor-not-allowed"
                        >
                          Finish
                        </button>
                      </div>
                    </>
                  )}

                  {/* IDLE STATE — model ready, waiting for user to click mic */}
                  {liveState === "idle" && whisperReady === "ready" && (
                    <>
                      <div className="flex-1 flex flex-col items-center justify-center text-center py-6">
                        <button
                          onClick={startLiveRecording}
                          className="w-16 h-16 rounded-full bg-white text-zinc-950 flex items-center justify-center shadow-2xl hover:scale-105 active:scale-95 transition-all duration-200 cursor-pointer group"
                          title="Start Transcribing"
                        >
                          <Mic size={24} className="text-zinc-950 group-hover:scale-110 transition-transform" />
                        </button>
                        <h4 className="text-base sm:text-lg font-bold text-white mt-4">
                          Click to start transcribing
                        </h4>
                        <p className="text-xs sm:text-sm text-zinc-400 mt-1 font-normal">
                          Experience the power of local Whisper Realtime
                        </p>
                      </div>

                      <div className="flex items-center justify-between pt-4 border-t border-white/10">
                        <div className="flex items-center gap-2">
                          <span className="w-2.5 h-2.5 rounded-full bg-emerald-400"></span>
                          <span className="text-xs text-zinc-400 font-medium">Whisper Ready</span>
                        </div>
                        <button
                          disabled
                          className="px-6 py-2 rounded-full text-xs font-semibold bg-white/10 text-zinc-500 cursor-not-allowed"
                        >
                          Finish
                        </button>
                      </div>
                    </>
                  )}

                  {/* RECORDING & TRANSCRIBING STATES */}
                  {(liveState === "recording" || liveState === "transcribing") && (
                    <>
                      <div className="flex-1 flex flex-col justify-start min-h-[140px] overflow-y-auto pr-1">
                        {liveTranscript ? (
                          <p className="text-base sm:text-xl text-white font-normal leading-relaxed whitespace-pre-wrap select-text animate-in fade-in duration-200">
                            {liveTranscript}
                          </p>
                        ) : (
                          <p className="text-base sm:text-xl text-zinc-500 font-normal italic">
                            Speak to transcribe...
                          </p>
                        )}
                      </div>

                      <div className="flex items-center justify-between pt-4 border-t border-white/10 mt-3">
                        <div className="flex items-center gap-2.5">
                          <span className="relative flex h-2.5 w-2.5">
                            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-rose-400 opacity-75"></span>
                            <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-rose-500"></span>
                          </span>
                          <span className="text-xs font-semibold text-white tracking-wide">
                            {liveState === "transcribing" ? "Finishing..." : "Transcribing..."}
                          </span>
                          <span className="text-xs text-zinc-400 font-mono">
                            ({formatSeconds(liveRecordingSeconds)})
                          </span>
                        </div>

                        <button
                          onClick={finishLiveRecording}
                          disabled={liveState === "transcribing"}
                          className="px-6 py-2 rounded-full text-xs font-bold bg-white text-zinc-950 hover:bg-zinc-200 active:scale-95 transition-all shadow-md disabled:opacity-50"
                        >
                          {liveState === "transcribing" ? (
                            <span className="flex items-center gap-1.5">
                              <Loader2 size={13} className="animate-spin text-zinc-950" />
                              <span>Finishing...</span>
                            </span>
                          ) : (
                            "Finish"
                          )}
                        </button>
                      </div>
                    </>
                  )}

                  {/* FINISHED STATE */}
                  {liveState === "finished" && (
                    <div className="flex flex-col justify-between h-full space-y-4">
                      <div className="flex items-center justify-between pb-3.5 border-b border-white/10">
                        <div className="flex items-center gap-2 text-xs text-zinc-400">
                          <span className="font-semibold text-white">Transcription Complete</span>
                          <span>·</span>
                          <span>{liveTranscript.split(/\s+/).filter(Boolean).length} words</span>
                          <span>·</span>
                          <span>{liveTranscript.length} chars</span>
                        </div>
                        <button
                          onClick={resetLiveRecording}
                          className="inline-flex items-center gap-1.5 text-xs text-zinc-300 hover:text-white transition-all px-3 py-1.5 rounded-full bg-white/5 hover:bg-white/10"
                        >
                          <RotateCcw size={12} />
                          <span>New Recording</span>
                        </button>
                      </div>

                      <div className="flex-1 overflow-y-auto pr-1 py-2">
                        <p className="text-base sm:text-lg text-white leading-relaxed whitespace-pre-wrap select-text">
                          {liveTranscript}
                        </p>
                      </div>

                      <div className="flex flex-wrap items-center justify-between gap-3 pt-4 border-t border-white/10">
                        <div className="flex items-center gap-2">
                          <button
                            onClick={() => {
                              navigator.clipboard.writeText(liveTranscript)
                              setCopiedLive(true)
                              setTimeout(() => setCopiedLive(false), 2000)
                              toast({ title: "Copied", description: "Transcript copied to clipboard." })
                            }}
                            className="inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-full bg-white/10 hover:bg-white/15 text-white text-xs font-semibold transition-all"
                          >
                            {copiedLive ? <CheckCheck size={13} className="text-emerald-400" /> : <Copy size={13} />}
                            <span>{copiedLive ? "Copied!" : "Copy Text"}</span>
                          </button>
                          <button
                            onClick={() => downloadTextFile(liveTranscript, "whisper_transcription.txt")}
                            className="inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-full bg-white/10 hover:bg-white/15 text-white text-xs font-semibold transition-all"
                          >
                            <Download size={13} />
                            <span>Download .txt</span>
                          </button>
                        </div>

                        <button
                          onClick={() => {
                            setInputText(liveTranscript)
                            setActiveTab("tts")
                            toast({ title: "Loaded into TTS", description: "Live transcription moved to Text-to-Speech prompt." })
                          }}
                          className="inline-flex items-center gap-1.5 px-5 py-2 rounded-full text-xs font-bold bg-white text-zinc-950 hover:bg-zinc-200 active:scale-95 transition-all shadow-md"
                        >
                          <span>Use in TTS Script</span>
                          <ArrowRight size={13} />
                        </button>
                      </div>
                    </div>
                  )}

                </div>
              )}

              {/* SUB-TAB 2: UPLOAD AUDIO */}
              {sttSubTab === "upload" && (
                <div className="w-full max-w-4xl lg:max-w-5xl h-[88%] min-h-[360px] max-h-[600px] rounded-2xl bg-[#131316]/95 backdrop-blur-2xl border border-white/12 shadow-[0_25px_60px_-15px_rgba(0,0,0,0.8)] p-6 sm:p-8 flex flex-col justify-between transition-all duration-300">

                  {/* IDLE / DROPZONE STATE */}
                  {!isUploadProcessing && !uploadTranscript && (
                    <div
                      onDragOver={(e) => {
                        e.preventDefault()
                        setIsDraggingFile(true)
                      }}
                      onDragLeave={() => setIsDraggingFile(false)}
                      onDrop={handleDropFile}
                      className={cn(
                        "flex-1 border-2 border-dashed rounded-xl p-8 flex flex-col items-center justify-center text-center transition-all duration-300",
                        isDraggingFile ? "border-primary bg-primary/10 scale-[1.01]" : "border-white/15 hover:border-white/30"
                      )}
                    >
                      <div className="w-14 h-14 rounded-2xl bg-white/10 text-white flex items-center justify-center shadow-lg mb-3">
                        <FileAudio size={28} />
                      </div>
                      <h4 className="text-base font-bold text-white">Upload Audio File</h4>
                      <p className="text-xs text-zinc-400 mt-1 max-w-sm">
                        Drag and drop your audio recording here or browse from your computer.
                      </p>

                      <div className="flex flex-wrap items-center justify-center gap-1.5 mt-4">
                        {["WAV", "MP3", "M4A", "FLAC", "OGG", "WebM"].map((fmt) => (
                          <span
                            key={fmt}
                            className="px-2.5 py-1 rounded-md text-[10px] font-mono font-bold bg-white/5 border border-white/10 text-zinc-300"
                          >
                            {fmt}
                          </span>
                        ))}
                      </div>

                      <label className="mt-5 px-6 py-2.5 rounded-full text-xs font-bold bg-white text-zinc-950 hover:bg-zinc-200 cursor-pointer shadow-lg transition-all active:scale-95 inline-flex items-center gap-2">
                        <Upload size={14} />
                        <span>Browse Audio Files</span>
                        <input
                          type="file"
                          accept="audio/*,.wav,.mp3,.m4a,.flac,.ogg,.webm"
                          onChange={(e) => {
                            const f = e.target.files?.[0]
                            if (f) handleUploadAudioFile(f)
                          }}
                          className="hidden"
                        />
                      </label>
                    </div>
                  )}

                  {/* PROCESSING STATE */}
                  {isUploadProcessing && (
                    <div className="flex-1 flex flex-col items-center justify-center text-center space-y-6">
                      <div className="flex items-center justify-center gap-1.5 h-16">
                        {[26, 48, 72, 35, 60, 85, 45, 68, 40, 75, 52, 30].map((h, i) => (
                          <div
                            key={i}
                            className="w-1.5 bg-gradient-to-t from-teal-400 to-amber-400 rounded-full animate-pulse"
                            style={{
                              height: `${h}%`,
                              animationDelay: `${i * 90}ms`,
                              animationDuration: "800ms",
                            }}
                          />
                        ))}
                      </div>
                      <div className="space-y-2">
                        <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full text-xs font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                          <Loader2 size={13} className="animate-spin" />
                          <span>Whisper Base · Neural Audio Analysis</span>
                        </div>
                        <h4 className="text-base font-bold text-white truncate max-w-md">
                          {uploadFile?.name || "Processing audio..."}
                        </h4>
                        <p className="text-xs text-zinc-400">
                          Elapsed: <span className="font-mono text-white font-bold">{uploadSeconds}s</span> · CUDA GPU Accelerated
                        </p>
                      </div>
                    </div>
                  )}

                  {/* COMPLETED STATE */}
                  {!isUploadProcessing && uploadTranscript && (
                    <div className="flex flex-col justify-between h-full space-y-4">
                      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3.5 border-b border-white/10">
                        <div className="space-y-0.5">
                          <div className="flex items-center gap-2">
                            <span className="font-bold text-sm text-white truncate max-w-xs sm:max-w-md">
                              {uploadFile?.name || "Uploaded Audio"}
                            </span>
                            {uploadFile && (
                              <span className="text-[11px] font-mono text-zinc-400">
                                ({uploadFile.size < 1024 * 1024
                                  ? `${(uploadFile.size / 1024).toFixed(1)} KB`
                                  : `${(uploadFile.size / (1024 * 1024)).toFixed(1)} MB`})
                              </span>
                            )}
                          </div>
                          <p className="text-xs text-zinc-400">
                            {uploadTranscript.split(/\s+/).filter(Boolean).length} words · {uploadTranscript.length} chars
                          </p>
                        </div>

                        <div className="flex items-center gap-2">
                          {uploadAudioUrl && (
                            <>
                              <audio
                                ref={uploadAudioRef}
                                src={uploadAudioUrl}
                                onEnded={() => setIsUploadAudioPlaying(false)}
                                className="hidden"
                              />
                              <button
                                onClick={() => {
                                  if (!uploadAudioRef.current) return
                                  if (isUploadAudioPlaying) {
                                    uploadAudioRef.current.pause()
                                    setIsUploadAudioPlaying(false)
                                  } else {
                                    uploadAudioRef.current.play()
                                    setIsUploadAudioPlaying(true)
                                  }
                                }}
                                className="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-white/10 hover:bg-white/15 text-white text-xs font-semibold transition-all"
                              >
                                {isUploadAudioPlaying ? <Pause size={13} /> : <Play size={13} />}
                                <span>{isUploadAudioPlaying ? "Pause Audio" : "Play Audio"}</span>
                              </button>
                            </>
                          )}
                          <button
                            onClick={() => {
                              setUploadFile(null)
                              setUploadTranscript(null)
                              if (uploadAudioUrl) URL.revokeObjectURL(uploadAudioUrl)
                              setUploadAudioUrl(null)
                            }}
                            className="inline-flex items-center gap-1 text-xs text-zinc-400 hover:text-white px-3 py-1.5 rounded-full bg-white/5 hover:bg-white/10 transition-all"
                          >
                            <RotateCcw size={12} />
                            <span>Upload Another</span>
                          </button>
                        </div>
                      </div>

                      <div className="flex-1 overflow-y-auto pr-1 py-2">
                        <p className="text-base sm:text-lg text-white leading-relaxed whitespace-pre-wrap select-text">
                          {uploadTranscript}
                        </p>
                      </div>

                      <div className="flex flex-wrap items-center justify-between gap-3 pt-4 border-t border-white/10">
                        <div className="flex items-center gap-2">
                          <button
                            onClick={() => {
                              navigator.clipboard.writeText(uploadTranscript)
                              setCopiedUpload(true)
                              setTimeout(() => setCopiedUpload(false), 2000)
                              toast({ title: "Copied", description: "Transcript copied to clipboard." })
                            }}
                            className="inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-full bg-white/10 hover:bg-white/15 text-white text-xs font-semibold transition-all"
                          >
                            {copiedUpload ? <CheckCheck size={13} className="text-emerald-400" /> : <Copy size={13} />}
                            <span>{copiedUpload ? "Copied!" : "Copy Text"}</span>
                          </button>
                          <button
                            onClick={() => downloadTextFile(uploadTranscript, `${uploadFile?.name || "audio"}_transcript.txt`)}
                            className="inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-full bg-white/10 hover:bg-white/15 text-white text-xs font-semibold transition-all"
                          >
                            <Download size={13} />
                            <span>Download .txt</span>
                          </button>
                        </div>

                        <button
                          onClick={() => {
                            setInputText(uploadTranscript)
                            setActiveTab("tts")
                            toast({ title: "Loaded into TTS", description: "Uploaded transcription moved to Text-to-Speech prompt." })
                          }}
                          className="inline-flex items-center gap-1.5 px-5 py-2 rounded-full text-xs font-bold bg-white text-zinc-950 hover:bg-zinc-200 active:scale-95 transition-all shadow-md"
                        >
                          <span>Use in TTS Script</span>
                          <ArrowRight size={13} />
                        </button>
                      </div>
                    </div>
                  )}

                </div>
              )}

            </div>
          </div>
        </div>
      ) : (
        /* Voice Cloning View */
        <div className="flex-1 overflow-y-auto p-4 sm:p-6 lg:p-8 space-y-6">
          <div className="max-w-4xl mx-auto space-y-6">
            <div className="rounded-3xl border border-white/10 bg-[#141414]/90 p-8 shadow-xl text-center space-y-5">
              <div className="w-16 h-16 rounded-2xl bg-primary/10 text-primary border border-primary/20 flex items-center justify-center mx-auto shadow-lg shadow-primary/10">
                <Sparkles size={28} />
              </div>

              <div className="space-y-2 max-w-lg mx-auto">
                <h3 className="text-xl font-bold text-foreground">Zero-Shot Voice Cloning</h3>
                <p className="text-xs text-muted-foreground leading-relaxed">
                  Fish Audio S1-mini natively supports zero-shot voice cloning by conditioning on a 5-10 second reference audio clip. You can synthesize any text in anyone's voice with exact tone, pitch, and accent.
                </p>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-left pt-2">
                <div className="p-4 rounded-2xl bg-black/30 border border-white/5 space-y-1.5">
                  <span className="text-[10px] font-bold text-primary uppercase font-mono">Step 1</span>
                  <h5 className="text-xs font-semibold text-foreground">Upload Reference</h5>
                  <p className="text-[11px] text-muted-foreground">Provide a clean 5-10 second WAV/MP3 recording of the target voice.</p>
                </div>
                <div className="p-4 rounded-2xl bg-black/30 border border-white/5 space-y-1.5">
                  <span className="text-[10px] font-bold text-primary uppercase font-mono">Step 2</span>
                  <h5 className="text-xs font-semibold text-foreground">Reference Text</h5>
                  <p className="text-[11px] text-muted-foreground">Whisper automatically transcribes the reference sample to align phonemes.</p>
                </div>
                <div className="p-4 rounded-2xl bg-black/30 border border-white/5 space-y-1.5">
                  <span className="text-[10px] font-bold text-primary uppercase font-mono">Step 3</span>
                  <h5 className="text-xs font-semibold text-foreground">Synthesize</h5>
                  <p className="text-[11px] text-muted-foreground">Generate endless new speech using the cloned vocal fingerprint.</p>
                </div>
              </div>

              <div className="pt-4 flex items-center justify-center gap-3">
                <button
                  onClick={() => setActiveTab("tts")}
                  className="px-6 py-2.5 rounded-full text-xs font-bold bg-primary hover:bg-primary/90 text-white transition-all shadow-md shadow-primary/25 active:scale-95"
                >
                  Open Text To Speech
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Hidden audio element for in-history inline playback */}
      <audio ref={historyAudioRef} onEnded={() => setPlayingHistoryId(null)} className="hidden" />
    </div>
  )
}
