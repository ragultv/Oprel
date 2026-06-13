'use client'

import React, { useState, useEffect, useCallback } from 'react'
import { useRouter } from 'next/navigation'
import * as Icons from 'lucide-react'
import {
  Settings2, Cpu, SlidersHorizontal, ChevronLeft,
  Plus, Trash2, RefreshCw, Eye, EyeOff, Check, X,
  Loader2, Zap, ChevronDown, ChevronUp, RotateCcw,
  Globe, KeyRound, Server, BookOpen, Save, Sparkles, Lock,
} from 'lucide-react'
import { useApp } from '@/services/context'
import { type Skill } from '@/services/skills'
import {
  type ProviderConfig, type ProviderType,
  PROVIDER_PRESETS, fetchProviderModels,
} from '@/services/providers'
import { cn } from '@/services/utils'

// ─── Skill Categories & Icons ────────────────────────────────────────────────
const CATEGORY_COLORS: Record<string, { text: string; bg: string; border: string }> = {
  Writing: { text: "text-cyan-500", bg: "bg-cyan-500/10 border-cyan-500/20", border: "border-cyan-500/20" },
  Development: { text: "text-indigo-500", bg: "bg-indigo-500/10 border-indigo-500/20", border: "border-indigo-500/20" },
  Research: { text: "text-emerald-500", bg: "bg-emerald-500/10 border-emerald-500/20", border: "border-emerald-500/20" },
  Documents: { text: "text-amber-500", bg: "bg-amber-500/10 border-amber-500/20", border: "border-amber-500/20" },
  Media: { text: "text-rose-500", bg: "bg-rose-500/10 border-rose-500/20", border: "border-rose-500/20" },
}

const AVAILABLE_ICONS = [
  "Brain", "Sparkles", "Code2", "Bug", "Eye", "Search", "Globe", "Mail", 
  "FileText", "Presentation", "ImageIcon", "Wand2", "Edit3", "Calculator", 
  "Terminal", "Briefcase", "TrendingUp", "Compass", "Layers", "Target"
]

const emptySkill: Skill = {
  id: '',
  name: '',
  description: '',
  command: '',
  icon: 'Sparkles',
  category: 'Writing',
  systemPrompt: '',
  temperature: 0.7,
  maxTokens: 4096,
  enabled: true,
  isPremium: false,
}

// ─── Sidebar Tab definitions ──────────────────────────────────────────────────

const TABS = [
  { id: 'config', label: 'Generation', icon: SlidersHorizontal },
  { id: 'presets', label: 'Prompt Presets', icon: BookOpen },
  { id: 'providers', label: 'AI Providers', icon: Globe },
  { id: 'skills', label: 'Skills', icon: Sparkles },
] as const

type TabId = (typeof TABS)[number]['id']

// ─── System Prompt Presets ────────────────────────────────────────────────────

const SYSTEM_PRESETS = [
  { label: 'General', prompt: 'You are a helpful AI assistant.' },
  {
    label: 'Coder',
    prompt:
      'You are an expert software engineer. Write clean, efficient, well-documented code. Prefer concise explanations. Always include code examples.',
  },
  {
    label: 'Diagrams',
    prompt:
      'You are a technical diagram expert. When asked to create diagrams, output valid Mermaid syntax inside ```mermaid code blocks. Always produce syntactically correct Mermaid.',
  },
  {
    label: 'Web Builder',
    prompt:
      'You are a senior frontend engineer. When asked to build UIs, produce complete, self-contained HTML files with embedded CSS and JS inside ```html code blocks.',
  },
  {
    label: 'Writer',
    prompt:
      'You are a professional writer and editor. Help craft clear, engaging, and well-structured prose. Adapt tone to the request.',
  },
  {
    label: 'Tutor',
    prompt:
      'You are a patient and thorough tutor. Explain concepts step-by-step, use analogies and examples, and check for understanding.',
  },
  {
    label: 'Analyst',
    prompt:
      'You are a data analyst. Provide structured analysis, identify trends, and support conclusions with reasoning and data.',
  },
  {
    label: 'Data / SQL',
    prompt:
      'You are a database expert specialising in SQL. Write performant, standards-compliant queries. Explain query plans when asked.',
  },
  {
    label: 'DevOps',
    prompt:
      'You are a DevOps and cloud infrastructure expert. Provide practical, secure, and scalable solutions using industry best practices.',
  },
  {
    label: 'Coach',
    prompt:
      'You are a supportive life and productivity coach. Help the user clarify goals, overcome obstacles, and build positive habits.',
  },
]

// ─── Slider helper ────────────────────────────────────────────────────────────

function SettingSlider({
  label, value, min, max, step, onChange, format,
}: {
  label: string
  value: number
  min: number
  max: number
  step: number
  onChange: (v: number) => void
  format?: (v: number) => string
}) {
  return (
    <div className="space-y-2">
      <div className="flex justify-between items-center">
        <label className="text-xs font-semibold text-muted-foreground tracking-widest uppercase">
          {label}
        </label>
        <span className="text-sm font-bold text-primary tabular-nums">
          {format ? format(value) : value}
        </span>
      </div>
      <input
        type="range" min={min} max={max} step={step} value={value}
        onChange={e => onChange(Number(e.target.value))}
        className="w-full h-1.5 bg-secondary rounded-full appearance-none cursor-pointer accent-primary"
      />
    </div>
  )
}

// ─── Provider type badge ──────────────────────────────────────────────────────

function ProviderBadge({ type }: { type: ProviderType }) {
  const preset = PROVIDER_PRESETS[type] || PROVIDER_PRESETS['openai-compatible']
  return (
    <span
      className="px-1.5 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider"
      style={{ color: preset.color, background: `${preset.color}22` }}
    >
      {preset.name}
    </span>
  )
}

// ─── Provider Form ────────────────────────────────────────────────────────────

function ProviderForm({
  initial,
  onSave,
  onCancel,
}: {
  initial?: Partial<ProviderConfig>
  onSave: (p: ProviderConfig) => Promise<void>
  onCancel: () => void
}) {
  const isEdit = Boolean(initial?.id)
  const [type, setType] = useState<ProviderType>(initial?.type ?? 'openai')
  const [name, setName] = useState(initial?.name ?? PROVIDER_PRESETS[initial?.type ?? 'openai'].name)
  const [apiKey, setApiKey] = useState(initial?.apiKey ?? '')
  const [baseUrl, setBaseUrl] = useState(initial?.baseUrl ?? PROVIDER_PRESETS[initial?.type ?? 'openai'].baseUrl)
  const [showKey, setShowKey] = useState(false)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')

  const preset = PROVIDER_PRESETS[type] || PROVIDER_PRESETS['openai-compatible']

  // Update defaults when type changes (only for new providers)
  useEffect(() => {
    if (!isEdit) {
      setName(PROVIDER_PRESETS[type].name)
      setBaseUrl(PROVIDER_PRESETS[type].baseUrl)
    }
  }, [type, isEdit])

  const handleSave = async () => {
    if (!apiKey.trim() && type !== 'openai-compatible') {
      setError('API key is required')
      return
    }
    setSaving(true)
    setError('')
    try {
      const id = initial?.id ?? `${type}-${Date.now()}`
      await onSave({
        id,
        name: name.trim() || preset.name,
        type,
        apiKey: apiKey.trim(),
        baseUrl: baseUrl.trim() || preset.baseUrl,
        enabled: initial?.enabled ?? true,
        enabledModelIds: initial?.enabledModelIds ?? [],
        availableModelIds: initial?.availableModelIds ?? [],
      })
    } catch (e: any) {
      setError(e.message || 'Failed to save provider')
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="space-y-4 border border-border rounded-xl p-5 bg-card/60">
      <h3 className="font-semibold text-foreground">{isEdit ? 'Edit Provider' : 'New Provider'}</h3>

      {/* Type selector */}
      {!isEdit && (
        <div>
          <label className="text-xs font-semibold text-muted-foreground uppercase tracking-widest block mb-2">
            Provider Type
          </label>
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
            {(Object.keys(PROVIDER_PRESETS) as ProviderType[]).map(t => {
              const p = PROVIDER_PRESETS[t]
              return (
                <button
                  key={t}
                  onClick={() => setType(t)}
                  className={cn(
                    'flex flex-col items-start gap-1 px-3 py-2.5 rounded-lg border text-left transition-all text-sm',
                    type === t
                      ? 'border-primary/50 bg-primary/10 text-foreground'
                      : 'border-border bg-secondary/40 text-muted-foreground hover:border-border/80 hover:bg-secondary/60'
                  )}
                >
                  <span className="font-semibold text-xs" style={{ color: p.color }}>{p.name}</span>
                  <span className="text-[10px] leading-tight opacity-70">{p.description}</span>
                </button>
              )
            })}
          </div>
        </div>
      )}

      {/* Name */}
      <div>
        <label className="text-xs font-semibold text-muted-foreground uppercase tracking-widest block mb-1.5">
          Display Name
        </label>
        <input
          value={name}
          onChange={e => setName(e.target.value)}
          placeholder={preset.name}
          className="w-full bg-secondary/60 border border-border rounded-lg px-3 py-2 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-primary/50"
        />
      </div>

      {/* API Key */}
      <div>
        <label className="text-xs font-semibold text-muted-foreground uppercase tracking-widest block mb-1.5">
          API Key
          {preset.docsUrl && (
            <a href={preset.docsUrl} target="_blank" rel="noopener noreferrer"
              className="ml-2 normal-case text-primary/70 hover:text-primary transition-colors">
              Get key ↗
            </a>
          )}
        </label>
        <div className="relative">
          <input
            type={showKey ? 'text' : 'password'}
            value={apiKey}
            onChange={e => setApiKey(e.target.value)}
            placeholder="sk-..."
            className="w-full bg-secondary/60 border border-border rounded-lg px-3 py-2 pr-10 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-primary/50 font-mono"
          />
          <button
            onClick={() => setShowKey(s => !s)}
            className="absolute right-2.5 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground transition-colors"
          >
            {showKey ? <EyeOff size={14} /> : <Eye size={14} />}
          </button>
        </div>
      </div>

      {/* Base URL (for custom / compatible) */}
      {(type === 'openai-compatible' || type === 'nvidia' || type === 'groq' || type === 'openrouter') && (
        <div>
          <label className="text-xs font-semibold text-muted-foreground uppercase tracking-widest block mb-1.5">
            Base URL
          </label>
          <input
            value={baseUrl}
            onChange={e => setBaseUrl(e.target.value)}
            placeholder={preset.baseUrl || 'https://...'}
            className="w-full bg-secondary/60 border border-border rounded-lg px-3 py-2 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-primary/50 font-mono"
          />
        </div>
      )}

      {error && (
        <p className="text-destructive text-xs bg-destructive/10 border border-destructive/20 px-3 py-2 rounded-lg">
          {error}
        </p>
      )}

      <div className="flex gap-2 justify-end">
        <button
          onClick={onCancel}
          className="px-4 py-2 text-sm rounded-lg border border-border text-muted-foreground hover:bg-secondary transition-all"
        >
          Cancel
        </button>
        <button
          onClick={handleSave}
          disabled={saving}
          className="px-4 py-2 text-sm rounded-lg bg-primary text-primary-foreground hover:bg-primary/90 transition-all flex items-center gap-2 disabled:opacity-60"
        >
          {saving ? <Loader2 size={14} className="animate-spin" /> : <Save size={14} />}
          {isEdit ? 'Save Changes' : 'Add Provider'}
        </button>
      </div>
    </div>
  )
}

// ─── Provider Card ────────────────────────────────────────────────────────────

function ProviderCard({
  provider,
  onUpdate,
  onDelete,
}: {
  provider: ProviderConfig
  onUpdate: (p: ProviderConfig) => Promise<void>
  onDelete: (id: string) => Promise<void>
}) {
  const [expanded, setExpanded] = useState(false)
  const [fetching, setFetching] = useState(false)
  const [fetchError, setFetchError] = useState('')
  const [deleting, setDeleting] = useState(false)
  const preset = PROVIDER_PRESETS[provider.type] || PROVIDER_PRESETS['openai-compatible']
  const enabledModelIds = Array.isArray(provider.enabledModelIds) ? provider.enabledModelIds : []
  const availableModelIds = Array.isArray(provider.availableModelIds) ? provider.availableModelIds : []

  const handleFetchModels = async () => {
    setFetching(true)
    setFetchError('')
    try {
      const models = await fetchProviderModels(provider)
      await onUpdate({
        ...provider,
        availableModelIds: Array.isArray(models) ? models.filter(Boolean) : [],
        lastFetched: new Date().toISOString(),
      })
    } catch (e: any) {
      setFetchError(e.message || 'Failed to fetch models')
    } finally {
      setFetching(false)
    }
  }

  const toggleModel = async (modelId: string) => {
    const already = enabledModelIds.includes(modelId)
    const next = already
      ? enabledModelIds.filter(m => m !== modelId)
      : [...enabledModelIds, modelId]
    try {
      await onUpdate({
        ...provider,
        enabledModelIds: next,
        availableModelIds,
      })
    } catch (e) {
      console.error('Failed to update provider model selection:', e)
    }
  }

  const toggleProvider = async () => {
    try {
      await onUpdate({
        ...provider,
        enabled: !provider.enabled,
        enabledModelIds,
        availableModelIds,
      })
    } catch (e) {
      console.error('Failed to toggle provider enabled state:', e)
    }
  }

  const handleDelete = async () => {
    if (!confirm(`Remove provider "${provider.name}"?`)) return
    setDeleting(true)
    try { await onDelete(provider.id) } finally { setDeleting(false) }
  }

  return (
    <div className={cn(
      'border rounded-xl overflow-hidden transition-all',
      provider.enabled ? 'border-border' : 'border-border/40 opacity-60'
    )}>
      {/* Header */}
      <div className="flex items-center gap-3 px-4 py-3 bg-card/60">
        <div
          className="w-8 h-8 rounded-lg flex items-center justify-center shrink-0 text-white font-bold text-xs"
          style={{ background: preset.color }}
        >
          {provider.name.slice(0, 2).toUpperCase()}
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className="font-semibold text-sm text-foreground truncate">{provider.name}</span>
            <ProviderBadge type={provider.type} />
          </div>
          <p className="text-xs text-muted-foreground">
            {enabledModelIds.length} model{enabledModelIds.length !== 1 ? 's' : ''} enabled
            {provider.lastFetched && (
              (() => {
                const date = new Date(provider.lastFetched);
                return isNaN(date.getTime()) ? '' : ` · fetched ${date.toLocaleDateString()}`;
              })()
            )}
          </p>
        </div>
        <div className="flex items-center gap-1.5 shrink-0">
          {/* Enable toggle */}
          <button
            onClick={toggleProvider}
            className={cn(
              'px-2.5 py-1 rounded-md text-xs font-medium transition-all border',
              provider.enabled
                ? 'bg-primary/10 border-primary/30 text-primary'
                : 'border-border text-muted-foreground hover:border-border'
            )}
          >
            {provider.enabled ? 'Enabled' : 'Disabled'}
          </button>
          <button
            onClick={() => setExpanded(e => !e)}
            className="w-7 h-7 rounded-lg flex items-center justify-center text-muted-foreground hover:text-foreground hover:bg-secondary transition-all"
          >
            {expanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
          </button>
          <button
            onClick={handleDelete}
            disabled={deleting}
            className="w-7 h-7 rounded-lg flex items-center justify-center text-muted-foreground hover:text-destructive hover:bg-destructive/10 transition-all"
          >
            {deleting ? <Loader2 size={12} className="animate-spin" /> : <Trash2 size={13} />}
          </button>
        </div>
      </div>

      {/* Model list */}
      {expanded && (
        <div className="px-4 py-4 border-t border-border/50 bg-background/30 space-y-3">
          <div className="flex items-center justify-between">
            <p className="text-xs text-muted-foreground font-medium">
              {availableModelIds.length > 0
                ? `${availableModelIds.length} models available — toggle ones to enable`
                : 'No models fetched yet'}
            </p>
            <button
              onClick={handleFetchModels}
              disabled={fetching}
              className="flex items-center gap-1.5 px-3 py-1.5 text-xs rounded-lg border border-border hover:bg-secondary text-muted-foreground hover:text-foreground transition-all disabled:opacity-60"
            >
              {fetching ? <Loader2 size={12} className="animate-spin" /> : <RefreshCw size={12} />}
              {fetching ? 'Fetching…' : 'Fetch Models'}
            </button>
          </div>

          {fetchError && (
            <p className="text-destructive text-xs bg-destructive/10 border border-destructive/20 px-3 py-2 rounded-lg">
              {fetchError}
            </p>
          )}

          {availableModelIds.length > 0 && (
            <div className="max-h-64 overflow-y-auto space-y-1 pr-1">
              {availableModelIds.map(modelId => {
                const enabled = enabledModelIds.includes(modelId)
                return (
                  <label
                    key={modelId}
                    className="flex items-center gap-3 px-3 py-2 rounded-lg hover:bg-secondary/60 cursor-pointer transition-all group"
                  >
                    <div className={cn(
                      'w-4 h-4 rounded border flex items-center justify-center shrink-0 transition-all',
                      enabled
                        ? 'bg-primary border-primary'
                        : 'border-border group-hover:border-primary/50'
                    )}>
                      {enabled && <Check size={10} className="text-primary-foreground" />}
                    </div>
                    <input
                      type="checkbox" checked={enabled} onChange={() => toggleModel(modelId)}
                      className="sr-only"
                    />
                    <span className="text-xs font-mono text-foreground/80 truncate flex-1">{modelId}</span>
                    {enabled && (
                      <span
                        className="text-[9px] font-bold px-1.5 py-0.5 rounded uppercase"
                        style={{ color: preset.color, background: `${preset.color}22` }}
                      >
                        Active
                      </span>
                    )}
                  </label>
                )
              })}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// ─── Main Settings Page ───────────────────────────────────────────────────────

export default function SettingsPage() {
  const router = useRouter()
  const {
    settings, setSettings, saveSettings,
    providers, saveProvider, removeProvider,
    skills, saveSkill, deleteSkill,
  } = useApp()
  const [activeTab, setActiveTab] = useState<TabId>('config')
  const [localSettings, setLocalSettings] = useState(settings)
  const [dirty, setDirty] = useState(false)
  const [saving, setSaving] = useState(false)
  const [savedFeedback, setSavedFeedback] = useState(false)
  const [showAddProvider, setShowAddProvider] = useState(false)

  // Skills Editor State
  const [activeSkill, setActiveSkill] = useState<Skill | null>(null)
  const [editSkill, setEditSkill] = useState<Skill | null>(null)
  const [panelOpen, setPanelOpen] = useState(false)
  const [validationError, setValidationError] = useState<string | null>(null)

  const handleOpenSkill = (skill: Skill) => {
    setActiveSkill(skill)
    setEditSkill(skill)
    setValidationError(null)
    setTimeout(() => setPanelOpen(true), 10)
  }

  const handleCloseSkill = () => {
    setPanelOpen(false)
    setValidationError(null)
    setTimeout(() => {
      setActiveSkill(null)
      setEditSkill(null)
    }, 300)
  }

  const updateEditSkill = <K extends keyof Skill>(key: K, value: Skill[K]) => {
    setEditSkill(prev => prev ? { ...prev, [key]: value } : null)
  }

  const validateSkill = (skill: Skill): string | null => {
    if (!skill.name.trim()) return "Skill name is required."
    if (!skill.command.trim()) return "Slash command is required."
    if (!/^[a-zA-Z0-9_-]+$/.test(skill.command)) return "Command must contain only alphanumeric characters, underscores, or hyphens."
    if (!skill.systemPrompt.trim()) return "System prompt is required."
    if (skill.temperature !== undefined && skill.temperature !== null && (skill.temperature < 0 || skill.temperature > 2)) {
      return "Temperature must be between 0.0 and 2.0."
    }
    // Check command uniqueness (case insensitive)
    const commandConflict = skills.some(s => s.command.toLowerCase() === skill.command.toLowerCase() && s.id !== skill.id)
    if (commandConflict) return `A skill with command /${skill.command} already exists.`
    return null
  }

  const handleSaveSkill = async () => {
    if (!editSkill) return
    const error = validateSkill(editSkill)
    if (error) {
      setValidationError(error)
      return
    }
    setValidationError(null)
    try {
      const finalSkill: Skill = {
        ...editSkill,
        id: editSkill.id || `custom-${Date.now()}`,
        command: editSkill.command.toLowerCase().trim(),
      }
      await saveSkill(finalSkill)
      handleCloseSkill()
    } catch (err: any) {
      setValidationError(err.message || "Failed to save skill.")
    }
  }

  const handleDeleteSkill = async () => {
    if (!editSkill?.id) return
    if (!confirm(`Are you sure you want to delete the skill "${editSkill.name}"?`)) return
    try {
      await deleteSkill(editSkill.id)
      handleCloseSkill()
    } catch (err: any) {
      setValidationError(err.message || "Failed to delete skill.")
    }
  }

  const handleToggleSkill = async (skill: Skill) => {
    try {
      const updatedSkill: Skill = {
        ...skill,
        enabled: skill.enabled === false ? true : false,
      }
      await saveSkill(updatedSkill)
    } catch (err) {
      console.error("Failed to toggle skill state:", err)
    }
  }

  // Sync when settings load from server
  useEffect(() => { setLocalSettings(settings) }, [settings])

  const updateSetting = <K extends keyof typeof localSettings>(key: K, value: typeof localSettings[K]) => {
    setLocalSettings(prev => ({ ...prev, [key]: value }))
    setDirty(true)
  }

  const handleSave = async () => {
    setSaving(true)
    try {
      await saveSettings(localSettings)
      setDirty(false)
      setSavedFeedback(true)
      setTimeout(() => setSavedFeedback(false), 2000)
    } finally {
      setSaving(false)
    }
  }

  const handleReset = () => {
    const defaults = {
      temperature: 0.7, topP: 0.9, topK: 40, maxTokens: 4096, repeatPenalty: 1.1,
      systemPrompt: 'You are a helpful AI assistant.',
    }
    setLocalSettings(prev => ({ ...prev, ...defaults }))
    setDirty(true)
  }

  return (
    <div className="flex h-screen w-full bg-background overflow-hidden">
      {/* ── Left sidebar ── */}
      <nav className="w-56 shrink-0 flex flex-col border-r border-border bg-card/40 px-3 py-4 gap-1">
        {/* Back */}
        <button
          onClick={() => router.push('/')}
          className="flex items-center gap-2 px-3 py-2 text-sm text-muted-foreground hover:text-foreground hover:bg-secondary rounded-lg transition-all mb-3"
        >
          <ChevronLeft size={15} />
          <span>Back</span>
        </button>

        <p className="text-[10px] font-bold text-muted-foreground/60 uppercase tracking-widest px-3 mb-1">
          Settings
        </p>

        {TABS.map(tab => {
          const Icon = tab.icon
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={cn(
                'flex items-center gap-2.5 px-3 py-2.5 rounded-xl text-sm font-medium transition-all text-left',
                activeTab === tab.id
                  ? 'bg-primary/10 text-primary border border-primary/20'
                  : 'text-muted-foreground hover:bg-secondary hover:text-foreground'
              )}
            >
              <Icon size={15} />
              {tab.label}
            </button>
          )
        })}
      </nav>

      {/* ── Content area ── */}
      <main className="flex-1 min-w-0 overflow-y-auto">
        <div className={cn(
          "mx-auto px-6 py-8 space-y-6 transition-all duration-300",
          activeTab === 'skills' ? 'max-w-5xl' : 'max-w-2xl'
        )}>

          {/* ── Generation Config ── */}
          {activeTab === 'config' && (
            <>
              <div className="flex items-center justify-between">
                <div>
                  <h1 className="text-xl font-bold text-foreground">Generation Settings</h1>
                  <p className="text-sm text-muted-foreground mt-0.5">
                    Controls how the model generates responses
                  </p>
                </div>
                <div className="flex gap-2">
                  <button
                    onClick={handleReset}
                    className="flex items-center gap-1.5 px-3 py-2 text-sm rounded-lg border border-border text-muted-foreground hover:bg-secondary transition-all"
                  >
                    <RotateCcw size={13} /> Reset
                  </button>
                  <button
                    onClick={handleSave}
                    disabled={!dirty || saving}
                    className={cn(
                      'flex items-center gap-1.5 px-4 py-2 text-sm rounded-lg transition-all font-medium',
                      dirty
                        ? 'bg-primary text-primary-foreground hover:bg-primary/90'
                        : 'bg-secondary text-muted-foreground cursor-not-allowed'
                    )}
                  >
                    {saving ? <Loader2 size={13} className="animate-spin" /> :
                      savedFeedback ? <Check size={13} /> : <Save size={13} />}
                    {savedFeedback ? 'Saved!' : 'Save'}
                  </button>
                </div>
              </div>

              <div className="space-y-6 bg-card/50 border border-border rounded-xl p-5">
                <SettingSlider
                  label="Temperature" value={localSettings.temperature} min={0} max={2} step={0.05}
                  onChange={v => updateSetting('temperature', v)}
                  format={v => v.toFixed(2)}
                />
                <SettingSlider
                  label="Top P" value={localSettings.topP} min={0} max={1} step={0.05}
                  onChange={v => updateSetting('topP', v)}
                  format={v => v.toFixed(2)}
                />
                <SettingSlider
                  label="Top K" value={localSettings.topK} min={1} max={100} step={1}
                  onChange={v => updateSetting('topK', v)}
                />
                <SettingSlider
                  label="Max Tokens" value={localSettings.maxTokens} min={256} max={32768} step={256}
                  onChange={v => updateSetting('maxTokens', v)}
                  format={v => v.toLocaleString()}
                />
                <SettingSlider
                  label="Repeat Penalty" value={localSettings.repeatPenalty} min={1} max={2} step={0.05}
                  onChange={v => updateSetting('repeatPenalty', v)}
                  format={v => v.toFixed(2)}
                />
              </div>
            </>
          )}

          {/* ── Prompt Presets ── */}
          {activeTab === 'presets' && (
            <>
              <div>
                <h1 className="text-xl font-bold text-foreground">Prompt Presets</h1>
                <p className="text-sm text-muted-foreground mt-0.5">
                  Choose a preset to instantly set the system prompt, or write your own below
                </p>
              </div>

              {/* Preset grid */}
              <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
                {SYSTEM_PRESETS.map(p => (
                  <button
                    key={p.label}
                    onClick={() => {
                      updateSetting('systemPrompt', p.prompt)
                    }}
                    className={cn(
                      'text-left px-3 py-2.5 rounded-xl border text-sm font-medium transition-all',
                      localSettings.systemPrompt === p.prompt
                        ? 'border-primary/50 bg-primary/10 text-primary'
                        : 'border-border bg-card/50 text-muted-foreground hover:border-border/80 hover:bg-secondary/60 hover:text-foreground'
                    )}
                  >
                    {p.label}
                  </button>
                ))}
              </div>

              {/* Editable textarea */}
              <div className="space-y-2">
                <label className="text-xs font-semibold text-muted-foreground uppercase tracking-widest">
                  System Prompt
                </label>
                <textarea
                  value={localSettings.systemPrompt}
                  onChange={e => updateSetting('systemPrompt', e.target.value)}
                  rows={8}
                  className="w-full bg-secondary/40 border border-border rounded-xl px-4 py-3 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-primary/40 resize-y font-mono leading-relaxed"
                  placeholder="You are a helpful AI assistant..."
                />
                <p className="text-xs text-muted-foreground">
                  Injected as the system message for every new conversation.
                </p>
              </div>

              <button
                onClick={handleSave}
                disabled={!dirty || saving}
                className={cn(
                  'flex items-center gap-1.5 px-4 py-2 text-sm rounded-lg transition-all font-medium',
                  dirty
                    ? 'bg-primary text-primary-foreground hover:bg-primary/90'
                    : 'bg-secondary text-muted-foreground cursor-not-allowed'
                )}
              >
                {saving ? <Loader2 size={13} className="animate-spin" /> :
                  savedFeedback ? <Check size={13} /> : <Save size={13} />}
                {savedFeedback ? 'Saved!' : 'Save Prompt'}
              </button>
            </>
          )}

          {/* ── AI Providers ── */}
          {activeTab === 'providers' && (
            <>
              <div className="flex items-center justify-between">
                <div>
                  <h1 className="text-xl font-bold text-foreground">AI Providers</h1>
                  <p className="text-sm text-muted-foreground mt-0.5">
                    Connect external providers - models appear in the model selector
                  </p>
                </div>
                <button
                  onClick={() => setShowAddProvider(p => !p)}
                  className="flex items-center gap-1.5 px-3 py-2 text-sm rounded-lg bg-primary text-primary-foreground hover:bg-primary/90 transition-all"
                >
                  <Plus size={14} />
                  Add Provider
                </button>
              </div>

              {/* Add provider form */}
              {showAddProvider && (
                <ProviderForm
                  onSave={async p => { await saveProvider(p); setShowAddProvider(false) }}
                  onCancel={() => setShowAddProvider(false)}
                />
              )}

              {/* Provider list */}
              <div className="space-y-3">
                {providers.length === 0 && !showAddProvider && (
                  <div className="flex flex-col items-center justify-center py-16 gap-3 text-center border border-dashed border-border/50 rounded-xl">
                    <Globe size={32} className="text-muted-foreground/40" />
                    <p className="text-sm text-muted-foreground">No providers configured yet</p>
                    <button
                      onClick={() => setShowAddProvider(true)}
                      className="text-xs text-primary hover:underline"
                    >
                      Add your first provider
                    </button>
                  </div>
                )}
                {providers.map(p => (
                  <ProviderCard
                    key={p.id}
                    provider={p}
                    onUpdate={saveProvider}
                    onDelete={removeProvider}
                  />
                ))}
              </div>

              {providers.length > 0 && (
                <p className="text-xs text-muted-foreground bg-secondary/40 rounded-lg px-4 py-3 border border-border/50">
                  <strong className="text-foreground">Tip:</strong> Enable individual models by expanding a provider and fetching its model list. Enabled models appear in the model selector with a provider tag.
                </p>
              )}
            </>
          )}

          {/* ── Skills Tab ── */}
          {activeTab === 'skills' && (
            <>
              <div className="flex items-center justify-between">
                <div>
                  <h1 className="text-xl font-bold text-foreground">Skills Settings</h1>
                  <p className="text-sm text-muted-foreground mt-0.5">
                    Configure and manage active slash-command skills
                  </p>
                </div>
                <button
                  onClick={() => handleOpenSkill(emptySkill)}
                  className="flex items-center gap-1.5 px-3 py-2 text-sm rounded-lg bg-primary text-primary-foreground hover:bg-primary/90 transition-all font-semibold cursor-pointer"
                >
                  <Plus size={14} />
                  New Skill
                </button>
              </div>

              {/* Grid Layout */}
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {/* "+" New Skill card */}
                <button
                  onClick={() => handleOpenSkill(emptySkill)}
                  className="border border-dashed border-border hover:border-primary/50 bg-secondary/10 hover:bg-primary/5 rounded-xl p-4 flex flex-col items-center justify-center h-44 transition-all duration-300 group text-muted-foreground hover:text-primary cursor-pointer"
                >
                  <div className="w-9 h-9 rounded-full border border-dashed border-border group-hover:border-primary/50 flex items-center justify-center mb-2">
                    <Plus size={16} />
                  </div>
                  <span className="text-xs font-semibold">Add Custom Skill</span>
                  <span className="text-[10px] opacity-60 mt-0.5">Build custom commands</span>
                </button>

                {skills.map(skill => {
                  const IconComponent = (Icons as any)[skill.icon] || Icons.Sparkles
                  const colors = CATEGORY_COLORS[skill.category] || { text: "text-primary", bg: "bg-primary/10 border-primary/20", border: "border-primary/20" }

                  return (
                    <div
                      key={skill.id}
                      onClick={() => handleOpenSkill(skill)}
                      className={cn(
                        "text-left border bg-card/40 rounded-xl p-4 flex flex-col justify-between h-44 hover:border-primary/50 hover:bg-secondary/20 transition-all duration-300 group shadow-md hover:shadow-lg relative overflow-hidden cursor-pointer",
                        skill.enabled === false ? "opacity-60 border-border/40" : "border-border"
                      )}
                    >
                      <div>
                        <div className="flex items-center justify-between gap-2 mb-3">
                          <div className={cn("w-8 h-8 rounded-lg flex items-center justify-center border", colors.bg, colors.text, colors.border)}>
                            <IconComponent size={16} />
                          </div>
                          <div className="flex items-center gap-1.5">
                            <span className="text-[10px] font-mono bg-secondary px-1.5 py-0.5 rounded text-muted-foreground/90 font-semibold">
                              /{skill.command}
                            </span>
                            {skill.isPremium && (
                              <span className="text-[8px] tracking-wide uppercase px-1 py-0.2 bg-primary/15 text-primary border border-primary/20 rounded font-bold scale-90">
                                PRO
                              </span>
                            )}
                            <span className={cn("text-[9px] font-bold uppercase tracking-wider px-1.5 py-0.5 rounded scale-90", colors.bg, colors.text)}>
                              {skill.category}
                            </span>
                            {/* Toggle Switch */}
                            <button
                              type="button"
                              onClick={(e) => {
                                e.stopPropagation();
                                handleToggleSkill(skill);
                              }}
                              className={cn(
                                "relative inline-flex h-4.5 w-8.5 shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none focus:ring-0 ml-1.5 z-10",
                                skill.enabled !== false ? "bg-primary" : "bg-neutral-700"
                              )}
                              title={skill.enabled !== false ? "Disable skill" : "Enable skill"}
                            >
                              <span
                                className={cn(
                                  "pointer-events-none inline-block h-3.5 w-3.5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out",
                                  skill.enabled !== false ? "translate-x-4" : "translate-x-0"
                                )}
                              />
                            </button>
                          </div>
                        </div>

                        <h3 className="font-bold text-sm text-foreground group-hover:text-primary transition-colors truncate">
                          {skill.name}
                        </h3>
                        <p className="text-xs text-muted-foreground line-clamp-2 mt-1 leading-relaxed">
                          {skill.description || "No description provided."}
                        </p>
                      </div>

                      <div className="mt-3 pt-3 border-t border-border/40 flex items-center justify-between text-[10px] text-muted-foreground/75 font-semibold">
                        <span>Temp: {skill.temperature ?? 0.7}</span>
                        <span>Tokens: {skill.maxTokens ?? 'Max'}</span>
                      </div>
                    </div>
                  )
                })}
              </div>
            </>
          )}

        </div>
      </main>

      {/* Sliding Sidepanel Drawer for Skill Editor */}
      {activeSkill && (
        <div className="fixed inset-0 z-50 flex justify-end">
          {/* Backdrop */}
          <div
            className={cn(
              "absolute inset-0 bg-black/60 backdrop-blur-sm transition-opacity duration-300",
              panelOpen ? "opacity-100" : "opacity-0"
            )}
            onClick={handleCloseSkill}
          />
          {/* Drawer */}
          <div
            className={cn(
              "relative w-full max-w-xl h-full bg-[#181818] border-l border-border shadow-2xl flex flex-col transition-all duration-300 ease-in-out transform z-10",
              panelOpen ? "translate-x-0" : "translate-x-full"
            )}
          >
            {/* Header */}
            <div className="px-6 py-5 border-b border-border/60 flex items-center justify-between bg-card/45 select-none">
              <div>
                <h2 className="text-base font-bold text-foreground">
                  {editSkill?.id ? `Edit Skill: ${editSkill.name}` : "Create Custom Skill"}
                </h2>
                <p className="text-xs text-muted-foreground mt-0.5">
                  Configure trigger, instructions, parameters, and aesthetics.
                </p>
              </div>
              <button
                onClick={handleCloseSkill}
                className="p-1 rounded-lg hover:bg-secondary text-muted-foreground hover:text-foreground transition-all cursor-pointer"
              >
                <X size={16} />
              </button>
            </div>

            {/* Scrollable Form */}
            <div className="flex-1 overflow-y-auto p-6 space-y-5">
              {validationError && (
                <div className="text-xs text-destructive bg-destructive/10 border border-destructive/20 px-3 py-2 rounded-lg leading-relaxed select-none animate-in fade-in duration-200">
                  {validationError}
                </div>
              )}

              {/* Name & Command */}
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="text-xs font-semibold text-muted-foreground uppercase tracking-widest block mb-1.5">
                    Skill Name
                  </label>
                  <input
                    type="text"
                    value={editSkill?.name || ''}
                    onChange={e => updateEditSkill('name', e.target.value)}
                    placeholder="e.g. Translator"
                    className="w-full bg-secondary/60 border border-border rounded-lg px-3 py-2 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-primary/50"
                  />
                </div>
                <div>
                  <label className="text-xs font-semibold text-muted-foreground uppercase tracking-widest block mb-1.5">
                    Slash Command
                  </label>
                  <div className="relative">
                    <span className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground text-sm font-mono font-bold">/</span>
                    <input
                      type="text"
                      value={editSkill?.command || ''}
                      onChange={e => updateEditSkill('command', e.target.value)}
                      placeholder="translate"
                      className="w-full bg-secondary/60 border border-border rounded-lg pl-6 pr-3 py-2 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-primary/50 font-mono"
                    />
                  </div>
                </div>
              </div>

              {/* Category & Selectable Icons */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="text-xs font-semibold text-muted-foreground uppercase tracking-widest block mb-1.5">
                    Category
                  </label>
                  <select
                    value={editSkill?.category || 'Writing'}
                    onChange={e => updateEditSkill('category', e.target.value as any)}
                    className="w-full bg-secondary/60 border border-border rounded-lg px-3 py-2 text-sm text-foreground focus:outline-none focus:ring-1 focus:ring-primary/50 cursor-pointer"
                  >
                    <option value="Writing">Writing</option>
                    <option value="Development">Development</option>
                    <option value="Research">Research</option>
                    <option value="Documents">Documents</option>
                    <option value="Media">Media</option>
                  </select>
                </div>
                
                <div>
                  <label className="text-xs font-semibold text-muted-foreground uppercase tracking-widest block mb-1.5">
                    Aesthetics: Active Icon
                  </label>
                  <div className="flex items-center gap-3 bg-secondary/20 border border-border/60 rounded-xl px-3 py-2">
                    <div className="w-8 h-8 rounded-lg flex items-center justify-center bg-primary/10 text-primary border border-primary/20 font-semibold">
                      {React.createElement((Icons as any)[editSkill?.icon || 'Sparkles'] || Icons.Sparkles, { size: 16 })}
                    </div>
                    <div className="text-xs">
                      <span className="font-bold text-foreground block">{editSkill?.icon || 'Sparkles'}</span>
                      <span className="text-[10px] text-muted-foreground">Lucide icon</span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Visual Icon Grid Selection */}
              <div>
                <label className="text-xs font-semibold text-muted-foreground uppercase tracking-widest block mb-1.5">
                  Pick an Icon
                </label>
                <div className="grid grid-cols-10 gap-1.5 border border-border/60 bg-secondary/20 rounded-xl p-2.5 max-h-36 overflow-y-auto">
                  {AVAILABLE_ICONS.map(iconName => {
                    const isSelected = editSkill?.icon === iconName;
                    return (
                      <button
                        key={iconName}
                        type="button"
                        onClick={() => updateEditSkill('icon', iconName)}
                        className={cn(
                          "p-2 rounded-lg border flex items-center justify-center transition-all hover:bg-secondary cursor-pointer",
                          isSelected
                            ? "border-primary bg-primary/15 text-primary"
                            : "border-transparent text-muted-foreground hover:text-foreground"
                        )}
                        title={iconName}
                      >
                        {React.createElement((Icons as any)[iconName] || Icons.HelpCircle, { size: 14 })}
                      </button>
                    )
                  })}
                </div>
              </div>

              {/* Description */}
              <div>
                <label className="text-xs font-semibold text-muted-foreground uppercase tracking-widest block mb-1.5">
                  Description
                </label>
                <input
                  type="text"
                  value={editSkill?.description || ''}
                  onChange={e => updateEditSkill('description', e.target.value)}
                  placeholder="Summarize the action..."
                  className="w-full bg-secondary/60 border border-border rounded-lg px-3 py-2 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-primary/50"
                />
              </div>

              {/* Parameters */}
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="text-xs font-semibold text-muted-foreground uppercase tracking-widest block mb-1.5">
                    Temperature (0.0 - 2.0)
                  </label>
                  <input
                    type="number"
                    step={0.1}
                    min={0}
                    max={2}
                    value={editSkill?.temperature !== undefined && editSkill.temperature !== null ? editSkill.temperature : ''}
                    onChange={e => {
                      const val = e.target.value === '' ? undefined : Number(e.target.value);
                      updateEditSkill('temperature', val);
                    }}
                    placeholder="0.7 (Default)"
                    className="w-full bg-secondary/60 border border-border rounded-lg px-3 py-2 text-sm text-foreground focus:outline-none focus:ring-1 focus:ring-primary/50"
                  />
                </div>
                <div>
                  <label className="text-xs font-semibold text-muted-foreground uppercase tracking-widest block mb-1.5">
                    Max Tokens
                  </label>
                  <input
                    type="number"
                    step={256}
                    min={256}
                    max={32768}
                    value={editSkill?.maxTokens !== undefined && editSkill.maxTokens !== null ? editSkill.maxTokens : ''}
                    onChange={e => {
                      const val = e.target.value === '' ? undefined : Number(e.target.value);
                      updateEditSkill('maxTokens', val);
                    }}
                    placeholder="4096 (Default)"
                    className="w-full bg-secondary/60 border border-border rounded-lg px-3 py-2 text-sm text-foreground focus:outline-none focus:ring-1 focus:ring-primary/50"
                  />
                </div>
              </div>

              {/* System Prompt (character & word count) */}
              <div className="space-y-1.5">
                <div className="flex justify-between items-center select-none">
                  <label className="text-xs font-semibold text-muted-foreground uppercase tracking-widest">
                    System Prompt / Instructions
                  </label>
                  <span className="text-[10px] text-muted-foreground/85 font-semibold font-mono bg-secondary px-2 py-0.5 rounded border border-border/40">
                    {editSkill?.systemPrompt ? editSkill.systemPrompt.trim().split(/\s+/).filter(Boolean).length : 0} words / {editSkill?.systemPrompt?.length || 0} chars
                  </span>
                </div>
                <textarea
                  value={editSkill?.systemPrompt || ''}
                  onChange={e => updateEditSkill('systemPrompt', e.target.value)}
                  rows={8}
                  className="w-full bg-secondary/40 border border-border rounded-xl px-4 py-3 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-primary/40 resize-y font-mono leading-relaxed overflow-y-auto"
                  placeholder="Enter custom instructions or system prompt..."
                />
              </div>

              {/* Toggles */}
              <div className="flex items-center gap-6 pt-2 select-none border-t border-border/40">
                <label className="flex items-center gap-2.5 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={editSkill?.enabled ?? true}
                    onChange={e => updateEditSkill('enabled', e.target.checked)}
                    className="rounded bg-secondary border-border text-primary focus:ring-0 focus:ring-offset-0 w-4 h-4 cursor-pointer"
                  />
                  <span className="text-xs font-semibold text-foreground/80">Enabled</span>
                </label>
                <label className="flex items-center gap-2.5 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={editSkill?.isPremium ?? false}
                    onChange={e => updateEditSkill('isPremium', e.target.checked)}
                    className="rounded bg-secondary border-border text-primary focus:ring-0 focus:ring-offset-0 w-4 h-4 cursor-pointer"
                  />
                  <span className="text-xs font-semibold text-foreground/80">Pro Level Skill</span>
                </label>
              </div>
            </div>

            {/* Footer */}
            <div className="px-6 py-4 border-t border-border/60 flex items-center justify-between bg-card/25 select-none">
              {editSkill?.id ? (
                <button
                  type="button"
                  onClick={handleDeleteSkill}
                  className="flex items-center gap-1.5 px-3 py-2 text-xs rounded-lg border border-destructive/30 hover:border-destructive text-destructive hover:bg-destructive/10 transition-all font-semibold cursor-pointer"
                >
                  <Trash2 size={13} /> Delete Skill
                </button>
              ) : <div />}
              
              <div className="flex gap-2">
                <button
                  type="button"
                  onClick={handleCloseSkill}
                  className="px-4 py-2 text-xs font-semibold rounded-lg border border-border text-muted-foreground hover:bg-secondary transition-all cursor-pointer"
                >
                  Cancel
                </button>
                <button
                  type="button"
                  onClick={handleSaveSkill}
                  className="flex items-center gap-1.5 px-4 py-2 text-xs font-bold rounded-lg bg-primary text-primary-foreground hover:bg-primary/90 transition-all cursor-pointer"
                >
                  <Save size={13} /> Save Skill
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
