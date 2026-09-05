"use client"

import { createContext, useContext, useState, useCallback, useEffect, useRef, type ReactNode } from "react"
import { DEFAULT_SETTINGS, type View, type Conversation, type ChatMessage, type AIModel, type GenerationSettings } from "@/services/data"
import { API, Model, Conversation as ApiConversation } from "@/services/api"
import { type Skill, ALL_SKILLS } from "@/services/skills"
import {
  type ProviderConfig,
  loadAllProviders,
  saveProvider as saveProviderToDb,
  deleteProvider as deleteProviderFromDb,
  providerModelsToAIModels,
  isProviderModel,
} from "@/services/providers"

interface AppContextType {
  // Navigation
  currentView: View
  setCurrentView: (view: View) => void

  // Conversations
  conversations: Conversation[]
  activeConversationId: string | null
  setActiveConversationId: (id: string | null) => void
  createConversation: () => Promise<string>
  deleteConversation: (id: string) => void
  addMessage: (conversationId: string, message: ChatMessage) => void
  setConversationMessages: (conversationId: string, messages: ChatMessage[]) => void
  linkConversation: (tempId: string, realId: string) => void
  refreshConversations: () => Promise<void>

  // Local Models
  models: AIModel[]          // All models (local + provider)
  localModels: AIModel[]     // Downloaded/on-disk models only
  activeModelId: string
  setActiveModelId: (id: string) => void
  selectedModelDetail: AIModel | null
  setSelectedModelDetail: (model: AIModel | null) => void
  refreshModels: () => Promise<void>

  // External Providers
  providers: ProviderConfig[]
  refreshProviders: () => Promise<void>
  saveProvider: (p: ProviderConfig) => Promise<void>
  removeProvider: (id: string) => Promise<void>

  // Settings
  settings: GenerationSettings
  setSettings: (s: GenerationSettings) => void
  saveSettings: (s: GenerationSettings) => Promise<void>
  settingsOpen: boolean
  setSettingsOpen: (v: boolean) => void

  // UI
  isGenerating: boolean
  setIsGenerating: (v: boolean) => void
  isModelLoading: boolean
  setIsModelLoading: (v: boolean) => void

  // User
  user: { name: string, role: string, initials: string }

  // Knowledge / RAG
  ragEnabled: boolean
  setRagEnabled: (v: boolean) => void

  // Skills CRUD
  skills: Skill[]
  refreshSkills: () => Promise<void>
  saveSkill: (skill: Skill) => Promise<void>
  deleteSkill: (id: string) => Promise<void>

  // Background Tasks
  backgroundTasks: Array<{ id: string; label: string }>
  addBackgroundTask: (id: string, label: string) => void
  removeBackgroundTask: (id: string) => void
}

const AppContext = createContext<AppContextType | null>(null)

export function AppProvider({ children }: { children: ReactNode }) {
  const [currentView, setCurrentView] = useState<View>("chat")
  const [conversations, setConversations] = useState<Conversation[]>([])
  const [activeConversationId, setActiveConversationId] = useState<string | null>(null)
  const [models, setModels] = useState<AIModel[]>([])
  const [localModels, setLocalModels] = useState<AIModel[]>([])
  const [providers, setProviders] = useState<ProviderConfig[]>([])
  const [activeModelId, setActiveModelId] = useState<string>("")
  const [selectedModelDetail, setSelectedModelDetail] = useState<AIModel | null>(null)
  const [settings, setSettings] = useState<GenerationSettings>(DEFAULT_SETTINGS)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [isGenerating, setIsGenerating] = useState(false)
  const [isModelLoading, setIsModelLoading] = useState(false)
  const [user, setUser] = useState({ name: "User", role: "Researcher", initials: "TR" })
  const [ragEnabled, setRagEnabled] = useState(false)
  const [skills, setSkills] = useState<Skill[]>(ALL_SKILLS)
  const [backgroundTasks, setBackgroundTasks] = useState<Array<{ id: string; label: string }>>([])

  const initialLoadAttempted = useRef(false)
  const isClient = useRef(false)

  // Initialization from localStorage (Client only)
  useEffect(() => {
    isClient.current = true;
    const savedView = localStorage.getItem('oprel_current_view') as View;
    if (savedView) setCurrentView(savedView);

    const savedModel = localStorage.getItem('oprel_active_model');
    if (savedModel) setActiveModelId(savedModel);

    const savedConv = localStorage.getItem('oprel_active_conv');
    if (savedConv) {
      setActiveConversationId(savedConv);
    }
  }, []);

  // Persistence Sync
  useEffect(() => {
    if (!isClient.current) return;
    localStorage.setItem('oprel_current_view', currentView);
  }, [currentView]);

  useEffect(() => {
    if (!isClient.current) return;
    if (activeModelId) {
      localStorage.setItem('oprel_active_model', activeModelId);
    }
  }, [activeModelId]);

  useEffect(() => {
    if (!isClient.current) return;
    if (activeConversationId) {
      localStorage.setItem('oprel_active_conv', activeConversationId);
    } else {
      localStorage.removeItem('oprel_active_conv');
    }
  }, [activeConversationId]);

  useEffect(() => {
    if (!isClient.current) return;
    const saved = localStorage.getItem('oprel_rag_enabled');
    if (saved !== null) setRagEnabled(saved === 'true');
  }, []);

  useEffect(() => {
    if (!isClient.current) return;
    localStorage.setItem('oprel_rag_enabled', ragEnabled ? 'true' : 'false');
  }, [ragEnabled]);

  const mapApiModels = (apiModels: Model[]): AIModel[] => {
    return apiModels.map(m => ({
      id: m.model_id,
      name: m.name || m.model_id,
      family: m.model_id.split('/')[0],
      size: m.size_gb ? `${m.size_gb.toFixed(1)} GB` : "Unknown",
      quantization: m.quantization || "Unknown",
      contextLength: 0, // Not provided by simple list
      ramRequired: 0,
      status: m.loaded ? "loaded" : "available",
      tags: [],
      description: "",
      publisher: m.model_id.split('/')[0],
      architecture: m.backend || "llama.cpp",
      parameters: "Unknown",
      license: "Unknown",
      compatibility: "compatible",
      speed: m.status === 'loaded' ? 'Active' : undefined
    }));
  };

  const refreshModels = useCallback(async () => {
    try {
      // 1. Fetch Local Models (Registry & Loaded status)
      const apiModels = await API.fetchModels();
      const mappedLocalRegistry: AIModel[] = apiModels.map(m => ({
        id: m.model_id,
        name: m.name || m.model_id,
        family: m.name.includes('-') ? m.name.split('-')[0] : m.name,
        size: "—",
        quantization: "Managed",
        contextLength: 0,
        ramRequired: 0,
        status: m.status as any,
        tags: m.tags || [],
        description: `${m.category || 'Text'} model`,
        publisher: m.model_id.includes('/') ? m.model_id.split('/')[0] : (m.name || '').split('-')[0],
        architecture: m.backend || "llama.cpp",
        parameters: "Unknown",
        license: "Proprietary",
        compatibility: "compatible",
        downloaded: !!m.downloaded,
        category: m.category,
      }));

      // 2. Fetch specific GGUF/Quantized local models for the switchers
      const perQuantLocal = await API.fetchLocalModels().catch(() => []);
      const mappedQuants: AIModel[] = (perQuantLocal || [])
        .filter(m => m.quantization && m.quantization !== 'Unknown' && m.quantization !== 'API')
        .filter(m => !isProviderModel(m.model_id))
        .map(m => ({
          id: `${m.model_id}::${m.quantization}`,
          name: m.name,
          family: (m.name || m.model_id).split('-')[0] || m.model_id,
          size: m.size_gb ? `${m.size_gb.toFixed(1)} GB` : "—",
          quantization: m.quantization || "Unknown",
          contextLength: 0,
          ramRequired: 0,
          status: m.loaded ? "loaded" as const : "available" as const,
          tags: [],
          description: "",
          publisher: (m.model_id || '').split('/')[0] || "",
          architecture: m.backend || "llama.cpp",
          parameters: "Unknown",
          license: "Proprietary",
          compatibility: "compatible" as const,
          speed: m.loaded ? 'Active' : undefined,
          downloaded: true,
          modelRepoId: m.model_id,
          category: m.category,
        }));

      // Deduplicate quants
      const uniqLocal = new Map<string, AIModel>();
      mappedQuants.forEach(m => {
        const existing = uniqLocal.get(m.id);
        if (!existing || (m.status === 'loaded' && existing.status !== 'loaded')) {
          uniqLocal.set(m.id, m);
        }
      });
      const finalLocal = Array.from(uniqLocal.values());
      setLocalModels(finalLocal);

      // 3. Fetch Cloud Providers and Merge
      const loadedProviders = await loadAllProviders().catch(() => []);
      setProviders(loadedProviders);
      const PROVIDER_TYPES = new Set(['openai', 'gemini', 'nvidia', 'groq', 'openrouter', 'openai-compatible']);
      const providerModels = providerModelsToAIModels(loadedProviders);

      // 4. Update Global Unified Models List
      setModels([...mappedLocalRegistry.filter(m => m.category !== 'external'), ...providerModels]);

      // 5. Active Model Logic
      const loadedInference = mappedLocalRegistry.find(m => m.status === 'loaded');
      const loadedInLocal = finalLocal.find(m => m.status === 'loaded');
      
      let candidateId = activeModelId;
      if (!candidateId && (loadedInference || loadedInLocal)) {
        candidateId = (loadedInLocal || loadedInference)!.id;
      }

      const candidate = finalLocal.find(m => m.id === candidateId) || 
                        finalLocal.find(m => m.id.split('::')[0] === candidateId) ||
                        mappedLocalRegistry.find(m => m.id === candidateId) ||
                        providerModels.find(m => m.id === candidateId) ||
                        loadedInLocal || loadedInference;

      if (candidate) {
        if (!activeModelId) setActiveModelId(candidate.id);
        setSelectedModelDetail(candidate);
      }

      // Auto-load Logic
      if (!loadedInLocal && !loadedInference && !initialLoadAttempted.current && candidate && candidate.downloaded && !candidate.id.includes('::')) {
        initialLoadAttempted.current = true;
        setIsModelLoading(true);
        API.loadModel(candidate.id).finally(() => {
          setIsModelLoading(false);
          refreshModels();
        });
      }
    } catch (error) {
      console.error("Failed to refresh models:", error);
    }
  }, [activeModelId]);

  // ── Provider refresh (now just calls unified refresh) ──────────────────────
  const refreshProviders = useCallback(async () => {
    await refreshModels();
  }, [refreshModels]);

  const saveProvider = useCallback(async (p: ProviderConfig) => {
    await saveProviderToDb(p)
    await refreshProviders()
  }, [refreshProviders])

  const removeProvider = useCallback(async (id: string) => {
    await deleteProviderFromDb(id)
    await refreshProviders()
  }, [refreshProviders])

  const refreshConversations = useCallback(async () => {
    try {
      const apiConvs = await API.fetchConversations();
      const emptyConversations = apiConvs.filter(c => c.message_count === 0);

      let emptyConversationToKeep: string | null = null;
      if (emptyConversations.length > 0) {
        emptyConversationToKeep =
          emptyConversations.find(c => c.id === activeConversationId)?.id ||
          [...emptyConversations]
            .sort((a, b) => new Date(b.last_updated || b.created_at).getTime() - new Date(a.last_updated || a.created_at).getTime())[0]
            .id;

        const redundantEmptyConversations = emptyConversations.filter(c => c.id !== emptyConversationToKeep);
        if (redundantEmptyConversations.length > 0) {
          await Promise.all(
            redundantEmptyConversations.map(c =>
              API.deleteConversation(c.id).catch(() => null)
            )
          );
        }
      }

      setConversations(prev => {
        // Build map of existing localized messages
        const messageMap = new Map(prev.map(c => [c.id, c.messages]));

        const updated = apiConvs
          .filter(c => c.message_count > 0 || c.id === emptyConversationToKeep)
          .map(c => {
          return {
            id: c.id,
            title: c.title || "New Chat",
            messages: messageMap.get(c.id) || [],
            messageCount: c.message_count,
            createdAt: new Date(c.created_at),
            updatedAt: new Date(c.last_updated || c.created_at),
            modelId: c.model_id
          };
        });

        return updated;
      });
    } catch (error) {
      console.error("Failed to fetch conversations:", error);
    }
  }, [activeConversationId]);

  const refreshSettings = useCallback(async () => {
    try {
      const apiSettings = await API.fetchSettings();
      setSettings({
        temperature: apiSettings.temperature,
        topP: apiSettings.top_p,
        topK: apiSettings.top_k,
        maxTokens: apiSettings.max_tokens,
        repeatPenalty: apiSettings.repeat_penalty,
        systemPrompt: apiSettings.system_instruction || DEFAULT_SETTINGS.systemPrompt
      });
    } catch (error) {
      console.error("Failed to fetch settings:", error);
    }
  }, []);

  const addBackgroundTask = useCallback((id: string, label: string) => {
    setBackgroundTasks(prev => {
      if (prev.some(t => t.id === id)) return prev
      return [...prev, { id, label }]
    })
  }, [])

  const removeBackgroundTask = useCallback((id: string) => {
    setBackgroundTasks(prev => prev.filter(t => t.id !== id))
  }, [])

  const refreshSkills = useCallback(async () => {
    try {
      const dbSkills = await API.fetchSkills()
      const mappedSkills: Skill[] = dbSkills.map((s: any) => ({
        id: s.id,
        name: s.name,
        description: s.description || "",
        command: s.command || s.id,
        icon: s.icon,
        category: s.category,
        systemPrompt: s.system_prompt,
        temperature: s.temperature ?? undefined,
        maxTokens: s.max_tokens ?? undefined,
        outputSchema: s.output_schema ?? undefined,
        isPremium: Boolean(s.is_premium),
        enabled: Boolean(s.enabled),
      }))
      setSkills(mappedSkills)
    } catch (err) {
      console.error("Failed to fetch skills, keeping defaults:", err)
    }
  }, [])

  const saveSkill = useCallback(async (skill: Skill) => {
    addBackgroundTask("save-skill", `Saving skill "${skill.name}" to database`)
    try {
      const dbPayload = {
        id: skill.id,
        name: skill.name,
        description: skill.description,
        command: skill.command,
        icon: skill.icon,
        category: skill.category,
        system_prompt: skill.systemPrompt,
        temperature: skill.temperature ?? null,
        max_tokens: skill.maxTokens ?? null,
        output_schema: skill.outputSchema ?? null,
        is_premium: skill.isPremium ? true : false,
        enabled: skill.enabled ?? true,
      }
      await API.saveSkill(dbPayload)
      await refreshSkills()
    } catch (err) {
      console.error("Failed to save skill:", err)
      throw err
    } finally {
      removeBackgroundTask("save-skill")
    }
  }, [refreshSkills, addBackgroundTask, removeBackgroundTask])

  const deleteSkill = useCallback(async (id: string) => {
    addBackgroundTask("delete-skill", "Deleting skill from database")
    try {
      await API.deleteSkill(id)
      await refreshSkills()
    } catch (err) {
      console.error("Failed to delete skill:", err)
      throw err
    } finally {
      removeBackgroundTask("delete-skill")
    }
  }, [refreshSkills, addBackgroundTask, removeBackgroundTask])

  const refreshUser = useCallback(async () => {
    try {
      const u = await API.fetchUser();
      setUser({
        name: u.name,
        role: u.role,
        initials: u.initials || u.name.split(' ').map(n => n[0]).join('').toUpperCase().slice(0, 2)
      });
    } catch {
      // Keep defaults
    }
  }, []);

  useEffect(() => {
    refreshModels();
    refreshConversations();
    refreshSettings();
    refreshUser();
    refreshSkills();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const createConversation = useCallback(async () => {
    const existingEmptyConversation = conversations.find(c => c.messageCount === 0);
    if (existingEmptyConversation) {
      setActiveConversationId(existingEmptyConversation.id);
      return existingEmptyConversation.id;
    }

    const created = await API.createConversation(activeModelId || 'chat');
    const now = new Date();
    const newConversation: Conversation = {
      id: created.id,
      title: created.title || 'New Chat',
      messages: [],
      messageCount: 0,
      createdAt: now,
      updatedAt: now,
      modelId: created.model_id,
    };

    setConversations(prev => {
      return [newConversation, ...prev.filter(c => c.id !== created.id)];
    });
    setActiveConversationId(created.id);
    return created.id;
  }, [activeModelId, conversations]);

  const deleteConversation = useCallback(async (id: string) => {
    try {
      await API.deleteConversation(id);
      setConversations((prev) => prev.filter((c) => c.id !== id));
      setActiveConversationId((prev) => (prev === id ? null : prev));
    } catch (error) {
      console.error("Failed to delete conversation:", error);
    }
  }, []);

  const saveSettingsToServer = useCallback(async (newSettings: GenerationSettings) => {
    try {
      await API.saveSettings({
        temperature: newSettings.temperature,
        top_p: newSettings.topP,
        top_k: newSettings.topK,
        max_tokens: newSettings.maxTokens,
        repeat_penalty: newSettings.repeatPenalty,
        system_instruction: newSettings.systemPrompt
      });
      setSettings(newSettings);
    } catch (error) {
      console.error("Failed to save settings:", error);
    }
  }, []);

  const addMessage = useCallback((conversationId: string, message: ChatMessage) => {
    setConversations((prev) => {
      const exists = prev.some(c => c.id === conversationId);
      if (!exists) {
        // Create new sidebar entry on first message (shouldn't normally happen now)
        const newConv: Conversation = {
          id: conversationId,
          title: message.role === 'user'
            ? (typeof message.content === 'string' ? message.content : '').slice(0, 50)
            : 'New Chat',
          messages: [message],
          messageCount: 1,
          createdAt: new Date(),
          updatedAt: new Date(),
          modelId: activeModelId,
        };
        return [newConv, ...prev];
      }

      return prev.map((c) => {
        if (c.id !== conversationId) return c
        const updated = {
          ...c,
          messages: [...c.messages, message],
          messageCount: Math.max(c.messageCount + 1, c.messages.length + 1),
          updatedAt: new Date(),
        }
        if ((!c.title || c.title === "New Chat") && message.role === "user") {
          const text = typeof message.content === 'string'
            ? message.content
            : (Array.isArray(message.content)
                ? message.content.find((p: any) => p.type === 'text')?.text || ''
                : '')
          updated.title = text.slice(0, 50) + (text.length > 50 ? "..." : "")
        }
        return updated
      })
    })
  }, [activeModelId])

  const linkConversation = useCallback((tempId: string, realId: string) => {
    setConversations((prev) =>
      prev.map((c) => {
        if (c.id !== tempId) return c
        return { ...c, id: realId }
      })
    )
    setActiveConversationId(realId)
  }, [])

  const setConversationMessages = useCallback((conversationId: string, messages: ChatMessage[]) => {
    setConversations((prev) =>
      prev.map((c) => {
        if (c.id !== conversationId) return c
        return { ...c, messages, messageCount: Math.max(c.messageCount, messages.length) }
      })
    )
  }, [])

  return (
    <AppContext.Provider
      value={{
        currentView,
        setCurrentView,
        conversations,
        activeConversationId,
        setActiveConversationId,
        createConversation,
        deleteConversation,
        addMessage,
        linkConversation,
        refreshConversations,
        models,
        activeModelId,
        setActiveModelId,
        selectedModelDetail,
        setSelectedModelDetail,
        refreshModels,
        setConversationMessages,
        localModels,
        providers,
        refreshProviders,
        saveProvider,
        removeProvider,
        settings,
        setSettings,
        saveSettings: saveSettingsToServer,
        settingsOpen,
        setSettingsOpen,
        isGenerating,
        setIsGenerating,
        isModelLoading,
        setIsModelLoading,
        user,
        ragEnabled,
        setRagEnabled,
        skills,
        refreshSkills,
        saveSkill,
        deleteSkill,
        backgroundTasks,
        addBackgroundTask,
        removeBackgroundTask
      }}
    >
      {children}
    </AppContext.Provider>
  )
}

export function useApp() {
  const ctx = useContext(AppContext)
  if (!ctx) throw new Error("useApp must be used within AppProvider")
  return ctx
}
