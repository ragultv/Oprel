export type View = "chat" | "images" | "models" | "dev" | "knowledge"

export type MessageRole = "user" | "assistant"

export interface ChatMessage {
  id: string
  role: MessageRole
  content: string | any[]
  timestamp: Date
  tps?: number
  skill?: {
    id: string
    name: string
  }
}

export interface Conversation {
  id: string
  title: string
  messages: ChatMessage[]
  createdAt: Date
  updatedAt?: Date
  modelId: string
}

export interface AIModel {
  id: string
  name: string
  family: string
  size: string
  quantization: string
  contextLength: number
  ramRequired: number
  status: "loaded" | "available" | "downloading" | "registry"
  downloadProgress?: number
  tags: string[]
  description: string
  publisher: string
  architecture: string
  parameters: string
  license: string
  compatibility: "compatible" | "hybrid" | "incompatible"
  speed?: string
  vramRequired?: number
  category?: string
  downloaded?: boolean
  modelRepoId?: string  // underlying HF repo_id, separate from composite id
}

export interface GenerationSettings {
  temperature: number
  topP: number
  topK: number
  maxTokens: number
  repeatPenalty: number
  systemPrompt: string
}

export const DEFAULT_SETTINGS: GenerationSettings = {
  temperature: 0.7,
  topP: 0.9,
  topK: 40,
  maxTokens: 4096,
  repeatPenalty: 1.1,
  systemPrompt: "You are a helpful AI assistant.",
}
