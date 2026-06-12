/**
 * API Service for Oprel SDK
 * Mirrors and enhances logic from legacy webui/js/api.js
 */

export interface Model {
  model_id: string;
  name: string;
  size_gb: number;
  quantization: string;
  backend: string;
  loaded: boolean;
  status: string;
  tags?: string[];
  category?: string;
  downloaded?: boolean;
}

export interface Conversation {
  id: string;
  title: string;
  created_at: string;
  last_updated: string;
  message_count: number;
  model_id: string;
}

export interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string | any[];
}

export interface Metrics {
  cpu_usage: number;
  ram_total_gb: number;
  ram_used_gb: number;
  gpu_name: string | null;
  gpu_usage: number | null;
  vram_total_mb: number | null;
  vram_used_mb: number | null;
  generation_speed: number;
}

export interface UserSettings {
  temperature: number;
  top_p: number;
  top_k: number;
  repeat_penalty: number;
  max_tokens: number;
  system_instruction: string | null;
}

export interface UserProfile {
  name: string;
  role: string;
  initials?: string;
}

export interface ImageGenerationData {
  url?: string;
  b64_json?: string;
  revised_prompt?: string;
}

export interface ImageGenerationResponse {
  created: number;
  data: ImageGenerationData[];
}

export interface ImageModel {
  id: string;
  repo_id: string;
  backend: string;
  downloaded: boolean;
  local_path?: string | null;
  quantization?: string | null;
  supported?: boolean;
  compatibility_reason?: string | null;
}

export interface ImageGenerationJob {
  id: string;
  status: 'queued' | 'running' | 'completed' | 'error';
  progress: number;
  message: string;
  created: number;
  error?: string | null;
  result?: ImageGenerationResponse;
}

export interface ModelDetailedInfo {
  repo_id: string;
  alias?: string;
  parameters: string;
  quantizations: string[];
  sizes: Record<string, number>;
  default_quantization: string | null;
}

const API_BASE = (typeof window !== 'undefined' && window.location.port === '3000') 
? 'http://localhost:11435' 
: ''; 

export const API = {
  async fetchModels(): Promise<Model[]> {
    const res = await fetch(`${API_BASE}/v1/models`);
    if (!res.ok) throw new Error('Failed to fetch models');
    const data = await res.json();
    
    // Convert OpenAI format to our internal Model format.
    // The server now returns alias names as `id` (e.g. "deepseek-r1-1.5b")
    // and sets `name` for unregistered local models.
    return data.data.map((m: any) => ({
      model_id: m.id,
      // Prefer server-supplied name (for unregistered models), otherwise use id (alias)
      name: m.name || m.id,
      size_gb: 0, 
      quantization: "Unknown",
      backend: "llama.cpp",
      loaded: !!m.loaded, 
      downloaded: !!m.downloaded,
      status: m.loaded ? 'loaded' : (m.downloaded ? 'available' : 'registry'),
      tags: m.tags || [],
      category: m.category || 'text-generation'
    }));
  },

  /** Fetch per-quant local models from /models endpoint (for Switch Model dropdown) */
  async fetchLocalModels(): Promise<Model[]> {
    try {
      const res = await fetch(`${API_BASE}/models`);
      if (!res.ok) return [];
      const data: any[] = await res.json();
      return data.map((m: any) => ({
        model_id: m.model_id || m.id || m.name,
        // Show alias + quant: e.g. "deepseek-r1-1.5b · Q8_0"
        name: m.quantization && m.quantization !== 'Unknown'
          ? `${m.name} · ${m.quantization}`
          : m.name,
        size_gb: m.size_gb || 0,
        quantization: m.quantization || 'Unknown',
        backend: m.backend || 'llama.cpp',
        loaded: !!m.loaded,
        status: m.loaded ? 'loaded' : 'available',
        downloaded: true,
        tags: [],
        category: m.category || 'text-generation',
      }));
    } catch {
      return [];
    }
  },

  async loadModel(modelId: string, params: any = {}): Promise<any> {
    const res = await fetch(`${API_BASE}/load`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model_id: modelId, ...params }),
    });
    if (!res.ok) throw new Error('Failed to load model');
    return res.json();
  },

  async unloadModel(modelId: string): Promise<any> {
    const res = await fetch(`${API_BASE}/unload/${encodeURIComponent(modelId)}`, {
      method: 'DELETE',
    });
    if (!res.ok) throw new Error('Failed to unload model');
    return res.json();
  },

  async deleteModelQuant(modelId: string, quantization: string): Promise<any> {
    const res = await fetch(`${API_BASE}/models/${encodeURIComponent(modelId)}/quant/${encodeURIComponent(quantization)}`, {
      method: 'DELETE',
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Failed to delete' }));
      throw new Error(err.detail || 'Failed to delete model');
    }
    return res.json();
  },

  async fetchDownloadLogs(limit = 100): Promise<any[]> {
    const res = await fetch(`${API_BASE}/download-logs?limit=${limit}`);
    if (!res.ok) return [];
    const data = await res.json();
    return data.logs || [];
  },

  async pullModel(modelId: string, quantization?: string): Promise<{
    success: boolean;
    model_id: string;
    quantization: string;
    download_id: string;
    message: string;
  }> {
    const res = await fetch(`${API_BASE}/pull`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        model_id: modelId,
        quantization: quantization 
      }),
    });
    if (!res.ok) throw new Error('Failed to start download');
    return res.json();
  },

  /**
   * Stream download progress via SSE
   */
  streamDownloadProgress(
    downloadId: string,
    onProgress: (progress: {
      model_id: string;
      quantization: string;
      status: string;
      progress: number;
      downloaded: number;
      total: number;
      speed: number;
      eta: number;
      error?: string;
    }) => void,
    onComplete: () => void,
    onError: (error: string) => void
  ): () => void {
    const eventSource = new EventSource(`${API_BASE}/downloads/progress?id=${encodeURIComponent(downloadId)}`);
    
    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        
        if (data.error) {
          onError(data.error);
          eventSource.close();
          return;
        }
        
        onProgress(data);
        
        if (data.status === 'completed') {
          onComplete();
          eventSource.close();
        } else if (data.status === 'error') {
          onError(data.error || 'Download failed');
          eventSource.close();
        }
      } catch (error) {
        console.warn('Error parsing SSE data:', error);
      }
    };
    
    eventSource.onerror = (error) => {
      // Suppress console.error, let the caller handle it via onError callback
      console.warn('SSE connection error - endpoint may not be available');
      onError('Connection lost. Please restart the server to enable real-time progress.');
      eventSource.close();
    };
    
    // Return cleanup function
    return () => {
      eventSource.close();
    };
  },

  async fetchConversations(): Promise<Conversation[]> {
    const res = await fetch(`${API_BASE}/conversations`);
    if (!res.ok) throw new Error('Failed to fetch conversations');
    return res.json();
  },

  async getConversation(id: string): Promise<ChatMessage[]> {
    const res = await fetch(`${API_BASE}/conversations/${id}`);
    if (!res.ok) throw new Error('Failed to load conversation');
    return res.json();
  },

  async deleteConversation(id: string): Promise<any> {
    const res = await fetch(`${API_BASE}/conversations/${id}`, {
      method: 'DELETE',
    });
    if (!res.ok) throw new Error('Failed to delete conversation');
    return res.json();
  },

  async renameConversation(id: string, title: string): Promise<any> {
    const res = await fetch(`${API_BASE}/conversations/${id}/title`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title }),
    });
    if (!res.ok) throw new Error('Failed to rename conversation');
    return res.json();
  },

  async getMetrics(): Promise<Metrics> {
    const res = await fetch(`${API_BASE}/system/metrics`);
    if (!res.ok) throw new Error('Failed to fetch metrics');
    return res.json();
  },
  
  async fetchAnalyticsSummary(days = 7): Promise<any> {
    const res = await fetch(`${API_BASE}/analytics/summary?days=${days}`);
    if (!res.ok) throw new Error('Failed to fetch analytics');
    return res.json();
  },

  async fetchRegistryModels(): Promise<any> {
    const res = await fetch(`${API_BASE}/registry/models`);
    if (!res.ok) throw new Error('Failed to fetch registry models');
    return res.json();
  },

  async fetchUser(): Promise<UserProfile> {
    const res = await fetch(`${API_BASE}/user`);
    if (!res.ok) throw new Error('Failed to fetch user');
    return res.json();
  },

  async saveUser(name: string, role: string): Promise<UserProfile> {
    const res = await fetch(`${API_BASE}/user`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, role }),
    });
    if (!res.ok) throw new Error('Failed to save user');
    return res.json();
  },

  async fetchSettings(): Promise<UserSettings> {
    const res = await fetch(`${API_BASE}/user/settings`);
    if (!res.ok) throw new Error('Failed to fetch settings');
    return res.json();
  },

  async saveSettings(settings: UserSettings): Promise<UserSettings> {
    const res = await fetch(`${API_BASE}/user/settings`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(settings),
    });
    if (!res.ok) throw new Error('Failed to save settings');
    return res.json();
  },

  async fetchSkills(): Promise<any[]> {
    const res = await fetch(`${API_BASE}/skills`);
    if (!res.ok) throw new Error('Failed to fetch skills');
    return res.json();
  },

  async saveSkill(skill: any): Promise<any> {
    const res = await fetch(`${API_BASE}/skills`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(skill),
    });
    if (!res.ok) throw new Error('Failed to save skill');
    return res.json();
  },

  async deleteSkill(skillId: string): Promise<any> {
    const res = await fetch(`${API_BASE}/skills/${encodeURIComponent(skillId)}`, {
      method: 'DELETE',
    });
    if (!res.ok) throw new Error('Failed to delete skill');
    return res.json();
  },

  async fetchModelInfo(modelId: string): Promise<ModelDetailedInfo> {
    const encodedId = encodeURIComponent(modelId);
    const res = await fetch(`${API_BASE}/models/info/${encodedId}`);
    if (!res.ok) throw new Error('Failed to fetch model info');
    return res.json();
  },

  async fetchLocalQuantizations(modelId: string): Promise<{
    model_id: string;
    repo_id: string;
    local_quantizations: string[];
    has_local: boolean;
  }> {
    try {
      const encodedId = encodeURIComponent(modelId);
      const res = await fetch(`${API_BASE}/models/${encodedId}/local-quants`);
      if (!res.ok) {
        // Return empty result on 404 or other errors
        return {
          model_id: modelId,
          repo_id: modelId,
          local_quantizations: [],
          has_local: false
        };
      }
      return res.json();
    } catch (error) {
      // Return empty result on network errors
      console.warn('Failed to fetch local quantizations:', error);
      return {
        model_id: modelId,
        repo_id: modelId,
        local_quantizations: [],
        has_local: false
      };
    }
  },

  /**
   * Helper for SSE streaming
   */
  async chatCompletionStream(
    payload: any,
    onToken: (token: string) => void,
    onConversationId?: (id: string) => void,
    signal?: AbortSignal
  ): Promise<void> {
    const { thinking, rag, maxTokens, topP, topK, repeatPenalty, ...rest } = payload;
    
    // Map camelCase to snake_case for the backend
    const bodyData = { 
      ...rest, 
      stream: true, 
      thinking: Boolean(thinking),
      rag: Boolean(rag),
      max_tokens: maxTokens,
      top_p: topP,
      top_k: topK,
      repeat_penalty: repeatPenalty,
    };
    
    const res = await fetch(`${API_BASE}/v1/chat/completions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(bodyData),
      signal,
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Unknown error' }));
      let msg = 'Streaming request failed';
      if (typeof err.detail === 'string') msg = err.detail;
      else if (Array.isArray(err.detail)) msg = err.detail.map((d: any) => d.msg).join(', ');
      else if (err.error) msg = err.error.message || err.error;
      throw new Error(msg);
    }

    const convId = res.headers.get('X-Conversation-ID');
    if (convId && onConversationId) {
      onConversationId(convId);
    }

    if (!res.body) return;

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      
      // Stop if aborted
      if (signal?.aborted) {
        await reader.cancel();
        return;
      }

      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        this._processLine(line, onToken);
      }
    }

    if (buffer.trim()) {
      this._processLine(buffer, onToken);
    }
  },

  _processLine(line: string, onToken: (token: string) => void) {
    const trimmed = line.trim();
    if (!trimmed || trimmed === 'data: [DONE]') return;
    
    if (trimmed.startsWith('data: ')) {
      const dataStr = trimmed.slice(6);
      
      if (dataStr.startsWith('[ERROR]')) {
        const errorMsg = dataStr.replace('[ERROR]', '').trim();
        throw new Error(errorMsg || 'Server-side generation error');
      }
      
      try {
        const json = JSON.parse(dataStr);
        const content = json.choices[0]?.delta?.content || '';
        if (content) onToken(content);
      } catch (e) {
        if (dataStr !== '[DONE]') {
          console.error('Error parsing SSE data:', e, line);
        }
      }
    }
  },

  async uploadDocument(file: File): Promise<any> {
    const formData = new FormData();
    formData.append('file', file);
    const res = await fetch(`${API_BASE}/index/upload`, {
      method: 'POST',
      body: formData,
    });
    if (!res.ok) throw new Error('Failed to upload document');
    return res.json();
  },

  async uploadChatDocument(file: File, modelId?: string, replyReserve?: number): Promise<any> {
    const formData = new FormData();
    formData.append('file', file);
    if (modelId) formData.append('model_id', modelId);
    if (replyReserve) formData.append('reply_reserve', String(replyReserve));
    const res = await fetch(`${API_BASE}/chat/upload`, {
      method: 'POST',
      body: formData,
    });
    if (!res.ok) {
      const error = await res.json().catch(() => ({ detail: 'Failed to extract file' }));
      throw new Error(error.detail || 'Failed to extract file');
    }
    return res.json();
  },

  async fetchDocuments(): Promise<any[]> {
    const res = await fetch(`${API_BASE}/index/documents`);
    if (!res.ok) throw new Error('Failed to fetch documents');
    return res.json();
  },

  async searchKnowledge(q: string, top_k = 5): Promise<any[]> {
    const res = await fetch(`${API_BASE}/index/search?q=${encodeURIComponent(q)}&top_k=${top_k}`);
    if (!res.ok) throw new Error('Failed to search knowledge base');
    return res.json();
  },

  async generateImage(payload: {
    model?: string;
    prompt: string;
    responseFormat?: 'url' | 'b64_json';
    size?: string;
    negativePrompt?: string;
    steps?: number;
    cfgScale?: number;
    seed?: number;
    sampler?: string;
  }): Promise<ImageGenerationResponse> {
    const res = await fetch(`${API_BASE}/v1/images/generations`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: payload.model,
        prompt: payload.prompt,
        response_format: payload.responseFormat || 'url',
        size: payload.size || '1024x1024',
        negative_prompt: payload.negativePrompt,
        steps: payload.steps,
        cfg_scale: payload.cfgScale,
        seed: payload.seed,
        sampler: payload.sampler,
      }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Unknown error' }));
      const message = typeof err.detail === 'string'
        ? err.detail
        : Array.isArray(err.detail)
          ? err.detail.map((d: any) => d.msg).join(', ')
          : err.error?.message || err.error || 'Image generation failed';
      throw new Error(message);
    }

    return res.json();
  },

  async fetchImageModels(): Promise<ImageModel[]> {
    const res = await fetch(`${API_BASE}/v1/images/models`);
    if (!res.ok) throw new Error('Failed to fetch image models');
    const data = await res.json();
    return data.data || [];
  },

  async pullImageModel(modelId: string, quantization?: string): Promise<{ success: boolean; model_id: string; quantization?: string; download_id: string; message: string }> {
    const res = await fetch(`${API_BASE}/v1/images/models/pull`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model_id: modelId, quantization }),
    });
    if (!res.ok) throw new Error('Failed to start image model download');
    return res.json();
  },

  async startImageGeneration(payload: {
    model?: string;
    prompt: string;
    responseFormat?: 'url' | 'b64_json';
    size?: string;
    negativePrompt?: string;
    steps?: number;
    cfgScale?: number;
    seed?: number;
    sampler?: string;
  }): Promise<ImageGenerationJob> {
    const res = await fetch(`${API_BASE}/v1/images/generations/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: payload.model,
        prompt: payload.prompt,
        response_format: payload.responseFormat || 'url',
        size: payload.size || '1024x1024',
        negative_prompt: payload.negativePrompt,
        steps: payload.steps,
        cfg_scale: payload.cfgScale,
        seed: payload.seed,
        sampler: payload.sampler,
      }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Unknown error' }));
      throw new Error(err.detail || 'Failed to start image generation');
    }

    return res.json();
  },

  async getImageGenerationJob(jobId: string): Promise<ImageGenerationJob> {
    const res = await fetch(`${API_BASE}/v1/images/generations/jobs/${encodeURIComponent(jobId)}`);
    if (!res.ok) throw new Error('Failed to fetch image generation job');
    return res.json();
  },

  streamImageGenerationProgress(
    jobId: string,
    onProgress: (job: ImageGenerationJob) => void,
    onComplete: (job: ImageGenerationJob) => void,
    onError: (error: string) => void
  ): () => void {
    const eventSource = new EventSource(`${API_BASE}/v1/images/generations/progress?id=${encodeURIComponent(jobId)}`);

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.error) {
          onError(data.error);
          eventSource.close();
          return;
        }

        onProgress(data as ImageGenerationJob);
        if (data.status === 'completed') {
          onComplete(data as ImageGenerationJob);
          eventSource.close();
        } else if (data.status === 'error') {
          onError(data.error || 'Image generation failed');
          eventSource.close();
        }
      } catch {
        onError('Failed to parse image progress update');
        eventSource.close();
      }
    };

    eventSource.onerror = () => {
      onError('Image progress stream disconnected');
      eventSource.close();
    };

    return () => eventSource.close();
  },
};

