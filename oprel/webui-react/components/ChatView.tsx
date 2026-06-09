"use client"

import { useEffect, useRef, useState, useCallback, useMemo } from "react"
import {
  Send,
  ChevronDown,
  Paperclip,
  FileText,
  ImageIcon,
  X,
  Zap,
  Brain,
  Bot,
  Copy,
  ThumbsUp,
  ThumbsDown,
  RotateCcw,
  Sparkles,
  PanelLeftClose,
  PanelLeftOpen,
  Square,
  Sun,
  Sunset,
  Moon,
  Search,
} from "lucide-react"
import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"
import remarkBreaks from "remark-breaks"
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter"
import { vscDarkPlus } from "react-syntax-highlighter/dist/esm/styles/prism"
import { cn } from "@/services/utils"
import { useApp } from "@/services/context"
import { API, ChatMessage as ApiChatMessage } from "@/services/api"
import type { ChatMessage, Conversation, AIModel } from "@/services/data"
import { PROVIDER_PRESETS, providerChatStream, type ProviderType } from "@/services/providers"
import { useToast } from "@/components/ui/use-toast"
import { ArtifactCard, ArtifactPanel, type Artifact } from "@/components/ArtifactPanel"

function TypingIndicator() {
  return (
    <div className="flex items-center gap-1 py-2">
      <div className="w-1.5 h-1.5 rounded-full bg-muted-foreground typing-dot" />
      <div className="w-1.5 h-1.5 rounded-full bg-muted-foreground typing-dot" />
      <div className="w-1.5 h-1.5 rounded-full bg-muted-foreground typing-dot" />
    </div>
  )
}

function getContentText(content: string | any[]): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    return content
      .map(part => {
        if (typeof part === 'string') return part;
        if (part.type === 'text') return part.text;
        return '';
      })
      .join('');
  }
  return '';
}

function renderImages(content: string | any[]) {
  if (typeof content === 'string') return null;
  if (Array.isArray(content)) {
    const images = content.filter(part => part.type === 'image_url');
    if (images.length === 0) return null;
    return (
      <div className="flex flex-wrap gap-2 mt-2 mb-2">
        {images.map((img, i) => (
          <img
            key={i}
            src={img.image_url.url}
            className="max-w-[400px] max-h-[300px] rounded-lg border border-border object-contain bg-black/20"
            alt="Uploaded content"
          />
        ))}
      </div>
    );
  }
  return null;
}
// Language display aliases (short names shown in code-block headers)
const LANG_ALIASES: Record<string, string> = {
  javascript: 'js', typescript: 'ts', python: 'py',
  markdown: 'md', dockerfile: 'docker', shellscript: 'sh',
  shell: 'sh', bash: 'bash', powershell: 'ps', java: 'java',
};
function displayLang(lang: string): string {
  const l = lang.toLowerCase();
  return LANG_ALIASES[l] || l;
}

/**
 * Full LLM output cleanup (run on final, completed messages).
 * Handles the common small-model issue of compressing markdown without newlines.
 */
function cleanLLMOutput(text: string): string {
  let r = text;

  // ── 1. Unicode table pipes ───────────────────────────────────────────────
  r = r.replace(/│/g, '|').replace(/┃/g, '|');

  // ── 2. Fix inline complete code blocks: ```lang content``` on one line ───
  //    e.g. "see: ```bash pip install foo``` next" → proper multi-line block
  r = r.replace(/```(\w+) ((?:(?!```|\n).)+)```/g, '\n```$1\n$2\n```\n');

  // ── 3. Code fences not at start of line → add newline before ────────────
  //    "text```python" → "text\n```python"
  r = r.replace(/([^\n`])(`{3})/g, '$1\n$2');

  // ── 4. Closing fence not followed by newline → add newline after ─────────
  //    "```\ncode```More text" → "```\ncode```\nMore text"
  //    Guard: don't add \n if next char is a-z (that would be an opening lang fence)
  r = r.replace(/(```)([ \t]*)([^\n`a-z])/g, '$1\n$3');

  // ── 5. Fix headers anywhere (not just line-start) ───────────────────────
  //    Step A: add space after ## if missing (anywhere in text)
  r = r.replace(/(#{2,6})([^\s#\n])/g, '$1 $2');
  //    Step B: add newlines before header if mid-line (now that space is present)
  r = r.replace(/([^\n])(#{2,6} )/g, '$1\n$2');

  // ── 6. Numbered list items compressed after punctuation ─────────────────
  //    "steps:1. Item" → "steps:\n1. Item"  (safe: only after : . ! ? ) ]
  r = r.replace(/([.!?:)\]])[ \t]*(\d+\.[ \t])/g, '$1\n$2');

  // ── 7. Bullet list items compressed after colon ──────────────────────────
  //    "structure: - Create" → "structure:\n- Create"
  r = r.replace(/:[ \t]*(-[ \t])/g, ':\n$1');

  // ── 8. Separator/rule fixes ──────────────────────────────────────────────
  r = r.replace(/^(---+)(#{2,6} )/gm, '$1\n\n$2');
  r = r.replace(/(\n---+)(#{2,6} )/g, '$1\n\n$2');

  return r;
}

/**
 * Parses user message content to separate attachments from main text for UI rendering.
 * Extracts [File: filename] ... ```code``` blocks.
 */
function parseUserContent(content: string | any[]): { text: string, files: Array<{ name: string, ext: string }> } {
  const rawText = getContentText(content);
  const files: Array<{ name: string, ext: string }> = [];
  
  // Regex to find [File: name] followed by optional code block
  // We want to hide these from the UI and only show Chips
  const fileRegex = /\[File: ([^\]]+)\](?:\r?\n)```(\w*)\r?\n[\s\S]*?```/g;
  
  let cleanText = rawText.replace(fileRegex, (match, name, ext) => {
    files.push({ name, ext });
    return ''; // Remove from text
  });

  // Also catch the "Here are the attached files:" prefix if it exists alone
  cleanText = cleanText.replace(/Here are the attached files:\s*$/, '').trim();
  
  return { 
    text: cleanText || (files.length > 0 ? "" : rawText), 
    files 
  };
}

/**
 * Light cleanup for streaming — only safest fixes, no aggressive rewrites.
 */
function cleanLLMOutputLight(text: string): string {
  let r = text;
  // Code fences not at line start (safe, no false positives)
  r = r.replace(/([^\n`])(`{3})/g, '$1\n$2');
  // Unicode pipes
  r = r.replace(/│/g, '|').replace(/┃/g, '|');
  return r;
}

// ── Time-aware greeting ──────────────────────────────────────────────────────
function getGreeting(name?: string): { text: string } {
  const hour = new Date().getHours()
  const suffix = name ? `, ${name.split(' ')[0]}` : ''
  if (hour < 12) return { text: `Good morning${suffix} ☀️` }
  if (hour < 17) return { text: `Good afternoon${suffix} 🌤` }
  return { text: `Good evening${suffix} 🌙` }
}

// ── AI-generated conversation title ─────────────────────────────────────────
const API_BASE_URL =
  typeof window !== 'undefined' && window.location.port === '3000'
    ? 'http://localhost:11435'
    : ''

async function generateConvTitle(
  apiModelId: string | undefined,
  userText: string
): Promise<string | null> {
  if (!apiModelId) return null
  try {
    const res = await fetch(`${API_BASE_URL}/v1/chat/completions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: apiModelId,
        messages: [
          {
            role: 'system',
            content:
              'You are a concise title generator. Generate a 3-5 word title for the conversation. Return ONLY the title text. No quotes, no intro, no "Title:", no punctuation.',
          },
          { role: 'user', content: `Summarize this message into a short title: ${userText.slice(0, 200)}` },
        ],
        max_tokens: 15,
        temperature: 0.3,
        stream: false,
        conversation_id: `ephemeral-title-${Date.now()}`,
      }),
    })
    if (!res.ok) return null
    const data = await res.json()
    const raw: string = data?.choices?.[0]?.message?.content || ''
    return raw.trim().replace(/^[«"']|[»"']$/g, '').slice(0, 60) || null
  } catch {
    return null
  }
}

function isVisionModel(model: AIModel | undefined) {
  if (!model) return false;
  const visionKeywords = ['vl', 'vision', 'llava', 'qwen2-vl', 'qwen2.5-vl', 'pixtral'];
  return visionKeywords.some(kw => model.id.toLowerCase().includes(kw) || model.name.toLowerCase().includes(kw));
}

function ThinkingBlock({ content, renderers }: { content: string, renderers: any }) {
  const [isExpanded, setIsExpanded] = useState(true);

  return (
    <div className={cn(
      "mb-6 rounded-xl bg-secondary/20 border border-primary/10 overflow-hidden transition-all duration-300",
      isExpanded ? "pb-4" : "pb-0"
    )}>
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full flex items-center justify-between px-4 py-3 text-primary/70 hover:text-primary hover:bg-primary/5 transition-all group"
      >
        <div className="flex items-center gap-2 text-[11px] font-bold uppercase tracking-widest">
          <Brain size={13} className={cn("transition-transform duration-500", isExpanded && "rotate-[360deg]")} />
          Thinking Process
        </div>
        <ChevronDown size={14} className={cn("transition-transform duration-300", isExpanded && "rotate-180")} />
      </button>

      {isExpanded && (
        <div className="px-4 text-muted-foreground italic text-sm border-t border-primary/5 pt-3 animate-in fade-in slide-in-from-top-2 duration-300">
          <ReactMarkdown remarkPlugins={[remarkGfm, remarkBreaks]} components={renderers}>{content}</ReactMarkdown>
        </div>
      )}
    </div>
  );
}

function MessageBubble({
  message,
  isStreaming = false,
  onOpenArtifact,
}: {
  message: ChatMessage
  isStreaming?: boolean
  onOpenArtifact?: (artifact: Artifact) => void
}) {
  const isUser = message.role === "user"
  const [copied, setCopied] = useState(false)

  const copyContent = () => {
    const text = getContentText(message.content);
    navigator.clipboard.writeText(text)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  // Build renderers — two modes just like ClaraVerse:
  //   STREAMING  → plain <pre><code> (fast, never crashes on partial tokens)
  //   COMPLETED  → react-syntax-highlighter with vscDarkPlus (full colors)
  const renderers = useMemo(() => ({
    code({ node, inline, className, children, ...props }: any) {
      const langMatch = /language-(\w+)/.exec(className || '');
      const lang = langMatch ? langMatch[1] : null;
      const isBlock = !inline && (lang || String(children ?? '').includes('\n'));
      // Safely extract code text — children can be string, array of strings, or undefined
      const codeText = (
        children == null ? ''
        : Array.isArray(children)
          ? (children as any[]).map(c => (c == null ? '' : String(c))).join('')
          : String(children)
      ).replace(/\n$/, '');

      if (isBlock) {
        // ── ARTIFACT: mermaid/html → card instead of code block ──────────
        if (!isStreaming && (lang === 'mermaid' || lang === 'html')) {
          const artifact: Artifact = { type: lang as 'mermaid' | 'html', code: codeText };
          return (
            <ArtifactCard
              artifact={artifact}
              onClick={() => onOpenArtifact?.(artifact)}
            />
          );
        }

        if (isStreaming) {
          // ── STREAMING: lightweight, plain, zero processing ────────────────
          return (
            <div className="oprel-streaming-code-block">
              <div className="oprel-streaming-code-header">
                <span className="oprel-streaming-code-lang">{lang ? displayLang(lang) : 'code'}</span>
              </div>
              <pre className="oprel-streaming-code-pre">
                <code>{codeText}</code>
              </pre>
            </div>
          );
        }

        // ── COMPLETED: full Prism syntax highlighting ─────────────────────
        return (
          <div className="oprel-code-block">
            <div className="oprel-code-header">
              <span className="oprel-code-lang">{lang ? displayLang(lang) : 'code'}</span>
              <button
                onClick={() => navigator.clipboard.writeText(codeText)}
                className="oprel-code-copy-btn"
              >
                <Copy size={12} />
                <span>Copy code</span>
              </button>
            </div>
            <div className="oprel-code-content">
              <SyntaxHighlighter
                language={lang || 'text'}
                style={vscDarkPlus}
                showLineNumbers={false}
                wrapLines
                wrapLongLines
                customStyle={{
                  margin: 0,
                  borderRadius: 0,
                  background: '#1e1e1e',
                  fontSize: '13px',
                  lineHeight: '1.6',
                  padding: '1rem',
                }}
              >
                {codeText}
              </SyntaxHighlighter>
            </div>
          </div>
        );
      }

      // Inline code
      return (
        <code className="oprel-inline-code" {...props}>
          {children}
        </code>
      );
    },
  }), [isStreaming]);

  // Handle thinking tags + clean LLM output quirks
  const processedContent = useMemo(() => {
    const rawText = getContentText(message.content);
    const images = renderImages(message.content);

    if (!rawText) return images;

    // Apply streaming-safe or full cleanup
    let cleaned = isStreaming ? cleanLLMOutputLight(rawText) : cleanLLMOutput(rawText);

    // Extract <think>…</think> block
    let thinking = "";
    const startIdx = cleaned.indexOf("<think>");
    if (startIdx !== -1) {
      const endIdx = cleaned.indexOf("</think>", startIdx + 7);
      if (endIdx !== -1) {
        thinking = cleaned.substring(startIdx + 7, endIdx).trim();
        cleaned = cleaned.substring(0, startIdx).trim() + "\n\n" + cleaned.substring(endIdx + 8).trim();
      } else {
        // Ongoing thinking (streaming)
        thinking = cleaned.substring(startIdx + 7).trim();
        cleaned = cleaned.substring(0, startIdx).trim();
      }
    }

    return (
      <div className="oprel-response">
        {images}
        {thinking && <ThinkingBlock content={thinking} renderers={renderers} />}
        {cleaned && (
          <div className="oprel-markdown">
            <ReactMarkdown
              remarkPlugins={[remarkGfm, remarkBreaks]}
              components={renderers}
            >
              {cleaned}
            </ReactMarkdown>
          </div>
        )}
      </div>
    );
  }, [message.content, renderers, isStreaming]);

  if (isUser) {
    const { text, files } = parseUserContent(message.content);
    const imgs = renderImages(message.content);
    const [copied, setCopied] = useState(false);

    const copyUserText = () => {
      navigator.clipboard.writeText(getContentText(message.content));
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    };

    return (
      <div className="flex justify-end mb-4 animate-in slide-in-from-right-2 duration-300">
        <div className="flex flex-col items-end gap-2 group/msg" style={{ maxWidth: '80%' }}>

          {/* File chips — standalone card per file, like the reference */}
          {files.map((file, i) => (
            <div key={i} className="flex items-center gap-3 px-3 py-2.5 rounded-2xl bg-[#1e1e1e] border border-white/[0.07] shadow-sm min-w-[200px]">
              {/* Blue file icon box */}
              <div className="w-9 h-9 rounded-lg bg-[#2563eb] flex items-center justify-center shrink-0">
                <FileText size={16} className="text-white" />
              </div>
              {/* Name & type */}
              <div className="flex flex-col min-w-0">
                <span className="text-[12px] font-semibold text-foreground truncate max-w-[220px] leading-tight">{file.name}</span>
                <span className="text-[10px] text-muted-foreground mt-0.5">
                  {/\.(pdf)$/i.test(file.name) ? 'PDF' :
                   /\.(txt|md)$/i.test(file.name) ? 'Text' :
                   /\.(jpg|jpeg|png|gif|webp|svg)$/i.test(file.name) ? 'Image' :
                   'Document'}
                </span>
              </div>
            </div>
          ))}

          {/* Image attachments */}
          {imgs}

          {/* Text bubble — simple rounded pill */}
          {text && (
            <div className="px-4 py-2.5 rounded-2xl bg-[#2d2d2d] border border-white/[0.05] text-foreground text-sm leading-relaxed whitespace-pre-wrap break-words shadow-sm">
              {text}
            </div>
          )}

          {/* Hover action icons — copy + edit — appear on hover below message */}
          <div className="flex items-center gap-1 opacity-0 group-hover/msg:opacity-100 transition-opacity duration-150 pr-1">
            <button
              onClick={copyUserText}
              className={cn(
                "w-7 h-7 rounded-lg flex items-center justify-center text-muted-foreground hover:text-foreground hover:bg-white/[0.06] transition-all",
                copied && "text-green-400"
              )}
              title="Copy"
            >
              <Copy size={13} />
            </button>
            <button
              className="w-7 h-7 rounded-lg flex items-center justify-center text-muted-foreground hover:text-foreground hover:bg-white/[0.06] transition-all"
              title="Edit"
            >
              {/* Pencil icon inline to avoid extra import */}
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/>
                <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/>
              </svg>
            </button>
          </div>
        </div>
      </div>
    )
  }


  return (
    <div className="mb-6 group">
      <div className="flex items-start gap-4 max-w-[780px]">
        <div className="w-7 h-7 rounded-full overflow-hidden shrink-0 mt-1 border border-border bg-secondary">
          <img src="/gui/logo1.png" alt="AI" className="w-full h-full object-cover" />
        </div>
        <div className="flex-1 min-w-0">
          {processedContent}
        </div>
      </div>
      {/* Action bar */}
      {!isStreaming && (
        <div className="flex items-center gap-1 mt-2 ml-11 transition-opacity">
          <button
            onClick={copyContent}
            className={cn(
              "p-1.5 rounded-md text-muted-foreground hover:text-foreground hover:bg-secondary transition-all text-xs flex items-center gap-1",
              copied && "text-green-500"
            )}
            title="Copy"
          >
            <Copy size={12} />
            {copied && <span className="text-[10px]">Copied</span>}
          </button>
          <button className="p-1.5 rounded-md text-muted-foreground hover:text-primary hover:bg-secondary transition-all" title="Good response">
            <ThumbsUp size={12} />
          </button>
          <button className="p-1.5 rounded-md text-muted-foreground hover:text-destructive hover:bg-secondary transition-all" title="Bad response">
            <ThumbsDown size={12} />
          </button>
          {message.tps && (
            <span className="ml-2 text-[10px] font-mono text-muted-foreground/50">
              {message.tps} t/s
            </span>
          )}
        </div>
      )}
    </div>
  )
}

export function ChatView({
}: {
}) {
  const {
    conversations,
    activeConversationId,
    setActiveConversationId,
    addMessage,
    linkConversation,
    refreshConversations,
    models,
    localModels,
    activeModelId,
    setActiveModelId,
    isGenerating,
    setIsGenerating,
    isModelLoading,
    setIsModelLoading,
    refreshModels,
    providers,
    refreshProviders,
    saveProvider,
    removeProvider,
    settings,
    setConversationMessages,
    user,
    ragEnabled,
    setRagEnabled
  } = useApp()

  const [input, setInput] = useState("")
  const [thinking, setThinking] = useState(false)
  const [modelDropdown, setModelDropdown] = useState(false)
  const [showTyping, setShowTyping] = useState(false)
  const [streamingMessage, setStreamingMessage] = useState<string | null>(null)
  const [selectedImage, setSelectedImage] = useState<string | null>(null)
  const [selectedImageName, setSelectedImageName] = useState<string | null>(null)
  const [activeArtifact, setActiveArtifact] = useState<Artifact | null>(null)
  const panelRef = useRef<HTMLDivElement>(null)
  // abort controller for stop-generation
  const abortRef = useRef<AbortController | null>(null)
  // Multi-file attachments (text/code/pdf etc.)
  const [attachments, setAttachments] = useState<Array<{
    name: string;
    type: 'image' | 'text';
    content: string;   // base64 data URL for images, raw text for text
    mimeType: string;
  }>>([])
  const fileInputRef = useRef<HTMLInputElement>(null)
  const { toast } = useToast()
  const [isHistoryLoading, setIsHistoryLoading] = useState(() => {
    // Initial state: true if we have an ID that needs loading
    return !!activeConversationId && !activeConversationId.startsWith('temp-');
  })

  const [prevActiveId, setPrevActiveId] = useState(activeConversationId);
  if (activeConversationId !== prevActiveId) {
    setPrevActiveId(activeConversationId);

    // Optimization: Only show loading screen if we don't have this conversation's messages in state yet
    const hasMessages = conversations.some(c => c.id === activeConversationId && c.messages.length > 0);

    if (activeConversationId && !activeConversationId.startsWith('temp-') && !hasMessages) {
      setIsHistoryLoading(true);
    } else {
      setIsHistoryLoading(false);
    }
  }

  const scrollRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  const activeConv = useMemo(() => conversations.find((c) => c.id === activeConversationId), [conversations, activeConversationId]);
  const chatLocalModels = useMemo(
    () => localModels.filter((m) => m.category !== 'text-to-image'),
    [localModels]
  );
  const chatModels = useMemo(
    () => models.filter((m) => m.category !== 'text-to-image'),
    [models]
  );
  // Prefer localModels (which have alias·quant names) for the active model display.
  // localModels IDs are "repo_id::QUANT", so match by activeModelId first, then by loaded status.
  const activeModel = useMemo(() => {
    if (!activeModelId) return undefined;
    // Exact match by composite ID in localModels
    const exactLocal = chatLocalModels.find(m => m.id === activeModelId)
    if (exactLocal) return exactLocal
    // Exact match in registry/external models
    const exactGlobal = chatModels.find((m) => m.id === activeModelId)
    if (exactGlobal) return exactGlobal

    // Fallback to loaded model from localModels
    const loadedLocal = chatLocalModels.find(m => m.status === 'loaded')
    if (loadedLocal) return loadedLocal
    // Fall back to any loaded model
    return chatModels.find(m => m.status === 'loaded')
  }, [chatLocalModels, chatModels, activeModelId]);
  // Use a ref so sendMessage always has latest conversation without being in deps
  const activeConvRef = useRef(activeConv);
  useEffect(() => { activeConvRef.current = activeConv; }, [activeConv]);

  // Global listener: If active model switches to an external model via ANY interaction
  // (Sidebar history click, dropdown selection), force-unload any running local models to save VRAM 
  useEffect(() => {
    if (activeModel?.category === 'external') {
      const loadedLocal = chatLocalModels.find(lm => lm.status === 'loaded');
      if (loadedLocal) {
        const repoToUnload = loadedLocal.modelRepoId || (loadedLocal.id.includes('::') ? loadedLocal.id.split('::')[0] : loadedLocal.id);
        API.unloadModel(repoToUnload)
          .then(() => refreshModels())
          .catch(err => console.error("Auto-unload failed:", err));
      }
    }
  }, [activeModel?.category, chatLocalModels, refreshModels]);

  // Load conversation history - only trigger when activeConversationId changes
  const loadedConvIdRef = useRef<string | null>(null);
  useEffect(() => {
    if (!activeConversationId || activeConversationId.startsWith('temp-')) {
      setIsHistoryLoading(false);
      return;
    }
    // Already loaded this convo's history
    const conv = conversations.find(c => c.id === activeConversationId);
    if (conv && conv.messages.length > 0) {
      setIsHistoryLoading(false);
      return;
    }
    // Don't reload if we already fetched this
    if (loadedConvIdRef.current === activeConversationId) {
      setIsHistoryLoading(false);
      return;
    }

    loadedConvIdRef.current = activeConversationId;
    setIsHistoryLoading(true);
    API.getConversation(activeConversationId).then(data => {
      // Backend returns a list of messages directly
      const messages = Array.isArray(data) ? data : (data as any).history || [];
      const history: ChatMessage[] = messages.map((m: any, i: number) => ({
        id: `${activeConversationId}-${i}`,
        role: m.role,
        content: m.content,
        timestamp: new Date(),
      }));
      setConversationMessages(activeConversationId, history);
    })
      .catch(err => {
        console.error(`[ChatView] API Fetch error for ${activeConversationId}:`, err);
        loadedConvIdRef.current = null;
      })
      .finally(() => {
        setIsHistoryLoading(false);
      });
    // Only depend on the ID changing, not on conversations array (to avoid loop)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeConversationId]);

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    if (!files.length) return;

    for (const file of files) {
      const isImage = file.type.startsWith('image/');

      if (isImage) {
        if (!isVisionModel(activeModel)) {
          toast({
            title: 'Vision model required',
            description: `"${file.name}" is an image. Switch to a vision model (e.g. Qwen2.5-VL) to attach images.`,
            variant: 'destructive',
          });
          continue;
        }

        const reader = new FileReader();
        reader.onload = ev => {
          if (ev.target?.result) {
            setSelectedImage(ev.target.result as string);
            setSelectedImageName(file.name);
          }
        };
        reader.readAsDataURL(file);
        continue;
      }

      try {
        const apiModelId = activeModelId?.includes('::') ? activeModelId.split('::')[0] : activeModelId;
        const res = await API.uploadChatDocument(file, apiModelId, settings.maxTokens);
        const suggested = res?.suggested_text || res?.extracted_text || `[File: ${file.name}]`;
        setAttachments(prev => [...prev, {
          name: file.name,
          type: 'text',
          content: suggested,
          mimeType: file.type || 'application/octet-stream',
        }]);
        toast({
          title: 'File extracted',
          description: `Extracted ${res?.chars ?? suggested.length} chars (~${res?.estimated_tokens ?? Math.ceil((suggested.length || 0) / 4)} tokens).`,
        });
      } catch (err: any) {
        console.error('Chat upload failed:', err);
        toast({
          title: 'Extraction failed',
          description: `Failed to extract ${file.name}: ${err?.message || 'Unknown error'}`,
          variant: 'destructive',
        });

        const looksLikeText = file.type.startsWith('text/') || /\.(txt|md|js|ts|jsx|tsx|py|json|yaml|yml|html|css|xml|sh|csv|log)$/i.test(file.name);
        if (looksLikeText) {
          const reader = new FileReader();
          reader.onload = ev => {
            const text = ev.target?.result as string;
            setAttachments(prev => [...prev, { name: file.name, type: 'text', content: text, mimeType: file.type || 'text/plain' }]);
          };
          reader.readAsText(file);
        } else {
          setAttachments(prev => [...prev, { name: file.name, type: 'text', content: `[File: ${file.name}]`, mimeType: file.type || 'application/octet-stream' }]);
        }
      }
    }

    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const removeAttachment = (idx: number) => {
    setAttachments(prev => prev.filter((_, i) => i !== idx));
  };

  const clearImage = () => {
    setSelectedImage(null);
    setSelectedImageName(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  // ── Stop generation ────────────────────────────────────────────────────────
  const stopGeneration = useCallback(() => {
    abortRef.current?.abort()
  }, [])

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [activeConv?.messages, showTyping, streamingMessage, selectedImage, attachments])

  const sendMessage = useCallback(async () => {
    const hasText = input.trim().length > 0;
    const hasAttachment = selectedImage || attachments.length > 0;
    if ((!hasText && !hasAttachment) || isGenerating) return;

    const currentInput = input.trim();
    const currentImage = selectedImage;
    const currentAttachments = [...attachments];

    let textWithFiles = currentInput;
    if (currentAttachments.length > 0) {
      const fileBlocks = currentAttachments.map(a => {
        const ext = a.name.split('.').pop() || '';
        return `\n\n[File: ${a.name}]\n\`\`\`${ext}\n${a.content}\n\`\`\``;
      }).join('');
      textWithFiles = currentInput ? currentInput + fileBlocks : `Here are the attached files:${fileBlocks}`;
    }

    let convId = activeConversationId;
    if (!convId) {
      convId = `temp-${Date.now()}`;
      setActiveConversationId(convId);
    }
    const finalConvId = convId;

    const userMsgId = `u-${Date.now()}`;
    const userContent = currentImage
      ? [{ type: 'image_url', image_url: { url: currentImage } }, { type: 'text', text: textWithFiles }]
      : textWithFiles;

    const userMsg: ChatMessage = {
      id: userMsgId,
      role: 'user',
      content: userContent,
      timestamp: new Date(),
    };

    const currentConv = activeConvRef.current;
    const history = (currentConv?.messages || []).map(m => ({ role: m.role, content: m.content }));
    const contextMessages = [
      ...(settings.systemPrompt ? [{ role: 'system', content: settings.systemPrompt }] : []),
      ...history,
      { role: 'user', content: userContent },
    ];

    addMessage(finalConvId, userMsg);
    setInput('');
    clearImage();
    setAttachments([]);
    setIsGenerating(true);
    setShowTyping(true);

    let currentResponse = '';
    let effectiveConvId = finalConvId;
    const isFirstMessage = !activeConvRef.current?.messages?.length;
    const abort = new AbortController();
    abortRef.current = abort;

    try {
      const apiModelId = activeModelId?.includes('::') ? activeModelId.split('::')[0] : activeModelId;
      const isExternal = activeModel?.category === 'external';

      if (isExternal && activeModelId?.includes('::')) {
        const [providerId, realModelId] = activeModelId.split('::');
        const provider = providers.find((p) => p.id === providerId);
        if (!provider) {
          throw new Error(`Provider ${providerId} not found`);
        }

        await providerChatStream(
          provider,
          realModelId,
          contextMessages,
          {
            max_tokens: settings.maxTokens,
            temperature: settings.temperature,
            top_p: settings.topP,
            conversation_id: finalConvId.startsWith('temp-') ? undefined : finalConvId,
            rag: ragEnabled,
          },
          (token) => {
            if (abort.signal.aborted) return;
            setShowTyping(false);
            currentResponse += token;
            setStreamingMessage(currentResponse);
          },
          (newId) => {
            if (finalConvId.startsWith('temp-')) {
              linkConversation(finalConvId, newId);
              window.history.replaceState(null, "", `/chat?conversationId=${newId}`);
              effectiveConvId = newId;
              refreshConversations();
            }
          },
          abort.signal,
        );
      } else {
        await API.chatCompletionStream(
          {
            model: apiModelId,
            messages: contextMessages,
            conversation_id: finalConvId.startsWith('temp-') ? undefined : finalConvId,
            thinking,
            temperature: settings.temperature,
            max_tokens: settings.maxTokens,
            top_p: settings.topP,
            top_k: settings.topK,
            repeat_penalty: settings.repeatPenalty,
            rag: ragEnabled,
          },
          (token) => {
            if (abort.signal.aborted) return;
            setShowTyping(false);
            currentResponse += token;
            setStreamingMessage(currentResponse);
          },
          (newId) => {
            if (finalConvId.startsWith('temp-')) {
              linkConversation(finalConvId, newId);
              window.history.replaceState(null, "", `/chat?conversationId=${newId}`);
              effectiveConvId = newId;
              refreshConversations();
            }
          },
          abort.signal,
        );
      }

      if (currentResponse.trim()) {
        addMessage(effectiveConvId, {
          id: `a-${Date.now()}`,
          role: 'assistant',
          content: currentResponse,
          timestamp: new Date(),
        });

        if (effectiveConvId && !effectiveConvId.startsWith('temp-')) {
          setTimeout(async () => {
            try {
              const data = await API.getConversation(effectiveConvId);
              const msgs = Array.isArray(data) ? data : (data as any).history || [];
              if (msgs.length > 0) {
                const normalized: ChatMessage[] = msgs.map((m: any, i: number) => ({
                  id: `${effectiveConvId}-${i}`,
                  role: m.role,
                  content: m.content,
                  timestamp: new Date(),
                }));
                setConversationMessages(effectiveConvId, normalized);
              }
            } catch {
              // keep local message if refresh fails
            }

            if (isFirstMessage && currentInput.trim()) {
              const title = await generateConvTitle(apiModelId, currentInput.trim());
              if (title) {
                try {
                  await API.renameConversation(effectiveConvId, title);
                  refreshConversations();
                } catch {
                  // best effort
                }
              }
            }
          }, 600);
        }
      }
    } catch (error: any) {
      if (error?.name !== 'AbortError' && !(error instanceof DOMException)) {
        console.error('Generation failed:', error);
        addMessage(effectiveConvId, {
          id: `err-${Date.now()}`,
          role: 'assistant',
          content: 'Error: ' + (error instanceof Error ? error.message : 'Unknown error occurred'),
          timestamp: new Date(),
        });
      } else if (currentResponse.trim()) {
        addMessage(effectiveConvId, {
          id: `a-${Date.now()}`,
          role: 'assistant',
          content: currentResponse + '\n\n*(Generation stopped)*',
          timestamp: new Date(),
        });
      }
    } finally {
      setIsGenerating(false);
      setShowTyping(false);
      setStreamingMessage(null);
    }
  }, [
    input,
    selectedImage,
    attachments,
    isGenerating,
    activeConversationId,
    activeModelId,
    activeModel,
    thinking,
    settings,
    addMessage,
    setIsGenerating,
    setActiveConversationId,
    setConversationMessages,
    refreshConversations,
    providers,
    ragEnabled,
    linkConversation,
  ]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  const tokenCount = Math.ceil(input.length / 4)

  return (
    <div className="flex h-full bg-background overflow-hidden">
      {/* Left: full chat column */}
      <div className="flex flex-col flex-1 min-w-0 overflow-hidden">
      {/* Header */}
      <header className="h-14 border-b border-border flex items-center justify-between px-5 shrink-0 bg-background/95 backdrop-blur-sm sticky top-0 z-20">
        <div className="flex items-center gap-3">
          <div>
            <div className="flex items-center gap-2">
              <span className="text-sm font-bold text-foreground">
                {isModelLoading ? "Initializing Model..." : (activeModel?.name ?? "No model")}
              </span>
              <span className={cn(
                "w-2 h-2 rounded-full pulse-dot",
                isModelLoading ? "bg-amber-500" : "bg-green-500"
              )} />
            </div>
            <div className="text-[10px] text-muted-foreground font-medium">
              {isModelLoading ? "Loading into VRAM..." : `Active · ${activeModel?.speed ?? "—"}`}
            </div>
          </div>
        </div>

        <div className="relative">
          <button
            onClick={() => setModelDropdown((v) => !v)}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-border text-xs font-semibold text-muted-foreground hover:text-foreground hover:bg-secondary transition-all"
          >
            <Zap size={12} className="text-primary" />
            Switch Model
            <ChevronDown size={12} />
          </button>
          {modelDropdown && (
            <>
              <div className="fixed inset-0 z-40" onClick={() => setModelDropdown(false)} />
              <div className="absolute right-0 top-10 w-72 bg-[#1e1e1e] border border-border rounded-xl shadow-2xl p-2 z-50 animate-fade-in-up">
                {/* Local Models Section */}
                <div className="px-2 py-1.5 text-[10px] font-bold text-muted-foreground uppercase tracking-wider bg-secondary/30 rounded-t-lg">
                  Local Inference
                </div>
                <div className="max-h-[30vh] overflow-y-auto">
                  {chatLocalModels.length > 0 ? (
                    chatLocalModels.map((m) => {
                      const repoId = m.modelRepoId || (m.id.includes('::') ? m.id.split('::')[0] : m.id)
                      const quant = m.quantization
                      const isActive = m.id === activeModelId
                      return (
                        <button
                          key={m.id}
                          onClick={async () => {
                            setModelDropdown(false)
                            if (!isActive) {
                              setIsModelLoading(true)
                              try {
                                setActiveModelId(m.id)
                                await API.loadModel(repoId, { quantization: quant })
                                await refreshModels()
                              } catch (err) {
                                console.error("Failed to switch model:", err)
                              } finally {
                                setIsModelLoading(false)
                              }
                            }
                          }}
                          className={cn(
                            "w-full text-left px-3 py-2 rounded-lg text-xs transition-all flex items-center justify-between gap-2 my-0.5",
                            isActive
                              ? "bg-primary/10 text-primary font-bold"
                              : "text-muted-foreground hover:bg-secondary hover:text-foreground"
                          )}
                        >
                          <div className="min-w-0">
                            <div className="font-semibold truncate">{m.name}</div>
                            <div className="text-[10px] opacity-60">{m.size}</div>
                          </div>
                          {m.status === "loaded" && (
                            <span className="text-[9px] px-1.5 py-0.5 rounded-md bg-green-500/10 text-green-500 font-bold shrink-0">LOADED</span>
                          )}
                        </button>
                      )
                    })
                  ) : (
                    <div className="px-3 py-4 text-center opacity-50 italic text-[10px]">No local models</div>
                  )}
                </div>

                {/* External Provider Models Section */}
                {chatModels.some(m => m.category === 'external') && (
                  <>
                    <div className="px-2 py-1.5 text-[10px] font-bold text-muted-foreground uppercase tracking-wider bg-secondary/30 border-t border-border/50">
                      External Providers
                    </div>
                    <div className="max-h-[30vh] overflow-y-auto pt-1">
                      {chatModels
                        .filter(m => m.category === 'external')
                        .map((m) => {
                          const isActive = m.id === activeModelId
                          const preset = PROVIDER_PRESETS[m.architecture as ProviderType]
                          const providerColor = preset?.color || "#6b7280"
                          return (
                            <button
                              key={m.id}
                              onClick={async () => {
                                setModelDropdown(false)
                                if (!isActive) {
                                  setActiveModelId(m.id)
                                }
                              }}
                              className={cn(
                                "w-full text-left px-3 py-2 rounded-lg text-xs transition-all flex items-center justify-between gap-2 my-0.5",
                                isActive
                                  ? "bg-primary/10 text-primary font-bold"
                                  : "text-muted-foreground hover:bg-secondary hover:text-foreground"
                              )}
                            >
                              <div className="min-w-0">
                                <div className="font-semibold truncate">{m.name}</div>
                                <div className="flex items-center gap-1.5">
                                  <div className="w-1.5 h-1.5 rounded-full" style={{ background: providerColor || 'var(--primary)' }} />
                                  <div className="text-[10px] opacity-60">{m.family}</div>
                                </div>
                              </div>
                              {isActive && (
                                <span className="text-[9px] px-1.5 py-0.5 rounded-md bg-primary/10 text-primary font-bold shrink-0">ACTIVE</span>
                              )}
                            </button>
                          )
                        })}
                    </div>
                  </>
                )}
              </div>
            </>
          )}
        </div>
      </header>

      {/* Messages area */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto px-5 py-6">
        <div className="max-w-[780px] mx-auto">
          {isHistoryLoading ? (
            <div className="flex items-center justify-center h-full min-h-[400px]">
              <div className="flex flex-col items-center gap-2">
                <div className="w-8 h-8 rounded-full border-2 border-primary/30 border-t-primary animate-spin" />
                <span className="text-xs text-muted-foreground font-medium">Loading conversation...</span>
              </div>
            </div>
          ) : (!activeConv && !isHistoryLoading) || (activeConv && activeConv.messages.length === 0 && !streamingMessage && !isHistoryLoading) ? (
            <div className="flex flex-col items-center justify-center h-full min-h-[400px] text-center gap-4">
              <div className="w-16 h-16 rounded-2xl bg-primary/10 border border-primary/20 flex items-center justify-center">
                <Sparkles size={30} className="text-primary" />
              </div>
              <div>
                <h2 className="text-xl font-bold text-foreground mb-1">
                  {getGreeting(user?.name).text}
                </h2>
                <p className="text-sm text-muted-foreground">
                  Ask anything — using <span className="font-semibold text-foreground/80">{activeModel?.name ?? 'your model'}</span>
                </p>
              </div>
              <div className="flex flex-wrap gap-2 justify-center mt-2">
                {["Explain transformers", "Write a React hook", "Optimize SQL query", "Debug my code"].map((s) => (
                  <button
                    key={s}
                    onClick={() => setInput(s)}
                    className="px-3 py-1.5 rounded-lg border border-border text-xs text-muted-foreground hover:text-foreground hover:bg-secondary transition-all"
                  >
                    {s}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <>
              {activeConv?.messages.map((msg) => (
                <MessageBubble
                  key={msg.id}
                  message={msg}
                  onOpenArtifact={setActiveArtifact}
                />
              ))}
              {streamingMessage && (
                <MessageBubble
                  isStreaming={true}
                  onOpenArtifact={setActiveArtifact}
                  message={{
                    id: 'streaming',
                    role: 'assistant',
                    content: streamingMessage,
                    timestamp: new Date(),
                  }}
                />
              )}
              {showTyping && (
                <div className="flex items-start gap-3 mb-6">
                  <div className="w-7 h-7 rounded-full overflow-hidden shrink-0 mt-0.5 border border-border bg-secondary">
                    <img src="/gui/logo1.png" alt="AI" className="w-full h-full object-cover" />
                  </div>
                  <div className="flex-1">
                    <TypingIndicator />
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      </div>

      {/* Input area */}
      <div className="px-5 pb-5 pt-2 bg-gradient-to-t from-background via-background to-transparent shrink-0">
        <div className="max-w-[780px] mx-auto">
          {/* Image preview */}
          {selectedImage && (
            <div className="mb-2 p-2 rounded-xl bg-secondary/50 border border-border flex items-center gap-3 animate-in fade-in slide-in-from-bottom-2 duration-200">
              <div className="relative group/img">
                <img src={selectedImage} className="w-12 h-12 rounded-lg object-cover border border-border" alt="Selected" />
                <button
                  onClick={clearImage}
                  className="absolute -top-1.5 -right-1.5 w-4 h-4 rounded-full bg-destructive text-white flex items-center justify-center opacity-0 group-hover/img:opacity-100 transition-opacity"
                >
                  <X size={9} />
                </button>
              </div>
              <div className="flex-1 min-w-0">
                <div className="text-[10px] font-bold text-foreground truncate">{selectedImageName}</div>
                <div className="text-[9px] text-muted-foreground">Image</div>
              </div>
              <button onClick={clearImage} className="text-[10px] font-bold text-muted-foreground hover:text-foreground px-2">Remove</button>
            </div>
          )}

          {/* Text/code file attachments tray */}
          {attachments.length > 0 && (
            <div className="mb-2 flex flex-wrap gap-1.5 animate-in fade-in duration-200">
              {attachments.map((att, i) => (
                <div
                  key={i}
                  className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-secondary/60 border border-border text-[11px] text-foreground max-w-[200px]"
                >
                  <FileText size={11} className="text-primary shrink-0" />
                  <span className="truncate">{att.name}</span>
                  <button
                    onClick={() => removeAttachment(i)}
                    className="ml-0.5 text-muted-foreground hover:text-foreground transition-colors shrink-0"
                  >
                    <X size={10} />
                  </button>
                </div>
              ))}
            </div>
          )}

          <div
            className={cn(
              "bg-[#1e1e1e] border border-border rounded-xl overflow-hidden transition-all",
              "focus-within:border-primary/50 focus-within:ring-1 focus-within:ring-primary/20"
            )}
          >
            <textarea
              ref={textareaRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Message..."
              rows={1}
              className="w-full bg-transparent text-foreground px-4 pt-4 pb-2 text-sm resize-none focus:outline-none placeholder:text-muted-foreground min-h-[52px] max-h-[200px]"
            />

            <input
              type="file"
              ref={fileInputRef}
              className="hidden"
              multiple
              accept="image/*,.pdf,.txt,.md,.py,.js,.ts,.jsx,.tsx,.java,.c,.cpp,.cs,.go,.rs,.rb,.php,.html,.css,.json,.yaml,.yml,.xml,.sh,.sql,.swift,.kt,.r,.scala,.lua,.toml"
              onChange={handleFileSelect}
            />

            <div className="flex items-center justify-between px-3 pb-3 pt-1">
              <div className="flex items-center gap-2">
                <button
                  onClick={() => fileInputRef.current?.click()}
                  className={cn(
                    "p-1.5 text-muted-foreground hover:text-foreground hover:bg-secondary rounded-lg transition-all",
                    (selectedImage || attachments.length > 0) && "text-primary bg-primary/10"
                  )}
                  title="Attach file (image for vision models, or any text/code/PDF file)"
                >
                  <Paperclip size={15} />
                </button>
                <div className="h-4 w-px bg-border mx-1" />

                {/* Mode toggle */}
                <div className="flex items-center bg-[#171717] border border-border rounded-full p-0.5 gap-0.5">
                  <button
                    onClick={() => setRagEnabled(false)}
                    className={cn(
                      "px-3 py-1 rounded-full text-[10px] font-bold transition-all",
                      !ragEnabled ? "bg-secondary text-foreground shadow-sm" : "text-muted-foreground hover:text-foreground"
                    )}
                  >
                    Normal
                  </button>
                  <button
                    onClick={() => setRagEnabled(true)}
                    className={cn(
                      "px-3 py-1 rounded-full text-[10px] font-bold transition-all flex items-center gap-1",
                      ragEnabled ? "bg-primary text-primary-foreground shadow-sm" : "text-muted-foreground hover:text-foreground"
                    )}
                    title={ragEnabled ? "RAG Enabled: Searching knowledge base" : "RAG Disabled: Using model knowledge only"}
                  >
                    <Search size={10} />
                    RAG
                  </button>
                </div>

                <div className="h-4 w-px bg-border mx-1" />
                <span className="text-[10px] font-mono text-muted-foreground">{tokenCount} / {settings.maxTokens}</span>
              </div>

              {/* Send / Stop button */}
              {isGenerating ? (
                <button
                  onClick={stopGeneration}
                  className="w-9 h-9 rounded-lg flex items-center justify-center transition-all bg-destructive/90 text-white hover:bg-destructive shadow-lg shadow-destructive/20 animate-pulse"
                  title="Stop generation"
                >
                  <Square size={14} fill="currentColor" />
                </button>
              ) : (
                <button
                  onClick={sendMessage}
                  disabled={!input.trim() && !selectedImage && attachments.length === 0}
                  className={cn(
                    "w-9 h-9 rounded-lg flex items-center justify-center transition-all",
                    (input.trim() || selectedImage || attachments.length > 0)
                      ? "bg-primary text-primary-foreground hover:bg-primary/90 shadow-lg shadow-primary/20"
                      : "bg-secondary text-muted-foreground cursor-not-allowed"
                  )}
                >
                  <Send size={15} />
                </button>
              )}
            </div>
          </div>

          <div className="text-center mt-2 text-[10px] text-muted-foreground">
            Using{" "}
            <span
              onClick={() => setModelDropdown(true)}
              className="text-muted-foreground hover:text-primary cursor-pointer transition-colors font-medium"
            >
              {activeModel?.name}
            </span>
          </div>
        </div>
      </div>
      {/* END left column */}
      </div>

      {/* Right: Artifact Panel */}
      {activeArtifact && (
        <div
          ref={panelRef}
          style={{ width: 480 }}
          className="shrink-0 border-l border-border/60 flex flex-col overflow-hidden animate-in slide-in-from-right-4 duration-300"
        >
          <ArtifactPanel
            artifact={activeArtifact}
            onClose={() => setActiveArtifact(null)}
            panelRef={panelRef as React.RefObject<HTMLDivElement>}
          />
        </div>
      )}
    </div>
  )
}
