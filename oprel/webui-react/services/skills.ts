export interface Skill {
  id: string;
  name: string;
  description: string;
  command: string; // Command without slash, e.g. "explain"
  icon: string;    // Lucide icon name matching our mapper
  category: 'Writing' | 'Development' | 'Research' | 'Documents' | 'Media';
  systemPrompt: string;
  temperature?: number;
  maxTokens?: number;
  outputSchema?: string;
  isPremium?: boolean;
  enabled?: boolean;
}

export const ALL_SKILLS: Skill[] = [
  // Writing & Productivity
  {
    id: "explain",
    name: "Explain",
    description: "Explain a concept simply",
    command: "explain",
    icon: "Brain",
    category: "Writing",
    systemPrompt: "You are an expert tutor. Break down complex concepts into simple, intuitive explanations using analogies, clear examples, and step-by-step guidance. Tailor your explanation to be easily understood by anyone.",
    temperature: 0.7,
    maxTokens: 4096,
    enabled: true
  },
  {
    id: "summarize",
    name: "Summarize",
    description: "Create a concise summary",
    command: "summarize",
    icon: "Sparkles",
    category: "Writing",
    systemPrompt: "You are a professional summarizer. Distill the following text into a structured, concise summary. Capture all key facts, critical arguments, and core conclusions. Organize the summary with key takeaways first, followed by a bulleted breakdown of crucial points.",
    temperature: 0.5,
    maxTokens: 4096,
    enabled: true
  },
  {
    id: "rewrite",
    name: "Rewrite",
    description: "Rewrite text casually or professionally",
    command: "rewrite",
    icon: "RefreshCw",
    category: "Writing",
    systemPrompt: "You are an expert editor. Rewrite the provided text according to the user's instructions while keeping the core message intact. Enhance clarity, flow, readability, and engagement. Adapt the tone (professional, casual, persuasive, etc.) as requested.",
    temperature: 0.7,
    maxTokens: 4096,
    enabled: true
  },
  {
    id: "translate",
    name: "Translate",
    description: "Translate text to another language",
    command: "translate",
    icon: "Globe",
    category: "Writing",
    systemPrompt: "You are a professional translator fluent in multiple languages. Translate the provided text into the target language accurately, preserving nuance, cultural context, and the original formatting. Do not add any introductory or explanatory text in your response.",
    temperature: 0.3,
    maxTokens: 4096,
    enabled: true
  },
  {
    id: "email",
    name: "Write Email",
    description: "Generate professional emails",
    command: "email",
    icon: "Mail",
    category: "Writing",
    systemPrompt: "You are an email drafting specialist. Craft a professional, clear, and well-structured email based on the user's prompt. Ensure the tone is appropriate for the context, includes a clear subject line, and ends with a professional sign-off.",
    temperature: 0.6,
    maxTokens: 2048,
    enabled: true
  },
  // Developer Tools
  {
    id: "code",
    name: "Generate Code",
    description: "Generate high-quality code snippets",
    command: "code",
    icon: "Code2",
    category: "Development",
    systemPrompt: "You are an elite software engineer. Write clean, efficient, and well-documented code that solves the user's problem. Follow industry best practices for the specified programming language. Include explanatory comments inside the code block for tricky parts.",
    temperature: 0.2,
    maxTokens: 8192,
    enabled: true
  },
  {
    id: "debug",
    name: "Debug Code",
    description: "Find and resolve code issues",
    command: "debug",
    icon: "Bug",
    category: "Development",
    systemPrompt: "You are a senior debugging specialist. Analyze the provided code for errors, performance bottlenecks, syntax bugs, and security weaknesses. Explain the root cause of the issue and provide the corrected code block showing how to resolve it.",
    temperature: 0.1,
    maxTokens: 8192,
    enabled: true
  },
  {
    id: "reviewcode",
    name: "Review Code",
    description: "Analyze code quality and security",
    command: "reviewcode",
    icon: "Eye",
    category: "Development",
    systemPrompt: "You are a principal code architect. Perform a comprehensive code review of the submitted source. Evaluate: code quality, performance, architecture patterns, security flaws, and maintainability. Provide a structured review report highlighting improvements.",
    temperature: 0.2,
    maxTokens: 8192,
    enabled: true
  },
  // Research & Analytics
  {
    id: "websearch",
    name: "Web Search",
    description: "Search the web for current information",
    command: "websearch",
    icon: "Search",
    category: "Research",
    systemPrompt: "You are a search assistant. Utilize search results to provide accurate, factual, and up-to-date answers. Cite sources appropriately, cross-reference assertions, and clarify any ambiguities in the retrieved data.",
    temperature: 0.4,
    maxTokens: 4096,
    enabled: true
  },
  {
    id: "deepresearch",
    name: "Deep Research",
    description: "Comprehensive multi-step research",
    command: "deepresearch",
    icon: "Layers",
    category: "Research",
    systemPrompt: "You are a lead researcher. Perform an exhaustive, rigorous deep dive into the query. Utilize a scientific research methodology: identify core themes, cross-reference facts, validate sources, and structure findings into a comprehensive research report. Format output with: Executive Summary, Detailed Comparison Framework, Source Validation, Fact-Checking Process, and Confidence Scoring.",
    temperature: 0.3,
    maxTokens: 8192,
    isPremium: true,
    enabled: true
  },
  {
    id: "competitoranalysis",
    name: "Competitor Analysis",
    description: "Analyze competitor options & features",
    command: "competitoranalysis",
    icon: "Target",
    category: "Research",
    systemPrompt: "You are a market analyst. Evaluate the competitor landscapes and perform a detailed SWOT and competitor analysis for the options specified. Analyze price points, feature parity, strengths, weaknesses, and market opportunities. Format as a structured comparison report.",
    temperature: 0.3,
    maxTokens: 8192,
    isPremium: true,
    enabled: true
  },
  // Documents
  {
    id: "analyzepdf",
    name: "Analyze PDF",
    description: "Extract insights from PDF documents",
    command: "analyzepdf",
    icon: "FileText",
    category: "Documents",
    systemPrompt: "You are a document extraction system. Read the text extracted from the PDF file and answer questions based strictly on the content of the file. Provide page number citations if available. If the answer cannot be found in the document, state that clearly.",
    temperature: 0.3,
    maxTokens: 4096,
    enabled: true
  },
  {
    id: "presentation",
    name: "Generate Presentation",
    description: "Create structured presentation slides",
    command: "presentation",
    icon: "Presentation",
    category: "Documents",
    systemPrompt: "You are a presentation design consultant. Structure slide outlines, talk tracks, and key content for a slide presentation deck. Group into logical sections: Intro, Main Arguments/Data, Conclusion, and Next Steps. For each slide, write the Slide Title, Key Bullet Points, and Speaker Notes.",
    temperature: 0.6,
    maxTokens: 4096,
    isPremium: true,
    enabled: true
  },
  // Media
  {
    id: "generateimage",
    name: "Generate Image",
    description: "Create images from text descriptions",
    command: "generateimage",
    icon: "ImageIcon",
    category: "Media",
    systemPrompt: "You are an AI image generator prompt engineer. Expand the user's description into a highly descriptive, professional prompt optimized for Stable Diffusion or Flux. Include camera angles, art style, lighting parameters (e.g. volumetric lighting, octane render), and specific color schemes.",
    temperature: 0.7,
    maxTokens: 2048,
    isPremium: true,
    enabled: true
  }
];

const LOCAL_STORAGE_RECENT_KEY = "oprel_recent_skills";
const LOCAL_STORAGE_USE_COUNT_KEY = "oprel_skills_use_count";

// Load recently used skill IDs
export function getRecentSkillIds(): string[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = localStorage.getItem(LOCAL_STORAGE_RECENT_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

// Save a skill selection to local storage history
export function recordSkillUsage(skillId: string) {
  if (typeof window === "undefined") return;
  try {
    // Update Recent List (up to 5 items, unique)
    const recents = getRecentSkillIds().filter(id => id !== skillId);
    recents.unshift(skillId);
    localStorage.setItem(LOCAL_STORAGE_RECENT_KEY, JSON.stringify(recents.slice(0, 5)));

    // Update Use Counts
    const countsRaw = localStorage.getItem(LOCAL_STORAGE_USE_COUNT_KEY);
    const counts = countsRaw ? JSON.parse(countsRaw) : {};
    counts[skillId] = (counts[skillId] || 0) + 1;
    localStorage.setItem(LOCAL_STORAGE_USE_COUNT_KEY, JSON.stringify(counts));
  } catch (e) {
    console.error("Failed to persist skill usage:", e);
  }
}

// Get the top most used skills (e.g. top 4)
export function getMostUsedSkills(skillsList: Skill[] = ALL_SKILLS): Skill[] {
  if (typeof window === "undefined") return [];
  try {
    const countsRaw = localStorage.getItem(LOCAL_STORAGE_USE_COUNT_KEY);
    if (!countsRaw) return skillsList.slice(0, 4); // default fallback to first 4
    
    const counts = JSON.parse(countsRaw);
    const sortedIds = Object.keys(counts).sort((a, b) => counts[b] - counts[a]);
    
    const matchedSkills = sortedIds
      .map(id => skillsList.find(s => s.id === id))
      .filter((s): s is Skill => !!s && s.enabled !== false);
      
    // Fill in remaining if we have fewer than 4 recorded usages
    const remaining = skillsList.filter(s => s.enabled !== false && !matchedSkills.includes(s));
    return [...matchedSkills, ...remaining].slice(0, 4);
  } catch {
    return skillsList.slice(0, 4);
  }
}

// Fuzzy matching filter helper
export function filterSkills(query: string, skillsList: Skill[] = ALL_SKILLS): Skill[] {
  const cleanQuery = query.toLowerCase().trim().replace(/^\//, "");
  if (!cleanQuery) {
    return skillsList.filter(s => s.enabled !== false);
  }

  return skillsList.filter(skill => {
    if (skill.enabled === false) return false;
    const name = (skill.name || "").toLowerCase();
    const cmd = (skill.command || skill.id || "").toLowerCase();
    const desc = (skill.description || "").toLowerCase();

    // 1. Exact match on command/name
    if (cmd.startsWith(cleanQuery) || name.startsWith(cleanQuery)) return true;

    // 2. Contains query match
    if (cmd.includes(cleanQuery) || name.includes(cleanQuery) || desc.includes(cleanQuery)) return true;

    // 3. Simple character subsequencing (fuzzy)
    let queryIndex = 0;
    for (let i = 0; i < name.length && queryIndex < cleanQuery.length; i++) {
      if (name[i] === cleanQuery[queryIndex]) {
        queryIndex++;
      }
    }
    return queryIndex === cleanQuery.length;
  });
}
