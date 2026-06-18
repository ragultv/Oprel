import sqlite3
import uuid
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, List
from oprel.core.config import Config

CONFIG = Config()
DB_PATH = CONFIG.cache_dir / "chat_history.db"

def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    cursor = conn.cursor()
    # Create conversations table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            title TEXT,
            model_id TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # Create messages table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
        )
    """)
    # Create users table for profile info
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            name TEXT NOT NULL,
            role TEXT NOT NULL,
            avatar_initials TEXT,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # Create user_settings table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_settings (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            temperature REAL,
            top_p REAL,
            top_k INTEGER,
            repeat_penalty REAL,
            max_tokens INTEGER,
            system_instruction TEXT,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # Create canvas_documents table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS canvas_documents (
            conversation_id TEXT PRIMARY KEY,
            title TEXT,
            content TEXT,
            card_timestamp TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
        )
    """)
    # Create download_logs table for persistent download history
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS download_logs (
            id TEXT PRIMARY KEY,
            model_id TEXT NOT NULL,
            model_name TEXT,
            quantization TEXT,
            status TEXT NOT NULL,
            size_bytes INTEGER DEFAULT 0,
            duration_seconds REAL DEFAULT 0,
            error TEXT,
            started_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            completed_at DATETIME
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_download_logs_time ON download_logs(started_at DESC)")
    
    # Create inference_logs table for analytics
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS inference_logs (
            id TEXT PRIMARY KEY,
            model_id TEXT NOT NULL,
            prompt_tokens INTEGER DEFAULT 0,
            completion_tokens INTEGER DEFAULT 0,
            latency_ms REAL DEFAULT 0,
            tps REAL DEFAULT 0,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_inference_logs_time ON inference_logs(created_at DESC)")

    # Create provider_configs table — API keys & enabled models for external providers
    # api_key is stored as-is; for production deployments consider encrypting at rest.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS provider_configs (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            type TEXT NOT NULL,
            api_key TEXT NOT NULL DEFAULT '',
            base_url TEXT DEFAULT '',
            enabled INTEGER NOT NULL DEFAULT 1,
            enabled_model_ids TEXT NOT NULL DEFAULT '[]',
            available_model_ids TEXT NOT NULL DEFAULT '[]',
            last_fetched TEXT DEFAULT NULL,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create skills table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS skills (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT,
            system_prompt TEXT NOT NULL,
            category TEXT NOT NULL,
            icon TEXT NOT NULL,
            temperature REAL,
            max_tokens INTEGER,
            output_schema TEXT,
            is_premium INTEGER DEFAULT 0,
            enabled INTEGER DEFAULT 1,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # ── Pre-populate or migrate default skills ────────────────────────────────
    default_skills = [
        ("explain", "Explain", "Explain a concept simply", "You are an expert tutor. Break down complex concepts into simple, intuitive explanations using analogies, clear examples, and step-by-step guidance. Tailor your explanation to be easily understood by anyone.", "Writing", "Brain", 0.7, 4096, None, 0, 1),
        ("summarize", "Summarize", "Create a concise summary", "You are a professional summarizer. Distill the following text into a structured, concise summary. Capture all key facts, critical arguments, and core conclusions. Organize the summary with key takeaways first, followed by a bulleted breakdown of crucial points.", "Writing", "Sparkles", 0.5, 4096, None, 0, 1),
        ("rewrite", "Rewrite", "Rewrite text casually or professionally", "You are an expert editor. Rewrite the provided text according to the user's instructions while keeping the core message intact. Enhance clarity, flow, readability, and engagement. Adapt the tone (professional, casual, persuasive, etc.) as requested.", "Writing", "RefreshCw", 0.7, 4096, None, 0, 1),
        ("translate", "Translate", "Translate text to another language", "You are a professional translator fluent in multiple languages. Translate the provided text into the target language accurately, preserving nuance, cultural context, and the original formatting. Do not add any introductory or explanatory text in your response.", "Writing", "Globe", 0.3, 4096, None, 0, 1),
        ("email", "Write Email", "Generate professional emails", "You are an email drafting specialist. Craft a professional, clear, and well-structured email based on the user's prompt. Ensure the tone is appropriate for the context, includes a clear subject line, and ends with a professional sign-off.", "Writing", "Mail", 0.6, 2048, None, 0, 1),
        ("code", "Generate Code", "Generate high-quality code snippets", "You are an elite software engineer. Write clean, efficient, and well-documented code that solves the user's problem. Follow industry best practices for the specified programming language. Include explanatory comments inside the code block for tricky parts.", "Development", "Code2", 0.2, 8192, None, 0, 1),
        ("debug", "Debug Code", "Find and resolve code issues", "You are a senior debugging specialist. Analyze the provided code for errors, performance bottlenecks, syntax bugs, and security weaknesses. Explain the root cause of the issue and provide the corrected code block showing how to resolve it.", "Development", "Bug", 0.1, 8192, None, 0, 1),
        ("reviewcode", "Review Code", "Analyze code quality and security", "You are a principal code architect. Perform a comprehensive code review of the submitted source. Evaluate: code quality, performance, architecture patterns, security flaws, and maintainability. Provide a structured review report highlighting improvements.", "Development", "Eye", 0.2, 8192, None, 0, 1),
        ("websearch", "Web Search", "Search the web for current information", "You are a search assistant. Utilize search results to provide accurate, factual, and up-to-date answers. Cite sources appropriately, cross-reference assertions, and clarify any ambiguities in the retrieved data.", "Research", "Search", 0.4, 4096, None, 0, 1),
        ("deepresearch", "Deep Research", "Comprehensive multi-step research", "You are a lead researcher. Perform an exhaustive, rigorous deep dive into the query. Utilize a scientific research methodology: identify core themes, cross-reference facts, validate sources, and structure findings into a comprehensive research report. Format output with: Executive Summary, Detailed Comparison Framework, Source Validation, Fact-Checking Process, and Confidence Scoring.", "Research", "Layers", 0.3, 8192, None, 1, 1),
        ("competitoranalysis", "Competitor Analysis", "Analyze competitor options & features", "You are a market analyst. Evaluate the competitor landscapes and perform a detailed SWOT and competitor analysis for the options specified. Analyze price points, feature parity, strengths, weaknesses, and market opportunities. Format as a structured comparison report.", "Research", "Target", 0.3, 8192, None, 1, 1),
        ("analyzepdf", "Analyze PDF", "Extract insights from PDF documents", "You are a document extraction system. Read the text extracted from the PDF file and answer questions based strictly on the content of the file. Provide page number citations if available. If the answer cannot be found in the document, state that clearly.", "Documents", "FileText", 0.3, 4096, None, 0, 1),
        ("presentation", "Generate Presentation", "Create structured presentation slides", "You are a presentation design consultant. Structure slide outlines, talk tracks, and key content for a slide presentation deck. Group into logical sections: Intro, Main Arguments/Data, Conclusion, and Next Steps. For each slide, write the Slide Title, Key Bullet Points, and Speaker Notes.", "Documents", "Presentation", 0.6, 4096, None, 1, 1),
        ("generateimage", "Generate Image", "Create images from text descriptions", "You are an AI image generator prompt engineer. Expand the user's description into a highly descriptive, professional prompt optimized for Stable Diffusion or Flux. Include camera angles, art style, lighting parameters (e.g. volumetric lighting, octane render), and specific color schemes.", "Media", "ImageIcon", 0.7, 2048, None, 1, 1),
        ("diagrams", "Diagrams", "Create Mermaid diagrams", "You are a technical diagram expert. When asked to create diagrams, output valid Mermaid syntax inside ```mermaid code blocks. Always produce syntactically correct Mermaid.\n\nCRITICAL RULES FOR MERMAID:\n1. NEVER use spaces in subgraph IDs. Format as: `subgraph ID [\"Title\"]` (e.g. `subgraph ClientLayer [\"Client Layer\"]`).\n2. ALWAYS wrap node labels containing parentheses or special characters in double quotes (e.g., `A[\"Input (X)\"]`).\n3. ALWAYS use newlines to separate statements (never put `subgraph` on the same line as a node).\n4. Use standard ASCII characters only. Avoid special hyphens.", "Development", "TrendingUp", 0.1, 4096, None, 0, 1),
        ("webbuilder", "Web Builder", "Build self-contained web UIs", "You are a senior frontend engineer. When asked to build UIs, produce complete, self-contained HTML files with embedded CSS and JS inside ```html code blocks.", "Development", "Code2", 0.2, 8192, None, 0, 1),
        ("writer", "Writer", "Professional writing and editing", "You are a professional writer and editor. Help craft clear, engaging, and well-structured prose. Adapt tone to the request.", "Writing", "Edit3", 0.7, 4096, None, 0, 1),
        ("analyst", "Data Analyst", "Analyze data and trends", "You are a data analyst. Provide structured analysis, identify trends, and support conclusions with reasoning and data.", "Research", "TrendingUp", 0.3, 4096, None, 0, 1),
        ("sql", "Data / SQL", "Write performant SQL queries", "You are a database expert specialising in SQL. Write performant, standards-compliant queries. Explain query plans when asked.", "Development", "Target", 0.1, 4096, None, 0, 1),
        ("devops", "DevOps", "Cloud infrastructure solutions", "You are a DevOps and cloud infrastructure expert. Provide practical, secure, and scalable solutions using industry best practices.", "Development", "Terminal", 0.2, 4096, None, 0, 1),
        ("coach", "Life Coach", "Supportive life & productivity coach", "You are a supportive life and productivity coach. Help the user clarify goals, overcome obstacles, and build positive habits.", "Writing", "Brain", 0.7, 4096, None, 0, 1)
    ]
    cursor.executemany("""
        INSERT OR IGNORE INTO skills (id, name, description, system_prompt, category, icon, temperature, max_tokens, output_schema, is_premium, enabled)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, default_skills)
    
    # ── OCR jobs table ────────────────────────────────────────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ocr_jobs (
            id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            image_data TEXT NOT NULL,
            result_json TEXT NOT NULL,
            full_text TEXT NOT NULL,
            word_count INTEGER NOT NULL DEFAULT 0,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # ── Migrations: add missing columns to existing tables ───────────────────
    # ── Oprel AI Groups Tables ────────────────────────────────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS groups (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            moderator_member_id TEXT,
            max_interrupt_rounds INTEGER DEFAULT 3,
            max_replies_per_agent INTEGER
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS group_members (
            id TEXT PRIMARY KEY,
            group_id TEXT NOT NULL,
            kind TEXT NOT NULL, -- 'cloud' or 'local'
            provider_id TEXT,
            model_id TEXT NOT NULL,
            display_name TEXT NOT NULL,
            role_description TEXT,
            is_moderator INTEGER DEFAULT 0,
            priority_order INTEGER DEFAULT 0,
            FOREIGN KEY (group_id) REFERENCES groups(id) ON DELETE CASCADE
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS group_rounds (
            id TEXT PRIMARY KEY,
            group_id TEXT NOT NULL,
            user_message_id TEXT NOT NULL,
            state TEXT NOT NULL, -- 'relevance', 'generation', 'interrupt', 'moderation', 'done'
            interrupt_count INTEGER DEFAULT 0,
            moderation_retries INTEGER DEFAULT 0,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (group_id) REFERENCES groups(id) ON DELETE CASCADE
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS group_messages (
            id TEXT PRIMARY KEY,
            group_id TEXT NOT NULL,
            round_id TEXT NOT NULL,
            sender_type TEXT NOT NULL, -- 'user', 'agent', 'system'
            member_id TEXT, -- nullable for user/system
            content TEXT NOT NULL,
            message_type TEXT NOT NULL, -- 'reply', 'interrupt', 'final_answer', 'user'
            sequence_number INTEGER NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (group_id) REFERENCES groups(id) ON DELETE CASCADE,
            FOREIGN KEY (round_id) REFERENCES group_rounds(id) ON DELETE CASCADE
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS group_reactions (
            id TEXT PRIMARY KEY,
            message_id TEXT NOT NULL,
            member_id TEXT NOT NULL,
            emoji TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (message_id) REFERENCES group_messages(id) ON DELETE CASCADE,
            FOREIGN KEY (member_id) REFERENCES group_members(id) ON DELETE CASCADE
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS agent_memory (
            group_id TEXT NOT NULL,
            member_id TEXT NOT NULL,
            rolling_summary TEXT NOT NULL DEFAULT '',
            last_compacted_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (group_id, member_id),
            FOREIGN KEY (group_id) REFERENCES groups(id) ON DELETE CASCADE,
            FOREIGN KEY (member_id) REFERENCES group_members(id) ON DELETE CASCADE
        )
    """)
    
    # This handles databases created before schema additions
    try:
        cursor.execute("ALTER TABLE canvas_documents ADD COLUMN card_timestamp TEXT")
    except Exception:
        pass  # column already exists

    # Fix diagrams skill prompt for existing databases to prevent Mermaid parse errors
    cursor.execute("""
        UPDATE skills 
        SET system_prompt = 'You are a technical diagram expert. When asked to create diagrams, output valid Mermaid syntax inside ```mermaid code blocks. Always produce syntactically correct Mermaid.\n\nCRITICAL RULES FOR MERMAID:\n1. NEVER use spaces in subgraph IDs. Format as: `subgraph ID ["Title"]` (e.g. `subgraph ClientLayer ["Client Layer"]`).\n2. ALWAYS wrap node labels containing parentheses or special characters in double quotes (e.g., `A["Input (X)"]`).\n3. ALWAYS use newlines to separate statements (never put `subgraph` on the same line as a node).\n4. Use standard ASCII characters only. Avoid special hyphens.'
        WHERE id = 'diagrams'
    """)

    conn.commit()
    conn.close()

def create_conversation(model_id: str, title: str = "New Chat", conversation_id: str = None) -> str:
    """Create a new conversation with optional custom ID"""
    conn = get_db()
    cursor = conn.cursor()
    conv_id = conversation_id if conversation_id else f"chat_{uuid.uuid4().hex[:12]}"
    now = datetime.now().isoformat()
    cursor.execute(
        "INSERT INTO conversations (id, title, model_id, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
        (conv_id, title, model_id, now, now)
    )
    conn.commit()
    conn.close()
    return conv_id
def delete_conversation(conversation_id: str):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM canvas_documents WHERE conversation_id = ?", (conversation_id,))
    cursor.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
    cursor.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
    conn.commit()
    conn.close()

def rename_conversation(conversation_id: str, new_title: str):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?", 
                   (new_title, datetime.now().isoformat(), conversation_id))
    conn.commit()
    conn.close()

def reset_conversation(conversation_id: str):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM canvas_documents WHERE conversation_id = ?", (conversation_id,))
    cursor.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
    now = datetime.now().isoformat()
    cursor.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conversation_id))
    conn.commit()
    conn.close()

def add_message(conversation_id: str, role: str, content: Any):
    import json
    raw_content = content
    if not isinstance(content, str):
        content = json.dumps(content)
        
    conn = get_db()
    cursor = conn.cursor()
    msg_id = f"msg_{uuid.uuid4().hex[:12]}"
    now = datetime.now().isoformat()
    cursor.execute(
        "INSERT INTO messages (id, conversation_id, role, content, created_at) VALUES (?, ?, ?, ?, ?)",
        (msg_id, conversation_id, role, content, now)
    )
    # Update timestamp and possibly title based on first user message
    cursor.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conversation_id))
    
    # Optional: Update title if it's the first user message
    if role == "user":
        # If content is a list (vision), use the text part if available
        title_content = raw_content
        if isinstance(raw_content, list):
            try:
                # Expecting [{type: text, text: ...}, ...]
                for item in raw_content:
                    if isinstance(item, dict) and item.get('type') == 'text':
                        title_content = item.get('text', '')
                        break
            except:
                pass

        cursor.execute("SELECT COUNT(*) FROM messages WHERE conversation_id = ? AND role = 'user'", (conversation_id,))
        count = cursor.fetchone()[0]
        if count == 1:
            title = str(title_content)[:30] + "..." if len(str(title_content)) > 30 else str(title_content)
            cursor.execute("UPDATE conversations SET title = ? WHERE id = ?", (title, conversation_id))
            
    conn.commit()
    conn.close()

def get_conversation_messages(conversation_id: str):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY created_at ASC",
        (conversation_id,)
    )
    rows = cursor.fetchall()
    conn.close()
    
    import json
    messages = []
    for row in rows:
        content = row["content"]
        if isinstance(content, str) and (content.startswith('[') or content.startswith('{')):
            try:
                content = json.loads(content)
            except:
                pass
        messages.append({"role": row["role"], "content": content})
    return messages

def list_conversations():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT c.id, c.title, c.model_id, c.created_at, c.updated_at, COUNT(m.id) as message_count
        FROM conversations c
        LEFT JOIN messages m ON c.id = m.conversation_id
        GROUP BY c.id
        ORDER BY c.updated_at DESC
        LIMIT 100
    """)
    rows = cursor.fetchall()
    conn.close()
    return [
        {
            "id": row["id"],
            "title": row["title"],
            "model_id": row["model_id"],
            "created_at": row["created_at"],
            "last_updated": row["updated_at"],
            "message_count": row["message_count"],
        }
        for row in rows
    ]

def get_active_conversation_count():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM conversations")
    count = cursor.fetchone()[0]
    conn.close()
    return count

def get_user():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT name, role, avatar_initials FROM users WHERE id = 1")
    row = cursor.fetchone()
    conn.close()
    if row:
        return {"name": row["name"], "role": row["role"], "initials": row["avatar_initials"]}
    return None

def set_user(name: str, role: str):
    conn = get_db()
    cursor = conn.cursor()
    # Get initials from name
    initials = "".join([n[0] for n in name.split()[:2]]).upper()
    
    cursor.execute("""
        INSERT OR REPLACE INTO users (id, name, role, avatar_initials, updated_at)
        VALUES (1, ?, ?, ?, ?)
    """, (name, role, initials, datetime.now().isoformat()))
    conn.commit()
    conn.close()
    return {"name": name, "role": role, "initials": initials}

def get_user_settings():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM user_settings WHERE id = 1")
    row = cursor.fetchone()
    conn.close()
    if row:
        return dict(row)
    return None

def set_user_settings(settings: dict):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO user_settings (id, temperature, top_p, top_k, repeat_penalty, max_tokens, system_instruction, updated_at)
        VALUES (1, ?, ?, ?, ?, ?, ?, ?)
    """, (
        settings.get("temperature"),
        settings.get("top_p"),
        settings.get("top_k"),
        settings.get("repeat_penalty"),
        settings.get("max_tokens"),
        settings.get("system_instruction"),
        datetime.now().isoformat()
    ))
    conn.commit()
    conn.close()
    return settings

# ──────────────────────────────────────────────────────────────────────────────
# Canvas CRUD
# ──────────────────────────────────────────────────────────────────────────────

def get_canvas_document(conversation_id: str) -> Optional[dict]:
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM canvas_documents WHERE conversation_id = ?", (conversation_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return dict(row)
    return None

def upsert_canvas_document(conversation_id: str, title: str, content: str, card_timestamp: str = None) -> dict:
    conn = get_db()
    cursor = conn.cursor()
    now = datetime.now().isoformat()
    cursor.execute("""
        INSERT INTO canvas_documents (conversation_id, title, content, card_timestamp, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(conversation_id) DO UPDATE SET
            title = excluded.title,
            content = excluded.content,
            card_timestamp = excluded.card_timestamp,
            updated_at = excluded.updated_at
    """, (conversation_id, title, content, card_timestamp, now, now))
    conn.commit()
    conn.close()
    return get_canvas_document(conversation_id)

# Initialize DB when module is loaded
init_db()


def save_download_log(model_id: str, model_name: str, quantization: str, status: str,
                      size_bytes: int = 0, duration_seconds: float = 0,
                      error: str = None, started_at: str = None, completed_at: str = None):
    """Persist a download event to the logs table."""
    conn = get_db()
    cursor = conn.cursor()
    log_id = f"dl_{uuid.uuid4().hex[:12]}"
    now = datetime.now().isoformat()
    cursor.execute("""
        INSERT INTO download_logs
            (id, model_id, model_name, quantization, status, size_bytes,
             duration_seconds, error, started_at, completed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        log_id, model_id, model_name, quantization, status,
        size_bytes, duration_seconds, error,
        started_at or now, completed_at or (now if status != 'downloading' else None)
    ))
    conn.commit()
    conn.close()
    return log_id


def list_download_logs(limit: int = 100):
    """Return recent download log entries, newest first."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, model_id, model_name, quantization, status, size_bytes,
               duration_seconds, error, started_at, completed_at
        FROM download_logs
        ORDER BY started_at DESC
        LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [
        {
            "id": r["id"],
            "model_id": r["model_id"],
            "model_name": r["model_name"],
            "quantization": r["quantization"],
            "status": r["status"],
            "size_bytes": r["size_bytes"],
            "duration_seconds": r["duration_seconds"],
            "error": r["error"],
            "started_at": r["started_at"],
            "completed_at": r["completed_at"],
        }
        for r in rows
    ]


def add_inference_log(model_id: str, prompt_tokens: int, completion_tokens: int, latency_ms: float, tps: float):
    """Log an inference event for analytics."""
    conn = get_db()
    cursor = conn.cursor()
    log_id = f"inf_{uuid.uuid4().hex[:12]}"
    now = datetime.now().isoformat()
    cursor.execute("""
        INSERT INTO inference_logs (id, model_id, prompt_tokens, completion_tokens, latency_ms, tps, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (log_id, model_id, prompt_tokens, completion_tokens, latency_ms, tps, now))
    conn.commit()
    conn.close()
    return log_id


def get_inference_summary(days: int = 7):
    """Retrieve summary stats for the last N days."""
    conn = get_db()
    cursor = conn.cursor()
    # Simplified time calculation for sqlite
    cursor.execute("""
        SELECT 
            COUNT(*) as total_requests,
            SUM(prompt_tokens) as total_prompt_tokens,
            SUM(completion_tokens) as total_completion_tokens,
            AVG(latency_ms) as avg_latency,
            AVG(tps) as avg_tps,
            model_id
        FROM inference_logs
        WHERE created_at >= datetime('now', '-' || ? || ' days')
        GROUP BY model_id
    """, (days,))
    rows = cursor.fetchall()
    
    # Get hourly distribution for charts
    cursor.execute("""
        SELECT 
            strftime('%Y-%m-%d %H:00:00', created_at) as hour,
            SUM(prompt_tokens + completion_tokens) as total_tokens,
            AVG(tps) as tps
        FROM inference_logs
        WHERE created_at >= datetime('now', '-' || ? || ' days')
        GROUP BY hour
        ORDER BY hour ASC
    """, (days,))
    timeline = cursor.fetchall()
    
    conn.close()
    return {
        "models": [dict(r) for r in rows],
        "timeline": [dict(t) for t in timeline]
    }


# ──────────────────────────────────────────────────────────────────────────────
# Provider config CRUD
# ──────────────────────────────────────────────────────────────────────────────

def list_providers() -> list:
    """Return all configured external AI providers."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM provider_configs ORDER BY updated_at DESC")
    rows = cursor.fetchall()
    conn.close()
    return [_row_to_provider(r) for r in rows]


def get_provider(provider_id: str) -> Optional[dict]:
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM provider_configs WHERE id = ?", (provider_id,))
    row = cursor.fetchone()
    conn.close()
    return _row_to_provider(row) if row else None


def upsert_provider(p: dict) -> dict:
    """Insert or update a provider config. `p` must have at least 'id', 'name', 'type', 'api_key'."""
    conn = get_db()
    cursor = conn.cursor()
    now = datetime.now().isoformat()
    cursor.execute("""
        INSERT INTO provider_configs
            (id, name, type, api_key, base_url, enabled, enabled_model_ids, available_model_ids, last_fetched, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            name = excluded.name,
            type = excluded.type,
            api_key = excluded.api_key,
            base_url = excluded.base_url,
            enabled = excluded.enabled,
            enabled_model_ids = excluded.enabled_model_ids,
            available_model_ids = excluded.available_model_ids,
            last_fetched = excluded.last_fetched,
            updated_at = excluded.updated_at
    """, (
        p["id"], p["name"], p["type"],
        p.get("api_key", ""),
        p.get("base_url", ""),
        1 if p.get("enabled", True) else 0,
        json.dumps(p.get("enabled_model_ids", [])),
        json.dumps(p.get("available_model_ids", [])),
        p.get("last_fetched"),
        now,
    ))
    conn.commit()
    conn.close()
    return get_provider(p["id"])


def delete_provider(provider_id: str):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM provider_configs WHERE id = ?", (provider_id,))
    conn.commit()
    conn.close()


def _row_to_provider(row) -> dict:
    d = dict(row)
    d["enabled"] = bool(d.get("enabled", 1))
    try:
        d["enabled_model_ids"] = json.loads(d.get("enabled_model_ids") or "[]")
    except Exception:
        d["enabled_model_ids"] = []
    try:
        d["available_model_ids"] = json.loads(d.get("available_model_ids") or "[]")
    except Exception:
        d["available_model_ids"] = []
    return d


# ──────────────────────────────────────────────────────────────────────────────
# Skills CRUD
# ──────────────────────────────────────────────────────────────────────────────

def list_skills() -> List[dict]:
    """Return all configured skills."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM skills ORDER BY category, name ASC")
    rows = cursor.fetchall()
    conn.close()
    
    skills = []
    for r in rows:
        d = dict(r)
        d["is_premium"] = bool(d.get("is_premium", 0))
        d["enabled"] = bool(d.get("enabled", 1))
        skills.append(d)
    return skills


def get_skill(skill_id: str) -> Optional[dict]:
    """Retrieve a single skill by ID."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM skills WHERE id = ?", (skill_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        d = dict(row)
        d["is_premium"] = bool(d.get("is_premium", 0))
        d["enabled"] = bool(d.get("enabled", 1))
        return d
    return None


def upsert_skill(s: dict) -> dict:
    """Insert or update a skill."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO skills (id, name, description, system_prompt, category, icon, temperature, max_tokens, output_schema, is_premium, enabled)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            name = excluded.name,
            description = excluded.description,
            system_prompt = excluded.system_prompt,
            category = excluded.category,
            icon = excluded.icon,
            temperature = excluded.temperature,
            max_tokens = excluded.max_tokens,
            output_schema = excluded.output_schema,
            is_premium = excluded.is_premium,
            enabled = excluded.enabled
    """, (
        s["id"], s["name"], s.get("description", ""), s["system_prompt"], s["category"], s["icon"],
        s.get("temperature"), s.get("max_tokens"), s.get("output_schema"),
        1 if s.get("is_premium") else 0, 1 if s.get("enabled", True) else 0
    ))
    conn.commit()
    conn.close()
    return get_skill(s["id"])


def delete_skill(skill_id: str):
    """Delete a skill by ID."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM skills WHERE id = ?", (skill_id,))
    conn.commit()
    conn.close()


# ── OCR CRUD ──────────────────────────────────────────────────────────────────

def add_ocr_job(job_id: str, filename: str, image_data: str, result_json: str, full_text: str, word_count: int) -> None:
    """Persist an OCR extraction result to the database."""
    conn = get_db()
    cursor = conn.cursor()
    now = datetime.now().isoformat()
    cursor.execute(
        "INSERT INTO ocr_jobs (id, filename, image_data, result_json, full_text, word_count, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (job_id, filename, image_data, result_json, full_text, word_count, now),
    )
    conn.commit()
    conn.close()


def get_ocr_jobs(limit: int = 50) -> list:
    """Return the most recent OCR jobs, newest first."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, filename, image_data, result_json, full_text, word_count, created_at FROM ocr_jobs ORDER BY created_at DESC LIMIT ?",
        (limit,),
    )
    rows = cursor.fetchall()
    conn.close()
    return [
        {
            "id": r["id"],
            "filename": r["filename"],
            "image_data": r["image_data"],
            "result_json": r["result_json"],
            "full_text": r["full_text"],
            "word_count": r["word_count"],
            "created_at": r["created_at"],
        }
        for r in rows
    ]


def delete_ocr_job(job_id: str) -> None:
    """Delete a single OCR job by ID."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM ocr_jobs WHERE id = ?", (job_id,))
    conn.commit()
    conn.close()

# ── Oprel AI Groups CRUD ────────────────────────────────────────────────────────

def create_group(name: str, max_interrupt_rounds: int = 3, max_replies_per_agent: Optional[int] = None) -> dict:
    conn = get_db()
    cursor = conn.cursor()
    group_id = f"grp_{uuid.uuid4().hex[:12]}"
    now = datetime.now().isoformat()
    cursor.execute(
        "INSERT INTO groups (id, name, created_at, max_interrupt_rounds, max_replies_per_agent) VALUES (?, ?, ?, ?, ?)",
        (group_id, name, now, max_interrupt_rounds, max_replies_per_agent)
    )
    conn.commit()
    conn.close()
    return get_group(group_id)

def get_group(group_id: str) -> Optional[dict]:
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM groups WHERE id = ?", (group_id,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        return None
        
    group = dict(row)
    cursor.execute("SELECT * FROM group_members WHERE group_id = ?", (group_id,))
    members = [dict(m) for m in cursor.fetchall()]
    group["members"] = members
    conn.close()
    return group

def list_groups() -> List[dict]:
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM groups ORDER BY created_at DESC")
    rows = cursor.fetchall()
    groups = []
    for r in rows:
        g = dict(r)
        cursor.execute("SELECT COUNT(*) FROM group_members WHERE group_id = ?", (g["id"],))
        g["member_count"] = cursor.fetchone()[0]
        groups.append(g)
    conn.close()
    return groups

def update_group(group_id: str, updates: dict) -> Optional[dict]:
    conn = get_db()
    cursor = conn.cursor()
    
    set_clauses = []
    values = []
    for k, v in updates.items():
        if k in ["name", "moderator_member_id", "max_interrupt_rounds", "max_replies_per_agent"]:
            set_clauses.append(f"{k} = ?")
            values.append(v)
            
    if not set_clauses:
        conn.close()
        return get_group(group_id)
        
    values.append(group_id)
    query = f"UPDATE groups SET {', '.join(set_clauses)} WHERE id = ?"
    cursor.execute(query, values)
    conn.commit()
    conn.close()
    return get_group(group_id)

def delete_group(group_id: str) -> None:
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM groups WHERE id = ?", (group_id,))
    conn.commit()
    conn.close()

def add_group_member(group_id: str, kind: str, model_id: str, display_name: str, 
                     provider_id: Optional[str] = None, role_description: Optional[str] = None,
                     is_moderator: bool = False, priority_order: int = 0) -> dict:
    conn = get_db()
    cursor = conn.cursor()
    member_id = f"mem_{uuid.uuid4().hex[:12]}"
    
    # Check constraints
    if kind == 'local':
        cursor.execute("SELECT COUNT(*) FROM group_members WHERE group_id = ? AND kind = 'local'", (group_id,))
        if cursor.fetchone()[0] > 0:
            conn.close()
            raise ValueError("Only one local member allowed per group")
            
    if is_moderator:
        cursor.execute("SELECT id FROM group_members WHERE group_id = ? AND is_moderator = 1", (group_id,))
        existing_mod = cursor.fetchone()
        if existing_mod:
            # Demote existing
            cursor.execute("UPDATE group_members SET is_moderator = 0 WHERE id = ?", (existing_mod["id"],))
            
    cursor.execute("""
        INSERT INTO group_members (id, group_id, kind, provider_id, model_id, display_name, role_description, is_moderator, priority_order)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (member_id, group_id, kind, provider_id, model_id, display_name, role_description, 1 if is_moderator else 0, priority_order))
    
    if is_moderator:
        cursor.execute("UPDATE groups SET moderator_member_id = ? WHERE id = ?", (member_id, group_id))
        
    conn.commit()
    
    cursor.execute("SELECT * FROM group_members WHERE id = ?", (member_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row)

def remove_group_member(group_id: str, member_id: str) -> None:
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM group_members WHERE group_id = ? AND id = ?", (group_id, member_id))
    cursor.execute("UPDATE groups SET moderator_member_id = NULL WHERE id = ? AND moderator_member_id = ?", (group_id, member_id))
    conn.commit()
    conn.close()

def get_group_messages(group_id: str, round_id: Optional[str] = None, limit: int = 100) -> list:
    conn = get_db()
    cursor = conn.cursor()
    
    if round_id:
        cursor.execute("""
            SELECT * FROM group_messages 
            WHERE group_id = ? AND round_id = ? 
            ORDER BY sequence_number ASC
        """, (group_id, round_id))
    else:
        cursor.execute("""
            SELECT * FROM group_messages 
            WHERE group_id = ? 
            ORDER BY created_at DESC LIMIT ?
        """, (group_id, limit))
        
    rows = cursor.fetchall()
    
    messages = []
    for r in rows:
        msg = dict(r)
        cursor.execute("SELECT * FROM group_reactions WHERE message_id = ?", (msg["id"],))
        msg["reactions"] = [dict(reac) for reac in cursor.fetchall()]
        messages.append(msg)
        
    conn.close()
    if not round_id:
        messages.reverse() # If fetched by DESC, reverse to chronological
    return messages

def add_reaction(message_id: str, member_id: str, emoji: str) -> None:
    conn = get_db()
    cursor = conn.cursor()
    reac_id = f"react_{uuid.uuid4().hex[:12]}"
    now = datetime.now().isoformat()
    cursor.execute("""
        INSERT INTO group_reactions (id, message_id, member_id, emoji, created_at)
        VALUES (?, ?, ?, ?, ?)
    """, (reac_id, message_id, member_id, emoji, now))
    conn.commit()
    conn.close()

def get_agent_memory(group_id: str, member_id: str) -> str:
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT rolling_summary FROM agent_memory WHERE group_id = ? AND member_id = ?", (group_id, member_id))
    row = cursor.fetchone()
    conn.close()
    if row:
        return row["rolling_summary"]
    return ""

def update_agent_memory(group_id: str, member_id: str, new_summary: str) -> None:
    from datetime import datetime
    conn = get_db()
    cursor = conn.cursor()
    now = datetime.now().isoformat()
    cursor.execute("""
        INSERT INTO agent_memory (group_id, member_id, rolling_summary, last_compacted_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(group_id, member_id) DO UPDATE SET
            rolling_summary = excluded.rolling_summary,
            last_compacted_at = excluded.last_compacted_at
    """, (group_id, member_id, new_summary, now))
    conn.commit()
    conn.close()

def create_group_round(group_id: str, user_message_id: str) -> dict:
    conn = get_db()
    cursor = conn.cursor()
    round_id = f"rnd_{uuid.uuid4().hex[:12]}"
    now = datetime.now().isoformat()
    cursor.execute("""
        INSERT INTO group_rounds (id, group_id, user_message_id, state, created_at)
        VALUES (?, ?, ?, 'relevance', ?)
    """, (round_id, group_id, user_message_id, now))
    conn.commit()
    cursor.execute("SELECT * FROM group_rounds WHERE id = ?", (round_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row)

def get_group_round(round_id: str) -> Optional[dict]:
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM group_rounds WHERE id = ?", (round_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def update_group_round_state(round_id: str, state: str, increment_interrupt: bool = False) -> None:
    conn = get_db()
    cursor = conn.cursor()
    if increment_interrupt:
        cursor.execute("UPDATE group_rounds SET state = ?, interrupt_count = interrupt_count + 1 WHERE id = ?", (state, round_id))
    else:
        cursor.execute("UPDATE group_rounds SET state = ? WHERE id = ?", (state, round_id))
    conn.commit()
    conn.close()

def get_max_sequence_number(group_id: str, round_id: str) -> int:
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(sequence_number) FROM group_messages WHERE group_id = ? AND round_id = ?", (group_id, round_id))
    row = cursor.fetchone()
    conn.close()
    return row[0] if row[0] is not None else 0

def add_group_message(group_id: str, round_id: str, sender_type: str, member_id: Optional[str], 
                      content: str, message_type: str, sequence_number: int, msg_id: Optional[str] = None) -> dict:
    conn = get_db()
    cursor = conn.cursor()
    msg_id = msg_id or f"gmsg_{uuid.uuid4().hex[:12]}"
    now = datetime.now().isoformat()
    cursor.execute("""
        INSERT INTO group_messages (id, group_id, round_id, sender_type, member_id, content, message_type, sequence_number, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (msg_id, group_id, round_id, sender_type, member_id, content, message_type, sequence_number, now))
    conn.commit()
    cursor.execute("SELECT * FROM group_messages WHERE id = ?", (msg_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row)
