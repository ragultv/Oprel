import asyncio
from oprel.server import db

MEMORY_COMPACTION_SYSTEM_PROMPT = """You maintain a compressed memory of a group discussion for one participant.

You will receive:
1. The participant's existing memory summary (may be empty on first round)
2. The raw messages from the round that just completed

Produce an updated summary under 150 words that preserves:
- Decisions or conclusions reached
- Open disagreements not yet resolved
- Facts or numbers that were verified or corrected
- Any commitment this participant made (e.g. "I will check X")

Drop: pleasantries, restated context, anything superseded by a later correction.
Output ONLY the updated summary text. No preamble, no headers.
"""

def format_round_messages(messages: list) -> str:
    """Format raw messages into a readable text block."""
    if not messages:
        return ""
    
    parts = []
    for msg in messages:
        # Determine sender name
        if msg["sender_type"] == "user":
            sender = "User"
        elif msg["sender_type"] == "system":
            sender = "System"
        else:
            # Look up agent display name if possible, or just use agent ID
            # In a real implementation we might join this in the SQL query
            sender = f"Agent {msg['member_id']}" 
            
        parts.append(f"[{sender}]: {msg['content']}")
        
    return "\n".join(parts)

def build_agent_context(group_id: str, member_id: str, round_id: str) -> str:
    """Builds the bounded context string for a specific agent generation call."""
    summary = db.get_agent_memory(group_id, member_id)
    raw_messages = db.get_group_messages(group_id, round_id=round_id)
    
    current_round_raw = format_round_messages(raw_messages)
    
    context_parts = []
    if summary:
        context_parts.append(f"--- Group Discussion Summary ---\n{summary}\n")
        
    context_parts.append(f"--- Current Round ---\n{current_round_raw}")
    return "\n".join(context_parts)

async def _compact_memory_for_member(group_id: str, round_id: str, member: dict, current_round_raw: str):
    """Internal function to run the compaction LLM call for a single member."""
    existing_summary = db.get_agent_memory(group_id, member["id"])
    
    prompt = f"Existing Memory:\n{existing_summary if existing_summary else '(None)'}\n\nNew Round Messages:\n{current_round_raw}"
    
    # MOCK LLM CALL
    # In full implementation, call the cheapest available model for this member's provider
    # updated_summary = await call_llm(system=MEMORY_COMPACTION_SYSTEM_PROMPT, user=prompt)
    updated_summary = existing_summary + "\n(Appended mock summary of new round)" if existing_summary else "(Initial mock summary)"
    
    db.update_agent_memory(group_id, member["id"], updated_summary.strip())

async def compact_memory_for_all_members(group_id: str, round_id: str):
    """
    Run memory compaction for all agents in the group.
    Called at the end of a round (state -> DONE).
    """
    group = db.get_group(group_id)
    if not group or not group.get("members"):
        return
        
    raw_messages = db.get_group_messages(group_id, round_id=round_id)
    current_round_raw = format_round_messages(raw_messages)
    
    if not current_round_raw:
        return
        
    # Run compactions in parallel to avoid blocking
    tasks = []
    for member in group["members"]:
        tasks.append(_compact_memory_for_member(group_id, round_id, member, current_round_raw))
        
    await asyncio.gather(*tasks, return_exceptions=True)
