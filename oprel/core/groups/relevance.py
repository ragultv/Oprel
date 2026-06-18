import json
import re
from typing import Optional
from pydantic import BaseModel, Field
from oprel.server.services.generation import generate_text, GenerateParams

class RelevanceDecision(BaseModel):
    decision: str = Field(..., description="'respond', 'interrupt', 'react', or 'silent'")
    emoji: Optional[str] = None
    reason: str

RELEVANCE_SYSTEM_PROMPT = """You are {agent_name}, a participant in a group chat with other AI agents and a human user.
Your role: {role_description}

You will see the conversation context and the latest message. Decide your participation level.

Rules:
- If the user explicitly @mentioned you by name, you MUST choose "respond".
- If you were not mentioned, only choose "respond" if you have something materially
  new, corrective, or specifically useful to add. Do not respond just to be agreeable
  or to restate what another agent already said.
- If another agent's message in this round contains a factual error, numeric error,
  or unsupported claim you can verify is wrong, choose "interrupt".
- If you have no new information but want to signal agreement or a quick reaction,
  choose "react" and pick exactly one emoji from: 👍 👎 🤔 🔥 ⚠️ 💡
- Otherwise choose "silent".

Respond with ONLY this JSON, nothing else:
{{"decision": "respond" | "interrupt" | "react" | "silent", "emoji": "<one emoji or null>", "reason": "<one short phrase>"}}
"""

def parse_relevance_response(llm_output: str, is_explicitly_mentioned: bool = False) -> RelevanceDecision:
    """Robustly parse the JSON output from the relevance gate LLM."""
    text = llm_output.strip()
    
    # Try direct parsing first
    try:
        data = json.loads(text)
        return RelevanceDecision(**data)
    except json.JSONDecodeError:
        pass
        
    # Try finding JSON block
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(1))
            return RelevanceDecision(**data)
        except Exception:
            pass
            
    # Try aggressive regex extraction
    obj_match = re.search(r'\{.*?\}', text, re.DOTALL)
    if obj_match:
        try:
            data = json.loads(obj_match.group(0))
            return RelevanceDecision(**data)
        except Exception:
            pass
            
    # Fallback heuristic if completely broken
    lower_text = text.lower()
    
    # Try finding the decision value with regex to handle missing quotes
    react_match = re.search(r'decision["\']?\s*:\s*["\']?react["\']?', lower_text)
    respond_match = re.search(r'decision["\']?\s*:\s*["\']?respond["\']?', lower_text)
    interrupt_match = re.search(r'decision["\']?\s*:\s*["\']?interrupt["\']?', lower_text)
    
    if react_match:
        return RelevanceDecision(decision="react", emoji="👍", reason="Fallback parsed as react")
    elif respond_match:
        return RelevanceDecision(decision="respond", emoji=None, reason="Fallback parsed as respond")
    elif interrupt_match:
        return RelevanceDecision(decision="interrupt", emoji=None, reason="Fallback parsed as interrupt")
    else:
        if is_explicitly_mentioned:
            # Safest default is respond so small models don't silently fail when tagged
            return RelevanceDecision(decision="respond", emoji=None, reason="Failed to parse output but was mentioned")
        else:
            # Default to silent if not mentioned and failed to parse, to avoid emoji spam
            return RelevanceDecision(decision="silent", emoji=None, reason="Failed to parse output, defaulting to silent")

async def gather_relevance_decisions(group_id: str, round_id: str, member: dict, context_prompt: str, scope: str = "all") -> RelevanceDecision:
    """
    Mock integration for the relevance gate.
    In the real implementation, this will:
    1. Look up the cheapest flash model for member.provider
    2. Build the prompt with RELEVANCE_SYSTEM_PROMPT
    3. Call the provider API
    4. Call parse_relevance_response
    """
    import asyncio
    import re
    # Use actual generation pipeline
    
    agent_name = member.get("display_name", "AI")
    
    # Mention logic
    has_any_mention = "@" in context_prompt
    is_mentioned = False
    if has_any_mention:
        # Check if this specific agent is mentioned
        is_mentioned = f"@{agent_name.lower()}" in context_prompt.lower()
        if not is_mentioned:
            # Another agent was mentioned, but not this one. Force silent.
            return RelevanceDecision(decision="silent", reason="Not mentioned")

    
    model_id = member["model_id"]
    if member.get("kind") == "cloud" and member.get("provider_id"):
        p_id = member["provider_id"]
        if not model_id.startswith(f"{p_id}::"):
            model_id = f"{p_id}::{model_id}"
            
    system_prompt = RELEVANCE_SYSTEM_PROMPT.format(
        agent_name=agent_name,
        role_description=member.get("role_description", "Participant")
    )
    
    params = GenerateParams(
        model_id=model_id,
        prompt=f"Scope: {scope}\nContext Prompt: {context_prompt}\nEvaluate your relevance now.",
        max_tokens=100,
        temperature=0.0,
        top_p=1.0,
        top_k=40,
        repeat_penalty=1.1,
        stream=False,
        images=None,
        conversation_id=None,
        system_prompt=system_prompt,
        reset_conversation=False,
        thinking=False,
        rag=False
    )
    
    try:
        res = await generate_text(params)
        return parse_relevance_response(res.text, is_explicitly_mentioned=is_mentioned)
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Relevance gate failed for {member.get('display_name')}: {e}")
        # Fallback heuristic
        if scope == "interrupt":
            return RelevanceDecision(decision="silent", reason="Fallback to silent on error")
        return RelevanceDecision(decision="respond", reason="Fallback to respond on error")
