import json
import re
from typing import Optional
from pydantic import BaseModel, Field
from oprel.server import db
from oprel.core.groups.memory import format_round_messages
from oprel.server.services.generation import generate_text, GenerateParams

class ModeratorDecision(BaseModel):
    consensus_reached: bool = Field(..., description="Whether a substantive consensus was reached.")
    final_summary: str = Field(..., description="The synthesized final answer if consensus was reached, otherwise a summary of the disagreement.")
    reasoning: str = Field(..., description="Explanation of why consensus was or was not reached.")

MODERATOR_SYSTEM_PROMPT = """You are the Moderator of this AI group discussion.
Your job is to read the transcript of the current round and determine if a substantive consensus has been reached to answer the user's implicit or explicit request.

Rules for Consensus:
1. Do not rely on simple emoji counting. Read the actual claims.
2. If agents hallucinated or agreed on something factually incorrect (hallucination amplification), and it was NOT corrected during the interrupt phase, you must flag consensus_reached as false if you detect it.
3. If there are unresolved disputes between agents, consensus_reached is false.
4. If a clear, correct, and unified answer has emerged, consensus_reached is true.

Respond with ONLY this JSON, nothing else:
{{"consensus_reached": true/false, "final_summary": "<synthesis of the answer>", "reasoning": "<why>"}}
"""

def parse_moderator_response(llm_output: str) -> ModeratorDecision:
    """Robustly parse the JSON output from the moderator engine LLM."""
    text = llm_output.strip()
    
    try:
        data = json.loads(text)
        return ModeratorDecision(**data)
    except json.JSONDecodeError:
        pass
        
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(1))
            return ModeratorDecision(**data)
        except Exception:
            pass
            
    obj_match = re.search(r'\{.*?\}', text, re.DOTALL)
    if obj_match:
        try:
            data = json.loads(obj_match.group(0))
            return ModeratorDecision(**data)
        except Exception:
            pass
            
    # Fallback heuristic
    lower_text = text.lower()
    consensus = False
    if "consensus_reached\": true" in lower_text or "consensus_reached\":true" in lower_text:
        consensus = True
        
    return ModeratorDecision(
        consensus_reached=consensus,
        final_summary="Failed to parse detailed summary.",
        reasoning="JSON parsing failed."
    )

async def run_moderator_engine(group_id: str, round_id: str) -> None:
    """
    Executes the Moderation Phase.
    Finds the designated moderator agent, sends them the full round transcript,
    and commits their synthesized final_answer to the database.
    """
    group = db.get_group(group_id)
    if not group or not group.get("members"):
        return
        
    # Find moderator
    moderator = next((m for m in group["members"] if m["is_moderator"]), None)
    if not moderator:
        # Fallback to the first agent if no explicit moderator
        moderator = group["members"][0]
        
    raw_messages = db.get_group_messages(group_id, round_id=round_id)
    transcript = format_round_messages(raw_messages)
    
    prompt = f"Round Transcript:\n{transcript}"
    
    # Try getting full model id
    model_id = moderator["model_id"]
    if moderator.get("kind") == "cloud" and moderator.get("provider_id"):
        p_id = moderator["provider_id"]
        if not model_id.startswith(f"{p_id}::"):
            model_id = f"{p_id}::{model_id}"
            
    params = GenerateParams(
        model_id=model_id,
        prompt=prompt,
        max_tokens=512,
        temperature=0.2,
        top_p=0.9,
        top_k=40,
        repeat_penalty=1.1,
        stream=False,
        images=None,
        conversation_id=None,
        system_prompt=MODERATOR_SYSTEM_PROMPT,
        reset_conversation=False,
        thinking=False,
        rag=False
    )
    
    try:
        res = await generate_text(params)
        decision = parse_moderator_response(res.text)
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Moderator failed: {e}")
        # Fallback if generation or parsing completely fails
        decision = ModeratorDecision(
            consensus_reached=True,
            final_summary=f"Consensus reached (fallback). Generation failed: {str(e)[:100]}",
            reasoning="Fallback due to generation error."
        )
    
    # Get current max sequence number to append at the end
    seq_num = db.get_max_sequence_number(group_id, round_id) + 1
    
    # Commit the moderator's final answer to the database
    db.add_group_message(
        group_id=group_id,
        round_id=round_id,
        sender_type="system", # We mark moderator synthesis as system, or "agent" with member_id
        member_id=moderator["id"],
        content=decision.final_summary,
        message_type="final_answer",
        sequence_number=seq_num
    )
    
    # Optional: We could store the consensus_reached boolean somewhere, perhaps in group_rounds
