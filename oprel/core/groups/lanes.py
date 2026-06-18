import asyncio
from dataclasses import dataclass
from oprel.server import db
from oprel.core.groups.memory import build_agent_context
from oprel.core.groups.relevance import RelevanceDecision
from oprel.server.services.generation import generate_text, GenerateParams
from oprel.server.services.providers import provider_chat_proxy
from oprel.server.schemas.providers import ProviderChatRequest

MAX_CONCURRENT_CLOUD_CALLS = 5
cloud_semaphore = asyncio.Semaphore(MAX_CONCURRENT_CLOUD_CALLS)
local_lock = asyncio.Lock()

@dataclass
class LaneResult:
    member: dict
    decision: RelevanceDecision
    text: str

def _get_full_model_id(member: dict) -> str:
    if member.get("kind") == "cloud" and member.get("provider_id"):
        p_id = member["provider_id"]
        m_id = member["model_id"]
        if m_id.startswith(f"{p_id}::"):
            return m_id
        return f"{p_id}::{m_id}"
    return member["model_id"]

async def run_cloud_agent(member: dict, prompt: str, decision: RelevanceDecision, system_prompt: str) -> LaneResult:
    """Executes a cloud model generation within the bounded semaphore."""
    async with cloud_semaphore:
        p_id = member.get("provider_id")
        m_id = member.get("model_id")
        
        body = ProviderChatRequest(
            model=m_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1024,
            temperature=0.7,
            top_p=0.9,
            stream=False,
            conversation_id=None
        )
        try:
            res = await provider_chat_proxy(p_id, body)
            text = res.text
        except Exception as e:
            text = f"[Error generating response: {e}]"
            
        return LaneResult(
            member=member,
            decision=decision,
            text=text
        )
async def run_local_agent(member: dict, prompt: str, decision: RelevanceDecision, system_prompt: str) -> LaneResult:
    """Executes a local model generation holding the single-slot lock."""
    async with local_lock:
        full_model_id = _get_full_model_id(member)
        params = GenerateParams(
            model_id=full_model_id,
            prompt=prompt,
            max_tokens=1024,
            temperature=0.7,
            top_p=0.9,
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
            text = res.text
        except Exception as e:
            text = f"[Error generating response: {e}]"
            
        return LaneResult(
            member=member,
            decision=decision,
            text=text
        )

def apply_reaction(message_id: str, member: dict, decision: RelevanceDecision) -> None:
    """Fast-path: Skips generation lane entirely, writes reaction directly."""
    if not decision.emoji:
        return
    db.add_reaction(message_id, member["id"], decision.emoji)

async def execute_lane_dispatch(group_id: str, round_id: str, responders: list) -> list[LaneResult]:
    """
    Dispatches generation tasks safely into Cloud and Local lanes.
    `responders` is a list of tuples: (member_dict, RelevanceDecision)
    """
    tasks = []
    group = db.get_group(group_id)
    all_members = group["members"] if group else []

    for member, decision in responders:
        prompt = build_agent_context(group_id, member["id"], round_id)
        
        # Build tagging instructions
        other_agents = [m["display_name"] for m in all_members if m["id"] != member["id"] and m.get("is_moderator") != 1]
        tagging_instruction = ""
        if other_agents:
            tagging_instruction = f"\n\nYou can tag other agents to ask for their input by using @AgentName in your response. Available agents: {', '.join(other_agents)}"
            
        system_prompt = (member.get("role_description") or "You are an AI participant in a group discussion.") + tagging_instruction
        
        if member["kind"] == "cloud":
            tasks.append(run_cloud_agent(member, prompt, decision, system_prompt))
        else:
            tasks.append(run_local_agent(member, prompt, decision, system_prompt))
            
    # Gather all results. 
    # Return exceptions so one failed agent doesn't crash the whole orchestrator.
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filter out exceptions and log them
    successful_results = []
    for r in results:
        if isinstance(r, Exception):
            # TODO: Log exception properly
            print(f"Agent failed in lane dispatch: {r}")
        else:
            successful_results.append(r)
            
    return successful_results
