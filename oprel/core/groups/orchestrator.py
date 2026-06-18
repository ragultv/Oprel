import asyncio
import logging
from enum import Enum
from oprel.server import db
from oprel.core.groups.relevance import gather_relevance_decisions
from oprel.core.groups.lanes import apply_reaction, execute_lane_dispatch
from oprel.core.groups.sequencer import sequence_and_commit_results
from oprel.core.groups.memory import compact_memory_for_all_members
from oprel.core.groups.moderator import run_moderator_engine
from oprel.server.routes.groups_ws import manager

logger = logging.getLogger(__name__)

class RoundState(Enum):
    RELEVANCE = "relevance"
    GENERATION = "generation"
    INTERRUPT = "interrupt"
    MODERATION = "moderation"
    DONE = "done"

async def run_round(group_id: str, round_id: str):
    """
    The strict finite state machine controlling a discussion round.
    This guarantees no infinite AI-to-AI spam loops.
    """
    group = db.get_group(group_id)
    if not group:
        logger.error(f"Group {group_id} not found.")
        return
        
    round_data = db.get_group_round(round_id)
    if not round_data:
        logger.error(f"Round {round_id} not found.")
        return
        
    max_interrupts = group.get("max_interrupt_rounds", 3)
    current_state = round_data["state"]
    
    # Context variable to pass responders from RELEVANCE to GENERATION
    # if executing in a single process flow.
    responders_queue = []
    
    while current_state != RoundState.DONE.value:
        logger.info(f"Group {group_id} | Round {round_id} | State: {current_state}")
        await manager.broadcast_to_group(group_id, {"type": "state_change", "state": current_state})
        
        if current_state == RoundState.RELEVANCE.value:
            # 1. Relevance Phase
            tasks = []
            from oprel.core.groups.memory import build_agent_context
            for member in group["members"]:
                if member.get("is_moderator") == 1:
                    continue
                
                # Fetch actual context including the user's new message
                context_prompt = build_agent_context(group_id, member["id"], round_id)
                prompt = f"Evaluate relevance based on this context:\n\n{context_prompt}" 
                tasks.append(gather_relevance_decisions(group_id, round_id, member, prompt))
            
            decisions = await asyncio.gather(*tasks, return_exceptions=True)
            
            responders_queue.clear()
            # Filter out moderators so we zip correctly with decisions
            active_members = [m for m in group["members"] if m.get("is_moderator") != 1]
            
            for member, decision in zip(active_members, decisions):
                if isinstance(decision, Exception):
                    logger.warning(f"Relevance gate failed for {member['display_name']}: {decision}")
                    continue
                    
                if decision.decision == "react":
                    message_id = round_data["user_message_id"]
                    apply_reaction(message_id, member, decision)
                    logger.info(f"Agent {member['display_name']} reacted with {decision.emoji}")
                    await manager.broadcast_to_group(group_id, {
                        "type": "reaction_added", 
                        "message_id": message_id,
                        "member_id": member["id"], 
                        "emoji": decision.emoji
                    })
                elif decision.decision in ["respond", "interrupt"]:
                    responders_queue.append((member, decision))
                    logger.info(f"Agent {member['display_name']} opted to {decision.decision}")
                    await manager.broadcast_to_group(group_id, {
                        "type": "agent_action", 
                        "member_id": member["id"], 
                        "action": decision.decision
                    })
                    
            if not responders_queue:
                # No agents want to generate text, transition immediately to DONE
                current_state = RoundState.DONE.value
                db.update_group_round_state(round_id, current_state)
                break
                
            current_state = RoundState.GENERATION.value
            db.update_group_round_state(round_id, current_state)
            
        elif current_state == RoundState.GENERATION.value:
            # 2. Generation Phase
            if not responders_queue:
                # Fallback if state was resumed without responders in memory
                current_state = RoundState.INTERRUPT.value
                db.update_group_round_state(round_id, current_state)
                continue
                
            lane_results = await execute_lane_dispatch(group_id, round_id, responders_queue)
            
            # Sequence and commit deterministically
            sequence_and_commit_results(group_id, round_id, lane_results)
            
            # Clear queue and move to interrupt
            responders_queue.clear()
            current_state = RoundState.INTERRUPT.value
            db.update_group_round_state(round_id, current_state)
            
        elif current_state == RoundState.INTERRUPT.value:
            # 3. Interrupt Phase
            if round_data["interrupt_count"] >= max_interrupts:
                logger.info(f"Max interrupt rounds ({max_interrupts}) reached.")
                current_state = RoundState.MODERATION.value
                db.update_group_round_state(round_id, current_state)
                continue
                
            tasks = []
            for member in group["members"]:
                if member.get("is_moderator") == 1:
                    continue
                context_prompt = build_agent_context(group_id, member["id"], round_id)
                prompt = f"Check for interruptions based on this context:\n\n{context_prompt}"
                tasks.append(gather_relevance_decisions(group_id, round_id, member, prompt, scope="interrupt"))
                
            decisions = await asyncio.gather(*tasks, return_exceptions=True)
            
            interrupt_responders = []
            active_members = [m for m in group["members"] if m.get("is_moderator") != 1]
            for member, decision in zip(active_members, decisions):
                if isinstance(decision, Exception):
                    continue
                if decision.decision in ["interrupt", "respond"]:
                    interrupt_responders.append((member, decision))
                    
            if not interrupt_responders:
                current_state = RoundState.MODERATION.value
                db.update_group_round_state(round_id, current_state)
            else:
                lane_results = await execute_lane_dispatch(group_id, round_id, interrupt_responders)
                sequence_and_commit_results(group_id, round_id, lane_results)
                
                db.update_group_round_state(round_id, current_state, increment_interrupt=True)
                round_data["interrupt_count"] += 1
                
        elif current_state == RoundState.MODERATION.value:
            # 4. Moderation Phase
            await run_moderator_engine(group_id, round_id)
            
            # Notify clients of final moderation complete
            await manager.broadcast_to_group(group_id, {"type": "moderation_complete"})
            
            current_state = RoundState.DONE.value
            db.update_group_round_state(round_id, current_state)
            
        elif current_state == RoundState.DONE.value:
            break
            
    # 5. Done Phase - Compact Memory
    logger.info(f"Group {group_id} | Round {round_id} completed. Compacting memory.")
    await manager.broadcast_to_group(group_id, {"type": "state_change", "state": RoundState.DONE.value})
    await compact_memory_for_all_members(group_id, round_id)
