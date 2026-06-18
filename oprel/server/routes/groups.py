from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
import uuid

from oprel.server.schemas.groups import GroupCreate, GroupUpdate, GroupMemberCreate, GroupResponse
from oprel.server import db

router = APIRouter()

@router.post("/groups", response_model=GroupResponse)
async def create_group_route(body: GroupCreate):
    # Check constraints: max 1 local member, exactly 1 moderator
    local_count = sum(1 for m in body.members if m.kind == 'local')
    if local_count > 1:
        raise HTTPException(status_code=400, detail="Only one local member is allowed per group")
        
    mod_count = sum(1 for m in body.members if m.is_moderator)
    if mod_count != 1 and body.members:
        raise HTTPException(status_code=400, detail="Exactly one member must be the moderator")
        
    group = db.create_group(body.name, body.max_interrupt_rounds, body.max_replies_per_agent)
    
    for idx, member in enumerate(body.members):
        db.add_group_member(
            group_id=group["id"],
            kind=member.kind,
            model_id=member.model_id,
            display_name=member.display_name,
            provider_id=member.provider_id,
            role_description=member.role_description,
            is_moderator=member.is_moderator,
            priority_order=member.priority_order or idx
        )
        
    return db.get_group(group["id"])

@router.get("/groups")
async def list_groups_route():
    return db.list_groups()

@router.get("/groups/{group_id}", response_model=GroupResponse)
async def get_group_route(group_id: str):
    group = db.get_group(group_id)
    if not group:
        raise HTTPException(status_code=404, detail="Group not found")
    return group

@router.patch("/groups/{group_id}", response_model=GroupResponse)
async def update_group_route(group_id: str, body: GroupUpdate):
    group = db.get_group(group_id)
    if not group:
        raise HTTPException(status_code=404, detail="Group not found")
        
    # Validation for moderator
    if body.moderator_member_id:
        is_valid_mod = any(m["id"] == body.moderator_member_id for m in group["members"])
        if not is_valid_mod:
            raise HTTPException(status_code=400, detail="Provided moderator_member_id does not belong to the group")
            
        # Update is_moderator flag on group_members directly to maintain consistency
        conn = db.get_db()
        cursor = conn.cursor()
        cursor.execute("UPDATE group_members SET is_moderator = 0 WHERE group_id = ?", (group_id,))
        cursor.execute("UPDATE group_members SET is_moderator = 1 WHERE id = ?", (body.moderator_member_id,))
        conn.commit()
        conn.close()
            
    updated_group = db.update_group(group_id, body.dict(exclude_unset=True))
    return db.get_group(group_id)

@router.delete("/groups/{group_id}")
async def delete_group_route(group_id: str):
    db.delete_group(group_id)
    return {"success": True}

@router.post("/groups/{group_id}/members")
async def add_member_route(group_id: str, body: GroupMemberCreate):
    group = db.get_group(group_id)
    if not group:
        raise HTTPException(status_code=404, detail="Group not found")
        
    try:
        member = db.add_group_member(
            group_id=group_id,
            kind=body.kind,
            model_id=body.model_id,
            display_name=body.display_name,
            provider_id=body.provider_id,
            role_description=body.role_description,
            is_moderator=body.is_moderator,
            priority_order=body.priority_order
        )
        return member
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.delete("/groups/{group_id}/members/{member_id}")
async def remove_member_route(group_id: str, member_id: str):
    db.remove_group_member(group_id, member_id)
    return {"success": True}

@router.get("/groups/{group_id}/messages")
async def get_messages_route(group_id: str, round_id: Optional[str] = None):
    return db.get_group_messages(group_id, round_id=round_id)

class PostMessageRequest(BaseModel):
    content: str

@router.post("/groups/{group_id}/messages")
async def post_message_route(group_id: str, body: PostMessageRequest, background_tasks: BackgroundTasks):
    group = db.get_group(group_id)
    if not group:
        raise HTTPException(status_code=404, detail="Group not found")
        
    # Generate the ID for the user's message
    user_msg_id = f"gmsg_{uuid.uuid4().hex[:12]}"
    
    # Create the round using this message ID
    round_data = db.create_group_round(group_id, user_msg_id)
    
    # Calculate sequence number
    seq_num = db.get_max_sequence_number(group_id, round_data["id"]) + 1
    
    # Insert the user message
    msg = db.add_group_message(
        group_id=group_id,
        round_id=round_data["id"],
        sender_type="user",
        member_id=None,
        content=body.content,
        message_type="trigger",
        sequence_number=seq_num,
        msg_id=user_msg_id
    )
    
    # Launch orchestrator in background
    from oprel.core.groups.orchestrator import run_round
    background_tasks.add_task(run_round, group_id, round_data["id"])
    
    return msg
