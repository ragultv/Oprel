from typing import Optional, List
from pydantic import BaseModel, Field

class GroupMemberCreate(BaseModel):
    kind: str = Field(..., description="'cloud' or 'local'")
    provider_id: Optional[str] = None
    model_id: str
    display_name: str
    role_description: Optional[str] = None
    is_moderator: bool = False
    priority_order: int = 0

class GroupCreate(BaseModel):
    name: str
    max_interrupt_rounds: int = 3
    max_replies_per_agent: Optional[int] = None
    members: List[GroupMemberCreate] = []

class GroupUpdate(BaseModel):
    name: Optional[str] = None
    max_interrupt_rounds: Optional[int] = None
    moderator_member_id: Optional[str] = None

class GroupResponse(BaseModel):
    id: str
    name: str
    created_at: str
    moderator_member_id: Optional[str]
    max_interrupt_rounds: int
    max_replies_per_agent: Optional[int]
    members: List[dict] = []
