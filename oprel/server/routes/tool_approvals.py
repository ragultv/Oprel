from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from oprel.server.security.local_auth import require_local_token
from oprel.server.tools import store
from oprel.server.tools.loop import ApprovalError, resume_after_approval
from oprel.server.tools.responses import json_response, prime, streaming_response

router = APIRouter(prefix="/tool-approvals", dependencies=[Depends(require_local_token)])


class ApprovalDecision(BaseModel):
    decision: Literal["approve", "reject"]
    # Optional user-edited arguments (validated against the tool schema before running)
    arguments: dict[str, Any] | None = None
    reason: str | None = None
    stream: bool = True


@router.get("")
async def list_pending(conversation_id: str | None = None):
    return store.list_pending_approvals(conversation_id)


@router.post("/{approval_id}")
async def decide(approval_id: str, body: ApprovalDecision):
    """Approve or reject a pending tool call, then continue the agent loop (streamed like a chat reply)."""
    approval = store.get_approval(approval_id)
    model = (approval or {}).get("loop_state", {}).get("model", "")
    gen = resume_after_approval(
        approval_id,
        approve=body.decision == "approve",
        edited_arguments=body.arguments,
        reason=body.reason,
    )
    try:
        first, gen = await prime(gen)
    except ApprovalError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc))
    conv_id = (approval or {}).get("conversation_id")
    if body.stream:
        return streaming_response(first, gen, model=model, conversation_id=conv_id)
    return await json_response(first, gen, model=model)
