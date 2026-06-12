from __future__ import annotations

from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from oprel.server import db

router = APIRouter()


class SkillSchema(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    system_prompt: str
    category: str
    icon: str
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    output_schema: Optional[str] = None
    is_premium: Optional[bool] = False
    enabled: Optional[bool] = True


@router.get("/skills")
async def list_skills():
    """Retrieve all skills from the database."""
    try:
        return db.list_skills()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to fetch skills: {exc}")


@router.post("/skills")
async def save_skill(skill: SkillSchema):
    """Save or update a skill configuration."""
    try:
        data = skill.dict()
        return db.upsert_skill(data)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save skill: {exc}")


@router.delete("/skills/{skill_id}")
async def delete_skill(skill_id: str):
    """Delete a skill configuration by ID."""
    try:
        # Check if skill exists
        existing = db.get_skill(skill_id)
        if not existing:
            raise HTTPException(status_code=404, detail="Skill not found")
        db.delete_skill(skill_id)
        return {"success": True, "message": f"Skill '{skill_id}' deleted successfully"}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to delete skill: {exc}")
