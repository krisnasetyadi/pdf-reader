# router/skills.py
"""
Skills CRUD (MS-251).

A skill is an uploaded instruction file (.md): the frontmatter becomes name /
slash_command / description, the body becomes `instruction`. This module only
stores and scopes them — running a skill is MS-252 and lives elsewhere.

Two visibility rules, both enforced here and not only in the UI:
  - scope "team"     — admin uploads, every member that admin created can use it
  - scope "personal" — anyone uploads, only the uploader can use it
A non-admin cannot create a team skill, and nobody can edit or delete a skill
they do not own.
"""
from fastapi import APIRouter, HTTPException, Depends
from typing import List
import asyncio
import uuid

from models import SkillCreate, SkillUpdate, SkillResponse
import storage as supabase_storage
from router.auth import get_current_user, UserRecord

router = APIRouter()


@router.post("/skills", response_model=SkillResponse, status_code=201)
async def create_skill(body: SkillCreate, user: UserRecord = Depends(get_current_user)):
    """Upload a skill. `scope="team"` is admin-only — checked server-side, since
    hiding the toggle in the UI would still leave the endpoint open."""
    if body.scope == "team" and user.role != "admin":
        raise HTTPException(
            status_code=403,
            detail="Only an admin can share a skill with the whole team",
        )

    skill_id = str(uuid.uuid4())
    ok = await asyncio.to_thread(
        supabase_storage.create_skill,
        skill_id=skill_id,
        name=body.name,
        slash_command=body.slash_command,
        instruction=body.instruction,
        owner_id=user.user_id,
        description=body.description,
        scope=body.scope,
    )
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to create skill")

    row = await asyncio.to_thread(
        supabase_storage.get_skill_for_user, skill_id, user.user_id, user.role == "admin"
    )
    if not row:
        raise HTTPException(status_code=500, detail="Skill created but could not be read back")
    return SkillResponse(**row)


@router.get("/skills", response_model=List[SkillResponse])
async def list_skills(user: UserRecord = Depends(get_current_user)):
    """Skills this account may use: its own, plus its admin's team skills."""
    rows = await asyncio.to_thread(
        supabase_storage.list_skills_for_user, user.user_id, user.role == "admin"
    )
    return [SkillResponse(**row) for row in rows]


@router.get("/skills/{skill_id}", response_model=SkillResponse)
async def get_skill(skill_id: str, user: UserRecord = Depends(get_current_user)):
    row = await asyncio.to_thread(
        supabase_storage.get_skill_for_user, skill_id, user.user_id, user.role == "admin"
    )
    if not row:
        raise HTTPException(status_code=404, detail="Skill not found")
    return SkillResponse(**row)


# PUT rather than PATCH even though the body is partial: that is the shape
# already used for partial updates in this codebase (router/sessions.py's
# rename_session) and the one the frontend RequestHandler speaks.
@router.put("/skills/{skill_id}", response_model=SkillResponse)
async def update_skill(
    skill_id: str,
    body: SkillUpdate,
    user: UserRecord = Depends(get_current_user),
):
    """Owner-only edit. A member can see an admin's team skill but not change it."""
    row = await asyncio.to_thread(
        supabase_storage.get_skill_for_user, skill_id, user.user_id, user.role == "admin"
    )
    if not row:
        raise HTTPException(status_code=404, detail="Skill not found")
    if row["owner_id"] != user.user_id:
        raise HTTPException(status_code=403, detail="Not allowed to edit this skill")
    if body.scope == "team" and user.role != "admin":
        raise HTTPException(
            status_code=403,
            detail="Only an admin can share a skill with the whole team",
        )

    fields = body.model_dump(exclude_none=True)
    if not fields:
        return SkillResponse(**row)

    updated = await asyncio.to_thread(
        supabase_storage.update_skill, skill_id, user.user_id, fields
    )
    if not updated:
        raise HTTPException(status_code=500, detail="Failed to update skill")
    return SkillResponse(**updated)


@router.delete("/skills/{skill_id}")
async def delete_skill(skill_id: str, user: UserRecord = Depends(get_current_user)):
    """Owner-only hard delete, matching how collections and runs are removed."""
    row = await asyncio.to_thread(
        supabase_storage.get_skill_for_user, skill_id, user.user_id, user.role == "admin"
    )
    if not row:
        raise HTTPException(status_code=404, detail="Skill not found")
    if row["owner_id"] != user.user_id:
        raise HTTPException(status_code=403, detail="Not allowed to delete this skill")

    ok = await asyncio.to_thread(supabase_storage.delete_skill, skill_id, user.user_id)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to delete skill")
    return {"skill_id": skill_id, "deleted": True}
