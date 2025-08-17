from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from src.database.connection import db_session
from src.database.models import DefectCatalog

router = APIRouter()

@router.get("/defects/list")
async def list_defects(session: AsyncSession = Depends(db_session)):
    res = await session.execute(select(DefectCatalog))
    rows = res.scalars().all()
    return [
        {
            'defect_id': r.defect_id,
            'name': r.name,
            'description': r.description,
            'severity_levels': (r.severity_scale.split(',') if r.severity_scale else [])
        } for r in rows
    ]
