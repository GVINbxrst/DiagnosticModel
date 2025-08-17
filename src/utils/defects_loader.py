"""Импорт дефектов из config/defects.yaml в DefectCatalog (однократно)."""
from __future__ import annotations

from pathlib import Path
from typing import List
import yaml
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models import DefectCatalog
from src.utils.logger import get_logger

logger = get_logger(__name__)

async def load_defects_yaml(session: AsyncSession, yaml_path: Path) -> int:
    if not yaml_path.exists():  # pragma: no cover
        logger.warning(f"defects.yaml not found: {yaml_path}")
        return 0
    # пропускаем если уже есть данные
    existing = await session.execute(select(DefectCatalog.defect_id))
    if existing.first():
        return 0
    with yaml_path.open('r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    defects: List[dict] = data.get('defects', [])
    rows = []
    for d in defects:
        sev = d.get('severity_levels') or []
        rows.append(DefectCatalog(
            defect_id=d['defect_id'],
            name=d.get('name',''),
            description=d.get('description'),
            severity_scale=','.join(sev)
        ))
    if not rows:
        return 0
    session.add_all(rows)
    await session.commit()
    logger.info(f"Imported {len(rows)} defects")
    return len(rows)
