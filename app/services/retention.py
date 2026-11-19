"""Retention policy for personal / behavioural data.

Detection events, Re-ID associations (global identities that link one person across cameras) and behaviour events are
personal data under GDPR / India's DPDP Act. They are deleted once older than the configured number of days.
Milvus embeddings are expired separately (EMBEDDING_RETENTION_DAYS, see analytics/mtmc.py). A value of 0 keeps data forever
(not recommended); audit-log retention is separate because it is a security record.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.db.models import AnalyticsEvent, AuditLog, DetectionEvent, ReidAssociation

log = logging.getLogger("retention")


async def purge_expired(db: AsyncSession, s: Settings, now: datetime | None = None) -> dict[str, int]:
    now = now or datetime.now(timezone.utc)
    plan = [
        ("detection_events", DetectionEvent, DetectionEvent.last_seen, s.detection_event_retention_days),
        ("reid_associations", ReidAssociation, ReidAssociation.last_seen, s.reid_retention_days),
        ("analytics_events", AnalyticsEvent, AnalyticsEvent.ts, s.analytics_event_retention_days),
        ("audit_log", AuditLog, AuditLog.ts, s.audit_retention_days),
    ]
    out: dict[str, int] = {}
    for name, model, col, days in plan:
        if days <= 0:
            continue
        res = await db.execute(delete(model).where(col < now - timedelta(days=days)))
        out[name] = res.rowcount or 0
    await db.commit()
    if any(out.values()):
        log.info("retention removed rows: %s", out)
    return out
