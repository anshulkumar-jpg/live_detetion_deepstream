"""Analytics read API (track-level summaries from PostgreSQL; full event search is in Elasticsearch/Kibana)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import Principal, get_session, viewer
from app.db.models import AnalyticsEvent, Camera, DetectionEvent, ReidAssociation

router = APIRouter(tags=["analytics"])


def _site_cams(who: Principal):
    """Sub-query of the camera ids a site-restricted user may see (None for users who see every site)."""
    return select(Camera.id).where(Camera.site.in_(who.sites)) if who.sites else None


def _since(hours: int) -> datetime:
    return datetime.now(timezone.utc) - timedelta(hours=hours)


@router.get("/events")
async def list_events(db: AsyncSession = Depends(get_session), who: Principal = Depends(viewer),
                      camera_id: str | None = None, class_name: str | None = None, model_id: str | None = None,
                      since: datetime | None = None, until: datetime | None = None,
                      limit: int = Query(100, ge=1, le=1000), offset: int = Query(0, ge=0)):
    conds = []
    if (sc := _site_cams(who)) is not None:
        conds.append(DetectionEvent.camera_id.in_(sc))
    if camera_id:
        conds.append(DetectionEvent.camera_id == camera_id)
    if class_name:
        conds.append(DetectionEvent.class_name == class_name)
    if model_id:
        conds.append(DetectionEvent.model_id == model_id)
    if since:
        conds.append(DetectionEvent.last_seen >= since)
    if until:
        conds.append(DetectionEvent.last_seen <= until)
    where = and_(*conds) if conds else True
    total = (await db.execute(select(func.count()).select_from(DetectionEvent).where(where))).scalar()
    rows = (await db.execute(select(DetectionEvent).where(where).order_by(DetectionEvent.last_seen.desc())
                             .limit(limit).offset(offset))).scalars()
    return {"total": total, "limit": limit, "offset": offset, "items": [{
        "event_id": r.event_id, "camera_id": r.camera_id, "worker_id": r.worker_id, "model_id": r.model_id,
        "object_id": r.object_id, "class_id": r.class_id, "class_name": r.class_name,
        "max_confidence": r.max_confidence, "bbox": r.last_bbox, "samples": r.sample_count,
        "global_id": r.global_id, "first_seen": r.first_seen, "last_seen": r.last_seen} for r in rows]}


@router.get("/analytics/summary")
async def summary(db: AsyncSession = Depends(get_session), who: Principal = Depends(viewer),
                  hours: int = Query(24, ge=1, le=24 * 30)):
    since = _since(hours)
    sc = _site_cams(who)
    scope = DetectionEvent.camera_id.in_(sc) if sc is not None else True
    by_class = (await db.execute(select(DetectionEvent.class_name, func.count()).where(
        DetectionEvent.last_seen >= since, scope).group_by(DetectionEvent.class_name))).all()
    by_cam = (await db.execute(select(DetectionEvent.camera_id, func.count()).where(
        DetectionEvent.last_seen >= since, scope).group_by(DetectionEvent.camera_id)
        .order_by(func.count().desc()).limit(20))).all()
    by_status = (await db.execute(select(Camera.status, func.count()).where(Camera.site.in_(who.sites) if who.sites else True)
                                  .group_by(Camera.status))).all()
    identities = (await db.execute(select(func.count(func.distinct(ReidAssociation.global_id))).where(
        ReidAssociation.last_seen >= since))).scalar()
    multi_cam = (await db.execute(select(func.count()).select_from(
        select(ReidAssociation.global_id).where(ReidAssociation.last_seen >= since)
        .group_by(ReidAssociation.global_id).having(func.count(func.distinct(ReidAssociation.camera_id)) > 1)
        .subquery()))).scalar()
    behaviors = (await db.execute(select(AnalyticsEvent.event_type, func.count()).where(
        AnalyticsEvent.ts >= since).group_by(AnalyticsEvent.event_type))).all()
    if who.sites:  # identities and behaviours span cameras of every site: not shown to a site-restricted user
        identities = multi_cam = None
        behaviors = []
    return {"window_hours": hours,
            "tracks_by_class": {k or "unknown": v for k, v in by_class},
            "top_cameras_by_tracks": [{"camera_id": c, "tracks": n} for c, n in by_cam],
            "cameras_by_status": {(s.value if hasattr(s, "value") else s): n for s, n in by_status},
            "reid": {"global_identities": identities, "identities_seen_on_multiple_cameras": multi_cam},
            "analytics_events": dict(behaviors)}


@router.get("/analytics/cameras/{camera_id}")
async def camera_analytics(camera_id: str, db: AsyncSession = Depends(get_session), who: Principal = Depends(viewer),
                           hours: int = Query(24, ge=1, le=24 * 30)):
    cam = await db.get(Camera, camera_id)
    if cam is None or not who.can_see_site(cam.site):
        raise HTTPException(404, "camera not found")
    since = _since(hours)
    base = and_(DetectionEvent.camera_id == camera_id, DetectionEvent.last_seen >= since)
    by_class = (await db.execute(select(DetectionEvent.class_name, func.count(), func.avg(DetectionEvent.max_confidence))
                                 .where(base).group_by(DetectionEvent.class_name))).all()
    if db.bind.dialect.name == "postgresql":
        bucket = func.date_trunc("hour", DetectionEvent.first_seen)
    else:
        bucket = func.strftime("%Y-%m-%dT%H:00:00", DetectionEvent.first_seen)
    hourly = (await db.execute(select(bucket, func.count()).where(base).group_by(bucket).order_by(bucket))).all()
    reid = (await db.execute(select(func.count()).select_from(ReidAssociation).where(and_(
        ReidAssociation.camera_id == camera_id, ReidAssociation.matched_camera_id.is_not(None),
        ReidAssociation.last_seen >= since)))).scalar()
    return {"camera_id": camera_id, "window_hours": hours,
            "classes": [{"class_name": c or "unknown", "tracks": n, "avg_max_confidence": float(a or 0)} for c, n, a in by_class],
            "tracks_per_hour": [{"hour": str(h), "tracks": n} for h, n in hourly],
            "cross_camera_matches": reid}
