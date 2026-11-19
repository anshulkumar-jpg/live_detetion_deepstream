"""Camera management, model assignment and per-camera stream endpoints.

Start/stop only change *desired state*; the scheduler converges actual state.
That is why they return 202 Accepted and the current status.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy import and_, delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.api.deps import Principal, audit, get_session, get_streams, operator, viewer
from app.core.config import Settings, get_settings
from app.core.security import create_video_token, get_cipher, strip_credentials
from app.db.models import (
    Camera,
    CameraModelAssignment,
    CameraStatus,
    DesiredState,
    ExecutionStrategy,
    Model,
    WorkerCameraAssignment,
)
from app.schemas.api import (
    CameraBulkCreate,
    CameraCreate,
    CameraModelsRequest,
    CameraOut,
    CameraUpdate,
    ModelAssignmentOut,
    Page,
)
from app.services.model_router import RoutingError, validate_assignment_request

router = APIRouter(prefix="/cameras", tags=["cameras"])

# fields that require rebuilding the camera's pipeline when changed
_PIPELINE_FIELDS = {"rtsp_url", "username", "password", "tracking_enabled", "tracker_type",
                    "execution_strategy", "stream_mode", "width", "height", "fps", "codec_hint",
                    "detection_interval"}


def _raw_guard(mode, settings: Settings) -> None:
    from app.db.models import StreamMode
    if StreamMode(mode) in (StreamMode.RAW, StreamMode.BOTH) and not settings.allow_raw_streams:
        raise HTTPException(422, "raw camera streams are disabled because they put the camera password into the video "
                                 "server; use stream_mode 'annotated', or set ALLOW_RAW_STREAMS=true to allow raw")


async def _worker_map(db: AsyncSession, ids: list[str]) -> dict[str, str]:
    if not ids:
        return {}
    rows = await db.execute(select(WorkerCameraAssignment.camera_id, WorkerCameraAssignment.worker_id).where(
        and_(WorkerCameraAssignment.camera_id.in_(ids), WorkerCameraAssignment.released_at.is_(None))))
    return dict(rows.all())


def to_out(cam: Camera, worker_id: str | None = None) -> CameraOut:
    return CameraOut(
        id=cam.id, name=cam.name, location=cam.location, site=cam.site, tags=cam.tags or {}, width=cam.width, height=cam.height,
        fps=cam.fps, codec_hint=cam.codec_hint, tracking_enabled=cam.tracking_enabled, tracker_type=cam.tracker_type,
        detection_interval=cam.detection_interval or 0,
        execution_strategy=cam.execution_strategy, stream_mode=cam.stream_mode, rtsp_url=cam.rtsp_url,
        has_credentials=bool(cam.username_enc), desired_state=cam.desired_state, status=cam.status,
        status_reason=cam.status_reason, config_generation=cam.config_generation, worker_id=worker_id,
        models=[ModelAssignmentOut.model_validate(a) for a in cam.assignments],
        created_at=cam.created_at, updated_at=cam.updated_at,
    )


def _new_camera(body: CameraCreate, settings: Settings, who: Principal | None = None) -> Camera:
    _raw_guard(body.stream_mode, settings)
    if who is not None and not who.can_see_site(body.site):
        raise HTTPException(403, f"you may not create cameras in site '{body.site}'")
    clean, url_user, url_pass = strip_credentials(body.rtsp_url)
    cipher = get_cipher(settings)
    return Camera(
        id=body.id, name=body.name, rtsp_url=clean,
        username_enc=cipher.encrypt(body.username or url_user), password_enc=cipher.encrypt(body.password or url_pass),
        location=body.location, site=body.site, tags=body.tags, width=body.width, height=body.height, fps=body.fps,
        codec_hint=body.codec_hint, tracking_enabled=body.tracking_enabled, tracker_type=body.tracker_type,
        detection_interval=body.detection_interval, execution_strategy=body.execution_strategy, stream_mode=body.stream_mode,
        desired_state=DesiredState.STOPPED, status=CameraStatus.REGISTERED,
    )


def _site_ok(cam: Camera, who: Principal) -> None:
    """A user restricted to some sites cannot see cameras of other sites (404, not 403: do not reveal they exist)."""
    if not who.can_see_site(cam.site):
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"camera '{cam.id}' not found")


async def _get(db: AsyncSession, camera_id: str, who: Principal | None = None) -> Camera:
    cam = (await db.execute(select(Camera).options(selectinload(Camera.assignments)).where(Camera.id == camera_id)
                            .execution_options(populate_existing=True))).scalar_one_or_none()
    if cam is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"camera '{camera_id}' not found")
    if who is not None:
        _site_ok(cam, who)
    return cam


@router.post("", response_model=CameraOut, status_code=201)
async def create_camera(body: CameraCreate, request: Request, db: AsyncSession = Depends(get_session),
                        who: Principal = Depends(operator), settings: Settings = Depends(get_settings)):
    if await db.get(Camera, body.id) is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, f"camera '{body.id}' already exists")
    cam = _new_camera(body, settings, who)
    db.add(cam)
    audit(db, request, who, "camera.create", "camera", cam.id, {"name": cam.name})
    await db.commit()
    return to_out(await _get(db, cam.id))


@router.post("/bulk", status_code=201)
async def bulk_create(body: CameraBulkCreate, request: Request, db: AsyncSession = Depends(get_session),
                      who: Principal = Depends(operator), settings: Settings = Depends(get_settings)):
    ids = [c.id for c in body.cameras]
    if len(ids) != len(set(ids)):
        raise HTTPException(422, "duplicate camera ids in request")
    existing = set((await db.execute(select(Camera.id).where(Camera.id.in_(ids)))).scalars())
    if existing:
        raise HTTPException(status.HTTP_409_CONFLICT, f"already exist: {sorted(existing)[:20]}")
    for c in body.cameras:
        db.add(_new_camera(c, settings, who))
    audit(db, request, who, "camera.bulk_create", "camera", None, {"count": len(ids)})
    await db.commit()
    return {"created": len(ids)}


@router.get("", response_model=Page)
async def list_cameras(db: AsyncSession = Depends(get_session), who: Principal = Depends(viewer),
                       status_: CameraStatus | None = Query(None, alias="status"),
                       desired_state: DesiredState | None = None,
                       limit: int = Query(100, ge=1, le=1000), offset: int = Query(0, ge=0)):
    q = select(Camera)
    cq = select(func.count()).select_from(Camera)
    if who.sites:  # site-restricted user: only their sites
        q, cq = q.where(Camera.site.in_(who.sites)), cq.where(Camera.site.in_(who.sites))
    if status_ is not None:
        q, cq = q.where(Camera.status == status_), cq.where(Camera.status == status_)
    if desired_state is not None:
        q, cq = q.where(Camera.desired_state == desired_state), cq.where(Camera.desired_state == desired_state)
    total = (await db.execute(cq)).scalar()
    cams = list((await db.execute(q.options(selectinload(Camera.assignments)).order_by(Camera.id).limit(limit).offset(offset))).scalars())
    wm = await _worker_map(db, [c.id for c in cams])
    return Page(total=total, limit=limit, offset=offset, items=[to_out(c, wm.get(c.id)) for c in cams])


@router.get("/{camera_id}", response_model=CameraOut)
async def get_camera(camera_id: str, db: AsyncSession = Depends(get_session), who: Principal = Depends(viewer)):
    cam = await _get(db, camera_id, who)
    return to_out(cam, (await _worker_map(db, [cam.id])).get(cam.id))


@router.put("/{camera_id}", response_model=CameraOut)
async def update_camera(camera_id: str, body: CameraUpdate, request: Request, db: AsyncSession = Depends(get_session),
                        who: Principal = Depends(operator), settings: Settings = Depends(get_settings)):
    cam = await _get(db, camera_id, who)
    data = body.model_dump(exclude_unset=True)
    if data.get("stream_mode") is not None:
        _raw_guard(data["stream_mode"], settings)
    if data.get("site") is not None and not who.can_see_site(data["site"]):
        raise HTTPException(403, f"you may not move cameras into site '{data['site']}'")
    cipher = get_cipher(settings)
    if "rtsp_url" in data:
        clean, u, p = strip_credentials(data.pop("rtsp_url"))
        cam.rtsp_url = clean
        if u and "username" not in data:
            data["username"] = u
        if p and "password" not in data:
            data["password"] = p
    if "username" in data:
        cam.username_enc = cipher.encrypt(data.pop("username"))
    if "password" in data:
        cam.password_enc = cipher.encrypt(data.pop("password"))
    for k, v in data.items():
        setattr(cam, k, v)
    if set(body.model_dump(exclude_unset=True)) & _PIPELINE_FIELDS:
        cam.config_generation += 1
    audit(db, request, who, "camera.update", "camera", cam.id, {"fields": sorted(body.model_dump(exclude_unset=True))})
    await db.commit()
    return to_out(await _get(db, camera_id))


@router.delete("/{camera_id}", status_code=202)
async def delete_camera(camera_id: str, request: Request, db: AsyncSession = Depends(get_session),
                        who: Principal = Depends(operator)):
    cam = await _get(db, camera_id, who)
    active = (await db.execute(select(func.count()).select_from(WorkerCameraAssignment).where(and_(
        WorkerCameraAssignment.camera_id == cam.id, WorkerCameraAssignment.released_at.is_(None))))).scalar()
    if active:
        # stop first; caller retries delete once the camera reports STOPPED
        cam.desired_state = DesiredState.STOPPED
        audit(db, request, who, "camera.delete_requested", "camera", cam.id)
        await db.commit()
        raise HTTPException(status.HTTP_409_CONFLICT, "camera is running; stop requested - retry delete when status is STOPPED")
    await db.delete(cam)
    audit(db, request, who, "camera.delete", "camera", camera_id)
    await db.commit()
    return {"deleted": camera_id}


@router.post("/{camera_id}/start", response_model=CameraOut, status_code=202)
async def start_camera(camera_id: str, request: Request, db: AsyncSession = Depends(get_session),
                       who: Principal = Depends(operator)):
    cam = await _get(db, camera_id, who)
    if not any(a.enabled for a in cam.assignments):
        raise HTTPException(422, "assign at least one enabled model before starting (POST /cameras/{id}/models)")
    cam.desired_state = DesiredState.RUNNING
    if cam.status in (CameraStatus.REGISTERED, CameraStatus.STOPPED, CameraStatus.FAILED):
        cam.status, cam.status_reason = CameraStatus.SCHEDULED, "start requested"
        cam.rtsp_failures = 0
    audit(db, request, who, "camera.start", "camera", cam.id)
    await db.commit()
    return to_out(await _get(db, camera_id))


@router.post("/{camera_id}/stop", response_model=CameraOut, status_code=202)
async def stop_camera(camera_id: str, request: Request, db: AsyncSession = Depends(get_session),
                      who: Principal = Depends(operator)):
    cam = await _get(db, camera_id, who)
    cam.desired_state = DesiredState.STOPPED
    audit(db, request, who, "camera.stop", "camera", cam.id)
    await db.commit()
    return to_out(await _get(db, camera_id))


@router.post("/{camera_id}/models", response_model=CameraOut)
async def assign_models(camera_id: str, body: CameraModelsRequest, request: Request,
                        db: AsyncSession = Depends(get_session), who: Principal = Depends(operator)):
    """Replace the camera's model assignments. Only this camera's pipeline is rebuilt;
    other cameras (including the one sharing its worker) keep running."""
    cam = await _get(db, camera_id, who)
    strategy = body.execution_strategy or cam.execution_strategy
    if any(m.parent_model_id for m in body.models) and body.execution_strategy is None:
        strategy = ExecutionStrategy.CASCADE
    models = {m.id: m for m in (await db.execute(select(Model))).scalars()}
    try:
        validate_assignment_request([m.model_dump() for m in body.models], models, strategy)
    except RoutingError as e:
        raise HTTPException(422, str(e))
    for m in body.models:
        if m.enabled and models[m.model_id].active_version_id is None:
            raise HTTPException(422, f"model '{m.model_id}' has no active version")
        if m.enabled and not models[m.model_id].enabled:
            raise HTTPException(422, f"model '{m.model_id}' is disabled")
    wanted = [m for m in body.models if m.detect_class_ids]
    if wanted:  # the camera may only pick classes the model offers
        from app.core.classes import selectable_classes
        from app.db.models import ModelVersion
        vers = {v.id: v for v in (await db.execute(select(ModelVersion).where(
            ModelVersion.id.in_([models[m.model_id].active_version_id for m in wanted])))).scalars()}
        for m in wanted:
            v = vers[models[m.model_id].active_version_id]
            ok = {i for i, _ in selectable_classes(v.nvinfer_properties or {}, v.labels or [])}
            bad = sorted(set(m.detect_class_ids) - ok)
            if bad:
                raise HTTPException(422, f"model '{m.model_id}' cannot detect class id(s) {bad}; allowed: {sorted(ok)}")
    await db.execute(delete(CameraModelAssignment).where(CameraModelAssignment.camera_id == cam.id))
    for i, m in enumerate(body.models):
        db.add(CameraModelAssignment(camera_id=cam.id, model_id=m.model_id, enabled=m.enabled,
                                     parent_model_id=m.parent_model_id, operate_on_class_ids=m.operate_on_class_ids,
                                     detect_class_ids=sorted(set(m.detect_class_ids)) if m.detect_class_ids else None,
                                     conf_threshold=m.conf_threshold, iou_threshold=m.iou_threshold, order=i))
    cam.execution_strategy = strategy
    if body.tracking_enabled is not None:
        cam.tracking_enabled = body.tracking_enabled
    if body.tracker_type is not None:
        cam.tracker_type = body.tracker_type
    if body.detection_interval is not None:
        cam.detection_interval = body.detection_interval
    cam.config_generation += 1
    audit(db, request, who, "camera.assign_models", "camera", cam.id, {"models": [m.model_dump() for m in body.models]})
    await db.commit()
    return to_out(await _get(db, camera_id), (await _worker_map(db, [camera_id])).get(camera_id))


@router.get("/{camera_id}/stream")
async def camera_stream(camera_id: str, db: AsyncSession = Depends(get_session), who: Principal = Depends(viewer),
                        streams=Depends(get_streams), settings: Settings = Depends(get_settings)):
    cam = await _get(db, camera_id, who)
    from app.db.models import User
    epoch = (await db.execute(select(User.token_epoch).where(User.username == who.subject))).scalar_one_or_none() or 0
    from app.db.models import StreamMode
    if cam.stream_mode == StreamMode.NONE:
        raise HTTPException(409, "streaming disabled for this camera (set stream_mode)")
    out = {"camera_id": cam.id, "stream_mode": cam.stream_mode.value, "streams": {}}
    # short-lived token for THIS camera's video only (not valid for the API or any other camera); pass it to the player
    # as "Authorization: Bearer <token>" (preferred) or "?jwt=<token>"
    out["video_token"] = create_video_token(who.subject, cam.id, who.role, epoch, settings)
    out["video_token_expires_in"] = settings.video_token_ttl_s
    if cam.stream_mode in (StreamMode.ANNOTATED, StreamMode.BOTH):
        out["streams"]["processed"] = streams.urls(cam.id, "processed")
    if cam.stream_mode in (StreamMode.RAW, StreamMode.BOTH) and streams.s.allow_raw_streams:
        out["streams"]["raw"] = streams.urls(cam.id, "raw")
    return out


@router.get("/{camera_id}/stream/status")
async def camera_stream_status(camera_id: str, db: AsyncSession = Depends(get_session), who: Principal = Depends(viewer),
                               streams=Depends(get_streams)):
    cam = await _get(db, camera_id, who)
    res = await streams.status(db, cam)
    await db.commit()
    return res
