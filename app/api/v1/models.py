"""Model registry: models, versions, activation and rollback."""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.api.deps import Principal, admin, audit, get_session, viewer
from app.db.models import InferenceBackend, Model, ModelVersion, ModelVersionStatus
from app.schemas.api import ModelCreate, ModelOut, ModelUpdate, ModelVersionIn, ModelVersionOut
from app.services.scheduler import bump_cameras_for_model

router = APIRouter(prefix="/models", tags=["models"])


async def _get(db: AsyncSession, model_id: str) -> Model:
    m = (await db.execute(select(Model).options(selectinload(Model.versions), selectinload(Model.active_version))
                          .where(Model.id == model_id).execution_options(populate_existing=True))).scalar_one_or_none()
    if m is None:
        raise HTTPException(404, f"model '{model_id}' not found")
    return m


def _out(m: Model) -> ModelOut:
    return ModelOut(
        id=m.id, name=m.name, task=m.task, backend=m.backend, description=m.description, enabled=m.enabled,
        active_version=ModelVersionOut.model_validate(m.active_version) if m.active_version else None,
        versions=[ModelVersionOut.model_validate(v) for v in sorted(m.versions, key=lambda v: v.id)],
    )


def _validate_version(m: Model, v: ModelVersionIn) -> None:
    if m.backend == InferenceBackend.NVINFER and not (v.onnx_file or v.engine_file):
        raise HTTPException(422, "nvinfer models need onnx_file and/or engine_file")
    if m.backend == InferenceBackend.NVINFERSERVER and not v.nvinferserver_config_file:
        raise HTTPException(422, "nvinferserver models need nvinferserver_config_file")
    if v.precision == "int8" and not (v.int8_calib_file or v.engine_file):
        raise HTTPException(422, "int8 precision needs int8_calib_file (or a prebuilt engine_file)")
    if m.task.value == "reid" and not v.embedding_dim:
        raise HTTPException(422, "reid models must declare embedding_dim")


async def _activate(db: AsyncSession, m: Model, v: ModelVersion) -> int:
    if v.status == ModelVersionStatus.FAILED:
        v.failure_reason = None
    for other in m.versions:
        if other.id != v.id and other.status == ModelVersionStatus.ACTIVE:
            other.status = ModelVersionStatus.RETIRED
    v.status = ModelVersionStatus.ACTIVE
    v.activated_at = datetime.now(timezone.utc)
    m.active_version_id = v.id
    await db.flush()
    return await bump_cameras_for_model(db, m.id)


@router.get("", response_model=list[ModelOut])
async def list_models(db: AsyncSession = Depends(get_session), _: Principal = Depends(viewer)):
    rows = (await db.execute(select(Model).options(selectinload(Model.versions), selectinload(Model.active_version))
                             .order_by(Model.id))).scalars()
    return [_out(m) for m in rows]


@router.get("/{model_id}", response_model=ModelOut)
async def get_model(model_id: str, db: AsyncSession = Depends(get_session), _: Principal = Depends(viewer)):
    return _out(await _get(db, model_id))


@router.post("", response_model=ModelOut, status_code=201)
async def create_model(body: ModelCreate, request: Request, db: AsyncSession = Depends(get_session),
                       who: Principal = Depends(admin)):
    if await db.get(Model, body.id) is not None:
        raise HTTPException(409, f"model '{body.id}' already exists")
    m = Model(id=body.id, name=body.name, task=body.task, backend=body.backend, description=body.description,
              enabled=body.enabled)
    db.add(m)
    await db.flush()
    if body.version is not None:
        _validate_version(m, body.version)
        v = ModelVersion(model_id=m.id, **body.version.model_dump(exclude={"activate"}))
        db.add(v)
        await db.flush()
        await db.refresh(m, ["versions"])
        if body.version.activate:
            await _activate(db, m, v)
    audit(db, request, who, "model.create", "model", m.id)
    await db.commit()
    return _out(await _get(db, m.id))


@router.put("/{model_id}", response_model=ModelOut)
async def update_model(model_id: str, body: ModelUpdate, request: Request, db: AsyncSession = Depends(get_session),
                       who: Principal = Depends(admin)):
    """Update metadata, enable/disable, register a new version, or activate/roll back
    to an existing version. Only cameras using this model get their pipelines rebuilt."""
    m = await _get(db, model_id)
    affected = 0
    if body.name is not None:
        m.name = body.name
    if body.description is not None:
        m.description = body.description
    if body.enabled is not None and body.enabled != m.enabled:
        m.enabled = body.enabled
        affected = await bump_cameras_for_model(db, m.id)
    if body.new_version is not None:
        if any(v.version == body.new_version.version for v in m.versions):
            raise HTTPException(409, f"version {body.new_version.version} already exists")
        _validate_version(m, body.new_version)
        v = ModelVersion(model_id=m.id, **body.new_version.model_dump(exclude={"activate"}))
        db.add(v)
        await db.flush()
        await db.refresh(m, ["versions"])
        if body.new_version.activate:
            affected = await _activate(db, m, v)
    if body.activate_version is not None:
        v = (await db.execute(select(ModelVersion).where(and_(
            ModelVersion.model_id == m.id, ModelVersion.version == body.activate_version)))).scalar_one_or_none()
        if v is None:
            raise HTTPException(404, f"version {body.activate_version} not found")
        affected = await _activate(db, m, v)
    audit(db, request, who, "model.update", "model", m.id,
          {"fields": sorted(body.model_dump(exclude_unset=True)), "cameras_affected": affected})
    await db.commit()
    return _out(await _get(db, m.id))
