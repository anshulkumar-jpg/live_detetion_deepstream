"""Request/response models for the public API. RTSP credentials are write-only."""
from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Literal

from app.core.trackers import TrackerType
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.db.models import (
    CameraStatus,
    DesiredState,
    ExecutionStrategy,
    InferenceBackend,
    ModelTask,
    ModelVersionStatus,
    StreamMode,
    WorkerState,
)

ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def _check_id(v: str) -> str:
    if not ID_RE.match(v):
        raise ValueError("must match [A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
    return v


# ------------------------------------------------------------------ auth
class TokenRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    role: str


class UserCreate(BaseModel):
    username: str = Field(min_length=3, max_length=64)
    password: str = Field(min_length=12, max_length=256)
    role: Literal["admin", "operator", "viewer"] = "viewer"


# ------------------------------------------------------------------ cameras
class CameraBase(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    location: str | None = None
    tags: dict[str, Any] = Field(default_factory=dict)
    width: int | None = Field(default=None, ge=16, le=8192)
    height: int | None = Field(default=None, ge=16, le=8192)
    fps: int = Field(default=30, ge=1, le=120)
    codec_hint: Literal["h264", "h265", "mjpeg"] | None = None
    tracking_enabled: bool = True
    tracker_type: TrackerType = "NvDCF"
    detection_interval: int = Field(default=0, ge=0, le=4)  # frames skipped between detections
    execution_strategy: ExecutionStrategy = ExecutionStrategy.INDEPENDENT
    stream_mode: StreamMode = StreamMode.NONE


class CameraCreate(CameraBase):
    id: str
    rtsp_url: str
    username: str | None = Field(default=None, max_length=255)
    password: str | None = Field(default=None, max_length=255)

    _id = field_validator("id")(_check_id)

    @field_validator("rtsp_url")
    @classmethod
    def _rtsp(cls, v: str) -> str:
        if not re.match(r"^(rtsp|rtsps)://[^\s]+$", v):
            raise ValueError("rtsp_url must be rtsp:// or rtsps://")
        return v


class CameraUpdate(BaseModel):
    name: str | None = None
    location: str | None = None
    tags: dict[str, Any] | None = None
    rtsp_url: str | None = None
    username: str | None = None
    password: str | None = None
    width: int | None = Field(default=None, ge=16, le=8192)
    height: int | None = Field(default=None, ge=16, le=8192)
    fps: int | None = Field(default=None, ge=1, le=120)
    codec_hint: Literal["h264", "h265", "mjpeg"] | None = None
    tracking_enabled: bool | None = None
    tracker_type: TrackerType | None = None
    detection_interval: int | None = Field(default=None, ge=0, le=4)
    execution_strategy: ExecutionStrategy | None = None
    stream_mode: StreamMode | None = None

    @field_validator("rtsp_url")
    @classmethod
    def _rtsp(cls, v: str | None) -> str | None:
        if v is not None and not re.match(r"^(rtsp|rtsps)://[^\s]+$", v):
            raise ValueError("rtsp_url must be rtsp:// or rtsps://")
        return v


class ModelAssignmentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())
    model_id: str
    enabled: bool
    parent_model_id: str | None = None
    operate_on_class_ids: list[int] | None = None
    detect_class_ids: list[int] | None = None
    conf_threshold: float | None = None
    iou_threshold: float | None = None
    order: int


class CameraOut(CameraBase):
    model_config = ConfigDict(from_attributes=True)
    id: str
    rtsp_url: str  # credentials are stored separately and never returned
    has_credentials: bool
    desired_state: DesiredState
    status: CameraStatus
    status_reason: str | None
    config_generation: int
    worker_id: str | None = None
    models: list[ModelAssignmentOut] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime


class CameraBulkCreate(BaseModel):
    cameras: list[CameraCreate] = Field(min_length=1, max_length=1000)


class Page(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[Any]


# ------------------------------------------------------------------ models
class ModelVersionIn(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    version: str = Field(min_length=1, max_length=32)
    onnx_file: str | None = None
    engine_file: str | None = None
    labels: list[str] = Field(default_factory=list)
    precision: Literal["fp32", "fp16", "int8"] = "fp16"
    int8_calib_file: str | None = None
    max_batch_size: int = Field(default=2, ge=1, le=256)
    input_shape: list[int] | None = None
    nvinfer_properties: dict[str, Any] = Field(default_factory=dict)
    class_attrs: dict[str, Any] = Field(default_factory=dict)
    nvinferserver_config_file: str | None = None
    embedding_dim: int | None = Field(default=None, ge=1, le=8192)
    est_memory_mb: int | None = Field(default=None, ge=0)
    activate: bool = True

    @field_validator("onnx_file", "engine_file", "int8_calib_file", "nvinferserver_config_file")
    @classmethod
    def _no_traversal(cls, v: str | None) -> str | None:
        if v is not None and (".." in v.split("/") or v.startswith("~")):
            raise ValueError("path traversal not allowed")
        return v

    @field_validator("nvinfer_properties")
    @classmethod
    def _props(cls, v: dict[str, Any]) -> dict[str, Any]:
        for k, val in v.items():
            if not re.match(r"^[a-z0-9-]+$", k):
                raise ValueError(f"invalid nvinfer property key '{k}'")
            if isinstance(val, str) and ("\n" in val or "[" in val):
                raise ValueError(f"invalid value for '{k}'")
        return v


class ModelCreate(BaseModel):
    id: str
    name: str
    task: ModelTask
    backend: InferenceBackend = InferenceBackend.NVINFER
    description: str | None = None
    enabled: bool = True
    version: ModelVersionIn | None = None

    _id = field_validator("id")(_check_id)


class ModelUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    enabled: bool | None = None
    new_version: ModelVersionIn | None = None
    activate_version: str | None = Field(default=None, description="activate (or roll back to) an existing version")


class ModelVersionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())
    id: int
    version: str
    status: ModelVersionStatus
    onnx_file: str | None
    engine_file: str | None
    labels: list[str]
    precision: str
    max_batch_size: int
    input_shape: list[int] | None
    nvinfer_properties: dict[str, Any]
    class_attrs: dict[str, Any] = Field(default_factory=dict)
    embedding_dim: int | None
    est_memory_mb: int | None
    failure_reason: str | None
    created_at: datetime
    activated_at: datetime | None


class ModelOut(BaseModel):
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())
    id: str
    name: str
    task: ModelTask
    backend: InferenceBackend
    description: str | None
    enabled: bool
    active_version: ModelVersionOut | None = None
    versions: list[ModelVersionOut] = Field(default_factory=list)


class CameraModelItem(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    enabled: bool = True
    parent_model_id: str | None = None
    operate_on_class_ids: list[int] | None = None
    detect_class_ids: list[int] | None = Field(default=None, min_length=1, max_length=256)  # None = all classes
    conf_threshold: float | None = Field(default=None, ge=0.01, le=0.99)  # min detection confidence (None = default)
    iou_threshold: float | None = Field(default=None, ge=0.05, le=0.95)   # NMS overlap limit (None = default)


class CameraModelsRequest(BaseModel):
    models: list[CameraModelItem] = Field(min_length=1, max_length=16)
    tracking_enabled: bool | None = None
    tracker_type: TrackerType | None = None
    detection_interval: int | None = Field(default=None, ge=0, le=4)
    execution_strategy: ExecutionStrategy | None = None


# ------------------------------------------------------------------ workers
class WorkerOut(BaseModel):
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())
    id: str
    node_id: str
    gpu_device_id: int | None
    pipeline_signature: str
    model_set: list[Any]
    capacity: int
    state: WorkerState
    desired_state: DesiredState
    health: str
    pid: int | None
    restart_count: int
    last_heartbeat_at: datetime | None
    started_at: datetime | None
    last_error: str | None
    camera_ids: list[str] = Field(default_factory=list)
    created_at: datetime


class CapacityProfileIn(BaseModel):
    gpu_name: str
    pipeline_signature: str
    model_set: list[Any] = Field(default_factory=list)
    input_resolution: str | None = None
    input_fps: int = 30
    max_streams: int = Field(ge=1)
    memory_mb_per_stream: float | None = None
    avg_fps_per_stream: float | None = None
    p95_latency_ms: float | None = None
    notes: str | None = None
