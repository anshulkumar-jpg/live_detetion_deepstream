"""PostgreSQL schema (SQLAlchemy 2.0). PostgreSQL is the source of truth for
configuration, desired state, assignments and worker state. High-frequency
per-frame detections go to Kafka -> Logstash -> Elasticsearch; only track-level
summaries are written here.
"""
from __future__ import annotations

import enum
from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

JsonType = JSON().with_variant(JSONB(), "postgresql")
# BigInteger autoincrement does not work on SQLite (used in unit tests only)
BigIntPK = BigInteger().with_variant(Integer(), "sqlite")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


# ------------------------------------------------------------------ enums
class DesiredState(str, enum.Enum):
    RUNNING = "RUNNING"
    STOPPED = "STOPPED"


class CameraStatus(str, enum.Enum):
    REGISTERED = "REGISTERED"
    PENDING_CAPACITY = "PENDING_CAPACITY"
    SCHEDULED = "SCHEDULED"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    RECONNECTING = "RECONNECTING"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"
    FAILED = "FAILED"


class WorkerState(str, enum.Enum):
    CREATED = "CREATED"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    DEGRADED = "DEGRADED"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"
    FAILED = "FAILED"


class ModelTask(str, enum.Enum):
    DETECTOR = "detector"
    CLASSIFIER = "classifier"
    POSE = "pose"
    SEGMENTATION = "segmentation"
    REID = "reid"  # embedding extractor


class InferenceBackend(str, enum.Enum):
    NVINFER = "nvinfer"            # TensorRT through Gst-nvinfer
    NVINFERSERVER = "nvinferserver"  # Triton through Gst-nvinferserver (custom/unsupported models)


class ModelVersionStatus(str, enum.Enum):
    REGISTERED = "REGISTERED"
    ACTIVE = "ACTIVE"
    FAILED = "FAILED"
    RETIRED = "RETIRED"


class ExecutionStrategy(str, enum.Enum):
    INDEPENDENT = "independent"  # every primary model runs on the full frame
    CASCADE = "cascade"          # secondary models run on parent model's objects


class StreamMode(str, enum.Enum):
    NONE = "none"
    ANNOTATED = "annotated"  # NVENC output pushed to MediaMTX
    RAW = "raw"              # MediaMTX proxies the original camera stream
    BOTH = "both"


class AssignmentState(str, enum.Enum):
    ASSIGNING = "ASSIGNING"
    ACTIVE = "ACTIVE"
    RELEASING = "RELEASING"
    RELEASED = "RELEASED"


def _enum(e: type[enum.Enum], name: str) -> Enum:
    return Enum(e, name=name, native_enum=False, length=32, validate_strings=True)


# ------------------------------------------------------------------ tables
class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(16), nullable=False, default="viewer")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Camera(Base):
    __tablename__ = "cameras"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    rtsp_url: Mapped[str] = mapped_column(Text, nullable=False)  # never contains credentials
    username_enc: Mapped[str | None] = mapped_column(Text)
    password_enc: Mapped[str | None] = mapped_column(Text)
    location: Mapped[str | None] = mapped_column(String(255))
    tags: Mapped[dict] = mapped_column(JsonType, default=dict)
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)
    fps: Mapped[int] = mapped_column(Integer, default=30)
    codec_hint: Mapped[str | None] = mapped_column(String(16))
    tracking_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    tracker_type: Mapped[str] = mapped_column(String(16), default="NvDCF")
    # frames skipped between detector runs (0 = detect every frame); the tracker still runs on every frame
    detection_interval: Mapped[int] = mapped_column(Integer, default=0, server_default="0", nullable=False)
    execution_strategy: Mapped[ExecutionStrategy] = mapped_column(
        _enum(ExecutionStrategy, "execution_strategy"), default=ExecutionStrategy.INDEPENDENT
    )
    stream_mode: Mapped[StreamMode] = mapped_column(_enum(StreamMode, "stream_mode"), default=StreamMode.NONE)
    desired_state: Mapped[DesiredState] = mapped_column(
        _enum(DesiredState, "desired_state"), default=DesiredState.STOPPED, nullable=False
    )
    status: Mapped[CameraStatus] = mapped_column(
        _enum(CameraStatus, "camera_status"), default=CameraStatus.REGISTERED, nullable=False
    )
    status_reason: Mapped[str | None] = mapped_column(Text)
    status_changed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # bumped on any change that requires the pipeline to be rebuilt for this camera
    config_generation: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    rtsp_failures: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    assignments: Mapped[list[CameraModelAssignment]] = relationship(
        back_populates="camera", cascade="all, delete-orphan", order_by="CameraModelAssignment.order"
    )

    __table_args__ = (
        Index("ix_cameras_desired_status", "desired_state", "status"),
    )


class Model(Base):
    __tablename__ = "models"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    task: Mapped[ModelTask] = mapped_column(_enum(ModelTask, "model_task"), nullable=False)
    backend: Mapped[InferenceBackend] = mapped_column(
        _enum(InferenceBackend, "inference_backend"), default=InferenceBackend.NVINFER
    )
    description: Mapped[str | None] = mapped_column(Text)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    active_version_id: Mapped[int | None] = mapped_column(
        ForeignKey("model_versions.id", use_alter=True, name="fk_models_active_version", ondelete="SET NULL")
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    versions: Mapped[list[ModelVersion]] = relationship(
        back_populates="model", foreign_keys="ModelVersion.model_id", cascade="all, delete-orphan"
    )
    active_version: Mapped[ModelVersion | None] = relationship(foreign_keys=[active_version_id], post_update=True)


class ModelVersion(Base):
    __tablename__ = "model_versions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_id: Mapped[str] = mapped_column(ForeignKey("models.id", ondelete="CASCADE"), nullable=False)
    version: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[ModelVersionStatus] = mapped_column(
        _enum(ModelVersionStatus, "model_version_status"), default=ModelVersionStatus.REGISTERED
    )
    # artifact locations inside the model repository volume
    onnx_file: Mapped[str | None] = mapped_column(Text)
    engine_file: Mapped[str | None] = mapped_column(Text)
    labels: Mapped[list] = mapped_column(JsonType, default=list)
    precision: Mapped[str] = mapped_column(String(8), default="fp16")  # fp32|fp16|int8
    int8_calib_file: Mapped[str | None] = mapped_column(Text)
    max_batch_size: Mapped[int] = mapped_column(Integer, default=2)
    input_shape: Mapped[list | None] = mapped_column(JsonType)  # [C,H,W]
    # extra Gst-nvinfer [property] keys (preprocessing, parser, clustering, ...)
    nvinfer_properties: Mapped[dict] = mapped_column(JsonType, default=dict)
    class_attrs: Mapped[dict] = mapped_column(JsonType, default=dict)  # per-class thresholds
    nvinferserver_config_file: Mapped[str | None] = mapped_column(Text)
    embedding_dim: Mapped[int | None] = mapped_column(Integer)
    est_memory_mb: Mapped[int | None] = mapped_column(Integer)  # measured by benchmark
    est_latency_ms: Mapped[float | None] = mapped_column(Float)
    failure_reason: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    activated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    model: Mapped[Model] = relationship(back_populates="versions", foreign_keys=[model_id])
    __table_args__ = (UniqueConstraint("model_id", "version", name="uq_model_version"),)


class CameraModelAssignment(Base):
    __tablename__ = "camera_model_assignments"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    camera_id: Mapped[str] = mapped_column(ForeignKey("cameras.id", ondelete="CASCADE"), nullable=False)
    model_id: Mapped[str] = mapped_column(ForeignKey("models.id", ondelete="RESTRICT"), nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    # cascade: run this model on objects produced by parent_model_id
    parent_model_id: Mapped[str | None] = mapped_column(ForeignKey("models.id", ondelete="RESTRICT"))
    operate_on_class_ids: Mapped[list | None] = mapped_column(JsonType)
    order: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    camera: Mapped[Camera] = relationship(back_populates="assignments")
    __table_args__ = (
        UniqueConstraint("camera_id", "model_id", name="uq_camera_model"),
        Index("ix_cma_model", "model_id"),
    )


class GpuDevice(Base):
    __tablename__ = "gpu_devices"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    node_id: Mapped[str] = mapped_column(String(64), nullable=False)
    gpu_index: Mapped[int] = mapped_column(Integer, nullable=False)
    uuid: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    name: Mapped[str | None] = mapped_column(String(128))
    memory_total_mb: Mapped[int | None] = mapped_column(Integer)
    driver_version: Mapped[str | None] = mapped_column(String(32))
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    healthy: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    latest_metrics: Mapped[dict] = mapped_column(JsonType, default=dict)
    last_seen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    __table_args__ = (UniqueConstraint("node_id", "gpu_index", name="uq_gpu_node_index"),)


class GpuResourceMetric(Base):
    __tablename__ = "gpu_resource_metrics"
    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    gpu_device_id: Mapped[int] = mapped_column(ForeignKey("gpu_devices.id", ondelete="CASCADE"), nullable=False)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    gpu_util_pct: Mapped[float | None] = mapped_column(Float)
    memory_used_mb: Mapped[float | None] = mapped_column(Float)
    memory_total_mb: Mapped[float | None] = mapped_column(Float)
    decoder_util_pct: Mapped[float | None] = mapped_column(Float)
    encoder_util_pct: Mapped[float | None] = mapped_column(Float)
    temperature_c: Mapped[float | None] = mapped_column(Float)
    power_w: Mapped[float | None] = mapped_column(Float)
    active_streams: Mapped[int | None] = mapped_column(Integer)
    __table_args__ = (Index("ix_gpu_metrics_dev_ts", "gpu_device_id", "ts"),)


class CapacityProfile(Base):
    """Benchmark-measured capacity. Admission control trusts these over defaults."""

    __tablename__ = "capacity_profiles"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    gpu_name: Mapped[str] = mapped_column(String(128), nullable=False)
    pipeline_signature: Mapped[str] = mapped_column(String(64), nullable=False)
    model_set: Mapped[list] = mapped_column(JsonType, default=list)
    input_resolution: Mapped[str | None] = mapped_column(String(16))
    input_fps: Mapped[int] = mapped_column(Integer, default=30)
    max_streams: Mapped[int] = mapped_column(Integer, nullable=False)
    memory_mb_per_stream: Mapped[float | None] = mapped_column(Float)
    avg_fps_per_stream: Mapped[float | None] = mapped_column(Float)
    p95_latency_ms: Mapped[float | None] = mapped_column(Float)
    notes: Mapped[str | None] = mapped_column(Text)
    measured_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    __table_args__ = (Index("ix_capacity_gpu_sig", "gpu_name", "pipeline_signature"),)


class Worker(Base):
    """A logical worker = one DeepStream pipeline group (default 2 cameras),
    run as an isolated OS process by the node agent."""

    __tablename__ = "workers"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    node_id: Mapped[str] = mapped_column(String(64), nullable=False)
    gpu_device_id: Mapped[int | None] = mapped_column(ForeignKey("gpu_devices.id", ondelete="SET NULL"))
    pipeline_signature: Mapped[str] = mapped_column(String(64), nullable=False)
    model_set: Mapped[list] = mapped_column(JsonType, default=list)
    capacity: Mapped[int] = mapped_column(Integer, nullable=False, default=2)
    state: Mapped[WorkerState] = mapped_column(_enum(WorkerState, "worker_state"), default=WorkerState.CREATED)
    desired_state: Mapped[DesiredState] = mapped_column(_enum(DesiredState, "worker_desired_state"), default=DesiredState.RUNNING)
    health: Mapped[str] = mapped_column(String(16), default="unknown")
    pid: Mapped[int | None] = mapped_column(Integer)
    restart_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    spec_generation: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    last_heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    state_changed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    stopped_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_error: Mapped[str | None] = mapped_column(Text)
    metrics: Mapped[dict] = mapped_column(JsonType, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)
    __table_args__ = (
        Index("ix_workers_state", "state"),
        Index("ix_workers_sig", "pipeline_signature", "state"),
    )


class WorkerCameraAssignment(Base):
    """Assignment history; rows with released_at IS NULL are current."""

    __tablename__ = "worker_camera_assignments"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    worker_id: Mapped[str] = mapped_column(ForeignKey("workers.id", ondelete="CASCADE"), nullable=False)
    camera_id: Mapped[str] = mapped_column(ForeignKey("cameras.id", ondelete="CASCADE"), nullable=False)
    camera_generation: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    state: Mapped[AssignmentState] = mapped_column(_enum(AssignmentState, "assignment_state"), default=AssignmentState.ASSIGNING)
    reason: Mapped[str | None] = mapped_column(String(255))
    assigned_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    released_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    __table_args__ = (
        Index(
            "uq_wca_active_camera",
            "camera_id",
            unique=True,
            postgresql_where=text("released_at IS NULL"),
            sqlite_where=text("released_at IS NULL"),
        ),
        Index("ix_wca_worker_active", "worker_id", "released_at"),
    )


class PipelineSession(Base):
    __tablename__ = "pipeline_sessions"
    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    worker_id: Mapped[str] = mapped_column(String(64), nullable=False)
    camera_id: Mapped[str] = mapped_column(ForeignKey("cameras.id", ondelete="CASCADE"), nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    end_reason: Mapped[str | None] = mapped_column(String(255))
    frames_processed: Mapped[int] = mapped_column(BigInteger, default=0)
    avg_fps: Mapped[float | None] = mapped_column(Float)
    __table_args__ = (Index("ix_pipeline_sessions_cam", "camera_id", "started_at"),)


class StreamSession(Base):
    __tablename__ = "stream_sessions"
    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    camera_id: Mapped[str] = mapped_column(ForeignKey("cameras.id", ondelete="CASCADE"), nullable=False)
    kind: Mapped[str] = mapped_column(String(16), nullable=False)  # processed | raw
    path: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(16), default="starting")  # starting|ready|not_ready|ended
    viewer_count: Mapped[int] = mapped_column(Integer, default=0)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    last_checked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    __table_args__ = (Index("ix_stream_sessions_cam", "camera_id", "ended_at"),)


class DetectionEvent(Base):
    """Track-level summary (one row per camera/model/track), upserted by analytics."""

    __tablename__ = "detection_events"
    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    event_id: Mapped[str] = mapped_column(String(36), unique=True, nullable=False)
    camera_id: Mapped[str] = mapped_column(String(64), nullable=False)
    worker_id: Mapped[str | None] = mapped_column(String(64))
    model_id: Mapped[str] = mapped_column(String(64), nullable=False)
    object_id: Mapped[str] = mapped_column(String(64), nullable=False)
    class_id: Mapped[int] = mapped_column(Integer, nullable=False)
    class_name: Mapped[str | None] = mapped_column(String(64))
    max_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    last_bbox: Mapped[dict] = mapped_column(JsonType, default=dict)
    sample_count: Mapped[int] = mapped_column(Integer, default=1)
    global_id: Mapped[str | None] = mapped_column(String(64))
    first_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    last_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    __table_args__ = (
        UniqueConstraint("camera_id", "model_id", "object_id", name="uq_detection_track"),
        Index("ix_detection_cam_time", "camera_id", "last_seen"),
        Index("ix_detection_class_time", "class_name", "last_seen"),
    )


class AnalyticsEvent(Base):
    __tablename__ = "analytics_events"
    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    event_id: Mapped[str] = mapped_column(String(36), unique=True, nullable=False)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    camera_id: Mapped[str | None] = mapped_column(String(64))
    payload: Mapped[dict] = mapped_column(JsonType, default=dict)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    __table_args__ = (Index("ix_analytics_type_ts", "event_type", "ts"), Index("ix_analytics_cam_ts", "camera_id", "ts"))


class ReidAssociation(Base):
    __tablename__ = "reid_associations"
    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    global_id: Mapped[str] = mapped_column(String(64), nullable=False)
    camera_id: Mapped[str] = mapped_column(String(64), nullable=False)
    object_id: Mapped[str] = mapped_column(String(64), nullable=False)
    model_id: Mapped[str | None] = mapped_column(String(64))
    similarity: Mapped[float | None] = mapped_column(Float)
    matched_camera_id: Mapped[str | None] = mapped_column(String(64))
    matched_object_id: Mapped[str | None] = mapped_column(String(64))
    first_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    last_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    __table_args__ = (
        UniqueConstraint("camera_id", "object_id", name="uq_reid_track"),
        Index("ix_reid_global", "global_id"),
    )


class SystemConfiguration(Base):
    __tablename__ = "system_configuration"
    key: Mapped[str] = mapped_column(String(128), primary_key=True)
    value: Mapped[dict] = mapped_column(JsonType, nullable=False)
    updated_by: Mapped[str | None] = mapped_column(String(64))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class AuditLog(Base):
    __tablename__ = "audit_logs"
    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, server_default=func.now())
    actor: Mapped[str] = mapped_column(String(64), nullable=False)
    action: Mapped[str] = mapped_column(String(64), nullable=False)
    resource_type: Mapped[str] = mapped_column(String(32), nullable=False)
    resource_id: Mapped[str | None] = mapped_column(String(128))
    details: Mapped[dict] = mapped_column(JsonType, default=dict)
    request_id: Mapped[str | None] = mapped_column(String(64))
    client_ip: Mapped[str | None] = mapped_column(String(64))
    __table_args__ = (Index("ix_audit_ts", "ts"), Index("ix_audit_resource", "resource_type", "resource_id"))


class ProcessedCommand(Base):
    """Idempotency ledger for commands/events consumed by the control plane."""

    __tablename__ = "processed_commands"
    command_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    source: Mapped[str] = mapped_column(String(64), nullable=False)
    processed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
