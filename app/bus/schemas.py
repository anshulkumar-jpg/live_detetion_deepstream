"""Versioned JSON event schemas shared by every service.

Every message on Kafka is an Envelope. `schema_version` lets consumers handle
old/new producers during rolling upgrades; unknown fields are preserved.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = 1


class Topics:
    CAMERA_COMMANDS = "camera.commands"
    CAMERA_STATUS = "camera.status"
    PIPELINE_COMMANDS = "pipeline.commands"
    PIPELINE_EVENTS = "pipeline.events"
    DETECTION_EVENTS = "detection.events"
    WORKER_HEARTBEAT = "worker.heartbeat"
    PIPELINE_METRICS = "pipeline.metrics"
    ANALYTICS_EVENTS = "analytics.events"
    REID_EVENTS = "reid.events"
    EMBEDDING_EVENTS = "embedding.events"  # sampled per-track embeddings for MTMC

    ALL = [
        CAMERA_COMMANDS, CAMERA_STATUS, PIPELINE_COMMANDS, PIPELINE_EVENTS, DETECTION_EVENTS,
        WORKER_HEARTBEAT, PIPELINE_METRICS, ANALYTICS_EVENTS, REID_EVENTS, EMBEDDING_EVENTS,
    ]

    @staticmethod
    def dlq(topic: str) -> str:
        return f"{topic}.dlq"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id() -> str:
    return str(uuid.uuid4())


class Envelope(BaseModel):
    model_config = ConfigDict(extra="allow")
    schema_version: int = SCHEMA_VERSION
    event_id: str = Field(default_factory=new_id)
    event_type: str
    source: str
    timestamp: str = Field(default_factory=now_iso)
    payload: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------- commands
class PipelineCommandType(str, Enum):
    START_WORKER = "START_WORKER"
    STOP_WORKER = "STOP_WORKER"
    RESTART_WORKER = "RESTART_WORKER"
    ADD_SOURCE = "ADD_SOURCE"
    REMOVE_SOURCE = "REMOVE_SOURCE"


class PipelineEventType(str, Enum):
    WORKER_STARTING = "WORKER_STARTING"
    WORKER_RUNNING = "WORKER_RUNNING"
    WORKER_DEGRADED = "WORKER_DEGRADED"
    WORKER_STOPPED = "WORKER_STOPPED"
    WORKER_FAILED = "WORKER_FAILED"
    SOURCE_ADDED = "SOURCE_ADDED"
    SOURCE_REMOVED = "SOURCE_REMOVED"
    SOURCE_RUNNING = "SOURCE_RUNNING"
    SOURCE_RECONNECTING = "SOURCE_RECONNECTING"
    SOURCE_FAILED = "SOURCE_FAILED"
    MODEL_LOAD_FAILED = "MODEL_LOAD_FAILED"
    COMMAND_REJECTED = "COMMAND_REJECTED"


# --------------------------------------------------------------- data events
class BBox(BaseModel):
    x: float
    y: float
    width: float
    height: float


class DetectionEventSchema(BaseModel):
    """Matches the spec's example detection event (plus schema_version / frame fields)."""

    model_config = ConfigDict(extra="allow")
    schema_version: int = SCHEMA_VERSION
    event_id: str = Field(default_factory=new_id)
    camera_id: str
    worker_id: str
    model_id: str
    event_type: str = "object_detected"
    class_id: int
    class_name: str | None = None
    confidence: float
    object_id: str | None = None
    bbox: BBox
    frame_num: int | None = None
    frame_width: int | None = None
    frame_height: int | None = None
    parent_object_id: str | None = None
    timestamp: str = Field(default_factory=now_iso)


class EmbeddingEventSchema(BaseModel):
    schema_version: int = SCHEMA_VERSION
    event_id: str = Field(default_factory=new_id)
    camera_id: str
    worker_id: str
    model_id: str            # embedding model
    model_version: str | None = None
    object_id: str
    class_name: str | None = None
    embedding: list[float]
    bbox: BBox | None = None
    timestamp: str = Field(default_factory=now_iso)


class HeartbeatSchema(BaseModel):
    model_config = ConfigDict(extra="allow")
    schema_version: int = SCHEMA_VERSION
    kind: str  # "worker" | "node"
    node_id: str
    worker_id: str | None = None
    state: str | None = None
    pid: int | None = None
    spec_generation: int | None = None
    sources: dict[str, Any] = Field(default_factory=dict)  # camera_id -> status/fps
    gpus: list[dict[str, Any]] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=now_iso)
