"""Metadata -> Kafka event logic (GPU-independent, unit tested).

The DeepStream probe converts pyds metadata into plain ``FrameResult`` objects;
everything below decides *what* to publish so the video thread stays cheap:

* a detection event is published when a track first appears and then every
  ``detection_event_interval_frames`` frames (per camera/model/track),
* embeddings are L2-normalised and published every ``embedding_interval_frames``
  per track,
* a per-camera token bucket caps events/s; excess is counted, never queued,
  so Kafka can never back-pressure the video pipeline.
"""
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

UNTRACKED = 0xFFFFFFFFFFFFFFFF


@dataclass
class ObjectResult:
    gie_id: int
    class_id: int
    label: str | None
    confidence: float
    tracker_id: int | None  # None / UNTRACKED when not tracked
    left: float
    top: float
    width: float
    height: float
    parent_tracker_id: int | None = None
    classifications: list[dict[str, Any]] = field(default_factory=list)
    embeddings: dict[int, list[float]] = field(default_factory=dict)  # reid gie_id -> vector


@dataclass
class FrameResult:
    source_id: int
    frame_num: int
    ntp_ns: int
    src_width: int
    src_height: int
    objects: list[ObjectResult]


class TokenBucket:
    def __init__(self, rate: float, burst: float | None = None, clock: Callable[[], float] = time.monotonic):
        self.rate = rate
        self.capacity = burst or rate
        self.tokens = self.capacity
        self.clock = clock
        self.last = clock()

    def take(self) -> bool:
        now = self.clock()
        self.tokens = min(self.capacity, self.tokens + (now - self.last) * self.rate)
        self.last = now
        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False


class SourceStats:
    """FPS / latency / deficit tracking for one camera inside a worker."""

    def __init__(self, expected_fps: int, window: int = 120):
        self.expected_fps = expected_fps
        self.arrivals: deque[float] = deque(maxlen=window)
        self.latencies: deque[float] = deque(maxlen=window)
        self.frames = 0
        self.last_frame_at: float | None = None
        self.deficit_frames = 0.0
        self._deficit_mark: float | None = None
        self._deficit_count = 0

    def on_frame(self, now: float, latency_ms: float | None) -> None:
        self.frames += 1
        self.last_frame_at = now
        self.arrivals.append(now)
        if latency_ms is not None and latency_ms >= 0:
            self.latencies.append(latency_ms)
        if self._deficit_mark is None:
            self._deficit_mark = now
        self._deficit_count += 1
        if now - self._deficit_mark >= 5.0:
            expected = (now - self._deficit_mark) * self.expected_fps
            self.deficit_frames += max(0.0, expected - self._deficit_count)
            self._deficit_mark, self._deficit_count = now, 0

    def fps(self) -> float:
        if len(self.arrivals) < 2:
            return 0.0
        span = self.arrivals[-1] - self.arrivals[0]
        return (len(self.arrivals) - 1) / span if span > 0 else 0.0

    def latency_ms(self) -> float | None:
        return sum(self.latencies) / len(self.latencies) if self.latencies else None


def l2norm(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v] if n > 0 else v


class EventEmitter:
    def __init__(self, *, worker_id: str, session_tag: str, gie_models: dict[int, dict[str, Any]],
                 publish: Callable[[str, dict, str], bool], detection_interval: int, embedding_interval: int,
                 max_events_per_s: int, clock: Callable[[], float] = time.monotonic):
        self.worker_id = worker_id
        self.session_tag = session_tag  # makes tracker IDs unique across pipeline restarts
        self.gie_models = gie_models    # gie_id -> {"model_id", "model_version", "task"}
        self.publish = publish
        self.det_iv = max(1, detection_interval)
        self.emb_iv = max(1, embedding_interval)
        self.max_eps = max_events_per_s
        self.clock = clock
        self._last_det: dict[tuple, int] = {}
        self._last_emb: dict[tuple, int] = {}
        self._buckets: dict[str, TokenBucket] = {}
        self.dropped = 0
        self.published = 0

    def object_id(self, tracker_id: int | None) -> str | None:
        if tracker_id is None or tracker_id == UNTRACKED:
            return None
        return f"TRACK_{self.session_tag}_{tracker_id}"

    def _bucket(self, cam: str) -> TokenBucket:
        b = self._buckets.get(cam)
        if b is None:
            b = self._buckets[cam] = TokenBucket(self.max_eps, clock=self.clock)
        return b

    def _send(self, topic: str, cam: str, payload: dict) -> None:
        if not self._bucket(cam).take():
            self.dropped += 1
            return
        if self.publish(topic, payload, cam):
            self.published += 1
        else:
            self.dropped += 1

    def forget_camera(self, cam: str) -> None:
        self._last_det = {k: v for k, v in self._last_det.items() if k[0] != cam}
        self._last_emb = {k: v for k, v in self._last_emb.items() if k[0] != cam}
        self._buckets.pop(cam, None)

    def process(self, camera_id: str, fr: FrameResult, mux_w: int, mux_h: int) -> None:
        from app.bus.schemas import Topics, new_id

        ts = datetime.fromtimestamp(fr.ntp_ns / 1e9, tz=timezone.utc).isoformat() if fr.ntp_ns else \
            datetime.now(timezone.utc).isoformat()
        sx = fr.src_width / mux_w if mux_w and fr.src_width else 1.0
        sy = fr.src_height / mux_h if mux_h and fr.src_height else 1.0
        for o in fr.objects:
            meta = self.gie_models.get(o.gie_id)
            if meta is None:
                continue
            oid = self.object_id(o.tracker_id)
            key = (camera_id, o.gie_id, oid if oid else f"untracked:{o.class_id}")
            last = self._last_det.get(key)
            bbox = {"x": round(o.left * sx, 1), "y": round(o.top * sy, 1),
                    "width": round(o.width * sx, 1), "height": round(o.height * sy, 1)}
            if last is None or fr.frame_num - last >= self.det_iv or fr.frame_num < last:
                self._last_det[key] = fr.frame_num
                ev = {
                    "schema_version": 1, "event_id": new_id(), "camera_id": camera_id, "worker_id": self.worker_id,
                    "model_id": meta["model_id"], "model_version": meta.get("model_version"),
                    "event_type": "object_detected" if last is None else "object_tracked",
                    "class_id": o.class_id, "class_name": o.label, "confidence": round(float(o.confidence), 4),
                    "object_id": oid, "bbox": bbox, "frame_num": fr.frame_num,
                    "frame_width": fr.src_width, "frame_height": fr.src_height, "timestamp": ts,
                }
                if o.parent_tracker_id is not None:
                    ev["parent_object_id"] = self.object_id(o.parent_tracker_id)
                if o.classifications:
                    ev["attributes"] = o.classifications
                self._send(Topics.DETECTION_EVENTS, camera_id, ev)
            if oid:
                for gid, vec in o.embeddings.items():
                    emeta = self.gie_models.get(gid)
                    if emeta is None:
                        continue
                    ekey = (camera_id, gid, oid)
                    el = self._last_emb.get(ekey)
                    if el is None or fr.frame_num - el >= self.emb_iv or fr.frame_num < el:
                        self._last_emb[ekey] = fr.frame_num
                        self._send(Topics.EMBEDDING_EVENTS, camera_id, {
                            "schema_version": 1, "event_id": new_id(), "camera_id": camera_id,
                            "worker_id": self.worker_id, "model_id": emeta["model_id"],
                            "model_version": emeta.get("model_version"), "object_id": oid,
                            "class_name": o.label, "embedding": l2norm(vec), "bbox": bbox, "timestamp": ts,
                        })
        # bound memory for long-running pipelines
        if len(self._last_det) > 50_000:
            cutoff = fr.frame_num - 10 * self.det_iv
            self._last_det = {k: v for k, v in self._last_det.items() if k[0] != camera_id or v >= cutoff}
        if len(self._last_emb) > 50_000:
            cutoff = fr.frame_num - 10 * self.emb_iv
            self._last_emb = {k: v for k, v in self._last_emb.items() if k[0] != camera_id or v >= cutoff}
