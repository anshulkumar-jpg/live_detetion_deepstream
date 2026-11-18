"""MTMC (multi-target multi-camera) service: appearance-based cross-camera Re-ID.

Scope (deliberately explicit):
* Local tracking stays in DeepStream (nvtracker) - IDs are only unique per camera session.
* This service links local tracks across cameras using Re-ID embeddings only:
  a track's embeddings are averaged; once it has ``mtmc_min_track_samples`` samples
  it is matched (cosine similarity) against recent tracks from *other* cameras of the
  same class inside ``mtmc_time_window_s``. Above ``mtmc_match_threshold`` the track
  joins that global identity, otherwise it starts a new one.
* No spatial/temporal camera-topology constraints and no calibration are applied.
  Accuracy depends on the Re-ID model and threshold and must be validated on your
  footage before global IDs are relied on. Calibration-aware matching (NVIDIA
  Calibration Toolkit output) can be plugged in via ``CandidateFilter``.

Run: python -m analytics.mtmc
"""
from __future__ import annotations

import asyncio
import os
import logging
import signal
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol

import numpy as np
from sqlalchemy import select, update

from app.bus.schemas import Topics, new_id, now_iso
from app.core.config import Settings, get_settings
from app.core.metrics import ANALYTICS
from app.db.models import DetectionEvent, ReidAssociation

log = logging.getLogger("mtmc")


class CandidateFilter(Protocol):
    """Hook for calibration/topology-aware gating (e.g. camera adjacency, travel time)."""

    def allow(self, track: "TrackState", candidate_meta: dict[str, Any]) -> bool: ...


class AllowAll:
    def allow(self, track, candidate_meta) -> bool:
        return True


@dataclass
class TrackState:
    camera_id: str
    object_id: str
    class_name: str | None
    model_id: str
    model_version: str | None
    sum_vec: np.ndarray
    count: int = 0
    global_id: str | None = None
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    matched: dict[str, Any] | None = None

    @property
    def key(self) -> str:
        return f"{self.camera_id}:{self.object_id}"

    def mean(self) -> list[float]:
        v = self.sum_vec / max(self.count, 1)
        n = np.linalg.norm(v)
        return (v / n if n else v).astype(np.float32).tolist()


class MTMCMatcher:
    def __init__(self, store, settings: Settings, bus, sessionmaker=None, candidate_filter: CandidateFilter | None = None,
                 clock=time.time):
        self.store = store
        self.s = settings
        self.bus = bus
        self.sm = sessionmaker
        self.filter = candidate_filter or AllowAll()
        self.clock = clock
        self.tracks: dict[str, TrackState] = {}

    def _ts(self, iso: str | None) -> float:
        try:
            return datetime.fromisoformat(iso).timestamp() if iso else self.clock()
        except ValueError:
            return self.clock()

    async def handle(self, topic: str, key: str | None, ev: dict[str, Any]) -> None:
        if topic != Topics.EMBEDDING_EVENTS:
            return
        t0 = time.perf_counter()
        vec = np.asarray(ev["embedding"], dtype=np.float32)
        if vec.shape[0] != self.s.embedding_dim:
            raise ValueError(f"embedding dim {vec.shape[0]} != configured {self.s.embedding_dim}")
        k = f"{ev['camera_id']}:{ev['object_id']}"
        ts = self._ts(ev.get("timestamp"))
        tr = self.tracks.get(k)
        if tr is None:
            tr = self.tracks[k] = TrackState(ev["camera_id"], ev["object_id"], ev.get("class_name"), ev["model_id"],
                                             ev.get("model_version"), np.zeros_like(vec), first_seen=ts)
        tr.sum_vec += vec
        tr.count += 1
        tr.last_seen = ts
        ANALYTICS.consumed.labels(topic).inc()
        if tr.count < self.s.mtmc_min_track_samples:
            return
        newly_assigned = False
        if tr.global_id is None:
            await self._assign(tr)
            newly_assigned = True
        meta = {"camera_id": tr.camera_id, "object_id": tr.object_id, "global_id": tr.global_id,
                "class_name": tr.class_name or "", "model_id": tr.model_id, "model_version": tr.model_version or "",
                "ts": int(tr.last_seen)}
        await asyncio.to_thread(self.store.upsert, tr.key, tr.mean(), meta)
        if newly_assigned:
            await self._persist(tr)
            self.bus.publish(Topics.REID_EVENTS, {
                "schema_version": 1, "event_id": new_id(), "event_type": "identity_assigned",
                "global_id": tr.global_id, "camera_id": tr.camera_id, "object_id": tr.object_id,
                "class_name": tr.class_name, "matched": tr.matched, "timestamp": now_iso()}, key=tr.global_id)
        ANALYTICS.reid_latency.observe(time.perf_counter() - t0)
        self._evict()

    async def _assign(self, tr: TrackState) -> None:
        min_ts = int(tr.last_seen - self.s.mtmc_time_window_s)
        hits = await asyncio.to_thread(self.store.search, tr.mean(), 5, exclude_camera=tr.camera_id,
                                       class_name=tr.class_name, min_ts=min_ts)
        for h in hits:
            if h.score < self.s.mtmc_match_threshold:
                break
            if h.meta.get("global_id") and self.filter.allow(tr, h.meta):
                tr.global_id = h.meta["global_id"]
                tr.matched = {"camera_id": h.meta.get("camera_id"), "object_id": h.meta.get("object_id"),
                              "similarity": round(h.score, 4)}
                ANALYTICS.reid_matches.labels("matched").inc()
                return
        tr.global_id = f"GID_{uuid.uuid4().hex[:12].upper()}"
        ANALYTICS.reid_matches.labels("new_identity").inc()

    async def _persist(self, tr: TrackState) -> None:
        if self.sm is None:
            return
        async with self.sm() as db:
            existing = (await db.execute(select(ReidAssociation).where(
                ReidAssociation.camera_id == tr.camera_id, ReidAssociation.object_id == tr.object_id))).scalar_one_or_none()
            now = datetime.fromtimestamp(tr.last_seen, tz=timezone.utc)
            if existing is None:
                db.add(ReidAssociation(
                    global_id=tr.global_id, camera_id=tr.camera_id, object_id=tr.object_id, model_id=tr.model_id,
                    similarity=(tr.matched or {}).get("similarity"),
                    matched_camera_id=(tr.matched or {}).get("camera_id"),
                    matched_object_id=(tr.matched or {}).get("object_id"),
                    first_seen=datetime.fromtimestamp(tr.first_seen, tz=timezone.utc), last_seen=now))
            else:
                existing.global_id, existing.last_seen = tr.global_id, now
            await db.execute(update(DetectionEvent).where(
                DetectionEvent.camera_id == tr.camera_id, DetectionEvent.object_id == tr.object_id
            ).values(global_id=tr.global_id))
            await db.commit()

    def _evict(self) -> None:
        if len(self.tracks) < 20_000:
            return
        cutoff = self.clock() - self.s.mtmc_time_window_s
        self.tracks = {k: t for k, t in self.tracks.items() if t.last_seen >= cutoff}

    def retention(self) -> int:
        return self.store.delete_older_than(int(self.clock() - self.s.embedding_retention_days * 86400))


async def main() -> None:
    from prometheus_client import start_http_server

    from analytics.vector_store import MilvusVectorStore
    from app.bus.kafka import KafkaBus, KafkaConsumerLoop, ensure_topics
    from app.core.logging import configure_logging
    from app.db.session import get_sessionmaker, init_engine, wait_for_db

    s = get_settings()
    configure_logging("mtmc", s.log_level, s.log_json)
    start_http_server(int(os.environ.get("METRICS_PORT", "9104")))
    init_engine(s)
    await wait_for_db()
    ensure_topics(s)
    store = MilvusVectorStore(s.milvus_uri, s.milvus_token, s.milvus_collection, s.embedding_dim,
                              ttl_s=s.embedding_retention_days * 86400)
    store.ensure_collection()
    bus = KafkaBus(s, client_id="mtmc")
    matcher = MTMCMatcher(store, s, bus, get_sessionmaker())
    # one consumer group; key=camera_id keeps per-camera ordering
    consumer = KafkaConsumerLoop([Topics.EMBEDDING_EVENTS], "mtmc", matcher.handle, bus, s)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    task = asyncio.create_task(consumer.run())
    while not stop.is_set():
        try:
            await asyncio.wait_for(stop.wait(), timeout=3600)
        except asyncio.TimeoutError:
            n = await asyncio.to_thread(matcher.retention)
            log.info("retention removed %d vectors", n)
    consumer.stop()
    task.cancel()
    consumer.close()
    bus.close()


if __name__ == "__main__":
    asyncio.run(main())
