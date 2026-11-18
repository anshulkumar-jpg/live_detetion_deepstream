"""Analytics Stream service (behavior analytics + track summaries).

Consumes ``detection.events`` and:
* keeps one aggregate per (camera, model, track) and periodically upserts it into
  PostgreSQL ``detection_events`` (track-level, NOT per frame),
* emits behavior events to ``analytics.events``:
    - ``track_started``  first sighting of a track
    - ``track_ended``    track unseen for ``track_timeout_s``; includes dwell time
    - ``occupancy``      per camera/class distinct active tracks, every ``occupancy_interval_s``
  ``track_ended`` is also stored in PostgreSQL ``analytics_events``; all analytics
  events flow to Elasticsearch through Logstash.

Run: python -m analytics.stream
"""
from __future__ import annotations

import asyncio
import os
import logging
import signal
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.bus.schemas import Topics, now_iso
from app.core.config import Settings, get_settings
from app.core.metrics import ANALYTICS
from app.db.models import AnalyticsEvent, DetectionEvent

log = logging.getLogger("analytics")


@dataclass
class TrackAgg:
    event_id: str
    camera_id: str
    worker_id: str | None
    model_id: str
    object_id: str
    class_id: int
    class_name: str | None
    max_confidence: float
    bbox: dict
    samples: int
    first_seen: datetime
    last_seen: datetime
    seen_mono: float
    dirty: bool = True


class AnalyticsStream:
    def __init__(self, settings: Settings, bus, sessionmaker, track_timeout_s: float = 10.0,
                 occupancy_interval_s: float = 60.0, clock=time.monotonic):
        self.s = settings
        self.bus = bus
        self.sm = sessionmaker
        self.timeout = track_timeout_s
        self.occ_iv = occupancy_interval_s
        self.clock = clock
        self.tracks: dict[tuple[str, str, str], TrackAgg] = {}
        self._last_occ = clock()

    @staticmethod
    def _dt(v: str | None) -> datetime:
        try:
            d = datetime.fromisoformat(v) if v else datetime.now(timezone.utc)
        except ValueError:
            d = datetime.now(timezone.utc)
        return d if d.tzinfo else d.replace(tzinfo=timezone.utc)

    async def handle(self, topic: str, key: str | None, ev: dict[str, Any]) -> None:
        if topic != Topics.DETECTION_EVENTS:
            return
        ANALYTICS.consumed.labels(topic).inc()
        oid = ev.get("object_id")
        if not oid:
            return  # untracked detections are only indexed in Elasticsearch
        k = (ev["camera_id"], ev["model_id"], oid)
        ts = self._dt(ev.get("timestamp"))
        a = self.tracks.get(k)
        if a is None:
            a = self.tracks[k] = TrackAgg(
                event_id=str(uuid.uuid5(uuid.NAMESPACE_URL, "/".join(k))), camera_id=ev["camera_id"],
                worker_id=ev.get("worker_id"), model_id=ev["model_id"], object_id=oid, class_id=int(ev["class_id"]),
                class_name=ev.get("class_name"), max_confidence=float(ev.get("confidence", 0)), bbox=ev.get("bbox", {}),
                samples=1, first_seen=ts, last_seen=ts, seen_mono=self.clock())
            self._emit("track_started", a, {})
        else:
            a.samples += 1
            a.max_confidence = max(a.max_confidence, float(ev.get("confidence", 0)))
            a.bbox = ev.get("bbox", a.bbox)
            a.last_seen = max(a.last_seen, ts)
            a.seen_mono = self.clock()
            a.dirty = True

    def _emit(self, etype: str, a: TrackAgg, extra: dict) -> dict:
        ev = {"schema_version": 1, "event_id": str(uuid.uuid4()), "event_type": etype, "camera_id": a.camera_id,
              "model_id": a.model_id, "object_id": a.object_id, "class_name": a.class_name,
              "timestamp": now_iso(), **extra}
        self.bus.publish(Topics.ANALYTICS_EVENTS, ev, key=a.camera_id)
        return ev

    async def tick(self) -> dict[str, int]:
        """Flush dirty aggregates, close timed-out tracks, publish occupancy."""
        now = self.clock()
        ended, ended_events = [], []
        for k, a in list(self.tracks.items()):
            if now - a.seen_mono > self.timeout:
                ended.append(a)
                del self.tracks[k]
                dwell = (a.last_seen - a.first_seen).total_seconds()
                ended_events.append(self._emit("track_ended", a, {
                    "dwell_s": round(dwell, 2), "first_seen": a.first_seen.isoformat(),
                    "last_seen": a.last_seen.isoformat(), "max_confidence": a.max_confidence, "samples": a.samples}))
        dirty = [a for a in self.tracks.values() if a.dirty] + ended
        if dirty or ended_events:
            await self._flush(dirty, ended_events)
        for a in dirty:
            a.dirty = False
        occ = 0
        if now - self._last_occ >= self.occ_iv:
            self._last_occ = now
            counts: dict[tuple[str, str], int] = defaultdict(int)
            for a in self.tracks.values():
                counts[(a.camera_id, a.class_name or "unknown")] += 1
            for (cam, cls), n in counts.items():
                self.bus.publish(Topics.ANALYTICS_EVENTS, {
                    "schema_version": 1, "event_id": str(uuid.uuid4()), "event_type": "occupancy",
                    "camera_id": cam, "class_name": cls, "active_tracks": n, "timestamp": now_iso()}, key=cam)
                occ += 1
        return {"flushed": len(dirty), "ended": len(ended), "occupancy": occ}

    async def _flush(self, aggs: list[TrackAgg], ended_events: list[dict]) -> None:
        async with self.sm() as db:
            pg = db.bind.dialect.name == "postgresql"
            for a in aggs:
                row = dict(event_id=a.event_id, camera_id=a.camera_id, worker_id=a.worker_id, model_id=a.model_id,
                           object_id=a.object_id, class_id=a.class_id, class_name=a.class_name,
                           max_confidence=a.max_confidence, last_bbox=a.bbox, sample_count=a.samples,
                           first_seen=a.first_seen, last_seen=a.last_seen)
                if pg:  # idempotent upsert (replays after a crash do not duplicate)
                    stmt = pg_insert(DetectionEvent).values(**row)
                    stmt = stmt.on_conflict_do_update(
                        constraint="uq_detection_track",
                        set_={"max_confidence": stmt.excluded.max_confidence, "last_bbox": stmt.excluded.last_bbox,
                              "sample_count": stmt.excluded.sample_count, "last_seen": stmt.excluded.last_seen})
                    await db.execute(stmt)
                else:
                    cur = (await db.execute(select(DetectionEvent).where(DetectionEvent.event_id == a.event_id))).scalar_one_or_none()
                    if cur is None:
                        db.add(DetectionEvent(**row))
                    else:
                        for f in ("max_confidence", "last_bbox", "sample_count", "last_seen"):
                            setattr(cur, f, row[f])
            for ev in ended_events:
                db.add(AnalyticsEvent(event_id=ev["event_id"], event_type=ev["event_type"], camera_id=ev["camera_id"],
                                      payload=ev, ts=datetime.now(timezone.utc)))
            await db.commit()


async def main() -> None:
    from prometheus_client import start_http_server

    from app.bus.kafka import KafkaBus, KafkaConsumerLoop, ensure_topics
    from app.core.logging import configure_logging
    from app.db.session import get_sessionmaker, init_engine, wait_for_db

    s = get_settings()
    configure_logging("analytics-stream", s.log_level, s.log_json)
    start_http_server(int(os.environ.get("METRICS_PORT", "9103")))
    init_engine(s)
    await wait_for_db()
    ensure_topics(s)
    bus = KafkaBus(s, client_id="analytics-stream")
    svc = AnalyticsStream(s, bus, get_sessionmaker())
    consumer = KafkaConsumerLoop([Topics.DETECTION_EVENTS], "analytics-stream", svc.handle, bus, s)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    task = asyncio.create_task(consumer.run())
    while not stop.is_set():
        try:
            await svc.tick()
        except Exception:
            log.exception("analytics flush failed (will retry)")
        try:
            await asyncio.wait_for(stop.wait(), timeout=1.0)
        except asyncio.TimeoutError:
            pass
    consumer.stop()
    await svc.tick()
    task.cancel()
    consumer.close()
    bus.close()


if __name__ == "__main__":
    asyncio.run(main())
