from datetime import datetime, timezone

import numpy as np
import pytest
from sqlalchemy import func, select

from analytics.mtmc import MTMCMatcher
from analytics.stream import AnalyticsStream
from analytics.vector_store import InMemoryVectorStore
from app.bus.schemas import Topics
from app.db.models import AnalyticsEvent, DetectionEvent, ReidAssociation

pytestmark = pytest.mark.integration


class Clock:
    t = 100.0

    def __call__(self):
        return self.t


def det(cam, oid, conf=0.8, frame=0):
    return {"event_id": f"e{frame}", "camera_id": cam, "worker_id": "W1", "model_id": "model_a", "class_id": 0,
            "class_name": "person", "confidence": conf, "object_id": oid,
            "bbox": {"x": 1, "y": 2, "width": 3, "height": 4}, "frame_num": frame,
            "timestamp": datetime.now(timezone.utc).isoformat()}


async def test_analytics_stream_track_lifecycle(sm, bus, settings):
    clock = Clock()
    svc = AnalyticsStream(settings, bus, sm, track_timeout_s=10, occupancy_interval_s=60, clock=clock)
    await svc.handle(Topics.DETECTION_EVENTS, "CAM_1", det("CAM_1", "TRACK_a_1", 0.7, 0))
    await svc.handle(Topics.DETECTION_EVENTS, "CAM_1", det("CAM_1", "TRACK_a_1", 0.95, 15))
    await svc.handle(Topics.DETECTION_EVENTS, "CAM_1", det("CAM_1", "TRACK_a_2", 0.6, 15))
    await svc.handle(Topics.DETECTION_EVENTS, "CAM_1", {**det("CAM_1", None), "object_id": None})  # untracked: ES only
    await svc.tick()
    await svc.tick()  # second flush with no changes must not duplicate
    async with sm() as db:
        rows = list((await db.execute(select(DetectionEvent).order_by(DetectionEvent.object_id))).scalars())
        assert [(r.object_id, r.sample_count, r.max_confidence) for r in rows] == [("TRACK_a_1", 2, 0.95), ("TRACK_a_2", 1, 0.6)]
    started = [e for e in bus.on(Topics.ANALYTICS_EVENTS) if e["event_type"] == "track_started"]
    assert len(started) == 2
    clock.t += 61
    stats = await svc.tick()
    assert stats["ended"] == 2
    ended = [e for e in bus.on(Topics.ANALYTICS_EVENTS) if e["event_type"] == "track_ended"]
    assert len(ended) == 2 and all("dwell_s" in e for e in ended)
    async with sm() as db:
        assert (await db.execute(select(func.count()).select_from(AnalyticsEvent))).scalar() == 2


async def test_analytics_occupancy(sm, bus, settings):
    clock = Clock()
    svc = AnalyticsStream(settings, bus, sm, track_timeout_s=1000, occupancy_interval_s=60, clock=clock)
    for i in range(3):
        await svc.handle(Topics.DETECTION_EVENTS, "CAM_2", det("CAM_2", f"T{i}", frame=i))
    clock.t += 60
    await svc.tick()
    occ = [e for e in bus.on(Topics.ANALYTICS_EVENTS) if e["event_type"] == "occupancy"]
    assert occ == [{**occ[0], "camera_id": "CAM_2", "class_name": "person", "active_tracks": 3}]


async def test_mtmc_persists_associations(sm, bus, settings):
    settings.embedding_dim = 4
    settings.mtmc_min_track_samples = 1
    async with sm() as db:
        now = datetime.now(timezone.utc)
        db.add(DetectionEvent(event_id="x1", camera_id="CAM_B", model_id="model_a", object_id="T9", class_id=0,
                              class_name="person", first_seen=now, last_seen=now))
        await db.commit()
    m = MTMCMatcher(InMemoryVectorStore(), settings, bus, sm)
    v = list(np.array([0.1, 0.9, 0.1, 0.0]))
    base = {"worker_id": "W", "model_id": "reid", "class_name": "person", "timestamp": datetime.now(timezone.utc).isoformat()}
    await m.handle(Topics.EMBEDDING_EVENTS, None, {**base, "camera_id": "CAM_A", "object_id": "T1", "embedding": v})
    await m.handle(Topics.EMBEDDING_EVENTS, None, {**base, "camera_id": "CAM_B", "object_id": "T9", "embedding": v})
    async with sm() as db:
        rows = list((await db.execute(select(ReidAssociation).order_by(ReidAssociation.camera_id))).scalars())
        assert len(rows) == 2 and rows[0].global_id == rows[1].global_id
        assert rows[1].matched_camera_id == "CAM_A" and rows[1].similarity > 0.99
        d = (await db.execute(select(DetectionEvent).where(DetectionEvent.object_id == "T9"))).scalar_one()
        assert d.global_id == rows[0].global_id
