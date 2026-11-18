"""MilvusVectorStore against Milvus Lite (same pymilvus client/API path as a Milvus server)."""
import time

import numpy as np
import pytest

from analytics.mtmc import MTMCMatcher
from analytics.vector_store import MilvusVectorStore
from app.bus.kafka import InMemoryBus
from app.bus.schemas import Topics

pytest.importorskip("milvus_lite")


@pytest.fixture
def store(tmp_path):
    s = MilvusVectorStore(str(tmp_path / "milvus.db"), "", "emb", 8, ttl_s=3600)
    s.ensure_collection()
    s.ensure_collection()  # idempotent
    return s


def meta(cam, oid, gid, cls, ts):
    return {"camera_id": cam, "object_id": oid, "global_id": gid, "class_name": cls, "model_id": "reid",
            "model_version": "1", "ts": ts}


def test_milvus_insert_search_filter_retention(store):
    v = np.random.default_rng(1).normal(size=8)
    v /= np.linalg.norm(v)
    now = int(time.time())
    store.upsert("A:T1", v.tolist(), meta("A", "T1", "G1", "person", now))
    store.upsert("B:T2", (-v).tolist(), meta("B", "T2", "G2", "person", now))
    store.upsert("C:T3", v.tolist(), meta("C", "T3", "G3", "car", now - 10_000))
    store.upsert("A:T1", v.tolist(), meta("A", "T1", "G1", "person", now))  # upsert, not duplicate
    hits = store.search(v.tolist(), 5, exclude_camera="B", class_name="person", min_ts=now - 600)
    assert [h.id for h in hits] == ["A:T1"] and hits[0].score == pytest.approx(1.0, abs=1e-4)
    assert hits[0].meta["global_id"] == "G1"
    assert store.search(v.tolist(), 5, exclude_camera="A", class_name="person", min_ts=None)[0].id == "B:T2"
    store.delete_older_than(now - 600)
    assert sorted(h.id for h in store.search(v.tolist(), 10, exclude_camera=None, class_name=None, min_ts=None)) == ["A:T1", "B:T2"]


async def test_mtmc_with_milvus(store, settings):
    settings.embedding_dim = 8
    settings.mtmc_min_track_samples = 1
    m = MTMCMatcher(store, settings, InMemoryBus())
    v = list(np.eye(8)[0])
    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc).isoformat()
    base = {"worker_id": "W", "model_id": "reid", "class_name": "person", "timestamp": ts}
    await m.handle(Topics.EMBEDDING_EVENTS, None, {**base, "camera_id": "A", "object_id": "T1", "embedding": v})
    await m.handle(Topics.EMBEDDING_EVENTS, None, {**base, "camera_id": "B", "object_id": "T7", "embedding": v})
    assert m.tracks["A:T1"].global_id == m.tracks["B:T7"].global_id
