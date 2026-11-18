"""Node agent process management (real subprocesses) and MTMC matching."""
import subprocess
import sys
import textwrap
import time
from datetime import datetime, timezone

import numpy as np
import pytest

from analytics.mtmc import MTMCMatcher
from analytics.vector_store import InMemoryVectorStore
from app.bus.kafka import InMemoryBus
from app.bus.schemas import Envelope, Topics
from worker.agent import NodeAgent

# Stand-in for worker.runner that speaks the same stdin/stdout protocol (no GPU here).
FAKE_RUNNER = textwrap.dedent("""
    import json, sys
    sources = set()
    for line in sys.stdin:
        cmd = json.loads(line)
        if cmd["type"] == "ADD_SOURCE":
            sources.add(cmd["source"]["camera_id"])
        elif cmd["type"] == "REMOVE_SOURCE":
            sources.discard(cmd["camera_id"])
        elif cmd["type"] == "CRASH":
            sys.exit(9)
        elif cmd["type"] == "STOP":
            sys.exit(0)
        print(json.dumps({"kind": "metrics", "metrics": {"sources": {c: {"fps": 30, "input_fps": 30} for c in sorted(sources)}}}), flush=True)
""")


def spawn(_wid, _spec):
    return subprocess.Popen([sys.executable, "-c", FAKE_RUNNER], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            text=True, bufsize=1)


def cmd(ctype, **payload):
    return Envelope(event_type=ctype, source="test", payload={"node_id": "gpu-node-1", **payload}).model_dump()


def wait_for(fn, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        if fn():
            return True
        time.sleep(0.05)
    return False


async def test_agent_lifecycle_and_crash_detection(settings):
    bus = InMemoryBus()
    agent = NodeAgent(settings, bus, spawn=spawn)
    await agent.handle_command(Topics.PIPELINE_COMMANDS, None,
                               cmd("START_WORKER", worker_id="W1", spec={"spec_generation": 1}))
    assert "W1" in agent.workers
    await agent.handle_command(Topics.PIPELINE_COMMANDS, None,
                               cmd("ADD_SOURCE", worker_id="W1", source={"camera_id": "CAM_1"}))
    assert wait_for(lambda: "CAM_1" in (agent.workers["W1"].last_metrics.get("sources") or {}))
    # duplicate START with same generation is idempotent (no second process)
    pid = agent.workers["W1"].proc.pid
    await agent.handle_command(Topics.PIPELINE_COMMANDS, None,
                               cmd("START_WORKER", worker_id="W1", spec={"spec_generation": 1}))
    assert agent.workers["W1"].proc.pid == pid
    # command for another node is ignored
    other = cmd("STOP_WORKER", worker_id="W1")
    other["payload"]["node_id"] = "other-node"
    await agent.handle_command(Topics.PIPELINE_COMMANDS, None, other)
    assert agent.workers["W1"].proc.poll() is None
    # ADD_SOURCE to an unknown worker -> SOURCE_FAILED so the scheduler reschedules
    await agent.handle_command(Topics.PIPELINE_COMMANDS, None,
                               cmd("ADD_SOURCE", worker_id="NOPE", source={"camera_id": "CAM_9"}))
    assert any(e["event_type"] == "SOURCE_FAILED" for e in bus.on(Topics.PIPELINE_EVENTS))
    # unexpected crash -> WORKER_FAILED
    agent._send(agent.workers["W1"], {"type": "CRASH"})
    assert wait_for(lambda: any(e["event_type"] == "WORKER_FAILED" for e in bus.on(Topics.PIPELINE_EVENTS)))
    assert "W1" not in agent.workers


async def test_agent_graceful_stop_and_stale_command(settings):
    bus = InMemoryBus()
    agent = NodeAgent(settings, bus, spawn=spawn)
    await agent.handle_command(Topics.PIPELINE_COMMANDS, None, cmd("START_WORKER", worker_id="W2", spec={}))
    proc = agent.workers["W2"].proc
    await agent.handle_command(Topics.PIPELINE_COMMANDS, None, cmd("STOP_WORKER", worker_id="W2"))
    assert wait_for(lambda: proc.poll() is not None)
    assert proc.returncode == 0
    assert not any(e["event_type"] == "WORKER_FAILED" for e in bus.on(Topics.PIPELINE_EVENTS))
    stale = cmd("START_WORKER", worker_id="W3", spec={})
    stale["timestamp"] = "2020-01-01T00:00:00+00:00"
    await agent.handle_command(Topics.PIPELINE_COMMANDS, None, stale)
    assert "W3" not in agent.workers


# ---------------------------------------------------------------- MTMC
def emb_event(cam, oid, vec, cls="person"):
    return {"camera_id": cam, "worker_id": "W", "model_id": "reid", "model_version": "1", "object_id": oid,
            "class_name": cls, "embedding": list(map(float, vec)), "timestamp": datetime.now(timezone.utc).isoformat()}


def unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


async def test_mtmc_matches_across_cameras(settings):
    settings.embedding_dim = 8
    settings.mtmc_min_track_samples = 2
    bus, store = InMemoryBus(), InMemoryVectorStore()
    m = MTMCMatcher(store, settings, bus)
    rng = np.random.default_rng(0)
    alice, bob = unit(rng.normal(size=8)), unit(rng.normal(size=8))
    for _ in range(3):
        await m.handle(Topics.EMBEDDING_EVENTS, None, emb_event("CAM_A", "T1", unit(alice + 0.05 * rng.normal(size=8))))
    await m.handle(Topics.EMBEDDING_EVENTS, None, emb_event("CAM_A", "T2", unit(bob + 0.05 * rng.normal(size=8))))
    await m.handle(Topics.EMBEDDING_EVENTS, None, emb_event("CAM_A", "T2", unit(bob + 0.05 * rng.normal(size=8))))
    for _ in range(2):  # alice re-appears on camera B
        await m.handle(Topics.EMBEDDING_EVENTS, None, emb_event("CAM_B", "T9", unit(alice + 0.05 * rng.normal(size=8))))
    gid_a = m.tracks["CAM_A:T1"].global_id
    assert gid_a and m.tracks["CAM_B:T9"].global_id == gid_a
    assert m.tracks["CAM_A:T2"].global_id != gid_a
    assert m.tracks["CAM_B:T9"].matched["camera_id"] == "CAM_A"
    reid = bus.on(Topics.REID_EVENTS)
    assert len(reid) == 3 and {e["global_id"] for e in reid} == {gid_a, m.tracks["CAM_A:T2"].global_id}


async def test_mtmc_never_matches_same_camera_or_other_class(settings):
    settings.embedding_dim = 4
    settings.mtmc_min_track_samples = 1
    m = MTMCMatcher(InMemoryVectorStore(), settings, InMemoryBus())
    v = [1, 0, 0, 0]
    await m.handle(Topics.EMBEDDING_EVENTS, None, emb_event("CAM_A", "T1", v))
    await m.handle(Topics.EMBEDDING_EVENTS, None, emb_event("CAM_A", "T2", v))
    await m.handle(Topics.EMBEDDING_EVENTS, None, emb_event("CAM_B", "T3", v, cls="car"))
    gids = {t.global_id for t in m.tracks.values()}
    assert len(gids) == 3
    with pytest.raises(ValueError):
        await m.handle(Topics.EMBEDDING_EVENTS, None, emb_event("CAM_A", "T4", [1, 0]))
