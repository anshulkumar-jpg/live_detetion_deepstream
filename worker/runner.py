"""Worker process: runs one DeepStream pipeline group.

Launched by the node agent:  python -m worker.runner --spec /path/spec.json
* control in:  JSON lines on stdin  ({"type": "ADD_SOURCE"|"REMOVE_SOURCE"|"STOP", ...})
* status out:  JSON lines on stdout (metrics snapshots for the agent's Prometheus exporter)
* events out:  Kafka (pipeline.events, worker.heartbeat, pipeline.metrics, detection/embedding events)
* logs:        stderr (JSON)
Exit codes: 0 graceful stop, 2 model load failure, 3 other pipeline failure.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import threading
import time

import numpy as _np  # noqa: F401  (ensures numpy is present for pyds tensor access)

from app.bus.schemas import Envelope, Topics
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.core.security import get_cipher

log = logging.getLogger("worker")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    args = ap.parse_args()
    s = get_settings()
    with open(args.spec) as f:
        spec = json.load(f)
    wid = spec["worker_id"]
    gen = spec.get("spec_generation", 1)
    # logs to stderr; stdout is reserved for the agent protocol
    configure_logging(f"worker:{wid}", s.log_level, s.log_json)
    logging.getLogger().handlers[0].stream = sys.stderr

    from app.bus.kafka import InMemoryBus, KafkaBus
    from worker.pipeline.events import EventEmitter

    # KAFKA_ENABLED=false is used by scripts/benchmark.py and scripts/validate_gpu_node.py
    bus = KafkaBus(s, client_id=wid) if s.kafka_enabled else InMemoryBus(retain=False)
    cipher = get_cipher(s)
    stdout_lock = threading.Lock()

    def emit_stdout(obj: dict) -> None:
        with stdout_lock:
            sys.stdout.write(json.dumps(obj, default=str) + "\n")
            sys.stdout.flush()

    def pipeline_event(etype: str, payload: dict) -> None:
        env = Envelope(event_type=etype, source=wid, payload={"worker_id": wid, "node_id": s.node_id, **payload})
        try:
            bus.publish_reliable(Topics.PIPELINE_EVENTS, env.model_dump(), key=wid, timeout=5)
        except Exception as e:
            log.error("failed to publish pipeline event %s: %s", etype, e)

    gie_models = {st["gie_unique_id"]: {"model_id": st["model_id"], "model_version": st["model_version"],
                                         "task": st["task"]} for st in spec["plan"]["stages"]}
    emitter = EventEmitter(
        worker_id=wid, session_tag=format(int(time.time()), "x"), gie_models=gie_models,
        publish=lambda topic, payload, key: bus.publish(topic, payload, key=key),
        detection_interval=s.detection_event_interval_frames, embedding_interval=s.embedding_interval_frames,
        max_events_per_s=s.events_max_per_second_per_camera,
    )

    pipeline_event("WORKER_STARTING", {"spec_generation": gen})
    try:
        from worker.pipeline.deepstream_pipeline import GLib, PipelineError, PipelineGroup
    except Exception as e:  # DeepStream runtime not present
        pipeline_event("WORKER_FAILED", {"reason": f"DeepStream runtime unavailable: {e}"})
        bus.flush()
        return 3

    config_dir = os.path.join(s.deepstream_config_dir, wid, f"g{gen}")
    try:
        group = PipelineGroup(spec, s, decrypt=cipher.decrypt, emitter=emitter, on_event=pipeline_event,
                              config_dir=config_dir)
        group.start(spec.get("sources", []))
    except PipelineError as e:
        et = "MODEL_LOAD_FAILED" if e.model_id else "WORKER_FAILED"
        pipeline_event(et, {"reason": str(e), "model_id": e.model_id, "model_version_id": e.model_version_id})
        bus.flush()
        return 2 if e.model_id else 3

    stopping = threading.Event()

    def do_stop(reason: str) -> bool:
        stopping.set()
        group.stop()
        pipeline_event("WORKER_STOPPED", {"reason": reason})
        return False

    signal.signal(signal.SIGTERM, lambda *_: GLib.idle_add(do_stop, "SIGTERM"))
    signal.signal(signal.SIGINT, lambda *_: GLib.idle_add(do_stop, "SIGINT"))

    def stdin_reader() -> None:
        for line in sys.stdin:
            try:
                cmd = json.loads(line)
            except json.JSONDecodeError:
                continue
            ct = cmd.get("type")
            if ct == "ADD_SOURCE":
                def _add(src=cmd["source"]):
                    try:
                        group.add_source(src)
                    except PipelineError as e:
                        pipeline_event("SOURCE_FAILED", {"camera_id": src["camera_id"], "reason": str(e)})
                    return False
                GLib.idle_add(_add)
            elif ct == "REMOVE_SOURCE":
                GLib.idle_add(lambda c=cmd["camera_id"]: (group.remove_source(c, "remove requested"), False)[1])
            elif ct == "STOP":
                GLib.idle_add(do_stop, cmd.get("reason", "stop requested"))
                return
        # agent went away: stop
        if not stopping.is_set():
            GLib.idle_add(do_stop, "agent pipe closed")

    threading.Thread(target=stdin_reader, daemon=True).start()

    running_announced = {"v": False}

    def heartbeat() -> bool:
        sources = group.snapshot()
        statuses = [v["status"] for v in sources.values()]
        state = "RUNNING" if statuses and all(x == "RUNNING" for x in statuses) else (
            "DEGRADED" if any(x in ("RECONNECTING", "FAILED") for x in statuses) else "RUNNING")
        lats = [v["latency_ms"] for v in sources.values() if v["latency_ms"] is not None]
        metrics = {"avg_latency_ms": sum(lats) / len(lats) if lats else None,
                   "events_published": emitter.published, "events_dropped": emitter.dropped,
                   "bus_dropped": sum(bus.stats.dropped.values()), "sources": sources}
        hb = {"schema_version": 1, "kind": "worker", "node_id": s.node_id, "worker_id": wid, "state": state,
              "pid": os.getpid(), "spec_generation": gen, "sources": sources, "metrics": metrics,
              "timestamp": Envelope(event_type="x", source="x").timestamp}
        bus.publish(Topics.WORKER_HEARTBEAT, hb, key=wid)
        bus.publish(Topics.PIPELINE_METRICS, {"worker_id": wid, **metrics, "timestamp": hb["timestamp"]}, key=wid)
        emit_stdout({"kind": "metrics", "worker_id": wid, "state": state, "metrics": metrics})
        if not running_announced["v"] and state == "RUNNING" and statuses:
            running_announced["v"] = True
            pipeline_event("WORKER_RUNNING", {"spec_generation": gen})
        return not stopping.is_set()

    GLib.timeout_add(int(s.heartbeat_interval_s * 1000), heartbeat)
    group.run()  # blocks until stop or fatal error

    if group.fatal is not None and not stopping.is_set():
        e = group.fatal
        group.stop()
        et = "MODEL_LOAD_FAILED" if e.model_id else "WORKER_FAILED"
        pipeline_event(et, {"reason": str(e), "model_id": e.model_id, "model_version_id": e.model_version_id})
        bus.flush(5)
        return 2 if e.model_id else 3
    bus.flush(5)
    return 0


if __name__ == "__main__":
    sys.exit(main())
