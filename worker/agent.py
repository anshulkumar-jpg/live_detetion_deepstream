"""Node agent: one per GPU server.

* Publishes node heartbeats with NVML telemetry for every GPU (GPU Resource Manager input).
* Consumes ``pipeline.commands`` for this node and runs each logical worker as an
  isolated ``worker.runner`` subprocess (crash isolation: one pipeline crashing
  never takes down the others).
* Forwards ADD/REMOVE_SOURCE to the right worker over its stdin.
* Detects unexpected worker exits and reports WORKER_FAILED immediately.
* Exposes Prometheus metrics on :9102 (GPU, per-camera FPS/latency, per-worker GPU memory).

Run: python -m worker.agent
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from prometheus_client import start_http_server

from app.bus.kafka import KafkaBus, KafkaConsumerLoop, ensure_topics
from app.bus.schemas import Envelope, PipelineCommandType, Topics
from app.core.command_auth import ReplayGuard, verify_command
from app.core.config import Settings, get_settings
from app.core.logging import configure_logging
from app.core.pipeline_guard import check_source, check_spec
from app.core.metrics import GPU, PIPE

log = logging.getLogger("agent")


@dataclass
class WorkerProc:
    worker_id: str
    spec: dict[str, Any]
    proc: subprocess.Popen
    started: float = field(default_factory=time.monotonic)
    stop_requested: bool = False
    last_metrics: dict[str, Any] = field(default_factory=dict)
    _prev: dict[str, float] = field(default_factory=dict)

    def delta(self, key: str, value: float) -> float:
        """Turn cumulative worker counters into Prometheus counter increments."""
        prev = self._prev.get(key, 0.0)
        self._prev[key] = value
        return value - prev if value >= prev else value


class NodeAgent:
    def __init__(self, settings: Settings, bus, spawn=None, collector=None):
        self.s = settings
        self.bus = bus
        self.workers: dict[str, WorkerProc] = {}
        self.lock = threading.RLock()
        self.spawn = spawn or self._spawn_process
        self.collector = collector
        self.max_command_age_s = 300
        self.replay = ReplayGuard()
        self._last_reject_event = 0.0

    # ------------------------------------------------------------ events
    def _event(self, etype: str, payload: dict[str, Any]) -> None:
        env = Envelope(event_type=etype, source=f"agent:{self.s.node_id}", payload={"node_id": self.s.node_id, **payload})
        try:
            self.bus.publish_reliable(Topics.PIPELINE_EVENTS, env.model_dump(), key=payload.get("worker_id") or self.s.node_id)
        except Exception as e:
            log.error("failed to publish %s: %s", etype, e)

    # ------------------------------------------------------------ processes
    def _spawn_process(self, worker_id: str, spec: dict[str, Any]) -> subprocess.Popen:
        d = os.path.join(self.s.deepstream_config_dir, worker_id)
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f"spec_g{spec.get('spec_generation', 1)}.json")
        with open(path, "w") as f:
            json.dump(spec, f)
        os.chmod(path, 0o600)  # contains encrypted credentials only, still keep private
        return subprocess.Popen([sys.executable, "-m", "worker.runner", "--spec", path],
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None, text=True, bufsize=1)

    def start_worker(self, worker_id: str, spec: dict[str, Any]) -> None:
        with self.lock:
            existing = self.workers.get(worker_id)
            if existing and existing.proc.poll() is None:
                if spec.get("spec_generation", 1) <= existing.spec.get("spec_generation", 1):
                    log.info("START_WORKER ignored: already running", extra={"worker_id": worker_id})
                    return
                self.stop_worker(worker_id, "restart with new spec", wait=True)
            proc = self.spawn(worker_id, spec)
            wp = WorkerProc(worker_id, spec, proc)
            self.workers[worker_id] = wp
        threading.Thread(target=self._monitor, args=(wp,), daemon=True, name=f"mon-{worker_id}").start()
        log.info("worker process started pid=%s", proc.pid, extra={"worker_id": worker_id})

    def _send(self, wp: WorkerProc, obj: dict[str, Any]) -> bool:
        try:
            wp.proc.stdin.write(json.dumps(obj) + "\n")
            wp.proc.stdin.flush()
            return True
        except (BrokenPipeError, OSError, ValueError):
            return False

    def stop_worker(self, worker_id: str, reason: str, wait: bool = False, grace_s: float = 15.0) -> None:
        with self.lock:
            wp = self.workers.get(worker_id)
        if wp is None or wp.proc.poll() is not None:
            # idempotent: report stopped so the scheduler can finalize
            self._event("WORKER_STOPPED", {"worker_id": worker_id, "reason": f"not running on node ({reason})"})
            return
        wp.stop_requested = True
        self._send(wp, {"type": "STOP", "reason": reason})

        def _ensure_dead():
            try:
                wp.proc.wait(timeout=grace_s)
            except subprocess.TimeoutExpired:
                wp.proc.terminate()
                try:
                    wp.proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    wp.proc.kill()
        if wait:
            _ensure_dead()
        else:
            threading.Thread(target=_ensure_dead, daemon=True).start()

    def _monitor(self, wp: WorkerProc) -> None:
        for line in wp.proc.stdout:  # agent protocol: JSON lines
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if msg.get("kind") == "metrics":
                wp.last_metrics = msg.get("metrics", {})
                dropped = (wp.last_metrics.get("events_dropped") or 0) + (wp.last_metrics.get("bus_dropped") or 0)
                PIPE.events_dropped.labels(wp.worker_id).inc(max(0.0, wp.delta("events_dropped", dropped)))
                for cam, info in (wp.last_metrics.get("sources") or {}).items():
                    PIPE.dropped.labels(cam, wp.worker_id).inc(max(0.0, wp.delta(f"deficit:{cam}", info.get("deficit_frames") or 0)))
                    PIPE.fps.labels(cam, wp.worker_id).set(info.get("fps") or 0)
                    if info.get("input_fps"):
                        PIPE.input_fps.labels(cam, wp.worker_id).set(info["input_fps"])
                    if info.get("latency_ms") is not None:
                        PIPE.latency.labels(cam, wp.worker_id).set(info["latency_ms"])
        code = wp.proc.wait()
        with self.lock:
            if self.workers.get(wp.worker_id) is wp:
                del self.workers[wp.worker_id]
        for cam in (wp.last_metrics.get("sources") or {}):
            for g in (PIPE.fps, PIPE.latency, PIPE.input_fps):
                try:
                    g.remove(cam, wp.worker_id)
                except KeyError:
                    pass
        if wp.stop_requested and code == 0:
            log.info("worker exited cleanly", extra={"worker_id": wp.worker_id})
            return
        if wp.stop_requested:
            self._event("WORKER_STOPPED", {"worker_id": wp.worker_id, "reason": f"stopped (exit {code})"})
            return
        # unexpected exit: the runner may already have published a specific reason (exit 2/3)
        PIPE.worker_restarts.labels(self.s.node_id).inc()
        log.error("worker exited unexpectedly code=%s", code, extra={"worker_id": wp.worker_id})
        if code not in (2, 3):
            self._event("WORKER_FAILED", {"worker_id": wp.worker_id, "reason": f"process exited with code {code}"})

    # ------------------------------------------------------------ commands
    def _reject(self, env: dict[str, Any], reason: str) -> None:
        """Refuse a command: log it, and tell the scheduler (at most once a second, so a flood cannot flood us)."""
        log.warning("REJECTED %s: %s", env.get("event_type"), reason)
        now = time.monotonic()
        if now - self._last_reject_event > 1.0:
            self._last_reject_event = now
            self._event("COMMAND_REJECTED", {"command": env.get("event_type"), "reason": reason[:300]})

    async def handle_command(self, topic: str, key: str | None, env: dict[str, Any]) -> None:
        p = env.get("payload", {})
        if p.get("node_id") != self.s.node_id:
            return
        if not self.s.allow_unsigned_commands:
            ok, why = verify_command(env, self.s.command_signing_key, self.s.node_id)
            if not ok:
                return self._reject(env, why)
            if not self.replay.first_time(env):
                return self._reject(env, "repeated command")
        ct0 = env.get("event_type")
        if ct0 in (PipelineCommandType.START_WORKER.value, PipelineCommandType.RESTART_WORKER.value):
            problems = check_spec(p.get("spec") or {}, self.s)
        elif ct0 == PipelineCommandType.ADD_SOURCE.value:
            problems = check_source(p.get("source") or {}, self.s.allow_file_sources)
        else:
            problems = []
        if problems:
            return self._reject(env, "; ".join(problems))
        try:
            age = (datetime.now(timezone.utc) - datetime.fromisoformat(env["timestamp"])).total_seconds()
            if age > self.max_command_age_s:
                log.warning("ignoring stale command %s (%.0fs old)", env.get("event_type"), age)
                return
        except (KeyError, ValueError):
            pass
        ct = env.get("event_type")
        wid = p.get("worker_id")
        loop = asyncio.get_running_loop()
        if ct == PipelineCommandType.START_WORKER.value:
            await loop.run_in_executor(None, self.start_worker, wid, p["spec"])
        elif ct == PipelineCommandType.RESTART_WORKER.value:
            await loop.run_in_executor(None, lambda: (self.stop_worker(wid, "restart", wait=True), self.start_worker(wid, p["spec"])))
        elif ct == PipelineCommandType.STOP_WORKER.value:
            await loop.run_in_executor(None, self.stop_worker, wid, p.get("reason", "stop"))
        elif ct in (PipelineCommandType.ADD_SOURCE.value, PipelineCommandType.REMOVE_SOURCE.value):
            with self.lock:
                wp = self.workers.get(wid)
            if ct == PipelineCommandType.ADD_SOURCE.value:
                if wp is None or not self._send(wp, {"type": "ADD_SOURCE", "source": p["source"]}):
                    self._event("SOURCE_FAILED", {"worker_id": wid, "camera_id": p["source"]["camera_id"],
                                                  "reason": "worker not running on node"})
            else:
                if wp is None or not self._send(wp, {"type": "REMOVE_SOURCE", "camera_id": p["camera_id"]}):
                    self._event("SOURCE_REMOVED", {"worker_id": wid, "camera_id": p["camera_id"],
                                                   "reason": "worker not running on node"})

    # ------------------------------------------------------------ telemetry
    def node_heartbeat(self) -> dict[str, Any]:
        gpus = self.collector.collect() if self.collector else []
        pid_to_worker = {}
        with self.lock:
            for wp in self.workers.values():
                pid_to_worker[wp.proc.pid] = wp.worker_id
        for g in gpus:
            lbl = (self.s.node_id, str(g["index"]), g["name"])
            GPU.util.labels(*lbl).set(g.get("gpu_util_pct") or 0)
            GPU.mem_used.labels(*lbl).set(g.get("memory_used_mb") or 0)
            GPU.mem_total.labels(*lbl).set(g.get("memory_total_mb") or 0)
            GPU.dec.labels(*lbl).set(g.get("decoder_util_pct") or 0)
            GPU.enc.labels(*lbl).set(g.get("encoder_util_pct") or 0)
            GPU.temp.labels(*lbl).set(g.get("temperature_c") or 0)
            GPU.power.labels(*lbl).set(g.get("power_w") or 0)
            g["worker_memory_mb"] = {}
            for pr in g.get("processes", []):
                w = pid_to_worker.get(pr["pid"])
                if w:
                    g["worker_memory_mb"][w] = pr["used_memory_mb"]
                    GPU.proc_mem.labels(self.s.node_id, w).set(pr["used_memory_mb"])
        hb = {"schema_version": 1, "kind": "node", "node_id": self.s.node_id, "gpus": gpus,
              "workers": sorted(pid_to_worker.values()), "timestamp": datetime.now(timezone.utc).isoformat()}
        self.bus.publish(Topics.WORKER_HEARTBEAT, hb, key=self.s.node_id)
        return hb

    def shutdown(self) -> None:
        with self.lock:
            ids = list(self.workers)
        for wid in ids:
            self.stop_worker(wid, "agent shutdown", wait=True)


async def main() -> None:
    s = get_settings()
    configure_logging(f"agent:{s.node_id}", s.log_level, s.log_json)
    if s.environment == "production":
        s.validate_for_production(needs_publish_token=True, needs_command_key=not s.allow_unsigned_commands)
    start_http_server(int(os.environ.get("METRICS_PORT", "9102")))
    from app.services.gpu import NvmlCollector

    try:
        collector = NvmlCollector()
    except Exception as e:
        log.error("NVML unavailable (%s): no GPU telemetry -> scheduler will not admit cameras on this node", e)
        collector = None
    ensure_topics(s)
    bus = KafkaBus(s, client_id=f"agent-{s.node_id}")
    agent = NodeAgent(s, bus, collector=collector)
    consumer = KafkaConsumerLoop([Topics.PIPELINE_COMMANDS], f"agent-{s.node_id}", agent.handle_command, bus, s,
                                 auto_offset_reset="earliest")
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    task = asyncio.create_task(consumer.run())
    while not stop.is_set():
        try:
            await loop.run_in_executor(None, agent.node_heartbeat)
        except Exception:
            log.exception("node heartbeat failed")
        try:
            await asyncio.wait_for(stop.wait(), timeout=s.heartbeat_interval_s)
        except asyncio.TimeoutError:
            pass
    consumer.stop()
    await loop.run_in_executor(None, agent.shutdown)
    task.cancel()
    consumer.close()
    bus.close()


if __name__ == "__main__":
    asyncio.run(main())
