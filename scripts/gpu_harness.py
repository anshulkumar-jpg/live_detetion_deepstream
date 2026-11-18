"""Shared harness: run real worker.runner processes (DeepStream) and sample NVML while they run."""
from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RunnerHandle:
    worker_id: str
    proc: subprocess.Popen
    metrics: list[dict[str, Any]] = field(default_factory=list)
    stderr_tail: list[str] = field(default_factory=list)


def start_runner(spec: dict, env_overrides: dict[str, str] | None = None) -> RunnerHandle:
    d = tempfile.mkdtemp(prefix=f"{spec['worker_id']}-")
    path = os.path.join(d, "spec.json")
    with open(path, "w") as f:
        json.dump(spec, f)
    env = {**os.environ, "KAFKA_ENABLED": "false", "HEARTBEAT_INTERVAL_S": "2",
           "DEEPSTREAM_CONFIG_DIR": d, "LOG_JSON": "false", **(env_overrides or {})}
    p = subprocess.Popen([sys.executable, "-m", "worker.runner", "--spec", path], stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, env=env)
    h = RunnerHandle(spec["worker_id"], p)

    def read_out():
        for line in p.stdout:
            try:
                msg = json.loads(line)
                if msg.get("kind") == "metrics":
                    msg["t"] = time.monotonic()
                    h.metrics.append(msg)
            except json.JSONDecodeError:
                pass

    def read_err():
        for line in p.stderr:
            h.stderr_tail.append(line.rstrip())
            del h.stderr_tail[:-40]

    threading.Thread(target=read_out, daemon=True).start()
    threading.Thread(target=read_err, daemon=True).start()
    return h


def stop_runner(h: RunnerHandle, timeout: float = 15) -> int:
    try:
        h.proc.stdin.write(json.dumps({"type": "STOP"}) + "\n")
        h.proc.stdin.flush()
    except (BrokenPipeError, OSError):
        pass
    try:
        return h.proc.wait(timeout)
    except subprocess.TimeoutExpired:
        h.proc.kill()
        return h.proc.wait()


class NvmlSampler:
    def __init__(self, gpu_index: int = 0, period: float = 1.0):
        from app.services.gpu import NvmlCollector

        self.c = NvmlCollector()
        self.idx = gpu_index
        self.period = period
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            g = self.c.collect()[self.idx]
            g["t"] = time.monotonic()
            self.samples.append(g)
            self._stop.wait(self.period)

    def start(self):
        self._t.start()
        return self

    def stop(self):
        self._stop.set()
        self._t.join(timeout=5)

    def name(self) -> str:
        return self.c.collect()[self.idx]["name"]

    def summary(self, since: float) -> dict[str, float]:
        s = [x for x in self.samples if x["t"] >= since]
        if not s:
            return {}

        def agg(k, fn=statistics.mean):
            vals = [x[k] for x in s if x.get(k) is not None]
            return round(fn(vals), 1) if vals else None
        return {"gpu_util_avg": agg("gpu_util_pct"), "gpu_util_max": agg("gpu_util_pct", max),
                "mem_used_max_mb": agg("memory_used_mb", max), "mem_total_mb": s[0]["memory_total_mb"],
                "nvdec_avg": agg("decoder_util_pct"), "nvdec_max": agg("decoder_util_pct", max),
                "nvenc_avg": agg("encoder_util_pct"), "temp_max": agg("temperature_c", max),
                "power_avg_w": agg("power_w")}


def stream_stats(handles: list[RunnerHandle], since: float) -> dict[str, Any]:
    """Per-camera FPS/latency from the last metrics snapshot after `since`."""
    per = {}
    for h in handles:
        snaps = [m for m in h.metrics if m["t"] >= since]
        if not snaps:
            continue
        for cam, info in snaps[-1]["metrics"].get("sources", {}).items():
            per[cam] = {"fps": info.get("fps") or 0.0, "latency_ms": info.get("latency_ms"), "status": info.get("status")}
    fps = [v["fps"] for v in per.values()]
    lat = [v["latency_ms"] for v in per.values() if v["latency_ms"] is not None]
    return {"streams": len(per), "fps_min": round(min(fps), 2) if fps else 0, "fps_avg": round(statistics.mean(fps), 2) if fps else 0,
            "latency_avg_ms": round(statistics.mean(lat), 1) if lat else None,
            "latency_p95_ms": round(sorted(lat)[int(0.95 * (len(lat) - 1))], 1) if lat else None,
            "not_running": [c for c, v in per.items() if v["status"] != "RUNNING"], "per_stream": per}
