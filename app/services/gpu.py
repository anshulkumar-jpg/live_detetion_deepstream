"""GPU Resource Manager: NVML telemetry collection + admission control.

Admission control never divides the GPU into fixed slices. It uses:
  1. live NVML measurements (utilisation, memory, NVDEC, NVENC, temperature),
  2. measured per-stream cost from benchmark capacity profiles when available,
  3. configured safety thresholds,
and also counts assignments that were admitted but have not yet shown up in the
measurements (so a burst of starts cannot overshoot).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from app.core.config import Settings

log = logging.getLogger(__name__)


# ------------------------------------------------------------- NVML collector
class NvmlCollector:
    """Reads GPU telemetry through NVML (nvidia-ml-py). Raises at init if NVML is unavailable."""

    def __init__(self) -> None:
        import pynvml

        self.nv = pynvml
        pynvml.nvmlInit()
        self.driver = self._s(pynvml.nvmlSystemGetDriverVersion())

    @staticmethod
    def _s(v: Any) -> str:
        return v.decode() if isinstance(v, bytes) else str(v)

    def _safe(self, fn, *args, default=None):
        try:
            return fn(*args)
        except self.nv.NVMLError:
            return default

    def collect(self) -> list[dict[str, Any]]:
        nv = self.nv
        out = []
        for i in range(nv.nvmlDeviceGetCount()):
            h = nv.nvmlDeviceGetHandleByIndex(i)
            mem = nv.nvmlDeviceGetMemoryInfo(h)
            util = self._safe(nv.nvmlDeviceGetUtilizationRates, h)
            dec = self._safe(nv.nvmlDeviceGetDecoderUtilization, h)
            enc = self._safe(nv.nvmlDeviceGetEncoderUtilization, h)
            power = self._safe(nv.nvmlDeviceGetPowerUsage, h)
            power_limit = self._safe(nv.nvmlDeviceGetEnforcedPowerLimit, h)
            procs = self._safe(nv.nvmlDeviceGetComputeRunningProcesses, h, default=[]) or []
            out.append({
                "index": i,
                "uuid": self._s(nv.nvmlDeviceGetUUID(h)),
                "name": self._s(nv.nvmlDeviceGetName(h)),
                "driver_version": self.driver,
                "memory_total_mb": mem.total / 2**20,
                "memory_used_mb": mem.used / 2**20,
                "memory_free_mb": mem.free / 2**20,
                "gpu_util_pct": util.gpu if util else None,
                "mem_bw_util_pct": util.memory if util else None,
                "decoder_util_pct": dec[0] if dec else None,
                "encoder_util_pct": enc[0] if enc else None,
                "temperature_c": self._safe(nv.nvmlDeviceGetTemperature, h, nv.NVML_TEMPERATURE_GPU),
                "power_w": power / 1000.0 if power is not None else None,
                "power_limit_w": power_limit / 1000.0 if power_limit is not None else None,
                "processes": [
                    {"pid": p.pid, "used_memory_mb": (p.usedGpuMemory or 0) / 2**20} for p in procs
                ],
                "collected_at": datetime.now(timezone.utc).isoformat(),
            })
        return out

    def close(self) -> None:
        try:
            self.nv.nvmlShutdown()
        except Exception:
            pass


# ------------------------------------------------------------- admission
@dataclass
class GpuSnapshot:
    device_id: int
    name: str
    healthy: bool
    metrics: dict[str, Any]
    last_seen_age_s: float
    active_streams: int          # assignments currently counted against this GPU
    unconfirmed_streams: int = 0  # admitted but not yet RUNNING (not in measurements yet)
    avg_inference_latency_ms: float | None = None


@dataclass
class StreamCost:
    memory_mb: float
    max_streams: int | None = None  # measured capacity for this pipeline signature on this GPU type
    source: str = "default"


@dataclass
class AdmissionDecision:
    admitted: bool
    reason: str
    headroom: float = 0.0
    checks: dict[str, Any] = field(default_factory=dict)


def admit(gpu: GpuSnapshot, cost: StreamCost, s: Settings) -> AdmissionDecision:
    m = gpu.metrics or {}
    checks: dict[str, Any] = {}

    def deny(reason: str) -> AdmissionDecision:
        return AdmissionDecision(False, reason, 0.0, checks)

    # 1. health / freshness
    if not gpu.healthy:
        return deny("gpu_unhealthy")
    if gpu.last_seen_age_s > s.gpu_metrics_stale_s:
        return deny(f"gpu_metrics_stale ({gpu.last_seen_age_s:.0f}s)")
    temp = m.get("temperature_c")
    checks["temperature_c"] = temp
    if temp is not None and temp >= s.gpu_max_temperature_c:
        return deny(f"gpu_temperature {temp}C >= {s.gpu_max_temperature_c}C")

    # 2. measured capacity limits
    cap = cost.max_streams or (s.default_max_streams_per_gpu or None)
    checks["max_streams"] = cap
    checks["active_streams"] = gpu.active_streams
    if cap is not None and gpu.active_streams + 1 > cap:
        return deny(f"stream_capacity_reached ({gpu.active_streams}/{cap}, source={cost.source})")

    # 3. memory (projected = measured + admitted-but-not-yet-measured + this stream)
    total = m.get("memory_total_mb") or 0
    used = m.get("memory_used_mb") or 0
    projected = used + gpu.unconfirmed_streams * cost.memory_mb + cost.memory_mb
    checks.update(memory_used_mb=used, memory_projected_mb=projected, memory_total_mb=total)
    if total:
        if projected > total - s.gpu_memory_reserve_mb:
            return deny(f"insufficient_gpu_memory (projected {projected:.0f}MB of {total:.0f}MB, reserve {s.gpu_memory_reserve_mb}MB)")
        if projected / total * 100 > s.gpu_max_memory_pct:
            return deny(f"gpu_memory_threshold ({projected / total * 100:.1f}% > {s.gpu_max_memory_pct}%)")

    # 4. engines: compute, NVDEC, NVENC utilisation
    for key, limit, label in (
        ("gpu_util_pct", s.gpu_max_utilization_pct, "gpu_utilization"),
        ("decoder_util_pct", s.gpu_max_decoder_pct, "nvdec_utilization"),
        ("encoder_util_pct", s.gpu_max_encoder_pct, "nvenc_utilization"),
    ):
        v = m.get(key)
        checks[key] = v
        if v is not None and v >= limit:
            return deny(f"{label} {v}% >= {limit}%")

    # 5. processing latency reported by running workers
    lat = gpu.avg_inference_latency_ms
    checks["avg_latency_ms"] = lat
    if lat is not None and lat >= s.gpu_max_inference_latency_ms:
        return deny(f"pipeline_latency {lat:.0f}ms >= {s.gpu_max_inference_latency_ms}ms")

    # headroom score for choosing among GPUs (higher is better)
    mem_head = 1 - (projected / total) if total else 0.5
    util_head = 1 - (m.get("gpu_util_pct") or 0) / 100
    dec_head = 1 - (m.get("decoder_util_pct") or 0) / 100
    headroom = min(mem_head, util_head, dec_head)
    return AdmissionDecision(True, "ok", headroom, checks)
