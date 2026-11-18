"""Prometheus metrics. Each process exposes its own registry on /metrics
(API) or a dedicated port (scheduler, agent, workers, analytics)."""
from __future__ import annotations

from types import SimpleNamespace

from prometheus_client import Counter, Gauge, Histogram

SCHED = SimpleNamespace(
    cameras=Gauge("vms_cameras", "Cameras by status", ["status"]),
    workers=Gauge("vms_workers", "Workers by state", ["state"]),
    commands=Counter("vms_scheduler_commands_total", "Commands issued", ["type"]),
    workers_created=Counter("vms_workers_created_total", "Workers created"),
    worker_failures=Counter("vms_worker_failures_total", "Workers marked FAILED"),
    reconcile_seconds=Histogram("vms_reconcile_seconds", "Reconcile pass duration"),
    is_leader=Gauge("vms_scheduler_leader", "1 if this scheduler holds the leader lock"),
)

GPU = SimpleNamespace(
    util=Gauge("vms_gpu_utilization_pct", "GPU utilisation", ["node", "gpu", "name"]),
    mem_used=Gauge("vms_gpu_memory_used_mb", "GPU memory used", ["node", "gpu", "name"]),
    mem_total=Gauge("vms_gpu_memory_total_mb", "GPU memory total", ["node", "gpu", "name"]),
    dec=Gauge("vms_gpu_nvdec_utilization_pct", "NVDEC utilisation", ["node", "gpu", "name"]),
    enc=Gauge("vms_gpu_nvenc_utilization_pct", "NVENC utilisation", ["node", "gpu", "name"]),
    temp=Gauge("vms_gpu_temperature_c", "GPU temperature", ["node", "gpu", "name"]),
    power=Gauge("vms_gpu_power_w", "GPU power draw", ["node", "gpu", "name"]),
    proc_mem=Gauge("vms_worker_gpu_memory_mb", "GPU memory used by worker process", ["node", "worker"]),
)

PIPE = SimpleNamespace(
    input_fps=Gauge("vms_camera_input_fps", "Configured camera input FPS", ["camera", "worker"]),
    fps=Gauge("vms_camera_processing_fps", "Measured processing FPS", ["camera", "worker"]),
    dropped=Counter("vms_camera_dropped_frames_total", "Frames missing in frame number sequence", ["camera", "worker"]),
    latency=Gauge("vms_pipeline_latency_ms", "Capture-to-metadata latency (avg over window)", ["camera", "worker"]),
    rtsp_failures=Counter("vms_rtsp_failures_total", "RTSP disconnect/reconnect events", ["camera"]),
    events_dropped=Counter("vms_events_dropped_total", "Events dropped by rate limit/backpressure", ["worker"]),
    model_load_failures=Counter("vms_model_load_failures_total", "Model engine load failures", ["model"]),
    worker_restarts=Counter("vms_worker_process_restarts_total", "Worker processes restarted by agent", ["node"]),
)

ANALYTICS = SimpleNamespace(
    consumed=Counter("vms_analytics_events_consumed_total", "Events consumed", ["topic"]),
    reid_latency=Histogram("vms_reid_latency_seconds", "Embedding -> match decision latency",
                           buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5)),
    reid_matches=Counter("vms_reid_matches_total", "Cross-camera matches", ["result"]),
    kafka_lag=Gauge("vms_kafka_consumer_lag", "Consumer lag", ["group", "topic"]),
)

STREAM = SimpleNamespace(
    ready=Gauge("vms_stream_ready", "1 if MediaMTX path is ready", ["camera", "kind"]),
    viewers=Gauge("vms_stream_viewers", "Readers on MediaMTX path", ["camera", "kind"]),
)

API = SimpleNamespace(
    requests=Counter("vms_api_requests_total", "API requests", ["method", "route", "status"]),
    latency=Histogram("vms_api_request_seconds", "API latency", ["method", "route"]),
)
