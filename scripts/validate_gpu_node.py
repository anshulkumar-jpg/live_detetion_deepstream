"""GPU node validation - run ON the GPU server inside the worker container:

    docker compose run --rm gpu-agent python3 scripts/validate_gpu_node.py \
        --models config/models/models.yaml [--rtsp rtsp://user:pass@camera/stream] [--mediamtx-api http://mediamtx:9997]

Checks (each PASS/FAIL, exit code != 0 on any failure):
  1. NVML / GPU visibility, driver, NVDEC/NVENC counters
  2. DeepStream GStreamer plugins + pyds bindings + tracker configs
  3. Every registered model alone (engine build/load + inference + tracking) on a sample clip
  4. Multi-model routing (all detectors on one camera) and cascade (if a pose/reid/classifier model exists)
  5. Optional: live RTSP ingestion, annotated NVENC output reaching MediaMTX
Writes docs/gpu_validation_report.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SAMPLE = "file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4"
ELEMENTS = ["nvurisrcbin", "nvstreammux", "nvinfer", "nvtracker", "nvstreamdemux", "nvvideoconvert", "nvdsosd",
            "nvv4l2h264enc", "h264parse", "rtspclientsink", "fakesink"]
results: list[dict] = []


def check(name: str, ok: bool, detail=None) -> bool:
    results.append({"check": name, "ok": bool(ok), "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail is not None else ""))
    return ok


def run_pipeline(name, plan, uris, seconds, annotated=False):
    from scripts.gpu_harness import start_runner, stop_runner, stream_stats
    from scripts.localplan import worker_spec

    spec = worker_spec(f"VAL_{name}", plan, uris, annotated=annotated)
    h = start_runner(spec)
    t0 = time.monotonic()
    time.sleep(seconds)
    alive = h.proc.poll() is None
    st = stream_stats([h], t0 + seconds / 2)
    published = h.metrics[-1]["metrics"].get("events_published", 0) if h.metrics else 0
    code = stop_runner(h)
    ok = alive and st["streams"] == len(uris) and st["fps_min"] > 0 and not st["not_running"]
    return ok, {"fps": st["fps_avg"], "latency_ms": st["latency_avg_ms"], "events_published": published,
                "exit": code, "stderr_tail": [] if ok else h.stderr_tail[-8:]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--sample", default=SAMPLE)
    ap.add_argument("--rtsp")
    ap.add_argument("--mediamtx-api")
    ap.add_argument("--seconds", type=float, default=45, help="per test (first run includes TensorRT engine build)")
    args = ap.parse_args()

    # 1. GPU
    try:
        from app.services.gpu import NvmlCollector
        g = NvmlCollector().collect()
        check("NVML: GPUs visible", len(g) > 0, [f"{x['index']}:{x['name']} {x['memory_total_mb']:.0f}MB" for x in g])
        check("NVML: NVDEC/NVENC utilisation readable", g and g[0]["decoder_util_pct"] is not None)
    except Exception as e:
        check("NVML: GPUs visible", False, repr(e))

    # 2. DeepStream runtime
    try:
        import gi
        gi.require_version("Gst", "1.0")
        from gi.repository import Gst
        Gst.init(None)
        missing = [e for e in ELEMENTS if Gst.ElementFactory.find(e) is None]
        check("GStreamer/DeepStream elements present", not missing, f"missing: {missing}" if missing else "all present")
    except Exception as e:
        check("GStreamer available", False, repr(e))
    try:
        import pyds  # noqa: F401
        check("pyds bindings import", True)
    except Exception as e:
        check("pyds bindings import", False, repr(e))
    from app.core.config import get_settings
    from worker.pipeline.config_gen import TRACKER_CONFIG_FILES
    s = get_settings()
    for t, f in TRACKER_CONFIG_FILES.items():
        p = os.path.join(s.tracker_config_dir, f)
        check(f"tracker config {t}", os.path.exists(p), p)
    check("tracker library", os.path.exists(s.tracker_lib), s.tracker_lib)

    # 3./4. models
    from scripts.localplan import load_catalog, plan_for
    cat = load_catalog(args.models)
    models, versions = cat
    for mid, m in models.items():
        v = versions[m.active_version_id]
        for f in (v.onnx_file, v.engine_file):
            if f:
                path = f if os.path.isabs(f) else os.path.join(s.model_repository_dir, f)
                check(f"{mid}: file {f}", os.path.exists(path), path)
    detectors = [mid for mid, m in models.items() if m.task.value == "detector"]
    standalone = [mid for mid, m in models.items() if m.task.value not in ("classifier", "reid")]
    for mid in standalone:
        ok, d = run_pipeline(mid, plan_for(cat, [mid], batch=2), [args.sample], args.seconds)
        check(f"single-model pipeline: {mid}", ok, d)
    if len(standalone) > 1:
        ok, d = run_pipeline("multi", plan_for(cat, standalone, batch=2), [args.sample, args.sample], args.seconds)
        check(f"multi-model pipeline ({'+'.join(standalone)}), 2 cameras/worker", ok, d)
    secondary = [mid for mid, m in models.items() if m.task.value in ("pose", "classifier", "reid")]
    if detectors and secondary:
        ok, d = run_pipeline("cascade", plan_for(cat, [detectors[0], *secondary], cascade_parent=detectors[0], batch=2),
                             [args.sample], args.seconds)
        check(f"cascade pipeline {detectors[0]} -> {'+'.join(secondary)}", ok, d)
    for tracker in ("NvSORT", "IOU"):
        ok, d = run_pipeline(f"trk_{tracker}", plan_for(cat, [standalone[0]], tracker=tracker, batch=2), [args.sample], args.seconds / 2)
        check(f"tracker {tracker}", ok, d)

    # 5. live RTSP + MediaMTX output
    if args.rtsp:
        ok, d = run_pipeline("rtsp", plan_for(cat, [standalone[0]], batch=2), [args.rtsp], args.seconds)
        check("live RTSP ingestion", ok, d)
    if args.mediamtx_api:
        import threading

        import httpx
        res = {}

        def poll():
            time.sleep(args.seconds * 0.7)
            try:
                res["path"] = httpx.get(f"{args.mediamtx_api}/v3/paths/get/cam/VAL_out_S0/processed", timeout=5).json()
            except Exception as e:
                res["error"] = repr(e)
        t = threading.Thread(target=poll)
        t.start()
        ok, d = run_pipeline("out", plan_for(cat, [standalone[0]], batch=2), [args.sample], args.seconds, annotated=True)
        t.join()
        ready = bool(res.get("path", {}).get("ready"))
        check("annotated NVENC stream published to MediaMTX", ok and ready, {**d, "mediamtx": res})

    os.makedirs("docs", exist_ok=True)
    with open("docs/gpu_validation_report.json", "w") as f:
        json.dump(results, f, indent=1, default=str)
    failed = [r for r in results if not r["ok"]]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
