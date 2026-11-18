"""GPU capacity benchmark - run ON the GPU server inside the worker container.

Ramps the number of streams (2 per worker process, exactly like production) for a
given model set and records the highest count that still processes every stream
at >= --fps-ratio of the target FPS within GPU memory/latency limits.

  # real-time capacity: feed RTSP (e.g. a sample clip looped into MediaMTX with ffmpeg -re)
  python3 scripts/benchmark.py --models config/models/models.yaml --model-set person_detection \
      --uri rtsp://mediamtx:8554/bench/clip --start 2 --step 2 --max 64 --duration 60 \
      --report docs/benchmark_results.md --post-api https://vms:8443
  # cascade: --model-set person_detection,pose_detection --cascade-parent person_detection

A file:// URI measures raw throughput (decoder runs faster than real time); use RTSP for
capacity planning. Results are written as JSON + Markdown and optionally posted to
POST /api/v1/gpu/capacity-profiles so admission control uses measured limits.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.gpu_harness import NvmlSampler, start_runner, stop_runner, stream_stats  # noqa: E402
from scripts.localplan import load_catalog, plan_for, worker_spec  # noqa: E402


def run_level(n: int, plan, uri: str, args) -> dict:
    handles = []
    for w in range(0, n, args.per_worker):
        k = min(args.per_worker, n - w)
        spec = worker_spec(f"BENCH_{w // args.per_worker:03d}", plan, [uri] * k, gpu=args.gpu,
                           capacity=args.per_worker, annotated=args.annotated, fps=args.fps)
        handles.append(start_runner(spec))
    t0 = time.monotonic()
    sampler = NvmlSampler(args.gpu).start()
    time.sleep(args.warmup)
    steady = time.monotonic()
    time.sleep(args.duration)
    st = stream_stats(handles, steady)
    gpu = sampler.summary(steady)
    sampler.stop()
    crashed = [h.worker_id for h in handles if h.proc.poll() not in (None, 0)]
    errors = {h.worker_id: h.stderr_tail[-5:] for h in handles if h.proc.poll() not in (None, 0)}
    for h in handles:
        stop_runner(h)
    target = args.fps * args.fps_ratio
    ok = (not crashed and st["streams"] == n and st["fps_min"] >= target and not st["not_running"]
          and (gpu.get("mem_used_max_mb") or 0) <= (gpu.get("mem_total_mb") or 1) * args.max_mem_ratio
          and (st["latency_p95_ms"] is None or st["latency_p95_ms"] <= args.max_latency_ms))
    return {"streams": n, "ok": ok, "startup_s": round(steady - t0, 1), **{k: v for k, v in st.items() if k != "per_stream"},
            "gpu": gpu, "crashed_workers": crashed, "errors": errors}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--model-set", required=True, help="comma separated model ids")
    ap.add_argument("--cascade-parent")
    ap.add_argument("--tracker", default="NvDCF")
    ap.add_argument("--uri", default="file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--fps-ratio", type=float, default=0.95)
    ap.add_argument("--max-latency-ms", type=float, default=500)
    ap.add_argument("--max-mem-ratio", type=float, default=0.9)
    ap.add_argument("--per-worker", type=int, default=2)
    ap.add_argument("--start", type=int, default=2)
    ap.add_argument("--step", type=int, default=2)
    ap.add_argument("--max", type=int, default=64)
    ap.add_argument("--warmup", type=float, default=30, help="seconds (first run builds TensorRT engines: use more)")
    ap.add_argument("--duration", type=float, default=60)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--annotated", action="store_true", help="include OSD + NVENC output branch per stream")
    ap.add_argument("--report", default="docs/benchmark_results.md")
    ap.add_argument("--post-api")
    args = ap.parse_args()

    catalog = load_catalog(args.models)
    ids = [m.strip() for m in args.model_set.split(",")]
    plan = plan_for(catalog, ids, cascade_parent=args.cascade_parent, tracker=args.tracker, batch=args.per_worker)
    gpu_name = NvmlSampler(args.gpu).name()
    realtime = not args.uri.startswith("file://")
    print(f"GPU={gpu_name} signature={plan.signature} models={ids} realtime={realtime}")
    results, best = [], None
    n = args.start
    while n <= args.max:
        r = run_level(n, plan, args.uri, args)
        results.append(r)
        print(json.dumps({k: r[k] for k in ("streams", "ok", "fps_min", "fps_avg", "latency_p95_ms")} | {"gpu": r["gpu"]}))
        if not r["ok"]:
            break
        best = r
        n += args.step

    out = {"date": dt.datetime.now(dt.timezone.utc).isoformat(), "gpu_name": gpu_name, "model_set": ids,
           "cascade_parent": args.cascade_parent, "tracker": args.tracker, "pipeline_signature": plan.signature,
           "uri_kind": "rtsp/real-time" if realtime else "file/throughput", "target_fps": args.fps,
           "fps_ratio": args.fps_ratio, "annotated_output": args.annotated,
           "max_streams": best["streams"] if best else 0, "levels": results}
    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
    with open(os.path.splitext(args.report)[0] + ".json", "w") as f:
        json.dump(out, f, indent=1, default=str)
    lines = [f"## Benchmark {out['date']}", "", f"- GPU: **{gpu_name}**", f"- Models: {', '.join(ids)}"
             + (f" (cascade on {args.cascade_parent})" if args.cascade_parent else ""),
             f"- Source: {out['uri_kind']}  target {args.fps} FPS, pass >= {args.fps_ratio:.0%}",
             f"- Annotated NVENC output: {args.annotated}", f"- **Max sustained streams: {out['max_streams']}**", "",
             "| streams | ok | min FPS | avg FPS | p95 latency ms | GPU util avg | mem max MB | NVDEC avg | NVENC avg |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        g = r["gpu"]
        lines.append(f"| {r['streams']} | {'yes' if r['ok'] else 'NO'} | {r['fps_min']} | {r['fps_avg']} | {r['latency_p95_ms']} | "
                     f"{g.get('gpu_util_avg')} | {g.get('mem_used_max_mb')} | {g.get('nvdec_avg')} | {g.get('nvenc_avg')} |")
    with open(args.report, "a") as f:
        f.write("\n".join(lines) + "\n\n")
    print(f"max sustained streams: {out['max_streams']}  (report: {args.report})")

    if args.post_api and best and realtime:
        import getpass

        import httpx
        c = httpx.Client(base_url=args.post_api, timeout=30, verify=os.environ.get("VMS_CA", True))
        tok = c.post("/api/v1/auth/token", json={"username": os.environ.get("VMS_USER", "admin"),
                                                 "password": os.environ.get("VMS_PASSWORD") or getpass.getpass()}).json()
        c.headers["Authorization"] = f"Bearer {tok['access_token']}"
        mem = best["gpu"].get("mem_used_max_mb")
        r = c.post("/api/v1/gpu/capacity-profiles", json={
            "gpu_name": gpu_name, "pipeline_signature": plan.signature, "model_set": plan.model_set,
            "input_fps": args.fps, "max_streams": best["streams"],
            "memory_mb_per_stream": round(mem / best["streams"], 1) if mem else None,
            "avg_fps_per_stream": best["fps_avg"], "p95_latency_ms": best["latency_p95_ms"],
            "notes": f"scripts/benchmark.py {out['date']} annotated={args.annotated}"})
        print("capacity profile:", r.status_code, r.text)
    return 0 if best else 1


if __name__ == "__main__":
    sys.exit(main())
