# Performance benchmarking

**Status: no hardware results yet.** The code was built in an environment without an NVIDIA GPU. No throughput or capacity figures have been measured, and none are claimed. Results go below once `scripts/benchmark.py` has been run on your GPU server.

## What has been measured (CPU-side control plane, no GPU)

| Test | Result |
|---|---|
| Register 1,000 cameras through the API (10 bulk calls, PostgreSQL 16) | 0.22 s |
| Scheduler reconcile pass over 1,000 cameras (793 pending, 207 assigned) | 1.3 s |
| Kafka produce/consume with per-camera ordering and DLQ (local broker 3.9.1) | pass |

## Procedure on the GPU server

1. **Real-time source.** Loop a representative clip into MediaMTX so decoding runs at camera pace:

   ```bash
   ffmpeg -re -stream_loop -1 -i your_clip_1080p.mp4 -c copy -f rtsp rtsp://worker:$INTERNAL_SERVICE_TOKEN@mediamtx:8554/bench/clip
   ```

   Use clips at your real camera resolutions and codecs, with H.264 and H.265 tested separately. A `file://` URI only measures raw decode throughput.

2. **Run each production model combination.** Every distinct model set has its own pipeline signature and its own capacity:

   ```bash
   python3 scripts/benchmark.py --models config/models/models.yaml --model-set person_detection \
       --uri rtsp://mediamtx:8554/bench/clip --start 2 --step 2 --max 64 --warmup 60 --duration 120 --post-api http://api:8000
   python3 scripts/benchmark.py ... --model-set person_detection,vehicle_detection
   python3 scripts/benchmark.py ... --model-set person_detection,pose_detection --cascade-parent person_detection
   python3 scripts/benchmark.py ... --annotated     # include OSD + NVENC output cost
   ```

3. A level passes when all of the following hold:
   - every stream is RUNNING at ≥95 % of the target FPS
   - p95 capture-to-metadata latency is ≤500 ms
   - GPU memory is ≤90 %
   - no worker crashed

   The highest passing level is posted as a capacity profile, and admission control enforces it from then on.

4. Raise `DEFAULT_MAX_STREAMS_PER_GPU` (or remove the cap) only after every model set you deploy has a profile.

**Engine builds.** The first run builds TensorRT engines from ONNX, which can take minutes. Use a long `--warmup` once, or pre-build engines with `trtexec` and set `engine_file`.

**Things to watch.**
- NVDEC is usually the first limit for many 1080p30 streams. `nvdec_avg` is reported per level.
- Each worker process loads its own copy of the TensorRT engines. Memory per stream therefore depends on `CAMERAS_PER_WORKER`. If memory is what limits you, benchmark with a larger value; the default stays 2.

## Results

<!-- scripts/benchmark.py appends to docs/benchmark_results.md; copy the summary here -->
_Not yet measured on target hardware._
