# DeepStream VMS Backend

Backend for registering up to 1,000 IP cameras and running GPU analytics on them. It uses NVIDIA DeepStream for video and AI, Kafka for events, FastAPI for the API, PostgreSQL for state, and MediaMTX for live video. The layout follows the NVIDIA MTMC deployment reference: Perception layer, then Kafka, then Analytics (Logstash, Elasticsearch, Kibana, MTMC, Milvus), then the Envoy gateway.

> **Capacity:** the system can **register** 1,000 cameras. How many it can **process** at the same time depends on your GPU. Run `scripts/benchmark.py` on your server to measure it. Until a benchmark result is recorded, the scheduler admits at most `DEFAULT_MAX_STREAMS_PER_GPU` streams (8 by default). Cameras over the limit wait in `PENDING_CAPACITY`, and the API reports why.

## Architecture

```
Client ──HTTPS──▶ Envoy GW ──▶ FastAPI (control plane) ──▶ PostgreSQL (desired + actual state)
                                   │ desired state
                                   ▼
                          Worker Scheduler (+ GPU admission control, leader-elected)
                                   │ pipeline.commands (Kafka, key=node)
                                   ▼
             GPU node agent (1 per GPU server; NVML telemetry → worker.heartbeat)
               ├─ worker process W1 ─ DeepStream pipeline group (2 cameras)
               ├─ worker process W2 ─ DeepStream pipeline group (2 cameras)
               └─ ...
   RTSP ─▶ nvurisrcbin(NVDEC) ─▶ nvstreammux ─▶ nvinfer(PGIEs) ─▶ nvtracker ─▶ nvinfer(SGIEs) ─▶ nvstreamdemux
                                                     │ probe: metadata/embeddings          └─▶ nvdsosd ─▶ NVENC ─▶ MediaMTX ─▶ HLS/WebRTC
                                                     ▼
          Kafka: detection.events · embedding.events · pipeline.events · worker.heartbeat · pipeline.metrics
             ├─▶ Analytics Stream ─▶ track summaries (PostgreSQL) + behavior events (analytics.events)
             ├─▶ MTMC (Re-ID) ◀──▶ Milvus  ─▶ reid.events + reid_associations
             └─▶ Logstash ─▶ Elasticsearch ─▶ Kibana
```

| Plane | Components |
|---|---|
| Control | `app/` FastAPI, `app/services/scheduler.py`, `app/services/model_router.py`, `app/services/gpu.py` |
| Video processing | `worker/agent.py` (node agent), `worker/runner.py` + `worker/pipeline/` (DeepStream) |
| Analytics | `analytics/stream.py`, `analytics/mtmc.py`, `analytics/vector_store.py`, Logstash/ES/Kibana |
| Storage | PostgreSQL (Alembic migrations), Milvus, Elasticsearch |
| Streaming | MediaMTX (`deploy/mediamtx`), `app/services/mediamtx.py`, VST adapter interface `app/services/vst.py` |

### Key design decisions

- **Logical worker = 1 DeepStream pipeline group = 1 OS process with up to `CAMERAS_PER_WORKER` cameras (default 2).** For example, 10 cameras run as 5 workers. A crash only affects that worker's cameras. The workers are *not* Kubernetes pods; one agent pod per GPU node runs them.
- **Cameras with the same models share a worker.** Cameras are grouped by an identical inference plan (the pipeline signature). That means a camera with a different model set never shares, or reloads, another camera's TensorRT engines.
- **Changes never interrupt other cameras.**
  - Adding a camera to a worker with a free slot uses `ADD_SOURCE`, which adds a source to the running pipeline.
  - Removing a camera or changing its models removes only that source.
  - Model version activation rebuilds only the cameras that use that model.
- **The scheduler reconciles from the database.** It compares desired state with actual state every 2 s and on every event. This handles lost messages, restarts and crashes. A PostgreSQL advisory lock makes one scheduler the leader; a second instance stands by.
- **Admission control uses measurements.** It combines NVML readings (GPU utilization, memory, NVDEC, NVENC, temperature), worker latency, and benchmark capacity profiles. It also counts streams that were admitted but haven't started yet. The GPU is never split into fixed slices.
- **Kafka never blocks video.** Events are sampled per track and rate-limited per camera with a token bucket. When the producer's queue is full, events are dropped and counted rather than waiting.

## Quick start (single GPU server, Docker Compose)

Prerequisites: NVIDIA driver, Docker, [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/), and NGC access for the DeepStream base image.

```bash
cp .env.example .env               # fill every CHANGE_ME (commands to generate keys are in the file)
mkdir -p models deploy/envoy/certs # put ONNX/engines/labels in ./models, TLS cert/key in deploy/envoy/certs
docker compose build
docker compose up -d                                  # core
docker compose --profile analytics --profile monitoring up -d   # + MTMC/Milvus/ELK + Prometheus/Grafana

# 1. validate the GPU node and your models (real DeepStream runs)
docker compose run --rm gpu-agent python3 scripts/validate_gpu_node.py --models config/models/models.yaml \
    --rtsp 'rtsp://user:pass@CAMERA/stream' --mediamtx-api http://mediamtx:9997
# 2. measure capacity and record it for admission control
docker compose run --rm gpu-agent python3 scripts/benchmark.py --models config/models/models.yaml \
    --model-set person_detection --uri rtsp://mediamtx:8554/bench/clip --post-api http://api:8000
# 3. register models, cameras, assign, start
python scripts/register_models.py --api https://HOST:8443 --user admin config/models/models.yaml
```

```bash
TOKEN=$(curl -sk -X POST https://HOST:8443/api/v1/auth/token -H 'content-type: application/json' \
        -d '{"username":"admin","password":"..."}' | jq -r .access_token)
curl -sk -X POST https://HOST:8443/api/v1/cameras -H "Authorization: Bearer $TOKEN" -H 'content-type: application/json' \
     -d '{"id":"CAM_101","name":"Gate","rtsp_url":"rtsp://user:pass@10.0.0.5/stream1","stream_mode":"annotated"}'
curl -sk -X POST https://HOST:8443/api/v1/cameras/CAM_101/models -H "Authorization: Bearer $TOKEN" -H 'content-type: application/json' \
     -d '{"models":[{"model_id":"person_detection"},{"model_id":"pose_detection","parent_model_id":"person_detection"}]}'
curl -sk -X POST https://HOST:8443/api/v1/cameras/CAM_101/start -H "Authorization: Bearer $TOKEN"
curl -sk https://HOST:8443/api/v1/cameras/CAM_101/stream -H "Authorization: Bearer $TOKEN"   # HLS / WebRTC URLs
```

Players add `?jwt=<token>` (or an `Authorization` header) when reading HLS/WebRTC. MediaMTX checks every read and publish against the API.

## Tests

```bash
pip install -r requirements-dev.txt
export TEST_DATABASE_URL=postgresql+asyncpg://vms:vms@localhost:5432/vms_test   # real PostgreSQL
export TEST_KAFKA_BOOTSTRAP=localhost:9092                                      # optional real Kafka
pytest -q
```

| Suite | What it does |
|---|---|
| Unit | Routing, config generation, admission control, event sampling/rate limiting, security, node agent (real subprocesses), MTMC |
| Integration | Scheduler lifecycle/recovery, full API with 1,000 cameras, analytics on PostgreSQL, Kafka with DLQ, scheduler → agent end-to-end, Milvus (Lite) |
| GPU | `scripts/validate_gpu_node.py` and `scripts/benchmark.py`, run on the GPU server |

See `docs/IMPLEMENTATION_REPORT.md` for what was verified and what still needs to be checked on the GPU server.

## Docs

| Document | Contents |
|---|---|
| `docs/IMPLEMENTATION_REPORT.md` | Phases, tests, open items |
| `docs/API.md` and `docs/openapi.json` | API reference (live docs at `/docs`) |
| `docs/DEPLOYMENT.md` | Compose, Kubernetes/Helm, security, operations |
| `docs/BENCHMARK.md` | Capacity measurement procedure and results log |
