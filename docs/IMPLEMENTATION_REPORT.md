# Implementation report

## Phase 1 – Repository audit

**The existing repository (`C:\Users\nj\github\deepstream`) could not be audited.** The build session had no access to that machine, and you asked for the code to be written to a new folder. As a result:

- Nothing in your repository was inspected, preserved or modified.
- The three existing AI models, their formats, pre/post-processing and class maps are **unknown**. No model files are included, and no detections were simulated.
- The Model Manager is generic: register each real model with its ONNX/engine, labels and Gst-nvinfer settings (`config/models/models.example.yaml`).

**Merging with your existing code.** Copy this tree beside your project and compare it against your `docker-compose.yml`, FastAPI routes, database models, Kafka topics and worker code. Keep any working pieces you prefer. The module boundaries below are designed so they can be swapped individually: router, scheduler, agent, pipeline, analytics. The audit can still be done properly if you share the repository (connect the folder or give a Git URL).

## Phases 2–11 – What was built

| Phase | Main files | Status |
|---|---|---|
| 2 Core backend | `app/core/*`, `app/db/models.py`, `migrations/`, `app/api/v1/*`, `app/main.py` | Built and tested on PostgreSQL 16 |
| 3 DeepStream integration | `worker/pipeline/deepstream_pipeline.py`, `worker/pipeline/config_gen.py`, `worker/runner.py` | Written; config generation and failure path tested; **GStreamer/pyds runtime not run (no GPU)** |
| 4 Model routing | `app/services/model_router.py` (modes A/B/C, signatures, validation, versioning, rollback) | Built and tested |
| 5 Worker management | `app/services/scheduler.py`, `worker/agent.py`, `app/scheduler_main.py` | Built and tested (real subprocesses, real Kafka) |
| 6 GPU Resource Manager | `app/services/gpu.py` (NVML collector + admission control), `/api/v1/gpu/*` | Admission tested; **NVML collector not run (no GPU)** |
| 7 Kafka | `app/bus/kafka.py`, `app/bus/schemas.py` | Tested on Kafka 3.9.1 (ordering, DLQ, retries) |
| 8 Analytics | `analytics/stream.py`, `analytics/mtmc.py`, `analytics/vector_store.py`, `deploy/logstash` | Stream and MTMC tested on PostgreSQL and Milvus Lite; Logstash/ES config not run |
| 9 Video output | NVENC branch in the pipeline, `app/services/mediamtx.py`, `deploy/mediamtx`, MediaMTX auth callback | API/auth tested with mocked MediaMTX HTTP; **real stream not run** |
| 10 Monitoring & reliability | `app/core/metrics.py`, `deploy/prometheus/*`, `deploy/grafana/*`, recovery logic | Metrics endpoint and recovery paths tested; Grafana/Prometheus not run |
| 11 Containers | `docker/Dockerfile.api`, `docker/Dockerfile.worker`, `docker-compose.yml`, `deploy/helm/*`, `deploy/envoy` | API image built and smoke-tested; compose validated; Helm lint/render passes; **worker image not built (NGC image)** |
| 12 Production validation | `scripts/validate_gpu_node.py`, `scripts/benchmark.py`, `docs/BENCHMARK.md` | Scripts ready; **must be run on your GPU server** |

## Commands executed and results

```
alembic upgrade head / downgrade base / upgrade head / check   -> clean on PostgreSQL 16 (19 tables)
pytest tests  (PostgreSQL 16 + Kafka 3.9.1 + Milvus Lite)       -> 62 passed
ruff check .                                                   -> clean
docker compose config (core + analytics + monitoring profiles) -> valid; required secrets enforced
docker build -f docker/Dockerfile.api                          -> built
container smoke test: migrate -> API -> scheduler -> camera start -> PENDING_CAPACITY "no GPU registered"
two schedulers: standby until leader stopped, then took over (advisory-lock failover)
docker build --check -f docker/Dockerfile.worker               -> no warnings (image not built)
helm lint / helm template deploy/helm/vms                      -> pass, 13 resources rendered
```

| Test file | Count | Covers |
|---|---|---|
| unit/test_router_admission_events.py | 17 | modes A/B/C, signature grouping, validation errors, every admission denial reason, event sampling, bbox scaling, rate limiting, FPS/drop statistics |
| unit/test_security_and_config.py | 9 | scrypt, JWT, credential split/encrypt, log redaction, nvinfer primary/secondary configs, labels, nvinferserver, tracker config |
| unit/test_agent_mtmc.py | 4 | agent start/add/idempotency/node filter/crash detection/graceful stop/stale commands (real subprocesses); MTMC cross-camera matching |
| unit/test_scripts_and_runner.py | 2 | benchmark plan generation; runner exits 3 cleanly without DeepStream |
| integration/test_scheduler.py | 16 | 4→2, 6→3, 10→5, 5→3 workers; model sets never mixed; add/remove without touching the other camera; admission and PENDING_CAPACITY recovery; no-GPU; overheating; capacity profile cap; crash recovery leaving healthy workers alone; zombie teardown; config-change rebuild; model-load rollback; idempotent events; failure backoff |
| integration/test_api.py | 7 | RBAC/401/403/internal token; credentials never returned; CRUD, versions, rollback, validation; end-to-end start → worker → RUNNING → stream URLs → restart → stop; **1,000-camera registration and scheduling**; MediaMTX auth callback |
| integration/test_analytics.py | 3 | track summaries, track_ended with dwell, occupancy, Re-ID persisted to PostgreSQL |
| integration/test_kafka.py | 2 | real broker: topics, per-key ordering, retries → DLQ with headers; scheduler → Kafka → agent → worker process end-to-end |
| integration/test_milvus.py | 2 | Milvus collection, upsert, filtered COSINE search, retention; MTMC on Milvus |

## Acceptance criteria

| # | Criterion | Status |
|---|---|---|
| 1 | Register 1,000 cameras | ✅ tested (0.22 s) |
| 2 | Start/stop through API | ✅ tested (control plane); real pipeline on GPU pending |
| 3 | Three existing models integrated | ⏳ needs your models registered + `validate_gpu_node.py` |
| 4/5 | Multi-model per camera / different models per camera | ✅ routing and scheduling tested; GPU run pending |
| 6/7 | DeepStream RTSP, detection, tracking | ⏳ code written; needs GPU validation |
| 8 | Embeddings | ✅ extraction path written (tensor meta), MTMC/Milvus tested; GPU run pending |
| 9/10 | Automatic worker creation/removal, 2 cameras/worker | ✅ tested |
| 11 | GPU monitoring | ⏳ NVML collector written; needs GPU |
| 12 | Admission control | ✅ tested |
| 13 | Kafka publish/consume | ✅ tested on real broker |
| 14 | MTMC + Milvus interfaces | ✅ tested (Milvus Lite) |
| 15 | MediaMTX processed video | ⏳ needs GPU (NVENC) |
| 16 | Worker failure recovery | ✅ tested (scheduler + agent) |
| 17 | Metrics exposed | ✅ |
| 18 | Docker on the GPU server | ⏳ API image verified; worker image to be built there |
| 19 | Integration tests | ✅ 62 passing |
| 20 | Capacity measured | ⏳ `scripts/benchmark.py` on target hardware |

## Remaining issues and risks

1. **No GPU runtime verification yet.** The DeepStream pipeline was written against the documented GStreamer/pyds APIs (nvurisrcbin, request pads, tensor-meta access) and is unverified until `validate_gpu_node.py` passes. Property names can differ between DeepStream releases. Pin the image to the release you validate.
2. **Pose output decoding is not implemented.** The pose model runs with `output-tensor-meta=1`, but turning its raw tensors into keypoints depends on your model. Supply your existing parser, either as an nvinfer `custom-lib-path` or as a Python decoder in `worker/pipeline`.
3. **MTMC is appearance-only** (cosine similarity on Re-ID embeddings). It has no camera topology or calibration constraints. Validate the threshold on your footage before relying on global IDs. The `CandidateFilter` hook is where Calibration Toolkit output would plug in; it is not integrated.
4. **VST/VMS:** interface only (`app/services/vst.py`), because the VST API depends on the version deployed.
5. **Dropped frames are estimated** from the measured FPS falling short of the configured input FPS. Frames that never reach the muxer cannot be counted directly.
6. **TensorRT engines are per process.** Two cameras per worker means each worker holds its own engine copy. Measure memory per stream with the benchmark.
7. **Admission before benchmarking** relies on the per-stream memory estimate plus `DEFAULT_MAX_STREAMS_PER_GPU`. Compute, NVDEC and latency limits only take effect once streams are running. Keep the conservative cap until capacity profiles exist.
8. Logstash, Elasticsearch, Kibana, Prometheus, Grafana, Envoy and the Helm chart are configured and linted but have not been run here.

## Next phase

Run `scripts/validate_gpu_node.py` on the GPU server with your three models, fix anything it reports, then run `scripts/benchmark.py` for each model set you deploy.
