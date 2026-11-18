# API reference

The machine-readable spec is in `docs/openapi.json`. Interactive docs are served at `/docs` on a running API.

Every `/api/v1` route except `POST /auth/token` needs `Authorization: Bearer <JWT>`. Roles are cumulative: `viewer` < `operator` < `admin`. Internal services can send `x-internal-token` instead of a JWT.

RTSP credentials are write-only. They can be given in the URL or as `username`/`password`. They are encrypted at rest with Fernet, never returned by the API, and redacted from logs.

| Method | Path | Role | Notes |
|---|---|---|---|
| POST | /api/v1/auth/token | - | `{username,password}` → JWT |
| POST | /api/v1/auth/users | admin | create user `{username,password,role}` |
| POST | /api/v1/cameras | operator | register; `id` is your camera id (e.g. `CAM_101`) |
| POST | /api/v1/cameras/bulk | operator | up to 1,000 per call |
| GET | /api/v1/cameras | viewer | `?status=&desired_state=&limit=&offset=` |
| GET/PUT/DELETE | /api/v1/cameras/{id} | viewer/operator | PUT on pipeline fields bumps `config_generation`, which rebuilds only this camera's source. DELETE on a running camera returns 409 and requests a stop. |
| POST | /api/v1/cameras/{id}/start · /stop | operator | 202; sets the desired state. Watch `status`: `SCHEDULED`, `STARTING`, `RUNNING`, `RECONNECTING`, `PENDING_CAPACITY`, `FAILED`, `STOPPED`. |
| POST | /api/v1/cameras/{id}/models | operator | replaces the camera's model assignments (modes A/B/C, see below) |
| GET | /api/v1/cameras/{id}/stream | viewer | HLS / WebRTC (WHEP) / RTSP URLs for `processed` and/or `raw` |
| GET | /api/v1/cameras/{id}/stream/status | viewer | MediaMTX readiness, viewer count, tracks; records `stream_sessions` |
| GET/POST | /api/v1/models | viewer/admin | register a model, optionally with its first version |
| GET/PUT | /api/v1/models/{id} | viewer/admin | `new_version`, `activate_version` (also used for rollback), `enabled` |
| GET | /api/v1/workers[/{id}[/metrics]] | viewer | state, cameras, heartbeat, per-source FPS/latency |
| POST | /api/v1/workers/{id}/restart · /stop | operator | restart in place / stop (its cameras are rescheduled) |
| GET | /api/v1/gpu/status · /metrics · /resources | viewer | devices, NVML history, admission view, pending cameras, capacity profiles |
| POST | /api/v1/gpu/capacity-profiles | admin | measured capacity (written by `scripts/benchmark.py`) |
| GET | /api/v1/events | viewer | track-level detections `?camera_id=&class_name=&model_id=&since=&until=` |
| GET | /api/v1/analytics/summary · /cameras/{id} | viewer | tracks by class and camera, Re-ID identities, behavior events |
| GET | /health · /ready · /metrics | - | liveness, readiness (database + bus), Prometheus |

## Model assignment modes

```jsonc
// Mode A – one model
{"models": [{"model_id": "person_detection"}]}
// Mode B – several full-frame models (disabled entries are kept but not run)
{"models": [{"model_id": "person_detection"}, {"model_id": "vehicle_detection"}, {"model_id": "pose_detection", "enabled": false}],
 "tracking_enabled": true, "tracker_type": "NvDCF"}
// Mode C – cascade: pose runs on person boxes (class 0) only
{"models": [{"model_id": "person_detection"},
            {"model_id": "pose_detection", "parent_model_id": "person_detection", "operate_on_class_ids": [0]}],
 "execution_strategy": "cascade"}
```

Validation rejects these requests with 422:

- unknown or disabled models
- models with no active version
- a parent model that isn't a detector, or isn't assigned and enabled
- cycles
- classifier or Re-ID models without a parent
- INT8 without a calibration file or prebuilt engine

## Kafka event schemas (`app/bus/schemas.py`)

All events carry `schema_version` and `event_id` (UUID, used for idempotency). Data topics are keyed by `camera_id`, which keeps per-camera ordering. Commands are keyed by `node_id`.

```json
{"schema_version":1,"event_id":"…","camera_id":"CAM_101","worker_id":"WORKER_1A2B3C4D5E","model_id":"person_detection",
 "model_version":"1","event_type":"object_detected","class_id":0,"class_name":"person","confidence":0.95,
 "object_id":"TRACK_6740f1a2_123","bbox":{"x":100,"y":120,"width":200,"height":300},"frame_num":4521,
 "frame_width":1920,"frame_height":1080,"timestamp":"2026-11-17T10:00:00.123+00:00"}
```

Topics:

| Topic | Contents |
|---|---|
| `camera.commands` | reserved |
| `camera.status` | camera status changes |
| `pipeline.commands` | START/STOP/RESTART_WORKER, ADD/REMOVE_SOURCE |
| `pipeline.events` | worker/source lifecycle, MODEL_LOAD_FAILED |
| `detection.events` | detections (example above) |
| `embedding.events` | sampled, L2-normalized embeddings |
| `worker.heartbeat` | worker and node heartbeats (NVML) |
| `pipeline.metrics` | pipeline metrics |
| `analytics.events` | track_started, track_ended + dwell, occupancy |
| `reid.events` | identity_assigned |

Every topic has a `<topic>.dlq` dead-letter queue. Failed messages land there with `x-error` headers.
