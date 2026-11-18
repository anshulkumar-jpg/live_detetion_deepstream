# Deployment and operations

## 1. Single GPU server (Docker Compose)

1. Install the NVIDIA driver and the NVIDIA Container Toolkit. Check that `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi` works.
2. Log in to NGC: `docker login nvcr.io` (username `$oauthtoken`, password = your NGC API key).
3. `cp .env.example .env`, then generate the secrets as described in the file.
4. Put your model files in `./models`, matching the paths in `config/models/models.yaml`. The folder must be writable, because nvinfer caches the TensorRT engines it builds there.
5. Put a TLS certificate and key at `deploy/envoy/certs/tls.crt` and `tls.key`.
6. `docker compose build && docker compose up -d`. Add `--profile analytics --profile monitoring` for MTMC, Milvus, ELK, Prometheus and Grafana.
7. Validate the GPU, then benchmark, then register models (README, steps 1–3).

**DeepStream image and pyds version.** `docker/Dockerfile.worker` defaults to `nvcr.io/nvidia/deepstream:7.1-triton-multiarch` and the pyds v1.2.0 wheel (cp310). If your driver needs another DeepStream release, override both:

```bash
docker compose build --build-arg DS_IMAGE=... --build-arg PYDS_WHEEL_URL=... gpu-agent
```

**Ports.**

| Port | Use |
|---|---|
| 8443 | Envoy: API, `/hls/`, `/webrtc/` |
| 8554 | RTSP read |
| 8189/udp | WebRTC ICE |

Kibana, Prometheus and Grafana are bound to localhost only.

**WebRTC from outside the host.** Set `webrtcAdditionalHosts` in `deploy/mediamtx/mediamtx.yml` to the server's reachable IP or name.

## 2. Kubernetes (multi-GPU / multi-node ready)

`deploy/helm/vms` is the platform chart. It deploys:

- API: Deployment + PodDisruptionBudget
- scheduler: 2 replicas, leader-elected
- **gpu-agent**: a DaemonSet on nodes labelled `nvidia.com/gpu.present=true`. It requests `nvidia.com/gpu` and has probes and a graceful-shutdown period.
- analytics-stream and MTMC
- MediaMTX
- a migration Job run as a Helm hook
- an optional PodMonitor

Stateful dependencies use upstream operators and charts. Their manifests and values are in `deploy/helm/third-party/`:

- Strimzi for Kafka
- CloudNativePG for PostgreSQL
- the Milvus Helm chart
- ECK for Elasticsearch, Kibana and Logstash
- kube-prometheus-stack for Prometheus and Grafana

```bash
kubectl create ns vms
kubectl -n vms create secret generic vms-secrets --from-literal=DATABASE_URL=... --from-literal=JWT_SECRET=... \
  --from-literal=CREDENTIAL_ENCRYPTION_KEY=... --from-literal=INTERNAL_SERVICE_TOKEN=... --from-literal=BOOTSTRAP_ADMIN_PASSWORD=...
helm install vms deploy/helm/vms -n vms -f my-values.yaml
```

**GPU capacity in Kubernetes.**

- One agent pod per GPU node manages all logical workers on that node as processes. The chart deliberately doesn't create a pod per 2 cameras.
- Kubernetes does not add GPU capacity. Scheduling is still decided by the platform's admission control, using benchmark profiles per GPU model.
- MIG, MPS and time-slicing are not treated as isolation. If you use MIG, each MIG device appears as a separate GPU to NVML and the agent. Benchmark each profile.

**Multiple GPUs per node.** The agent reports every GPU, and the scheduler picks the GPU with the most headroom. Workers pin to a GPU through the DeepStream `gpu-id`.

## 3. Security checklist

- Secrets come only from the environment or Kubernetes Secrets. Production startup refuses missing or weak secrets.
- RTSP credentials are Fernet-encrypted in PostgreSQL. Workers decrypt them in memory, and logs redact `user:pass@` from URLs.
- JWT with RBAC (admin, operator, viewer). Every write is recorded in `audit_logs`.
- Envoy terminates TLS (TLS 1.2 minimum) and blocks `/internal/*` from outside.
- MediaMTX uses `authMethod: http`:
  - reading a stream requires a platform JWT
  - publishing requires a private source IP and the internal token
- For Kafka across hosts, set `KAFKA_SECURITY_PROTOCOL=SASL_SSL` and the SASL variables. The Strimzi manifest defines a SCRAM-authenticated TLS listener.
- Containers: the API image runs as non-root. The Helm chart sets `runAsNonRoot`, a read-only root filesystem and drops all capabilities. The GPU agent needs the NVIDIA runtime.

## 4. Operations

| Situation | What happens / what to do |
|---|---|
| Camera stuck in `PENDING_CAPACITY` | `GET /api/v1/gpu/resources` shows the failing admission check. Add GPU capacity, raise limits after benchmarking, or stop other cameras. Retries are automatic. |
| Camera `RECONNECTING` | nvurisrcbin reconnects every `RTSP_RECONNECT_INTERVAL_S`. A stall watchdog (5 s with no frames) marks the source, and source errors re-add only that source after a backoff. |
| Camera `FAILED` | Retried with exponential backoff (30 s up to 10 min). `status_reason` holds the last error. |
| Worker crash | The agent reports `WORKER_FAILED` immediately. A missed heartbeat (20 s) has the same effect. The scheduler marks the worker FAILED, re-admits its cameras on new workers and stops any surviving process. Healthy workers are not touched. |
| Model engine fails to load | `MODEL_LOAD_FAILED`: the version is marked FAILED and the model rolls back automatically to the last good version. Only cameras that use the model are rebuilt. |
| Manual rollback | `PUT /api/v1/models/{id}` with `{"activate_version":"1"}` |
| Scheduler restart | All state is in PostgreSQL. The next reconcile pass converges, and the standby takes over within about 5 s. |
| Kafka backlog | Alert `KafkaBacklog`. Detection events are sampled (`DETECTION_EVENT_INTERVAL_FRAMES`) and rate-limited (`EVENTS_MAX_PER_SECOND_PER_CAMERA`). Tune these rather than letting Kafka throttle video. |

**Retention.**

- Kafka: 24 h.
- Milvus: TTL plus an hourly delete (`EMBEDDING_RETENTION_DAYS`).
- PostgreSQL holds track-level rows only. Prune `detection_events`, `gpu_resource_metrics` and `analytics_events` with a cron job, or partition them by month for large fleets.
