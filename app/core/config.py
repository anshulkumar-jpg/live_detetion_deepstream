"""Environment-based configuration for every service in the platform.

All values come from environment variables (or a local .env file). Nothing secret
has a usable default: production startup fails fast if secrets are missing.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ---- general --------------------------------------------------------
    environment: Literal["development", "test", "production"] = "development"
    service_name: str = "vms-api"
    log_level: str = "INFO"
    log_json: bool = True

    # ---- database (source of truth) ------------------------------------
    database_url: str = "postgresql+asyncpg://vms:vms@localhost:5432/vms"
    db_pool_size: int = 20
    db_max_overflow: int = 10

    # ---- security -------------------------------------------------------
    jwt_secret: str = Field(default="", description="HS256 signing secret")
    jwt_algorithm: str = "HS256"
    # key rotation: the previous signing secret is still accepted (to verify only) until old tokens have expired
    jwt_secret_previous: str = ""
    jwt_expire_minutes: int = 60
    video_token_ttl_s: int = 300  # camera-scoped video tokens live this long (a player asks for a new one)
    # sign-in protection: a user name is locked after this many failures within the window, a client address after more
    login_max_failures_per_user: int = 5
    login_max_failures_per_ip: int = 20
    login_window_s: int = 900
    login_lockout_s: int = 900
    # Fernet key (urlsafe base64, 32 bytes) used to encrypt RTSP credentials at rest
    credential_encryption_key: str = ""
    # comma-separated older keys, only used to decrypt until scripts/rotate_credentials.py has re-encrypted everything
    credential_encryption_keys_old: str = ""
    # bootstrap admin, created on first start only if no users exist
    bootstrap_admin_username: str = "admin"
    bootstrap_admin_password: str = ""
    # Password DeepStream workers use to publish video to MediaMTX. It can publish and do nothing else; it is only
    # given to the API (to check it) and the GPU agent (to use it), not to every service.
    mediamtx_publish_token: str = ""
    # HMAC key for pipeline commands: the scheduler signs them, GPU agents check them. Nobody else holds it.
    command_signing_key: str = ""
    command_ttl_s: int = 60  # how long a signed command stays valid (max 300)
    allow_unsigned_commands: bool = False  # development only: accept and send commands without a signature
    # shared libraries a model may load (custom-lib-path). Keep this folder out of the writable models volume.
    allowed_lib_dirs: list[str] = ["/opt/vms/lib"]
    allow_file_sources: bool = False  # file:// camera sources (benchmarks only)
    cors_origins: list[str] = []

    # ---- kafka ----------------------------------------------------------
    kafka_enabled: bool = True
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_security_protocol: str = "PLAINTEXT"  # PLAINTEXT | SASL_SSL | SASL_PLAINTEXT | SSL
    kafka_sasl_mechanism: str = ""  # PLAIN | SCRAM-SHA-256 | SCRAM-SHA-512
    kafka_sasl_username: str = ""
    kafka_sasl_password: str = ""
    kafka_ssl_ca_location: str = ""
    kafka_topic_prefix: str = ""
    kafka_default_partitions: int = 12
    kafka_replication_factor: int = 1
    kafka_max_retries: int = 3
    kafka_producer_queue_max: int = 100_000

    # ---- scheduler / workers -------------------------------------------
    cameras_per_worker: int = 2  # project requirement: exactly 2 by default
    scheduler_interval_s: float = 2.0
    heartbeat_interval_s: float = 5.0
    heartbeat_timeout_s: float = 20.0
    worker_start_timeout_s: float = 120.0
    max_worker_restarts: int = 5
    node_id: str = "gpu-node-1"

    # ---- GPU admission control -----------------------------------------
    gpu_max_utilization_pct: float = 85.0
    gpu_max_memory_pct: float = 90.0
    gpu_max_decoder_pct: float = 85.0
    gpu_max_encoder_pct: float = 85.0
    gpu_max_temperature_c: float = 85.0
    gpu_max_inference_latency_ms: float = 200.0
    gpu_memory_reserve_mb: int = 1024
    # Used when no benchmark-derived per-stream cost exists for a model set.
    default_stream_memory_mb: int = 350
    # Hard cap of streams per GPU until a benchmark sets a measured value. 0 = no cap.
    default_max_streams_per_gpu: int = 0
    gpu_metrics_stale_s: float = 30.0

    # ---- DeepStream worker ---------------------------------------------
    deepstream_config_dir: str = "/opt/vms/generated"
    model_repository_dir: str = "/models"
    tracker_lib: str = "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so"
    # official tracker configs shipped in the DeepStream container
    tracker_config_dir: str = "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app"
    source_stall_timeout_s: float = 5.0
    source_readd_backoff_s: float = 15.0
    rtsp_reconnect_interval_s: int = 10
    rtsp_reconnect_attempts: int = -1
    rtsp_latency_ms: int = 200
    rtsp_tcp: bool = True
    streammux_width: int = 1920
    streammux_height: int = 1080
    streammux_batched_push_timeout_us: int = 40000
    detection_event_interval_frames: int = 15
    embedding_interval_frames: int = 30
    events_max_per_second_per_camera: int = 60
    encoder_bitrate_bps: int = 4_000_000

    # ---- video plane ----------------------------------------------------
    mediamtx_api_url: str = "http://localhost:9997"
    mediamtx_api_user: str = "vms-api"  # login the API and scheduler use for the MediaMTX control API
    mediamtx_api_password: str = ""
    mediamtx_metrics_password: str = ""  # login (user "metrics") Prometheus uses to read MediaMTX metrics
    # Raw (un-annotated) camera streams make MediaMTX pull the camera itself, which puts the camera password into
    # the video server. Off by default; annotated streams (published by the workers) need no camera password.
    allow_raw_streams: bool = False
    mediamtx_rtsp_push_url: str = "rtsp://localhost:8554"
    mediamtx_public_hls_url: str = "http://localhost:8888"
    mediamtx_public_webrtc_url: str = "http://localhost:8889"
    mediamtx_public_rtsp_url: str = "rtsp://localhost:8554"
    stream_push_protocol: Literal["rtsp", "rtmp"] = "rtsp"
    mediamtx_rtmp_push_url: str = "rtmp://localhost:1935"
    vst_base_url: str = ""  # optional VST/VMS adapter

    # ---- analytics plane -----------------------------------------------
    milvus_uri: str = "http://localhost:19530"
    milvus_token: str = ""
    milvus_collection: str = "object_embeddings"
    embedding_dim: int = 256
    mtmc_match_threshold: float = 0.75  # cosine similarity
    mtmc_time_window_s: int = 600
    mtmc_min_track_samples: int = 3
    embedding_retention_days: int = 7
    # retention of personal / behavioural data in PostgreSQL (days; 0 = keep forever). See app/services/retention.py
    detection_event_retention_days: int = 30
    reid_retention_days: int = 7  # cross-camera identities: keep as short as the embeddings
    analytics_event_retention_days: int = 30
    audit_retention_days: int = 365
    elasticsearch_url: str = ""
    elasticsearch_index_prefix: str = "vms-detections"

    @field_validator("cameras_per_worker")
    @classmethod
    def _positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("cameras_per_worker must be >= 1")
        return v

    def validate_for_production(self, needs_publish_token: bool = False, needs_command_key: bool = False,
                                needs_mediamtx_api: bool = False, needs_mediamtx_metrics: bool = False) -> None:
        required = ["jwt_secret", "credential_encryption_key"]
        if needs_command_key:
            required.append("command_signing_key")
        if needs_mediamtx_api:
            required.append("mediamtx_api_password")
        if needs_mediamtx_metrics:
            required.append("mediamtx_metrics_password")
        if needs_publish_token:
            required.append("mediamtx_publish_token")
        missing = [name for name in required if not getattr(self, name)]
        if missing:
            raise RuntimeError(f"Missing required secrets: {', '.join(missing).upper()}")
        if len(self.jwt_secret) < 32:
            raise RuntimeError("JWT_SECRET must be at least 32 characters")
        if needs_publish_token and len(self.mediamtx_publish_token) < 24:
            raise RuntimeError("MEDIAMTX_PUBLISH_TOKEN must be at least 24 characters")
        if needs_command_key and len(self.command_signing_key) < 32:
            raise RuntimeError("COMMAND_SIGNING_KEY must be at least 32 characters")
        if needs_mediamtx_api and len(self.mediamtx_api_password) < 24:
            raise RuntimeError("MEDIAMTX_API_PASSWORD must be at least 24 characters")
        if needs_mediamtx_metrics and len(self.mediamtx_metrics_password) < 24:
            raise RuntimeError("MEDIAMTX_METRICS_PASSWORD must be at least 24 characters")

    def topic(self, name: str) -> str:
        return f"{self.kafka_topic_prefix}{name}"


@lru_cache
def get_settings() -> Settings:
    return Settings()
