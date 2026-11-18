"""One command to run the whole system and see which model runs on which camera.

    python scripts/run_all.py            # start stack, register models, start all cameras, open live viewer, show table
    python scripts/run_all.py --watch    # same, then keep refreshing the table every 5 s (Ctrl+C to stop watching)
    python scripts/run_all.py --status   # only print the table (does not start anything)

Credentials are read from .env (BOOTSTRAP_ADMIN_PASSWORD). Cameras must already be registered
(see scripts/register_cameras.py); this script starts them and shows their state.
"""
from __future__ import annotations

import argparse
import os
import shutil
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
CORE = ["postgres", "kafka", "migrate", "api", "scheduler", "mediamtx", "envoy", "analytics-stream", "gpu-agent"]
G, Y, R, C, D, B, X = "\033[32m", "\033[33m", "\033[31m", "\033[36m", "\033[2m", "\033[1m", "\033[0m"

if os.name == "nt":
    os.system("")  # enable ANSI colours in the Windows console


def say(msg: str) -> None:
    print(f"{C}>>{X} {msg}", flush=True)


def load_env() -> dict[str, str]:
    env: dict[str, str] = {}
    f = ROOT / ".env"
    if f.exists():
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, **kw)


def docker_ready() -> bool:
    return run(["docker", "info"]).returncode == 0


def ensure_docker() -> None:
    if docker_ready():
        return
    say("Docker is not running - starting Docker Desktop...")
    exe = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Docker" / "Docker" / "Docker Desktop.exe"
    if exe.exists():
        subprocess.Popen([str(exe)])
    for _ in range(60):
        if docker_ready():
            return
        time.sleep(5)
    sys.exit(f"{R}Docker did not start. Start Docker Desktop manually and run this again.{X}")


class Api:
    def __init__(self, base: str, user: str, password: str) -> None:
        self.base = base
        self.http = httpx.Client(verify=False, timeout=30)
        self.user, self.password = user, password
        self.token = ""

    def login(self) -> bool:
        try:
            r = self.http.post(f"{self.base}/api/v1/auth/token", json={"username": self.user, "password": self.password})
        except httpx.HTTPError:
            return False
        if r.status_code != 200:
            return False
        self.token = r.json()["access_token"]
        return True

    def get(self, path: str, **kw):
        return self.http.get(f"{self.base}/api/v1{path}", headers={"Authorization": f"Bearer {self.token}"}, **kw)

    def post(self, path: str, **kw):
        return self.http.post(f"{self.base}/api/v1{path}", headers={"Authorization": f"Bearer {self.token}"}, **kw)


def listening(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1", port)) == 0


def cameras(api: Api) -> list[dict]:
    d = api.get("/cameras", params={"limit": 500}).json()
    return d.get("items", d) if isinstance(d, dict) else d


LABELS: dict[str, list[str]] = {}


def model_text(c: dict) -> str:
    ms = sorted(c.get("models", []), key=lambda m: m.get("order", 0))
    parts = []
    for m in ms:
        name = m["model_id"].replace("_detection", "")
        ids = m.get("detect_class_ids")
        if ids:  # the camera detects only some of the model's classes
            lab = LABELS.get(m["model_id"], [])
            name += "[" + ",".join(lab[i] if i < len(lab) else str(i) for i in ids) + "]"
        if m.get("conf_threshold") or m.get("iou_threshold"):  # camera uses its own sensitivity
            name += " c" + format(m["conf_threshold"], "g") if m.get("conf_threshold") else ""
            name += " i" + format(m["iou_threshold"], "g") if m.get("iou_threshold") else ""
        parts.append(f"{name} (posture)" if m.get("parent_model_id") else name)
    return " -> ".join(parts) if len(parts) > 1 and any(m.get("parent_model_id") for m in ms) else " + ".join(parts) or "-"


def worker_stats(api: Api) -> dict[str, dict]:
    """camera_id -> {fps, latency, worker}"""
    out: dict[str, dict] = {}
    try:
        for w in api.get("/workers").json():
            m = api.get(f"/workers/{w['id']}/metrics").json().get("metrics", {})
            for cam, s in (m.get("sources") or {}).items():
                out[cam] = {"fps": s.get("fps"), "lat": s.get("latency_ms"), "worker": w["id"][-6:]}
    except Exception:
        pass
    return out


def gpu_line(api: Api) -> str:
    r = run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"])
    if r.returncode != 0 or not r.stdout.strip():
        return ""
    name, util, used, total, temp = [x.strip() for x in r.stdout.strip().splitlines()[0].split(",")]
    return f"GPU: {name} | util {util}% | memory {used}/{total} MiB | {temp} C"


def show(api: Api, viewer_port: int) -> int:
    cams = cameras(api)
    if any(m.get("detect_class_ids") for c in cams for m in c.get("models", [])):
        try:
            for md in api.get("/models").json():
                LABELS[md["id"]] = (md.get("active_version") or {}).get("labels") or []
        except Exception:
            pass
    stats = worker_stats(api)
    running = sum(1 for c in cams if c["status"] == "RUNNING")
    print("\033[2J\033[H" if os.name != "nt" else "", end="")
    print(f"{B}=== DeepStream VMS  |  cameras live: {running}/{len(cams)} ==={X}")
    print(f"{D}{gpu_line(api)}{X}\n")
    print(f"{B}{'CAMERA':<11} {'STATUS':<9} {'MODEL RUNNING':<40} {'DETECTION':<16} {'TRACKER':<11}{'FPS':>4} {'LAG':>5}  WORKER{X}")
    print("-" * 110)
    for c in sorted(cams, key=lambda c: c["id"]):
        st = c["status"]
        col = G if st == "RUNNING" else (Y if st in ("STARTING", "SCHEDULED", "PENDING_CAPACITY") else R)
        s = stats.get(c["id"], {})
        fps = f"{s['fps']:.0f}" if s.get("fps") else "-"
        lat = f"{s['lat']:.0f}" if s.get("lat") else "-"
        n = c.get("detection_interval") or 0
        if c.get("tracker_type") in ("ByteTrack", "OCSORT", "BoTSORT", "StrongSORT"):
            n = 0  # python trackers need detection on every frame
        det = "every frame" if n == 0 else f"every {n + 1}nd frame" if n == 1 else f"every {n + 1}rd frame" if n == 2 else f"every {n + 1}th frame"
        trk = (c.get("tracker_type") or "NvDCF") if c.get("tracking_enabled", True) else "off"
        print(f"{c['id']:<11} {col}{st:<9}{X} {model_text(c):<40} {det:<16} {trk:<11}{fps:>4} {lat:>5}  {s.get('worker', '-')}")
        if st not in ("RUNNING", "STOPPED") and c.get("status_reason"):
            print(f"{'':<11} {D}{c['status_reason']}{X}")
    print(f"\n{B}Live video (all cameras):{X} http://localhost:{viewer_port}")
    return running


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", action="store_true", help="keep refreshing the table")
    ap.add_argument("--status", action="store_true", help="only print the table")
    ap.add_argument("--no-open", action="store_true", help="do not open the browser")
    ap.add_argument("--api", default="https://localhost:8443")
    ap.add_argument("--viewer-port", type=int, default=8080)
    a = ap.parse_args()

    env = load_env()
    pw = os.environ.get("VMS_PASSWORD") or env.get("BOOTSTRAP_ADMIN_PASSWORD", "")
    if not pw:
        sys.exit(f"{R}No admin password: set VMS_PASSWORD or BOOTSTRAP_ADMIN_PASSWORD in .env{X}")
    api = Api(a.api, env.get("BOOTSTRAP_ADMIN_USERNAME", "admin"), pw)

    if not a.status:
        ensure_docker()
        say("Starting services (database, Kafka, API, scheduler, video server, GPU agent)...")
        r = run(["docker", "compose", "up", "-d", *CORE])
        if r.returncode != 0:
            sys.exit(f"{R}docker compose failed:{X}\n{r.stderr[-1500:]}")

    say("Waiting for the API...")
    for _ in range(90):
        if api.login():
            break
        time.sleep(2)
    else:
        sys.exit(f"{R}API did not become ready (check: docker compose logs api){X}")

    if not a.status:
        say("Registering models (no change if already up to date)...")
        r = run([sys.executable, "scripts/register_models.py", "config/models/models.yaml", "--api", a.api,
                 "--ca", "deploy/envoy/certs/tls.crt"], env={**os.environ, "VMS_PASSWORD": pw})
        for line in (r.stdout or "").strip().splitlines():
            print(f"   {D}{line}{X}")

        cams = cameras(api)
        if not cams:
            print(f"{Y}No cameras registered yet - add them with scripts/register_cameras.py{X}")
        for c in cams:
            if c["status"] not in ("RUNNING", "STARTING", "SCHEDULED"):
                api.post(f"/cameras/{c['id']}/start")
        say(f"Starting {len(cams)} cameras...")

        if not listening(a.viewer_port):
            say("Starting the live viewer...")
            kw = {"creationflags": 0x00000008 | 0x00000200} if os.name == "nt" else {"start_new_session": True}
            subprocess.Popen([sys.executable, str(ROOT / "scripts" / "live_viewer.py"), "--port", str(a.viewer_port),
                              "--api", a.api], cwd=ROOT, env={**os.environ, "VMS_PASSWORD": pw, "PYTHONUNBUFFERED": "1"},
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **kw)
        deadline = time.time() + 240
        while time.time() < deadline:
            n_run = show(api, a.viewer_port)
            if n_run == len(cameras(api)):
                break
            time.sleep(5)
        if not a.no_open:
            webbrowser.open(f"http://localhost:{a.viewer_port}")

    try:
        while True:
            show(api, a.viewer_port)
            if not a.watch:
                break
            time.sleep(5)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
