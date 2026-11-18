"""Register NVR channels as cameras and assign models round-robin.

Credentials come from env (VMS_PASSWORD for the API, NVR_PASSWORD for the NVR), never from the command line.
Example:
  python scripts/register_cameras.py --api https://localhost:8443 --ca deploy/envoy/certs/tls.crt \
      --nvr-ip 192.168.1.100 --nvr-user admin --nvr-prefix NVR1 --channels 2-9,17 \
      --models person_detection,vehicle_detection,fall_detection [--start]
"""
from __future__ import annotations

import argparse
import os
import sys

import httpx


def parse_channels(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="https://localhost:8443")
    ap.add_argument("--ca", default=None)
    ap.add_argument("--user", default="admin")
    ap.add_argument("--nvr-ip", required=True)
    ap.add_argument("--nvr-user", required=True)
    ap.add_argument("--nvr-prefix", required=True)
    ap.add_argument("--channels", required=True)
    ap.add_argument("--models", required=True, help="comma list; cameras get them round-robin, one model each")
    ap.add_argument("--rtsp-template", default="rtsp://{ip}:554/cam/realmonitor?channel={ch}&subtype={sub}")
    ap.add_argument("--subtype", type=int, default=0)
    ap.add_argument("--stream-mode", default="annotated")
    ap.add_argument("--start", action="store_true")
    a = ap.parse_args()
    c = httpx.Client(base_url=a.api, verify=a.ca or True, timeout=30)
    r = c.post("/api/v1/auth/token", json={"username": a.user, "password": os.environ["VMS_PASSWORD"]})
    r.raise_for_status()
    c.headers["Authorization"] = f"Bearer {r.json()['access_token']}"
    models = a.models.split(",")
    rc = 0
    for i, ch in enumerate(parse_channels(a.channels)):
        cid = f"{a.nvr_prefix}_CH{ch:02d}"
        body = {"id": cid, "name": f"{a.nvr_prefix} channel {ch}", "stream_mode": a.stream_mode,
                "rtsp_url": a.rtsp_template.format(ip=a.nvr_ip, ch=ch, sub=a.subtype),
                "username": a.nvr_user, "password": os.environ["NVR_PASSWORD"]}
        x = c.post("/api/v1/cameras", json=body)
        print(cid, "create", x.status_code, "" if x.status_code < 300 else x.text[:200])
        if x.status_code >= 300 and x.status_code != 409:
            rc = 1
            continue
        m = models[i % len(models)]
        x = c.post(f"/api/v1/cameras/{cid}/models", json={"models": [{"model_id": m}]})
        print(cid, "models", m, x.status_code, "" if x.status_code < 300 else x.text[:200])
        if x.status_code >= 300:
            rc = 1
        if a.start:
            x = c.post(f"/api/v1/cameras/{cid}/start")
            print(cid, "start", x.status_code, x.text[:160] if x.status_code >= 300 else "")
    return rc


if __name__ == "__main__":
    sys.exit(main())
