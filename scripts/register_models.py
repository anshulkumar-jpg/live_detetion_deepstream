"""Register (or add versions of) models from a YAML file through the API.

    python scripts/register_models.py --api https://vms:8443 --user admin config/models/models.yaml
Password is read from VMS_PASSWORD or prompted. Idempotent: existing versions are skipped.
"""
from __future__ import annotations

import argparse
import getpass
import os
import sys

import httpx
import yaml


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--user", default="admin")
    ap.add_argument("--ca", default=None, help="CA bundle for the gateway TLS certificate")
    args = ap.parse_args()
    pw = os.environ.get("VMS_PASSWORD") or getpass.getpass("password: ")
    c = httpx.Client(base_url=args.api, verify=args.ca or True, timeout=30)
    tok = c.post("/api/v1/auth/token", json={"username": args.user, "password": pw})
    tok.raise_for_status()
    c.headers["Authorization"] = f"Bearer {tok.json()['access_token']}"
    spec = yaml.safe_load(open(args.file))
    rc = 0
    for m in spec.get("models", []):
        r = c.get(f"/api/v1/models/{m['id']}")
        if r.status_code == 404:
            r = c.post("/api/v1/models", json=m)
            print(f"{m['id']}: create -> {r.status_code} {'' if r.is_success else r.text}")
        else:
            existing = {v["version"] for v in r.json()["versions"]}
            v = m.get("version")
            if v and v["version"] not in existing:
                r = c.put(f"/api/v1/models/{m['id']}", json={"new_version": v})
                print(f"{m['id']}: new version {v['version']} -> {r.status_code} {'' if r.is_success else r.text}")
            else:
                print(f"{m['id']}: up to date")
                continue
        rc |= 0 if r.is_success else 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
