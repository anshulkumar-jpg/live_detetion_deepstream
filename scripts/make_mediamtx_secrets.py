"""Create the logins that protect the MediaMTX control API and its metrics.

    python scripts/make_mediamtx_secrets.py

Safe to run again: it only creates what is missing.
  .env                                      MEDIAMTX_API_PASSWORD (API + scheduler), MEDIAMTX_METRICS_PASSWORD (API checks it)
  deploy/prometheus/secrets/mediamtx_metrics  the metrics password as a file, for Prometheus (password_file)
"""
from __future__ import annotations

import secrets
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV = ROOT / ".env"
SECRETS = ROOT / "deploy" / "prometheus" / "secrets"


def main() -> int:
    raw = ENV.read_bytes().decode("utf-8") if ENV.exists() else ""
    nl = "\r\n" if "\r\n" in raw else "\n"
    lines = raw.split(nl) if raw else []
    added = []

    def secret(name: str) -> str:
        for line in lines:
            if line.startswith(name + "=") and line.split("=", 1)[1].strip():
                return line.split("=", 1)[1].strip()
        value = secrets.token_urlsafe(32)
        lines.append(f"{name}={value}")
        added.append(name)
        return value

    secret("MEDIAMTX_API_PASSWORD")
    metrics = secret("MEDIAMTX_METRICS_PASSWORD")
    SECRETS.mkdir(parents=True, exist_ok=True)
    (SECRETS / ".gitkeep").touch()
    (SECRETS / "mediamtx_metrics").write_text(metrics, encoding="utf-8")
    if added:
        ENV.write_bytes(nl.join(lines).encode("utf-8"))
        print("added to .env:", ", ".join(added))
    else:
        print(".env already has the MediaMTX logins")
    print("wrote", (SECRETS / "mediamtx_metrics").relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
