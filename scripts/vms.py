"""DeepStream VMS command line (Windows, Linux and macOS). Replaces the *.bat scripts.

    python scripts/vms.py init                    create every secret that is missing in .env, and the Kafka / MediaMTX security files
    python scripts/vms.py up [--all] [--analytics] [--monitoring]
    python scripts/vms.py down                    stop (containers kept); add --remove to delete them
    python scripts/vms.py status                  docker compose ps
    python scripts/vms.py logs [service]
    python scripts/vms.py run [args]              start the cameras and the live viewer (scripts/run_all.py)
    python scripts/vms.py rotate-key {jwt|credential|command|publish}   generate a new key in .env (see docs/SECRETS.md)

Secrets are only ever created when missing; existing values are never overwritten, except by rotate-key.
"""
from __future__ import annotations

import argparse
import base64
import os
import secrets
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV = ROOT / ".env"
EXAMPLE = ROOT / ".env.example"

# name -> generator. Everything not listed here (hosts, limits, ...) is taken from .env.example as written.
GENERATED = {
    "POSTGRES_PASSWORD": lambda: secrets.token_urlsafe(24),
    "JWT_SECRET": lambda: secrets.token_urlsafe(48),
    "CREDENTIAL_ENCRYPTION_KEY": lambda: base64.urlsafe_b64encode(os.urandom(32)).decode(),
    "MEDIAMTX_PUBLISH_TOKEN": lambda: secrets.token_urlsafe(32),
    "COMMAND_SIGNING_KEY": lambda: secrets.token_urlsafe(48),
    "BOOTSTRAP_ADMIN_PASSWORD": lambda: secrets.token_urlsafe(18),
    "GRAFANA_ADMIN_PASSWORD": lambda: secrets.token_urlsafe(18),
    "ELASTIC_PASSWORD": lambda: secrets.token_urlsafe(24),
    "KIBANA_SYSTEM_PASSWORD": lambda: secrets.token_urlsafe(24),
    "LOGSTASH_ES_PASSWORD": lambda: secrets.token_urlsafe(24),
}
ROTATABLE = {"jwt": "JWT_SECRET", "credential": "CREDENTIAL_ENCRYPTION_KEY", "command": "COMMAND_SIGNING_KEY",
             "publish": "MEDIAMTX_PUBLISH_TOKEN"}
OLD_SLOT = {"JWT_SECRET": "JWT_SECRET_PREVIOUS", "CREDENTIAL_ENCRYPTION_KEY": "CREDENTIAL_ENCRYPTION_KEYS_OLD"}


def read_env() -> list[str]:
    if ENV.exists():
        return ENV.read_text(encoding="utf-8").replace("\r\n", "\n").split("\n")
    return EXAMPLE.read_text(encoding="utf-8").replace("\r\n", "\n").split("\n") if EXAMPLE.exists() else []


def write_env(lines: list[str]) -> None:
    ENV.write_text("\n".join(lines).rstrip("\n") + "\n", encoding="utf-8")
    try:
        os.chmod(ENV, 0o600)
    except OSError:
        pass


def get_value(lines: list[str], name: str) -> str | None:
    for ln in lines:
        if ln.startswith(name + "="):
            return ln.split("=", 1)[1].strip()
    return None


def set_value(lines: list[str], name: str, value: str) -> None:
    for i, ln in enumerate(lines):
        if ln.startswith(name + "="):
            lines[i] = f"{name}={value}"
            return
    lines.append(f"{name}={value}")


def py(*args: str) -> int:
    return subprocess.call([sys.executable, *args], cwd=ROOT)


def compose(*args: str) -> int:
    return subprocess.call(["docker", "compose", *args], cwd=ROOT)


def cmd_init(_a) -> int:
    lines = read_env()
    made = []
    for name, gen in GENERATED.items():
        cur = get_value(lines, name)
        if not cur or cur.startswith("CHANGE_ME"):
            set_value(lines, name, gen())
            made.append(name)
    write_env(lines)
    print("created in .env:", ", ".join(made) if made else "nothing (all secrets already set)")
    if "BOOTSTRAP_ADMIN_PASSWORD" in made:
        print("first admin sign-in: user 'admin', password =", get_value(lines, "BOOTSTRAP_ADMIN_PASSWORD"), "(change it after sign-in)")
    rc = py("scripts/make_kafka_security.py")
    return rc or py("scripts/make_mediamtx_secrets.py")


def cmd_up(a) -> int:
    if cmd_init(a):
        return 1
    profiles: list[str] = []
    if a.all or a.analytics:
        profiles += ["--profile", "analytics"]
    if a.all or a.monitoring:
        profiles += ["--profile", "monitoring"]
    return compose(*profiles, "up", "-d", *(["--build"] if a.build else []))


def cmd_down(a) -> int:
    prof = ["--profile", "analytics", "--profile", "monitoring"]
    return compose(*prof, "down") if a.remove else compose(*prof, "stop")


def cmd_status(_a) -> int:
    return compose("--profile", "analytics", "--profile", "monitoring", "ps")


def cmd_logs(a) -> int:
    return compose("logs", "--tail", "200", "-f", *([a.service] if a.service else []))


def cmd_run(a) -> int:
    return py("scripts/run_all.py", *a.rest)


def cmd_rotate(a) -> int:
    name = ROTATABLE[a.which]
    lines = read_env()
    old = get_value(lines, name)
    new = GENERATED[name]()
    if name in OLD_SLOT and old:
        keep = get_value(lines, OLD_SLOT[name])
        set_value(lines, OLD_SLOT[name], old if name == "JWT_SECRET" or not keep else f"{old},{keep}")
    set_value(lines, name, new)
    write_env(lines)
    print(f"{name} replaced. Restart the services that use it (python scripts/vms.py up).")
    if name == "CREDENTIAL_ENCRYPTION_KEY":
        print("Then run: python scripts/rotate_credentials.py   and remove CREDENTIAL_ENCRYPTION_KEYS_OLD from .env.")
    if name == "JWT_SECRET":
        print("Everyone stays signed in until their token expires; remove JWT_SECRET_PREVIOUS from .env after that.")
    if name in ("COMMAND_SIGNING_KEY", "MEDIAMTX_PUBLISH_TOKEN"):
        print("Restart the scheduler / API and ALL GPU agents together: commands signed with the old key are refused.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="vms", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init").set_defaults(fn=cmd_init)
    up = sub.add_parser("up")
    up.add_argument("--all", action="store_true")
    up.add_argument("--analytics", action="store_true")
    up.add_argument("--monitoring", action="store_true")
    up.add_argument("--build", action="store_true")
    up.set_defaults(fn=cmd_up)
    dn = sub.add_parser("down")
    dn.add_argument("--remove", action="store_true")
    dn.set_defaults(fn=cmd_down)
    sub.add_parser("status").set_defaults(fn=cmd_status)
    lg = sub.add_parser("logs")
    lg.add_argument("service", nargs="?")
    lg.set_defaults(fn=cmd_logs)
    rn = sub.add_parser("run")
    rn.add_argument("rest", nargs=argparse.REMAINDER)
    rn.set_defaults(fn=cmd_run)
    rk = sub.add_parser("rotate-key")
    rk.add_argument("which", choices=sorted(ROTATABLE))
    rk.set_defaults(fn=cmd_rotate)
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
