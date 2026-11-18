"""Interactive: list the registered cameras, choose which to delete, confirm, delete.

    python scripts/delete_camera.py
    delete_camera.bat                 # same, double-click on Windows

A running camera is stopped first (its pipeline is removed), then deleted. Other cameras keep running.
"""
from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
API = os.environ.get("VMS_API", "https://localhost:8443")
G, Y, R, C, D, B, X = "\033[32m", "\033[33m", "\033[31m", "\033[36m", "\033[2m", "\033[1m", "\033[0m"
if os.name == "nt":
    os.system("")


def load_env() -> dict[str, str]:
    env: dict[str, str] = {}
    f = ROOT / ".env"
    if f.exists():
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env


def model_text(c: dict) -> str:
    ms = sorted(c.get("models", []), key=lambda m: m.get("order", 0))
    return " + ".join(m["model_id"].replace("_detection", "") for m in ms) or "-"


def host_of(c: dict) -> str:
    m = re.match(r"rtsp[s]?://(?:[^@/]*@)?([^:/]+)", c.get("rtsp_url") or "")
    return m.group(1) if m else "-"


def label(c: dict) -> str:
    return f"{c['id']}  \"{c.get('name') or c['id']}\"  ({host_of(c)})"


def parse_selection(text: str, n: int, ids: list[str] | None = None) -> list[int]:
    """'all', '1,3,5-7' (1-based numbers) or camera ids -> sorted list of 0-based indexes."""
    text = text.strip().lower()
    if text in ("a", "all"):
        return list(range(n))
    out: set[int] = set()
    lowered = [i.lower() for i in (ids or [])]
    for part in text.replace(" ", "").split(","):
        if part in lowered:  # typed the camera id
            out.add(lowered.index(part))
            continue
        m = re.fullmatch(r"(\d+)(?:-(\d+))?", part)
        if not m:
            raise ValueError(f"cannot understand '{part}' - type the number or the camera id from the list")
        lo, hi = int(m.group(1)), int(m.group(2) or m.group(1))
        if lo < 1 or hi > n or lo > hi:
            raise ValueError(f"numbers must be between 1 and {n}")
        out.update(range(lo - 1, hi))
    return sorted(out)


def main() -> int:
    env = load_env()
    pw = os.environ.get("VMS_PASSWORD") or env.get("BOOTSTRAP_ADMIN_PASSWORD", "")
    http = httpx.Client(verify=False, timeout=30)
    try:
        r = http.post(f"{API}/api/v1/auth/token", json={"username": env.get("BOOTSTRAP_ADMIN_USERNAME", "admin"), "password": pw})
        r.raise_for_status()
    except Exception:
        print(f"{R}Cannot reach the VMS API at {API}. Start the system first (start_services.bat), then try again.{X}")
        return 1
    h = {"Authorization": f"Bearer {r.json()['access_token']}"}

    def call(method: str, path: str) -> httpx.Response:
        return http.request(method, f"{API}/api/v1{path}", headers=h)

    data = call("GET", "/cameras?limit=500").json()
    cams = sorted(data.get("items", data) if isinstance(data, dict) else data, key=lambda c: c["id"])
    if not cams:
        print("No cameras are registered.")
        return 0

    print(f"\n{B}=== Delete camera ==={X}\n")
    print(f"{B}{'No':>3}  {'CAMERA ID':<12} {'NAME':<22} {'IP ADDRESS':<15} {'STATUS':<9} MODEL{X}")
    for i, c in enumerate(cams, 1):
        col = G if c["status"] == "RUNNING" else Y
        print(f"{i:>3}  {c['id']:<12} {(c.get('name') or '-')[:21]:<22} {host_of(c):<15} {col}{c['status']:<9}{X} {model_text(c)}")

    all_ids = [c["id"] for c in cams]
    while True:
        text = input(f"\n{B}Which camera(s) to delete?{X} [number like 2, or 1,3 or 2-4, or camera id, or all; Enter = cancel]: ").strip()
        if not text:
            print("Cancelled.")
            return 0
        try:
            picked = parse_selection(text, len(cams), all_ids)
            break
        except ValueError as e:
            print(f"  {R}{e}{X}")

    ids = [cams[i]["id"] for i in picked]
    names = {c["id"]: label(c) for c in cams}
    print(f"\n{B}You are about to DELETE {len(ids)} camera(s):{X}")
    for cid in ids:
        print(f"  {R}x{X} {names[cid]}")
    print(f"{D}Their live stream stops and the camera is removed from the system. This cannot be undone.{X}")
    expected = "yes"
    if input(f"{B}Type {expected} to confirm{X}: ").strip().lower() != expected:
        print("Cancelled - nothing was deleted.")
        return 0

    failed = 0
    for cid in ids:
        call("POST", f"/cameras/{cid}/stop")  # ignored if already stopped
        deadline = time.time() + 90
        code = 0
        while time.time() < deadline:
            r = call("DELETE", f"/cameras/{cid}")
            code = r.status_code
            if code in (200, 202, 204, 404):
                break
            if code == 409:  # still running / stopping: wait and retry
                time.sleep(3)
                continue
            break
        if code in (200, 202, 204):
            print(f"  {G}deleted{X}  {names[cid]}")
        elif code == 404:
            print(f"  {Y}already gone{X}  {names[cid]}")
        else:
            failed += 1
            print(f"  {R}could not delete (HTTP {code}){X}  {names[cid]}  {r.text[:150]}")

    left = call("GET", "/cameras?limit=500").json()
    left = left.get("items", left) if isinstance(left, dict) else left
    print(f"\n{G if not failed else Y}Done.{X} {len(left)} camera(s) remain. The live viewer drops deleted tiles when you refresh the page.")
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nCancelled.")
        sys.exit(130)
