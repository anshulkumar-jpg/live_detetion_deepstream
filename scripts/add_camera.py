"""Interactive: add cameras (or all channels of an NVR) and choose which model runs on each.

    python scripts/add_camera.py            # asks: IP -> username -> password -> channels -> model
    add_camera.bat                          # same, double-click on Windows

It finds the working RTSP address automatically, lists the live channels, lets you pick them,
then shows a model menu, registers the cameras and starts them.
The camera password is typed hidden and sent only to your own API (stored encrypted there).
"""
from __future__ import annotations

import argparse
import getpass
import ipaddress
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
G, Y, R, C, D, B, X = "\033[32m", "\033[33m", "\033[31m", "\033[36m", "\033[2m", "\033[1m", "\033[0m"
if os.name == "nt":
    os.system("")

# RTSP address patterns tried in order. {ch} = channel number, {sub} = 0 main stream.
TEMPLATES = [
    ("Dahua / CP Plus / most Indian NVRs", "/cam/realmonitor?channel={ch}&subtype={sub}"),
    ("Hikvision / Hikvision-compatible", "/Streaming/Channels/{ch}0{sub1}"),
    ("Uniview", "/unicast/c{ch}/s{sub}/live"),
    ("Generic (Axis style)", "/axis-media/media.amp"),
]

# Menu: key -> (title, description, [(model_id, parent_model_id or None)], cascade?)
MODELS = {
    "1": ("Person detection", "finds every person (YOLO26s)", [("person_detection", None)], False),
    "2": ("Vehicle detection", "car, motorcycle, bus, truck", [("vehicle_detection", None)], False),
    "3": ("Person + posture", "person boxes labelled STANDING / SITTING / FALLEN (fall alert)",
          [("person_detection", None), ("fall_detection", "person_detection")], True),
    "4": ("Fall detection only", "raw fall model on the full frame (weak, not recommended)", [("fall_detection", None)], False),
}


# Detection frequency. Tracking ALWAYS runs on every frame; this only controls how often the detector runs.
# key -> (frames skipped between detections, title, description)
INTERVALS = {
    "1": (0, "Every frame", "most accurate, heaviest on the GPU"),
    "2": (1, "Skip 1 frame", "detect on every 2nd frame"),
    "3": (2, "Skip 2 frames", "detect on every 3rd frame"),
    "4": (3, "Skip 3 frames", "detect on every 4th frame"),
    "5": (4, "Skip 4 frames", "detect on every 5th frame, lightest"),
}


def load_env() -> dict[str, str]:
    env: dict[str, str] = {}
    f = ROOT / ".env"
    if f.exists():
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env


def ask(prompt: str, default: str | None = None, validate=None) -> str:
    while True:
        suffix = f" [{default}]" if default else ""
        v = input(f"{B}{prompt}{X}{suffix}: ").strip() or (default or "")
        if not v:
            print(f"  {R}Please enter a value.{X}")
            continue
        if validate:
            err = validate(v)
            if err:
                print(f"  {R}{err}{X}")
                continue
        return v


def valid_ip(v: str) -> str | None:
    try:
        ipaddress.ip_address(v)
        return None
    except ValueError:
        return "That is not a valid IP address (example: 192.168.1.100)."


def port_open(ip: str, port: int, timeout: float = 3) -> bool:
    with socket.socket() as s:
        s.settimeout(timeout)
        return s.connect_ex((ip, port)) == 0


def ffprobe(url: str, timeout_s: int = 6) -> str | None:
    """Return 'codec WxH' if the stream opens, else None."""
    exe = shutil.which("ffprobe")
    if not exe:
        sys.exit(f"{R}ffprobe not found. Install ffmpeg (winget install Gyan.FFmpeg) and try again.{X}")
    try:
        r = subprocess.run([exe, "-v", "error", "-rtsp_transport", "tcp", "-timeout", str(timeout_s * 1_000_000),
                            "-select_streams", "v:0", "-show_entries", "stream=codec_name,width,height",
                            "-of", "csv=p=0", url], capture_output=True, text=True, timeout=timeout_s + 8)
    except subprocess.TimeoutExpired:
        return None
    out = r.stdout.strip().splitlines()
    if r.returncode == 0 and out and out[0].count(",") == 2:
        codec, w, h = out[0].split(",")
        return f"{codec} {w}x{h}"
    return None


def rtsp_url(ip: str, port: int, path: str, user: str, pw: str, with_auth: bool) -> str:
    from urllib.parse import quote
    auth = f"{quote(user, safe='')}:{quote(pw, safe='')}@" if with_auth else ""
    return f"rtsp://{auth}{ip}:{port}{path}"


def discover(ip: str, port: int, user: str, pw: str, max_ch: int) -> tuple[str, list[tuple[int, str]]]:
    """Find the working address pattern, then scan channels one by one (NVRs block parallel logins)."""
    chosen = None
    for name, tpl in TEMPLATES:
        for ch in range(1, 5):
            path = tpl.format(ch=ch, sub=0, sub1=1)
            if ffprobe(rtsp_url(ip, port, path, user, pw, True), 5):
                chosen = (name, tpl)
                break
            time.sleep(0.5)
        if chosen:
            break
        # also try just channel 1 pattern variants for single cameras
    if not chosen:
        return "", []
    name, tpl = chosen
    print(f"  {G}Address pattern found:{X} {name}")
    if "{ch}" not in tpl:
        return tpl, [(1, ffprobe(rtsp_url(ip, port, tpl, user, pw, True)) or "")]
    live: list[tuple[int, str]] = []
    print(f"  Scanning channels 1-{max_ch} (one at a time, please wait)...")
    for ch in range(1, max_ch + 1):
        info = ffprobe(rtsp_url(ip, port, tpl.format(ch=ch, sub=0, sub1=1), user, pw, True), 5)
        print(f"    channel {ch:>2}: " + (f"{G}LIVE{X} {info}" if info else f"{D}no video{X}"))
        if info:
            live.append((ch, info))
        time.sleep(0.7)
    return tpl, live


def parse_selection(text: str, available: list[int]) -> list[int]:
    text = text.strip().lower()
    if text in ("", "all", "a"):
        return available
    chosen: set[int] = set()
    for part in text.replace(" ", "").split(","):
        m = re.fullmatch(r"(\d+)(?:-(\d+))?", part)
        if not m:
            raise ValueError(f"cannot understand '{part}'")
        lo, hi = int(m.group(1)), int(m.group(2) or m.group(1))
        chosen.update(range(lo, hi + 1))
    bad = sorted(chosen - set(available))
    if bad:
        raise ValueError(f"channel(s) {bad} are not live")
    return sorted(chosen)


def model_menu(title: str) -> str:
    print(f"\n{B}{title}{X}")
    for k, (name, desc, *_rest) in MODELS.items():
        print(f"  {C}{k}{X}) {name:<20} {D}{desc}{X}")
    while True:
        v = input(f"{B}Choose model [1-{len(MODELS)}]{X}: ").strip()
        if v in MODELS:
            return v
        print(f"  {R}Type a number from the list.{X}")


def interval_menu(title: str) -> int:
    print(f"\n{B}{title}{X}")
    print(f"{D}Tracking (person IDs) always runs on every frame. This chooses how often the detector runs.{X}")
    for k, (_, name, desc) in INTERVALS.items():
        print(f"  {C}{k}{X}) {name:<14} {D}{desc}{X}")
    while True:
        v = input(f"{B}Choose [1-{len(INTERVALS)}]{X} (Enter = 1): ").strip() or "1"
        if v in INTERVALS:
            return INTERVALS[v][0]
        print(f"  {R}Type a number from the list.{X}")


class Api:
    def __init__(self, base: str, user: str, password: str) -> None:
        self.base, self.http = base, httpx.Client(verify=False, timeout=30)
        r = self.http.post(f"{base}/api/v1/auth/token", json={"username": user, "password": password})
        r.raise_for_status()
        self.h = {"Authorization": f"Bearer {r.json()['access_token']}"}

    def call(self, method: str, path: str, **kw) -> httpx.Response:
        return self.http.request(method, f"{self.base}/api/v1{path}", headers=self.h, **kw)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="https://localhost:8443")
    ap.add_argument("--max-channels", type=int, default=0, help="channels to scan (asked if not given)")
    a = ap.parse_args()

    env = load_env()
    admin_pw = os.environ.get("VMS_PASSWORD") or env.get("BOOTSTRAP_ADMIN_PASSWORD", "")
    try:
        api = Api(a.api, env.get("BOOTSTRAP_ADMIN_USERNAME", "admin"), admin_pw)
    except Exception:
        sys.exit(f"{R}Cannot reach the VMS API at {a.api}. Start the system first (run_all.bat), then try again.{X}")

    print(f"\n{B}=== Add camera / NVR ==={X}\n")
    ip = ask("1) Camera or NVR IP address", validate=valid_ip)
    port = int(ask("   RTSP port", "554", lambda v: None if v.isdigit() else "Port must be a number."))
    if not port_open(ip, port):
        print(f"  {Y}Warning: {ip}:{port} did not answer. Check the IP and that this PC is on the same network.{X}")
        if input("  Continue anyway? [y/N]: ").strip().lower() != "y":
            return 1
    user = ask("2) Camera username", "admin")
    pw = os.environ.get("NVR_PASSWORD") or getpass.getpass(f"{B}3) Camera password{X} (hidden): ")
    if not pw:
        sys.exit(f"{R}Password is empty.{X}")

    max_ch = a.max_channels or int(ask("   How many channels to scan (1 for a single camera)", "16",
                                      lambda v: None if v.isdigit() and 0 < int(v) <= 128 else "Enter 1-128."))
    print(f"\n{C}Checking the camera...{X}")
    tpl, live = discover(ip, port, user, pw, max_ch)
    if not live:
        print(f"\n{R}No working video stream found.{X} Check the IP, username/password, and that RTSP is enabled on the device.")
        return 1
    print(f"\n{G}{len(live)} live channel(s):{X} " + ", ".join(f"{c} ({i})" for c, i in live))
    print(f"{D}Note: an NVR may also expose its own multi-screen view as a channel - skip it if a channel shows a grid of cameras.{X}")
    while True:
        try:
            picked = parse_selection(input(f"{B}Which channels to add?{X} [all, or e.g. 2,3,5-7]: "), [c for c, _ in live])
            break
        except ValueError as e:
            print(f"  {R}{e}{X}")
    default_prefix = "CAM" + ip.split(".")[-1] if len(live) == 1 and max_ch == 1 else "NVR" + ip.split(".")[-1]
    prefix = ask("   Name prefix for the cameras", default_prefix,
                 lambda v: None if re.fullmatch(r"[A-Za-z0-9_.-]{1,40}", v) else "Use letters, numbers, _ . - only.")

    mode = "1"
    if len(picked) > 1:
        print(f"\n{B}Model for these {len(picked)} cameras:{X}")
        print(f"  {C}1{X}) Same model on all of them")
        print(f"  {C}2{X}) Choose a model for each camera")
        mode = ask("Choose", "1", lambda v: None if v in ("1", "2") else "Type 1 or 2.")
    plan: dict[int, str] = {}
    if mode == "1":
        key = model_menu("Which model should run on your camera(s)?")
        plan = {ch: key for ch in picked}
    else:
        for ch in picked:
            plan[ch] = model_menu(f"Model for channel {ch}")

    skip: dict[int, int] = {}
    if len(picked) > 1 and ask("\nUse the same detection frequency for all cameras? (y/n)", "y",
                               lambda v: None if v.lower() in ("y", "n") else "Type y or n.").lower() == "n":
        for ch in picked:
            skip[ch] = interval_menu(f"Detection frequency for channel {ch}")
    else:
        every = interval_menu("How often should detection run?")
        skip = {ch: every for ch in picked}

    # check models exist in the system
    have = {m["id"] for m in api.call("GET", "/models").json()}
    for ch, key in plan.items():
        missing = [m for m, _ in MODELS[key][2] if m not in have]
        if missing:
            sys.exit(f"{R}Model(s) {missing} are not registered. Run run_all.bat first.{X}")

    print(f"\n{B}Summary{X}")
    for ch in picked:
        n = skip[ch]
        print(f"  {prefix}_CH{ch:02d}  ->  {MODELS[plan[ch]][0]}, detection "
              + ("every frame" if n == 0 else f"every {n + 1}th frame (skip {n})").replace("1th", "1st") + ", tracking every frame")
    if input(f"\n{B}Add and start these cameras? [Y/n]{X}: ").strip().lower() == "n":
        print("Cancelled.")
        return 0

    for ch in picked:
        cid = f"{prefix}_CH{ch:02d}"
        _, _, items, cascade = MODELS[plan[ch]]
        body = {"id": cid, "name": f"{prefix} channel {ch}", "stream_mode": "annotated",
                "rtsp_url": rtsp_url(ip, port, tpl.format(ch=ch, sub=0, sub1=1), user, pw, False),
                "username": user, "password": pw}
        r = api.call("POST", "/cameras", json=body)
        if r.status_code == 409:
            print(f"  {Y}{cid} already exists - updating its models{X}")
        elif r.status_code >= 300:
            print(f"  {R}{cid}: could not add ({r.status_code}) {r.text[:150]}{X}")
            continue
        req = {"models": [{"model_id": m, **({"parent_model_id": p, "operate_on_class_ids": [0]} if p else {})}
                          for m, p in items], "tracking_enabled": True, "detection_interval": skip[ch]}
        if cascade:
            req.update({"execution_strategy": "cascade"})
        r = api.call("POST", f"/cameras/{cid}/models", json=req)
        if r.status_code >= 300:
            print(f"  {R}{cid}: models not set ({r.status_code}) {r.text[:150]}{X}")
            continue
        r = api.call("POST", f"/cameras/{cid}/start")
        print(f"  {G}{cid}: added, model = {MODELS[plan[ch]][0]}, starting{X}" if r.status_code < 300
              else f"  {R}{cid}: start failed ({r.status_code}){X}")

    print(f"\n{G}Done.{X} Watch live: http://localhost:8080   |   status table: python scripts/run_all.py --status")
    print(f"{D}A new camera can take 1-2 minutes to show video.{X}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nCancelled.")
        sys.exit(130)
