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

try:
    from scripts._tls import tls_verify
except ImportError:  # run as "python scripts/x.py"
    from _tls import tls_verify

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


# Tracking algorithm. key -> (tracker_type for the API, title, description, runs in python?)
TRACKERS = {
    "1": ("NvDCF", "NvDCF", "DeepStream GPU tracker, follows objects by appearance even when detection is skipped (default)", False),
    "2": ("NvSORT", "SORT", "fast and simple, motion only", False),
    "3": ("IOU", "IOU", "simplest and lightest", False),
    "4": ("NvDeepSORT", "DeepSORT", "uses appearance, needs a Re-ID model on the camera", False),
    "5": ("ByteTrack", "ByteTrack", "great for crowds, keeps low-confidence boxes alive", True),
    "6": ("OCSORT", "OC-SORT", "best through occlusion and crossing paths", True),
    "7": ("BoTSORT", "BoT-SORT", "ByteTrack style + appearance when a Re-ID model is available", True),
    "8": ("StrongSORT", "StrongSORT", "DeepSORT style + appearance when a Re-ID model is available", True),
}


PY_TRACKERS = {v[0] for v in TRACKERS.values() if v[3]}


def load_env() -> dict[str, str]:
    env: dict[str, str] = {}
    f = ROOT / ".env"
    if f.exists():
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env


def read_password(prompt: str) -> str:
    """Password prompt that shows * for every key you type (so typos are visible); plain hidden prompt elsewhere."""
    if os.name == "nt" and sys.stdin.isatty():
        import msvcrt
        print(prompt, end="", flush=True)
        buf: list[str] = []
        while True:
            ch = msvcrt.getwch()
            if ch in ("\r", "\n"):
                print()
                return "".join(buf)
            if ch == "\x03":
                raise KeyboardInterrupt
            if ch == "\b":
                if buf:
                    buf.pop()
                    print("\b \b", end="", flush=True)
            elif ch in ("\x00", "\xe0"):
                msvcrt.getwch()  # arrow / function key: ignore
            else:
                buf.append(ch)
                print("*", end="", flush=True)
    return getpass.getpass(prompt)


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


def probe(url: str, timeout_s: int = 6) -> tuple[str | None, str]:
    """Open one RTSP stream. Returns (info, reason): info is 'codec WxH' on success; reason is
    ok | auth | forbidden | notfound | network | nodata | other."""
    exe = shutil.which("ffprobe")
    if not exe:
        sys.exit(f"{R}ffprobe not found. Install ffmpeg (winget install Gyan.FFmpeg) and try again.{X}")
    try:
        r = subprocess.run([exe, "-v", "error", "-rtsp_transport", "tcp", "-timeout", str(timeout_s * 1_000_000),
                            "-select_streams", "v:0", "-show_entries", "stream=codec_name,width,height",
                            "-of", "csv=p=0", url], capture_output=True, text=True, timeout=timeout_s + 8)
    except subprocess.TimeoutExpired:
        return None, "network"
    out = r.stdout.strip().splitlines()
    if r.returncode == 0 and out and out[0].count(",") == 2:
        codec, w, h = out[0].split(",")
        return f"{codec} {w}x{h}", "ok"
    err = (r.stderr or "").lower()
    if "401" in err or "unauthorized" in err:
        return None, "auth"
    if "403" in err or "forbidden" in err:
        return None, "forbidden"
    if "404" in err or "not found" in err:
        return None, "notfound"
    if any(t in err for t in ("refused", "timed out", "timeout", "no route", "unreachable", "network")):
        return None, "network"
    if "invalid data" in err:
        return None, "nodata"
    return None, "other"


def ffprobe(url: str, timeout_s: int = 6) -> str | None:
    return probe(url, timeout_s)[0]


def rtsp_url(ip: str, port: int, path: str, user: str, pw: str, with_auth: bool) -> str:
    from urllib.parse import quote
    auth = f"{quote(user, safe='')}:{quote(pw, safe='')}@" if with_auth else ""
    return f"rtsp://{auth}{ip}:{port}{path}"


def discover(ip: str, port: int, user: str, pw: str, max_ch: int) -> tuple[str, list[tuple[int, str]], str | None]:
    """Find the working address pattern, then scan channels one by one (NVRs block parallel logins).

    Returns (pattern, live_channels, problem). problem is None, 'auth', 'network' or 'nopattern'.
    The scan stops at the FIRST rejected login: repeating a wrong password makes many NVRs lock the account.
    """
    chosen = None
    for name, tpl in TEMPLATES:
        for ch in range(1, 5):
            info, why = probe(rtsp_url(ip, port, tpl.format(ch=ch, sub=0, sub1=1), user, pw, True), 5)
            if why == "auth":
                return "", [], "auth"          # wrong login: stop immediately, do not try more
            if why == "network" and ch == 1 and tpl == TEMPLATES[0][1]:
                return "", [], "network"       # nothing answers on this address
            if info:
                chosen = (name, tpl)
                break
            time.sleep(0.5)
        if chosen:
            break
    if not chosen:
        return "", [], "nopattern"
    name, tpl = chosen
    print(f"  {G}Address pattern found:{X} {name}")
    if "{ch}" not in tpl:
        return tpl, [(1, ffprobe(rtsp_url(ip, port, tpl, user, pw, True)) or "")], None
    live: list[tuple[int, str]] = []
    print(f"  Scanning channels 1-{max_ch} (one at a time, please wait)...")
    for ch in range(1, max_ch + 1):
        info, why = probe(rtsp_url(ip, port, tpl.format(ch=ch, sub=0, sub1=1), user, pw, True), 5)
        print(f"    channel {ch:>2}: " + (f"{G}LIVE{X} {info}" if info else f"{D}no video{X}"))
        if why == "auth":
            print(f"  {Y}The NVR started rejecting the login during the scan - stopping so it does not lock the account.{X}")
            return tpl, live, ("auth" if not live else None)
        if info:
            live.append((ch, info))
        time.sleep(0.7)
    return tpl, live, None


def explain_problem(problem: str | None, ip: str, port: int) -> None:
    if problem == "auth":
        print(f"\n{R}The device rejected the username or password (401 Unauthorized).{X}")
        print("  - Only ONE login was tried, so this tool did not cause a lock.")
        print(f"  - {Y}Careful:{X} many NVRs lock the account for 10-30 minutes after a few wrong logins.")
        print("    If you are sure the password is right, wait a while, or unlock/reset the user in the NVR web page")
        print(f"    (open http://{ip} in a browser > user management), then run this again.")
    elif problem == "network":
        print(f"\n{R}Cannot reach {ip}:{port}.{X} Check the IP, the RTSP port, and that this PC is on the same network.")
    else:
        print(f"\n{R}Logged in, but no video stream was found.{X} RTSP may be disabled on the device, or it uses an address")
        print("  pattern this tool does not know. Check the RTSP settings in the NVR/camera web page.")


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


def model_classes(api, model_id: str) -> list[tuple[int, str]]:
    """Classes of a model that a camera may choose from (those the model does not already filter out)."""
    r = api.call("GET", f"/models/{model_id}")
    v = (r.json().get("active_version") or {}) if r.status_code == 200 else {}
    props, labels = v.get("nvinfer_properties") or {}, v.get("labels") or []
    n = int(props.get("num-detected-classes") or len(labels) or 0)
    raw = props.get("filter-out-class-ids")
    out = {int(x) for x in str(raw).replace(",", ";").split(";") if x.strip()} if raw not in (None, "") else set()
    return [(i, labels[i] if i < len(labels) else str(i)) for i in range(n) if i not in out]


def model_defaults(api, model_id: str) -> tuple[float, float]:
    """The model's registered confidence and NMS IoU (what is used when the camera sets nothing)."""
    r = api.call("GET", f"/models/{model_id}")
    v = (r.json().get("active_version") or {}) if r.status_code == 200 else {}
    a = (v.get("class_attrs") or {}).get("all", {})
    return float(a.get("pre-cluster-threshold", 0.25)), float(a.get("nms-iou-threshold", 0.45))


def ask_float(prompt: str, default: float, lo: float, hi: float) -> float:
    while True:
        text = input(f"{B}{prompt}{X} [{lo:g}-{hi:g}] (Enter = {default:g}): ").strip()
        if not text:
            return default
        try:
            v = float(text)
            if lo <= v <= hi:
                return round(v, 3)
        except ValueError:
            pass
        print(f"  {R}Enter a number between {lo:g} and {hi:g}, for example {default:g}.{X}")


def class_menu(title: str, classes: list[tuple[int, str]]) -> list[int] | None:
    """Ask which classes to use. Returns the chosen class ids, or None for 'all'."""
    names = ", ".join(n for _, n in classes)
    print(f"\n{B}{title}{X}")
    print(f"  {C}1{X}) All of them  {D}({names}){X}")
    print(f"  {C}2{X}) Only some of them")
    if ask("Choose", "1", lambda v: None if v in ("1", "2") else "Type 1 or 2.") == "1":
        return None
    print("  " + "    ".join(f"{C}{k}{X}) {n}" for k, (_, n) in enumerate(classes, 1)))
    while True:
        text = input(f"{B}Enter the numbers you want{X} (for example 1,3): ").strip()
        try:
            picked = [n for n in parse_selection(text, list(range(1, len(classes) + 1)))]
            if text and picked:
                ids = [classes[n - 1][0] for n in picked]
                return None if len(ids) == len(classes) else ids
        except ValueError:
            pass
        print(f"  {R}Type numbers from the list above, like 1,3 or 2-3.{X}")


def model_menu(title: str) -> str:
    print(f"\n{B}{title}{X}")
    for k, (name, desc, *_rest) in MODELS.items():
        print(f"  {C}{k}{X}) {name:<20} {D}{desc}{X}")
    while True:
        v = input(f"{B}Choose model [1-{len(MODELS)}]{X}: ").strip()
        if v in MODELS:
            return v
        print(f"  {R}Type a number from the list.{X}")


def tracker_menu(title: str) -> str:
    print(f"\n{B}{title}{X}")
    for k, (_, name, desc, py) in TRACKERS.items():
        tag = f"{Y}python{X}" if py else f"{G}GPU{X}   "
        print(f"  {C}{k}{X}) {name:<11} [{tag}] {D}{desc}{X}")
    print(f"{D}[python] trackers run in the worker and need detection on every frame (the detection-frequency question is skipped).{X}")
    while True:
        v = input(f"{B}Choose tracker [1-{len(TRACKERS)}]{X} (Enter = 1): ").strip() or "1"
        if v in TRACKERS:
            return TRACKERS[v][0]
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
        self.base, self.http = base, httpx.Client(verify=tls_verify(), timeout=30)
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
    pw = os.environ.get("NVR_PASSWORD") or read_password(f"{B}3) Camera password{X} (shown as ****, check the length): ")
    if not pw:
        sys.exit(f"{R}Password is empty.{X}")

    max_ch = a.max_channels or int(ask("   How many channels to scan (1 for a single camera)", "16",
                                      lambda v: None if v.isdigit() and 0 < int(v) <= 128 else "Enter 1-128."))
    retried = False
    while True:
        print(f"\n{C}Checking the camera...{X}")
        tpl, live, problem = discover(ip, port, user, pw, max_ch)
        if live:
            break
        explain_problem(problem, ip, port)
        if (problem == "auth" and not retried and not os.environ.get("NVR_PASSWORD")
                and input(f"\n{B}Re-enter the password and try once more? [y/N]{X}: ").strip().lower() == "y"):
            retried = True
            pw = read_password(f"{B}Camera password{X} (shown as ****, check the length): ")
            continue
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

    # Tell the user BEFORE asking about models which of these channels are already registered.
    existing = {c["id"] for c in (api.call("GET", "/cameras", params={"limit": 1000}).json() or {}).get("items", [])}
    dup = [ch for ch in picked if f"{prefix}_CH{ch:02d}" in existing]
    if dup:
        print(f"\n{Y}Already added (camera id exists):{X} " + ", ".join(f"channel {ch} = {prefix}_CH{ch:02d}" for ch in dup))
        if input(f"{B}Change the models/settings of these existing cameras? [y/N]{X}: ").strip().lower() != "y":
            picked = [ch for ch in picked if ch not in dup]
            if not picked:
                print("Nothing new to add. Cancelled.")
                return 0
            print(f"{D}Skipping them; continuing with the new channels: {', '.join(map(str, picked))}{X}")

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

    trk: dict[int, str] = {}
    if len(picked) > 1 and ask("\nUse the same tracker for all cameras? (y/n)", "y",
                               lambda v: None if v.lower() in ("y", "n") else "Type y or n.").lower() == "n":
        for ch in picked:
            trk[ch] = tracker_menu(f"Tracker for channel {ch}")
    else:
        one = tracker_menu("Which tracking algorithm should run on your camera(s)?")
        trk = {ch: one for ch in picked}

    skip: dict[int, int] = {}
    native = [ch for ch in picked if trk[ch] not in PY_TRACKERS]
    if not native:
        skip = {ch: 0 for ch in picked}
    elif len(native) > 1 and ask("\nUse the same detection frequency for all cameras? (y/n)", "y",
                               lambda v: None if v.lower() in ("y", "n") else "Type y or n.").lower() == "n":
        skip = {ch: 0 for ch in picked}
        for ch in native:
            skip[ch] = interval_menu(f"Detection frequency for channel {ch}")
    else:
        every = interval_menu("How often should detection run?")
        skip = {ch: (every if ch in native else 0) for ch in picked}

    # check models exist in the system
    have = {m["id"] for m in api.call("GET", "/models").json()}
    for ch, key in plan.items():
        missing = [m for m, _ in MODELS[key][2] if m not in have]
        if missing:
            sys.exit(f"{R}Model(s) {missing} are not registered. Run run_all.bat first.{X}")

    # which classes of each model should be detected (asked only when a model offers more than one class)
    info = {m: model_classes(api, m) for key in set(plan.values()) for m, _ in MODELS[key][2]}
    det: dict[int, dict[str, list[int] | None]] = {ch: {} for ch in picked}

    def ask_for(key: str, who: str) -> dict[str, list[int] | None]:
        out: dict[str, list[int] | None] = {}
        for m, parent in MODELS[key][2]:
            classes = info[m]
            if len(classes) < 2:
                continue
            title = (f"Which postures should be highlighted on each person{who}?" if parent
                     else f"Which classes should {m.replace('_', ' ')} detect{who}?")
            out[m] = class_menu(title, classes)
        return out

    if mode == "1":
        shared = ask_for(plan[picked[0]], "")
        for ch in picked:
            det[ch] = dict(shared)
    else:
        for ch in picked:
            det[ch] = ask_for(plan[ch], f" on channel {ch}")

    defaults = {m: model_defaults(api, m) for m in info}
    sens: dict[int, dict[str, tuple[float, float | None]]] = {ch: {} for ch in picked}

    def ask_sens(key: str, who: str) -> dict[str, tuple[float, float | None]]:
        items_ = MODELS[key][2]
        shown = ", ".join(f"{m.replace('_detection', '')} conf {defaults[m][0]:g}" + ("" if p else f" / IoU {defaults[m][1]:g}")
                          for m, p in items_)
        print(f"\n{B}Detection sensitivity{who}{X}")
        print(f"{D}Confidence = how sure the model must be before a box is shown. Lower finds more, with more false alarms.{X}")
        print(f"{D}IoU = how much two boxes may overlap before they count as one object. Lower merges more, higher keeps crowds apart.{X}")
        print(f"  {C}1{X}) Recommended settings  {D}({shown}){X}")
        print(f"  {C}2{X}) Set my own confidence and IoU")
        if ask("Choose", "1", lambda v: None if v in ("1", "2") else "Type 1 or 2.") == "1":
            return {}
        out_: dict[str, tuple[float, float | None]] = {}
        for m, parent in items_:
            name = m.replace("_", " ")
            if parent:
                conf = ask_float(f"Confidence for {name} (how sure before a posture label is shown)", defaults[m][0], 0.01, 0.99)
                out_[m] = (conf, None)
            else:
                conf = ask_float(f"Confidence for {name}", defaults[m][0], 0.01, 0.99)
                iou = ask_float(f"IoU (NMS overlap) for {name}", defaults[m][1], 0.05, 0.95)
                out_[m] = (conf, iou)
        return out_

    if mode == "1":
        shared_s = ask_sens(plan[picked[0]], "")
        for ch in picked:
            sens[ch] = dict(shared_s)
    else:
        for ch in picked:
            sens[ch] = ask_sens(plan[ch], f" on channel {ch}")

    def names_of(model_id: str, ids: list[int] | None) -> str:
        names = dict(info[model_id])
        return "all" if ids is None else ",".join(names.get(i, str(i)) for i in ids)

    print(f"\n{B}Summary{X}")
    for ch in picked:
        n = skip[ch]
        tname = next(v[1] for v in TRACKERS.values() if v[0] == trk[ch])
        print(f"  {prefix}_CH{ch:02d}  ->  {MODELS[plan[ch]][0]}, detection "
              + ("every frame" if n == 0 else f"every {n + 1}th frame (skip {n})").replace("1th", "1st")
              + f", tracker {tname}")
        for m, ids in det[ch].items():
            print(f"      {D}{m.replace('_', ' ')} classes:{X} {names_of(m, ids)}")
        for m, (c_, i_) in sens[ch].items():
            print(f"      {D}{m.replace('_', ' ')} sensitivity:{X} confidence {c_:g}" + (f", IoU {i_:g}" if i_ is not None else ""))
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
            print(f"  {Y}{cid} already exists - its models are being updated{X}")
        elif r.status_code >= 300:
            print(f"  {R}{cid}: could not add ({r.status_code}) {r.text[:150]}{X}")
            continue
        req = {"models": [{"model_id": m, **({"parent_model_id": p, "operate_on_class_ids": [0]} if p else {}),
                           **({"detect_class_ids": det[ch][m]} if det[ch].get(m) else {}),
                           **({"conf_threshold": sens[ch][m][0]} if m in sens[ch] else {}),
                           **({"iou_threshold": sens[ch][m][1]} if m in sens[ch] and sens[ch][m][1] is not None else {})}
                          for m, p in items], "tracking_enabled": True, "tracker_type": trk[ch],
                          "detection_interval": skip[ch]}
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
