"""Download a few seconds of an annotated HLS stream through the gateway and save it as mp4.
usage: python scripts/grab_hls_clip.py CAMERA_ID out.mp4 [--api https://localhost:8443] [--segments 4]
Needs VMS_PASSWORD in the environment and ffmpeg on PATH.
"""
import argparse, os, re, subprocess, sys, tempfile, time, urllib.parse
import httpx

ap = argparse.ArgumentParser()
ap.add_argument("camera"); ap.add_argument("out")
ap.add_argument("--api", default="https://localhost:8443"); ap.add_argument("--user", default="admin")
ap.add_argument("--segments", type=int, default=4)
a = ap.parse_args()
c = httpx.Client(verify=False, timeout=120)
tok = c.post(f"{a.api}/api/v1/auth/token", json={"username": a.user, "password": os.environ["VMS_PASSWORD"]}).json()["access_token"]
base = f"{a.api}/hls/cam/{a.camera}/processed/"
def get(u): return c.get(u, params={"jwt": tok} if "jwt=" not in u else None)
master = get(base + "index.m3u8").text
sub = next(l for l in master.splitlines() if l and not l.startswith("#"))
seen, blob, init = set(), b"", None
deadline = time.time() + 40
while len(seen) < a.segments and time.time() < deadline:
    pl = get(base + sub).text
    if init is None:
        m = re.search(r'#EXT-X-MAP:URI="([^"]+)"', pl)
        if m: init = get(base + m.group(1)).content; blob += init
    for l in pl.splitlines():
        if l and not l.startswith("#") and l not in seen and len(seen) < a.segments:
            r = get(base + l)
            if r.status_code == 200: blob += r.content; seen.add(l)
    time.sleep(1)
if len(seen) == 0: sys.exit("no segments received")
with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f: f.write(blob); tmp = f.name
subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", tmp, "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-an", a.out], check=True)
os.unlink(tmp); print("saved", a.out, len(seen), "segments")
