"""Local live viewer: one web page with every annotated camera stream, no certificate warnings.

WebRTC (near real-time) is used first, HLS only as a fallback. The page logs in through this small server, which
keeps the token fresh and passes the video requests to the VMS gateway.

usage:  VMS_PASSWORD=... python scripts/live_viewer.py [--port 8080] [--api https://localhost:8443]
then open http://localhost:8080
"""
import argparse
import json
import os
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8080)
ap.add_argument("--api", default="https://localhost:8443")
ap.add_argument("--user", default="admin")
a = ap.parse_args()
client = httpx.Client(verify=False, timeout=httpx.Timeout(60, connect=10), limits=httpx.Limits(max_connections=64))
tok = {"v": None, "t": 0.0}
lock = threading.Lock()


def token() -> str:
    with lock:
        if not tok["v"] or time.time() - tok["t"] > 1800:
            r = client.post(f"{a.api}/api/v1/auth/token", json={"username": a.user, "password": os.environ["VMS_PASSWORD"]})
            r.raise_for_status()
            tok["v"], tok["t"] = r.json()["access_token"], time.time()
        return tok["v"]


def cameras():
    r = client.get(f"{a.api}/api/v1/cameras", params={"limit": 200}, headers={"Authorization": f"Bearer {token()}"})
    d = r.json()
    d = d.get("items", d) if isinstance(d, dict) else d
    return [(c["id"], ",".join(m["model_id"] for m in c["models"])) for c in d if c["status"] == "RUNNING"]


PAGE = """<!doctype html><meta charset=utf-8><title>Live detection</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/hls.js/1.5.13/hls.min.js"></script>
<style>body{margin:0;background:#111;color:#eee;font:13px sans-serif}h3{margin:8px 12px}
#g{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:6px;padding:6px}
.c{position:relative;background:#000}video{width:100%;display:block;background:#000}
.l{position:absolute;left:6px;top:4px;background:#000a;padding:2px 6px;border-radius:3px}
.s{position:absolute;right:6px;top:4px;background:#000a;padding:2px 6px;border-radius:3px;color:#fc6}
.s.ok{color:#6f6}</style>
<h3>Live detection streams <small id=n></small></h3><div id=g></div>
<script>
const tiles = new Map();

class Tile {
  constructor(id, model) {
    this.id = id; this.gen = 0; this.tries = 0; this.pc = null; this.hls = null; this.timers = [];
    const d = this.el = document.createElement('div'); d.className = 'c';
    d.innerHTML = `<video muted autoplay playsinline controls></video><div class=l>${id} &middot; ${model}</div><div class=s>connecting...</div>`;
    this.video = d.querySelector('video'); this.badge = d.querySelector('.s');
    document.getElementById('g').appendChild(d);
    this.connect();
  }
  say(text, ok) { this.badge.textContent = text; this.badge.className = 's' + (ok ? ' ok' : ''); }
  later(fn, ms) { const t = setTimeout(fn, ms); this.timers.push(t); return t; }
  clear() { this.timers.forEach(clearTimeout); this.timers = []; }
  close() {
    this.gen++; this.clear();
    if (this.pc) { try { this.pc.close(); } catch (e) {} this.pc = null; }
    if (this.hls) { try { this.hls.destroy(); } catch (e) {} this.hls = null; }
  }
  remove() { this.close(); this.el.remove(); }

  retry(why) {            // reconnect with a growing pause, never a tight loop
    this.close();
    this.tries++;
    if (this.tries > 3) { this.say('WebRTC unavailable - HLS'); return this.useHls(); }
    this.say(why + ' - reconnecting...');
    this.later(() => this.connect(), Math.min(1000 * this.tries, 5000));
  }

  async connect() {
    const gen = ++this.gen;
    try {
      const pc = this.pc = new RTCPeerConnection({iceServers: []});
      pc.addTransceiver('video', {direction: 'recvonly'});
      pc.ontrack = e => { if (gen === this.gen) { this.video.srcObject = e.streams[0]; this.watch(gen); } };
      pc.onconnectionstatechange = () => {
        if (gen !== this.gen) return;
        const st = pc.connectionState;
        if (st === 'connected') { this.tries = 0; this.clear(); this.say('live', true); this.watch(gen); }
        else if (st === 'failed' || st === 'closed') this.retry('connection lost');
        else if (st === 'disconnected') {      // usually heals by itself - give it a few seconds before rebuilding
          this.later(() => { if (gen === this.gen && pc.connectionState !== 'connected') this.retry('connection lost'); }, 6000);
        }
      };
      await pc.setLocalDescription(await pc.createOffer());
      await new Promise(r => { if (pc.iceGatheringState === 'complete') return r(); pc.onicegatheringstatechange = () => pc.iceGatheringState === 'complete' && r(); setTimeout(r, 1500); });
      const ctl = new AbortController(); const t = setTimeout(() => ctl.abort(), 10000);
      const r = await fetch(`/webrtc/cam/${this.id}/processed/whep`, {method: 'POST', headers: {'Content-Type': 'application/sdp'}, body: pc.localDescription.sdp, signal: ctl.signal});
      clearTimeout(t);
      if (gen !== this.gen) return;
      if (!r.ok) throw new Error('whep ' + r.status);
      await pc.setRemoteDescription({type: 'answer', sdp: await r.text()});
      this.later(() => { if (gen === this.gen && pc.connectionState !== 'connected') this.retry('no connection'); }, 10000);
    } catch (err) {
      if (gen === this.gen) this.retry(err && err.name === 'AbortError' ? 'timeout' : 'error');
    }
  }

  // every 2 s: show fps + delay, and rebuild the stream if the picture froze
  watch(gen) {
    if (this.watching === gen) return; this.watching = gen;
    let last = -1, still = 0;
    const tick = async () => {
      if (gen !== this.gen || !this.pc) return;
      try {
        const stats = await this.pc.getStats();
        stats.forEach(s => {
          if (s.type !== 'inbound-rtp' || s.kind !== 'video') return;
          const delay = s.jitterBufferEmittedCount ? Math.round(1000 * s.jitterBufferDelay / s.jitterBufferEmittedCount) : 0;
          this.say(`live ${Math.round(s.framesPerSecond || 0)} fps - ${delay} ms`, true);
          still = s.framesDecoded === last ? still + 1 : 0; last = s.framesDecoded;
        });
        if (still >= 4) return this.retry('frozen');
      } catch (e) {}
      this.later(tick, 2000);
    };
    this.later(tick, 2000);
  }

  useHls() {
    const src = `/hls/cam/${this.id}/processed/index.m3u8`;
    if (window.Hls && Hls.isSupported()) {
      const h = this.hls = new Hls({lowLatencyMode: true, liveSyncDurationCount: 2, liveMaxLatencyDurationCount: 5,
                                    maxLiveSyncPlaybackRate: 1.3, manifestLoadingMaxRetry: 20, levelLoadingMaxRetry: 20});
      h.loadSource(src); h.attachMedia(this.video);
      h.on(Hls.Events.FRAG_LOADED, () => this.say('live (HLS)', true));
      h.on(Hls.Events.ERROR, (e, x) => { if (x.fatal) { h.destroy(); this.hls = null; this.later(() => this.useHls(), 5000); } });
    } else { this.video.src = src; }
  }
}

async function refresh() {            // add new cameras, drop deleted ones
  try {
    const list = await (await fetch('/cameras')).json();
    const ids = new Set(list.map(x => x[0]));
    for (const [id, model] of list) if (!tiles.has(id)) tiles.set(id, new Tile(id, model));
    for (const [id, t] of tiles) if (!ids.has(id)) { t.remove(); tiles.delete(id); }
    document.getElementById('n').textContent = `(${tiles.size} cameras, WebRTC)`;
  } catch (e) {}
}
refresh(); setInterval(refresh, 10000);
</script>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *x):
        pass

    def send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        u = urllib.parse.urlsplit(self.path)
        if u.path.startswith("/webrtc/") and u.path.endswith("/whep"):
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            try:
                r = client.post(f"{a.api}{u.path}", params={"jwt": token()}, content=body, headers={"Content-Type": "application/sdp"})
            except Exception as e:
                return self.send(504, str(e).encode(), "text/plain")
            return self.send(r.status_code, r.content, "application/sdp")
        self.send(404, b"not found", "text/plain")

    def do_GET(self):
        u = urllib.parse.urlsplit(self.path)
        if u.path == "/":
            return self.send(200, PAGE.encode(), "text/html; charset=utf-8")
        if u.path == "/cameras":
            try:
                return self.send(200, json.dumps(cameras()).encode(), "application/json")
            except Exception as e:
                return self.send(502, str(e).encode(), "text/plain")
        if u.path.startswith("/hls/"):
            q = [(k, v) for k, v in urllib.parse.parse_qsl(u.query) if k != "jwt"] + [("jwt", token())]
            try:  # stream the answer through instead of buffering whole video segments
                with client.stream("GET", f"{a.api}{u.path}", params=q) as r:
                    self.send_response(r.status_code)
                    self.send_header("Content-Type", r.headers.get("content-type", "application/octet-stream"))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    for chunk in r.iter_bytes(65536):
                        self.wfile.write(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as e:
                try:
                    self.send(504, str(e).encode(), "text/plain")
                except Exception:
                    pass
            return
        self.send(404, b"not found", "text/plain")


ThreadingHTTPServer.daemon_threads = True
print(f"Live viewer: http://localhost:{a.port}", flush=True)
ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()
