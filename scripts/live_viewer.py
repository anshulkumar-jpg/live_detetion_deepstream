"""Local live viewer: one web page with every annotated camera stream (HLS), no certificate warnings.

It logs in to the VMS gateway itself, keeps the token fresh, and proxies HLS from the gateway.
usage:  VMS_PASSWORD=... python scripts/live_viewer.py [--port 8080] [--api https://localhost:8443]
then open http://localhost:8080
"""
import argparse, os, threading, time, urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import httpx

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8080)
ap.add_argument("--api", default="https://localhost:8443")
ap.add_argument("--user", default="admin")
a = ap.parse_args()
client = httpx.Client(verify=False, timeout=120)
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
.c{position:relative;background:#000}video{width:100%;display:block}
.l{position:absolute;left:6px;top:4px;background:#000a;padding:2px 6px;border-radius:3px}
.s{position:absolute;right:6px;top:4px;background:#000a;padding:2px 6px;border-radius:3px;color:#fc6}</style>
<h3>Live detection streams <small>(WebRTC, near real-time)</small></h3><div id=g></div>
<script>
fetch('/cameras').then(r=>r.json()).then(list=>{
 for(const [id,model] of list){
  const d=document.createElement('div');d.className='c';
  d.innerHTML=`<video muted autoplay playsinline controls></video><div class=l>${id} &middot; ${model}</div><div class=s>connecting...</div>`;
  document.getElementById('g').appendChild(d);
  const v=d.querySelector('video'),s=d.querySelector('.s'),src=`/hls/cam/${id}/processed/index.m3u8`;
  const hls=()=>{ if(window.Hls&&Hls.isSupported()){const h=new Hls({lowLatencyMode:true,liveSyncDurationCount:2,manifestLoadingMaxRetry:60,manifestLoadingRetryDelay:3000,levelLoadingMaxRetry:60,levelLoadingRetryDelay:3000});
    h.loadSource(`/hls/cam/${id}/processed/index.m3u8`);h.attachMedia(v);h.on(Hls.Events.FRAG_LOADED,()=>s.remove());
    h.on(Hls.Events.ERROR,(e,x)=>{if(x.fatal){s.textContent='retrying...';h.destroy();setTimeout(hls,3000)}});}};
  const go=async()=>{
    try{
      const pc=new RTCPeerConnection({iceServers:[]});
      pc.addTransceiver('video',{direction:'recvonly'});
      pc.ontrack=e=>{v.srcObject=e.streams[0];s.textContent='live'; setTimeout(()=>s.remove(),1500)};
      pc.onconnectionstatechange=()=>{if(['failed','disconnected','closed'].includes(pc.connectionState)){pc.close();s.textContent='reconnecting...';d.appendChild(s);setTimeout(go,2000)}};
      await pc.setLocalDescription(await pc.createOffer());
      await new Promise(r=>{if(pc.iceGatheringState==='complete')return r();pc.onicegatheringstatechange=()=>pc.iceGatheringState==='complete'&&r();setTimeout(r,2000)});
      const r=await fetch(`/webrtc/cam/${id}/processed/whep`,{method:'POST',headers:{'Content-Type':'application/sdp'},body:pc.localDescription.sdp});
      if(!r.ok) throw new Error('whep '+r.status);
      await pc.setRemoteDescription({type:'answer',sdp:await r.text()});
      setTimeout(()=>{if(pc.connectionState!=='connected'){pc.close();s.textContent='WebRTC blocked - using HLS';hls()}},8000);
    }catch(err){console.log(id,err);s.textContent='using HLS...';hls()}
  };
  go();
 }});
</script>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *x): pass

    def send(self, code, body, ctype):
        self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(body)

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
            import json
            return self.send(200, json.dumps(cameras()).encode(), "application/json")
        if u.path.startswith("/hls/"):
            q = [(k, v) for k, v in urllib.parse.parse_qsl(u.query) if k != "jwt"] + [("jwt", token())]
            try:
                r = client.get(f"{a.api}{u.path}", params=q)
            except Exception as e:
                return self.send(504, str(e).encode(), "text/plain")
            return self.send(r.status_code, r.content, r.headers.get("content-type", "application/octet-stream"))
        self.send(404, b"not found", "text/plain")


print(f"Live viewer: http://localhost:{a.port}", flush=True)
ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()
