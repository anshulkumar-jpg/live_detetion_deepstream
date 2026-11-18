"""MediaMTX integration (video streaming plane), via the MediaMTX Control API v3.

* Processed streams: DeepStream workers push NVENC H.264 to
  ``rtsp://mediamtx:8554/cam/<id>/processed`` (publisher). Nothing to configure
  here except reading status.
* Raw streams: MediaMTX pulls the original camera on demand
  (``cam/<id>/raw``, ``sourceOnDemand``), so no GPU work is spent on it.
* Viewers use HLS (``:8888/<path>/index.m3u8``) or WebRTC/WHEP (``:8889/<path>/whep``).
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import httpx
from sqlalchemy import and_, select

from app.core.config import Settings
from app.core.security import get_cipher, inject_credentials
from app.db.models import Camera, StreamMode, StreamSession

log = logging.getLogger(__name__)


def path_for(camera_id: str, kind: str) -> str:
    return f"cam/{camera_id}/{kind}"


class MediaMTXClient:
    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None):
        self.s = settings
        auth = (settings.mediamtx_api_user, settings.mediamtx_api_password) if settings.mediamtx_api_user else None
        self.http = client or httpx.AsyncClient(base_url=settings.mediamtx_api_url, auth=auth, timeout=5.0)

    async def get_path(self, name: str) -> dict[str, Any] | None:
        r = await self.http.get(f"/v3/paths/get/{name}")
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()

    async def list_paths(self) -> list[dict[str, Any]]:
        items, page = [], 0
        while True:
            r = await self.http.get("/v3/paths/list", params={"page": page, "itemsPerPage": 500})
            r.raise_for_status()
            data = r.json()
            items.extend(data.get("items", []))
            page += 1
            if page >= data.get("pageCount", 1):
                return items

    async def add_or_replace_path(self, name: str, conf: dict[str, Any]) -> None:
        r = await self.http.post(f"/v3/config/paths/add/{name}", json=conf)
        if r.status_code == 400 and "already exists" in r.text:
            r = await self.http.post(f"/v3/config/paths/replace/{name}", json=conf)
        r.raise_for_status()

    async def delete_path(self, name: str) -> None:
        r = await self.http.delete(f"/v3/config/paths/delete/{name}")
        if r.status_code not in (200, 404):
            r.raise_for_status()

    async def ping(self) -> bool:
        try:
            r = await self.http.get("/v3/config/global/get")
            return r.status_code == 200
        except httpx.HTTPError:
            return False


class StreamManager:
    """Manages per-camera stream paths, sessions and output URLs."""

    def __init__(self, settings: Settings, client: MediaMTXClient | None = None):
        self.s = settings
        self.mtx = client or MediaMTXClient(settings)

    def urls(self, camera_id: str, kind: str) -> dict[str, str]:
        p = path_for(camera_id, kind)
        return {
            "path": p,
            "hls": f"{self.s.mediamtx_public_hls_url}/{p}/index.m3u8",
            "webrtc": f"{self.s.mediamtx_public_webrtc_url}/{p}",
            "whep": f"{self.s.mediamtx_public_webrtc_url}/{p}/whep",
            "rtsp": f"{self.s.mediamtx_public_rtsp_url}/{p}",
        }

    async def ensure_raw_path(self, cam: Camera) -> None:
        """MediaMTX pulls the camera directly (credentials are only sent to MediaMTX)."""
        cipher = get_cipher(self.s)
        src = inject_credentials(cam.rtsp_url, cipher.decrypt(cam.username_enc), cipher.decrypt(cam.password_enc))
        try:
            await self.mtx.add_or_replace_path(path_for(cam.id, "raw"), {
                "source": src, "sourceOnDemand": True, "rtspTransport": "tcp",
            })
        except httpx.HTTPError as e:
            log.warning("failed to configure raw path: %s", type(e).__name__, extra={"camera_id": cam.id})

    async def remove_raw_path(self, camera_id: str) -> None:
        try:
            await self.mtx.delete_path(path_for(camera_id, "raw"))
        except httpx.HTTPError as e:
            log.warning("failed to delete raw path: %s", type(e).__name__, extra={"camera_id": camera_id})

    async def status(self, db, cam: Camera) -> dict[str, Any]:
        kinds = []
        if cam.stream_mode in (StreamMode.ANNOTATED, StreamMode.BOTH):
            kinds.append("processed")
        if cam.stream_mode in (StreamMode.RAW, StreamMode.BOTH):
            kinds.append("raw")
        out: dict[str, Any] = {"camera_id": cam.id, "stream_mode": cam.stream_mode.value, "streams": []}
        for kind in kinds:
            info: dict[str, Any] = {"kind": kind, **self.urls(cam.id, kind)}
            try:
                p = await self.mtx.get_path(path_for(cam.id, kind))
                info["ready"] = bool(p and p.get("ready"))
                info["viewers"] = len(p.get("readers", [])) if p else 0
                info["tracks"] = p.get("tracks", []) if p else []
                info["mediamtx_reachable"] = True
            except httpx.HTTPError:
                info.update(ready=False, viewers=0, mediamtx_reachable=False)
            await self._record_session(db, cam.id, kind, info)
            out["streams"].append(info)
        return out

    async def _record_session(self, db, camera_id: str, kind: str, info: dict[str, Any]) -> None:
        now = datetime.now(timezone.utc)
        sess = (await db.execute(select(StreamSession).where(and_(
            StreamSession.camera_id == camera_id, StreamSession.kind == kind,
            StreamSession.ended_at.is_(None))))).scalar_one_or_none()
        if sess is None:
            sess = StreamSession(camera_id=camera_id, kind=kind, path=info["path"])
            db.add(sess)
        sess.status = "ready" if info.get("ready") else "not_ready"
        sess.viewer_count = info.get("viewers", 0)
        sess.last_checked_at = now
