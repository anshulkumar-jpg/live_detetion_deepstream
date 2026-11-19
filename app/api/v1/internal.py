"""Internal endpoints called by infrastructure (not part of the public API).

POST /internal/mediamtx/auth - MediaMTX `authMethod: http` callback.
  * read / playback: requires a valid, un-revoked platform JWT of an active user, passed by the
    player as `Authorization: Bearer` or `?jwt=` (MediaMTX forwards it as `token`).
  * publish: only DeepStream workers (user 'worker') on private networks, only to cam/<id>/processed,
    authenticated with the dedicated MEDIAMTX_PUBLISH_TOKEN as the RTSP password (publish only, no API access).
  * api (control API, can read camera sources): only the API/scheduler login, from a private network.
  * metrics: only the 'metrics' login (Prometheus). pprof and anything unknown: never.
"""
from __future__ import annotations

import hmac
import ipaddress
import re

from fastapi import APIRouter, Depends, Response
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import AuthError, get_session, video_principal_from_token
from app.core.config import Settings, get_settings
from app.core.security import publish_token_valid

router = APIRouter(prefix="/internal", tags=["internal"], include_in_schema=False)
PROCESSED_RE = re.compile(r"^cam/([A-Za-z0-9][A-Za-z0-9_.-]{0,63})/(processed|raw)$")


class MediaMTXAuth(BaseModel):
    user: str | None = ""
    password: str | None = ""
    token: str | None = ""
    ip: str | None = ""
    action: str
    path: str | None = ""
    protocol: str | None = ""
    id: str | None = None
    query: str | None = ""


def _private(ip: str | None) -> bool:
    try:
        return ipaddress.ip_address(ip or "").is_private
    except ValueError:
        return False


def _secret_ok(given: str | None, expected: str) -> bool:
    return bool(expected) and given is not None and hmac.compare_digest(given.encode(), expected.encode())


@router.post("/mediamtx/auth")
async def mediamtx_auth(req: MediaMTXAuth, settings: Settings = Depends(get_settings),
                        db: AsyncSession = Depends(get_session)) -> Response:
    ok = False
    if req.action in ("read", "playback"):
        tok = req.token or ""
        if not tok and req.query:
            m = re.search(r"(?:^|&)jwt=([^&]+)", req.query)
            tok = m.group(1) if m else ""
        m = PROCESSED_RE.match(req.path or "")
        try:  # camera-scoped token: must be for exactly this camera, and the user must still be active
            if m:
                await video_principal_from_token(db, tok, m.group(1), settings)
                ok = True
        except AuthError:
            ok = False
    elif req.action == "publish":
        ok = (_private(req.ip) and req.user == "worker" and bool(req.path) and req.path.endswith("/processed")
              and bool(PROCESSED_RE.match(req.path or "")) and publish_token_valid(req.password, settings))
    elif req.action == "api":
        ok = _private(req.ip) and req.user == settings.mediamtx_api_user and _secret_ok(req.password, settings.mediamtx_api_password)
    elif req.action == "metrics":
        ok = _private(req.ip) and req.user == "metrics" and _secret_ok(req.password, settings.mediamtx_metrics_password)
    # pprof (debug dumps) and any other action stay refused
    return Response(status_code=200 if ok else 401)
