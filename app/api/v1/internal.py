"""Internal endpoints called by infrastructure (not part of the public API).

POST /internal/mediamtx/auth - MediaMTX `authMethod: http` callback.
  * read / playback: requires a valid platform JWT (viewer or above), passed by the
    player as `Authorization: Bearer` or `?jwt=` (MediaMTX forwards it as `token`).
  * publish: only DeepStream workers on private networks, only to cam/<id>/processed,
    authenticated with the internal service token as the RTSP password.
  * api / metrics / pprof: private networks only.
"""
from __future__ import annotations

import ipaddress
import re

import jwt
from fastapi import APIRouter, Depends, Response
from pydantic import BaseModel

from app.core.config import Settings, get_settings
from app.core.security import decode_token, internal_token_valid

router = APIRouter(prefix="/internal", tags=["internal"], include_in_schema=False)
PROCESSED_RE = re.compile(r"^cam/[A-Za-z0-9][A-Za-z0-9_.-]{0,63}/(processed|raw)$")


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


@router.post("/mediamtx/auth")
async def mediamtx_auth(req: MediaMTXAuth, settings: Settings = Depends(get_settings)) -> Response:
    ok = False
    if req.action in ("read", "playback"):
        tok = req.token or ""
        if not tok and req.query:
            m = re.search(r"(?:^|&)jwt=([^&]+)", req.query)
            tok = m.group(1) if m else ""
        try:
            decode_token(tok, settings)
            ok = bool(req.path and PROCESSED_RE.match(req.path))
        except jwt.PyJWTError:
            ok = False
    elif req.action == "publish":
        ok = (_private(req.ip) and bool(req.path) and req.path.endswith("/processed")
              and bool(PROCESSED_RE.match(req.path or "")) and internal_token_valid(req.password, settings))
    elif req.action in ("api", "metrics", "pprof"):
        ok = _private(req.ip)
    return Response(status_code=200 if ok else 401)
