"""Signed pipeline commands.

The scheduler is the only program allowed to tell a GPU node what to run. Every command it puts on Kafka is signed
with HMAC-SHA256 using COMMAND_SIGNING_KEY, which only the scheduler (to sign) and the GPU agents (to check) hold.
A GPU agent runs a command only if all of this is true:

* the signature is valid, so nobody who merely has access to Kafka can invent or change a command,
* the command names this node (``payload.node_id``), and that name is part of what is signed,
* it has not expired (``exp``, a few seconds to minutes after it was signed),
* it is not a repeat of a command already run (the agent keeps the ids it has seen until they expire).
"""
from __future__ import annotations

import hashlib
import hmac
import json
import time
from typing import Any

SIGNED_FIELDS = ("event_id", "event_type", "source", "timestamp", "exp", "payload")
MAX_TTL_S = 300  # a command claiming to live longer than this is rejected, which also limits replay


def _canonical(msg: dict[str, Any]) -> bytes:
    return json.dumps({k: msg.get(k) for k in SIGNED_FIELDS}, sort_keys=True, separators=(",", ":"), default=str).encode()


def _mac(msg: dict[str, Any], key: str) -> str:
    return hmac.new(key.encode(), _canonical(msg), hashlib.sha256).hexdigest()


def sign_command(msg: dict[str, Any], key: str, ttl_s: int = 60, now: float | None = None) -> dict[str, Any]:
    """Return a copy of ``msg`` with an expiry time and a signature added."""
    if not key:
        raise ValueError("COMMAND_SIGNING_KEY is not set: refusing to send an unsigned pipeline command")
    out = dict(msg)
    out["exp"] = int((now if now is not None else time.time()) + min(ttl_s, MAX_TTL_S))
    out["sig"] = _mac(out, key)
    return out


def verify_command(msg: dict[str, Any], key: str, node_id: str, now: float | None = None) -> tuple[bool, str]:
    """(ok, reason). Never raises on bad input."""
    if not key:
        return False, "no COMMAND_SIGNING_KEY configured on this node"
    sig, exp = msg.get("sig"), msg.get("exp")
    if not isinstance(sig, str) or isinstance(exp, bool) or not isinstance(exp, (int, float)):
        return False, "command is not signed"
    if not hmac.compare_digest(sig, _mac(msg, key)):
        return False, "bad signature"
    t = now if now is not None else time.time()
    if exp < t:
        return False, "command expired"
    if exp > t + MAX_TTL_S + 5:
        return False, "expiry too far in the future"
    payload = msg.get("payload")
    if not isinstance(payload, dict) or payload.get("node_id") != node_id:
        return False, "command is for another node"
    return True, "ok"


class ReplayGuard:
    """Remembers command ids until they expire, so a captured command cannot be run twice."""

    def __init__(self) -> None:
        self._seen: dict[str, float] = {}

    def first_time(self, msg: dict[str, Any], now: float | None = None) -> bool:
        t = now if now is not None else time.time()
        for k in [k for k, exp in self._seen.items() if exp < t]:
            del self._seen[k]
        eid = str(msg.get("event_id"))
        if eid in self._seen:
            return False
        self._seen[eid] = float(msg.get("exp", t + MAX_TTL_S))
        return True
