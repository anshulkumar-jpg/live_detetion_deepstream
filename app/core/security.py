"""JWT authentication, RBAC, password hashing and RTSP credential encryption."""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets
from datetime import datetime, timedelta, timezone
from enum import Enum
from urllib.parse import quote, urlsplit, urlunsplit

import jwt
from cryptography.fernet import Fernet, InvalidToken

from app.core.config import Settings, get_settings


class Role(str, Enum):
    ADMIN = "admin"        # everything incl. users, models, system config
    OPERATOR = "operator"  # cameras, assignments, workers
    VIEWER = "viewer"      # read-only


ROLE_RANK = {Role.VIEWER: 0, Role.OPERATOR: 1, Role.ADMIN: 2}


# ---------------------------------------------------------------- passwords
_SCRYPT_N, _SCRYPT_R, _SCRYPT_P = 2**14, 8, 1


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    dk = hashlib.scrypt(password.encode(), salt=salt, n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P, dklen=32)
    return "scrypt$" + base64.b64encode(salt).decode() + "$" + base64.b64encode(dk).decode()


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, salt_b64, dk_b64 = stored.split("$")
        if algo != "scrypt":
            return False
        salt, expected = base64.b64decode(salt_b64), base64.b64decode(dk_b64)
        dk = hashlib.scrypt(password.encode(), salt=salt, n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P, dklen=32)
        return hmac.compare_digest(dk, expected)
    except Exception:
        return False


# ---------------------------------------------------------------- JWT
def create_access_token(subject: str, role: Role, settings: Settings | None = None) -> str:
    s = settings or get_settings()
    now = datetime.now(timezone.utc)
    payload = {
        "sub": subject,
        "role": role.value,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=s.jwt_expire_minutes)).timestamp()),
        "jti": secrets.token_hex(8),
    }
    return jwt.encode(payload, s.jwt_secret, algorithm=s.jwt_algorithm)


def decode_token(token: str, settings: Settings | None = None) -> dict:
    s = settings or get_settings()
    return jwt.decode(token, s.jwt_secret, algorithms=[s.jwt_algorithm])


def internal_token_valid(token: str | None, settings: Settings | None = None) -> bool:
    s = settings or get_settings()
    return bool(s.internal_service_token) and token is not None and hmac.compare_digest(
        token, s.internal_service_token
    )


# ---------------------------------------------------------------- credentials
class CredentialCipher:
    """Fernet (AES-128-CBC + HMAC-SHA256) encryption for RTSP credentials at rest."""

    def __init__(self, key: str):
        if not key:
            raise RuntimeError("CREDENTIAL_ENCRYPTION_KEY is not set")
        self._f = Fernet(key.encode() if isinstance(key, str) else key)

    def encrypt(self, plaintext: str | None) -> str | None:
        if plaintext is None or plaintext == "":
            return None
        return self._f.encrypt(plaintext.encode()).decode()

    def decrypt(self, token: str | None) -> str | None:
        if not token:
            return None
        try:
            return self._f.decrypt(token.encode()).decode()
        except InvalidToken as e:
            raise RuntimeError("Unable to decrypt stored credential (wrong key?)") from e


def get_cipher(settings: Settings | None = None) -> CredentialCipher:
    s = settings or get_settings()
    return CredentialCipher(s.credential_encryption_key)


def strip_credentials(url: str) -> tuple[str, str | None, str | None]:
    """Split credentials out of an RTSP URL. Returns (clean_url, username, password)."""
    parts = urlsplit(url)
    if parts.username is None and parts.password is None:
        return url, None, None
    host = parts.hostname or ""
    if ":" in host:  # IPv6
        host = f"[{host}]"
    if parts.port:
        host = f"{host}:{parts.port}"
    clean = urlunsplit((parts.scheme, host, parts.path, parts.query, parts.fragment))
    from urllib.parse import unquote

    return clean, unquote(parts.username or "") or None, unquote(parts.password or "") or None


def inject_credentials(url: str, username: str | None, password: str | None) -> str:
    """Build the runtime RTSP URL. Only ever used inside the worker; never logged."""
    if not username:
        return url
    parts = urlsplit(url)
    userinfo = quote(username, safe="")
    if password:
        userinfo += ":" + quote(password, safe="")
    netloc = f"{userinfo}@{parts.netloc.split('@')[-1]}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))
