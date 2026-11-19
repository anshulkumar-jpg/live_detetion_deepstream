"""JWT authentication, RBAC, password hashing and RTSP credential encryption."""
from __future__ import annotations

import base64
import functools
import hashlib
import hmac
import os
import secrets
from datetime import datetime, timedelta, timezone
from enum import Enum
from urllib.parse import quote, urlsplit, urlunsplit

import jwt
from cryptography.fernet import Fernet, InvalidToken, MultiFernet

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


@functools.lru_cache(maxsize=1)
def dummy_password_hash() -> str:
    """A real hash of a random password: checking a login against it costs the same as checking a real user, so a wrong
    user name cannot be told apart from a wrong password by timing."""
    return hash_password(secrets.token_urlsafe(24))


# ---------------------------------------------------------------- JWT
TOKEN_ISSUER = "vms"
AUD_API = "vms-api"
AUD_VIDEO = "vms-video"


def create_access_token(subject: str, role: Role, settings: Settings | None = None, epoch: int = 0) -> str:
    s = settings or get_settings()
    now = datetime.now(timezone.utc)
    payload = {
        "sub": subject,
        "role": role.value,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=s.jwt_expire_minutes)).timestamp()),
        "jti": secrets.token_hex(8),
        "te": int(epoch),  # token epoch: see User.token_epoch
        "iss": TOKEN_ISSUER,
        "aud": AUD_API,
    }
    return jwt.encode(payload, s.jwt_secret, algorithm=s.jwt_algorithm)


def _decode(token: str, s: Settings, aud: str, require: list[str]) -> dict:
    """Check with the current signing key; during a key rotation also with JWT_SECRET_PREVIOUS (never to sign)."""
    keys = [s.jwt_secret] + ([s.jwt_secret_previous] if s.jwt_secret_previous else [])
    err: Exception | None = None
    for k in keys:
        try:
            return jwt.decode(token, k, algorithms=[s.jwt_algorithm], audience=aud, issuer=TOKEN_ISSUER,
                              options={"require": require})
        except jwt.InvalidSignatureError as e:
            err = e
    raise err or jwt.InvalidTokenError("invalid token")


def decode_token(token: str, settings: Settings | None = None) -> dict:
    """Decode a login (API) token. Issuer and audience are enforced, so a video token cannot be used on the API."""
    s = settings or get_settings()
    return _decode(token, s, AUD_API, ["exp", "sub", "aud", "iss"])


def create_video_token(subject: str, camera_id: str, role: Role, epoch: int, settings: Settings | None = None) -> str:
    """Short-lived token that opens the video of ONE camera and nothing else (not the API, not other cameras)."""
    s = settings or get_settings()
    now = datetime.now(timezone.utc)
    return jwt.encode({
        "sub": subject, "role": role.value, "cam": camera_id, "te": int(epoch),
        "iat": int(now.timestamp()), "exp": int((now + timedelta(seconds=s.video_token_ttl_s)).timestamp()),
        "jti": secrets.token_hex(8), "iss": TOKEN_ISSUER, "aud": AUD_VIDEO,
    }, s.jwt_secret, algorithm=s.jwt_algorithm)


def decode_video_token(token: str, settings: Settings | None = None) -> dict:
    s = settings or get_settings()
    return _decode(token, s, AUD_VIDEO, ["exp", "sub", "aud", "iss", "cam"])


def publish_token_valid(token: str | None, settings: Settings | None = None) -> bool:
    """Password a DeepStream worker uses to publish annotated video to MediaMTX. Grants publishing only."""
    s = settings or get_settings()
    return bool(s.mediamtx_publish_token) and token is not None and hmac.compare_digest(
        token.encode(), s.mediamtx_publish_token.encode()
    )


# ---------------------------------------------------------------- credentials
class CredentialCipher:
    """Fernet (AES-128-CBC + HMAC-SHA256) encryption for RTSP credentials at rest."""

    def __init__(self, key: str, old_keys: str | list[str] = ""):
        """``key`` encrypts. ``old_keys`` (comma separated) are only used to decrypt data written before a rotation."""
        if not key:
            raise RuntimeError("CREDENTIAL_ENCRYPTION_KEY is not set")
        olds = [k.strip() for k in (old_keys.split(",") if isinstance(old_keys, str) else old_keys) if k.strip()]
        self._f = MultiFernet([Fernet(k.encode() if isinstance(k, str) else k) for k in [key, *olds]])

    def rotate(self, token: str | None) -> str | None:
        """Re-encrypt a stored value under the current key (used by scripts/rotate_credentials.py)."""
        return self._f.rotate(token.encode()).decode() if token else token

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
    return CredentialCipher(s.credential_encryption_key, s.credential_encryption_keys_old)


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
