"""Re-encrypt every stored camera credential under the current CREDENTIAL_ENCRYPTION_KEY.

Rotation steps (see docs/SECRETS.md):
  1. python scripts/vms.py rotate-key credential      -> puts the new key in .env, the old one in CREDENTIAL_ENCRYPTION_KEYS_OLD
  2. restart the API / scheduler so they use the new key (old data is still readable through the old key)
  3. python scripts/rotate_credentials.py             -> re-encrypts all rows (this script)
  4. remove CREDENTIAL_ENCRYPTION_KEYS_OLD from .env and restart

    python scripts/rotate_credentials.py            # do it
    python scripts/rotate_credentials.py --dry-run  # only count
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.core.security import get_cipher  # noqa: E402
from app.db.models import Camera  # noqa: E402
from app.db.session import get_sessionmaker, init_engine  # noqa: E402


async def rotate(dry_run: bool = False) -> int:
    s = get_settings()
    init_engine(s)
    cipher = get_cipher(s)
    changed = 0
    async with get_sessionmaker()() as db:
        for cam in (await db.execute(select(Camera))).scalars():
            if cam.username_enc or cam.password_enc:
                changed += 1
                if not dry_run:
                    cam.username_enc = cipher.rotate(cam.username_enc)
                    cam.password_enc = cipher.rotate(cam.password_enc)
        if not dry_run:
            await db.commit()
    return changed


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    n = asyncio.run(rotate(ap.parse_args().dry_run))
    print(f"{n} camera(s) {'would be' if '--dry-run' in sys.argv else 'were'} re-encrypted")
