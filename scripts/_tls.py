"""TLS settings shared by the helper scripts.

The scripts always verify the server certificate:

* VMS_CA_BUNDLE=<file>  trust this CA / certificate file;
* otherwise, if the gateway uses a SELF-SIGNED certificate at deploy/envoy/certs/tls.crt (the usual local setup), that
  one certificate is trusted, and only it;
* otherwise the normal system certificate store is used (a certificate from a public or company CA).

VMS_INSECURE_TLS=1 switches verification off for a throw-away lab only, and prints a warning every time.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

LOCAL_CERT = Path(__file__).resolve().parent.parent / "deploy" / "envoy" / "certs" / "tls.crt"


def _self_signed(path: Path) -> bool:
    try:
        from cryptography import x509

        cert = x509.load_pem_x509_certificate(path.read_bytes())
        return cert.issuer == cert.subject
    except Exception:
        return False


def tls_verify() -> bool | str:
    if os.environ.get("VMS_INSECURE_TLS") == "1":
        print("WARNING: TLS certificate verification is OFF (VMS_INSECURE_TLS=1). Do not use this outside a lab.",
              file=sys.stderr)
        return False
    if os.environ.get("VMS_CA_BUNDLE"):
        return os.environ["VMS_CA_BUNDLE"]
    if LOCAL_CERT.exists() and _self_signed(LOCAL_CERT):
        return str(LOCAL_CERT)
    return True
