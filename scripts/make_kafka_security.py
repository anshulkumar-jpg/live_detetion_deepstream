"""Create the TLS certificates and the per-service logins that secure Kafka.

    python scripts/make_kafka_security.py

Safe to run again: it only creates what is missing and never replaces a certificate or password that exists.
Pass --rotate-certs to issue a new certificate authority and server certificate (clients get the new ca.pem on restart).

Writes
  deploy/kafka/certs/ca.pem       public certificate authority (clients use it to trust the broker)
  deploy/kafka/certs/kafka.p12    broker key + certificate (secret, stays on the broker)
  deploy/kafka/certs/ca.p12       truststore for Java clients (Logstash)
  .env                            KAFKA_KEYSTORE_PASSWORD and one KAFKA_PASS_<SERVICE> per service (SCRAM-SHA-512 logins)

Needs: pip install cryptography
"""
from __future__ import annotations

import argparse
import datetime as dt
import ipaddress
import secrets
import sys
from pathlib import Path

try:
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.hazmat.primitives.serialization import pkcs12
    from cryptography.x509.oid import NameOID
except ImportError:
    sys.exit("This script needs the 'cryptography' package:  pip install cryptography")

ROOT = Path(__file__).resolve().parent.parent
CERTS = ROOT / "deploy" / "kafka" / "certs"
ENV = ROOT / ".env"
SERVICES = ["SCHEDULER", "AGENT", "ANALYTICS", "MTMC", "API", "LOGSTASH", "EXPORTER"]


def read_env() -> tuple[list[str], str]:
    raw = ENV.read_bytes().decode("utf-8") if ENV.exists() else ""
    nl = "\r\n" if "\r\n" in raw else "\n"
    return (raw.split(nl) if raw else []), nl


def has(lines: list[str], name: str) -> bool:
    return any(line.startswith(name + "=") and line.split("=", 1)[1].strip() for line in lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rotate-certs", action="store_true", help="issue a new CA and broker certificate")
    a = ap.parse_args()

    lines, nl = read_env()
    added: list[str] = []

    def secret(name: str, nbytes: int = 24) -> str:
        if not has(lines, name):
            lines.append(f"{name}={secrets.token_urlsafe(nbytes)}")
            added.append(name)
        return next(line.split("=", 1)[1].strip() for line in lines if line.startswith(name + "="))

    store_pw = secret("KAFKA_KEYSTORE_PASSWORD")
    for svc in SERVICES:
        secret(f"KAFKA_PASS_{svc}")

    CERTS.mkdir(parents=True, exist_ok=True)
    (CERTS / ".gitkeep").touch()
    if a.rotate_certs or not (CERTS / "kafka.p12").exists() or not (CERTS / "ca.pem").exists():
        now = dt.datetime.now(dt.timezone.utc)
        ca_key = ec.generate_private_key(ec.SECP256R1())
        ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "DeepStream VMS Kafka CA")])
        ca_cert = (x509.CertificateBuilder().subject_name(ca_name).issuer_name(ca_name).public_key(ca_key.public_key())
                   .serial_number(x509.random_serial_number()).not_valid_before(now - dt.timedelta(minutes=5))
                   .not_valid_after(now + dt.timedelta(days=3650))
                   .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
                   .add_extension(x509.KeyUsage(digital_signature=True, key_cert_sign=True, crl_sign=True, content_commitment=False,
                                                key_encipherment=False, data_encipherment=False, key_agreement=False,
                                                encipher_only=False, decipher_only=False), critical=True)
                   .sign(ca_key, hashes.SHA256()))
        key = ec.generate_private_key(ec.SECP256R1())
        cert = (x509.CertificateBuilder().subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "kafka")]))
                .issuer_name(ca_name).public_key(key.public_key()).serial_number(x509.random_serial_number())
                .not_valid_before(now - dt.timedelta(minutes=5)).not_valid_after(now + dt.timedelta(days=825))
                .add_extension(x509.SubjectAlternativeName([x509.DNSName("kafka"), x509.DNSName("localhost"),
                                                            x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]), critical=False)
                .add_extension(x509.ExtendedKeyUsage([x509.oid.ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
                .sign(ca_key, hashes.SHA256()))
        enc = serialization.BestAvailableEncryption(store_pw.encode())
        (CERTS / "ca.pem").write_bytes(ca_cert.public_bytes(serialization.Encoding.PEM))
        (CERTS / "kafka.p12").write_bytes(pkcs12.serialize_key_and_certificates(b"kafka", key, cert, [ca_cert], enc))
        (CERTS / "ca.p12").write_bytes(pkcs12.serialize_java_truststore([pkcs12.PKCS12Certificate(ca_cert, b"vms-ca")], enc))
        print("created certificates in", CERTS.relative_to(ROOT))
    else:
        print("certificates already exist (use --rotate-certs to replace them)")

    # the Kafka image reads the keystore password from a file (it does not accept it as a plain setting)
    (CERTS / "store.creds").write_text(store_pw, encoding="utf-8")

    if added:
        ENV.write_bytes(nl.join(lines).encode("utf-8"))
        print("added to .env:", ", ".join(added))
    else:
        print(".env already has all Kafka passwords")
    print("Next: docker compose up -d   (the broker creates the logins and permissions when it starts)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
