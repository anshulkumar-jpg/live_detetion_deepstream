# Secrets and key rotation

`python scripts/vms.py init` creates every missing secret (it never overwrites). On a shared server keep `.env` at mode 600 and out of backups that other people can read. For Kubernetes put the same names in a Kubernetes Secret, or sync them from your secrets manager (External Secrets Operator, Vault, AWS/GCP Secret Manager); the Helm chart reads Secrets, not values files.

| Secret | Used by | Rotate with |
|---|---|---|
| `JWT_SECRET` | API (signs and checks login + video tokens) | `vms.py rotate-key jwt`: the old value moves to `JWT_SECRET_PREVIOUS` (verify-only), so nobody is signed out. Remove `JWT_SECRET_PREVIOUS` after `JWT_EXPIRE_MINUTES` (default 60). |
| `CREDENTIAL_ENCRYPTION_KEY` | API, scheduler, GPU agent (camera passwords at rest) | 1. `vms.py rotate-key credential` (old key goes to `CREDENTIAL_ENCRYPTION_KEYS_OLD`) 2. restart 3. `python scripts/rotate_credentials.py` 4. delete `CREDENTIAL_ENCRYPTION_KEYS_OLD`, restart |
| `COMMAND_SIGNING_KEY` | scheduler, GPU agents | `vms.py rotate-key command`, then restart the scheduler **and all GPU agents together**. Commands signed with the old key are refused (they live at most 5 minutes). |
| `MEDIAMTX_PUBLISH_TOKEN` | API, GPU agents/workers | `vms.py rotate-key publish`; restart API and agents. Running workers must restart to publish again. |
| `MEDIAMTX_API_PASSWORD`, `MEDIAMTX_METRICS_PASSWORD`, `KAFKA_PASS_*`, `ELASTIC_PASSWORD`, ... | one service each | delete the line from `.env`, run `vms.py init`, restart that service. Kafka logins are re-created by `kafka-init.sh` on start. |
| Kafka TLS certificate | all Kafka clients | delete `deploy/kafka/certs/*` (not `.gitkeep`), run `vms.py init`, restart everything. |

Rotate on a schedule (at least yearly), immediately after anyone with access leaves, and after any suspected leak.
