# Data retention and privacy

Detection events, cross-camera identities (Re-ID) and person embeddings can identify people, so they fall under GDPR and India's DPDP Act. The platform deletes them automatically. The periods below are the **defaults**: set them to what your legal basis and policy allow, and document why.

| Data | Where | Default | Setting |
|---|---|---|---|
| Track-level detection events | PostgreSQL `detection_events` | 30 days | `DETECTION_EVENT_RETENTION_DAYS` |
| Re-ID associations (global identity to camera tracks) | PostgreSQL `reid_associations` | 7 days | `REID_RETENTION_DAYS` |
| Embeddings | Milvus | 7 days | `EMBEDDING_RETENTION_DAYS` |
| Behaviour events | PostgreSQL `analytics_events` | 30 days | `ANALYTICS_EVENT_RETENTION_DAYS` |
| Audit log (security record) | PostgreSQL `audit_logs` | 365 days | `AUDIT_RETENTION_DAYS` |
| Kafka topics | Kafka | 24 hours | broker `log.retention.hours` |
| Elasticsearch event copies | `vms-*` daily indices | until deleted | add an ILM delete policy that matches the periods above |

`0` keeps the data forever (not recommended). The scheduler leader runs the PostgreSQL purge every hour (`app/services/retention.py`, tested in `tests/integration/test_retention.py`). Milvus expires embeddings in `analytics/mtmc.py`.

Also: restrict who can read analytics with **sites** (docs/API.md), keep `ALLOW_RAW_STREAMS=false` unless you need un-annotated video, and answer data-subject requests by deleting the rows for the relevant `global_id` and time range.
