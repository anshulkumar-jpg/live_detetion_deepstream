"""Vector storage for Re-ID embeddings: Milvus (production) + in-memory (tests)."""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class VectorHit:
    id: str
    score: float  # cosine similarity
    meta: dict[str, Any]


class VectorStore(ABC):
    @abstractmethod
    def ensure_collection(self) -> None: ...

    @abstractmethod
    def upsert(self, id: str, vector: list[float], meta: dict[str, Any]) -> None: ...

    @abstractmethod
    def search(self, vector: list[float], top_k: int, *, exclude_camera: str | None, class_name: str | None,
               min_ts: int | None) -> list[VectorHit]: ...

    @abstractmethod
    def delete_older_than(self, ts: int) -> int: ...


class MilvusVectorStore(VectorStore):
    """pymilvus MilvusClient; COSINE metric over L2-normalised vectors with an HNSW index."""

    FIELDS = ("camera_id", "object_id", "global_id", "class_name", "model_id", "model_version", "ts")

    def __init__(self, uri: str, token: str, collection: str, dim: int, ttl_s: int | None = None):
        from pymilvus import MilvusClient

        self.client = MilvusClient(uri=uri, token=token or "")
        self.collection = collection
        self.dim = dim
        self.ttl_s = ttl_s
        self.lite = uri.endswith(".db")

    def ensure_collection(self) -> None:
        from pymilvus import DataType, MilvusClient

        if self.client.has_collection(self.collection):
            self.client.load_collection(self.collection)
            return
        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=200)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=self.dim)
        for f in ("camera_id", "object_id", "global_id", "class_name", "model_id", "model_version"):
            schema.add_field(f, DataType.VARCHAR, max_length=128)
        schema.add_field("ts", DataType.INT64)
        idx = self.client.prepare_index_params()
        if self.lite:  # Milvus Lite (local .db file, dev/tests) supports a reduced index set
            idx.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="COSINE")
        else:
            idx.add_index(field_name="vector", index_type="HNSW", metric_type="COSINE", params={"M": 16, "efConstruction": 200})
            idx.add_index(field_name="camera_id", index_type="INVERTED")
            idx.add_index(field_name="ts", index_type="INVERTED")
        kwargs = {"properties": {"collection.ttl.seconds": self.ttl_s}} if self.ttl_s and not self.lite else {}
        self.client.create_collection(self.collection, schema=schema, index_params=idx, **kwargs)
        self.client.load_collection(self.collection)
        log.info("created Milvus collection %s (dim=%d)", self.collection, self.dim)

    def upsert(self, id: str, vector: list[float], meta: dict[str, Any]) -> None:
        row = {"id": id, "vector": vector, **{f: meta.get(f) or "" for f in self.FIELDS if f != "ts"},
               "ts": int(meta.get("ts") or time.time())}
        self.client.upsert(self.collection, [row])

    @staticmethod
    def _q(v: str) -> str:
        return v.replace("\\", "\\\\").replace('"', '\\"')

    def search(self, vector, top_k, *, exclude_camera=None, class_name=None, min_ts=None) -> list[VectorHit]:
        conds = []
        if exclude_camera:
            conds.append(f'camera_id != "{self._q(exclude_camera)}"')
        if class_name:
            conds.append(f'class_name == "{self._q(class_name)}"')
        if min_ts:
            conds.append(f"ts >= {int(min_ts)}")
        params = {"metric_type": "COSINE"} if self.lite else {"metric_type": "COSINE", "params": {"ef": 64}}
        res = self.client.search(self.collection, data=[vector], limit=top_k, filter=" and ".join(conds),
                                 output_fields=list(self.FIELDS), search_params=params)
        hits = []
        for h in res[0] if res else []:
            hits.append(VectorHit(id=h["id"], score=float(h["distance"]), meta=h.get("entity", {})))
        return hits

    def delete_older_than(self, ts: int) -> int:
        r = self.client.delete(self.collection, filter=f"ts < {int(ts)}")
        try:
            return int(r["delete_count"])
        except (KeyError, TypeError):
            return int(getattr(r, "delete_count", 0) or 0)


class InMemoryVectorStore(VectorStore):
    """Exact cosine search with numpy. For unit tests and single-node development only."""

    def __init__(self) -> None:
        self.rows: dict[str, tuple[np.ndarray, dict[str, Any]]] = {}

    def ensure_collection(self) -> None:
        pass

    def upsert(self, id, vector, meta) -> None:
        v = np.asarray(vector, dtype=np.float32)
        n = np.linalg.norm(v)
        self.rows[id] = (v / n if n else v, dict(meta))

    def search(self, vector, top_k, *, exclude_camera=None, class_name=None, min_ts=None) -> list[VectorHit]:
        q = np.asarray(vector, dtype=np.float32)
        n = np.linalg.norm(q)
        q = q / n if n else q
        out = []
        for id_, (v, m) in self.rows.items():
            if exclude_camera and m.get("camera_id") == exclude_camera:
                continue
            if class_name and m.get("class_name") != class_name:
                continue
            if min_ts and (m.get("ts") or 0) < min_ts:
                continue
            out.append(VectorHit(id_, float(np.dot(q, v)), m))
        return sorted(out, key=lambda h: h.score, reverse=True)[:top_k]

    def delete_older_than(self, ts) -> int:
        old = [k for k, (_, m) in self.rows.items() if (m.get("ts") or 0) < ts]
        for k in old:
            del self.rows[k]
        return len(old)
