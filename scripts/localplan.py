"""Build worker specs from the models YAML without the database (used by GPU validation and benchmarks).
Uses the same Model Routing Engine as the scheduler, so the generated pipeline is identical."""
from __future__ import annotations

import yaml

from app.db.models import (
    Camera,
    CameraModelAssignment,
    ExecutionStrategy,
    InferenceBackend,
    Model,
    ModelTask,
    ModelVersion,
    ModelVersionStatus,
)
from app.services.model_router import PipelinePlan, build_plan


def load_catalog(path: str) -> tuple[dict[str, Model], dict[int, ModelVersion]]:
    spec = yaml.safe_load(open(path))
    models, versions = {}, {}
    for i, m in enumerate(spec["models"], start=1):
        v = m["version"]
        model = Model(id=m["id"], name=m.get("name", m["id"]), task=ModelTask(m["task"]),
                      backend=InferenceBackend(m.get("backend", "nvinfer")), enabled=True, active_version_id=i)
        mv = ModelVersion(id=i, model_id=m["id"], version=str(v["version"]), status=ModelVersionStatus.ACTIVE,
                          onnx_file=v.get("onnx_file"), engine_file=v.get("engine_file"), labels=v.get("labels", []),
                          precision=v.get("precision", "fp16"), int8_calib_file=v.get("int8_calib_file"),
                          max_batch_size=v.get("max_batch_size", 2), nvinfer_properties=v.get("nvinfer_properties", {}),
                          class_attrs=v.get("class_attrs", {}), nvinferserver_config_file=v.get("nvinferserver_config_file"),
                          embedding_dim=v.get("embedding_dim"))
        models[model.id], versions[i] = model, mv
    return models, versions


def plan_for(catalog, model_ids: list[str], *, cascade_parent: str | None = None, tracker: str = "NvDCF",
             batch: int = 2, tracking: bool = True) -> PipelinePlan:
    models, versions = catalog
    cam = Camera(id="BENCH", name="bench", rtsp_url="", tracking_enabled=tracking, tracker_type=tracker,
                 execution_strategy=ExecutionStrategy.CASCADE if cascade_parent else ExecutionStrategy.INDEPENDENT)
    assigns = [CameraModelAssignment(camera_id="BENCH", model_id=mid, enabled=True, order=i,
                                     parent_model_id=cascade_parent if cascade_parent and mid != cascade_parent else None)
               for i, mid in enumerate(model_ids)]
    return build_plan(cam, assigns, models, versions, batch)


def worker_spec(worker_id: str, plan: PipelinePlan, uris: list[str], gpu: int = 0, capacity: int = 2,
                annotated: bool = False, fps: int = 30) -> dict:
    return {"worker_id": worker_id, "spec_generation": 1, "gpu_index": gpu, "capacity": capacity,
            "plan": plan.to_dict(),
            "sources": [{"camera_id": f"{worker_id}_S{i}", "uri": u, "username_enc": None, "password_enc": None,
                         "camera_generation": 1, "fps": fps, "annotated_output": annotated,
                         "output_path": f"cam/{worker_id}_S{i}/processed"} for i, u in enumerate(uris)]}
