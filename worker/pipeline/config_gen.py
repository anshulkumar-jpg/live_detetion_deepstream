"""Generate Gst-nvinfer / Gst-nvtracker configuration from a pipeline plan.

Pure functions (no GStreamer import) so they are unit-testable without a GPU.
Keys follow the Gst-nvinfer configuration-file spec ([property], [class-attrs-*]).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

NETWORK_MODE = {"fp32": 0, "int8": 1, "fp16": 2}
NETWORK_TYPE = {"detector": 0, "classifier": 1, "segmentation": 2, "pose": 100, "reid": 100}

# Official tracker configs shipped with DeepStream (samples/configs/deepstream-app)
TRACKER_CONFIG_FILES = {
    "NvDCF": "config_tracker_NvDCF_perf.yml",
    "NvSORT": "config_tracker_NvSORT.yml",
    "IOU": "config_tracker_IOU.yml",
    "NvDeepSORT": "config_tracker_NvDeepSORT.yml",
}


def resolve(path: str | None, repo: str) -> str | None:
    if not path:
        return None
    return path if os.path.isabs(path) else os.path.join(repo, path)


def expected_engine_path(onnx_path: str, batch: int, gpu: int, precision: str) -> str:
    """Name nvinfer uses when it serialises an engine built from ONNX, so the next start reuses it."""
    return f"{onnx_path}_b{batch}_gpu{gpu}_{precision}.engine"


def _fmt(v: Any) -> str:
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, (list, tuple)):
        return ";".join(str(x) for x in v)
    return str(v)


def nvinfer_config(stage: dict[str, Any], gpu_id: int, repo: str, labels_path: str | None, interval: int = 0) -> str:
    """Render one nvinfer config file for a stage of the plan.

    ``interval`` = batches the primary detector skips between runs (0 = every frame). Secondary models always run
    (their re-inference rate is secondary-reinfer-interval); the tracker runs on every frame regardless."""
    props: dict[str, Any] = {
        "gpu-id": gpu_id,
        "gie-unique-id": stage["gie_unique_id"],
        "process-mode": stage["process_mode"],
        "network-type": NETWORK_TYPE.get(stage["task"], 100),
        "network-mode": NETWORK_MODE[stage.get("precision", "fp16")],
        "batch-size": stage.get("max_batch_size", 1),
        "interval": max(0, int(interval)) if stage["process_mode"] == 1 else 0,
    }
    onnx = resolve(stage.get("onnx_file"), repo)
    engine = resolve(stage.get("engine_file"), repo)
    if onnx:
        props["onnx-file"] = onnx
    if engine:
        props["model-engine-file"] = engine
    elif onnx:
        props["model-engine-file"] = expected_engine_path(onnx, props["batch-size"], gpu_id, stage.get("precision", "fp16"))
    calib = resolve(stage.get("int8_calib_file"), repo)
    if calib:
        props["int8-calib-file"] = calib
    if labels_path:
        props["labelfile-path"] = labels_path
    if stage["process_mode"] == 2:
        props["operate-on-gie-id"] = stage["operate_on_gie_id"]
        if stage.get("operate_on_class_ids"):
            props["operate-on-class-ids"] = stage["operate_on_class_ids"]
    if stage.get("output_tensor_meta"):
        props["output-tensor-meta"] = 1
    if stage["task"] == "classifier":
        props.setdefault("classifier-async-mode", 0)
    # model-specific preprocessing / parser keys from the registry override defaults
    for k, v in (stage.get("nvinfer_properties") or {}).items():
        if k.startswith("x-vms-"):
            continue  # platform-private hints, not nvinfer keys
        props[k] = v
    if stage.get("detect_class_ids") and stage["process_mode"] == 1:  # camera picked a subset of the model's classes
        from app.core.classes import filter_out_for
        drop = filter_out_for(props, stage.get("labels") or [], stage["detect_class_ids"])
        if drop:
            props["filter-out-class-ids"] = drop
        else:
            props.pop("filter-out-class-ids", None)
    for k in ("custom-lib-path", "onnx-file", "model-engine-file", "int8-calib-file", "labelfile-path"):
        if k in props and isinstance(props[k], str):
            props[k] = resolve(props[k], repo)

    lines = ["[property]"] + [f"{k}={_fmt(v)}" for k, v in props.items()]
    class_attrs = stage.get("class_attrs") or {}
    if stage["task"] == "detector" and "all" not in class_attrs:
        class_attrs = {"all": {"pre-cluster-threshold": 0.25}, **class_attrs}
    conf, iou = stage.get("conf_threshold"), stage.get("iou_threshold")
    if stage["task"] == "detector" and stage["process_mode"] == 1 and (conf or iou):
        patched = {}  # the camera chose its own sensitivity: it replaces the model default in every class block
        for cls, attrs in class_attrs.items():
            attrs = dict(attrs)
            if conf:
                attrs["pre-cluster-threshold"] = float(conf)
            if iou:
                attrs["nms-iou-threshold"] = float(iou)
            patched[cls] = attrs
        class_attrs = patched
    for cls, attrs in class_attrs.items():
        lines.append("")
        lines.append(f"[class-attrs-{cls}]")
        lines.extend(f"{k}={_fmt(v)}" for k, v in attrs.items())
    return "\n".join(lines) + "\n"


def write_stage_configs(plan: dict[str, Any], gpu_id: int, repo: str, out_dir: str) -> dict[int, str]:
    """Write config + labels files for every nvinfer stage. Returns {gie_id: config_path}."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    paths: dict[int, str] = {}
    for st in plan["stages"]:
        gid = st["gie_unique_id"]
        if st["backend"] == "nvinferserver":
            paths[gid] = resolve(st["nvinferserver_config_file"], repo)  # type: ignore[assignment]
            continue
        labels_path = None
        if st.get("labels"):
            labels_path = os.path.join(out_dir, f"labels_{st['model_id']}_{st['model_version']}.txt")
            sep = ";" if st["task"] == "classifier" else "\n"
            Path(labels_path).write_text(sep.join(st["labels"]) + "\n")
        cfg = os.path.join(out_dir, f"pgie_{gid}_{st['model_id']}.txt" if st["process_mode"] == 1
                           else f"sgie_{gid}_{st['model_id']}.txt")
        Path(cfg).write_text(nvinfer_config(st, gpu_id, repo, labels_path, int(plan.get("detection_interval") or 0)))
        paths[gid] = cfg
    return paths


def tracker_properties(tracker_type: str, settings, gpu_id: int) -> dict[str, Any]:
    return {
        "ll-lib-file": settings.tracker_lib,
        "ll-config-file": os.path.join(settings.tracker_config_dir, TRACKER_CONFIG_FILES[tracker_type]),
        "tracker-width": 640,
        "tracker-height": 384,
        "gpu-id": gpu_id,
        "display-tracking-id": 1,
    }
