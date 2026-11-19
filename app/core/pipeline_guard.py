"""Checks that a worker spec cannot make a GPU node load code or read files it should not.

A worker spec is turned into DeepStream config files, and some settings in it are *loaded as code*
(``custom-lib-path`` makes nvinfer dlopen() a shared library). So the spec is checked in three places:
when a model version is registered (API), when the GPU agent receives a command, and when the worker starts.

Rules
* A model may only set settings from an ALLOW-LIST (MODEL_NVINFER_KEYS). Anything else is refused, so a new or
  dangerous DeepStream option can never slip in. Settings the platform controls itself (batch size, gpu, ids, the
  main model file, ...) cannot be changed by a model.
* ``custom-lib-path`` must be an absolute path inside ALLOWED_LIB_DIRS (default /opt/vms/lib, built into the image,
  not writable by the models volume). Nothing else may load code.
* Model files (onnx, engine, calibration, labels, ...) must stay inside the model repository, no ``..``, no other folders.
* Settings that name a function must be plain identifiers.
* Per-class settings (class_attrs) are allow-listed too, and must be numbers.
* Camera sources must be rtsp:// or rtsps:// (file:// only when ALLOW_FILE_SOURCES is on, for benchmarks).
"""
from __future__ import annotations

import posixpath
import re
from typing import Any

from app.core.config import Settings
from app.core.trackers import ALL_TRACKERS

IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Settings a model version may set (Gst-nvinfer [property] keys that describe the model, not the platform).
MODEL_NVINFER_KEYS = frozenset({
    # pre-processing
    "net-scale-factor", "offsets", "model-color-format", "maintain-aspect-ratio", "symmetric-padding",
    "scaling-filter", "scaling-compute-hw", "infer-dims", "network-input-order", "uff-input-order",
    "uff-input-blob-name", "input-tensor-meta",
    # outputs and post-processing
    "num-detected-classes", "cluster-mode", "filter-out-class-ids", "output-blob-names", "output-tensor-meta",
    "network-type", "classifier-threshold", "classifier-async-mode", "segmentation-threshold",
    "segmentation-output-order", "disable-output-host-copy", "crop-objects-to-roi-boundary",
    # secondary models
    "secondary-reinfer-interval", "input-object-min-width", "input-object-min-height",
    "input-object-max-width", "input-object-max-height",
    # custom code: only from the allowed library folder (checked separately)
    "custom-lib-path", "parse-bbox-func-name", "parse-bbox-instance-mask-func-name", "parse-classifier-func-name",
    "engine-create-func-name",
    # other model files: only inside the model repository (checked separately)
    "model-file", "proto-file", "uff-file", "tlt-encoded-model", "mean-file", "custom-network-config", "tlt-model-key",
    # engine build options
    "workspace-size", "force-implicit-batch-dim", "layer-device-precision", "enable-dla", "use-dla-core",
    "tensor-meta-pool-size",
})
# Set by the platform from the plan; a model must not override them.
PIPELINE_CONTROLLED_KEYS = frozenset({
    "gpu-id", "gie-unique-id", "process-mode", "network-mode", "batch-size", "interval", "operate-on-gie-id",
    "operate-on-class-ids", "labelfile-path", "onnx-file", "model-engine-file", "int8-calib-file",
})
CLASS_ATTR_KEYS = frozenset({
    "pre-cluster-threshold", "post-cluster-threshold", "eps", "group-threshold", "minBoxes", "dbscan-min-score",
    "nms-iou-threshold", "threshold", "roi-top-offset", "roi-bottom-offset", "detected-min-w", "detected-min-h",
    "detected-max-w", "detected-max-h", "topk",
})
LIB_KEYS = {"custom-lib-path"}
PATH_KEYS = {"model-file", "proto-file", "uff-file", "tlt-encoded-model", "mean-file", "custom-network-config"}
STAGE_PATH_FIELDS = ("onnx_file", "engine_file", "int8_calib_file", "nvinferserver_config_file")
BACKENDS = {"nvinfer", "nvinferserver"}


def _norm(p: str) -> str:
    return posixpath.normpath(str(p).replace("\\", "/"))


def _clean(p: Any) -> bool:
    return isinstance(p, str) and p != "" and "\x00" not in p and "\n" not in p and "\r" not in p


def _inside(path: str, base: str) -> bool:
    p, b = _norm(path), _norm(base).rstrip("/")
    if not posixpath.isabs(p):  # relative paths are resolved under the model repository
        return p != ".." and not p.startswith("../")
    return p == b or p.startswith(b + "/")


def property_problems(props: Any, name: str = "model") -> list[str]:
    """Allow-list check of a model's nvinfer settings. Empty list = fine."""
    if not isinstance(props, dict):
        return [f"{name}: nvinfer_properties must be an object"]
    problems: list[str] = []
    for key, val in props.items():
        if not isinstance(key, str):
            problems.append(f"{name}: setting names must be text")
            continue
        if key.startswith("x-vms-"):  # platform hints, never passed to nvinfer
            if isinstance(val, (dict, list)):
                problems.append(f"{name}: '{key}' must be a plain value")
            continue
        if key in PIPELINE_CONTROLLED_KEYS:
            problems.append(f"{name}: '{key}' is set by the platform; a model cannot change it")
        elif key not in MODEL_NVINFER_KEYS:
            problems.append(f"{name}: setting '{key}' is not allowed in a model")
        elif isinstance(val, (dict, list)) or val is None:
            problems.append(f"{name}: '{key}' must be a number, text or true/false")
        elif isinstance(val, str) and (len(val) > 512 or any(c in val for c in "\n\r\x00")):
            problems.append(f"{name}: invalid value for '{key}'")
    return problems


def class_attr_problems(class_attrs: Any, name: str = "model") -> list[str]:
    if not isinstance(class_attrs, dict):
        return [f"{name}: class_attrs must be an object"]
    problems: list[str] = []
    for cls, attrs in class_attrs.items():
        if not (cls == "all" or (isinstance(cls, str) and cls.isdigit())):
            problems.append(f"{name}: class_attrs name '{cls}' must be 'all' or a class number")
        if not isinstance(attrs, dict):
            problems.append(f"{name}: class_attrs['{cls}'] must be an object")
            continue
        for k, v in attrs.items():
            if k not in CLASS_ATTR_KEYS:
                problems.append(f"{name}: class setting '{k}' is not allowed")
            elif isinstance(v, bool) or not isinstance(v, (int, float)):
                problems.append(f"{name}: class setting '{k}' must be a number")
    return problems


def check_stage(stage: dict[str, Any], model_dir: str, lib_dirs: list[str]) -> list[str]:
    """Return a list of problems (empty = fine) for one inference stage / model version."""
    name = stage.get("model_id") or "model"
    problems: list[str] = []
    if stage.get("backend") not in (None, *BACKENDS):
        problems.append(f"{name}: unknown backend {stage.get('backend')!r}")
    for f in STAGE_PATH_FIELDS:
        v = stage.get(f)
        if v in (None, ""):
            continue
        if not _clean(v) or not _inside(v, model_dir):
            problems.append(f"{name}: {f} must stay inside the model repository ({model_dir})")
    props = stage.get("nvinfer_properties") or {}
    problems += property_problems(props, name)
    problems += class_attr_problems(stage.get("class_attrs") or {}, name)
    if isinstance(props, dict):
        for key, val in props.items():
            if not isinstance(key, str) or key.startswith("x-vms-"):
                continue
            if key in LIB_KEYS:
                if not _clean(val) or not posixpath.isabs(_norm(val)) or not any(_inside(val, d) for d in lib_dirs):
                    problems.append(f"{name}: custom-lib-path must be an absolute path inside {', '.join(lib_dirs)}")
            elif key in PATH_KEYS:
                if not _clean(val) or not _inside(val, model_dir):
                    problems.append(f"{name}: '{key}' must stay inside the model repository ({model_dir})")
            elif key.endswith("-func-name") and key in MODEL_NVINFER_KEYS:
                if not isinstance(val, str) or not IDENT.match(val):
                    problems.append(f"{name}: '{key}' must be a plain function name")
    return problems


def check_source(src: dict[str, Any], allow_file: bool = False) -> list[str]:
    uri = str(src.get("uri") or src.get("rtsp_url") or "")
    ok = uri.startswith(("rtsp://", "rtsps://")) or (allow_file and uri.startswith("file://"))
    if not ok or "\x00" in uri or "\n" in uri or "\r" in uri:
        return [f"camera {src.get('camera_id', '?')}: source address must start with rtsp:// or rtsps://"]
    return []


def check_spec(spec: dict[str, Any], s: Settings, allow_file: bool | None = None) -> list[str]:
    """Check a whole worker spec (stages + sources + tracker)."""
    problems: list[str] = []
    plan = spec.get("plan") or {}
    if plan.get("tracker_type") not in ALL_TRACKERS:
        problems.append(f"unknown tracker {plan.get('tracker_type')!r}")
    for st in plan.get("stages") or []:
        problems += check_stage(st, s.model_repository_dir, s.allowed_lib_dirs)
    for src in spec.get("sources") or []:
        problems += check_source(src, s.allow_file_sources if allow_file is None else allow_file)
    return problems
