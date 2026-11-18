"""Plan generation used by GPU validation/benchmark scripts, and the runner's failure path."""
import json
import os
import subprocess
import sys

from scripts.localplan import load_catalog, plan_for, worker_spec
from worker.pipeline.config_gen import write_stage_configs

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXAMPLE = os.path.join(ROOT, "config", "models", "models.example.yaml")


def test_example_catalog_builds_all_modes(tmp_path):
    cat = load_catalog(EXAMPLE)
    single = plan_for(cat, ["person_detection"])
    multi = plan_for(cat, ["person_detection", "vehicle_detection"])
    cascade = plan_for(cat, ["person_detection", "pose_detection"], cascade_parent="person_detection")
    assert len({single.signature, multi.signature, cascade.signature}) == 3
    assert cascade.stages[1].process_mode == 2 and cascade.stages[1].operate_on_gie_id == 1
    spec = worker_spec("W", multi, ["file:///x.mp4", "file:///x.mp4"])
    assert len(spec["sources"]) == 2 and spec["capacity"] == 2
    paths = write_stage_configs(spec["plan"], 0, "/models", str(tmp_path))
    txt = open(paths[1]).read()
    assert "custom-lib-path" not in txt and "maintain-aspect-ratio=1" in txt and "pre-cluster-threshold=0.4" in txt


def test_runner_reports_missing_deepstream_runtime(tmp_path, settings):
    cat = load_catalog(EXAMPLE)
    spec = worker_spec("W_FAIL", plan_for(cat, ["person_detection"]), ["file:///x.mp4"])
    p = tmp_path / "spec.json"
    p.write_text(json.dumps(spec))
    env = {**os.environ, "KAFKA_ENABLED": "false", "CREDENTIAL_ENCRYPTION_KEY": settings.credential_encryption_key,
           "DEEPSTREAM_CONFIG_DIR": str(tmp_path), "LOG_JSON": "true"}
    try:
        import gi  # noqa: F401
        import pyds  # noqa: F401
        import pytest
        pytest.skip("DeepStream runtime present; covered by scripts/validate_gpu_node.py")
    except ImportError:
        pass
    r = subprocess.run([sys.executable, "-m", "worker.runner", "--spec", str(p)], cwd=ROOT, env=env,
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 3
    assert r.stdout == ""  # stdout is reserved for the agent protocol
