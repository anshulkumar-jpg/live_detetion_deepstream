import logging

import jwt
import pytest

from app.core.logging import JsonFormatter, redact
from app.core.security import (
    CredentialCipher,
    Role,
    create_access_token,
    decode_token,
    hash_password,
    inject_credentials,
    strip_credentials,
    verify_password,
)
from worker.pipeline.config_gen import expected_engine_path, nvinfer_config, tracker_properties, write_stage_configs


def test_password_hash_roundtrip():
    h = hash_password("correct horse battery")
    assert h.startswith("scrypt$")
    assert verify_password("correct horse battery", h)
    assert not verify_password("wrong", h)


def test_jwt_roundtrip_and_tamper(settings):
    tok = create_access_token("alice", Role.OPERATOR, settings)
    claims = decode_token(tok, settings)
    assert claims["sub"] == "alice" and claims["role"] == "operator"
    with pytest.raises(jwt.PyJWTError):
        decode_token(tok[:-2] + "xx", settings)


def test_credentials_split_and_inject():
    clean, u, p = strip_credentials("rtsp://admin:p%40ss@10.0.0.5:554/Streaming/101")
    assert clean == "rtsp://10.0.0.5:554/Streaming/101"
    assert (u, p) == ("admin", "p@ss")
    assert inject_credentials(clean, u, p) == "rtsp://admin:p%40ss@10.0.0.5:554/Streaming/101"
    assert strip_credentials("rtsp://cam/1") == ("rtsp://cam/1", None, None)


def test_cipher(settings):
    c = CredentialCipher(settings.credential_encryption_key)
    t = c.encrypt("secret")
    assert t != "secret" and c.decrypt(t) == "secret"
    assert c.encrypt(None) is None and c.decrypt(None) is None


def test_log_redaction():
    assert redact("connecting rtsp://admin:hunter2@cam.local/x") == "connecting rtsp://***:***@cam.local/x"
    rec = logging.LogRecord("t", logging.INFO, "f", 1, "uri=rtsp://u:p@h/s", None, None)
    rec.camera_id = "CAM_1"
    out = JsonFormatter("svc").format(rec)
    assert "u:p@" not in out and '"camera_id": "CAM_1"' in out


def _stage(**kw):
    base = dict(model_id="person_det", model_version="1", model_version_id=1, task="detector", backend="nvinfer",
                gie_unique_id=1, process_mode=1, precision="fp16", onnx_file="person/model.onnx", engine_file=None,
                labels=["person", "bag"], max_batch_size=2, nvinfer_properties={"net-scale-factor": 0.0039215697,
                                                                               "x-vms-note": "ignored"},
                class_attrs={}, output_tensor_meta=False, operate_on_gie_id=None, operate_on_class_ids=None)
    base.update(kw)
    return base


def test_nvinfer_primary_config():
    cfg = nvinfer_config(_stage(), 0, "/models", "/gen/labels.txt")
    assert "[property]" in cfg
    assert "onnx-file=/models/person/model.onnx" in cfg
    assert f"model-engine-file={expected_engine_path('/models/person/model.onnx', 2, 0, 'fp16')}" in cfg
    assert "network-mode=2" in cfg and "process-mode=1" in cfg and "gie-unique-id=1" in cfg
    assert "net-scale-factor=0.0039215697" in cfg and "x-vms-note" not in cfg
    assert "[class-attrs-all]" in cfg


def test_nvinfer_secondary_cascade_config():
    cfg = nvinfer_config(_stage(model_id="reid", task="reid", gie_unique_id=3, process_mode=2, operate_on_gie_id=1,
                                operate_on_class_ids=[0], output_tensor_meta=True, labels=[]), 0, "/models", None)
    assert "process-mode=2" in cfg and "operate-on-gie-id=1" in cfg and "operate-on-class-ids=0" in cfg
    assert "output-tensor-meta=1" in cfg and "network-type=100" in cfg
    assert "[class-attrs-all]" not in cfg


def test_write_stage_configs(tmp_path):
    plan = {"stages": [_stage(), _stage(model_id="pose", task="pose", gie_unique_id=2, labels=[]),
                       _stage(model_id="ext", backend="nvinferserver", gie_unique_id=3,
                              nvinferserver_config_file="ext/triton.txt")]}
    paths = write_stage_configs(plan, 0, "/models", str(tmp_path))
    assert set(paths) == {1, 2, 3}
    assert (tmp_path / "labels_person_det_1.txt").read_text() == "person\nbag\n"
    assert paths[3] == "/models/ext/triton.txt"


def test_tracker_properties(settings):
    p = tracker_properties("NvDCF", settings, 0)
    assert p["ll-config-file"].endswith("config_tracker_NvDCF_perf.yml")
    assert p["ll-lib-file"].endswith("libnvds_nvmultiobjecttracker.so")
