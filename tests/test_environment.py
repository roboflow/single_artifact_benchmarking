import hashlib

from sab.environment import collect_environment, file_sha256

EXPECTED_KEYS = {
    "timestamp_utc",
    "hostname",
    "platform",
    "python_version",
    "sab_commit",
    "base_image",
    "hardware_label",
    "gpu_name",
    "driver_version",
    "tensorrt_version",
    "torch_version",
    "torch_cuda_version",
    "onnxruntime_version",
    "l4t_release",
    "jetson_device_model",
}


def test_collect_environment_never_raises_and_has_all_keys():
    env = collect_environment()
    assert EXPECTED_KEYS <= set(env)
    assert env["timestamp_utc"] is not None
    assert env["platform"] is not None


def test_collect_environment_reads_baked_env_vars(monkeypatch):
    monkeypatch.setenv("SAB_GIT_SHA", "abc123")
    monkeypatch.setenv("SAB_BASE_IMAGE", "roboflow/some-image:tag@sha256:deadbeef")
    monkeypatch.setenv("SAB_HARDWARE", "T4")
    env = collect_environment()
    assert env["sab_commit"] == "abc123"
    assert env["base_image"] == "roboflow/some-image:tag@sha256:deadbeef"
    assert env["hardware_label"] == "T4"


def test_collect_environment_ignores_unknown_sha_sentinel(monkeypatch):
    # Dockerfiles default SAB_GIT_SHA=unknown; that must not masquerade as a commit.
    monkeypatch.setenv("SAB_GIT_SHA", "unknown")
    env = collect_environment()
    assert env["sab_commit"] != "unknown"


def test_file_sha256(tmp_path):
    payload = b"sab environment test payload"
    f = tmp_path / "artifact.onnx"
    f.write_bytes(payload)
    assert file_sha256(str(f)) == hashlib.sha256(payload).hexdigest()
