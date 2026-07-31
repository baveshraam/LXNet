"""The GPU guard: a silent CPU fallback is the expensive failure, so it must abort."""
import pytest
import tensorflow as tf

from lxnet import train


@pytest.fixture
def no_gpu(monkeypatch):
    monkeypatch.setattr(tf.config, "list_physical_devices", lambda kind: [])


def test_refuses_to_train_on_cpu_by_default(no_gpu, tmp_path):
    with pytest.raises(SystemExit, match="refusing to train on CPU"):
        train.main(["--out-dir", str(tmp_path), "--data-root", str(tmp_path)])


def test_allow_cpu_opts_back_in(no_gpu, tmp_path):
    """Reaching dataset indexing proves the guard let it through."""
    with pytest.raises(ValueError, match="no class directories"):
        train.main(["--allow-cpu", "--out-dir", str(tmp_path), "--data-root", str(tmp_path)])
