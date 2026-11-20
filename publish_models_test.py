"""Tests for publish_models.py: README handling, model list, upload loop."""
# pylint: disable=missing-docstring,redefined-outer-name
import os
import sys
import unittest.mock as mock

import pytest

sys.path.insert(0, os.path.dirname(__file__))
import publish_models  # noqa: E402
from gen_readmes import quant_summary  # noqa: E402


def test_model_list_matches_models_dir():
    for name in publish_models.MODELS:
        assert os.path.isdir(os.path.join(publish_models.ROOT, name)), name


@pytest.fixture()
def model_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(publish_models, "ROOT", str(tmp_path))
    d = tmp_path / "fake-model"
    d.mkdir()
    (d / "openvino_model.bin").write_bytes(b"x" * 8)
    (d / "config.json").write_text('{"architectures": ["Qwen2ForCausalLM"]}', encoding="utf-8")
    (d / "openvino_config.json").write_text("{}", encoding="utf-8")
    return d


def _publish(api, folder, dry_run=False):
    return publish_models.publish(api, folder, dry_run)


def test_publish_generates_readme(model_dir):
    assert _publish(mock.Mock(), "fake-model", dry_run=True)
    md = (model_dir / "README.md").read_text(encoding="utf-8")
    assert "# fake-model" in md
    assert "m4gshm/fake-model" in md


def test_publish_keeps_existing_readme(model_dir):
    (model_dir / "README.md").write_text("# hand written\n", encoding="utf-8")
    _publish(mock.Mock(), "fake-model", dry_run=True)
    assert (model_dir / "README.md").read_text(encoding="utf-8") == "# hand written\n"


def test_upload_skips_readme_already_on_hf(model_dir):
    api = mock.Mock()
    api.list_repo_files.return_value = ["README.md", "config.json"]
    _publish(api, "fake-model")
    assert api.upload_folder.call_args.kwargs["ignore_patterns"] == ["README.md"]


def test_upload_includes_readme_for_new_repo(model_dir):
    api = mock.Mock()
    api.list_repo_files.return_value = ["config.json"]
    _publish(api, "fake-model")
    assert not api.upload_folder.call_args.kwargs.get("ignore_patterns")


def test_publish_missing_folder_skipped():
    assert not _publish(mock.Mock(), "nope", dry_run=True)


def test_publish_upload_calls_hf(model_dir):
    api = mock.Mock()
    api.list_repo_files.return_value = ["config.json"]
    assert _publish(api, "fake-model")
    api.create_repo.assert_called_once()
    assert api.create_repo.call_args.args[0] == "m4gshm/fake-model"
    api.upload_folder.assert_called_once()
    assert api.upload_folder.call_args.kwargs["folder_path"] == str(model_dir)


def test_quant_summary_from_folder_name():
    qs = quant_summary({"quantization_config": {"dataset": "gsm8k", "default_config": {"quant_method": "awq"}}},
                       "m-int4-sym-g128-se-awq")
    assert "group_size=128" in qs[0] and "AWQ" in qs[0] and "gsm8k" in qs[0]
    qs8 = quant_summary({}, "m-int8-sym")
    assert qs8[0] == "- Quantization: weights `int8`, symmetric"
