"""Tests for self-contained classifier model bundle resolution."""

from pathlib import Path

import pytest

from smart_router_proxy.bert_classifier import resolve_base_model


def test_resolve_base_model_prefers_bundled_weights(tmp_path: Path) -> None:
    model_dir = tmp_path / "classifier-model"
    base_dir = model_dir / "base_model"
    base_dir.mkdir(parents=True)
    (base_dir / "config.json").write_text("{}")
    (base_dir / "model.safetensors").write_bytes(b"weights")

    source, local_only = resolve_base_model(model_dir, "distilbert-base-uncased")

    assert source == str(base_dir)
    assert local_only is True


def test_resolve_base_model_rejects_incomplete_bundle(tmp_path: Path) -> None:
    model_dir = tmp_path / "classifier-model"
    base_dir = model_dir / "base_model"
    base_dir.mkdir(parents=True)
    (base_dir / "config.json").write_text("{}")

    with pytest.raises(FileNotFoundError, match="model.safetensors"):
        resolve_base_model(model_dir, "distilbert-base-uncased")


def test_resolve_base_model_keeps_legacy_huggingface_fallback(tmp_path: Path) -> None:
    source, local_only = resolve_base_model(tmp_path, "distilbert-base-uncased")

    assert source == "distilbert-base-uncased"
    assert local_only is False
