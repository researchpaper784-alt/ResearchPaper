"""Phase 2.1 tests — model factory and metrics."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from fedswarm.eval.metrics import compute_metrics
from fedswarm.models.factory import build_model, count_parameters
from fedswarm.models.resnet import build_resnet18
from fedswarm.models.simple_cnn import SimpleCNN


def test_simple_cnn_is_under_one_million_params() -> None:
    model = SimpleCNN(num_classes=4, norm="groupnorm")
    assert count_parameters(model) < 1_000_000


def test_simple_cnn_forward_shape() -> None:
    model = SimpleCNN(num_classes=4, norm="groupnorm")
    x = torch.randn(2, 3, 112, 112)
    out = model(x)
    assert out.shape == (2, 4)


@pytest.mark.parametrize("norm", ["groupnorm", "batchnorm"])
def test_simple_cnn_builds_with_both_norms(norm: str) -> None:
    model = SimpleCNN(num_classes=4, norm=norm)
    out = model(torch.randn(2, 3, 64, 64))
    assert out.shape == (2, 4)


def test_simple_cnn_rejects_unknown_norm() -> None:
    with pytest.raises(ValueError, match="Unknown norm"):
        SimpleCNN(norm="layernorm")


def test_resnet18_groupnorm_has_no_batchnorm_layers() -> None:
    model = build_resnet18(num_classes=4, pretrained=False, norm="groupnorm")
    for module in model.modules():
        assert not isinstance(module, nn.BatchNorm2d), "GroupNorm swap left a BatchNorm2d"


def test_resnet18_batchnorm_keeps_batchnorm_layers() -> None:
    model = build_resnet18(num_classes=4, pretrained=False, norm="batchnorm")
    assert any(isinstance(m, nn.BatchNorm2d) for m in model.modules())


def test_resnet18_forward_shape_and_head() -> None:
    model = build_resnet18(num_classes=4, pretrained=False, norm="groupnorm")
    out = model(torch.randn(2, 3, 112, 112))
    assert out.shape == (2, 4)
    assert model.fc.out_features == 4


def test_factory_dispatches_to_simple_cnn() -> None:
    model = build_model("simple_cnn", num_classes=4, pretrained=False, norm="gn")
    assert isinstance(model, SimpleCNN)


def test_factory_rejects_pretrained_simple_cnn() -> None:
    with pytest.raises(ValueError, match="pretrained"):
        build_model("simple_cnn", pretrained=True)


def test_factory_rejects_unknown_model() -> None:
    with pytest.raises(ValueError, match="Unknown model"):
        build_model("vit_tiny")


def test_factory_norm_aliases_are_equivalent() -> None:
    a = build_model("simple_cnn", norm="gn")
    b = build_model("simple_cnn", norm="groupnorm")
    assert type(a) is type(b)
    assert count_parameters(a) == count_parameters(b)


# --- metrics ---------------------------------------------------------------------------


def test_compute_metrics_perfect_predictions() -> None:
    labels = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    metrics = compute_metrics(labels, labels.copy())

    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert all(r == 1.0 for r in metrics["per_class_recall"].values())


def test_compute_metrics_detects_majority_class_shortcut() -> None:
    """A model that always predicts the majority class should score well on accuracy
    but poorly on macro-F1 -- the exact failure mode macro-F1 is chosen to catch."""
    labels = np.array([0] * 90 + [1] * 10)
    predictions = np.zeros(100, dtype=int)

    metrics = compute_metrics(labels, predictions)

    assert metrics["accuracy"] == 0.9
    assert metrics["macro_f1"] < 0.5
    assert metrics["per_class_recall"]["glioma"] == 1.0
    assert metrics["per_class_recall"]["meningioma"] == 0.0


def test_compute_metrics_auc_with_probabilities() -> None:
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 4, size=200)
    probs = np.eye(4)[labels] * 0.7 + rng.random((200, 4)) * 0.3
    probs /= probs.sum(axis=1, keepdims=True)

    metrics = compute_metrics(labels, probs.argmax(axis=1), probs)

    assert metrics["auc_ovr_macro"] is not None
    assert 0.5 < metrics["auc_ovr_macro"] <= 1.0


def test_compute_metrics_auc_handles_missing_class_gracefully() -> None:
    """A small federated client under label skew may hold only 2 of 4 classes; AUC
    must degrade to None rather than crash."""
    labels = np.array([0, 0, 1, 1])
    probs = np.tile([0.4, 0.3, 0.2, 0.1], (4, 1))

    metrics = compute_metrics(labels, np.zeros(4, dtype=int), probs)

    assert metrics["auc_ovr_macro"] is None
