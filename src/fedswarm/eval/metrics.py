"""Classification metrics. Macro-F1 is the plan's primary metric (§6.3): the dataset is
class-imbalanced (after Phase 1.2 de-duplication -- notumor is the minority class here,
not the majority as in the raw variant) and a missed tumour is a clinically asymmetric
error; accuracy alone would let a method look good by exploiting the majority class.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, recall_score, roc_auc_score

from fedswarm.data.download import CLASSES


def compute_metrics(
    labels: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray | None = None
) -> dict:
    """`probabilities` is (N, num_classes) softmax output, needed only for AUC."""
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    accuracy = float((labels == predictions).mean())

    macro_f1 = float(f1_score(labels, predictions, average="macro", zero_division=0))
    per_class_recall = {
        cls: float(r)
        for cls, r in zip(
            CLASSES, recall_score(labels, predictions, average=None, zero_division=0, labels=range(len(CLASSES)))
        )
    }

    metrics = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class_recall": per_class_recall,
        "n_samples": int(len(labels)),
    }

    if probabilities is not None:
        try:
            metrics["auc_ovr_macro"] = float(
                roc_auc_score(labels, probabilities, multi_class="ovr", average="macro", labels=range(len(CLASSES)))
            )
        except ValueError:
            # Can happen if a class is absent from this batch of labels (e.g. a small
            # federated client holding only 2 of 4 classes) -- report as unavailable
            # rather than crash, since this is expected under label-skewed partitions.
            metrics["auc_ovr_macro"] = None

    return metrics
