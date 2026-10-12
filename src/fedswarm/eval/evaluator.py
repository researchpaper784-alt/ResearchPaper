"""A full forward pass over a dataloader, turned into metrics."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from fedswarm.eval.metrics import compute_metrics


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict:
    # Moving the MODEL is not the caller's job, because the data movement below is not
    # either. This function took a `device`, moved the batches onto it and left the
    # weights wherever they were, so the contract was half-stated and every caller had to
    # remember the other half. `local_train`, `build_evaluate_fn`, `ServerValFitness` and
    # `run_experiment` all did; `fl/app.py::evaluate_handler` did not, and on Kaggle's T4
    # that ended every round with 10 of 10 clients failing:
    #   RuntimeError: Input type (torch.cuda.FloatTensor) and weight type
    #                 (torch.FloatTensor) should be the same
    # Invisible on CPU, where both halves are the same device, which is why it survived
    # 457 tests and every CPU run this project has ever done. `.to()` is in-place for an
    # nn.Module and a no-op when the weights are already there, so this is free for the
    # callers that were already correct.
    model.to(device)
    model.eval()
    all_labels, all_preds, all_probs, total_loss = [], [], [], 0.0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        total_loss += F.cross_entropy(logits, labels, reduction="sum").item()
        probs = F.softmax(logits, dim=1)

        all_labels.append(labels.cpu().numpy())
        all_preds.append(probs.argmax(dim=1).cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    labels_arr = np.concatenate(all_labels)
    metrics = compute_metrics(labels_arr, np.concatenate(all_preds), np.concatenate(all_probs))
    metrics["loss"] = total_loss / len(labels_arr)
    return metrics
