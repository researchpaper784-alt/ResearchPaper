"""A full forward pass over a dataloader, turned into metrics."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from fedswarm.eval.metrics import compute_metrics


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict:
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
