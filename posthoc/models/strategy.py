from __future__ import annotations

import logging

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

logger = logging.getLogger(__name__)


class LogisticStrategy:
    def __init__(self) -> None:
        self.loss_fn = nn.BCEWithLogitsLoss()

    def compute_loss(
        self, model: nn.Module, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = model(x).squeeze(-1)
        loss = self.loss_fn(logits, y)
        return loss, logits

    def metric(self, y_true: np.ndarray, y_pred_raw: np.ndarray) -> float:
        if len(np.unique(y_true)) < 2:
            logger.warning(
                "Cannot compute AUC: validation set contains only one class "
                "(%d positives, %d total).",
                int(y_true.sum()),
                len(y_true),
            )
            return -1.0
        probs = 1.0 / (1.0 + np.exp(-y_pred_raw))
        return float(roc_auc_score(y_true, probs))
