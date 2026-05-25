import torch
import torch.nn as nn


class QLIKELoss(nn.Module):
    def __init__(self, epsilon=1e-5, reduction="mean"):
        super(QLIKELoss, self).__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        # Clamp both prediction and target to positive values for numerical stability.
        y_pred = torch.clamp(y_pred, min=self.epsilon)
        y_true = torch.clamp(y_true, min=self.epsilon)

        # Compute elementwise QLIKE loss.
        ratio = y_true / y_pred
        loss = ratio - torch.log(ratio) - 1

        # Aggregate based on the configured reduction mode.
        if self.reduction == "mean":
            return torch.mean(loss)
        if self.reduction == "sum":
            return torch.sum(loss)
        if self.reduction == "none":
            return loss
        raise ValueError(f"Unsupported reduction type: {self.reduction}")
