"""Multi-class Focal Loss for class-imbalanced leaf disease classification."""

from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiClassFocalLoss(nn.Module):
    """Focal Loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t).

    Standard cross-entropy weighs every sample equally, so a model can drive
    down loss by getting easy, over-represented classes (e.g. the majority
    disease class) right while still misclassifying hard/rare ones (e.g.
    'Healthy', which is often under-represented and visually close to early
    disease stages). Two correction terms are applied on top of CE:

    - alpha (per-class weight): rebalances class frequency, same role as
      class_weight in weighted CE.
    - (1 - p_t)^gamma (focusing term): down-weights *easy* examples (p_t
      close to 1, term goes to 0) and leaves *hard* examples (p_t close to
      0, term goes to 1) with close to full loss, so training focuses on
      what the model is still getting wrong.
    """

    def __init__(
        self,
        alpha: Optional[Union[float, Sequence[float], torch.Tensor]] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"reduction must be 'mean', 'sum' or 'none', got {reduction!r}")
        self.gamma = gamma
        self.reduction = reduction

        if alpha is None:
            self.alpha = None
        else:
            if not isinstance(alpha, torch.Tensor):
                alpha = torch.tensor(alpha, dtype=torch.float32)
            # Buffer (not Parameter): moves with .to(device)/.cuda(), never trained.
            self.register_buffer("alpha", alpha.float())

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """inputs: raw logits (N, C). targets: class indices (N,)."""
        log_probs = F.log_softmax(inputs, dim=1)

        # log p_t and p_t for each sample's true class only.
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()

        focal_term = (1.0 - pt).pow(self.gamma)
        loss = -focal_term * log_pt

        if self.alpha is not None:
            if self.alpha.numel() == 1:
                alpha_t = self.alpha  # scalar weight applied uniformly
            else:
                alpha_t = self.alpha.gather(0, targets)  # per-sample class weight
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
