"""Post-training temperature scaling (Guo et al., 2017) for confidence calibration."""

import torch
import torch.nn as nn


class ModelWithTemperature(nn.Module):
    """Wraps a trained classifier with a single learned temperature T.

    Softmax(logits / T) leaves argmax (and thus accuracy) unchanged, because
    dividing all logits by the same positive scalar preserves their rank
    order. It only reshapes how peaked the distribution is: T > 1 flattens
    probabilities (fixes overconfidence, the common case for modern CNNs),
    T < 1 sharpens them. T is fit once, after training, on held-out data.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        # Initialized > 1 (typical CNNs start overconfident); learned via set_temperature.
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(x)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits: torch.Tensor) -> torch.Tensor:
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    def set_temperature(
        self,
        valid_loader,
        device: str = "cpu",
        lr: float = 0.01,
        max_iter: int = 50,
    ) -> "ModelWithTemperature":
        """Fit T by minimizing NLL (cross-entropy) on the validation logits.

        The backbone is frozen (no_grad forward pass, weights untouched);
        only the scalar `self.temperature` receives gradients. This is a
        1-D convex optimization problem, so L-BFGS converges in a handful
        of iterations without a learning-rate schedule.
        """
        self.to(device)
        self.model.eval()

        nll_criterion = nn.CrossEntropyLoss().to(device)
        ece_criterion = _ECELoss().to(device)

        logits_list, labels_list = [], []
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs = inputs.to(device)
                logits_list.append(self.model(inputs))
                labels_list.append(labels)
        logits = torch.cat(logits_list).to(device)
        labels = torch.cat(labels_list).to(device)

        before_nll = nll_criterion(logits, labels).item()
        before_ece = ece_criterion(logits, labels).item()

        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss

        optimizer.step(closure)

        after_nll = nll_criterion(self.temperature_scale(logits), labels).item()
        after_ece = ece_criterion(self.temperature_scale(logits), labels).item()

        print(f"Optimal temperature: {self.temperature.item():.4f}")
        print(f"NLL:  {before_nll:.4f} -> {after_nll:.4f}")
        print(f"ECE:  {before_ece:.4f} -> {after_ece:.4f}")
        return self


class _ECELoss(nn.Module):
    """Expected Calibration Error: bins predictions by confidence and compares
    each bin's average confidence to its actual accuracy. A well-calibrated
    model has |confidence - accuracy| ~= 0 in every bin.
    """

    def __init__(self, n_bins: int = 15):
        super().__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        confidences, predictions = torch.max(probs, dim=1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for lower, upper in zip(self.bin_lowers, self.bin_uppers):
            in_bin = confidences.gt(lower.item()) & confidences.le(upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                acc_in_bin = accuracies[in_bin].float().mean()
                conf_in_bin = confidences[in_bin].mean()
                ece += (conf_in_bin - acc_in_bin).abs() * prop_in_bin
        return ece
