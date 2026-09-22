"""Calibrated inference with rejection for out-of-distribution / ambiguous inputs."""

from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from PIL import Image

from .augmentation import get_val_transforms

STATUS_OK = "OK"
STATUS_OOD = "Uncertain / Unknown (OOD)"
STATUS_AMBIGUOUS = "Ambiguous prediction (Needs recapture)"


@torch.no_grad()
def infer_with_rejection(
    image: Image.Image,
    model: torch.nn.Module,
    temperature_scaler: Optional[Union[torch.nn.Module, float, torch.Tensor]],
    class_names: List[str],
    conf_threshold: float = 0.75,
    margin_threshold: float = 0.25,
    device: str = "cpu",
    img_size: int = 224,
) -> Dict[str, Any]:
    """Predict with two independent rejection gates instead of trusting argmax blindly.

    1. Low top-1 confidence -> the input doesn't look like any training
       class with conviction, e.g. a non-leaf / OOD photo.
    2. Small top1-vs-top2 margin -> the model is genuinely torn between two
       classes (common for visually similar diseases), so the single label
       is unreliable even if its raw confidence is high enough to pass (1).
    """
    model.eval()
    transform = get_val_transforms(img_size)
    x = transform(image).unsqueeze(0).to(device)

    logits = model(x)

    # Calibrate with T before taking softmax: p = softmax(logits / T).
    if temperature_scaler is None:
        calibrated_logits = logits
    elif hasattr(temperature_scaler, "temperature_scale"):
        calibrated_logits = temperature_scaler.temperature_scale(logits)
    else:
        calibrated_logits = logits / temperature_scaler

    probs = F.softmax(calibrated_logits, dim=1).squeeze(0)

    k = min(2, probs.numel())
    top_probs, top_idx = torch.topk(probs, k=k)
    top1_prob = top_probs[0].item()
    top1_idx = top_idx[0].item()
    top2_prob = top_probs[1].item() if k > 1 else 0.0
    margin = top1_prob - top2_prob

    probability_dist = {class_names[i]: probs[i].item() for i in range(probs.numel())}
    predicted_class = class_names[top1_idx]

    if top1_prob < conf_threshold:
        status = STATUS_OOD
        predicted_class = None
    elif margin < margin_threshold:
        status = STATUS_AMBIGUOUS
    else:
        status = STATUS_OK

    return {
        "status": status,
        "predicted_class": predicted_class,
        "confidence": top1_prob,
        "margin": margin,
        "probabilities": probability_dist,
    }
