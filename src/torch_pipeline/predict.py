"""
predict.py (torch_pipeline)
-----------------------------
Calibrated, rejection-aware prediction on a single image from the command
line, using a checkpoint produced by torch_pipeline/train.py.

Run:
    python src/torch_pipeline/predict.py path/to/image.jpg
"""

import pathlib
import sys

import torch
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import model_utils as mu  # noqa: E402

from torch_pipeline import (  # noqa: E402
    LeafDiseaseClassifier,
    ModelWithTemperature,
    infer_with_rejection,
)

MODEL_PATH = mu.MODELS_DIR / "leaf_disease_model_torch.pt"


def load_calibrated_model(device: str = "cpu"):
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)

    model = LeafDiseaseClassifier(
        num_classes=len(checkpoint["class_names"]),
        backbone=checkpoint["backbone"],
        pretrained=False,  # weights are overwritten by the checkpoint below
        dropout=checkpoint["dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    calibrated_model = ModelWithTemperature(model).to(device)
    calibrated_model.temperature.data.fill_(checkpoint["temperature"])

    return model, calibrated_model, checkpoint["class_names"], checkpoint["img_size"]


def predict_image(image_path: str, conf_threshold: float = 0.75, margin_threshold: float = 0.25):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, calibrated_model, class_names, img_size = load_calibrated_model(device)

    image = Image.open(image_path).convert("RGB")
    return infer_with_rejection(
        image=image,
        model=model,
        temperature_scaler=calibrated_model,
        class_names=class_names,
        conf_threshold=conf_threshold,
        margin_threshold=margin_threshold,
        device=device,
        img_size=img_size,
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python src/torch_pipeline/predict.py path/to/image.jpg")
        sys.exit(1)

    result = predict_image(sys.argv[1])

    print(f"\nPrediction for {sys.argv[1]}:")
    print(f"  Status:     {result['status']}")
    print(f"  Prediction: {result['predicted_class']}")
    print(f"  Confidence: {result['confidence']:.1%}")
    print(f"  Margin:     {result['margin']:.1%}")
    print("\n  All classes:")
    for name, prob in sorted(result["probabilities"].items(), key=lambda kv: -kv[1]):
        bar = "█" * int(prob * 40)
        print(f"    {mu.display_name(name):35s} {prob:6.1%}  {bar}")
