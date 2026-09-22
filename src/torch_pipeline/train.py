"""
train.py (torch_pipeline)
--------------------------
End-to-end training script that wires the robust-pipeline building blocks
(augmentation, focal loss, transfer-learning model, temperature scaling)
into a runnable equivalent of src/train.py, but on PyTorch.

Two-stage transfer learning, same idea as the Keras version:
  1. Train only the head (backbone frozen)
  2. Unfreeze the backbone and fine-tune end-to-end at a low LR
Then the best checkpoint is calibrated with temperature scaling on the
validation set, and evaluated on the held-out test set.

Run:
    python src/torch_pipeline/train.py
"""

import argparse
import json
import pathlib
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import model_utils as mu  # noqa: E402 - shared paths/constants from the Keras pipeline

from torch_pipeline import (  # noqa: E402
    LeafDiseaseClassifier,
    ModelWithTemperature,
    MultiClassFocalLoss,
    get_train_transforms,
    get_val_transforms,
)

MODEL_PATH = mu.MODELS_DIR / "leaf_disease_model_torch.pt"
HISTORY_PLOT_PATH = mu.MODELS_DIR / "training_history_torch.png"
CONFUSION_MATRIX_PATH = mu.MODELS_DIR / "confusion_matrix_torch.png"
METRICS_PATH = mu.MODELS_DIR / "metrics_torch.json"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backbone", default="efficientnet_b0",
                    help="timm/torchvision backbone name (default: efficientnet_b0)")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs-head", type=int, default=5)
    p.add_argument("--epochs-finetune", type=int, default=3)
    p.add_argument("--lr-head", type=float, default=3e-4)
    p.add_argument("--lr-finetune", type=float, default=1e-5)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--gamma", type=float, default=2.0, help="focal loss focusing parameter")
    p.add_argument("--num-workers", type=int, default=0)
    return p.parse_args()


def build_dataloaders(img_size: int, batch_size: int, num_workers: int):
    train_ds = ImageFolder(mu.DATA_DIR / "train", transform=get_train_transforms(img_size))
    val_ds = ImageFolder(mu.DATA_DIR / "val", transform=get_val_transforms(img_size))
    test_ds = ImageFolder(mu.DATA_DIR / "test", transform=get_val_transforms(img_size))

    # ImageFolder assigns indices by sorted folder name - same order as the
    # Keras pipeline's image_dataset_from_directory, so class_names line up.
    class_names = train_ds.classes

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, class_names


def compute_class_weights(train_ds: ImageFolder, num_classes: int) -> torch.Tensor:
    """Inverse-frequency alpha for MultiClassFocalLoss: rarer classes get a
    higher weight, so e.g. an under-represented 'Healthy' folder isn't
    drowned out by more common disease classes."""
    counts = np.bincount([label for _, label in train_ds.samples], minlength=num_classes)
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(model, loader, criterion, device, optimizer=None):
    """One pass over `loader`. Trains if `optimizer` is given, else evaluates."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(is_train):
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)
            loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += inputs.size(0)

    return total_loss / total, correct / total


def train_stage(model, train_loader, val_loader, criterion, optimizer, epochs, device, history, best_state):
    for epoch in range(epochs):
        start = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_state["val_acc"]:
            best_state["val_acc"] = val_acc
            best_state["state_dict"] = {k: v.clone() for k, v in model.state_dict().items()}

        print(f"  epoch {epoch + 1}/{epochs} "
              f"[{time.time() - start:.0f}s] "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.2%} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.2%}")


def plot_history(history: dict, switch_epoch: int, out_path: pathlib.Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(history["train_acc"], label="Train accuracy")
    axes[0].plot(history["val_acc"], label="Validation accuracy")
    axes[0].axvline(switch_epoch - 0.5, color="gray", linestyle="--", label="Fine-tuning starts")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history["train_loss"], label="Train loss")
    axes[1].plot(history["val_loss"], label="Validation loss")
    axes[1].axvline(switch_epoch - 0.5, color="gray", linestyle="--", label="Fine-tuning starts")
    axes[1].set_title("Loss (focal)")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"Training plot saved to {out_path}")


@torch.no_grad()
def evaluate_on_test(calibrated_model, test_loader, class_names, device):
    calibrated_model.eval()
    y_true, y_pred = [], []
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        probs = torch.softmax(calibrated_model(inputs), dim=1)
        y_pred.extend(probs.argmax(dim=1).cpu().tolist())
        y_true.extend(labels.tolist())

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    accuracy = float((y_true == y_pred).mean())
    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    print(f"\nTest accuracy (calibrated): {accuracy:.2%}")
    print("\n" + classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    short_names = [n.replace("Tomato___", "").replace("_", " ") for n in class_names]
    ax.set_xticklabels(short_names, rotation=45, ha="right")
    ax.set_yticklabels(short_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"Confusion Matrix - torch pipeline (accuracy: {accuracy:.1%})")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(CONFUSION_MATRIX_PATH, dpi=120)
    print(f"Confusion matrix saved to {CONFUSION_MATRIX_PATH}")

    return accuracy, report


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    mu.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading datasets...")
    train_loader, val_loader, test_loader, class_names = build_dataloaders(
        args.img_size, args.batch_size, args.num_workers
    )
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}")

    alpha = compute_class_weights(train_loader.dataset, num_classes)
    print(f"Focal loss alpha (inverse class frequency): {alpha.tolist()}")
    criterion = MultiClassFocalLoss(alpha=alpha, gamma=args.gamma).to(device)

    print(f"\nBuilding model ({args.backbone} + Transfer Learning)...")
    model = LeafDiseaseClassifier(
        num_classes=num_classes, backbone=args.backbone,
        pretrained=True, dropout=args.dropout, freeze_backbone=True,
    ).to(device)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_state = {"val_acc": -1.0, "state_dict": None}

    print(f"\n=== Stage 1/2: Training head ({args.epochs_head} epochs, backbone frozen) ===")
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr_head
    )
    train_stage(model, train_loader, val_loader, criterion, optimizer,
                args.epochs_head, device, history, best_state)

    print(f"\n=== Stage 2/2: Fine-tuning ({args.epochs_finetune} epochs, backbone unfrozen) ===")
    model.unfreeze_backbone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr_finetune)
    train_stage(model, train_loader, val_loader, criterion, optimizer,
                args.epochs_finetune, device, history, best_state)

    print(f"\nBest validation accuracy during training: {best_state['val_acc']:.2%}")
    model.load_state_dict(best_state["state_dict"])

    plot_history(history, switch_epoch=args.epochs_head, out_path=HISTORY_PLOT_PATH)

    print("\n=== Calibrating confidence (temperature scaling) on the validation set ===")
    calibrated_model = ModelWithTemperature(model).to(device)
    calibrated_model.set_temperature(val_loader, device=device)

    accuracy, report = evaluate_on_test(calibrated_model, test_loader, class_names, device)

    print(f"\nSaving model checkpoint to {MODEL_PATH}")
    torch.save({
        "model_state_dict": model.state_dict(),
        "temperature": calibrated_model.temperature.item(),
        "class_names": class_names,
        "backbone": args.backbone,
        "img_size": args.img_size,
        "dropout": args.dropout,
    }, MODEL_PATH)

    metrics = {
        "test_accuracy": accuracy,
        "temperature": calibrated_model.temperature.item(),
        "backbone": args.backbone,
        "per_class": {
            class_names[i]: {
                "precision": report[class_names[i]]["precision"],
                "recall": report[class_names[i]]["recall"],
                "f1_score": report[class_names[i]]["f1-score"],
                "support": report[class_names[i]]["support"],
            }
            for i in range(num_classes)
        },
        "macro_avg_f1": report["macro avg"]["f1-score"],
        "weighted_avg_f1": report["weighted avg"]["f1-score"],
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"Metrics saved to {METRICS_PATH}")
    print("\nDone. Use infer_with_rejection() from torch_pipeline for calibrated,"
          " rejection-aware predictions with this checkpoint.")


if __name__ == "__main__":
    main()
