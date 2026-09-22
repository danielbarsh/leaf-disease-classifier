"""
model_utils.py
----------------
Constants and helper functions shared across all scripts (training, evaluation,
prediction, demo app). Keeping all these "sources of truth" in one place
prevents inconsistencies between the scripts.
"""

import json
import pathlib

# --- Paths ---
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "leaf_split"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "leaf_disease_model.keras"
CLASS_NAMES_PATH = MODELS_DIR / "class_names.json"
HISTORY_PLOT_PATH = MODELS_DIR / "training_history.png"
CONFUSION_MATRIX_PATH = MODELS_DIR / "confusion_matrix.png"
METRICS_PATH = MODELS_DIR / "metrics.json"

# --- Model hyperparameters ---
IMG_SIZE = (160, 160)          # image size the model expects (MobileNetV2)
BATCH_SIZE = 32
SEED = 42

# Friendly display names for the classes, shown in the app
CLASS_DISPLAY_NAMES = {
    "Tomato___healthy": "Healthy leaf",
    "Tomato___Late_blight": "Late Blight",
    "Tomato___Early_blight": "Early Blight",
    "Tomato___Leaf_Mold": "Leaf Mold",
    "Tomato___Septoria_leaf_spot": "Septoria Leaf Spot",
    "Tomato___Bacterial_spot": "Bacterial Spot",
}


def save_class_names(class_names):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(CLASS_NAMES_PATH, "w", encoding="utf-8") as f:
        json.dump(list(class_names), f, ensure_ascii=False, indent=2)


def load_class_names():
    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def display_name(class_name: str) -> str:
    return CLASS_DISPLAY_NAMES.get(class_name, class_name)
