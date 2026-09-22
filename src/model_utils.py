"""
model_utils.py
----------------
קבועים ופונקציות עזר משותפות לכל הסקריפטים (אימון, הערכה, חיזוי, אפליקציית הדמו).
לשמור את כל ה"מקורות אמת" האלה במקום אחד מונע חוסר-התאמות בין הסקריפטים.
"""

import json
import pathlib

# --- נתיבים ---
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "leaf_split"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "leaf_disease_model.keras"
CLASS_NAMES_PATH = MODELS_DIR / "class_names.json"
HISTORY_PLOT_PATH = MODELS_DIR / "training_history.png"
CONFUSION_MATRIX_PATH = MODELS_DIR / "confusion_matrix.png"
METRICS_PATH = MODELS_DIR / "metrics.json"

# --- היפר-פרמטרים של המודל ---
IMG_SIZE = (160, 160)          # גודל התמונה שהמודל מצפה לו (MobileNetV2)
BATCH_SIZE = 32
SEED = 42

# שמות הקלאסים לתצוגה ידידותית (בעברית) באפליקציה
CLASS_DISPLAY_NAMES = {
    "Tomato___healthy": "עלה בריא (Healthy)",
    "Tomato___Late_blight": "כשות מאוחרת (Late Blight)",
    "Tomato___Early_blight": "כשות מוקדמת (Early Blight)",
    "Tomato___Leaf_Mold": "עובש עלים (Leaf Mold)",
    "Tomato___Septoria_leaf_spot": "כתמי ספטוריה (Septoria Leaf Spot)",
    "Tomato___Bacterial_spot": "כתמים חיידקיים (Bacterial Spot)",
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
