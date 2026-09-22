"""
evaluate.py
-----------
מעריך את המודל המאומן על סט הבדיקה (test) - הסט שהמודל מעולם לא ראה,
לא באימון ולא בבחירת ההיפר-פרמטרים. זה הכי קרוב ל"ביצועים בעולם האמיתי".

מייצר:
- דיוח סיווג (precision/recall/f1 לכל קלאס)
- מטריצת בלבול (confusion matrix) כתמונה
- קובץ metrics.json עם המספרים המרכזיים (שימושי גם לאפליקציית הדמו ול-README)

הרצה:
    python src/evaluate.py
"""

import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import model_utils as mu


def main():
    print(f"טוען מודל מ-{mu.MODEL_PATH}...")
    model = tf.keras.models.load_model(mu.MODEL_PATH)
    class_names = mu.load_class_names()

    test_ds = tf.keras.utils.image_dataset_from_directory(
        mu.DATA_DIR / "test",
        image_size=mu.IMG_SIZE,
        batch_size=mu.BATCH_SIZE,
        shuffle=False,
    )

    print("מריץ חיזויים על סט הבדיקה...")
    y_true = np.concatenate([y.numpy() for _, y in test_ds])
    y_pred_probs = model.predict(test_ds, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    accuracy = float(np.mean(y_true == y_pred))
    print(f"\nדיוק (accuracy) על סט הבדיקה: {accuracy:.2%}")

    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    print("\n" + classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    # --- מטריצת בלבול ---
    # הערה: תוויות הגרף באנגלית בכוונה - matplotlib לא תומך כראוי ב-RTL
    # (עברית תוצג הפוכה/משובשת), ואנגלית גם נפוצה יותר בפורטפוליו טכני.
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
    ax.set_title(f"Confusion Matrix (overall accuracy: {accuracy:.1%})")

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(mu.CONFUSION_MATRIX_PATH, dpi=120)
    print(f"\nמטריצת הבלבול נשמרה ב-{mu.CONFUSION_MATRIX_PATH}")

    # --- שמירת מטריקות לקובץ JSON ---
    metrics = {
        "test_accuracy": accuracy,
        "per_class": {
            class_names[i]: {
                "precision": report[class_names[i]]["precision"],
                "recall": report[class_names[i]]["recall"],
                "f1_score": report[class_names[i]]["f1-score"],
                "support": report[class_names[i]]["support"],
            }
            for i in range(len(class_names))
        },
        "macro_avg_f1": report["macro avg"]["f1-score"],
        "weighted_avg_f1": report["weighted avg"]["f1-score"],
    }
    with open(mu.METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"המטריקות נשמרו ב-{mu.METRICS_PATH}")


if __name__ == "__main__":
    main()
