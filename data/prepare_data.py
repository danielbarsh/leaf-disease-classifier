"""
prepare_data.py
----------------
מוריד תת-קבוצה מתוך דאטהסט PlantVillage (תמונות אמיתיות של עלי עגבנייה,
בריאים וחולים ב-6 מחלות/מצבים שונים) ומחלק אותה לתיקיות train/val/test.

למה ה"הורדה" נעשית עם git ולא עם קישור ישיר?
מכיוון שהדאטהסט מתארח כקבצים רגילים בתוך ריפו ב-GitHub, אנחנו עושים
"sparse checkout" חלקי - מורידים רק את תיקיות הקלאסים שבחרנו, ולא
את כל הריפו (שהוא ענק, כמה ג'יגה עם כל הגרסאות: color/grayscale/segmented).

הרצה:
    python data/prepare_data.py
"""

import os
import pathlib
import random
import shutil
import stat
import subprocess
import sys

REPO_URL = "https://github.com/spMohanty/PlantVillage-Dataset.git"

# 6 קלאסים של עלי עגבנייה: בריא + 5 מחלות נפוצות
CLASSES = [
    "Tomato___healthy",
    "Tomato___Late_blight",
    "Tomato___Early_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Bacterial_spot",
]

# תרגום שמות הקלאסים לתצוגה נעימה יותר באפליקציה
CLASS_DISPLAY_NAMES = {
    "Tomato___healthy": "עלה בריא (Healthy)",
    "Tomato___Late_blight": "כשות מאוחרת (Late Blight)",
    "Tomato___Early_blight": "כשות מוקדמת (Early Blight)",
    "Tomato___Leaf_Mold": "עובש עלים (Leaf Mold)",
    "Tomato___Septoria_leaf_spot": "כתמי ספטוריה (Septoria Leaf Spot)",
    "Tomato___Bacterial_spot": "כתמים חיידקיים (Bacterial Spot)",
}

BASE_DIR = pathlib.Path(__file__).resolve().parent
CLONE_DIR = BASE_DIR / "_plantvillage_repo"
SPLIT_DIR = BASE_DIR / "leaf_split"

# כמה תמונות לקחת מכל קלאס לכל היותר (כדי שהאימון יהיה מהיר וסביר בזמן/משאבים).
# אפשר להגדיל את זה כשמריצים על המחשב שלכם (יש אלפי תמונות זמינות לכל קלאס).
MAX_IMAGES_PER_CLASS = 400

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
SEED = 42


def run(cmd, cwd=None):
    if cwd is not None and cmd[0] == "git":
        # עוקף את בדיקת ה"dubious ownership" של git עבור הריפו הזמני הזה בלבד,
        # בלי לגעת בהגדרות ה-git הגלובליות של המשתמש (רלוונטי בעיקר ב-Windows).
        cmd = [cmd[0], "-c", f"safe.directory={cwd}", *cmd[1:]]
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)


def download_subset():
    """מוריד רק את תיקיות הקלאסים הרלוונטיות מהריפו, בלי ההיסטוריה/שאר הקבצים."""
    if CLONE_DIR.exists():
        print("הריפו כבר קיים מקומית, מדלג על הורדה חוזרת.")
        return

    run([
        "git", "clone",
        "--filter=blob:none",
        "--no-checkout",
        "--depth", "1",
        REPO_URL,
        str(CLONE_DIR),
    ])

    sparse_paths = [f"raw/color/{c}" for c in CLASSES]
    run(["git", "sparse-checkout", "init", "--cone"], cwd=CLONE_DIR)
    run(["git", "sparse-checkout", "set", *sparse_paths], cwd=CLONE_DIR)
    run(["git", "checkout", "master"], cwd=CLONE_DIR)


def split_dataset():
    random.seed(SEED)

    if SPLIT_DIR.exists():
        shutil.rmtree(SPLIT_DIR)

    total_images = 0
    for class_name in CLASSES:
        src_dir = CLONE_DIR / "raw" / "color" / class_name
        images = sorted(src_dir.glob("*.JPG")) + sorted(src_dir.glob("*.jpg"))
        random.shuffle(images)
        images = images[:MAX_IMAGES_PER_CLASS]

        n = len(images)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)

        splits = {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:],
        }

        for split_name, split_images in splits.items():
            out_dir = SPLIT_DIR / split_name / class_name
            out_dir.mkdir(parents=True, exist_ok=True)
            for img_path in split_images:
                shutil.copy(img_path, out_dir / img_path.name)

        total_images += n
        print(
            f"  {class_name:32s}: {n:4d} תמונות -> "
            f"train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}"
        )

    print(f"\nסה\"כ {total_images} תמונות חולקו ל-{SPLIT_DIR}")


def _force_remove_readonly(func, path, exc_info):
    # קבצי git (כמו pack files) נוצרים לפעמים כ-read-only ב-Windows,
    # מה שגורם ל-shutil.rmtree להיכשל עם PermissionError.
    os.chmod(path, stat.S_IWRITE)
    func(path)


def cleanup():
    """מוחק את שכפול ה-git הגולמי אחרי החילוץ, כדי לחסוך מקום בדיסק."""
    if CLONE_DIR.exists():
        shutil.rmtree(CLONE_DIR, onerror=_force_remove_readonly)
        print("נוקה תיקיית ה-git הזמנית.")


if __name__ == "__main__":
    print("שלב 1/3: מוריד תת-קבוצה של הדאטהסט (עלי עגבנייה, PlantVillage)...")
    download_subset()

    print("\nשלב 2/3: מחלק לתיקיות train/val/test...")
    split_dataset()

    print("\nשלב 3/3: ניקוי קבצים זמניים...")
    cleanup()

    print("\nהכנת הדאטה הושלמה בהצלחה!")
    print(f"הדאטה המחולק נמצא ב: {SPLIT_DIR}")
