"""
prepare_data.py
----------------
Downloads a subset of the PlantVillage dataset (real photos of tomato leaves,
healthy and affected by 6 different diseases/conditions) and splits it into
train/val/test folders.

Why download with git instead of a direct link?
Since the dataset is hosted as plain files inside a GitHub repo, we do a
partial "sparse checkout" - only downloading the class folders we chose, not
the entire repo (which is huge, several GB with all versions: color/grayscale/segmented).

Run:
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

# 6 tomato leaf classes: healthy + 5 common diseases
CLASSES = [
    "Tomato___healthy",
    "Tomato___Late_blight",
    "Tomato___Early_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Bacterial_spot",
]

# Friendlier display names for the classes, used in the app
CLASS_DISPLAY_NAMES = {
    "Tomato___healthy": "Healthy leaf",
    "Tomato___Late_blight": "Late Blight",
    "Tomato___Early_blight": "Early Blight",
    "Tomato___Leaf_Mold": "Leaf Mold",
    "Tomato___Septoria_leaf_spot": "Septoria Leaf Spot",
    "Tomato___Bacterial_spot": "Bacterial Spot",
}

BASE_DIR = pathlib.Path(__file__).resolve().parent
CLONE_DIR = BASE_DIR / "_plantvillage_repo"
SPLIT_DIR = BASE_DIR / "leaf_split"

# Max number of images to take per class (keeps training fast and reasonable in time/resources).
# You can increase this when running on your own machine (there are thousands of images available per class).
MAX_IMAGES_PER_CLASS = 400

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
SEED = 42


def run(cmd, cwd=None):
    if cwd is not None and cmd[0] == "git":
        # Bypasses git's "dubious ownership" check for this temporary repo only,
        # without touching the user's global git config (mainly relevant on Windows).
        cmd = [cmd[0], "-c", f"safe.directory={cwd}", *cmd[1:]]
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)


def download_subset():
    """Downloads only the relevant class folders from the repo, without history/other files."""
    if CLONE_DIR.exists():
        print("Repo already exists locally, skipping re-download.")
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
            f"  {class_name:32s}: {n:4d} images -> "
            f"train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}"
        )

    print(f"\nTotal {total_images} images split into {SPLIT_DIR}")


def _force_remove_readonly(func, path, exc_info):
    # Git files (like pack files) are sometimes created as read-only on Windows,
    # which causes shutil.rmtree to fail with PermissionError.
    os.chmod(path, stat.S_IWRITE)
    func(path)


def cleanup():
    """Deletes the raw git clone after extraction, to save disk space."""
    if CLONE_DIR.exists():
        shutil.rmtree(CLONE_DIR, onerror=_force_remove_readonly)
        print("Temporary git folder cleaned up.")


if __name__ == "__main__":
    print("Step 1/3: Downloading a subset of the dataset (tomato leaves, PlantVillage)...")
    download_subset()

    print("\nStep 2/3: Splitting into train/val/test folders...")
    split_dataset()

    print("\nStep 3/3: Cleaning up temporary files...")
    cleanup()

    print("\nData preparation completed successfully!")
    print(f"The split data is located at: {SPLIT_DIR}")
