"""
train.py
--------
מאמן מודל לסיווג תמונות עלי עגבנייה (בריא / 5 מחלות) בעזרת Transfer Learning
על גבי MobileNetV2 שאומן מראש על ImageNet.

למה Transfer Learning?
במקום לאמן רשת חדשה מאפס (שדורשת המון תמונות וזמן חישוב), אנחנו לוקחים
רשת שכבר "יודעת לראות" (זוהתה על מיליוני תמונות), מקפיאים אותה, ומוסיפים
עליה שכבות סיווג קטנות שמתאימות במיוחד למשימה שלנו. זו הגישה הסטנדרטית
בתעשייה כשיש לך דאטהסט קטן-בינוני.

הרצה:
    python src/train.py
"""

import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")  # לא צריך חלון גרפי בשרת
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import model_utils as mu

EPOCHS_HEAD = 8           # אימון השכבות העליונות (הבסיס קפוא)
EPOCHS_FINE_TUNE = 4      # שלב שני: "מפשירים" חלק מהבסיס ומעדנים בקצב איטי
FINE_TUNE_AT_LAYER = 100  # מאיזו שכבה בבסיס להתחיל לאמן (שכבות נמוכות יותר נשארות קפואות)
LEARNING_RATE_HEAD = 1e-3
LEARNING_RATE_FINE_TUNE = 1e-5


def build_datasets():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        mu.DATA_DIR / "train",
        image_size=mu.IMG_SIZE,
        batch_size=mu.BATCH_SIZE,
        seed=mu.SEED,
        shuffle=True,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        mu.DATA_DIR / "val",
        image_size=mu.IMG_SIZE,
        batch_size=mu.BATCH_SIZE,
        seed=mu.SEED,
        shuffle=False,
    )
    class_names = train_ds.class_names

    # אופטימיזציה של קלט/פלט - טוען מראש את הבאטצ' הבא בזמן שהמודל מעבד את הנוכחי
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=autotune)
    val_ds = val_ds.prefetch(buffer_size=autotune)

    return train_ds, val_ds, class_names


def load_pretrained_base():
    """
    טוען את MobileNetV2 עם משקולות ImageNet.

    ברשתות ארגוניות/מוגבלות מסוימות ההורדה הרגילה מ-storage.googleapis.com
    עלולה להיחסם. אם זה קורה, נופלים בחזרה למראה (mirror) ציבורי של אותם
    משקולות בדיוק שמתארח ב-GitHub Releases, ושמים אותו בקאש של Keras
    כדי שההורדה הרגילה של Keras "תמצא" אותו ולא תנסה שוב את הכתובת החסומה.
    """
    try:
        return tf.keras.applications.MobileNetV2(
            input_shape=mu.IMG_SIZE + (3,),
            include_top=False,
            weights="imagenet",
        )
    except Exception as e:
        print(f"הורדה רגילה של משקולות ImageNet נכשלה ({e}).")
        print("מנסה מראה חלופי ב-GitHub...")

        import urllib.request
        cache_path = pathlib.Path.home() / ".keras" / "models" / (
            "mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_160_no_top.h5"
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        mirror_url = (
            "https://github.com/JonathanCMitchell/mobilenet_v2_keras/"
            "releases/download/v1.1/"
            "mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_160_no_top.h5"
        )
        urllib.request.urlretrieve(mirror_url, cache_path)
        print(f"המשקולות הורדו בהצלחה מהמראה החלופי ל-{cache_path}")

        return tf.keras.applications.MobileNetV2(
            input_shape=mu.IMG_SIZE + (3,),
            include_top=False,
            weights="imagenet",
        )


def build_model(num_classes: int):
    # שכבות "הגברה" (Data Augmentation) - מגוונות מעט את תמונות האימון
    # (סיבוב, היפוך, זום) כדי שהמודל יכליל טוב יותר ולא ישנן בעל פה.
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ], name="data_augmentation")

    # MobileNetV2 מצפה לקלט בטווח [-1, 1]
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

    base_model = load_pretrained_base()
    base_model.trainable = False  # שלב 1: קופאים על הבסיס

    inputs = tf.keras.Input(shape=mu.IMG_SIZE + (3,))
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    return model, base_model


def plot_history(history_head, history_fine, out_path: pathlib.Path):
    acc = history_head.history["accuracy"] + history_fine.history["accuracy"]
    val_acc = history_head.history["val_accuracy"] + history_fine.history["val_accuracy"]
    loss = history_head.history["loss"] + history_fine.history["loss"]
    val_loss = history_head.history["val_loss"] + history_fine.history["val_loss"]

    switch_epoch = len(history_head.history["accuracy"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(acc, label="Train accuracy")
    axes[0].plot(val_acc, label="Validation accuracy")
    axes[0].axvline(switch_epoch - 0.5, color="gray", linestyle="--", label="Fine-tuning starts")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(loss, label="Train loss")
    axes[1].plot(val_loss, label="Validation loss")
    axes[1].axvline(switch_epoch - 0.5, color="gray", linestyle="--", label="Fine-tuning starts")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"גרף האימון נשמר ב-{out_path}")


def main():
    mu.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("טוען דאטהסטים...")
    train_ds, val_ds, class_names = build_datasets()
    print(f"קלאסים: {class_names}")
    mu.save_class_names(class_names)

    print("\nבונה מודל (MobileNetV2 + Transfer Learning)...")
    model, base_model = build_model(num_classes=len(class_names))
    model.summary()

    # --- שלב 1: אימון השכבות העליונות בלבד ---
    print(f"\n=== שלב 1/2: אימון השכבות העליונות ({EPOCHS_HEAD} epochs, בסיס קפוא) ===")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_HEAD),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    history_head = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_HEAD)

    # --- שלב 2: Fine-tuning - מפשירים חלק מהשכבות העליונות בבסיס ---
    print(f"\n=== שלב 2/2: Fine-tuning ({EPOCHS_FINE_TUNE} epochs, קצב למידה נמוך) ===")
    base_model.trainable = True
    for layer in base_model.layers[:FINE_TUNE_AT_LAYER]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_FINE_TUNE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    history_fine = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_HEAD + EPOCHS_FINE_TUNE,
        initial_epoch=history_head.epoch[-1] + 1,
    )

    print(f"\nשומר את המודל ב-{mu.MODEL_PATH}")
    model.save(mu.MODEL_PATH)

    plot_history(history_head, history_fine, mu.HISTORY_PLOT_PATH)

    final_val_acc = history_fine.history["val_accuracy"][-1]
    print(f"\nדיוק סופי על סט הוולידציה: {final_val_acc:.2%}")
    print("אימון הושלם בהצלחה! עכשיו אפשר להריץ python src/evaluate.py")


if __name__ == "__main__":
    main()
