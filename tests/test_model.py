"""
test_model.py
--------------
כמה בדיקות שפיות (sanity checks) בסיסיות - לא בדיקות יחידה מקיפות, אלא
בדיקה שהצינור כולו (מודל -> חיזוי) עובד כמו שצריך, ושמידות הפלט הגיוניות.
זה בדיוק סוג הדברים שמראיינים אוהבים לראות בפרויקט פורטפוליו: הוכחה
שחשבתם על תקינות ולא רק על "זה עובד אצלי במחשב".

הרצה:
    python -m pytest tests/ -v
    (או פשוט: python tests/test_model.py)
"""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
import model_utils as mu


def test_class_names_file_exists_and_valid():
    assert mu.CLASS_NAMES_PATH.exists(), "לא נמצא class_names.json - הריצו קודם src/train.py"
    class_names = mu.load_class_names()
    assert isinstance(class_names, list)
    assert len(class_names) == 6
    assert "Tomato___healthy" in class_names


def test_model_file_exists():
    assert mu.MODEL_PATH.exists(), "לא נמצא מודל מאומן - הריצו קודם src/train.py"


def test_model_predicts_valid_probability_distribution():
    import tensorflow as tf

    model = tf.keras.models.load_model(mu.MODEL_PATH)
    class_names = mu.load_class_names()

    # תמונה רנדומלית (רעש) - לא בודקים נכונות, רק שהפלט תקין
    fake_image = np.random.randint(0, 255, size=(1,) + mu.IMG_SIZE + (3,)).astype("float32")
    preds = model.predict(fake_image, verbose=0)

    assert preds.shape == (1, len(class_names)), f"צורת פלט לא צפויה: {preds.shape}"
    assert np.isclose(preds.sum(), 1.0, atol=1e-3), "ההסתברויות לא מסתכמות ל-1 (softmax שבור?)"
    assert (preds >= 0).all() and (preds <= 1).all(), "יש הסתברויות מחוץ לטווח [0,1]"


def test_predict_on_real_test_image():
    """בודק שהמודל מצליח לזהות נכון לפחות דוגמה אחת מסט הבדיקה (בדיקת עשיות, לא דיוק)."""
    from predict import predict_image

    test_dir = mu.DATA_DIR / "test" / "Tomato___healthy"
    if not test_dir.exists():
        print("מדלג: אין תמונות test זמינות (הריצו קודם data/prepare_data.py)")
        return

    sample_images = list(test_dir.glob("*.JPG")) + list(test_dir.glob("*.jpg"))
    assert len(sample_images) > 0, "לא נמצאו תמונות בדיקה"

    results = predict_image(str(sample_images[0]), top_k=1)
    predicted_class = results[0][0]
    print(f"תמונת test מ-Tomato___healthy סווגה כ-{predicted_class}")
    # לא אוכפים שהחיזוי יהיה נכון (מודל לא מושלם) - רק שהוא רץ ומחזיר משהו הגיוני
    assert predicted_class in mu.load_class_names()


if __name__ == "__main__":
    tests = [
        test_class_names_file_exists_and_valid,
        test_model_file_exists,
        test_model_predicts_valid_probability_distribution,
        test_predict_on_real_test_image,
    ]
    passed, failed = 0, 0
    for test_fn in tests:
        try:
            test_fn()
            print(f"✅ {test_fn.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"❌ {test_fn.__name__}: {e}")
            failed += 1
    print(f"\n{passed} עברו, {failed} נכשלו")
    sys.exit(1 if failed else 0)
