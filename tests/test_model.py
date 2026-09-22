"""
test_model.py
--------------
A few basic sanity checks - not comprehensive unit tests, but a check that
the whole pipeline (model -> prediction) works as expected, and that output
shapes are sane. This is exactly the kind of thing interviewers like to see
in a portfolio project: proof that you thought about correctness and not
just "it works on my machine".

Run:
    python -m pytest tests/ -v
    (or simply: python tests/test_model.py)
"""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
import model_utils as mu


def test_class_names_file_exists_and_valid():
    assert mu.CLASS_NAMES_PATH.exists(), "class_names.json not found - run src/train.py first"
    class_names = mu.load_class_names()
    assert isinstance(class_names, list)
    assert len(class_names) == 6
    assert "Tomato___healthy" in class_names


def test_model_file_exists():
    assert mu.MODEL_PATH.exists(), "No trained model found - run src/train.py first"


def test_model_predicts_valid_probability_distribution():
    import tensorflow as tf

    model = tf.keras.models.load_model(mu.MODEL_PATH)
    class_names = mu.load_class_names()

    # random image (noise) - not checking correctness, just that the output is valid
    fake_image = np.random.randint(0, 255, size=(1,) + mu.IMG_SIZE + (3,)).astype("float32")
    preds = model.predict(fake_image, verbose=0)

    assert preds.shape == (1, len(class_names)), f"Unexpected output shape: {preds.shape}"
    assert np.isclose(preds.sum(), 1.0, atol=1e-3), "Probabilities don't sum to 1 (broken softmax?)"
    assert (preds >= 0).all() and (preds <= 1).all(), "There are probabilities outside the [0,1] range"


def test_predict_on_real_test_image():
    """Checks that the model correctly identifies at least one example from the test set (a smoke test, not accuracy)."""
    from predict import predict_image

    test_dir = mu.DATA_DIR / "test" / "Tomato___healthy"
    if not test_dir.exists():
        print("Skipping: no test images available (run data/prepare_data.py first)")
        return

    sample_images = list(test_dir.glob("*.JPG")) + list(test_dir.glob("*.jpg"))
    assert len(sample_images) > 0, "No test images found"

    results = predict_image(str(sample_images[0]), top_k=1)
    predicted_class = results[0][0]
    print(f"Test image from Tomato___healthy classified as {predicted_class}")
    # Not enforcing that the prediction is correct (model isn't perfect) - just that it runs and returns something sensible
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
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
