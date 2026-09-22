"""
predict.py
----------
חיזוי על תמונה בודדת מה-command line. שימושי לבדיקה מהירה בלי לפתוח את
אפליקציית ה-Streamlit.

הרצה:
    python src/predict.py path/to/image.jpg
"""

import pathlib
import sys

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import model_utils as mu


def predict_image(image_path: str, top_k: int = 3):
    model = tf.keras.models.load_model(mu.MODEL_PATH)
    class_names = mu.load_class_names()

    img = tf.keras.utils.load_img(image_path, target_size=mu.IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array, verbose=0)[0]
    top_indices = predictions.argsort()[::-1][:top_k]

    results = [
        (class_names[i], mu.display_name(class_names[i]), float(predictions[i]))
        for i in top_indices
    ]
    return results


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("שימוש: python src/predict.py path/to/image.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    results = predict_image(image_path)

    print(f"\nתחזיות עבור {image_path}:\n")
    for class_name, display, prob in results:
        bar = "█" * int(prob * 40)
        print(f"  {display:35s} {prob:6.1%}  {bar}")
