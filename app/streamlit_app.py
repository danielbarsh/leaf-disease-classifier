"""
streamlit_app.py
-----------------
Live demo app: the user uploads a photo of a tomato leaf, and the app shows
the model's prediction (healthy / which disease) with a confidence level for
each class. Lets you pick between the two trained models:

- Keras (MobileNetV2) - the original model, always returns a class.
- PyTorch (EfficientNet-B0, calibrated) - src/torch_pipeline, returns a
  calibrated confidence and can reject low-confidence/ambiguous predictions
  instead of guessing.

Each backend's heavy dependency (tensorflow / torch) is imported lazily,
inside its own load/predict functions, so the app only needs whichever
backend you actually select to be installed.

Run locally:
    streamlit run app/streamlit_app.py

Free deployment to the web (to have a link to share instead of a CV):
    See the instructions in README.md under "Deployment".
"""

import json
import pathlib
import sys

import streamlit as st
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
import model_utils as mu

TORCH_MODEL_PATH = mu.MODELS_DIR / "leaf_disease_model_torch.pt"
TORCH_METRICS_PATH = mu.MODELS_DIR / "metrics_torch.json"

st.set_page_config(
    page_title="Tomato Leaf Disease Detector",
    page_icon="🍅",
    layout="centered",
)


# --- Keras (MobileNetV2) backend ---

@st.cache_resource
def load_keras_model():
    import tensorflow as tf
    model = tf.keras.models.load_model(mu.MODEL_PATH)
    class_names = mu.load_class_names()
    return model, class_names


def predict_keras(model, class_names, image: Image.Image):
    import numpy as np
    import tensorflow as tf
    image = image.convert("RGB").resize(mu.IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array, verbose=0)[0]
    order = preds.argsort()[::-1]
    return [(class_names[i], mu.display_name(class_names[i]), float(preds[i])) for i in order]


# --- PyTorch (EfficientNet-B0, calibrated) backend ---

@st.cache_resource
def load_torch_model():
    import torch
    from torch_pipeline import LeafDiseaseClassifier, ModelWithTemperature

    checkpoint = torch.load(TORCH_MODEL_PATH, map_location="cpu", weights_only=False)
    model = LeafDiseaseClassifier(
        num_classes=len(checkpoint["class_names"]),
        backbone=checkpoint["backbone"],
        pretrained=False,  # weights come from the checkpoint below
        dropout=checkpoint["dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    calibrated_model = ModelWithTemperature(model)
    calibrated_model.temperature.data.fill_(checkpoint["temperature"])

    return model, calibrated_model, checkpoint["class_names"], checkpoint["img_size"]


def predict_torch(model, calibrated_model, class_names, img_size, image: Image.Image):
    from torch_pipeline import infer_with_rejection
    return infer_with_rejection(
        image=image.convert("RGB"),
        model=model,
        temperature_scaler=calibrated_model,
        class_names=class_names,
        device="cpu",
        img_size=img_size,
    )


def main():
    st.title("🍅 Tomato Leaf Disease Detector")
    st.markdown(
        """
        A sample **Computer Vision** project: detects 6 possible conditions in a
        tomato leaf - a healthy leaf or one of 5 common diseases. Based on the
        [PlantVillage](https://github.com/spMohanty/PlantVillage-Dataset) dataset.
        """
    )

    backend = st.radio(
        "Model",
        ["Keras (MobileNetV2)", "PyTorch (EfficientNet-B0, calibrated)"],
        horizontal=True,
    )
    is_torch = backend.startswith("PyTorch")

    if is_torch and not TORCH_MODEL_PATH.exists():
        st.error(
            "No trained PyTorch model found. Run this first:\n\n"
            "```\npython src/torch_pipeline/train.py\n```"
        )
        st.stop()
    if not is_torch and not mu.MODEL_PATH.exists():
        st.error(
            "No trained Keras model found. Run this first:\n\n"
            "```\npython src/train.py\n```"
        )
        st.stop()

    if is_torch:
        model, calibrated_model, class_names, img_size = load_torch_model()
    else:
        model, class_names = load_keras_model()

    st.divider()

    tab_upload, tab_camera = st.tabs(["📁 Upload image", "📷 Camera"])
    image = None

    with tab_upload:
        uploaded_file = st.file_uploader(
            "Upload a photo of a tomato leaf (JPG/PNG)", type=["jpg", "jpeg", "png"]
        )
        if uploaded_file is not None:
            image = Image.open(uploaded_file)

    with tab_camera:
        camera_file = st.camera_input("Take a photo of a tomato leaf")
        if camera_file is not None:
            image = Image.open(camera_file)

    if image is not None:
        col1, col2 = st.columns([1, 1.2])

        with col1:
            st.image(image, caption="Uploaded image", use_container_width=True)

        with st.spinner("Running the model..."):
            if is_torch:
                result = predict_torch(model, calibrated_model, class_names, img_size, image)
                ranked = sorted(result["probabilities"].items(), key=lambda kv: -kv[1])
            else:
                ranked_raw = predict_keras(model, class_names, image)
                ranked = [(name, prob) for name, _display, prob in ranked_raw]

        with col2:
            st.subheader("Prediction results")

            if is_torch and result["status"] != "OK":
                st.error(f"**{result['status']}**")
                st.caption(
                    f"Top confidence {result['confidence']:.1%}, "
                    f"margin {result['margin']:.1%} - below the thresholds for a "
                    "reliable prediction. Try a clearer, closer photo of the leaf."
                )
            else:
                top_class, top_prob = ranked[0]
                top_display = mu.display_name(top_class)
                if top_class == "Tomato___healthy":
                    st.success(f"**{top_display}** ({top_prob:.1%} confidence)")
                else:
                    st.warning(f"**{top_display}** ({top_prob:.1%} confidence)")

            st.caption("All classes, ranked by confidence:")
            for class_name, prob in ranked:
                st.progress(prob, text=f"{mu.display_name(class_name)} - {prob:.1%}")

        st.divider()
        st.caption(
            "⚠️ This is a demo project for portfolio purposes. Do not use it for "
            "real agricultural decisions without expert verification."
        )

    with st.sidebar:
        st.header("ℹ️ About the project")
        if is_torch:
            st.markdown(
                """
                **Task type:** Image Classification (calibrated, rejection-aware)

                **Architecture:** EfficientNet-B0 + Transfer Learning,
                Focal Loss, Temperature Scaling

                **Dataset:** PlantVillage (real photos of tomato leaves)

                **Classes:**
                """
            )
        else:
            st.markdown(
                """
                **Task type:** Image Classification

                **Architecture:** MobileNetV2 + Transfer Learning

                **Dataset:** PlantVillage (real photos of tomato leaves)

                **Classes:**
                """
            )
        for c in class_names:
            st.markdown(f"- {mu.display_name(c)}")

        metrics_path = TORCH_METRICS_PATH if is_torch else mu.METRICS_PATH
        if metrics_path.exists():
            with open(metrics_path, encoding="utf-8") as f:
                metrics = json.load(f)
            st.metric("Accuracy on test set", f"{metrics['test_accuracy']:.1%}")

        st.markdown("---")
        st.markdown("[Project source code](.) · built as a portfolio project for job hunting")


if __name__ == "__main__":
    main()
