"""
streamlit_app.py
-----------------
Live demo app: the user uploads a photo of a tomato leaf, and the app shows
the model's prediction (healthy / which disease) with a confidence level for
each class.

Run locally:
    streamlit run app/streamlit_app.py

Free deployment to the web (to have a link to share instead of a CV):
    See the instructions in README.md under "Deployment".
"""

import pathlib
import sys

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
import model_utils as mu

st.set_page_config(
    page_title="Tomato Leaf Disease Detector",
    page_icon="🍅",
    layout="centered",
)


@st.cache_resource
def load_model_and_classes():
    model = tf.keras.models.load_model(mu.MODEL_PATH)
    class_names = mu.load_class_names()
    return model, class_names


def predict(model, class_names, image: Image.Image):
    image = image.convert("RGB").resize(mu.IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array, verbose=0)[0]
    order = preds.argsort()[::-1]
    return [(class_names[i], mu.display_name(class_names[i]), float(preds[i])) for i in order]


def main():
    st.title("🍅 Tomato Leaf Disease Detector")
    st.markdown(
        """
        A sample **Computer Vision** project: a model trained with *Transfer Learning*
        on top of **MobileNetV2**, detecting 6 possible conditions in a tomato leaf - a
        healthy leaf or one of 5 common diseases. Based on the
        [PlantVillage](https://github.com/spMohanty/PlantVillage-Dataset) dataset.
        """
    )

    if not mu.MODEL_PATH.exists():
        st.error(
            "No trained model found. Run this first:\n\n"
            "```\npython src/train.py\n```"
        )
        st.stop()

    model, class_names = load_model_and_classes()

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
            results = predict(model, class_names, image)

        with col2:
            st.subheader("Prediction results")
            top_class, top_display, top_prob = results[0]

            if top_class == "Tomato___healthy":
                st.success(f"**{top_display}** ({top_prob:.1%} confidence)")
            else:
                st.warning(f"**{top_display}** ({top_prob:.1%} confidence)")

            st.caption("All classes, ranked by confidence:")
            for class_name, display, prob in results:
                st.progress(prob, text=f"{display} - {prob:.1%}")

        st.divider()
        st.caption(
            "⚠️ This is a demo project for portfolio purposes. Do not use it for "
            "real agricultural decisions without expert verification."
        )

    with st.sidebar:
        st.header("ℹ️ About the project")
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

        if mu.METRICS_PATH.exists():
            import json
            with open(mu.METRICS_PATH, encoding="utf-8") as f:
                metrics = json.load(f)
            st.metric("Accuracy on test set", f"{metrics['test_accuracy']:.1%}")

        st.markdown("---")
        st.markdown("[Project source code](.) · built as a portfolio project for job hunting")


if __name__ == "__main__":
    main()
