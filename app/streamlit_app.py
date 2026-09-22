"""
streamlit_app.py
-----------------
אפליקציית דמו חיה: המשתמש מעלה תמונה של עלה עגבנייה, והאפליקציה מציגה
את התחזית של המודל (בריא / איזו מחלה) עם רמת ביטחון לכל קלאס.

הרצה מקומית:
    streamlit run app/streamlit_app.py

פריסה חינמית לאינטרנט (כדי שיהיה קישור לשתף במקום CV):
    ראו את ההוראות ב-README.md תחת "פריסה (Deployment)".
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
    page_title="מזהה מחלות בעלי עגבנייה",
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
    st.title("🍅 מזהה מחלות בעלי עגבנייה")
    st.markdown(
        """
        פרויקט **Computer Vision** לדוגמה: מודל שאומן עם *Transfer Learning*
        על גבי **MobileNetV2**, מזהה 6 מצבים אפשריים בעלה עגבנייה - עלה בריא
        או אחת מ-5 מחלות נפוצות. מבוסס על דאטהסט
        [PlantVillage](https://github.com/spMohanty/PlantVillage-Dataset).
        """
    )

    if not mu.MODEL_PATH.exists():
        st.error(
            "לא נמצא מודל מאומן. הריצו קודם:\n\n"
            "```\npython src/train.py\n```"
        )
        st.stop()

    model, class_names = load_model_and_classes()

    st.divider()

    tab_upload, tab_camera = st.tabs(["📁 העלאת תמונה", "📷 מצלמה"])
    image = None

    with tab_upload:
        uploaded_file = st.file_uploader(
            "העלו תמונה של עלה עגבנייה (JPG/PNG)", type=["jpg", "jpeg", "png"]
        )
        if uploaded_file is not None:
            image = Image.open(uploaded_file)

    with tab_camera:
        camera_file = st.camera_input("צלמו עלה עגבנייה")
        if camera_file is not None:
            image = Image.open(camera_file)

    if image is not None:
        col1, col2 = st.columns([1, 1.2])

        with col1:
            st.image(image, caption="התמונה שהועלתה", use_container_width=True)

        with st.spinner("מריץ את המודל..."):
            results = predict(model, class_names, image)

        with col2:
            st.subheader("תוצאות החיזוי")
            top_class, top_display, top_prob = results[0]

            if top_class == "Tomato___healthy":
                st.success(f"**{top_display}** ({top_prob:.1%} ביטחון)")
            else:
                st.warning(f"**{top_display}** ({top_prob:.1%} ביטחון)")

            st.caption("כל הקלאסים, לפי סדר ביטחון:")
            for class_name, display, prob in results:
                st.progress(prob, text=f"{display} - {prob:.1%}")

        st.divider()
        st.caption(
            "⚠️ זהו פרויקט הדגמה לצרכי פורטפוליו. אין להשתמש בו לקבלת "
            "החלטות אמיתיות בגידול חקלאי בלי אימות של מומחה."
        )

    with st.sidebar:
        st.header("ℹ️ על הפרויקט")
        st.markdown(
            """
            **סוג המשימה:** סיווג תמונות (Image Classification)

            **ארכיטקטורה:** MobileNetV2 + Transfer Learning

            **דאטהסט:** PlantVillage (תמונות אמיתיות של עלי עגבנייה)

            **קלאסים:**
            """
        )
        for c in class_names:
            st.markdown(f"- {mu.display_name(c)}")

        if mu.METRICS_PATH.exists():
            import json
            with open(mu.METRICS_PATH, encoding="utf-8") as f:
                metrics = json.load(f)
            st.metric("דיוק על סט הבדיקה", f"{metrics['test_accuracy']:.1%}")

        st.markdown("---")
        st.markdown("[קוד המקור בפרויקט](.) · נבנה כפרויקט פורטפוליו לחיפוש עבודה")


if __name__ == "__main__":
    main()
