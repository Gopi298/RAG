import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Accident Detection AI",
    page_icon="🚗",
    layout="centered"
)

# =========================================================
# TITLE
# =========================================================

st.title("🚗 Accident Detection AI")

st.write(
    "Upload a vehicle/road image and the AI model will "
    "predict whether an accident is detected."
)

# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        "accident_model.keras"
    )

    with open("class_names.json", "r") as f:
        class_names = json.load(f)

    return model, class_names


model, class_names = load_model()

# =========================================================
# IMAGE UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "Upload Accident Image",
    type=["jpg", "jpeg", "png"]
)

# =========================================================
# PREDICTION
# =========================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # Resize
    img = image.resize((224, 224))

    # Convert to array
    img_array = np.array(img)

    # Add batch dimension
    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    # MobileNetV2 preprocessing
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(
        img_array.astype(np.float32)
    )

    # Prediction
    prediction = model.predict(
        img_array,
        verbose=0
    )[0][0]

    # =====================================================
    # IMPORTANT
    # =====================================================

    # Find which class has index 1
    positive_class = class_names[1]

    if positive_class.lower() == "accident":

        accident_probability = prediction

    else:

        accident_probability = 1 - prediction

    non_accident_probability = 1 - accident_probability

    # =====================================================
    # RESULT
    # =====================================================

    st.subheader("Prediction Result")

    if accident_probability >= 0.60:

        st.error("🚨 ACCIDENT DETECTED")

        st.metric(
            "Accident Probability",
            f"{accident_probability * 100:.2f}%"
        )

    else:

        st.success("✅ NO ACCIDENT DETECTED")

        st.metric(
            "Non-Accident Probability",
            f"{non_accident_probability * 100:.2f}%"
        )

    # =====================================================
    # PROBABILITY BAR
    # =====================================================

    st.write("Accident Probability")

    st.progress(
        float(accident_probability)
    )

    st.write(
        f"Accident: {accident_probability * 100:.2f}%"
    )

    st.write(
        f"Non-Accident: {non_accident_probability * 100:.2f}%"
    )

    # =====================================================
    # WARNING
    # =====================================================

    st.warning(
        "This is an AI image-classification model. "
        "It should not be used as the sole basis for "
        "real-world emergency decisions."
    )
