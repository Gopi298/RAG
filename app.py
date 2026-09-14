```python
import os
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# ============================================================
# STREAMLIT PAGE
# ============================================================

st.set_page_config(
    page_title="AI Accident Detection",
    page_icon="🚨",
    layout="wide"
)

# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "models/accident_detection_final.keras"

IMAGE_SIZE = (224, 224)

# IMPORTANT
# Training class order:
#
# 0 = accident
# 1 = non_accident
#
# Verify this from your Colab:
# print(train_ds.class_names)

ACCIDENT_THRESHOLD = 0.80

# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):

        return None

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    return model


model = load_model()

# ============================================================
# HEADER
# ============================================================

st.title("🚨 AI Vehicle Accident Detection")

st.subheader(
    "Computer Vision + CNN + Streamlit"
)

st.write(
    "Detect vehicle accidents using an AI image "
    "classification model."
)

# ============================================================
# MODEL CHECK
# ============================================================

if model is None:

    st.error(
        "❌ Model not found."
    )

    st.write(
        "Please upload your trained model to:"
    )

    st.code(
        "models/accident_detection_final.keras"
    )

    st.stop()

st.success(
    "✅ Accident Detection Model Loaded"
)

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Detection Settings")

threshold = st.sidebar.slider(
    "Accident Confidence Threshold",
    0.50,
    0.99,
    0.80,
    0.01
)

st.sidebar.write(
    f"Threshold: {threshold * 100:.0f}%"
)

st.sidebar.divider()

st.sidebar.write(
    "CNN Model: EfficientNetB0"
)

st.sidebar.write(
    "Input Size: 224 × 224"
)

st.sidebar.write(
    "Classes: 2"
)

# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_accident(image):

    # Convert image to RGB
    image = image.convert("RGB")

    # Resize
    image = image.resize(
        IMAGE_SIZE
    )

    # Convert to NumPy
    image_array = np.array(
        image
    ).astype(
        np.float32
    )

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # Model prediction
    prediction = model.predict(
        image_array,
        verbose=0
    )

    # --------------------------------------------------------
    # MODEL OUTPUT
    # --------------------------------------------------------
    #
    # With:
    #
    # ['accident', 'non_accident']
    #
    # and sigmoid output:
    #
    # probability = probability of class 1
    #
    # Therefore:
    #
    # non_accident = prediction
    # accident = 1 - prediction
    #

    non_accident_probability = float(
        prediction[0][0]
    )

    accident_probability = (
        1.0 -
        non_accident_probability
    )

    # --------------------------------------------------------
    # FINAL DECISION
    # --------------------------------------------------------

    if accident_probability >= threshold:

        result = "ACCIDENT"

        confidence = accident_probability

    else:

        result = "NON-ACCIDENT"

        confidence = non_accident_probability

    return (
        result,
        confidence,
        accident_probability,
        non_accident_probability
    )


# ============================================================
# TABS
# ============================================================

camera_tab, image_tab = st.tabs(
    [
        "📷 CAMERA",
        "🖼️ IMAGE UPLOAD"
    ]
)

# ============================================================
# CAMERA TAB
# ============================================================

with camera_tab:

    st.header(
        "📷 Camera Accident Detection"
    )

    st.write(
        "Allow camera permission in your browser."
    )

    camera_image = st.camera_input(
        "Take a vehicle picture"
    )

    if camera_image is not None:

        image = Image.open(
            camera_image
        )

        # Display image
        st.image(
            image,
            caption="Camera Image",
            use_container_width=True
        )

        # Predict
        (
            result,
            confidence,
            accident_probability,
            non_accident_probability
        ) = predict_accident(
            image
        )

        st.divider()

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        if result == "ACCIDENT":

            st.error(
                "🚨 ACCIDENT DETECTED"
            )

        else:

            st.success(
                "✅ NO ACCIDENT DETECTED"
            )

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Prediction",
                result
            )

        with col2:

            st.metric(
                "Confidence",
                f"{confidence * 100:.2f}%"
            )

        # ----------------------------------------------------
        # PROBABILITIES
        # ----------------------------------------------------

        st.subheader(
            "Prediction Probability"
        )

        st.write(
            f"🚨 Accident: "
            f"{accident_probability * 100:.2f}%"
        )

        st.progress(
            accident_probability
        )

        st.write(
            f"✅ Non-Accident: "
            f"{non_accident_probability * 100:.2f}%"
        )

        st.progress(
            non_accident_probability
        )

# ============================================================
# IMAGE UPLOAD TAB
# ============================================================

with image_tab:

    st.header(
        "🖼️ Upload Vehicle Image"
    )

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp"
        ]
    )

    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        )

        st.image(
            image,
            caption="Uploaded Vehicle Image",
            use_container_width=True
        )

        if st.button(
            "🔍 DETECT ACCIDENT",
            type="primary",
            use_container_width=True
        ):

            (
                result,
                confidence,
                accident_probability,
                non_accident_probability
            ) = predict_accident(
                image
            )

            st.divider()

            # ------------------------------------------------
            # RESULT
            # ------------------------------------------------

            if result == "ACCIDENT":

                st.error(
                    "🚨 ACCIDENT DETECTED"
                )

            else:

                st.success(
                    "✅ NON-ACCIDENT"
                )

            # ------------------------------------------------
            # METRICS
            # ------------------------------------------------

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Prediction",
                    result
                )

            with col2:

                st.metric(
                    "Confidence",
                    f"{confidence * 100:.2f}%"
                )

            # ------------------------------------------------
            # PROBABILITY
            # ------------------------------------------------

            st.subheader(
                "AI Prediction"
            )

            st.write(
                f"🚨 Accident Probability: "
                f"{accident_probability * 100:.2f}%"
            )

            st.progress(
                accident_probability
            )

            st.write(
                f"✅ Non-Accident Probability: "
                f"{non_accident_probability * 100:.2f}%"
            )

            st.progress(
                non_accident_probability
            )

# ============================================================
# PROJECT INFORMATION
# ============================================================

st.divider()

st.header(
    "🧠 Project Information"
)

col1, col2, col3 = st.columns(3)

with col1:

    st.write(
        "**Model**"
    )

    st.write(
        "EfficientNetB0 CNN"
    )

with col2:

    st.write(
        "**Input**"
    )

    st.write(
        "224 × 224 pixels"
    )

with col3:

    st.write(
        "**Output**"
    )

    st.write(
        "Accident / Non-Accident"
    )

st.info(
    """
    This application is an AI classification demonstration.
    Always verify an accident visually before taking
    emergency action.
    """
)
```
