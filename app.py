import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

st.set_page_config(page_title="Vehicle Accident Detection", page_icon="🚗", layout="centered")

MODEL_PATH = "models/accident_model.keras"
IMG_SIZE = (224, 224)

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("Model file not found. Please train the model first and place accident_model.keras in the models/ folder.")
        st.stop()
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# Class mapping – adjust if your train_gen showed the opposite
# Usually: 0 = accident, 1 = non_accident  (check train.py output)
CLASS_NAMES = {0: "ACCIDENT", 1: "NON-ACCIDENT"}

st.title("🚗 Vehicle Accident Detection AI")
st.markdown("Upload an image of a road scene. The model will classify it as **ACCIDENT** or **NON-ACCIDENT** based on actual collision/impact.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    img = image.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    with st.spinner("Analyzing..."):
        pred_prob = model.predict(img_array, verbose=0)[0][0]
        # Adjust threshold or mapping if needed
        pred_class = 1 if pred_prob > 0.5 else 0
        confidence = pred_prob if pred_class == 1 else 1 - pred_prob

        label = CLASS_NAMES[pred_class]
        color = "red" if label == "ACCIDENT" else "green"

    st.markdown(f"### Prediction: <span style='color:{color}'>{label}</span>", unsafe_allow_html=True)
    st.metric("Confidence", f"{confidence*100:.1f}%")

    st.progress(float(confidence))

    with st.expander("Model details"):
        st.write(f"Raw sigmoid output: {pred_prob:.4f}")
        st.write("Rules followed: only true collision / impact / visible crash damage is labeled Accident.")
