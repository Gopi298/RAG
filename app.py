import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np

st.set_page_config(page_title="Vehicle Accident Detection", page_icon="🚗", layout="centered")

@st.cache_resource
def load_accident_model():
    return tf.keras.models.load_model("accident_model.h5")

try:
    model = load_accident_model()
except Exception as e:
    st.error(f"Failed to load model file 'accident_model.h5'. Ensure it is committed to your repository root. Details: {e}")

st.title("🚗 Vehicle Accident Detection AI")
st.write("Upload a vehicle or road image to evaluate for collisions or normal traffic conditions.")

uploaded_file = st.file_uploader("Choose an image file...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    if st.button("Analyze Image"):
        with st.spinner("Processing image..."):
            img_resized = image.resize((224, 224))
            img_array = img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)
            img_preprocessed = preprocess_input(img_array)
            
            prediction = model.predict(img_preprocessed)[0][0]
            
            if prediction >= 0.5:
                confidence = prediction * 100
                st.error("🚨 **Prediction: ACCIDENT**")
                st.metric(label="Confidence Score", value=f"{confidence:.2f}%")
            else:
                confidence = (1 - prediction) * 100
                st.success("✅ **Prediction: NON-ACCIDENT**")
                st.metric(label="Confidence Score", value=f"{confidence:.2f}%")

st.markdown("---")
st.caption("Detection model focuses on structural impact, deformation, and damage indicators rather than proximity.")
