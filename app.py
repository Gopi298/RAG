import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np

# Set page layout
st.set_page_config(page_title="Vehicle Accident Detection", page_icon="🚗", layout="centered")

# Load pre-trained model once
@st.cache_resource
def load_accident_model():
    model = tf.keras.models.load_model("accident_model.h5")
    return model

model = load_accident_model()

st.title("🚗 Vehicle Accident Detection AI")
st.write("Upload a vehicle or road image to analyze for accident impact or normal traffic conditions.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Analyze Image"):
        with st.spinner("Analyzing image..."):
            # Preprocess image
            img_resized = image.resize((224, 224))
            img_array = img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)
            img_preprocessed = preprocess_input(img_array)
            
            # Predict
            prediction = model.predict(img_preprocessed)[0][0]
            
            # Prediction logic (Threshold: 0.5)
            if prediction >= 0.5:
                confidence = prediction * 100
                st.error(f"🚨 **Prediction: ACCIDENT**")
                st.metric(label="Confidence Score", value=f"{confidence:.2f}%")
            else:
                confidence = (1 - prediction) * 100
                st.success(f"✅ **Prediction: NON-ACCIDENT**")
                st.metric(label="Confidence Score", value=f"{confidence:.2f}%")

st.markdown("---")
st.caption("Note: Detection is based on physical collision and visible damage indicators rather than proximity between vehicles.")
