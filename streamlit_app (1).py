
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load the trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('accident_detection_model.keras')
    return model

model = load_model()

# Define image size (must match training size)
IMAGE_SIZE = (224, 224)

st.title("🚗 Vehicle Accident Detection")
st.write("Upload an image to detect if it's an accident or non-accident scene.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")

    # Preprocess the image for model prediction
    img_array = np.array(image.resize(IMAGE_SIZE)) / 255.0
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension

    # Make prediction
    prediction = model.predict(img_array)
    confidence = prediction[0][0]

    st.subheader("Prediction:")
    if confidence > 0.5:
        st.error(f"ACCIDENT (Confidence: {confidence:.2f})")
    else:
        st.success(f"NON-ACCIDENT (Confidence: {1 - confidence:.2f})")

st.write("\n--- Notes ---")
st.write("This application uses a deep learning model to classify images as 'Accident' or 'Non-Accident'.")
st.write("The model was trained on a dataset of 400 images.")
