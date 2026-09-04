import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Configure Streamlit UI
st.set_page_config(page_title="CCTV Accident Detection", page_icon="🚨", layout="centered")

st.title("🚨 Live CCTV Accident Detection")
st.write("Real-time browser webcam stream evaluated by standard CNN architecture.")

# Load the trained Keras model
@st.cache_resource
def load_cnn_model():
    return tf.keras.models.load_model("accident_model.h5")

try:
    model = load_cnn_model()
    st.sidebar.success("Model loaded successfully!")
except Exception as e:
    st.sidebar.error("Model file 'accident_model.h5' not found. Train and save the model first.")
    st.stop()

# Set up detection threshold slider
threshold = st.sidebar.slider("Accident Probability Threshold", 0.1, 0.9, 0.5, 0.05)

# WebRTC Video Processor Class
class AccidentDetector(VideoTransformerBase):
    def __init__(self):
        self.model = model
        self.threshold = threshold

    def transform(self, frame):
        # Convert WebRTC frame to OpenCV BGR format
        img = frame.to_ndarray(format="bgr24")

        # Preprocess frame for CNN inference
        resized_img = cv2.resize(img, (180, 180))
        rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        normalized_img = rgb_img / 255.0
        input_tensor = np.expand_dims(normalized_img, axis=0)

        # Run Prediction
        prediction = self.model.predict(input_tensor, verbose=0)[0][0]

        # Draw overlays on video output
        if prediction > self.threshold:
            label = f"ACCIDENT DETECTED ({prediction*100:.1f}%)"
            color = (0, 0, 255)  # Red
            # Banner alert box
            cv2.rectangle(img, (0, 0), (img.shape[1], 50), (0, 0, 255), -1)
            cv2.putText(img, label, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        else:
            label = f"Normal Flow ({prediction*100:.1f}%)"
            color = (0, 255, 0)  # Green
            cv2.putText(img, label, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        return img

# Streamlit WebRTC Streamer
webrtc_streamer(
    key="accident-detection",
    video_transformer_factory=AccidentDetector,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)
