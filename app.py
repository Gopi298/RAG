import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from PIL import Image
from ultralytics import YOLO

# Page Configuration
st.set_page_config(
    page_title="Car Accident Detection",
    page_icon="🚗",
    layout="centered"
)

st.title("🚗 Car Accident Detection System")
st.write("Upload an image or video to analyze traffic conditions and detect accidents using YOLOv8.")

# Sidebar Configuration
st.sidebar.header("Model Configuration")
model_file = st.sidebar.text_input("Weights File / Model Name", value="yolov8n.pt")
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.05,
    max_value=1.0,
    value=0.25,
    step=0.05
)

# Cached Model Loader
@st.cache_resource
def load_yolo_model(model_path):
    return YOLO(model_path)

try:
    model = load_yolo_model(model_file)
except Exception as e:
    st.error(f"Failed to load model from '{model_file}': {e}")
    st.stop()

# Input Mode Tabs
tab_image, tab_video = st.tabs(["🖼️ Image Detection", "🎥 Video Detection"])

# --- IMAGE DETECTION ---
with tab_image:
    uploaded_image = st.file_uploader("Choose an image file...", type=["jpg", "jpeg", "png"])
    
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        if st.button("Run Image Detection"):
            with st.spinner("Analyzing image..."):
                # Convert PIL Image to OpenCV BGR
                img_array = np.array(image.convert("RGB"))
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

                # Predict
                results = model.predict(source=img_bgr, conf=confidence_threshold)
                
                # Render results
                res_plotted = results[0].plot()
                res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)

            st.subheader("Detection Results")
            st.image(res_rgb, caption="Annotated Image", use_container_width=True)

            # Detection Summary
            boxes = results[0].boxes
            if len(boxes) > 0:
                st.subheader("Detected Objects")
                class_names = model.names
                counts = {}
                for box in boxes:
                    cid = int(box.cls[0])
                    cname = class_names[cid]
                    counts[cname] = counts.get(cname, 0) + 1
                
                for obj_name, count in counts.items():
                    st.write(f"- **{obj_name.capitalize()}**: {count}")
            else:
                st.info("No objects detected at the current confidence threshold.")

# --- VIDEO DETECTION ---
with tab_video:
    uploaded_video = st.file_uploader("Choose a video file...", type=["mp4", "avi", "mov", "mkv"])
    
    if uploaded_video is not None:
        # Save uploaded file to a temporary file for OpenCV capture
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_video.read())
        tfile.close()
        
        st.video(tfile.name)
        
        if st.button("Process Video"):
            st_frame = st.empty()
            cap = cv2.VideoCapture(tfile.name)
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                results = model.predict(source=frame, conf=confidence_threshold, verbose=False)
                res_plotted = results[0].plot()
                res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                
                # Render stream
                st_frame.image(res_rgb, caption="Processing Video Stream...", use_container_width=True)
            
            cap.release()
            os.remove(tfile.name)
            st.success("Video processing complete!")
