import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

# Page configuration
st.set_page_config(
    page_title="Car Accident Detection",
    page_icon="🚗",
    layout="centered"
)

st.title("🚗 Car Accident & Traffic Detection (YOLOv8)")
st.write("Upload an image to detect vehicles, traffic elements, and potential accidents.")

# Load the YOLO model once and cache it for performance
@st.cache_resource
def load_model():
    # Uses yolov8x.pt (downloads automatically if not present locally)
    return YOLO("yolov8x.pt")

model = load_model()

# Sidebar options
st.sidebar.header("Detection Settings")
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold", 
    min_value=0.05, 
    max_value=1.0, 
    value=0.20, 
    step=0.05
)

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Read image using PIL
    image = Image.open(uploaded_file)
    
    # Display raw image
    st.image(image, caption="Uploaded Image", use_container_width=True)
    st.write("")
    
    # Predict button
    if st.button("Run Detection"):
        with st.spinner("Processing image..."):
            # Convert PIL Image to OpenCV format (BGR)
            img_array = np.array(image.convert("RGB"))
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

            # Run YOLOv8 inference
            results = model.predict(source=img_bgr, conf=confidence_threshold)
            
            # Extract annotated frame
            res_plotted = results[0].plot()
            res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)

        st.subheader("Detection Results")
        st.image(res_rgb, caption="Processed Image", use_container_width=True)

        # Show detected counts summary
        st.subheader("Object Summary")
        boxes = results[0].boxes
        if len(boxes) > 0:
            class_ids = boxes.cls.cpu().numpy().astype(int)
            class_names = model.names
            
            counts = {}
            for cid in class_ids:
                cname = class_names[cid]
                counts[cname] = counts.get(cname, 0) + 1
            
            for item, count in counts.items():
                st.write(f"- **{item.capitalize()}**: {count}")
        else:
            st.info("No objects detected above the selected confidence threshold.")
