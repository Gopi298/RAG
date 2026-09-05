import os
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
import tempfile
import urllib.request

# Set page layout
st.set_page_config(
    page_title="Accident Detection System",
    page_icon="🚨",
    layout="wide"
)

# ---------------------------------------------------------
# 1. MODEL DOWNLOADER & LOAD
# ---------------------------------------------------------
MODEL_PATH = "data.pkl"

# REPLACE THIS WITH YOUR GOOGLE DRIVE FILE ID
GDRIVE_FILE_ID = "YOUR_GOOGLE_DRIVE_FILE_ID_HERE"

def download_model_from_gdrive(file_id, destination):
    """Downloads model weights directly from Google Drive."""
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    with st.spinner("Downloading accident detection model weights (~100MB)... Please wait."):
        urllib.request.urlretrieve(url, destination)

@st.cache_resource
def load_accident_model():
    """Downloads weights if missing, builds ResNet-50 architecture, and loads weights."""
    if not os.path.exists(MODEL_PATH):
        download_model_from_gdrive(GDRIVE_FILE_ID, MODEL_PATH)

    # Reconstruct ResNet-50 architecture matching your data.pkl structure
    model = models.resnet50(weights=None)
    
    # 2 output classes: [0: Normal, 1: Accident]
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    
    try:
        # Weights loaded safely on CPU
        state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=False)
        
        # Unroll state_dict if wrapped in a dict key
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
            
        model.load_state_dict(state_dict)
    except Exception as e:
        st.error(f"Failed to load model architecture/weights: {e}")
        st.stop()
        
    model.eval()
    return model

model = load_accident_model()

# ---------------------------------------------------------
# 2. IMAGE PREPROCESSING & INFERENCE
# ---------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

def predict_frame(image):
    """Runs a single PIL image through PyTorch model."""
    img_t = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_t)
        probabilities = torch.softmax(outputs, dim=1)[0]
        confidence, predicted_class = torch.max(probabilities, 0)
        
    labels = {0: "Normal", 1: "Accident Detected"}
    return labels[predicted_class.item()], confidence.item()

# ---------------------------------------------------------
# 3. STREAMLIT INTERFACE
# ---------------------------------------------------------
st.title("🚨 Real-Time Accident Detection")

tab1, tab2 = st.tabs(["🖼️ Image Inference", "🎥 Video Inference"])

# --- TAB 1: IMAGE INFERENCE ---
with tab1:
    uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_image is not None:
        image = Image.open(uploaded_image).convert("RGB")
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Uploaded Input Image", use_column_width=True)
            
        with col2:
            label, score = predict_frame(image)
            st.subheader("Model Result")
            
            if label == "Accident Detected":
                st.error(f"**Alert:** {label}")
            else:
                st.success(f"**Status:** {label}")
                
            st.metric(label="Confidence", value=f"{score * 100:.2f}%")

# --- TAB 2: VIDEO INFERENCE ---
with tab2:
    uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
    
    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        
        cap = cv2.VideoCapture(tfile.name)
        st_frame = st.empty()
        stop_btn = st.button("Stop Stream")
        
        while cap.isOpened() and not stop_btn:
            ret, frame = cap.read()
            if not ret:
                break
                
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_frame)
            
            label, score = predict_frame(pil_img)
            
            # Draw frame bounding text
            color = (0, 0, 255) if label == "Accident Detected" else (0, 255, 0)
            text = f"{label}: {score * 100:.1f}%"
            cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
            
            st_frame.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)
            
        cap.release()
