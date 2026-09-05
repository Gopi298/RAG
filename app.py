import os
import tempfile
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
from huggingface_hub import hf_hub_download

# Page Layout
st.set_page_config(
    page_title="Accident Detection System",
    page_icon="🚨",
    layout="wide"
)

# ---------------------------------------------------------
# 1. MODEL DOWNLOAD & ARCHITECTURE SETUP
# ---------------------------------------------------------
# CHANGE THESE TWO LINES TO YOUR HUGGINGFACE REPO DETAILS
HF_REPO_ID = "YOUR_USERNAME/YOUR_MODEL_REPO"  # e.g., "johndoe/accident-resnet"
HF_FILENAME = "data.pkl"

@st.cache_resource
def load_accident_model():
    """Downloads model weights from Hugging Face Hub and constructs PyTorch ResNet-50."""
    try:
        with st.spinner("Downloading model weights from Hugging Face..."):
            model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME)
    except Exception as e:
        st.error(f"Failed to download model from Hugging Face: {e}")
        st.stop()

    # Reconstruct ResNet-50 architecture matching data.pkl structure
    model = models.resnet50(weights=None)
    
    # Adjust output layer for binary classification: [0: Normal, 1: Accident]
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    
    try:
        # Load weights onto CPU safely
        state_dict = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        
        # Unroll state_dict if wrapped inside a dictionary key
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
            
        model.load_state_dict(state_dict)
    except Exception as e:
        st.error(f"Error loading state dictionary into architecture: {e}")
        st.stop()
        
    model.eval()
    return model

model = load_accident_model()

# ---------------------------------------------------------
# 2. INFERENCE PREPROCESSING
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
    """Executes single frame inference through the PyTorch model."""
    img_t = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_t)
        probabilities = torch.softmax(outputs, dim=1)[0]
        confidence, predicted_class = torch.max(probabilities, 0)
        
    labels = {0: "Normal", 1: "Accident Detected"}
    return labels[predicted_class.item()], confidence.item()

# ---------------------------------------------------------
# 3. STREAMLIT USER INTERFACE
# ---------------------------------------------------------
st.title("🚨 Real-Time Accident Detection App")

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
            st.subheader("Model Result")
            label, score = predict_frame(image)
            
            if label == "Accident Detected":
                st.error(f"**Status:** {label}")
            else:
                st.success(f"**Status:** {label}")
                
            st.metric(label="Confidence Score", value=f"{score * 100:.2f}%")

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
            
            # Annotate video frame with OpenCV
            color = (0, 0, 255) if label == "Accident Detected" else (0, 255, 0)
            text = f"{label}: {score * 100:.1f}%"
            cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
            
            st_frame.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)
            
        cap.release()
