import os
import streamlit as st
import gdown
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
import tempfile

MODEL_PATH = "data.pkl"

# Paste your file ID here (the long string from your Google Drive link)
# Example link: https://drive.google.com/file/d/1A2B3C4D5E6F7G8H9I0J/view?usp=sharing
# File ID is: 1A2B3C4D5E6F7G8H9I0J
GDRIVE_FILE_ID = "YOUR_GOOGLE_DRIVE_FILE_ID_HERE"

@st.cache_resource
def load_accident_model():
    """Downloads weights if missing, builds ResNet-50 architecture, and loads weights."""
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model weights from Google Drive (~100MB)... Please wait."):
            url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
            gdown.download(url, MODEL_PATH, quiet=False)

    # Reconstruct ResNet-50 architecture
    model = models.resnet50(weights=None)
    
    # 2 output classes: [0: Normal, 1: Accident]
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    
    try:
        state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=False)
        
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
            
        model.load_state_dict(state_dict)
    except Exception as e:
        st.error(f"Failed to load model architecture/weights: {e}")
        st.stop()
        
    model.eval()
    return model

model = load_accident_model()
