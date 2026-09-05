import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
import tempfile
import os

# Set page layout
st.set_page_config(
    page_title="Accident Detection System",
    page_icon="🚨",
    layout="wide"
)

# ---------------------------------------------------------
# 1. MODEL DEFINITION & LOADING
# ---------------------------------------------------------
@st.cache_resource
def load_accident_model(weights_path):
    """
    Reconstructs the ResNet architecture and loads the saved weights.
    """
    # Initialize a ResNet model (ResNet-50 based on the deeper layer structure in the pickle file)
    # If your original model was ResNet18/34, change models.resnet50 to models.resnet18/resnet34
    model = models.resnet50(weights=None)
    
    # Adjust final classification layer for binary output (0: Normal, 1: Accident)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    
    # Load weights from data.pkl
    try:
        state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
        
        # Handle cases where state_dict is wrapped in another object/dict key
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
            
        model.load_state_dict(state_dict)
    except Exception as e:
        st.error(f"Error loading model weights: {e}")
        st.stop()
        
    model.eval()
    return model

# Load model (Path based on your file tree)
MODEL_PATH = "accident_detection_model/data.pkl"

if not os.path.exists(MODEL_PATH):
    st.error(f"Model weight file not found at path: {MODEL_PATH}")
    st.stop()

model = load_accident_model(MODEL_PATH)

# ---------------------------------------------------------
# 2. INPUT PREPROCESSING
# ---------------------------------------------------------
# Standard ImageNet transformations commonly used for ResNet inference
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

def predict_frame(image):
    """Passes an image frame through the model and returns class & confidence."""
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
st.write("Upload an image or a video feed to run the PyTorch Accident Detection model.")

tab1, tab2 = st.tabs(["🖼️ Image Inference", "🎥 Video Inference"])

# --- TAB 1: IMAGE PROCESSING ---
with tab1:
    uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_image is not None:
        image = Image.open(uploaded_image).convert("RGB")
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Uploaded Input", use_column_width=True)
            
        with col2:
            st.subheader("Model Prediction")
            label, score = predict_frame(image)
            
            if label == "Accident Detected":
                st.error(f"**Status:** {label}")
            else:
                st.success(f"**Status:** {label}")
                
            st.metric(label="Confidence Score", value=f"{score * 100:.2f}%")

# --- TAB 2: VIDEO PROCESSING ---
with tab2:
    uploaded_video = st.file_uploader("Upload Video File", type=["mp4", "avi", "mov"])
    
    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        
        cap = cv2.VideoCapture(tfile.name)
        st_frame = st.empty()
        
        stop_btn = st.button("Stop Processing")
        
        while cap.isOpened() and not stop_btn:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR (OpenCV format) to RGB (PIL format)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_frame)
            
            # Perform inference
            label, score = predict_frame(pil_img)
            
            # Annotate frame with OpenCV
            color = (0, 0, 255) if label == "Accident Detected" else (0, 255, 0)
            text = f"{label}: {score * 100:.1f}%"
            cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
            
            # Render frame back to Streamlit
            st_frame.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)
            
        cap.release()
