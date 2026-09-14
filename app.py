import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Page Configuration
st.set_page_config(
    page_title="Accident Detection System",
    page_icon="🚨",
    layout="centered"
)

CLASS_NAMES = ["Accident", "Non-Accident"]
MODEL_PATH = "accident_detection_mobilenet.pth"

@st.cache_resource
def load_trained_model():
    """Loads and caches the model to optimize Streamlit Cloud performance."""
    device = torch.device("cpu")
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, 2)
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        return model, True
    except Exception as e:
        return model, False

def transform_image(image):
    """Preprocessing transformation matching training validation set."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# Application UI
st.title("🚨 Real-Time Accident Detection")
st.write("Upload an image (various angles, weather conditions, objects, or pedestrians) to detect potential road accidents.")

model, model_loaded = load_trained_model()

if not model_loaded:
    st.warning("⚠️ Weights file (`accident_detection_mobilenet.pth`) not found. Running with uncalibrated baseline weights. Execute `train.py` locally first.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    with st.spinner("Analyzing image features..."):
        input_tensor = transform_image(image)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            
        accident_prob = probabilities[0].item() * 100
        non_accident_prob = probabilities[1].item() * 100
        predicted_class = CLASS_NAMES[torch.argmax(probabilities).item()]

    st.markdown("---")
    st.subheader("Analysis Result")
    
    if predicted_class == "Accident":
        st.error(f"**Status: ACCIDENT DETECTED** ({accident_prob:.1f}% Confidence)")
    else:
        st.success(f"**Status: NO ACCIDENT DETECTED** ({non_accident_prob:.1f}% Confidence)")

    st.write("**Confidence Breakdown:**")
    st.progress(int(accident_prob), text=f"Accident Risk: {accident_prob:.1f}%")
    st.progress(int(non_accident_prob), text=f"Safe/Normal: {non_accident_prob:.1f}%")
