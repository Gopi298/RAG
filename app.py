import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go

# --- Page Configuration ---
st.set_page_config(
    page_title="Diabetes Risk Classifier AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling ---
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f8fafc;
    }
    .stMetric {
        background: #1e293b;
        border: 1px solid #334155;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #6366f1 0%, #a855f7 100%);
        color: white;
        font-weight: bold;
        border-radius: 8px;
        border: none;
        padding: 12px;
        font-size: 16px;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #4f46e5 0%, #9333ea 100%);
    }
    </style>
""", unsafe_allow_html=True)

# --- Load Model & Scaler ---
@st.cache_resource
def load_artifacts():
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except FileNotFoundError:
        return None, None

model, scaler = load_artifacts()

# --- Sidebar Inputs ---
st.sidebar.header("🩺 Patient Clinical Metrics")

pregnancies = st.sidebar.number_input("Pregnancies", min_value=0, max_value=20, value=1)
glucose = st.sidebar.slider("Glucose Level (mg/dL)", 40, 200, 120)
blood_pressure = st.sidebar.slider("Blood Pressure (mm Hg)", 40, 140, 70)
skin_thickness = st.sidebar.slider("Skin Thickness (mm)", 7, 99, 20)
insulin = st.sidebar.slider("Insulin Level (mu U/ml)", 14, 846, 79)
bmi = st.sidebar.slider("BMI (Body Mass Index)", 15.0, 60.0, 25.0)
dpf = st.sidebar.slider("Diabetes Pedigree Function", 0.07, 2.5, 0.375)
age = st.sidebar.slider("Age (Years)", 21, 90, 33)

# Feature Calculation
bmi_cat = 0 if bmi < 18.5 else (1 if bmi <= 24.9 else (2 if bmi <= 29.9 else 3))
gi_ratio = glucose / (insulin + 1)
age_bmi = age * bmi

input_data = pd.DataFrame([[
    pregnancies, glucose, blood_pressure, skin_thickness, insulin, 
    bmi, dpf, age, bmi_cat, gi_ratio, age_bmi
]], columns=[
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
    'BMI', 'DiabetesPedigreeFunction', 'Age', 'BMI_Category',
    'Glucose_Insulin_Ratio', 'Age_BMI_Product'
])

# --- Main Dashboard ---
st.title("🩺 Diabetes Risk Assessment Platform")
st.markdown("Interactive machine learning diagnostic tool for diabetes probability analysis.")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📋 Patient Summary")
    st.dataframe(pd.DataFrame({
        "Metric": ["Glucose", "BMI", "Age", "Insulin", "Blood Pressure"],
        "Value": [f"{glucose} mg/dL", f"{bmi}", f"{age} yrs", f"{insulin} mu U/ml", f"{blood_pressure} mm Hg"]
    }), use_container_width=True)

with col2:
    st.subheader("⚡ Prediction Engine")
    if model is not None and scaler is not None:
        scaled_input = scaler.transform(input_data)
        prediction = model.predict(scaled_input)[0]
        prob = model.predict_proba(scaled_input)[0][1]

        st.metric(label="Diabetes Risk Probability", value=f"{prob * 100:.1f}%")

        if prediction == 1:
            st.error("⚠️ **High Risk of Diabetes Detected**")
        else:
            st.success("✅ **Low Risk of Diabetes Detected**")
            
        # Gauge Chart
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = prob * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Risk Score %"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "#ef4444" if prob >= 0.5 else "#22c55e"},
                'steps': [
                    {'range': [0, 35], 'color': "#15803d"},
                    {'range': [35, 65], 'color': "#eab308"},
                    {'range': [65, 100], 'color': "#b91c1c"}
                ]
            }
        ))
        fig.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Please run `train_and_save.py` first to generate `model.pkl` and `scaler.pkl`.")
