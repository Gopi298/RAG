import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go

# --- Page Configuration ---
st.set_page_config(
    page_title="Diabetes AI Diagnostic System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling ---
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f8fafc;
    }
    .stMetric {
        background-color: #1e293b;
        border: 1px solid #334155;
        padding: 15px;
        border-radius: 10px;
    }
    .stButton>button {
        background: linear-gradient(90deg, #6366f1 0%, #a855f7 100%);
        color: white;
        font-weight: bold;
        border-radius: 8px;
        border: none;
        padding: 12px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Artifact Loader ---
@st.cache_resource
def load_artifacts():
    try:
        with open('model.pkl', 'rb') as f_model:
            model = pickle.load(f_model)
        with open('scaler.pkl', 'rb') as f_scaler:
            scaler = pickle.load(f_scaler)
        return model, scaler
    except Exception as e:
        st.error(f"Error loading artifacts: {e}")
        return None, None

model, scaler = load_artifacts()

# --- Sidebar Input Controls ---
st.sidebar.header("🩺 Patient Parameters")

pregnancies = st.sidebar.number_input("Pregnancies", min_value=0, max_value=20, value=1)
glucose = st.sidebar.slider("Glucose (mg/dL)", 40, 200, 120)
blood_pressure = st.sidebar.slider("Blood Pressure (mm Hg)", 40, 140, 70)
skin_thickness = st.sidebar.slider("Skin Thickness (mm)", 7, 99, 20)
insulin = st.sidebar.slider("Insulin Level (mu U/ml)", 14, 846, 79)
bmi = st.sidebar.slider("BMI", 15.0, 60.0, 25.0)
dpf = st.sidebar.slider("Diabetes Pedigree Function", 0.07, 2.50, 0.375, step=0.01)
age = st.sidebar.slider("Age (Years)", 21, 90, 33)

# Feature Calculations
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

# --- Main Dashboard Layout ---
st.title("🩺 Diabetes Risk Prediction Platform")
st.markdown("Clinical Machine Learning Inference Interface")

col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("📋 Input Metrics Summary")
    st.table(pd.DataFrame({
        "Metric": ["Glucose", "BMI", "Age", "Insulin", "Blood Pressure", "Pregnancies", "Pedigree Function"],
        "Value": [f"{glucose} mg/dL", f"{bmi}", f"{age} yrs", f"{insulin} mu U/ml", f"{blood_pressure} mm Hg", str(pregnancies), str(dpf)]
    }))

with col_right:
    st.subheader("⚡ Diagnostic Result")
    if model is not None and scaler is not None:
        scaled_features = scaler.transform(input_data)
        prediction = model.predict(scaled_features)[0]
        risk_proba = model.predict_proba(scaled_features)[0][1]

        st.metric(label="Calculated Diabetes Risk", value=f"{risk_proba * 100:.1f}%")

        if prediction == 1:
            st.error("⚠️ **High Probability of Diabetes**")
        else:
            st.success("✅ **Low Probability of Diabetes**")

        # Gauge Visualization
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_proba * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Risk Score Percentage"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#ef4444" if risk_proba >= 0.5 else "#22c55e"},
                'steps': [
                    {'range': [0, 35], 'color': "#15803d"},
                    {'range': [35, 65], 'color': "#eab308"},
                    {'range': [65, 100], 'color': "#b91c1c"}
                ]
            }
        ))
        fig.update_layout(height=260, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Model artifacts not found. Please run `python train_model.py` first.")
