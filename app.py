import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("models/disaster_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Page Configuration
st.set_page_config(
    page_title="Disaster Severity Prediction",
    page_icon="🌍",
    layout="centered"
)

# Title
st.title("🌍 Disaster Severity Prediction System")

st.write(
    "Predict whether a disaster event is classified as a "
    "Major Disaster or Non-Major Disaster."
)

# User Inputs

severity = st.slider(
    "Severity Level",
    min_value=1,
    max_value=10,
    value=5
)

population = st.number_input(
    "Affected Population",
    min_value=0,
    value=1000
)

economic_loss = st.number_input(
    "Estimated Economic Loss (USD)",
    min_value=0.0,
    value=100000.0,
    step=1000.0
)

damage_index = st.slider(
    "Infrastructure Damage Index",
    min_value=0.0,
    max_value=100.0,
    value=50.0
)

# Prediction

if st.button("Predict"):

    input_data = pd.DataFrame({
        "severity_level": [severity],
        "affected_population": [population],
        "estimated_economic_loss_usd": [economic_loss],
        "infrastructure_damage_index": [damage_index]
    })

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.error("🚨 Major Disaster")
    else:
        st.success("✅ Non-Major Disaster")