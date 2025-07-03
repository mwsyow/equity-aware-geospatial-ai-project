import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import sys
import os
import streamlit.components.v1 as components
import time

# Add the src directory to the path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'index_travel_accessibility')))
from travel_time_and_centroid import get_hospital_df

    
st.set_page_config(page_title="Equity-Aware Geospatial AI Dashboard", layout="wide")

st.title("Equity-Aware Geospatial AI: Scenario Simulation Dashboard")

# Inject custom CSS for pointer cursor on selectbox
st.markdown("""
    <style>
    /* Make the entire selectbox container show pointer on hover */
    .stSelectbox:hover, .stSelectbox:hover * {
        cursor: pointer !important;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
    /* Disable Streamlit's default widget transition/fade effect */
    .element-container {
        transition: none !important;
        animation: none !important;
    }
    </style>
""", unsafe_allow_html=True)

# Dropdown for model selection
model_files = {
    "Main Model": "main_model_run.html",
    "Policy Maker Model": "policy_maker_model.html",
    "Deprivation Aware Model": "depravation_aware.html",
    "Demand Based Model": "demand_based_model.html",
    "Accessibility Based Model": "accessibility_based_model.html"
}

model_name = st.selectbox(
    "Select Model to View Map",
    options=list(model_files.keys()),
    index=0,
    format_func=lambda x: x
)

# Show spinner for a short time to simulate loading
with st.spinner("Loading map..."):
    time.sleep(0.5)  # Short delay for UX feedback

selected_file = model_files[model_name]
html_file_path = f"src/web_dashboard/{selected_file}"
with open(html_file_path, 'r', encoding='utf-8') as f:
    html_content = f.read()
components.html(html_content, height=600, scrolling=True)

# Add a legend below the map
st.markdown("""
<div style='display: flex; align-items: center; margin-bottom: 32px;'>
    <div style='width: 20px; height: 20px; background: rgb(0,0,255); border-radius: 50%; margin-right: 8px;'></div>
    <span style='margin-right: 24px;'>Current Hospital</span>
    <div style='width: 20px; height: 20px; background: rgb(0,255,0); border-radius: 50%; margin-right: 8px;'></div>
    <span>Predicted Hospital</span>
</div>
""", unsafe_allow_html=True)

# Dummy metrics
equity_index = np.random.uniform(0.5, 1.0)
hdr = np.random.uniform(0.2, 0.8)
overserved_areas = np.random.randint(0, 5)

# st.subheader("Key Metrics")
# col1, col2, col3 = st.columns(3)
# col1.metric("Equity Index", f"{equity_index:.2f}")
# col2.metric("HDR", f"{hdr:.2f}")
# col3.metric("Overserved Areas", f"{overserved_areas}")