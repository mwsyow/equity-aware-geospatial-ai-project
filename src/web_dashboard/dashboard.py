import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import sys
import os
import streamlit.components.v1 as components
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'index_travel_accessibility')))
from travel_time_and_centroid import get_hospital_df

    
st.set_page_config(page_title="Equity-Aware Geospatial AI Dashboard", layout="wide")

st.title("Equity-Aware Geospatial AI: Scenario Simulation Dashboard")

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

model_descriptions = {
    "Main Model": "Optimized hospital locations balancing travel time and equity across Saarland districts.",
    "Policy Maker Model": "Hospital allocation based on policy maker decisions and existing planning guidelines.",
    "Accessibility Based Model": "Hospitals placed to maximize accessibility and minimize average travel time for residents.",
    "Demand Based Model": "Hospital locations optimized to meet projected healthcare demand in each district.",
    "Deprivation Aware Model": "Hospital allocation prioritizing underserved and socioeconomically deprived areas."
}

model_name = st.selectbox(
    "Select Model to View Map",
    options=list(model_files.keys()),
    index=0,
    format_func=lambda x: x
)

with st.spinner("Loading map..."):
    time.sleep(0.5) 

selected_file = model_files[model_name]
html_file_path = f"src/web_dashboard/{selected_file}"
with open(html_file_path, 'r', encoding='utf-8') as f:
    html_content = f.read()
components.html(html_content, height=600, scrolling=True)

st.markdown(f"**Model Description:** {model_descriptions[model_name]}")
