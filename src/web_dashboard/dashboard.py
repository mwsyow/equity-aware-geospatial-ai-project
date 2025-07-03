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

# Custom centered spinner HTML
spinner_html = """
<div style="display: flex; justify-content: center; align-items: center; height: 400px;">
  <div>
    <svg width="60" height="60" viewBox="0 0 44 44" stroke="#1f77b4">
      <g fill="none" fill-rule="evenodd" stroke-width="2">
        <circle cx="22" cy="22" r="1">
          <animate attributeName="r"
            begin="0s" dur="1.8s"
            values="1; 20"
            calcMode="spline"
            keyTimes="0; 1"
            keySplines="0.165, 0.84, 0.44, 1"
            repeatCount="indefinite" />
          <animate attributeName="stroke-opacity"
            begin="0s" dur="1.8s"
            values="1; 0"
            calcMode="spline"
            keyTimes="0; 1"
            keySplines="0.3, 0.61, 0.355, 1"
            repeatCount="indefinite" />
        </circle>
        <circle cx="22" cy="22" r="1">
          <animate attributeName="r"
            begin="-0.9s" dur="1.8s"
            values="1; 20"
            calcMode="spline"
            keyTimes="0; 1"
            keySplines="0.165, 0.84, 0.44, 1"
            repeatCount="indefinite" />
          <animate attributeName="stroke-opacity"
            begin="-0.9s" dur="1.8s"
            values="1; 0"
            calcMode="spline"
            keyTimes="0; 1"
            keySplines="0.3, 0.61, 0.355, 1"
            repeatCount="indefinite" />
        </circle>
      </g>
    </svg>
  </div>
</div>
"""

# Show the centered spinner
spinner_placeholder = st.empty()
spinner_placeholder.markdown(spinner_html, unsafe_allow_html=True)
time.sleep(2)  # Duration of the loader

# Replace spinner with the map
spinner_placeholder.empty()

# Sidebar controls
# st.sidebar.header("Scenario Controls")

# Load real current hospitals
df_current = get_hospital_df()
df_current = df_current.rename(columns={"Lon": "lon", "Lat": "lat", "HospitalAddress": "name"})
df_current["type"] = "Current"
df_current["color"] = [[0, 0, 255] for _ in range(len(df_current))]  # Blue

# Path to your HTML file
html_file_path = "src/web_dashboard/main_model_run.html"

st.subheader("Hospital Locations (Current and Predicted)")

# Read and display the HTML file
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