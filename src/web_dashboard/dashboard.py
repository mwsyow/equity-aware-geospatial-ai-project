import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import sys
import os

# Add the src directory to the path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'index_travel_accessibility')))
from travel_time_and_centroid import get_hospital_df

st.set_page_config(page_title="Equity-Aware Geospatial AI Dashboard", layout="wide")

st.title("Equity-Aware Geospatial AI: Scenario Simulation Dashboard")

# Sidebar controls
st.sidebar.header("Scenario Controls")
num_hospitals = st.sidebar.slider("Number of Hospitals", min_value=1, max_value=20, value=2, step=1)

# Load real current hospitals
df_current = get_hospital_df()
df_current = df_current.rename(columns={"Lon": "lon", "Lat": "lat", "HospitalAddress": "name"})
df_current["type"] = "Current"
df_current["color"] = [[0, 0, 255] for _ in range(len(df_current))]  # Blue

# Simulate predicted hospitals (dummy for now)
predicted_hospitals = pd.DataFrame({
    'lat': [49.24 + 0.01*i for i in range(num_hospitals)],
    'lon': [6.99 + 0.01*i for i in range(num_hospitals)],
    'name': [f'Predicted {i+1}' for i in range(num_hospitals)],
    'type': ['Predicted']*num_hospitals,
    'color': [[0, 255, 0]]*num_hospitals  # Green
})

# Combine for map
df_map = pd.concat([df_current[['lat', 'lon', 'name', 'type', 'color']], predicted_hospitals], ignore_index=True)

# Pydeck map
layer = pdk.Layer(
    'ScatterplotLayer',
    df_map,
    get_position='[lon, lat]',
    get_color='color',
    get_radius=200,
    pickable=True
)
view_state = pdk.ViewState(latitude=49.25, longitude=7.0, zoom=10, pitch=0)
st.subheader("Hospital Locations (Current and Predicted)")
st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text": "{name} ({type})"}))

st.write("Predicted hospitals (dummy):")
st.dataframe(predicted_hospitals)

# Dummy metrics
equity_index = np.random.uniform(0.5, 1.0)
hdr = np.random.uniform(0.2, 0.8)
overserved_areas = np.random.randint(0, 5)

# st.subheader("Key Metrics")
# col1, col2, col3 = st.columns(3)
# col1.metric("Equity Index", f"{equity_index:.2f}")
# col2.metric("HDR", f"{hdr:.2f}")
# col3.metric("Overserved Areas", f"{overserved_areas}")