import sys
import os

# Add src directory to path for imports
current_dir = os.path.dirname(__file__)
src_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from index_demand_forecast.demand_forecast import (
    df_saarland_diseases_history as loading_hospital_inpatients_per_district,
    forecast_diseases_history as forecast_diseases_history_hfdr,
    df_per_capita_demand as dfpcd_hfdr,
    grid_search_ARIMA as grid_search_ARIMA_hfdr,
    forecast_ARIMA as forecast_ARIMA_hfdr,
    forecast_demand as forecast_demand_hfdr
    
)
import pandas as pd
# StrEnum compatibility for Python < 3.11
try:
    from enum import StrEnum
except ImportError:
    from enum import Enum
    class StrEnum(str, Enum):
        pass

CUT_OFF_YEAR = 2021
YEAR = 'time'
REGION_CODE = '1_variable_attribute_code'
VALUE = 'value'
ICD_VARIANT = '2_variable_attribute_label'
PROJECTION_VARIANT = '2_variable_attribute_code'
DISTRICT_CODE = 'district_code'

SAARLAND_AGS = {
    "Regionalverband Saarbrücken": "10041",
    "Merzig-Wadern": "10042",
    "Neunkirchen": "10043",
    "Saarlouis": "10044",
    "Saarpfalz-Kreis": "10045",
    "St. Wendel": "10046"
}


class ProjectionVariant(StrEnum):
    """Population projection variants for demographic forecasting.
    
    Enumeration of different population projection scenarios used by the demographic model.
    """
    VAR01 = 'BEV-VARIANTE-01'
    VAR02 = 'BEV-VARIANTE-02'
    VAR03 = 'BEV-VARIANTE-03'
    VAR04 = 'BEV-VARIANTE-04'
    VAR05 = 'BEV-VARIANTE-05'

def compute_demand_for_saarland(region_code=10, period=9):
    dfpcd = dfpcd_hfdr()
    per_capita_demand = dfpcd.loc[region_code]

    best_model, _ = grid_search_ARIMA_hfdr(
        per_capita_demand, 
        p_values=[3], 
        d_values=[0], 
        q_values=[0]
    )
    
    forecast, conf_int = forecast_ARIMA_hfdr(best_model, period)
    demand, _ = forecast_demand_hfdr(forecast, region_code, ProjectionVariant.VAR01, conf_int)

    df = loading_hospital_inpatients_per_district()
    forecast_diseases, diseases_conf_int = forecast_diseases_history_hfdr(df, period, [1], [1], [0])
    forecast_diseases = forecast_diseases.reset_index().rename(columns={'index': YEAR})
    forecast_diseases = forecast_diseases.pivot(
        index=DISTRICT_CODE,
        columns=YEAR,
        values=VALUE
    )

    forecast_diseases_standardized = forecast_diseases.div(forecast_diseases.sum(axis=0), axis=1)

    demand_per_district = forecast_diseases_standardized.mul(demand, axis=1)

    average_forecasted_demand_per_district = demand_per_district.mean(axis=1)
    
    return average_forecasted_demand_per_district


def load_hospital_data(hospital_file_path: str) -> pd.DataFrame:
    """
    Loads and processes hospital data from a model result Excel file.

    The function performs the following operations:
    1. Reads the model result Excel file
    2. Extracts bed allocation data
    3. Aggregates beds by district

    Returns:
        pd.DataFrame: A DataFrame containing columns:
            - district (str): District code
            - beds (int): Number of hospital beds
    """
    # Read the model result file
    try:
        df = pd.read_excel(hospital_file_path, engine="openpyxl")
    except Exception as e:
        print(f"❌ Error reading file {hospital_file_path}: {e}")
        print(f"📁 File exists: {os.path.exists(hospital_file_path)}")
        print(f"📁 File size: {os.path.getsize(hospital_file_path) if os.path.exists(hospital_file_path) else 'N/A'}")
        # Return empty DataFrame with expected columns
        return pd.DataFrame(columns=['district', 'beds'])
    
    # Check if this is a model result file (has bed_allocation column)
    if 'bed_allocation' in df.columns:
        # This is a model result file
        print(f"📊 Loading model result file: {hospital_file_path}")
        
        # Group by district_code and sum bed_allocation
        beds_by_district = df.groupby('district_code')['bed_allocation'].sum().reset_index()
        beds_by_district = beds_by_district.rename(columns={'district_code': 'district', 'bed_allocation': 'beds'})
        
        # Convert district codes to string format expected by other functions
        beds_by_district['district'] = beds_by_district['district'].astype(str)
        
        return beds_by_district
    else:
        # Fallback to original hospital capacity file format
        print(f"🏥 Loading hospital capacity file: {hospital_file_path}")
        df = pd.read_excel(hospital_file_path, sheet_name='KHV_2021', header=4, engine="openpyxl")
        df.columns = df.columns.str.strip()

        # Rename columns for consistency
        df = df.rename(columns={
            "Land": "region",
            "Kreis": "district", 
            "INSG": "beds"
        })

        # Convert beds to numeric, handling any non-numeric values
        df["beds"] = pd.to_numeric(df["beds"], errors='coerce')

        # Clean the data by dropping rows with missing or zero beds
        df = df.dropna(subset=["beds"])
        df = df[df["beds"] > 0]

        # Filter for Saarland (region code 10) and convert district codes to match inpatient data
        df = df[df["region"] == 10].copy()
        df["district"] = 10000 + df["district"].astype(int)
        df["district"] = df["district"].astype(str)

        return df


def calculate_hfdr(hospital_file_path: str):
    try:
        # Load and process hospital data
        hospital_df = load_hospital_data(hospital_file_path)
        
        # Check if we have valid data
        if hospital_df.empty:
            print(f"❌ No hospital data found for {hospital_file_path}")
            # Return neutral values for all districts
            return pd.Series([0.5] * len(SAARLAND_AGS), index=list(SAARLAND_AGS.keys()))
        
        hospital_df["district"] = hospital_df["district"].astype(str)

        # Filter only Saarland districts
        saarland_districts = set(SAARLAND_AGS.values())
        hospital_df = hospital_df[hospital_df["district"].isin(saarland_districts)]

        if hospital_df.empty:
            print(f"❌ No Saarland district data found in {hospital_file_path}")
            return pd.Series([0.5] * len(SAARLAND_AGS), index=list(SAARLAND_AGS.keys()))

        # Sum beds per district
        beds_per_district = hospital_df.groupby("district")["beds"].sum()

        # Compute average forecasted demand per district
        forecasted_demand_per_district = compute_demand_for_saarland()
        forecasted_demand_per_district.index = forecasted_demand_per_district.index.astype(str)
        forecasted_demand_per_district = forecasted_demand_per_district[forecasted_demand_per_district.index.isin(saarland_districts)]

        # Calculate hfdr ratio
        hfdr_ratio = beds_per_district / forecasted_demand_per_district

        # Reindex by Saarland district code order
        ordered_district_codes = list(SAARLAND_AGS.values())
        hfdr_ratio = hfdr_ratio.reindex(ordered_district_codes)

        # Fill NaN values with neutral value
        hfdr_ratio = hfdr_ratio.fillna(0.5)

        # Rename index to readable district names
        hfdr_ratio.index = [name for name in SAARLAND_AGS]

        print("Ratio of total beds to average forecasted demand per district (hfdr):")
        print(hfdr_ratio)

        return hfdr_ratio
        
    except Exception as e:
        print(f"❌ Error calculating HFDR: {e}")
        return pd.Series([0.5] * len(SAARLAND_AGS), index=list(SAARLAND_AGS.keys()))


