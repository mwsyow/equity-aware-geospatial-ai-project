from index_demand_forecast.demand_forecast import (
    df_saarland_diseases_history as loading_hospital_inpatients_per_district,
    forecast_diseases_history as forecast_diseases_history_hfdr,
    df_per_capita_demand as dfpcd_hfdr,
    grid_search_ARIMA as grid_search_ARIMA_hfdr,
    forecast_ARIMA as forecast_ARIMA_hfdr,
    forecast_demand as forecast_demand_hfdr
    
)
import pandas as pd
from enum import StrEnum

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
    Loads and processes hospital data from an Excel file.

    The function performs the following operations:
    1. Dynamically finds the header row in the Excel file
    2. Filters data for the specified region (Saarland)
    3. Extracts relevant columns (region, district, beds)
    4. Cleans the data by removing rows with missing or zero beds

    Returns:
        pd.DataFrame: A DataFrame containing columns:
            - region (int): Region/Land code
            - district (float): District/Kreis code
            - beds (int): Number of hospital beds
    """
    # Re-read with correct header
    df = pd.read_excel(hospital_file_path, engine="openpyxl")
    df.columns = df.columns.str.strip()


    # Adapt to new Excel format
    # Rename columns for consistency
    # Rename columns based on the new Excel format
    df = df.rename(columns={
        "district_code": "district",
        "bed_allocation": "beds"
    })


    # Clean the data by dropping rows with missing or zero beds
    df = df.dropna(subset=["beds"])
    df = df[df["beds"] > 0]

    return df


def calculate_hfdr(hospital_file_path: str):
    # Load and process hospital data
    hospital_df = load_hospital_data(hospital_file_path)
    hospital_df["district"] = hospital_df["district"].astype(str)

    # Filter only Saarland districts
    saarland_districts = set(SAARLAND_AGS.values())
    hospital_df = hospital_df[hospital_df["district"].isin(saarland_districts)]

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

    # Rename index to readable district names
    hfdr_ratio.index = [name for name in SAARLAND_AGS]

    print("Ratio of total beds to average forecasted demand per district (hfdr):")
    print(hfdr_ratio)

    return hfdr_ratio


