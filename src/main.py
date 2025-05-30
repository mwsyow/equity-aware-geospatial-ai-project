import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from enum import StrEnum

from ai_planner.agentic_ai_planner_2 import AgenticPlannerWithPrediction
from index_demand_forecast.demand_forecast import (
    forecast_demand_per_district_in_saarland as run_demand_forecast,
    df_hospital_inpatients,
    df_saarland_diseases_history,
    CUT_OFF_YEAR, YEAR, REGION_CODE, VALUE
)
from index_demand_forecast.demand_forecast import forecast_demand_per_district_in_saarland as run_demand_forecast
from index_elderly_share.elderly_share import run as run_elderly_share
from index_gisd.gisd import run as run_gisd
from index_hospital_capacity.hospital_capacity_index_dict import calculate_hospital_capacity_index as run_hospital_capacity_index
from index_travel_accessibility.TAI import RUN as run_travel_time_index

class Index(StrEnum):
    FORECAST_DEMAND = "forecast_demand_index"
    ELDERLY_SHARE = "elderly_share_index"
    GISD = "gisd_index"
    HOSPITAL_CAPACITY = "hospital_capacity_index"
    TRAVEL_TIME = "travel_time_index"
    ACCESSIBILITY = "accessibility_index"

INDEX_FUNC_MAP = {
    Index.FORECAST_DEMAND: run_demand_forecast,
    Index.ELDERLY_SHARE: run_elderly_share,
    Index.GISD: run_gisd,
    Index.HOSPITAL_CAPACITY: run_hospital_capacity_index,
    Index.TRAVEL_TIME: run_travel_time_index
}

def assemble_indexes() -> pd.DataFrame:
    combinded_df = []
    for index in INDEX_FUNC_MAP.values():
        res = index()
        # If res is a dict of scalars, wrap values in a list
        if isinstance(res, dict) and all(not isinstance(v, (list, pd.Series, np.ndarray, pd.DataFrame)) for v in res.values()):
            res = {k: [v] for k, v in res.items()}
        combinded_df.append(pd.DataFrame(res))
    df = pd.concat(combinded_df)
    df = df.transpose()
    df.columns = list(INDEX_FUNC_MAP.keys())
    return df

#EquityIndex = w1 x DemandForecastingIndex + w2 x GISD-Index + w3 x TravelTimeIndex + w4 x AccesibilityIndex

# where AccesibilityIndex = w5 x ElderlyShare + w6 x HospitalCapacityIndex
# This formula is applicable at a district level

#Higher value of EquityIndex should mean worse equity and vice versa

def equity_index(index_df: pd.DataFrame, weights: dict) -> pd.Series:
    """
    Calculate the Equity Index based on the provided index DataFrame and weights.
    
    Parameters:
        index_df (pd.DataFrame): DataFrame containing the index values.
        weights (dict): Dictionary containing the weights for each index.
        
    Returns:
        pd.Series: Equity Index values for each district.
    """
    equity = []
    for district, index in index_df.iterrows():
        # Calculate Accessibility Index
        accessibility_index = (
            weights[Index.ELDERLY_SHARE] * index[Index.ELDERLY_SHARE] +
            weights[Index.HOSPITAL_CAPACITY] * index[Index.HOSPITAL_CAPACITY]
        )
        
        # Calculate Equity Index
        equity_index = (
            weights[Index.FORECAST_DEMAND] * index[Index.FORECAST_DEMAND] +
            weights[Index.GISD] * index[Index.GISD] +
            weights[Index.TRAVEL_TIME] * index[Index.TRAVEL_TIME] +
            weights[Index.ACCESSIBILITY] * accessibility_index
        )
        equity.append(equity_index)
    equity_index = pd.Series(equity, index=index_df.index, name="EquityIndex")
    
    return equity_index

def current_hospital_demand() -> pd.Series:
    dfhi = df_hospital_inpatients()
    curr_saarland_hospital_inpatients = dfhi[(dfhi[YEAR]==CUT_OFF_YEAR) & (dfhi[REGION_CODE]==10)][VALUE].values[0]
    dfdsh = df_saarland_diseases_history()
    dfdsh = dfdsh[CUT_OFF_YEAR].map(lambda x: x/dfdsh[CUT_OFF_YEAR].sum())
    curr_demand = dfdsh * curr_saarland_hospital_inpatients
    return curr_demand

def main():
    """
        Initialize planner with:
        - hospitals: DataFrame with SiteID, Lon, Lat, CostPerBed, MaxBeds
        - districts: DataFrame with AGS_CODE, Demand, EquityIndex, centroid (geometry)
        - travel_time: dict of dicts: travel_time[site][district] in minutes
        - budget_beds: total beds available to allocate
        - max_open_sites: max hospitals that can be open simultaneously
        - alpha: weight for cost vs equity penalty in objective
        - max_travel: max travel time threshold (minutes)
    """

    
    # Randomize MaxBeds for hospitals
    
    
    #IMPORT THE DATA SETS AND REPLACE THE DUMMY BELOW 
    
    
    # Create dummy hospital data
    hospitals_df = pd.DataFrame({
        "SiteID": ["H1", "H2"],
        "Lon": [10.0, 10.5],
        "Lat": [50.0, 50.5],
    })
    
    
    hospitals_df_dict = hospitals_df.to_dict('list')
    
    np.random.seed(42)
    maxBeds = np.random.randint(200, 1001, size=len(hospitals_df_dict["SiteID"]))
    
    hospitals_df["MaxBeds"] = maxBeds
    
    
    
    # Create dummy district data with centroids
    districts_df = pd.DataFrame({
        "AGS_CODE": ["D1", "D2", "D3"],
        "Demand": [300, 400, 500],
        "EquityIndex": [1.0, 0.8, 0.4],
        "Lon": [10.1, 10.3, 10.6],
        "Lat": [50.1, 50.2, 50.4]
    })
    districts_df["centroid"] = districts_df.apply(lambda row: Point(row["Lon"], row["Lat"]), axis=1)
    districts_gdf = gpd.GeoDataFrame(districts_df, geometry="centroid")
    
    # Create dummy travel_time dict: travel_time[hospital][district] in minutes
    travel_time_dict = {
        "H1": {"D1": 15, "D2": 25, "D3": 35},
        "H2": {"D1": 20, "D2": 10, "D3": 30}
    } 
    
    #travel time planner 
    
    # Run planner
    planner = AgenticPlannerWithPrediction(
        hospitals_df,
        districts_gdf,
        travel_time_dict,
        budget_beds=1000,
        max_open_sites=2,
        alpha=0.6,
        max_travel=30,
        predict_new=3
    )
    
    plan = planner.optimize_with_bayesian()
    
    print("Open sites:", plan["open_sites"])
    print("Beds allocation:", plan["beds_alloc"])
    print("Objective value:", plan["objective"])
    
    # planner.plot_predicted_hospitals()
    
    planner.plot_predicted_hospitals()
    
if __name__ == "__main__":
    print(assemble_indexes())
