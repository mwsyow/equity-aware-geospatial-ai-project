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
from index_travel_accessibility.travel_time_and_centroid import (
    get_centroids, get_hospital_df, get_travel_time_matrix
)
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

def centroid() -> pd.Series:
    """
    Returns a Series with AGS_CODE as index and centroid coordinates as values.
    """
    centroids = get_centroids()
    centroid_series = pd.Series({k: Point(v["lon"], v["lat"]) for k, v in centroids.items()})
    return centroid_series

def current_hospital_demand() -> pd.Series:
    dfhi = df_hospital_inpatients()
    curr_saarland_hospital_inpatients = dfhi[(dfhi[YEAR]==CUT_OFF_YEAR) & (dfhi[REGION_CODE]==10)][VALUE].values[0]
    dfdsh = df_saarland_diseases_history()
    dfdsh = dfdsh[CUT_OFF_YEAR].map(lambda x: x/dfdsh[CUT_OFF_YEAR].sum())
    curr_demand = dfdsh * curr_saarland_hospital_inpatients
    curr_demand.index = [str(i) for i in curr_demand.index]
    return curr_demand

def main():
    """
        Initialize planner with:
        - hospitals: DataFrame with SiteID, HospitalAddress, Lon, Lat, MaxBeds
        - districts: DataFrame with AGS_CODE, Demand, EquityIndex, centroid (geometry)
        - travel_time: dict of dicts: travel_time[site][district] in minutes
        - budget_beds: total beds available to allocate
        - max_open_sites: max hospitals that can be open simultaneously
        - alpha: weight for cost vs equity penalty in objective
        - max_travel: max travel time threshold (minutes)
    """
    hospitals_df = get_hospital_df()
    hospitals_df = hospitals_df.set_index("SiteID")
        
    index_df = assemble_indexes()
    weight = {
        Index.FORECAST_DEMAND: 0.25,
        Index.ELDERLY_SHARE: 0.25,
        Index.GISD: 0.25,
        Index.HOSPITAL_CAPACITY: 0.25,
        Index.TRAVEL_TIME: 0.25,
        Index.ACCESSIBILITY: 0.25,
    }
    curr_hospital_demand = current_hospital_demand()
    idx = curr_hospital_demand.index
    districts_df = pd.concat([
        curr_hospital_demand.reindex(idx),
        equity_index(index_df, weight).reindex(idx),
        centroid().reindex(idx)
    ], axis=1, keys=["Demand", "EquityIndex", "centroid"])
    districts_gdf = gpd.GeoDataFrame(districts_df, geometry="centroid")

    travel_time_dict = get_travel_time_matrix()

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
    main()
