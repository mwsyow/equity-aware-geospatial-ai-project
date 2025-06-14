import os
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from enum import StrEnum

from .ai_planner.planner import HospitalPlanner, CRS
from .index_demand_forecast.demand_forecast import (
    forecast_demand_per_district_in_saarland as run_demand_forecast,
    df_hospital_inpatients,
    df_saarland_diseases_history,
    CUT_OFF_YEAR, YEAR, REGION_CODE, VALUE
)
from .index_demand_forecast.demand_forecast import forecast_demand_per_district_in_saarland as run_demand_forecast
from .index_elderly_share.elderly_share import run as run_elderly_share
from .index_gisd.gisd import run as run_gisd
from .index_hospital_capacity.hospital_capacity_index_dict import calculate_hospital_capacity_index as run_hospital_capacity_index
from .index_travel_accessibility.TAI import RUN as run_travel_time_index
from .index_travel_accessibility.travel_time_and_centroid import (
    get_centroids, 
    map_predicted_and_existing_hospitals
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
    Index.TRAVEL_TIME: run_travel_time_index,
}

def assemble_indexes() -> pd.DataFrame:
    """
    Assembles all individual indexes into a combined DataFrame.
    
    The function iterates through the INDEX_FUNC_MAP, calls each index calculation function,
    and combines the results into a single DataFrame.
    
    Returns:
        pd.DataFrame: Combined DataFrame where:
            - Rows are districts
            - Columns are different indexes (forecast_demand, elderly_share, etc.)
            - Values are the calculated index values for each district
    """
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

def equity_index(index_df: pd.DataFrame, weights: dict) -> pd.Series:
    """
    Calculate the Equity Index based on weighted combinations of individual indexes.
    
    The equity index is calculated as:
    EquityIndex = w1*DemandForecast + w2*GISD + w3*TravelTime + w4*Accessibility
    where Accessibility = w5*ElderlyShare + w6*HospitalCapacity
    
    Higher values indicate worse equity conditions.
    
    Args:
        index_df (pd.DataFrame): DataFrame containing the index values for each district
        weights (dict): Dictionary mapping Index enum values to their respective weights
    
    Returns:
        pd.Series: Equity Index values for each district
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
    Calculate the centroid coordinates for each district.
    
    Retrieves district centroids from the travel time and centroid module
    and converts them to Shapely Point objects.
    
    Returns:
        pd.Series: District centroids where:
            - Index: AGS_CODE (district identifier)
            - Values: Shapely Point objects containing (longitude, latitude)
    """
    centroids = get_centroids()
    centroid_series = pd.Series({k: Point(v["lon"], v["lat"]) for k, v in centroids.items()})
    return centroid_series

def current_hospital_demand() -> pd.Series:
    """
    Calculate current hospital demand based on historical inpatient data.
    
    Processes hospital inpatient data and disease history to calculate
    the current demand distribution across districts.
    
    Returns:
        pd.Series: Current hospital demand where:
            - Index: District identifiers as strings
            - Values: Calculated demand values for each district
    """
    dfhi = df_hospital_inpatients()
    curr_saarland_hospital_inpatients = dfhi[(dfhi[YEAR]==CUT_OFF_YEAR) & (dfhi[REGION_CODE]==10)][VALUE].values[0]
    dfdsh = df_saarland_diseases_history()
    dfdsh = dfdsh[CUT_OFF_YEAR].map(lambda x: x/dfdsh[CUT_OFF_YEAR].sum())
    curr_demand = dfdsh * curr_saarland_hospital_inpatients
    curr_demand.index = [str(i) for i in curr_demand.index]
    return curr_demand

def get_district_gdf(equity_df: pd.DataFrame) -> gpd.GeoDataFrame:# Load the Excel file
    file_path = os.path.join(os.path.dirname(__file__), "index_travel_accessibility", "data", "processed", "saarland_districts_sample_points.xlsx")
    df_sample_points = pd.read_excel(file_path)
    SAARLAND_DISTRICT_MAPPING = {
        "Saarlouis": 10044,
        "St. Wendel": 10046,
        "Saarpfalz-Kreis": 10045,
        "Merzig-Wadern": 10042,
        "Neunkirchen": 10043,
        "Regionalverband Saarbrücken": 10041 
    }
    # Display basic information about the dataframe
    df_sample_points = df_sample_points.replace(SAARLAND_DISTRICT_MAPPING)
    df_sample_points = df_sample_points.set_index('district')

    #STEP1: Load and Prepare District Data (Centroids)
    curr_hospital_demand = current_hospital_demand()
    idx = curr_hospital_demand.index

    # Create the districts dataframe without MultiIndex
    districts_df = pd.DataFrame({
        "demand": curr_hospital_demand.reindex(idx),
        "equity_index": equity_df.reindex(idx),
        "centroid": centroid().reindex(idx)
    })
    districts_df.index = districts_df.index.astype(int)
    districts_df = df_sample_points.join(districts_df[['demand', 'equity_index']], how='left')
    districts_df["centroid"] = districts_df.apply(
        lambda row: Point(row["longitude"], row["latitude"]),
        axis=1
    )
    # Create GeoDataFrame properly
    districts_gdf = gpd.GeoDataFrame(districts_df, geometry="centroid", crs=CRS)
    districts_gdf = districts_gdf.reset_index().rename(columns={'district': 'district_code'})
    return districts_gdf

def main():
    index_df = assemble_indexes()
    # Set the weights for the indexes
    weight = {
        Index.FORECAST_DEMAND: 0.25,
        Index.ELDERLY_SHARE: 0.25,
        Index.GISD: 0.25,
        Index.HOSPITAL_CAPACITY: 0.25,
        Index.TRAVEL_TIME: 0.25,
        Index.ACCESSIBILITY: 0.25,
    }
    equity_df = equity_index(index_df, weight)
    districts_gdf = get_district_gdf(equity_df)
    planner = HospitalPlanner(districts_gdf)
    n_samples_per_centroid = 1
    radius_km = 5
    planner.generate_candidates(n_samples_per_centroid=n_samples_per_centroid, radius_km=radius_km)
    planner.init_graph()
    planner.build_matrix()
    
    initial_weights = {
        'equity_weight': 0.75,
        'travel_weight': 0.065,
        'beds_weight': 0.065,
        'supply_weight': 0.065,
    }

    selected_candidates = planner.run(
        initial_weights=initial_weights,
        step_size=0.01,
        num_neighbors=2,
        eh_distance_threshold=7_000,
        c2c_distance_threshold=7_000,
        time_threshold=30,
        max_beds_per_hospital=300,
        min_beds_per_hospital=20,
        max_beds=1500,
        k=2,
    )
    path = os.path.join(os.path.dirname(__file__), "results", "sample_points.html")
    map_predicted_and_existing_hospitals(path, selected_candidates)
    
if __name__ == "__main__":
    main()
