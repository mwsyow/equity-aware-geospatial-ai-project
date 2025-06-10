import os
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
    get_centroids, get_hospital_df, get_travel_time_matrix,
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

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

def compute_pca_weights(index_df: pd.DataFrame) -> dict[Index, float]:
    """
    Derive weights from PCA loadings on the first principal component.
    Returns normalized weights for each index (excluding the composite ACCESSIBILITY index),
    and computes the ACCESSIBILITY index weight as the sum of its components.
    
    ACCESSIBILITY = ELDERLY_SHARE + HOSPITAL_CAPACITY (weighted sum)
    """
    # Ensure columns match Enum values
    df = index_df.copy()
    
    # Drop ACCESSIBILITY because it's composite
    df = df.drop(columns=[Index.ACCESSIBILITY.value], errors='ignore')

    # Standardize data
    scaler = StandardScaler()
    X = scaler.fit_transform(df.values)

    # Perform PCA
    pca = PCA(n_components=1)
    pca.fit(X)

    # Get absolute PCA loadings for interpretability
    loadings = np.abs(pca.components_[0])

    # Map column names to loadings
    weights_raw = dict(zip(df.columns, loadings))

    # Normalize weights so they sum to 1
    total = sum(weights_raw.values())
    normalized_weights = {Index(col): val / total for col, val in weights_raw.items()}

    # Compute ACCESSIBILITY as the sum of its components' weights
    accessibility_weight = (
        normalized_weights.get(Index.ELDERLY_SHARE, 0.0) +
        normalized_weights.get(Index.HOSPITAL_CAPACITY, 0.0)
    )
    normalized_weights[Index.ACCESSIBILITY] = accessibility_weight

    return normalized_weights

def main():
    """
    Main entry point for the equity-aware hospital planning system.
    
    Workflow:
    1. Loads hospital and district data
    2. Assembles equity indexes with weights
    3. Calculates current demand
    4. Initializes AI planner with parameters:
        - hospitals: DataFrame with hospital information
        - districts: GeoDataFrame with district metrics
        - travel_time: Matrix of travel times between sites
        - budget_beds: Total beds available (1000)
        - max_open_sites: Maximum number of hospitals (2)
        - alpha: Cost vs equity weight (0.6)
        - max_travel: Maximum travel time in minutes (30)
    5. Runs optimization
    6. Visualizes results on map
    
    The function saves the resulting map visualization to the results directory.
    """
    hospitals_df = get_hospital_df()
    hospitals_df = hospitals_df.set_index("SiteID")
        
    index_df = assemble_indexes()
    # Compute PCA weights for the indexes
    weight = compute_pca_weights(index_df)
    # weight = {
    #     Index.FORECAST_DEMAND: 0.25,
    #     Index.ELDERLY_SHARE: 0.25,
    #     Index.GISD: 0.25,
    #     Index.HOSPITAL_CAPACITY: 0.25,
    #     Index.TRAVEL_TIME: 0.25,
    #     Index.ACCESSIBILITY: 0.25,
    # }
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
        predict_new=4
    )

    plan = planner.optimize_with_bayesian()

    print("Open sites:", plan["open_sites"])
    print("Beds allocation:", plan["beds_alloc"])
    print("Objective value:", plan["objective"])

    save_path = os.path.join(os.path.dirname(__file__), "results", "saarland_map_with_predicted_and_exisiting_hospitals.html")
    # planner.plot_predicted_hospitals()
    map_predicted_and_existing_hospitals(
        save_path,
        planner.predicted_hospitals
    )

if __name__ == "__main__":
    main()
