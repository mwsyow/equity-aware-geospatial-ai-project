import os
import time
import folium
import requests
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter


# -----------------------------------------------------------------------------------------
# Configurations
# -----------------------------------------------------------------------------------------


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
OUT_DIR  = os.path.join(BASE_DIR, "data", "processed")

# MUST HAVE
HOSPITAL_DATA_PATH = os.path.join(RAW_DIR, "Krankenhausverzeichnis_2021.xlsx" )
NUTS_DATA_PATH = os.path.join(RAW_DIR, "NUTS_RG_01M_2021_4326_LEVL_3.shp")

STATUS_QUO_MODEL_PATH = os.path.join(OUT_DIR, "saarland_hospitals_with_coords.xlsx") 
POLICY_MAKER_MODEL_PATH = os.path.join(OUT_DIR, "policy_maker_model.xlsx") 
MAIN_MODEL_PATH = os.path.join(OUT_DIR, "main.xlsx") 
DEMAND_BASED_MODEL_PATH = os.path.join(OUT_DIR, "demand_based_model.xlsx") 
DEPRIVATION_AWARE_MODEL_PATH = os.path.join(OUT_DIR, "deprivation_aware_model.xlsx") 
ACCESSIBILITY_BASED_MODEL_PATH = os.path.join(OUT_DIR, "accessibility_based_model.xlsx") 

SAARLAND_DISTRICTS_SAMPLE_POINTS_PATH = os.path.join(OUT_DIR, "saarland_districts_sample_points.xlsx") 
NEAREST_HOSPITALS_TO_SAMPLE_POINTS_PATH = os.path.join(OUT_DIR, "nearest_hospitals_to_sample_points.xlsx") 
# TRAVEL_TIME_FROM_SAMPLE_TO_HOSPITAL_PATH = os.path.join(OUT_DIR, "travel_time_from_sample_to_hospital.xlsx") 
# CALCULATED_METRICS_FOR_TRAVEL_TIME_PATH = os.path.join(OUT_DIR, "calculated_metrics_from_travel_time.xlsx") 


SAARLAND_AGS = {
    "Regionalverband Saarbrücken": "10041",
    "Merzig-Wadern": "10042",
    "Neunkirchen": "10043",
    "Saarlouis": "10044",
    "Saarpfalz-Kreis": "10045",
    "St. Wendel": "10046"
}


# MUST HAVE DATASETS TO RUN THIS PIPELINE   ------> ./data/raw/filename
# 1. NUTS_RG_01M_2021_4326_LEVL_3.shp

# -----------------------------------------------------------------------------------------

    
    # MODEL_PATHS = {
    #     "status_quo_model": "equity-aware-geospatial-ai-project\\src\\evaluation\\data\\processed\\RES_status_quo_model.xlsx",
    #     "demand_based_model": "equity-aware-geospatial-ai-project\\src\\evaluation\\data\\processed\\RES_demand_based_model.xlsx",
    #     "deprivation_aware_model": "equity-aware-geospatial-ai-project\\src\\evaluation\\data\\processed\\RES_deprivation_aware_model.xlsx",
    #     "accessibility_based_model": "equity-aware-geospatial-ai-project\\src\\evaluation\\data\\processed\\RES_accessibility_based_model.xlsx",
    #     "main_model": "equity-aware-geospatial-ai-project\\src\\evaluation\\data\\processed\\RES_main_model.xlsx",
    # }


def get_TAI_scaled_for_model(model_name: str) -> pd.DataFrame:
    """
    Returns a DataFrame with mean travel time scaled for each district for a single model.

    Args:
        model_name (str): Name of the model to retrieve TAI values for. Must be one of the keys in MODEL_PATHS.

    Returns:
        pd.DataFrame: DataFrame with 'district_code' and model-specific travel time scaled values.
    """

    MODEL_PATHS = {
        "status_quo_model": "equity-aware-geospatial-ai-project\\src\\evaluation\\data\\processed\\RES_status_quo_model.xlsx",
        "policy_maker_model": "equity-aware-geospatial-ai-project\\src\\evaluation\\data\\processed\\RES_policy_maker_model.xlsx",
        "demand_based_model": "equity-aware-geospatial-ai-project\\src\\evaluation\\data\\processed\\RES_demand_based_model.xlsx",
        "deprivation_aware_model": "equity-aware-geospatial-ai-project\\src\\evaluation\\data\\processed\\RES_deprivation_aware_model.xlsx",
        "accessibility_based_model": "equity-aware-geospatial-ai-project\\src\\evaluation\\data\\processed\\RES_accessibility_based_model.xlsx",
        "main_model": "equity-aware-geospatial-ai-project\\src\\evaluation\\data\\processed\\RES_main_model.xlsx",
    }

    if model_name not in MODEL_PATHS:
        raise ValueError(f"Invalid model name. Choose from: {list(MODEL_PATHS.keys())}")

    path = MODEL_PATHS[model_name]
    df = pd.read_excel(path)
    df["district_code"] = df["district"].apply(lambda d: SAARLAND_AGS.get(d, d))
    df = df[["district_code", "mean_travel_time_mins_scaled"]].rename(
        columns={"mean_travel_time_mins_scaled": model_name}
    )

    return df


# mean_travel_time_scaled_df = get_TAI_scaled_for_model("policy_maker_model")
# print("\n\n✅ Mean Travel Time Scaled DataFrame across all models ✅\n\n")
# print(mean_travel_time_scaled_df)
# print("\n\n")








def accessibility_score():
    """
    Accessibility Score : Summary Statistics of Travel Time in MINUTES
    For each model, computes:
        - Mean travel time
        - Median travel time
        - 95th percentile (P95) travel time
    across the entire state (all sample points), and saves to Excel.
    """

    MODEL_PATHS = {
        "status_quo_model": "equity-aware-geospatial-ai-project\\src\\evaluation\\data\\processed\\RES_status_quo_model_travel_time_from_sample_to_hospital.xlsx",
        "policy_maker_model": "equity-aware-geospatial-ai-project\\src\\evaluation\\data\\processed\\RES_policy_maker_model_travel_time_from_sample_to_hospital.xlsx",
        "demand_based_model": "equity-aware-geospatial-ai-project\\src\\evaluation\\data\\processed\\RES_demand_based_model_travel_time_from_sample_to_hospital.xlsx",
        "deprivation_aware_model": "equity-aware-geospatial-ai-project\\src\\evaluation\\data\\processed\\RES_deprivation_aware_model_travel_time_from_sample_to_hospital.xlsx",
        "accessibility_based_model": "equity-aware-geospatial-ai-project\\src\\evaluation\\data\\processed\\RES_accessibility_based_model_travel_time_from_sample_to_hospital.xlsx",
        "main_model": "equity-aware-geospatial-ai-project\\src\\evaluation\\data\\processed\\RES_main_model_travel_time_from_sample_to_hospital.xlsx",
    }

    results = []

    for model_name, path in MODEL_PATHS.items():
        df = pd.read_excel(path)
        mean_tt = df["travel_time_minutes"].mean()
        median_tt = df["travel_time_minutes"].median()
        p95_tt = df["travel_time_minutes"].quantile(0.95)

        results.append({
            "model": model_name,
            "mean_travel_time_mins": mean_tt,
            "median_travel_time_mins": median_tt,
            "p95_travel_time_mins": p95_tt
        })

    results_df = pd.DataFrame(results).round(2)
    
    # Save to Excel
    output_path = "equity-aware-geospatial-ai-project\\src\\evaluation\\data\\processed\\accessibility_score.xlsx"
    results_df.to_excel(output_path, index=False)

    print("✅ Saved accessibility score (mean, median, p95) to:")
    print(f"📄 {output_path}")

    return results_df



# accessibility_score = accessibility_score()

# print("\n\n✅ Accessibility Score DataFrame ✅\n\n")
# print(accessibility_score)
# print("\n\n")




def generate_sample_points_per_district():
    """
    Generates 5 evenly spread sample points per district assuming potential start points to the nearest hospital location.
    Better than the centroid method as it is more biased.
    """   
        
    # Load your Saarland districts shapefile (adjust path if needed)
    nuts_gdf = gpd.read_file(NUTS_DATA_PATH)

    # Filter Saarland districts (6 districts)
    saarland_districts = nuts_gdf[(nuts_gdf["CNTR_CODE"] == "DE") & (nuts_gdf["NUTS_ID"].str.startswith("DEC"))]

    def generate_evenly_spread_points(polygon, n_points=5):
        """
        Generate n_points inside polygon evenly spread using its bounding box and filtering inside polygon.
        """
        minx, miny, maxx, maxy = polygon.bounds
        
        points = []
        # Create a grid of points, more than needed, then filter inside polygon
        grid_size = int(np.ceil(np.sqrt(n_points)*3))  # oversample grid
        
        xs = np.linspace(minx, maxx, grid_size)
        ys = np.linspace(miny, maxy, grid_size)
        
        candidate_points = [Point(x,y) for x in xs for y in ys]
        # Keep only points inside polygon
        inside_points = [pt for pt in candidate_points if polygon.contains(pt)]
        
        # Pick evenly spaced points from inside points
        if len(inside_points) < n_points:
            # if too few points inside, just return all
            return inside_points
        else:
            indices = np.linspace(0, len(inside_points)-1, n_points, dtype=int)
            return [inside_points[i] for i in indices]

    # Prepare list to collect points data
    points_data = []

    for idx, row in saarland_districts.iterrows():
        district_name = row["NUTS_NAME"]
        polygon = row["geometry"]
        points = generate_evenly_spread_points(polygon, 5)
        for pt in points:
            points_data.append({
                "district": district_name,
                "Lat": pt.y,
                "Lon": pt.x
            })

    # Convert to DataFrame and save CSV
    points_df = pd.DataFrame(points_data)
    points_df.to_excel(SAARLAND_DISTRICTS_SAMPLE_POINTS_PATH, index=False)
    
    print("*"*80)
    print("✅ Saved 5 evenly spread sample points per district")
    




def find_nearest_hospitals(HOSPITAL_DATA):
    """
    Finds and saves the excel for nearest hospital details for each sample point.
    """
    
    # Load sample points and hospital data
    sample_points = pd.read_excel(SAARLAND_DISTRICTS_SAMPLE_POINTS_PATH)
    hospitals = pd.read_excel(HOSPITAL_DATA)

    # Ensure valid coordinates
    sample_points = sample_points.dropna(subset=["Lat", "Lon"])


    # Compute nearest hospital for each sample point
    results = []

    for i, point in sample_points.iterrows():
        point_coord = (point["Lat"], point["Lon"])
        min_dist = float("inf")
        nearest_hospital = None
        nearest_hospital_coord = None

        for j, hospital in hospitals.iterrows():
            hospital_coord = (hospital["Lat"], hospital["Lon"])
            dist = geodesic(point_coord, hospital_coord).kilometers
            if dist < min_dist:
                min_dist = dist
                # nearest_hospital = hospital["Adresse_Name"]
                nearest_hospital_coord = hospital_coord

        results.append({
            "district": point["district"],
            "sample_point_lat": point_coord[0],
            "sample_point_lon": point_coord[1],
            # "nearest_hospital": nearest_hospital,
            "hospital_lat": nearest_hospital_coord[0],
            "hospital_lon": nearest_hospital_coord[1],
            "distance_km": round(min_dist, 3)
        })

    # Save results to Excel
    results_df = pd.DataFrame(results)
    results_df.to_excel(NEAREST_HOSPITALS_TO_SAMPLE_POINTS_PATH, index=False)

    print("*"*80)
    print("✅ Found nearest hospitals to all sample points.")
    


def get_travel_time(MODEL_NAME):
    """
    Gets the travel time by cars in minutes  - FROM a sample point TO the nearest hospital
    """
    
    # Load the sample points + nearest hospital coordinates
    df = pd.read_excel(NEAREST_HOSPITALS_TO_SAMPLE_POINTS_PATH)  # Must include 'district', 'sample_point_lat', 'sample_point_lon', 'hospital_lat', 'hospital_lon'

    results = []

    for idx, row in df.iterrows():
        district = row.get("district", "Unknown")
        nearest_hospital = row.get("nearest_hospital", "Unknown")
        lat1, lon1 = row["sample_point_lat"], row["sample_point_lon"]
        lat2, lon2 = row["hospital_lat"], row["hospital_lon"]

        # Build OSRM URL
        url = f"http://router.project-osrm.org/route/v1/driving/{lon1},{lat1};{lon2},{lat2}?overview=false"

        try:
            r = requests.get(url)
            data = r.json()

            if "routes" in data and len(data["routes"]) > 0:
                duration_sec = data["routes"][0]["duration"]
                distance_m = data["routes"][0]["distance"]

                results.append({
                    "district": district,
                    "sample_lat": lat1,
                    "sample_lon": lon1,
                    "nearest_hospital": nearest_hospital,
                    "hospital_lat": lat2,
                    "hospital_lon": lon2,
                    "travel_time_minutes": round(duration_sec / 60, 2),
                    "travel_distance_km": round(distance_m / 1000, 2)
                })

            else:
                print(f"No route found for row {idx}")
                results.append({
                    "district": district,
                    "sample_lat": lat1,
                    "sample_lon": lon1,
                    "hospital_lat": lat2,
                    "hospital_lon": lon2,
                    "travel_time_minutes": None,
                    "travel_distance_km": None
                })

        except Exception as e:
            print(f"Error at row {idx}: {e}")

        time.sleep(1)  # Be kind to the demo server

    # Save to Excel
    out_df = pd.DataFrame(results)
    out_df.to_excel( f"equity-aware-geospatial-ai-project\\src\\evaluation\\data\\processed\\{MODEL_NAME}_travel_time_from_sample_to_hospital.xlsx", index=False)
    
    print("*"*80)
    print("✅ Generated Travel times from sample points to the nearest hospitals")
    

def calc_metrics_from_travel_time(MODEL_NAME):
    """
    Calculates and saves the MEAN, MEDIAN and 95th PERCENTILE of the travel time for EACH district.
    """

    # Load the data
    df = pd.read_excel(f"equity-aware-geospatial-ai-project\\src\\evaluation\\data\\processed\\{MODEL_NAME}_travel_time_from_sample_to_hospital.xlsx")

    # Group by district and compute metrics
    summary_df = df.groupby("district")["travel_time_minutes"].agg([
        ("mean_travel_time_mins", "mean"),
        ("median_travel_time_mins", "median"),
        ("p95_travel_time_mins", lambda x: x.quantile(0.95))
    ]).reset_index()

    # Min-max scaling
    for col in ["mean_travel_time_mins", "median_travel_time_mins", "p95_travel_time_mins"]:
        min_val = summary_df[col].min()
        max_val = summary_df[col].max()
        summary_df[f"{col}_scaled"] = (summary_df[col] - min_val) / (max_val - min_val)

    # Save result
    # summary_df.to_excel(CALCULATED_METRICS_FOR_TRAVEL_TIME_PATH, index=False)
    summary_df.to_excel(f"equity-aware-geospatial-ai-project\\src\\evaluation\\data\\processed\\{MODEL_NAME}.xlsx", index=False)

    print("*"*80)
    print(f"✅ Calculated statistics of the travel time for EACH district for {MODEL_NAME}")
    
    
def RUN(HOSPITAL_DATA, MODEL_NAME):
    """
    Main function to run the entire pipeline for calculating travel accessibility index in Saarland.
    This function orchestrates the entire process from fetching hospital data to calculating travel times and metrics.
    """
    
    print("*"*80)
    print("Estimated time to run the pipeline ---->  2-3 mins")
    
    # Create all required directories
    required_dirs = [RAW_DIR, OUT_DIR]
    for directory in required_dirs:
        os.makedirs(directory, exist_ok=True)
    
    # pipeline
      
    # get_hospitals_in_saarland()
    
    # coordinates_from_address()
    
  
    generate_sample_points_per_district()
    
    find_nearest_hospitals(HOSPITAL_DATA)
    
    get_travel_time(MODEL_NAME)
    
    calc_metrics_from_travel_time(MODEL_NAME)
    





# ---------------------------------------------------------------------------------------------------------------------------
# Main entry point to run the pipeline 

# if __name__ == "__main__":
#     RUN(POLICY_MAKER_MODEL_PATH, "RES_policy_maker_model")

# Models 
# The models are run separately to get the results for each model.

# 1. STATUS_QUO_MODEL_PATH, "RES_status_quo_model"
# 2. POLICY_MAKER_MODEL_PATH, "RES_policy_maker_model"
# 2. MAIN_MODEL_PATH , "RES_main_model"
# 3. DEMAND_BASED_MODEL_PATH , "RES_demand_based_model"
# 4. DEPRIVATION_AWARE_MODEL_PATH , "RES_deprivation_aware_model"
# 5. ACCESSIBILITY_BASED_MODEL_PATH , "RES_accessibility_based_model"

# ---------------------------------------------------------------------------------------------------------------------------









# ---------------------------------------------------------------------------------------------------------------------------
#                                                        THANK YOU 
#                                                      ~ Piyush Pant 
# ---------------------------------------------------------------------------------------------------------------------------