#! python src/travel_accessibility_index/TAI.py

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


BASE_DIR = os.path.dirname(__file__)
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
OUT_DIR  = os.path.join(BASE_DIR, "data", "processed")
MAP_DIR  = os.path.join(OUT_DIR, "maps")

# MUST HAVE
HOSPITAL_DATA_PATH = os.path.join(RAW_DIR, "Krankenhausverzeichnis_2021.xlsx" )
NUTS_DATA_PATH = os.path.join(RAW_DIR, "NUTS_RG_01M_2021_4326_LEVL_3.shp")


SAARLAND_HOSPITALS_PATH = os.path.join(OUT_DIR, "saarland_hospitals.xlsx") 
SAARLAND_HOSPITALS_WITH_COORDINATES_PATH = os.path.join(OUT_DIR, "saarland_hospitals_with_coords.xlsx") 
SAARLAND_DISTRICTS_SAMPLE_POINTS_PATH = os.path.join(OUT_DIR, "saarland_districts_sample_points.xlsx") 
NEAREST_HOSPITALS_TO_SAMPLE_POINTS_PATH = os.path.join(OUT_DIR, "nearest_hospitals_to_sample_points.xlsx") 
TRAVEL_TIME_FROM_SAMPLE_TO_HOSPITAL_PATH = os.path.join(OUT_DIR, "travel_time_from_sample_to_hospital.xlsx") 
CALCULATED_METRICS_FOR_TRAVEL_TIME_PATH = os.path.join(OUT_DIR, "calculated_metrics_from_travel_time.xlsx") 


SAARLAND_MAP_WITH_BORDERS_AND_HOSPITALS_PATH = os.path.join(MAP_DIR , "saarland_map_with_borders_and_hospitals.html")
SAARLAND_SAMPLE_POINTS_MAP_PATH = os.path.join(MAP_DIR , "saarland_sample_points_map.html")
SAARLAND_HOSPITALS_AND_SAMPLE_POINTS_MAP_PATH = os.path.join(MAP_DIR , "saarland_hospital_and_sample_points_map.html")
NEAREST_HOSPITALS_TO_SAMPLE_POINTS_MAP_PATH = os.path.join(MAP_DIR , "nearest_hospitals_to_sample_points_map.html")


SAARLAND_AGS = {
    "Regionalverband Saarbrücken": "10041",
    "Merzig-Wadern": "10042",
    "Neunkirchen": "10043",
    "Saarlouis": "10044",
    "Saarpfalz-Kreis": "10045",
    "St. Wendel": "10046"
}


# MUST HAVE DATASETS TO RUN THIS PIPELINE   ------> ./data/raw/filename
# 1. Krankenhausverzeichnis_2021.xlsx
# 2. NUTS_RG_01M_2021_4326_LEVL_3.shp

# -----------------------------------------------------------------------------------------



def get_hospitals_in_saarland():
    """
    Saves the excel containing saarland hospital data
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        df = pd.read_excel(HOSPITAL_DATA_PATH, sheet_name="KHV_2021", header=4)

    # Saarland has Land code '10' (German state codes)
    df_saarland = df[df['Land'] == 10]

    df_saarland.to_excel(SAARLAND_HOSPITALS_PATH, index=False)

    print("*"*80)
    print("✅ Saved Saarland Hospital Dataset")
    
    

def coordinates_from_address():
    """
    Saves Coordinates of Hospital addresses along with the address in an excel file
    """
    
    # Load filtered Saarland hospitals Excel file
    df = pd.read_excel(SAARLAND_HOSPITALS_PATH)

    # Create full address string
    df["Full_Address"] = (
        df["Adresse_Strasse_Standort"].astype(str) + " " +
        df["Adresse_Haus-Nr._Standort"].astype(str) + ", " +
        df["Adresse_Postleitzahl_Standort"].astype(str) + " " +
        df["Adresse_Ort_Standort"].astype(str) + ", Germany"
    )

    # Initialize geocoder
    geolocator = Nominatim(user_agent="saarland_hospital_locator")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

    # Get coordinates
    df["location"] = df["Full_Address"].apply(geocode)
    df["latitude"] = df["location"].apply(lambda loc: loc.latitude if loc else None)
    df["longitude"] = df["location"].apply(lambda loc: loc.longitude if loc else None)

    df.to_excel(SAARLAND_HOSPITALS_WITH_COORDINATES_PATH, index=False)

    df = pd.read_excel(SAARLAND_HOSPITALS_WITH_COORDINATES_PATH)

    # SINCE THE FREE API IS NOT GETTING THE COORDINATES OF THESE HOSPITALS, THUS ADDING THEM MANUALLY
    # THE GOOGLE MAPS API (PAID) CAN GET THEIR COORDINATES BUT IT IS LIMITED
    df.loc[df["Full_Address"] == "Friedenstraße 2, 66822 Lebach, Germany", ["latitude", "longitude"]] = [49.412574, 6.909871]
    df.loc[df["Full_Address"] == "Kirrberger Straße nan, 66421 Homburg, Germany", ["latitude", "longitude"]] = [49.308199, 7.351777]
    df.loc[df["Full_Address"] == "Hospitalhof nan, 66606 St. Wendel, Germany", ["latitude", "longitude"]] = [49.454659, 7.186793]

    # Save updated file
    df.to_excel(SAARLAND_HOSPITALS_WITH_COORDINATES_PATH, index=False)


    print("*"*80)
    print("✅ Saved Hospital Coordinates Dataset")

    
def map_hospitals():
    """
    Saves map that shows all the hospitals in Saarland
    """

    # Load hospital data
    df = pd.read_excel(SAARLAND_HOSPITALS_WITH_COORDINATES_PATH)

    # Saarland center
    saarland_center = [49.3964, 7.0220]

    # Create map
    m = folium.Map(location=saarland_center, zoom_start=10, tiles="OpenStreetMap")
    # CartoDB positron, OpenStreetMap

    # Add hospitals
    for _, row in df.iterrows():
        if pd.notnull(row["latitude"]) and pd.notnull(row["longitude"]):
            popup = folium.Popup(f"{row['Adresse_Name']}", max_width=250)
            folium.Marker(
                location=[row["latitude"], row["longitude"]],
                popup=popup,
                icon=folium.Icon(color="red", icon="plus-sign")
            ).add_to(m)

    # Load NUTS shapefile
    nuts_gdf = gpd.read_file(NUTS_DATA_PATH)

    # Filter Saarland districts (NUTS-3 = DECxx)
    districts = nuts_gdf[nuts_gdf["NUTS_ID"].str.startswith("DEC")]

    # Get Saarland state boundary from NUTS-2 (DEC0)
    saarland_state = nuts_gdf[nuts_gdf["NUTS_ID"] == "DEC0"]

    # Ensure CRS is WGS84
    districts = districts.to_crs(epsg=4326)
    saarland_state = saarland_state.to_crs(epsg=4326)

    # Add Saarland state border
    folium.GeoJson(
        saarland_state,
        name="Saarland State Border",
        style_function=lambda x: {
            'fillColor': 'none',
            'color': 'red',
            'weight': 3,
            'dashArray': '5, 5'
        }
    ).add_to(m)

    # Add district borders
    folium.GeoJson(
        districts,
        name="Saarland Districts",
        style_function=lambda x: {
            'fillColor': 'none',
            'color': 'black',
            'weight': 3,
            'opacity': 0.7
        },
        tooltip=folium.GeoJsonTooltip(fields=["NUTS_NAME"])
    ).add_to(m)

    # Add layer control
    folium.LayerControl().add_to(m)

    # Save map
    m.save(SAARLAND_MAP_WITH_BORDERS_AND_HOSPITALS_PATH)
    
    print("*"*80)
    print("✅ Map saved with Saarland district borders and Hospitals!")
    




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
                "latitude": pt.y,
                "longitude": pt.x
            })

    # Convert to DataFrame and save CSV
    points_df = pd.DataFrame(points_data)
    points_df.to_excel(SAARLAND_DISTRICTS_SAMPLE_POINTS_PATH, index=False)
    
    print("*"*80)
    print("✅ Saved 5 evenly spread sample points per district")
    


def map_saarland_districts_sample_points():
    """
    Generates the map showing all sample points per district
    """        

    # Load sample points CSV
    points_df = pd.read_excel(SAARLAND_DISTRICTS_SAMPLE_POINTS_PATH)

    # Saarland center coordinates
    saarland_center = [49.3964, 7.0220]

    # Create folium map with CartoDB positron style
    m = folium.Map(location=saarland_center, zoom_start=10, tiles="OpenStreetMap")

    # Add sample points as blue circle markers with popup of district name
    for _, row in points_df.iterrows():
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=5,
            color='blue',
            fill=True,
            fill_opacity=0.7,
            popup=f"District: {row['district']}"
        ).add_to(m)

    # Load Saarland district borders shapefile
    nuts_gdf = gpd.read_file(NUTS_DATA_PATH)
    saarland_districts = nuts_gdf[(nuts_gdf["CNTR_CODE"] == "DE") & (nuts_gdf["NUTS_ID"].str.startswith("DEC"))]

    # Add Saarland district borders to map
    folium.GeoJson(
        saarland_districts,
        name="Saarland Districts",
        style_function=lambda x: {
            'fillColor': 'none',
            'color': 'black',
            'weight': 3,
            'opacity': 0.7
        },
        tooltip=folium.GeoJsonTooltip(fields=["NUTS_NAME"])
    ).add_to(m)

    # Add layer control
    folium.LayerControl().add_to(m)

    # Save and show map
    m.save(SAARLAND_SAMPLE_POINTS_MAP_PATH)
    
    print("*"*80)
    print("✅ Map saved with sample points per district in Saarland")



def map_hospitals_and_sample_points():
    """
    Generates map with hospitals and sample points in Saarland
    """
    
    # Load sample points
    points_df = pd.read_excel(SAARLAND_DISTRICTS_SAMPLE_POINTS_PATH)

    # Load hospital data
    hospitals_df = pd.read_excel(SAARLAND_HOSPITALS_WITH_COORDINATES_PATH)

    # Load Saarland district borders (NUTS level 3)
    nuts_gdf = gpd.read_file(NUTS_DATA_PATH)
    saarland_districts = nuts_gdf[(nuts_gdf["CNTR_CODE"] == "DE") & (nuts_gdf["NUTS_ID"].str.startswith("DEC"))]

    # Map center
    saarland_center = [49.3964, 7.0220]
    m = folium.Map(location=saarland_center, zoom_start=10, tiles="OpenStreetMap")

    # Add hospitals (red markers)
    for _, row in hospitals_df.iterrows():
        if pd.notnull(row["latitude"]) and pd.notnull(row["longitude"]):
            popup = folium.Popup(f"{row['Adresse_Name']}", max_width=250)
            folium.Marker(
                location=[row["latitude"], row["longitude"]],
                popup=popup,
                icon=folium.Icon(color="red", icon="plus-sign")
            ).add_to(m)

    # Add sample points (blue circle markers)
    for _, row in points_df.iterrows():
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=5,
            color='blue',
            fill=True,
            fill_opacity=0.7,
            popup=f"District: {row['district']}"
        ).add_to(m)

    # Add district borders
    folium.GeoJson(
        saarland_districts,
        name="Saarland Districts",
        style_function=lambda x: {
            'fillColor': 'none',
            'color': 'black',
            'weight': 2,
            'opacity': 0.8
        },
        tooltip=folium.GeoJsonTooltip(fields=["NUTS_NAME"])
    ).add_to(m)

    # Add layer control
    folium.LayerControl().add_to(m)

    # Save map
    m.save(SAARLAND_HOSPITALS_AND_SAMPLE_POINTS_MAP_PATH )

    print("*"*80)
    print("✅ Map saved with hospitals and sample points per district in Saarland")
    


def find_nearest_hospitals():
    """
    Finds and saves the excel for nearest hospital details for each sample point.
    """
    
    # Load sample points and hospital data
    sample_points = pd.read_excel(SAARLAND_DISTRICTS_SAMPLE_POINTS_PATH)
    hospitals = pd.read_excel(SAARLAND_HOSPITALS_WITH_COORDINATES_PATH)

    # Ensure valid coordinates
    sample_points = sample_points.dropna(subset=["latitude", "longitude"])


    # Compute nearest hospital for each sample point
    results = []

    for i, point in sample_points.iterrows():
        point_coord = (point["latitude"], point["longitude"])
        min_dist = float("inf")
        nearest_hospital = None
        nearest_hospital_coord = None

        for j, hospital in hospitals.iterrows():
            hospital_coord = (hospital["latitude"], hospital["longitude"])
            dist = geodesic(point_coord, hospital_coord).kilometers
            if dist < min_dist:
                min_dist = dist
                nearest_hospital = hospital["Adresse_Name"]
                nearest_hospital_coord = hospital_coord

        results.append({
            "district": point["district"],
            "sample_point_lat": point_coord[0],
            "sample_point_lon": point_coord[1],
            "nearest_hospital": nearest_hospital,
            "hospital_lat": nearest_hospital_coord[0],
            "hospital_lon": nearest_hospital_coord[1],
            "distance_km": round(min_dist, 3)
        })

    # Save results to Excel
    results_df = pd.DataFrame(results)
    results_df.to_excel(NEAREST_HOSPITALS_TO_SAMPLE_POINTS_PATH, index=False)

    print("*"*80)
    print("✅ Found nearest hospitals to all sample points.")
    

def map_nearest_hospitals_to_sample_points():
    """
    Generates map connecting sample points and nearest hospitals
    """
    
    # Load sample point to nearest hospital data
    samples = pd.read_excel(NEAREST_HOSPITALS_TO_SAMPLE_POINTS_PATH)

    # Load full hospital list (to show all hospitals, not just nearest ones)
    all_hospitals = pd.read_excel(SAARLAND_HOSPITALS_WITH_COORDINATES_PATH)
    all_hospitals = all_hospitals.dropna(subset=["latitude", "longitude"])

    # Load district borders
    districts = gpd.read_file(NUTS_DATA_PATH)
    districts = districts[districts["CNTR_CODE"] == "DE"]
    districts = districts[districts["NUTS_ID"].str.startswith("DEC")]

    # Saarland center
    saarland_center = [49.3964, 7.0220]

    # Create folium map
    m = folium.Map(location=saarland_center, zoom_start=10, tiles="OpenStreetMap")

    # Add district borders
    folium.GeoJson(
        districts,
        name="Saarland Districts",
        style_function=lambda x: {
            'fillColor': 'none',
            'color': 'black',
            'weight': 3,
            'opacity': 0.7
        },
        tooltip=folium.GeoJsonTooltip(fields=["NUTS_NAME"])
    ).add_to(m)

    # Add ALL hospitals (from full list)
    for _, row in all_hospitals.iterrows():
        hosp_coord = (row["latitude"], row["longitude"])
        folium.Marker(
            location=hosp_coord,
            icon=folium.Icon(color="red", icon="plus-sign"),
            popup=row["Adresse_Name"]
        ).add_to(m)

    # Add sample points and connecting lines to their nearest hospital
    for _, row in samples.iterrows():
        sp_coord = (row["sample_point_lat"], row["sample_point_lon"])
        hosp_coord = (row["hospital_lat"], row["hospital_lon"])

        # Sample point marker
        folium.CircleMarker(
            location=sp_coord,
            radius=5,
            color="blue",
            fill=True,
            fill_opacity=1,
            popup=f"Sample Point ({row['district']})"
        ).add_to(m)

        # Line connecting sample point to nearest hospital
        folium.PolyLine(
            locations=[sp_coord, hosp_coord],
            color="blue",
            weight=2,
            dash_array="5,5",
            tooltip=f"{row['distance_km']} km"
        ).add_to(m)

    # Add layer control
    folium.LayerControl().add_to(m)

    # Save map
    m.save(NEAREST_HOSPITALS_TO_SAMPLE_POINTS_MAP_PATH)

    print("*"*80)
    print("✅ Map saved connecting sample points to nearest hospitals")
    

def get_travel_time():
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
    out_df.to_excel(TRAVEL_TIME_FROM_SAMPLE_TO_HOSPITAL_PATH, index=False)
    
    print("*"*80)
    print("✅ Generated Travel times from sample points to the nearest hospitals")
    

def calc_metrics_from_travel_time():
    """
    Calculates and saves the MEAN, MEDIAN and 95th PERCENTILE of the travel time for EACH district.
    """

    # Load the data
    df = pd.read_excel(TRAVEL_TIME_FROM_SAMPLE_TO_HOSPITAL_PATH)

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
    summary_df.to_excel(CALCULATED_METRICS_FOR_TRAVEL_TIME_PATH, index=False)
    
    print("*"*80)
    print("✅ Calculated MEAN, MEDIAN and 95th PERCENTILE of the Travel time for EACH district")
    
    
def RUN():
    """
    Workflow ---> Travel Accessibility Index  #
    
    1. Gets the hospitals in Saarland (in 2021) from the main dataset
    2. Gets coordinates for each of the hospital in Saarland
    3. Generates 5 evenly spread sample points per district (which are potential starting points to the nearest hospitals)
    4. Finds the nearest hospital to each sample point (allows cross-district hospitals if they are near)
    5. Finds the Travel Time by car in minutes - FROM Sample point TO nearest Hospital
    6. Calculates MEAN, MEDIAN, 95th PERCENTILE of the Travel Time
    7. Uses scaled (0-1) MEAN for each district as the TAI = average travel time to the nearest hospital per district

    PS - Generates various MAPS for a more realisitc approach and better visualization.
    
    RETURNS A DICTIONARY WITH TAI SCALED SCORES FOR EACH DISTRICT.
    """
    
    print("*"*80)
    print("Estimated time to run the pipeline ---->  2-3 mins")
    
    # Create all required directories
    required_dirs = [RAW_DIR, OUT_DIR, MAP_DIR]
    for directory in required_dirs:
        os.makedirs(directory, exist_ok=True)
    
    # pipeline
      
    get_hospitals_in_saarland()
    
    coordinates_from_address()
    
    generate_sample_points_per_district()
    
    find_nearest_hospitals()
    
    get_travel_time()
    
    calc_metrics_from_travel_time()
    
    
    # generate all MAPS
    
    map_hospitals()
    
    map_saarland_districts_sample_points()
    
    map_hospitals_and_sample_points()
    
    map_nearest_hospitals_to_sample_points()
    
    
    # Returning the final Travel Accessbility Index per district in uniform order
    # Average travel time to the nearest hospital per district
    
    df = pd.read_excel(CALCULATED_METRICS_FOR_TRAVEL_TIME_PATH)
    
    TAI = {
        SAARLAND_AGS[row["district"]]: row["mean_travel_time_mins_scaled"]
        for _, row in df.iterrows()
    }
    
    print("*"*80)
    print("✅ Pipeline Run Successful! ")
    print("*"*80)
    print(f"TAI = {TAI}")
    
    return TAI



if __name__ == "__main__":
    RUN()


# HIGH VALUE = WORSE ACCESSIBILITY = MORE AVG. TIME TO HOSPITAL

#* RESULTS : BELOW ARE THE FINAL TRAVEL ACCESSIBILITY INDEX (TAI). WE ARE USING SCALED MEAN TRAVEL TIME

# TAI = {
#     '10041': 0.0, 
#     '10042': 0.771978021978022, 
#     '10043': 0.5515873015873013, 
#     '10044': 0.7118437118437116, 
#     '10045': 1.0, 
#     '10046': 0.996031746031746
# }














# ---------------------------------------------------------------------------------------------------------------------------
#                                                        THANK YOU 
#                                                      ~ Piyush Pant 
# ---------------------------------------------------------------------------------------------------------------------------