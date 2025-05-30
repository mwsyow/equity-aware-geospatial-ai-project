import os
import time
import requests
import pandas as pd
import geopandas as gpd

BASE_DIR = os.path.dirname(__file__)
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
NUTS_DATA_PATH = os.path.join(RAW_DIR, "NUTS_RG_01M_2021_4326_LEVL_3.shp")
HOSPITAL_DATA_PATH = os.path.join(RAW_DIR, "Krankenhausverzeichnis_2021.xlsx" )
HOSPITAL_WITH_COORDS_DATA_PATH = os.path.join(RAW_DIR, "saarland_hospitals_with_coords.xlsx" )


DISTRICT_TO_AGS = {
    "Regionalverband Saarbrücken": "10041",
    "Merzig-Wadern": "10042",
    "Neunkirchen": "10043",
    "Saarlouis": "10044",
    "Saarpfalz-Kreis": "10045",
    "St. Wendel": "10046", 
}


def get_centroids():
    """
    Returns a Dictionary containing AGS code with centroids (lat, long) in coordinates format.
    """
    
    # Load NUTS shapefile
    nuts = gpd.read_file(NUTS_DATA_PATH)

    # Filter only Saarland districts using NUTS_NAME
    nuts_saarland = nuts[nuts["NUTS_NAME"].isin(DISTRICT_TO_AGS.keys())].copy()

    # Map AGS codes
    nuts_saarland["AGS"] = nuts_saarland["NUTS_NAME"].map(DISTRICT_TO_AGS)

    # Compute centroids
    nuts_saarland["centroid"] = nuts_saarland.geometry.centroid

    # Construct the centroid dictionary
    centroid_dict = {
        row["AGS"]: {"lat": row["centroid"].y, "lon": row["centroid"].x}
        for _, row in nuts_saarland.iterrows()
    }
    
    return centroid_dict



def get_hospital_df():
    """
    Returns a Dataframe with Hospital_address, Lat, Lon, INSG.
    Missing INSG values are filled with the AVERAGE INSG. 
    """
    df = pd.read_excel(HOSPITAL_WITH_COORDS_DATA_PATH)

    hospitals_df = pd.DataFrame({
        "SiteID": [f"H{i+1}" for i in range(len(df))],  # Create hospital IDs like H1, H2, ...
        "HospitalAddress": df["Full_Address"],
        "Lon": df["longitude"],
        "Lat": df["latitude"],
        "MaxBeds": df["INSG"]
    })

    # Calculate average of INSG ignoring NaNs
    insg_avg = round(hospitals_df["MaxBeds"].mean())

    # Fill NaN values with the average
    hospitals_df["MaxBeds"] = hospitals_df["MaxBeds"].fillna(insg_avg)

    return hospitals_df


def get_travel_time_matrix():
    """
    Returns a nested dict: {HospitalID: {DistrictAGS: travel_time_in_minutes}}
    Uses hospital coordinates and district centroids, queries OSRM API for driving times.
    """
    centroid_dict = get_centroids()
    hospitals_df = get_hospital_df()

    travel_time_dict = {}

    for i, hosp_row in hospitals_df.iterrows():
        hosp_id = hosp_row["SiteID"]
        hosp_lon = hosp_row["Lon"]
        hosp_lat = hosp_row["Lat"]

        travel_time_dict[hosp_id] = {}

        for ags, centroid in centroid_dict.items():
            dest_lon = centroid["lon"]
            dest_lat = centroid["lat"]

            url = f"http://router.project-osrm.org/route/v1/driving/{hosp_lon},{hosp_lat};{dest_lon},{dest_lat}?overview=false"
            try:
                response = requests.get(url)
                data = response.json()
                # Duration in seconds, convert to minutes
                travel_time = data["routes"][0]["duration"] / 60
                travel_time_dict[hosp_id][ags] = round(travel_time)
            except Exception as e:
                print(f"Error fetching OSRM data for hospital {hosp_id} to district {ags}: {e}")
                travel_time_dict[hosp_id][ags] = None
            
            time.sleep(0.1)  # polite delay to avoid API overload

    return travel_time_dict

#! RESULTS

# CENTROIDS
# {
#     '10041': {'lat': 49.25139128171354, 'lon': 6.957087371708875}, 
#     '10042': {'lat': 49.495749727139255, 'lon': 6.680204517952003}, 
#     '10043': {'lat': 49.375798012926424, 'lon': 7.117022598583442}, 
#     '10044': {'lat': 49.35546377690791, 'lon': 6.775424811890773}, 
#     '10045': {'lat': 49.24875499039272, 'lon': 7.242227630019494}, 
#     '10046': {'lat': 49.51960704059241, 'lon': 7.100470861356751}
# }



# HOSPITAL_DF

#  Hospital_address       Lon        Lat    INSG
# 0   Klosterstr. 14, 66125 Saarbrücken-Dudweiler, G...  7.041318  49.277066   157.0
# 1             Rheinstr. 2, 66113 Saarbrücken, Germany  6.959970  49.247555   441.0
# 2     Bahnhofstraße 76-78, 66111 Saarbrücken, Germany  6.992591  49.237085     2.0
# 3            Winterberg 1, 66119 Saarbrücken, Germany  6.994143  49.221221   571.0
# 4           Lahnstraße 19, 66113 Saarbrücken, Germany  6.963554  49.246999   277.0
# 5     Sonnenbergstraße 10, 66119 Saarbrücken, Germany  7.011476  49.200393   369.0
# 6   Karlstr. 67, 66125 Saarbrücken-Herrensohr, Ger...  7.009256  49.273860     6.0
# 7   Großherzog-Friedrich-Straße 44, 66111 Saarbrüc...  7.003564  49.232725   124.0
# 8     Waldstraße 40, 66287 Kleinblittersdorf, Germany  7.048336  49.153009    26.0
# 9          In der Humes 35, 66346 Püttlingen, Germany  6.870121  49.276946   405.0
# 10          An der Klinik 10, 66280 Sulzbach, Germany  7.054714  49.297734   298.0
# 11       Richardstraße 5-9, 66333 Völklingen, Germany  6.852541  49.257259   411.0
# 12          Trierer Straße 148, 66663 Merzig, Germany  6.631578  49.457186   298.0
# 13         Saaruferstraße 10, 66693 Mettlach, Germany  6.596771  49.495769    31.0
# 14            Kräwigstraße 2-6, 66687 Wadern, Germany  6.889340  49.537868   277.0
# 15       Brunnenstraße 20, 66538 Neunkirchen, Germany  7.183021  49.340959   309.0
# 16          Klinikweg 1-5, 66539 Neunkirchen, Germany  7.227015  49.317756   236.0
# 17  Theodor-Fliedner-Straße 12, 66538 Neunkirchen,...  7.185803  49.348293   120.0
# 18             Heeresstraße 49, 66822 Lebach, Germany  6.889790  49.419092   208.0
# 19             Friedenstraße 2, 66822 Lebach, Germany  6.909871  49.412574   277.0
# 20          Vaubanstraße 25, 66740 Saarlouis, Germany  6.744279  49.313649   223.0
# 21       Kapuzinerstrasse 4, 66740 Saarlouis, Germany  6.756487  49.315236   411.0
# 22    Orannastraße 55, 66802 Überherrn-Berus, Germany  6.682970  49.266127    32.0
# 23      Hospitalstraße 5, 66798 Wallerfangen, Germany  6.714977  49.328624   111.0
# 24      Kirrberger Straße nan, 66421 Homburg, Germany  7.351777  49.308199  1335.0
# 25  Klaus-Tussing-Straße 1, 66386 St. Ingbert, Ger...  7.113175  49.290552   181.0
# 26        Am Hirschberg 1a, 66606 St. Wendel, Germany  7.178492  49.453979   346.0
# 27         Hospitalhof nan, 66606 St. Wendel, Germany  7.186793  49.454659   277.0



# TRAVEL TIME MATRIX
#
# {
#     'H1': 
#     {'10041': 17, '10042': 65, '10043': 25, '10044': 27, '10045': 26, '10046': 41}, 
#     'H2': {'10041': 2, '10042': 58, '10043': 30, '10044': 27, '10045': 30, '10046': 39}, 
#     'H3': {'10041': 9, '10042': 59, '10043': 28, '10044': 28, '10045': 28, '10046': 44}, 
#     'H4': {'10041': 17, '10042': 66, '10043': 37, '10044': 35, '10045': 28, '10046': 53}, 
#     'H5': {'10041': 4, '10042': 60, '10043': 30, '10044': 27, '10045': 31, '10046': 39}, 
#     'H6': {'10041': 16, '10042': 64, '10043': 35, '10044': 34, '10045': 28, '10046': 51}, 
#     'H7': {'10041': 18, '10042': 66, '10043': 26, '10044': 29, '10045': 29, '10046': 43}, 
#     'H8': {'10041': 11, '10042': 60, '10043': 30, '10044': 29, '10045': 25, '10046': 47}, 
#     'H9': {'10041': 24, '10042': 73, '10043': 43, '10044': 42, '10045': 33, '10046': 60}, 
#     'H10': {'10041': 20, '10042': 57, '10043': 36, '10044': 20, '10045': 42, '10046': 45}, 
#     'H11': {'10041': 18, '10042': 61, '10043': 21, '10044': 24, '10045': 25, '10046': 38}, 
#     'H12': {'10041': 15, '10042': 53, '10043': 39, '10044': 22, '10045': 36, '10046': 50}, 
#     'H13': {'10041': 36, '10042': 30, '10043': 46, '10044': 19, '10045': 55, '10046': 56}, 
#     'H14': {'10041': 40, '10042': 25, '10043': 50, '10044': 24, '10045': 60, '10046': 52}, 
#     'H15': {'10041': 41, '10042': 36, '10043': 47, '10044': 37, '10045': 59, '10046': 27}, 
#     'H16': {'10041': 25, '10042': 63, '10043': 14, '10044': 26, '10045': 21, '10046': 31}, 
#     'H17': {'10041': 27, '10042': 65, '10043': 18, '10044': 28, '10045': 19, '10046': 36}, 
#     'H18': {'10041': 27, '10042': 65, '10043': 12, '10044': 27, '10045': 23, '10046': 29}, 
#     'H19': {'10041': 32, '10042': 44, '10043': 38, '10044': 22, '10045': 48, '10046': 33}, 
#     'H20': {'10041': 28, '10042': 45, '10043': 34, '10044': 21, '10045': 44, '10046': 30}, 
#     'H21': {'10041': 24, '10042': 40, '10043': 36, '10044': 9, '10045': 44, '10046': 46}, 
#     'H22': {'10041': 25, '10042': 43, '10043': 37, '10044': 12, '10045': 46, '10046': 47}, 
#     'H23': {'10041': 31, '10042': 51, '10043': 47, '10044': 21, '10045': 52, '10046': 57}, 
#     'H24': {'10041': 25, '10042': 41, '10043': 37, '10044': 10, '10045': 45, '10046': 47}, 
#     'H25': {'10041': 36, '10042': 74, '10043': 28, '10044': 37, '10045': 19, '10046': 46}, 
#     'H26': {'10041': 25, '10042': 63, '10043': 20, '10044': 26, '10045': 20, '10046': 37}, 
#     'H27': {'10041': 40, '10042': 72, '10043': 20, '10044': 43, '10045': 42, '10046': 17}, 
#     'H28': {'10041': 41, '10042': 72, '10043': 21, '10044': 43, '10045': 43, '10046': 16}
# }

