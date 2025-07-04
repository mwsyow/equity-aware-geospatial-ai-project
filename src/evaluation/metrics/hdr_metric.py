import sys
import os

# Add src directory to path for imports
current_dir = os.path.dirname(__file__)
src_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from index_demand_forecast.demand_forecast import df_saarland_diseases_history as loading_hospital_inpatients_per_district
import pandas as pd


SAARLAND_AGS = {
    "Regionalverband Saarbrücken": "10041",
    "Merzig-Wadern": "10042",
    "Neunkirchen": "10043",
    "Saarlouis": "10044",
    "Saarpfalz-Kreis": "10045",
    "St. Wendel": "10046"
}

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


def load_hospital_inpatient_data() -> pd.DataFrame:
    """
    Loads and processes hospital inpatient data from an Excel file.

    The function performs the following operations:
    1. Dynamically finds the header row in the Excel file
    2. Filters data for the specified region (Saarland)
    3. Extracts relevant columns (year, region, district, value)

    Returns:
        pd.DataFrame: A DataFrame containing columns:
            - year (int): Year of the data
            - region (int): Region/Land code
            - district (float): District/Kreis code
            - value (int): Number of hospital inpatients
    """
    
    hospital_inpatient_df = loading_hospital_inpatients_per_district()
    # Use 2021 as the year column (since columns are integers)
    hospital_inpatient_df_2021 = hospital_inpatient_df[2021]
    return hospital_inpatient_df_2021.reset_index().rename(columns={"district_code": "district", 2021: "value"})



def calculate_hdr(hospital_file_path: str):
    try:
        # Load the data
        hospital_data = load_hospital_data(hospital_file_path)
        hospital_inpatient_data = load_hospital_inpatient_data()

        # Check if we have valid data
        if hospital_data.empty:
            print(f"❌ No hospital data found for {hospital_file_path}")
            # Return neutral values for all districts
            return pd.Series([0.5] * len(SAARLAND_AGS), index=list(SAARLAND_AGS.keys()))

        # Convert district codes to string for consistency
        hospital_data["district"] = hospital_data["district"].astype(str)
        hospital_inpatient_data["district"] = hospital_inpatient_data["district"].astype(str)

        # Filter only Saarland districts
        saarland_districts = set(SAARLAND_AGS.values())
        hospital_data = hospital_data[hospital_data["district"].isin(saarland_districts)]
        hospital_inpatient_data = hospital_inpatient_data[hospital_inpatient_data["district"].isin(saarland_districts)]

        if hospital_data.empty:
            print(f"❌ No Saarland district data found in {hospital_file_path}")
            return pd.Series([0.5] * len(SAARLAND_AGS), index=list(SAARLAND_AGS.keys()))

        # Calculate total beds and inpatients per district
        beds_by_district = hospital_data.groupby("district")["beds"].sum()
        inpatients_by_district = hospital_inpatient_data.groupby("district")["value"].sum()

        # Compute the HDR ratio
        hdr_ratio = beds_by_district / inpatients_by_district

        # Reindex and sort by SAARLAND_AGS order
        ordered_district_codes = list(SAARLAND_AGS.values())
        hdr_ratio = hdr_ratio.reindex(ordered_district_codes)

        # Fill NaN values with neutral value
        hdr_ratio = hdr_ratio.fillna(0.5)

        # Optional: rename the index to district names
        hdr_ratio.index = [name for name in SAARLAND_AGS]

        print("Ratio (Total Beds / Hospital Inpatients) per district:")
        print(hdr_ratio)

        return hdr_ratio
        
    except Exception as e:
        print(f"❌ Error calculating HDR: {e}")
        return pd.Series([0.5] * len(SAARLAND_AGS), index=list(SAARLAND_AGS.keys()))
    
   #here we load the function to load the hospital inpatient data