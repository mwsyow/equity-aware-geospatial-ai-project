import sys
import os

# Add src directory to path for imports
current_dir = os.path.dirname(__file__)
src_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from index_demand_forecast.demand_forecast import forecast_demand_per_district_in_saarland as run_demand_forecast
from index_elderly_share.elderly_share import run as run_elderly_share
from index_gisd.gisd import run as run_gisd
from index_hospital_capacity.hospital_capacity_index_dict import calculate_hospital_capacity_index as run_hospital_capacity_index
from .accessibility_score_metric import get_TAI_scaled_for_model as run_TAI_scaled_for_model
import os
import pandas as pd
# StrEnum compatibility for Python < 3.11
try:
    from enum import StrEnum
except ImportError:
    from enum import Enum
    class StrEnum(str, Enum):
        pass
import numpy as np



SAARLAND_AGS = [
    "10041",  # Regionalverband Saarbrücken
    "10042",  # Merzig-Wadern
    "10043",  # Neunkirchen
    "10044",  # Saarlouis
    "10045",  # Saarpfalz-Kreis
    "10046",  # St. Wendel
]

REGION_CODE = 10  # Saarland Land code  
DATA_PATH = os.path.join(os.path.dirname(__file__), "data")


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
    Index.TRAVEL_TIME: lambda: run_TAI_scaled_for_model("status_quo_model"),  # Default model
    }


def calculate_new_hospital_capacity_index(hospital_file_path: str) -> dict:
    """
    Calculates the Hospital Capacity Index for each district in Saarland.

    The index is calculated using the following steps:
    1. Aggregates total hospital beds per district from model result file
    2. Combines bed data with population data
    3. Calculates adjusted beds per capita
    4. Normalizes and inverts the values to create the final index
    5. Maps district codes to AGS (Amtlicher Gemeindeschlüssel) codes

    The Hospital Capacity Index ranges from 0 to 1, where:
    - Higher values indicate lower hospital capacity relative to population
    - Lower values indicate higher hospital capacity relative to population

    Returns:
        dict: A dictionary mapping AGS codes to Hospital Capacity Index values,
              with values rounded to 4 decimal places
    """
    try:
        # Load model result data directly
        print(f"🏥 Loading hospital capacity from model result: {hospital_file_path}")
        df = pd.read_excel(hospital_file_path, engine="openpyxl")
        
        if 'bed_allocation' not in df.columns:
            print(f"❌ No bed_allocation column found in {hospital_file_path}")
            return {ags: 0.5 for ags in SAARLAND_AGS}  # Return neutral values
        
        # Aggregate total beds per district
        beds_per_district = df.groupby('district_code')['bed_allocation'].sum().reset_index()
        beds_per_district = beds_per_district.rename(columns={'district_code': 'district', 'bed_allocation': 'TotalBeds'})
        
        # Convert 5-digit district codes to 2-digit codes for population data matching
        beds_per_district['district'] = beds_per_district['district'].astype(str).str[-2:].astype(int)
        
        print(f"🔍 Beds per district data:")
        print(f"   Shape: {beds_per_district.shape}")
        print(f"   Columns: {beds_per_district.columns.tolist()}")
        print(f"   District values: {beds_per_district['district'].tolist()}")
        print(f"   Data types: {beds_per_district.dtypes.to_dict()}")
        
        # Load population data
        population_df = load_population_data()
        
        print(f"🔍 Population data:")
        print(f"   Shape: {population_df.shape}")
        print(f"   Columns: {population_df.columns.tolist()}")
        print(f"   District values: {population_df['district'].tolist()}")
        print(f"   Data types: {population_df.dtypes.to_dict()}")
        
        # Merge with population data
        merged_df = pd.merge(beds_per_district, population_df, on="district", how="inner")
        
        print(f"🔍 After merge:")
        print(f"   Shape: {merged_df.shape}")
        print(f"   Columns: {merged_df.columns.tolist()}")
        
        if merged_df.empty:
            print(f"❌ No data after merging with population data")
            print(f"   Beds districts: {beds_per_district['district'].tolist()}")
            print(f"   Population districts: {population_df['district'].tolist()}")
            return {ags: 0.5 for ags in SAARLAND_AGS}
        
        # Compute adjusted beds per capita
        merged_df["AdjBeds"] = merged_df["TotalBeds"] / merged_df["population"]
        
        # Normalize and invert to compute HospitalCapacityIndex
        min_adj = merged_df["AdjBeds"].min()
        max_adj = merged_df["AdjBeds"].max()
        
        if max_adj == min_adj:
            # All districts have same capacity, return neutral values
            merged_df["HospitalCapacityIndex"] = 0.5
        else:
            merged_df["HospitalCapacityIndex"] = 1 - (merged_df["AdjBeds"] - min_adj) / (max_adj - min_adj)
        
        # Ensure HospitalCapacityIndex is numeric and handle any NaN values
        merged_df["HospitalCapacityIndex"] = pd.to_numeric(merged_df["HospitalCapacityIndex"], errors='coerce').fillna(0.5)
        
        # Convert to dictionary: {district: HospitalCapacityIndex}
        result_dict = dict(zip(merged_df["district"], merged_df["HospitalCapacityIndex"].round(4)))
        
        # Map result_dict keys to AGS
        mapped_result = {}
        for ags in SAARLAND_AGS:
            # Try to find the district in the result
            district_code = int(ags[-2:])  # Extract last 2 digits
            
            if district_code in result_dict:
                mapped_result[ags] = result_dict[district_code]
            else:
                mapped_result[ags] = 0.5  # Default neutral value
        
        print(f"✅ Hospital capacity index calculated: {mapped_result}")
        return mapped_result
        
    except Exception as e:
        print(f"❌ Error calculating hospital capacity index: {e}")
        return {ags: 0.5 for ags in SAARLAND_AGS}  # Return neutral values on error



def assemble_indexes() -> pd.DataFrame:
    combined = []
    # Iterate with both key (Index enum) and function
    for name, fn in INDEX_FUNC_MAP.items():
        # 1) sanity check: fn must be callable
        if not callable(fn):
            raise TypeError(f"INDEX_FUNC_MAP[{name!r}] is not callable (got {type(fn)})")

        # 2) optional logging - shows you exactly what's running
        print(f"→ computing index {name!r} using {fn.__name__}")

        # 3) actually call it
        res = fn()

        # 4) wrap scalars-in-dict into lists so DataFrame(res) works
        if isinstance(res, dict):
            if all(not isinstance(v, (list, pd.Series, np.ndarray, pd.DataFrame))
                   for v in res.values()):
                res = {k: [v] for k, v in res.items()}
            df_i = pd.DataFrame(res)

        # 5) if it already returned a DataFrame, extract the relevant column
        elif isinstance(res, pd.DataFrame):
            # For DataFrames, we need to extract the relevant column
            # The TAI function returns a DataFrame with 'district_code' and model name columns
            res_df = res  # Type hint for the linter
            if 'district_code' in res_df.columns:
                # Extract the model-specific column (should be the second column)
                model_col = [col for col in res_df.columns if col != 'district_code'][0]
                # Create a DataFrame with districts as columns (like the dict results)
                df_i = res_df.set_index('district_code')[model_col].to_frame().T
            else:
                # If it's a single-column DataFrame, use it as is
                df_i = res_df

        else:
            # 6) blow up on anything else
            raise TypeError(f"Index function {name!r} returned unsupported type {type(res)}")

        combined.append(df_i)

    # 7) concatenate, transpose, and set column names
    df = pd.concat(combined, axis=0).transpose()
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
    # Read the correct sheet with proper header
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

    return df

def load_population_data() -> pd.DataFrame:
    """
    Loads and processes population data from an Excel file.

    The function performs the following operations:
    1. Dynamically finds the header row in the Excel file
    2. Filters data for the specified region (Saarland)
    3. Extracts relevant columns (district, population)

    Returns:
        pd.DataFrame: A DataFrame containing columns:
            - district (float): District/Kreis code
            - population (int): Population count for the district
    """
    file_path = file_path = os.path.join(DATA_PATH, "District-Population.xlsx")
    xl = pd.ExcelFile(file_path, engine="openpyxl")
    sheet_df = xl.parse(xl.sheet_names[0], header=None)

    # Detect first non-empty row as header
    for i, row in sheet_df.iterrows():
        if row.notna().sum() > 2:
            header_row = i
            break

    # Re-read with correct header
    df = pd.read_excel(file_path, skiprows=header_row, engine="openpyxl")
    df.columns = df.columns.str.strip()

    df = df[df["Land"] == REGION_CODE]
    df = df[["Kreis", "Population"]].rename(columns={"Kreis": "district", "Population": "population"})

    return df



def calculate_new_equity_index(hospital_file_path: str, modelname: str):
    """
    Runs the equity index calculation pipeline with a custom hospital data file.

    Args:
        hospital_file_path (str): Path to the hospital capacity Excel file.
        modelname (str): Name of the model to use for travel time calculations.
        weights (dict): Dictionary mapping Index enum values to their respective weights.

    Returns:
        pd.Series: Equity Index values for each district.
    """
    print(f"🔍 Calculating equity index for model: {modelname}")
    
    try:
        # Define the path to the hospital data file (not the model results file)
        hospital_data_path = os.path.join(os.path.dirname(__file__), "..", "..", "index_hospital_capacity", "data", "Krankenhausverzeichnis_2021.xlsx")
        print(f"🏥 Hospital data path: {hospital_data_path}")
        print(f"🏥 Hospital data exists: {os.path.exists(hospital_data_path)}")

        # Override the INDEX_FUNC_MAP entries with the custom functions
        INDEX_FUNC_MAP[Index.HOSPITAL_CAPACITY] = lambda: calculate_new_hospital_capacity_index(hospital_file_path)
        INDEX_FUNC_MAP[Index.TRAVEL_TIME] = lambda: run_TAI_scaled_for_model(modelname)

        print("📊 Assembling indexes...")
        index_df = assemble_indexes()
        print(f"📊 Indexes assembled. Shape: {index_df.shape}")
        print(f"📊 Indexes columns: {list(index_df.columns)}")
        print(f"📊 Indexes head:\n{index_df.head()}")
        
        weight = {
            Index.FORECAST_DEMAND: 0.25,
            Index.ELDERLY_SHARE: 0.25,
            Index.GISD: 0.25,
            Index.HOSPITAL_CAPACITY: 0.25,
            Index.TRAVEL_TIME: 0.25,
            Index.ACCESSIBILITY: 0.25,
        }

        print("⚖️ Calculating equity index...")
        result = equity_index(index_df, weight)
        print(f"✅ Equity index calculated: {result}")
        return result
        
    except Exception as e:
        print(f"❌ Error calculating equity index for {modelname}: {e}")
        import traceback
        traceback.print_exc()
        return pd.Series([np.nan] * len(SAARLAND_AGS), index=SAARLAND_AGS, name="EquityIndex")