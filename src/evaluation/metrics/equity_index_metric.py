from index_demand_forecast.demand_forecast import forecast_demand_per_district_in_saarland as run_demand_forecast
from index_elderly_share.elderly_share import run as run_elderly_share
from index_gisd.gisd import run as run_gisd
from index_hospital_capacity.hospital_capacity_index_dict import calculate_hospital_capacity_index as run_hospital_capacity_index
from accessibility_score_metric import get_TAI_scaled_for_model as run_TAI_scaled_for_model
import os
import pandas as pd

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
    }


def calculate_new_hospital_capacity_index(hospital_file_path: str) -> dict:
    """
    Calculates the Hospital Capacity Index for each district in Saarland.

    The index is calculated using the following steps:
    1. Aggregates total hospital beds per district
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
    # Load cleaned datasets
    hospital_df = load_hospital_data(hospital_file_path)
    population_df = load_population_data()

    # Aggregate total beds per Kreis
    beds_per_kreis = hospital_df.groupby("district")["beds"].sum().reset_index()
    beds_per_kreis.rename(columns={"beds": "TotalBeds"}, inplace=True)

    # Merge with population data
    merged_df = pd.merge(beds_per_kreis, population_df, on="district", how="inner")

    # Compute adjusted beds per capita
    merged_df["AdjBeds"] = merged_df["TotalBeds"] / merged_df["population"]

    # Normalize and invert to compute HospitalCapacityIndex
    min_adj = merged_df["AdjBeds"].min()
    max_adj = merged_df["AdjBeds"].max()
    merged_df["HospitalCapacityIndex"] = 1 - (merged_df["AdjBeds"] - min_adj) / (max_adj - min_adj)
    # Ensure HospitalCapacityIndex is numeric and handle any NaN values
    merged_df["HospitalCapacityIndex"] = pd.to_numeric(merged_df["HospitalCapacityIndex"], errors='coerce').fillna(1)
    # Convert to dictionary: {district: HospitalCapacityIndex}
    result_dict = dict(zip(merged_df["district"], merged_df["HospitalCapacityIndex"].round(4)))

    # Map result_dict keys to AGS
    mapped_result = {
    ags: result_dict.get(float(ags[-2:]))
    for ags in SAARLAND_AGS
    }

    return mapped_result



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
    xl = pd.ExcelFile(hospital_file_path, engine="openpyxl")
    #sheet_df = xl.parse("KHV_2021", header=None)


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
        population_data (pd.DataFrame): DataFrame with columns ['district', 'population'].
        weights (dict): Dictionary mapping Index enum values to their respective weights.

    Returns:
        pd.Series: Equity Index values for each district.
    """

    INDEX_FUNC_MAP[Index.HOSPITAL_CAPACITY] = lambda: calculate_new_hospital_capacity_index(hospital_file_path)
    INDEX_FUNC_MAP[Index.TRAVEL_TIME] = run_TAI_scaled_for_model(modelname)

    index_df = assemble_indexes()
    weight = {
        Index.FORECAST_DEMAND: 0.25,
        Index.ELDERLY_SHARE: 0.25,
        Index.GISD: 0.25,
        Index.HOSPITAL_CAPACITY: 0.25,
        Index.TRAVEL_TIME: 0.25,
        Index.ACCESSIBILITY: 0.25,
    }

    return equity_index(index_df, weight)

    



