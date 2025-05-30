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

def load_hospital_data() -> pd.DataFrame:
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
    file_path = os.path.join(DATA_PATH, "Krankenhausverzeichnis_2021.xlsx")
    xl = pd.ExcelFile(file_path, engine="openpyxl")
    sheet_df = xl.parse("KHV_2021", header=None)

    # Detect first non-empty row as header
    for i, row in sheet_df.iterrows():
        if row.notna().sum() > 3:  # assuming at least 4 non-empty values in the header row
            header_row = i
            break

    # Re-read with correct header
    df = pd.read_excel(file_path, sheet_name="KHV_2021", skiprows=header_row, engine="openpyxl")
    df.columns = df.columns.str.strip()

    df = df[df["Land"] == REGION_CODE]
    df = df[["Land", "Kreis", "INSG"]].rename(columns={"Land": "region", "Kreis": "district", "INSG": "beds"})
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

def calculate_hospital_capacity_index() -> dict:
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
    hospital_df = load_hospital_data()
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

    # Convert to dictionary: {district: HospitalCapacityIndex}
    result_dict = dict(zip(merged_df["district"], merged_df["HospitalCapacityIndex"].round(4)))

    # Map result_dict keys to AGS
    mapped_result = {
    ags: result_dict.get(float(ags[-2:]))
    for ags in SAARLAND_AGS
    }

    return mapped_result
