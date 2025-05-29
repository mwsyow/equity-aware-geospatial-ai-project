### Step 1: Compute Adjusted Beds per District

# AdjBeds₍d₎ = Beds₍d₎ ÷ Population₍d₎

### Step 2: Normalize & Invert to get Hospital Capacity Index

# HospitalCapacityIndex₍d₎ = 1 − (AdjBeds₍d₎ − min(AdjBeds)) ÷ (max(AdjBeds) − min(AdjBeds))

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


def load_hospital_data() -> pd.DataFrame:
    """
    Loads the hospital dataset from Excel, finds the header row dynamically,
    filters for a specific Land code, and returns a DataFrame with 'region', 'district', and 'beds'.
    """
    file_path = "data/Krankenhausverzeichnis_2021.xlsx"
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
    Loads population data, finds the header row dynamically,
    filters for a specific Land code, and returns a DataFrame with 'district' and 'population'.
    """
    file_path = "data/District-Population.xlsx"
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

if __name__ == "__main__":
    result = calculate_hospital_capacity_index()
    print(result)
