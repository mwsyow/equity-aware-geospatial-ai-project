# src/index_gisd/gisd.py

import os
import pandas as pd
from sklearn.preprocessing import minmax_scale

# ——————————————————————————————
# Configuration
# ——————————————————————————————
BASE_DIR      = os.path.dirname(__file__)
RAW_CSV       = os.path.join(BASE_DIR, "data", "raw", "GISD_Bund_Kreis.csv")
OUT_DIR       = os.path.join(BASE_DIR, "data", "processed")
OUT_CSV       = os.path.join(OUT_DIR, "gisd_norm_saarland.csv")
GISD_YEAR     = 2021

# Only the six Saarland districts by AGS code
SAARLAND_AGS = [
    "10041",  # Regionalverband Saarbrücken
    "10042",  # Merzig-Wadern
    "10043",  # Neunkirchen
    "10044",  # Saarlouis
    "10045",  # Saarpfalz-Kreis
    "10046",  # St. Wendel
]
# ——————————————————————————————

def load_gisd() -> pd.DataFrame:
    """
    Reads the raw GISD CSV (columns: kreis_id, kreis_name, year,
    gisd_score, gisd_5, gisd_10, gisd_k) and returns a DataFrame
    with AGS, year, and gisd_value (the composite score).
    """
    df = pd.read_csv(RAW_CSV, dtype={"kreis_id": str})
    # rename to our internal names
    df = df.rename(columns={
        "kreis_id":   "AGS",
        "year":       "year",
        "gisd_score": "gisd_value"
    })
    # select only the three we need
    return df[["AGS", "year", "gisd_value"]]

def filter_year_and_region(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Keep only rows for GISD_YEAR and filter to the six Saarland AGS codes.
    """
    df_y = df[df["year"] == year].copy()
    df_y["AGS"] = df_y["AGS"].str.zfill(5)
    # Filter to Saarland districts
    df_y = df_y[df_y["AGS"].isin(SAARLAND_AGS)]
    return df_y[["AGS", "gisd_value"]].reset_index(drop=True)

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Min–Max scale gisd_value into [0,1], returning AGS + gisd_norm.
    """
    df = df.copy()
    df["gisd_norm"] = minmax_scale(df["gisd_value"])
    return df[["AGS", "gisd_norm"]]

def run() -> pd.DataFrame:
    """
    Full GISD workflow for Saarland:
      1) Load raw data
      2) Filter to 2021 and Saarland AGS
      3) Normalize values
      4) Save output CSV
    """
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1. Load
    raw_df = load_gisd()

    # 2. Filter to 2021 + Saarland
    filtered_df = filter_year_and_region(raw_df, GISD_YEAR)

    # 3. Normalize
    norm_df = normalize(filtered_df)

    # 4. Save
    norm_df.to_csv(OUT_CSV, index=False)
    print(f"[GISD] Wrote {len(norm_df)} Saarland districts to {OUT_CSV}")

    return norm_df

if __name__ == "__main__":
    run()