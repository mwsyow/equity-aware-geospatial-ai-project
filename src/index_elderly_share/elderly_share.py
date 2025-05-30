# src/index_elderly/elderly.py

import os
import pandas as pd
import csv

BASE_DIR = os.path.dirname(__file__)
RAW_CSV  = os.path.join(BASE_DIR, "data", "raw", "pop_age_groups.csv")
OUT_DIR  = os.path.join(BASE_DIR, "data", "processed")
OUT_CSV  = os.path.join(OUT_DIR, "elderly_share_saarland.csv")

SAARLAND_AGS = {"10041","10042","10043","10044","10045","10046"}

DEF_DATE    = "2021-12-31"

def load_raw():
    """
    Skip the first 6 descriptive lines, then read only these columns:
      0 = reference date
      1 = region code (AGS)
     33 = '65 to under 75 years' value
     35 = '75 years and over' value
     37 = 'Total' value
    (The odd‐numbered columns in between are the little 'e' footnote columns.)
    """

    # first pass: find which line data actually begins on
    with open(RAW_CSV, newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=";")
        for i, row in enumerate(reader):
            if row and row[0] == DEF_DATE:
                header_line = i
                break
        else:
            raise ValueError(f"Could not find a row starting with {DEF_DATE}")

    # second pass: tell pandas exactly how many rows to skip
    df = pd.read_csv(
        RAW_CSV,
        sep=";",
        skiprows=header_line,
        header=None,        # we already know the column positions
        usecols=[0, 1, 33, 35, 37],
        dtype={1: str},
        names=["ref_date", "AGS", "65_to_75", "75_plus", "pop_total"]
    )

    return df

def trim_by_year(df, date):

    # forward-fill the header dates
    df["ref_date"] = df["ref_date"].ffill()

    # now only keep the required date block
    df_by_date = df[df["ref_date"] == date].copy()

    df_by_date.reset_index(drop=True, inplace=True)

    return df_by_date

def trim_by_ags_codes(df, ags_codes):
    mask = df["AGS"].isin(ags_codes)

    # pull out only those rows
    df_by_ags_codes = df[mask].reset_index(drop=True)
    return df_by_ags_codes

def compute_share(df):
    for col in ["65_to_75", "75_plus", "pop_total"]:
        df[col] = pd.to_numeric(df[col], errors="raise").astype(int)

    df["elderly_share"] = (df["65_to_75"] + df["75_plus"]) / df["pop_total"]
    return df

def get_dict(df):
    return dict(zip(df["AGS"], df["elderly_share"]))


def run():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_raw()
    def_by_date = trim_by_year(df, DEF_DATE)
    saarland_def = trim_by_ags_codes(def_by_date, SAARLAND_AGS)
    final_def = compute_share(saarland_def)

    return get_dict(final_def)

if __name__ == "__main__":
    run()