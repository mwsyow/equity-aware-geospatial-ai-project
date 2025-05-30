# src/index_elderly/elderly.py

"""
Calculate the share of elderly population (age 65+) for districts in Saarland.

This script processes demographic data to compute the percentage of elderly residents
in each district (AGS) of Saarland, using population data for age groups 65-75 and 75+.
"""

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
    Load raw demographic data from CSV file.

    Returns:
        pandas.DataFrame: DataFrame containing columns:
            - ref_date: Reference date of the data
            - AGS: District identifier code
            - 65_to_75: Population count for age group 65-75
            - 75_plus: Population count for age group 75+
            - pop_total: Total population
    
    Raises:
        ValueError: If the specified reference date is not found in the CSV
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
    """
    Filter DataFrame to keep only rows for a specific reference date.

    Args:
        df (pandas.DataFrame): Input DataFrame with demographic data
        date (str): Reference date to filter by

    Returns:
        pandas.DataFrame: Filtered DataFrame containing only rows for specified date
    """

    # forward-fill the header dates
    df["ref_date"] = df["ref_date"].ffill()

    # now only keep the required date block
    df_by_date = df[df["ref_date"] == date].copy()

    df_by_date.reset_index(drop=True, inplace=True)

    return df_by_date

def trim_by_ags_codes(df, ags_codes):
    """
    Filter DataFrame to keep only rows for specified district codes (AGS).

    Args:
        df (pandas.DataFrame): Input DataFrame with demographic data
        ags_codes (set): Set of district codes to include

    Returns:
        pandas.DataFrame: Filtered DataFrame containing only rows for specified districts
    """
    mask = df["AGS"].isin(ags_codes)

    # pull out only those rows
    df_by_ags_codes = df[mask].reset_index(drop=True)
    return df_by_ags_codes

def compute_share(df):
    """
    Calculate the share of elderly population (65+) for each district.

    Args:
        df (pandas.DataFrame): Input DataFrame with demographic data

    Returns:
        pandas.DataFrame: DataFrame with additional 'elderly_share' column containing
            the proportion of population aged 65 and over
    """

    for col in ["65_to_75", "75_plus", "pop_total"]:
        df[col] = pd.to_numeric(df[col], errors="raise").astype(int)

    df["elderly_share"] = (df["65_to_75"] + df["75_plus"]) / df["pop_total"]
    return df

def get_dict(df):
    """
    Convert DataFrame results to a dictionary mapping district codes to elderly shares.

    Args:
        df (pandas.DataFrame): Input DataFrame with AGS codes and elderly_share column

    Returns:
        dict: Dictionary with AGS codes as keys and elderly share values as values
    """

    return dict(zip(df["AGS"], df["elderly_share"]))


def run():
    """
    Execute the complete workflow to calculate elderly population shares.

    Creates output directory if it doesn't exist, loads raw data,
    processes it for Saarland districts, and computes elderly shares.

    Returns:
        dict: Dictionary mapping Saarland district codes to their elderly population shares
    """

    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_raw()
    def_by_date = trim_by_year(df, DEF_DATE)
    saarland_def = trim_by_ags_codes(def_by_date, SAARLAND_AGS)
    final_def = compute_share(saarland_def)

    return get_dict(final_def)
