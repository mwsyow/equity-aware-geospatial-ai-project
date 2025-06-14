from statistics import mean, stdev
import pandas as pd

from metrics.hdr_metric import calculate_hdr  # Must return a dict or Series with district codes as keys

SAARLAND_AGS = {
    "Regionalverband Saarbrücken": "10041",
    "Merzig-Wadern": "10042",
    "Neunkirchen": "10043",
    "Saarlouis": "10044",
    "Saarpfalz-Kreis": "10045",
    "St. Wendel": "10046"
}

def compute_overserved_area_count(hospital_file_path: str) -> pd.Series:
    """
    Computes a binary indicator for overserved districts.

    For each district, computes HDR using hdr_metric, then determines if the district is overserved.
    A district is considered overserved if HDR > mean(HDRs) + 1.5 * std(HDRs).

    Returns:
        pd.Series: Series with district names as index and values 1 (overserved) or 0.
    """
    hdr_values = calculate_hdr(hospital_file_path)  # Should return dict or Series {district_code: HDR}
    if not hdr_values:
        return pd.Series(dtype=int)

    # Ensure we have only Saarland district HDRs and cast to float
    hdr_series = pd.Series(hdr_values)
    hdr_series.index = hdr_series.index.astype(str)
    hdr_series = hdr_series[hdr_series.index.isin(SAARLAND_AGS.values())]
    hdr_series = hdr_series.astype(float)

    # Compute threshold
    hdr_mean = hdr_series.mean()
    hdr_std = hdr_series.std(ddof=1) if len(hdr_series) > 1 else 0
    threshold = hdr_mean + 1.5 * hdr_std

    # Compute binary indicators per district
    overserved_flags = (hdr_series > threshold).astype(int)

    # Reindex and relabel to readable district names
    ordered_codes = list(SAARLAND_AGS.values())
    overserved_flags = overserved_flags.reindex(ordered_codes)
    overserved_flags.index = list(SAARLAND_AGS.keys())

    return overserved_flags