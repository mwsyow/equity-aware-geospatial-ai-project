from statistics import mean, stdev
import pandas as pd

# StrEnum compatibility for Python < 3.11
try:
    from enum import StrEnum
except ImportError:
    from enum import Enum
    class StrEnum(str, Enum):
        pass

from .hdr_metric import calculate_hdr  # Must return a dict or Series with district codes as keys

SAARLAND_AGS = {
    "Regionalverband Saarbrücken": "10041",
    "Merzig-Wadern": "10042",
    "Neunkirchen": "10043",
    "Saarlouis": "10044",
    "Saarpfalz-Kreis": "10045",
    "St. Wendel": "10046"
}

def compute_overserved_area_count(hospital_file_path: str, modelname: str = None) -> pd.Series:
    """
    Computes a binary indicator for overserved districts.

    For each district, computes HDR using hdr_metric, then determines if the district is overserved.
    A district is considered overserved if HDR > mean(HDRs) + 1.5 * std(HDRs).

    Returns:
        pd.Series: Series with district names as index and values 1 (overserved) or 0.
    """
    print(f"🔍 Computing overserved area count for: {hospital_file_path}")
    hdr_values = calculate_hdr(hospital_file_path)  # Should return dict or Series {district_code: HDR}
    
    print(f"🔍 HDR values received:")
    print(f"   Type: {type(hdr_values)}")
    print(f"   Empty: {hdr_values.empty if hasattr(hdr_values, 'empty') else 'N/A'}")
    print(f"   Values: {hdr_values}")
    
    if hasattr(hdr_values, 'empty') and hdr_values.empty:
        print(f"❌ HDR values are empty, returning empty series")
        return pd.Series(dtype=int)

    # Ensure we have only Saarland district HDRs and cast to float
    hdr_series = pd.Series(hdr_values).astype(float)
    print(f"🔍 HDR series created:")
    print(f"   Shape: {hdr_series.shape}")
    print(f"   Index: {hdr_series.index.tolist()}")
    print(f"   Values: {hdr_series.values}")
    
    # The HDR values already have district names as index, so we don't need to filter
    # Just ensure they are float values
    hdr_series = hdr_series.astype(float)
    
    print(f"🔍 After filtering for Saarland districts:")
    print(f"   Shape: {hdr_series.shape}")
    print(f"   Index: {hdr_series.index.tolist()}")
    print(f"   Values: {hdr_series.values}")

    # Compute threshold
    hdr_mean = hdr_series.mean()
    hdr_std = hdr_series.std(ddof=1) if len(hdr_series) > 1 else 0
    threshold = hdr_mean + 1.5 * hdr_std
    
    print(f"🔍 Threshold calculation:")
    print(f"   Mean: {hdr_mean}")
    print(f"   Std: {hdr_std}")
    print(f"   Threshold: {threshold}")

    # Compute binary indicators per district
    overserved_flags = (hdr_series > threshold).astype(int)
    
    print(f"🔍 Overserved flags:")
    print(f"   Values: {overserved_flags.values}")

    # The overserved_flags already have district names as index, so we just need to ensure order
    ordered_names = list(SAARLAND_AGS.keys())
    overserved_flags = overserved_flags.reindex(ordered_names)
    
    print(f"🔍 Final result:")
    print(f"   Values: {overserved_flags.values}")
    print(f"   Index: {overserved_flags.index.tolist()}")

    return overserved_flags