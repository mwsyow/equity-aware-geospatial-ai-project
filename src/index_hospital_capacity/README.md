# Hospital Capacity Index

## Overview
This module calculates a hospital capacity index for districts in Saarland, Germany by analyzing the relationship between available hospital beds and population size. The index ranges from 0 to 1, where higher values indicate greater need for additional capacity.

## Methodology

### 1. Data Processing
- Loads hospital bed data per district
- Loads population data per district 
- Aggregates total beds per district (Kreis)

### 2. Index Calculation
The index is calculated in two steps:

1. **Compute Adjusted Beds per District**
   ```math
   AdjBeds₍d₎ = Beds₍d₎ ÷ Population₍d₎
   ```
   Where:
   - `d` is the district
   - `Beds₍d₎` is total hospital beds in district
   - `Population₍d₎` is district population

2. **Normalize & Invert to get Hospital Capacity Index**
   ```math
   HospitalCapacityIndex₍d₎ = 1 − (AdjBeds₍d₎ − min(AdjBeds)) ÷ (max(AdjBeds) − min(AdjBeds))
   ```

## Usage

```python
from hospital_capacity_index_dict import calculate_hospital_capacity_index

# Calculate index for all Saarland districts
result = calculate_hospital_capacity_index()
print(result)
```

## Output Format
Returns a dictionary mapping district AGS codes to their capacity index:
```python
{
    "10041": 0.xxxx,  # Regionalverband Saarbrücken
    "10042": 0.xxxx,  # Merzig-Wadern 
    "10043": 0.xxxx,  # Neunkirchen
    "10044": 0.xxxx,  # Saarlouis
    "10045": 0.xxxx,  # Saarpfalz-Kreis
    "10046": 0.xxxx   # St. Wendel
}
```

## Dependencies
- pandas==2.2.3
- openpyxl==3.1.5

## Data Sources
- Hospital bed data per district
- Population statistics per district