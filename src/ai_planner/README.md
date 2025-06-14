# AI Planner for Equity-Aware Hospital Location Optimization

## Overview
This module implements an AI-driven hospital location and capacity planning system that considers both operational efficiency and healthcare equity. The planner uses multi-objective optimization with CMA-ES (Covariance Matrix Adaptation Evolution Strategy) to determine optimal hospital locations and bed allocations while balancing multiple factors including equity, travel time, and resource constraints.

## Key Features
- Optimizes hospital locations and bed allocations
- Considers multiple factors:
  - Population demand coverage
  - Equity considerations for underserved areas
  - Travel time accessibility
  - Proximity to existing hospitals
  - Resource supply constraints
  - Bed allocation optimization
- Uses CMA-ES for efficient weight optimization
- Supports both existing and new hospital locations
- Handles geospatial constraints and road network accessibility

## Classes

### HospitalPlanner
Main planner class that implements multi-objective optimization for hospital placement.

**Key Methods:**
- `generate_candidates`: Creates potential hospital locations around district centroids
- `init_baseline_model`: Sets up the optimization model with equity, travel time, and bed allocation objectives
- `predict_location`: Determines optimal hospital locations for given weights
- `run`: Executes CMA-ES optimization to find best weights and locations

**Parameters:**
- `districts_gdf`: GeoDataFrame with district data (demand, equity_index, centroid)
- `weights`: Dictionary of optimization weights:
  - equity_weight: Weight for equity improvement
  - travel_weight: Weight for travel time minimization
  - beds_weight: Weight for bed allocation optimization
  - supply_weight: Weight for supply oversaturation penalty
- `time_threshold`: Maximum travel time in minutes
- `max_beds_per_hospital`: Maximum beds per hospital
- `min_beds_per_hospital`: Minimum beds per hospital
- `max_beds`: Total maximum beds across all hospitals
- `num_neighbors`: Maximum allowed existing hospital neighbors
- `eh_distance_threshold`: Minimum distance from existing hospitals (meters)
- `c2c_distance_threshold`: Minimum distance between new hospitals (meters)
- `k`: Number of hospitals to build

## Usage Example

```python
# Initialize planner
planner = HospitalPlanner(districts_gdf)

# Generate candidate locations
planner.generate_candidates(n_samples_per_centroid=10, radius_km=5)

# Initialize road network
planner.init_graph()

# Build necessary matrices
planner.build_matrix()

# Run optimization with initial weights
initial_weights = {
    'equity_weight': 0.75,
    'travel_weight': 0.065,
    'beds_weight': 0.065,
    'supply_weight': 0.065
}

# Get optimal hospital locations
selected_candidates = planner.run(
    initial_weights=initial_weights,
    step_size=0.1,
    num_neighbors=0,
    eh_distance_threshold=7000,
    c2c_distance_threshold=7000,
    time_threshold=30,
    max_beds_per_hospital=300,
    min_beds_per_hospital=20,
    max_beds=1500,
    k=10
)
```

## Dependencies
```
geopandas>=1.0.1
numpy>=2.2.6
pandas>=2.2.3
networkx>=3.0
osmnx>=1.0
shapely>=2.1.1
pulp>=2.7
cma>=3.0
```

## Optimization Details

### Objective Function
The planner optimizes a weighted combination of:
- Equity term: Weighted sum of district equity indices
- Travel time term: Minimization of travel times between districts and hospitals
- Bed allocation term: Optimization of bed distribution based on demand
- Supply penalty: Penalty for oversaturation near existing hospitals

### Constraints
- Maximum and minimum beds per hospital
- Total beds must not exceed maximum
- Travel time to hospitals must be within threshold
- Minimum distance from existing hospitals
- Minimum distance between new hospitals
- Maximum number of existing hospital neighbors
- Exact number of hospitals to build (k)

## Output
The optimization returns a GeoDataFrame containing:
- Selected candidate locations with optimal bed allocations
- Geographic coordinates (Lat, Lon)
- Bed allocation per hospital
- Type indicator ('predicted' vs 'existing')