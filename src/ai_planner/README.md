# AI Planner for Equity-Aware Hospital Location Optimization

## Overview
This module implements an AI-driven hospital location and capacity planning system that considers both operational efficiency and healthcare equity. The planner uses Bayesian optimization to determine optimal hospital locations and bed allocations while balancing costs against equity measures.

## Key Features
- Optimizes hospital locations and bed allocations
- Considers multiple equity factors:
  - Demand forecasting
  - Socioeconomic deprivation (GISD)
  - Travel time accessibility
  - Elderly population share
  - Hospital capacity
- Supports both existing and predicted new hospital locations
- Uses Bayesian optimization for efficient solution search

## Classes

### AgenticPlanner
Base planner class that optimizes hospital locations and bed allocations.

**Parameters:**
- `hospitals`: DataFrame with hospital data (Lon, Lat, CostPerBed, MaxBeds)
- `districts`: DataFrame with district metrics (Demand, EquityIndex, centroid)
- `travel_time`: Travel time matrix between sites and districts
- `budget_beds`: Total available beds to allocate
- `max_open_sites`: Maximum number of hospitals to open
- `alpha`: Weight between cost and equity (0-1)
- `max_travel`: Maximum acceptable travel time in minutes

### AgenticPlannerWithPrediction
Extends AgenticPlanner with capability to predict new optimal hospital locations.

**Additional Features:**
- Predicts new hospital locations using KMeans clustering
- Estimates travel times for predicted locations
- Visualizes existing and predicted hospitals on map

## Usage Example

```python
# Initialize planner
planner = AgenticPlannerWithPrediction(
    hospitals_df,
    districts_gdf,
    travel_time_dict,
    budget_beds=1000,
    max_open_sites=2,
    alpha=0.6,
    max_travel=30,
    predict_new=4
)

# Run optimization
plan = planner.optimize_with_bayesian()

# Access results
print("Open sites:", plan["open_sites"])
print("Beds allocation:", plan["beds_alloc"])
print("Objective value:", plan["objective"])

# Visualize results
planner.plot_predicted_hospitals()
```

## Dependencies
```
geopandas==1.0.1
numpy==2.2.6
pandas==2.2.3
scikit-learn==1.6.1
matplotlib==3.10.3
shapely==2.1.1
bayesian-optimization==2.0.4
```

## Optimization Details

### Objective Function
The planner optimizes a weighted combination of:
- Cost term: Sum of beds × cost per bed for each open site
- Equity term: Weighted sum of unmet demand adjusted by district equity indices

### Constraints
- Total beds must not exceed budget
- Number of open sites must not exceed maximum
- Travel time to hospitals must be within specified limit

## Output
The optimization returns a dictionary containing:
- `open_sites`: Set of selected hospital locations
- `beds_alloc`: Bed allocation per hospital
- `coverage`: District-level coverage metrics
- `objective`: Final objective function value