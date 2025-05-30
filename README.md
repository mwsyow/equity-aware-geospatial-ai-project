# equity-aware-geospatial-ai-project

This project provides an equity-aware geospatial AI planner for hospital bed allocation across districts, balancing cost and equity using multiple health and accessibility indexes.

## Main Script: `src/main.py`

### Overview

The `main.py` script integrates outputs from several index modules (demand forecasting, elderly share, GISD, hospital capacity, travel accessibility) and runs an AI-based planner to optimize hospital bed allocation.

### Key Features

- **Index Calculation:** Aggregates multiple health and accessibility indexes per district.
- **Equity Index:** Computes a composite equity index using weighted sub-indexes.
- **Hospital Bed Planning:** Uses an agentic AI planner to allocate beds to hospitals under constraints (budget, max open sites, travel time).
- **Geospatial Output:** Generates a GeoDataFrame for visualization and further analysis.

### Usage

1. **Install dependencies:**  
   Make sure all required Python packages are installed (see `requirements.txt`).

2. **Run the main script:**  
   ```bash
   python3 src/main.py
   ```

3. **Output:**  
   - Prints the selected hospital sites, bed allocations, and objective value.
   - Saves a map visualization of predicted and existing hospitals to the `results/` directory.

### Main Components

- **assemble_indexes:** Aggregates all index values into a DataFrame.
- **equity_index:** Calculates the equity index for each district.
- **current_hospital_demand:** Computes current hospital demand per district.
- **centroid:** Gets the centroid (geometry) for each district.
- **AgenticPlannerWithPrediction:** Optimizes hospital bed allocation using Bayesian optimization.

### Customization

- Adjust weights for the equity index in the `main()` function.
- Modify planner parameters (e.g., `budget_beds`, `max_open_sites`, `alpha`, `max_travel`) as needed.

### Directory Structure

```
src/
  main.py
  ai_planner/
  index_demand_forecast/
  index_elderly_share/
  index_gisd/
  index_hospital_capacity/
  index_travel_accessibility/
results/
  saarland_map_with_predicted_and_exisiting_hospitals.html
```

---

For more details, see the code comments in `src/main.py`. 