# equity-aware-geospatial-ai-project

This project provides an equity-aware geospatial AI planner for hospital bed allocation across districts in Saarland, Germany, balancing cost and equity using multiple health and accessibility indexes.

## Project Structure

```
src/
├── main.py                 # Main script orchestrating the planning process
├── main.ipynb             # Jupyter notebook for interactive development
├── requirements.txt       # Python package dependencies
├── environment.yml        # Conda environment configuration
├── ai_planner/           # AI-based hospital location optimization
├── index_demand_forecast/ # Demand forecasting module
├── index_elderly_share/   # Elderly population analysis
├── index_gisd/           # German Index of Socioeconomic Deprivation
├── index_hospital_capacity/ # Hospital capacity analysis
├── index_travel_accessibility/ # Travel time and accessibility analysis
├── evaluation/           # Evaluation metrics and analysis
├── experiments/          # Experimental configurations and results
├── results/             # Output visualizations and results
└── cache/               # Cached data for faster processing
```

## Implementation Overview

### Core Components

1. **Index Calculation Modules**
   - Demand Forecast Index: Predicts future healthcare demand
   - Elderly Share Index: Analyzes elderly population distribution
   - GISD Index: Measures socioeconomic deprivation
   - Hospital Capacity Index: Evaluates existing hospital capacity
   - Travel Time Index: Assesses accessibility via road network

2. **Equity Index System**
   - Combines multiple health and accessibility indexes
   - Uses weighted combinations to compute composite equity scores
   - Considers both current and projected healthcare needs

3. **Hospital Planning System**
   - AI-driven optimization using CMA-ES
   - Multi-objective optimization considering:
     - Equity improvement
     - Travel time minimization
     - Bed allocation optimization
     - Supply distribution
   - Geospatial constraints handling
   - Road network integration

### Key Features

- **Data Integration:** Combines multiple data sources for comprehensive analysis
- **Spatial Analysis:** Utilizes geospatial data for location optimization
- **Equity Focus:** Prioritizes healthcare access in underserved areas
- **Constraint Handling:** Manages multiple operational and spatial constraints
- **Visualization:** Generates interactive maps for result analysis

## Getting Started

### Prerequisites

1. **Environment Setup**
   ```bash
   # Using conda
   conda env create -f environment.yml
   conda activate equity-aware-geospatial-ai

   # Or using pip
   pip install -r requirements.txt
   ```

2. **Data Preparation**
   - Ensure all required data files are in their respective module directories
   - Check the `cache/` directory for any cached data requirements

### Running the Project

1. **Interactive Development**
   ```bash
   jupyter notebook src/main.ipynb
   ```

2. **Command Line Execution**
   ```bash
   python src/main.py
   ```

3. **Output**
   - Results are saved in the `results/` directory
   - Interactive maps show both existing and predicted hospital locations
   - Detailed metrics and analysis available in the evaluation module

### Development Workflow

1. **Data Processing**
   - Each index module processes its specific data
   - Results are cached for faster subsequent runs

2. **Planning Process**
   - Index calculation and aggregation
   - Equity index computation
   - Candidate location generation
   - Optimization execution
   - Result visualization

3. **Evaluation**
   - Performance metrics calculation
   - Equity impact assessment
   - Accessibility analysis
   - Resource utilization evaluation



For more details, see the code comments in `src/main.py`. 