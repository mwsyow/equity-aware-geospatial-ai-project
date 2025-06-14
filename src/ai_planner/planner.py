"""
Hospital Location Planning using Multi-Objective Optimization.

This module provides functionality for optimal hospital placement using mathematical 
optimization techniques. It considers multiple factors including population demand, 
equity indices, travel times, and existing healthcare infrastructure to determine 
the best locations for new hospitals.

The optimization model balances:
- Population demand coverage
- Equity considerations for underserved areas  
- Travel time accessibility
- Proximity to existing hospitals
- Resource supply constraints
- Bed allocation optimization

Dependencies:
    - networkx: For road network graph operations
    - numpy: For numerical computations
    - pandas: For data manipulation
    - geopandas: For geospatial data handling
    - shapely: For geometric operations
    - osmnx: For OpenStreetMap data retrieval
    - pulp: For linear programming optimization
    - cma: For CMA-ES optimization
"""

import networkx as nx
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import osmnx as ox
import pulp
import cma
from .utils import get_existing_hospitals_gdf

# Geographic coordinate system configuration
CRS_NUM = 4326
CRS = f"EPSG:{CRS_NUM}"
DEFAULT_GRAPH_LOCATION = "Saarland, Germany"


class HospitalPlanner:
    """
    A multi-objective optimization system for hospital location planning.
    
    This class implements a mathematical programming approach to determine optimal
    hospital locations considering demand, equity, accessibility, and existing
    infrastructure constraints.
    
    Attributes:
        districts_gdf (gpd.GeoDataFrame): Geographic data of districts with demand/equity info
        candidates_gdf (gpd.GeoDataFrame): Generated candidate hospital locations
        G (networkx.Graph): Road network graph for travel time calculations
        model (pulp.LpProblem): Linear programming optimization model
        D (list): List of district indices
        P (list): List of candidate location indices
        x (dict): Binary decision variables for hospital placement
        y (dict): Binary decision variables for district-hospital assignments
    """
    
    def __init__(self, districts_gdf: gpd.GeoDataFrame):
        """
        Initialize the HospitalPlanner with district data.
        
        Args:
            districts_gdf (gpd.GeoDataFrame): GeoDataFrame containing district polygons
                with required columns: 'demand', 'equity_index', 'centroid'
        """
        self.districts_gdf = districts_gdf.to_crs(epsg=CRS_NUM)
        
    @staticmethod
    def generate_points(centroid, radius_km, n, bound_poly) -> list[Point]:
        """
        Generate random points within a circular area around a centroid.
        
        Uses uniform random sampling within a circle, constrained by a bounding polygon.
        Points are generated using polar coordinates with appropriate density correction.
        
        Args:
            centroid (Point): Center point for generation
            radius_km (float): Maximum radius in kilometers
            n (int): Number of points to attempt generating
            bound_poly (Polygon): Bounding polygon to constrain points
            
        Returns:
            list[Point]: List of valid points within the bounded area
        """
        r_deg = radius_km / 111.0  # Rough conversion from km to degrees
        points = []
        for _ in range(n):
            # Generate random point using polar coordinates
            # sqrt ensures uniform density distribution
            r = r_deg * np.sqrt(np.random.rand())
            theta = np.random.rand() * 2 * np.pi
            dx, dy = r * np.cos(theta), r * np.sin(theta)
            
            cand = Point(centroid.x + dx, centroid.y + dy)
            
            # Only include points within the bounding polygon
            if bound_poly.contains(cand):
                points.append(cand)
        return points
    
    @staticmethod
    def normalize_metric(metric: dict) -> dict:
        """
        Normalize metric values to [0,1] range using min-max normalization.
        
        Handles NaN and infinity values by setting them to 0. Prevents division
        by zero when all values are identical.
        
        Args:
            metric (dict): Dictionary mapping keys to numeric values
            
        Returns:
            dict: Dictionary with normalized values in [0,1] range
        """
        # Filter out NaN and infinity values for min/max calculation
        valid_values = [v for v in metric.values() if np.isfinite(v)]
        
        if not valid_values:
            # If all values are NaN or infinity, return zeros
            return {k: 0.0 for k in metric.keys()}
        
        min_val = min(valid_values)
        max_val = max(valid_values)
        
        # Avoid division by zero
        if max_val == min_val:
            return {k: 0.0 for k in metric.keys()}
        
        # Normalize, setting NaN/infinity values to 0
        normalized = {}
        for k, v in metric.items():
            if np.isfinite(v):
                normalized[k] = (v - min_val) / (max_val - min_val)
            else:
                normalized[k] = 0.0
        
        return normalized
    
    def generate_candidates(self, n_samples_per_centroid: int, radius_km: int):
        """
        Generate candidate hospital locations around district centroids.
        
        Creates potential hospital sites by sampling points around each district's
        centroid within the specified radius and regional boundaries.
        
        Args:
            n_samples_per_centroid (int): Number of candidate points per district
            radius_km (int): Search radius in kilometers around each centroid
            
        Side Effects:
            Sets self.candidates_gdf with generated candidate locations
        """
        # Get regional boundary for constraining candidates
        gdf_saarland = ox.geocode_to_gdf(DEFAULT_GRAPH_LOCATION)
        gdf_saarland_utm = gdf_saarland.to_crs(epsg=CRS_NUM)
        poly_utm = gdf_saarland_utm.loc[0, "geometry"] 

        candidates_dfs = []
        for _, row in self.districts_gdf.iterrows():
            candidates = HospitalPlanner.generate_points(
                row['centroid'], 
                radius_km=radius_km, 
                n=n_samples_per_centroid, 
                bound_poly=poly_utm
            )
            candidates_df = pd.DataFrame(candidates, columns=['geometry'])
            candidates_df['district_code'] = row['district_code']
            candidates_dfs.append(candidates_df)
            
        self.candidates_gdf = gpd.GeoDataFrame(pd.concat(candidates_dfs), crs=CRS).reset_index(drop=True)
    
    def init_graph(self, graph_path: str = None) -> nx.Graph:
        """
        Initialize road network graph for travel time calculations.
        
        Loads or downloads OpenStreetMap road network data, projects to appropriate
        coordinate system, and adds travel time attributes. Maps district centroids
        and candidate locations to nearest graph nodes.
        
        Args:
            graph_path (str, optional): Path to saved GraphML file. If None,
                downloads from OpenStreetMap. Defaults to None.
                
        Side Effects:
            Sets self.G with the road network graph
            Adds 'node' column to districts_gdf and candidates_gdf
        """
        if graph_path is None:
            G = ox.graph_from_place(DEFAULT_GRAPH_LOCATION, network_type='drive')
        else:
            G = ox.load_graphml(graph_path)
            
        G = ox.project_graph(G, to_crs=CRS)

        # Add travel time attributes to edges
        G = ox.add_edge_speeds(G)
        G = ox.add_edge_travel_times(G)

        # Find nearest graph nodes for district centroids
        self.districts_gdf['node'] = ox.nearest_nodes(
            G,
            self.districts_gdf.geometry.x,
            self.districts_gdf.geometry.y
        )

        # Find nearest graph nodes for candidate locations
        self.candidates_gdf['node'] = ox.nearest_nodes(
            G,
            self.candidates_gdf.geometry.x,
            self.candidates_gdf.geometry.y
        )
        
        self.G = G
    
    @staticmethod
    def calculate_ttm(G: nx.Graph, gdf1: gpd.GeoDataFrame, gdf2: gpd.GeoDataFrame, weight: str) -> dict:
        """
        Build travel time/distance matrix between two sets of locations.
        
        Computes shortest path distances or travel times between all pairs of
        locations from two GeoDataFrames using the road network graph.
        
        Args:
            G (nx.Graph): Road network graph with edge weights
            gdf1 (gpd.GeoDataFrame): Source locations with 'node' column
            gdf2 (gpd.GeoDataFrame): Destination locations with 'node' column  
            weight (str): Edge attribute to use as weight ('travel_time' or 'length')
            
        Returns:
            dict: Mapping from (source_idx, dest_idx) to shortest path cost
        """
        ttm = {}
        for i, d_row in gdf1.iterrows():
            for j, c_row in gdf2.iterrows():
                try:
                    t = nx.shortest_path_length(G, d_row['node'], c_row['node'], weight=weight, method='dijkstra')
                except Exception as e:
                    # Set to infinity if no path exists
                    t = float('inf')
                ttm[(i, j)] = t
        return ttm
    
    def build_travel_time_matrix_d2c(self) -> None:
        """
        Build travel time matrix (district -> candidate) in minutes
        """
        # Build travel time matrix (district -> candidate) in minutes
        ttm_district_to_candidate = HospitalPlanner.calculate_ttm(self.G, self.districts_gdf, self.candidates_gdf, 'travel_time')
        self.ttm_district_to_candidate = {k: v / 60 for k, v in ttm_district_to_candidate.items()}  # convert seconds to minutes
    
    def build_travel_time_matrix_eh2d(self) -> None:
        """
        Build travel time matrix (existing hospital -> district) in minutes
        """
        hospital_gdf = get_existing_hospitals_gdf()
        
        # Map existing hospitals to graph nodes
        hospital_gdf['node'] = ox.nearest_nodes(
            self.G,
            hospital_gdf.geometry.x,
            hospital_gdf.geometry.y
        )
        ttm_district_to_existing_hospital = HospitalPlanner.calculate_ttm(self.G, self.districts_gdf, hospital_gdf, 'travel_time')
        self.ttm_district_to_existing_hospital = {k: v / 60 for k, v in ttm_district_to_existing_hospital.items()}  # convert seconds to minutes
    
    def init_baseline_model(self, 
        equity_weight: float = 0.75,
        travel_weight: float = 0.065,
        beds_weight: float = 0.065,
        time_threshold: int = 30,
        max_beds_per_hospital: int = 300,
        min_beds_per_hospital: int = 20,
        max_beds: int = 1500,
    ) -> pulp.LpProblem:
        """
        Initialize the baseline multi-objective optimization model.
        
        Creates a linear programming model that maximizes weighted coverage
        considering demand satisfaction, equity improvement, and travel time
        minimization. Uses maximum coverage formulation with time thresholds.
        
        Args:
            equity_weight (float): Weight for equity improvement objective (0-1)
            travel_weight (float): Weight for travel time minimization objective (0-1)
            beds_weight (float): Weight for bed allocation optimization (0-1)
            time_threshold (int): Maximum travel time in minutes for coverage
            max_beds_per_hospital (int): Maximum number of beds per hospital
            min_beds_per_hospital (int): Minimum number of beds per hospital
            max_beds (int): Maximum total number of beds across all hospitals
            
        Side Effects:
            Sets self.model, self.D, self.P, self.x, self.y, self.z with optimization components
        """
        # Define problem data from existing variables
        D = list(self.districts_gdf.index)  # districts
        P = list(self.candidates_gdf.index)  # candidate locations
        c = self.ttm_district_to_candidate
        
        # Extract and normalize objective metrics
        demand = dict(zip(self.districts_gdf.index, self.districts_gdf['demand']))
        equity_raw = dict(zip(self.districts_gdf.index, self.districts_gdf['equity_index']))

        equity_norm = self.normalize_metric(equity_raw)
        demand_norm = self.normalize_metric(demand)
        c_norm = self.normalize_metric(c)
        
        # Coverage matrix: 1 if candidate can serve district within time threshold
        a = {(d, p): int(c[(d, p)] <= time_threshold) for d in D for p in P}
        
        # Initialize optimization model
        model = pulp.LpProblem("Equity_Supply_Travel_MaxCover", pulp.LpMaximize)

        # Decision variables
        x = pulp.LpVariable.dicts("open", P, cat="Binary")  # 1 if hospital opened at candidate p
        y = pulp.LpVariable.dicts("assign", (D, P), cat="Binary")  # 1 if district d assigned to hospital p
        z = pulp.LpVariable.dicts("bed_allocation", P,                 
                                lowBound=0.0, 
                                upBound=max_beds_per_hospital,
                                cat="Integer") 
        # number of beds allocated to hospital p
        beds_reward = {
            p: sum(demand_norm[d] * a[(d, p)] for d in D)
            for p in P
        }
        bed_term = pulp.lpSum(
            beds_reward[p] * z[p]/max_beds_per_hospital
            for p in P
        )
        model += (
            equity_weight * pulp.lpSum(
                equity_norm[d] * y[d][p] 
                for d in D 
                for p in P 
            )
            - travel_weight * pulp.lpSum(
                c_norm[(d, p)] * y[d][p]       
                for d in D 
                for p in P 
            )
            + beds_weight * bed_term
        ), "Total_Score"
        
        # Constraints
        for d in D:
            for p in P:
                # Can only assign to open hospitals
                model += y[d][p] <= x[p], f"Assign_if_open_{d}_{p}"
                # Can only assign within time threshold
                model += y[d][p] <= a[(d, p)], f"Assign_within_Tmax_{d}_{p}"

        for p in P:
            # Can only allocate beds if hospital is open
            model += z[p] <= x[p] * max_beds_per_hospital, f"Beds_if_open_{p}"
            model += z[p] >= min_beds_per_hospital * x[p], f"Beds_min_if_open_{p}"
            
        # Total allocated beds should not exceed max_beds
        model += pulp.lpSum(z[p] for p in P) <= max_beds, "Total_Beds_Limit"

        self.model = model
        self.D = D
        self.P = P
        self.x = x
        self.y = y
        self.z = z
        
    def build_distance_metric_c2eh(self) -> None:
        hospital_gdf = get_existing_hospitals_gdf()
        
        # Map existing hospitals to graph nodes
        hospital_gdf['node'] = ox.nearest_nodes(
            self.G,
            hospital_gdf.geometry.x,
            hospital_gdf.geometry.y
        )
        
        # Calculate distances from candidates to existing hospitals
        self.distance_existing_hospitals = HospitalPlanner.calculate_ttm(
            self.G,
            self.candidates_gdf,
            hospital_gdf,
            'length'
        )
        self.hospital_gdf = hospital_gdf
    
    def add_existing_hospitals_constraints(self, 
        supply_weight: float = 0.065, 
        num_neighbors: int = 0,
        distance_threshold: int = 7_000
        ) -> pulp.LpProblem:
        """
        Add constraints considering existing hospital infrastructure.
        
        Modifies the optimization model to account for existing hospitals by:
        1. Adding supply oversaturation penalty
        2. Preventing placement too close to existing facilities
        
        Args:
            gdf (gpd.GeoDataFrame): Existing hospitals with 'MaxBeds' column
            supply_weight (float): Weight for supply oversaturation penalty (0-1)
            distance_threshold (int): Minimum distance from existing hospitals (meters)
            num_neighbors (int): Maximum allowed existing hospital neighbors
            
        Side Effects:
            Modifies self.model by adding supply penalty and proximity constraints
        """
        candidates_gdf = self.candidates_gdf.copy()
        
        # Find existing hospital neighbors for each candidate
        is_existing_hospitals_neighbors = {}
        for h in candidates_gdf.index: 
            is_existing_hospitals_neighbors[h] = []
            for (i, j), dist in self.distance_existing_hospitals.items():
                if dist <= distance_threshold and i == h and i != j:
                    is_existing_hospitals_neighbors[h].append(j)
        # Calculate total supply (existing + new hospital capacity)
        candidates_gdf['total_supply'] = [
            self.hospital_gdf.iloc[is_existing_hospitals_neighbors[h]]['bed_allocation'].sum() + 1 
            for h in candidates_gdf.index
        ]
        
        supply_raw = dict(zip(candidates_gdf.index, candidates_gdf['total_supply']))
        supply_norm = self.normalize_metric(supply_raw)
        
        # Add supply oversaturation penalty to objective
        self.model.objective -= (
            supply_weight * pulp.lpSum(
                supply_norm[p] * self.x[p] 
                for p in self.P 
            )
        )
        
        # Add proximity constraints: prevent placement near existing hospitals
        for p in self.P:
            if len(is_existing_hospitals_neighbors[p]) >= num_neighbors:
                self.model += self.x[p] == 0, f"Close_if_has_more_than_{num_neighbors}_existing_hospitals_neighbors_{p}"
    
    def build_distance_metric_c2c(self) -> None:
        self.is_candidate_neighbors = HospitalPlanner.calculate_ttm(
            self.G,
            self.candidates_gdf,
            self.candidates_gdf,
            'length'
        )
    
    def add_neighbor_constraints(self, distance_threshold: int = 7_000) -> pulp.LpProblem:
        """
        Add constraints to prevent placing hospitals too close to each other.
        
        Creates mutual exclusion constraints between candidate locations that are
        within the specified distance threshold to avoid oversaturation.
        
        Args:
            distance_threshold (int): Minimum distance between new hospitals (meters)
            
        Side Effects:
            Modifies self.model by adding neighbor exclusion constraints
        """
        is_candidate_neighbors = {k: v for k, v in self.is_candidate_neighbors.items() if v <= distance_threshold}
        
        # Add mutual exclusion constraints for neighboring candidates
        for (p1, p2), is_neighbor in is_candidate_neighbors.items():
            if p1 != p2 and is_neighbor == 1:
                self.model += self.x[p1] + self.x[p2] <= 1, f"No_candidate_neighbors_{p1}_{p2}"
    
    def set_num_predictions(self, k: int) -> pulp.LpProblem:
        """
        Set the exact number of hospitals to be built.
        
        Adds or updates the constraint specifying how many hospitals should be
        opened in the optimal solution.
        
        Args:
            k (int): Number of hospitals to build
            
        Side Effects:
            Modifies self.model by adding/updating facility count constraint
        """
        # Remove existing constraint if present
        if "Open_k_Facilities" in self.model.constraints:
            self.model.constraints.pop("Open_k_Facilities")
        
        # Add new constraint for exact number of facilities
        self.model += pulp.lpSum(self.x[p] for p in self.P) == k, "Open_k_Facilities"
        
    def predict_hospital_locations(self) -> tuple[list[int], list[tuple[int, int]]]:
        """
        Solve the optimization model and return optimal hospital locations.
        
        Executes the linear programming solver to find the optimal solution
        and extracts hospital locations and district assignments.
        
        Returns:
            tuple[list[int], list[tuple[int, int]]]: 
                - List of selected candidate indices for hospital placement
                - List of (district, hospital) assignment pairs
                
        Raises:
            Exception: If the optimization model fails to solve optimally
        """
        # Solve the optimization model
        self.model.solve(pulp.PULP_CBC_CMD(msg=False))
        
        # Extract solution: selected hospital locations
        selected = [p for p in self.P if self.x[p].value() > 0.5]
        
        # Extract solution: district-hospital assignments  
        assigns = [(d, p) for d in self.D for p in self.P if self.y[d][p].value() > 0.5]
        
        # Extract solution: bed allocations
        beds = {p: self.z[p].value() for p in self.P}
        
        return selected, assigns, beds

    def build_matrix(self):
        self.build_travel_time_matrix_d2c()
        self.build_travel_time_matrix_eh2d()
        self.build_distance_metric_c2eh()
        self.build_distance_metric_c2c()
    
    def compute_loss(self, 
        selected_candidates: gpd.GeoDataFrame, 
        hospital_gdf: gpd.GeoDataFrame,
        ) -> dict[str, float]:
        """
        Compute loss function for each district
        """
        loss = {}
        district_codes = self.districts_gdf['district_code'].unique().tolist()
        for district_code in district_codes:
            centroids = self.districts_gdf[self.districts_gdf['district_code'] == district_code]['centroid']
            candidates = selected_candidates[selected_candidates['district_code'] == district_code]
            hospitals = hospital_gdf[hospital_gdf['district_code'] == district_code]
            temp_cent = []
            if len(candidates) > 0 or len(hospitals) > 0:
                for centroid in centroids.index:
                    temp = []
                    for candidate in candidates.index:
                        travel_time = self.ttm_district_to_candidate[(centroid, candidate)]
                        temp.append(travel_time if travel_time < float('inf') else 0)
                    for hospital in hospitals.index:
                        travel_time = self.ttm_district_to_existing_hospital[(centroid, hospital)]
                        temp.append(travel_time if travel_time < float('inf') else 0)
                    if len(temp) > 0:
                        temp_cent.append(sum(temp) / len(temp))
                loss[district_code] = sum(temp_cent) / len(temp_cent) if len(temp_cent) > 0 else 0
        return sum(list(loss.values())) / len(loss)
    
    def predict_location(self, 
        weights: dict[str, float],
        time_threshold: int = 30,
        max_beds_per_hospital: int = 300,
        min_beds_per_hospital: int = 20,
        max_beds: int = 1500,
        num_neighbors: int = 0,
        eh_distance_threshold: int = 7_000,
        c2c_distance_threshold: int = 7_000,
        k: int = 10,
    ):
        """
        Predict optimal hospital locations using the configured optimization model.
        
        Args:
            weights (dict[str, float]): Dictionary of optimization weights:
                - equity_weight: Weight for equity improvement
                - travel_weight: Weight for travel time minimization
                - beds_weight: Weight for bed allocation optimization
                - supply_weight: Weight for supply oversaturation penalty
            time_threshold (int): Maximum travel time in minutes for coverage
            max_beds_per_hospital (int): Maximum number of beds per hospital
            min_beds_per_hospital (int): Minimum number of beds per hospital
            max_beds (int): Maximum total number of beds across all hospitals
            num_neighbors (int): Maximum allowed existing hospital neighbors
            eh_distance_threshold (int): Minimum distance from existing hospitals (meters)
            c2c_distance_threshold (int): Minimum distance between new hospitals (meters)
            k (int): Number of hospitals to build
            
        Returns:
            tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]: 
                - Selected candidate locations with bed allocations
                - Existing hospital locations
        """
        self.init_baseline_model(
            equity_weight=weights['equity_weight'],
            travel_weight=weights['travel_weight'],
            beds_weight=weights['beds_weight'],
            time_threshold=time_threshold,
            max_beds_per_hospital=max_beds_per_hospital,
            min_beds_per_hospital=min_beds_per_hospital,
            max_beds=max_beds,
        )
        self.add_existing_hospitals_constraints(
            supply_weight=weights['supply_weight'],
            num_neighbors=num_neighbors,
            distance_threshold=eh_distance_threshold,
        )
        self.add_neighbor_constraints(
            distance_threshold=c2c_distance_threshold,
        )
        self.set_num_predictions(k)
        selected, assigns, beds = self.predict_hospital_locations()
        selected_candidates = self.candidates_gdf.iloc[selected]
        selected_candidates['bed_allocation'] = selected_candidates.index.map(lambda id: beds[id])
        selected_candidates = selected_candidates.assign(
            Lat=selected_candidates.geometry.y,
            Lon=selected_candidates.geometry.x
        )
        hospital_gdf = self.hospital_gdf.copy()
        hospital_gdf['type'] = 'existing'
        selected_candidates['type'] = 'predicted'

        return selected_candidates, hospital_gdf
    
    def run(self, 
        initial_weights: dict[str, float], 
        step_size: float,
        num_neighbors: int = 0,
        eh_distance_threshold: int = 7_000,
        c2c_distance_threshold: int = 7_000,
        time_threshold: int = 30,
        max_beds_per_hospital: int = 300,
        min_beds_per_hospital: int = 20,
        max_beds: int = 1500,
        k: int = 10,
    ):
        """
        Run the hospital location optimization using CMA-ES.
        
        Uses Covariance Matrix Adaptation Evolution Strategy (CMA-ES) to find
        optimal weights for the multi-objective optimization model.
        
        Args:
            initial_weights (dict[str, float]): Initial optimization weights
            step_size (float): Initial step size for CMA-ES
            num_neighbors (int): Maximum allowed existing hospital neighbors
            eh_distance_threshold (int): Minimum distance from existing hospitals (meters)
            c2c_distance_threshold (int): Minimum distance between new hospitals (meters)
            time_threshold (int): Maximum travel time in minutes for coverage
            max_beds_per_hospital (int): Maximum number of beds per hospital
            min_beds_per_hospital (int): Minimum number of beds per hospital
            max_beds (int): Maximum total number of beds across all hospitals
            k (int): Number of hospitals to build
            
        Returns:
            gpd.GeoDataFrame: Selected candidate locations with optimal bed allocations
        """
        weights = list(initial_weights.values())
        losses = []
        es = cma.CMAEvolutionStrategy(x0=weights, sigma0=step_size)
        while not es.stop():
            # generate λ candidate weight-vectors
            solutions = es.ask()
            # evaluate in parallel however you like
            for solution in solutions:
                weights = dict(zip(initial_weights.keys(), solution))
                selected_candidates, hospital_gdf = self.predict_location(weights,
                    num_neighbors=num_neighbors,
                    eh_distance_threshold=eh_distance_threshold,
                    c2c_distance_threshold=c2c_distance_threshold,
                    time_threshold=time_threshold,
                    max_beds_per_hospital=max_beds_per_hospital,
                    min_beds_per_hospital=min_beds_per_hospital,
                    max_beds=max_beds,
                    k=k,
                )
                loss = self.compute_loss(selected_candidates, hospital_gdf)
                losses.append(loss)
            # feed back to CMA-ES
            es.tell(solutions, losses)
            es.disp()     # optional progress print

        # final best
        optimal_weight = es.best.get()[0]
        weights = dict(zip(initial_weights.keys(), optimal_weight))
        selected_candidates, hospital_gdf = self.predict_location(weights,
            num_neighbors=num_neighbors,
            eh_distance_threshold=eh_distance_threshold,
            c2c_distance_threshold=c2c_distance_threshold,
            time_threshold=time_threshold,
            max_beds_per_hospital=max_beds_per_hospital,
            min_beds_per_hospital=min_beds_per_hospital,
            max_beds=max_beds,
            k=k,
        )
        return selected_candidates