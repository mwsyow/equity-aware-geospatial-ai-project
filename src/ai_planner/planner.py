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

Dependencies:
    - networkx: For road network graph operations
    - numpy: For numerical computations
    - pandas: For data manipulation
    - geopandas: For geospatial data handling
    - shapely: For geometric operations
    - osmnx: For OpenStreetMap data retrieval
    - pulp: For linear programming optimization
"""

import networkx as nx
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import osmnx as ox
import pulp

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

        candidates = []
        for _, row in self.districts_gdf.iterrows():
            candidates += HospitalPlanner.generate_points(
                row['centroid'], 
                radius_km=radius_km, 
                n=n_samples_per_centroid, 
                bound_poly=poly_utm
            )

        self.candidates_gdf = gpd.GeoDataFrame(geometry=candidates, crs=CRS)
    
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
    def build_ttm(G: nx.Graph, gdf1: gpd.GeoDataFrame, gdf2: gpd.GeoDataFrame, weight: str) -> dict:
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
    
    def init_baseline_model(self, 
                      demand_weight: float = 0.065,
                      equity_weight: float = 0.75,
                      travel_weight: float = 0.065,
                      time_threshold: int = 30) -> pulp.LpProblem:
        """
        Initialize the baseline multi-objective optimization model.
        
        Creates a linear programming model that maximizes weighted coverage
        considering demand satisfaction, equity improvement, and travel time
        minimization. Uses maximum coverage formulation with time thresholds.
        
        Args:
            demand_weight (float): Weight for demand coverage objective (0-1)
            equity_weight (float): Weight for equity improvement objective (0-1)  
            travel_weight (float): Weight for travel time minimization objective (0-1)
            time_threshold (int): Maximum travel time in minutes for coverage
            
        Side Effects:
            Sets self.model, self.D, self.P, self.x, self.y with optimization components
        """
        # Define problem data from existing variables
        D = list(self.districts_gdf.index)  # districts
        P = list(self.candidates_gdf.index)  # candidate locations

        # Build travel time matrix (district -> candidate) in minutes
        c = HospitalPlanner.build_ttm(self.G, self.districts_gdf, self.candidates_gdf, 'travel_time')
        c = {k: v / 60 for k, v in c.items()}  # convert seconds to minutes
        
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
        
        # Multi-objective function: maximize weighted sum of objectives
        model += (
            demand_weight * pulp.lpSum(
                demand_norm[d] * y[d][p] 
                for d in D 
                for p in P 
            )
            + equity_weight * pulp.lpSum(
                equity_norm[d] * y[d][p] 
                for d in D 
                for p in P 
            )
            + (1 - travel_weight * pulp.lpSum(
                c_norm[(d, p)] * y[d][p]       
                for d in D 
                for p in P 
            ))
        ), "Total_Score"
        
        # Constraints
        for d in D:
            for p in P:
                # Can only assign to open hospitals
                model += y[d][p] <= x[p], f"Assign_if_open_{d}_{p}"
                # Can only assign within time threshold
                model += y[d][p] <= a[(d, p)], f"Assign_within_Tmax_{d}_{p}"

        self.model = model
        self.D = D
        self.P = P
        self.x = x
        self.y = y
    
    def add_existing_hospitals_constraints(self, 
        gdf: gpd.GeoDataFrame, 
        supply_weight: float = 0.065, 
        distance_threshold: int = 7_000,
        num_neighbors: int = 0
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
        hospital_gdf = gdf.copy()
        
        # Map existing hospitals to graph nodes
        hospital_gdf['node'] = ox.nearest_nodes(
            self.G,
            hospital_gdf.geometry.x,
            hospital_gdf.geometry.y
        )
        
        # Calculate distances from candidates to existing hospitals
        distance_existing_hospitals = HospitalPlanner.build_ttm(
            self.G,
            candidates_gdf,
            hospital_gdf,
            'length'
        )
        
        # Find existing hospital neighbors for each candidate
        is_existing_hospitals_neighbors = {}
        for h in candidates_gdf.index: 
            is_existing_hospitals_neighbors[h] = []
            for (i, j), dist in distance_existing_hospitals.items():
                if dist <= distance_threshold and i == h and i != j:
                    is_existing_hospitals_neighbors[h].append(j)
        
        # Calculate total supply (existing + new hospital capacity)
        candidates_gdf['total_supply'] = [
            hospital_gdf.iloc[is_existing_hospitals_neighbors[h]]['MaxBeds'].sum() + 1 
            for h in candidates_gdf.index
        ]
        
        supply_raw = dict(zip(candidates_gdf.index, candidates_gdf['total_supply']))
        supply_norm = self.normalize_metric(supply_raw)
        
        # Add supply oversaturation penalty to objective
        self.model += (
            (1 - supply_weight * pulp.lpSum(
                supply_norm[p] * self.x[p] 
                for p in self.P 
            ))
        ), "Total_Supply"
        
        # Add proximity constraints: prevent placement near existing hospitals
        for p in self.P:
            if len(is_existing_hospitals_neighbors[p]) >= num_neighbors:
                self.model += self.x[p] == 0, f"Close_if_has_more_than_{num_neighbors}_existing_hospitals_neighbors_{p}"
    
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
        # Calculate distances between all candidate pairs
        is_candidate_neighbors = HospitalPlanner.build_ttm(
            self.G,
            self.candidates_gdf,
            self.candidates_gdf,
            'length'
        )
        
        # Filter to only include neighbors within threshold
        is_candidate_neighbors = {k: v for k, v in is_candidate_neighbors.items() if v <= distance_threshold}
        
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
        
    def predict(self) -> tuple[list[int], list[tuple[int, int]]]:
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
        
        return selected, assigns
    
    def compute_equity_metric(self) -> float:
        """
        Compute an equity disparity metric after optimization:
        the difference between max and min district coverage ratios.
        """
        coverage = {}
        # Calculate coverage fraction per district safely (check variable values)
        for d in self.D:
            coverage[d] = sum(
                (self.y[d][p].value() if self.y[d][p] is not None and self.y[d][p].value() is not None else 0)
                * self.districts_gdf.loc[d, 'demand']
                for p in self.P
            )
        vals = list(coverage.values())
        max_cov = max(vals)
        min_cov = min(vals)
        disparity = max_cov - min_cov
        # Save coverage info for monitoring/logging if needed
        self.last_coverage = coverage
        self.last_max_coverage = max_cov
        self.last_min_coverage = min_cov
        return disparity

    def optimize_with_agent(self,
                            demand_weight=0.065,
                            equity_weight=0.75,
                            travel_weight=0.065,
                            time_threshold=30,
                            equity_tolerance=0.05,
                            max_iter=5,
                            adaptive_factor=1.2,
                            decay_factor=0.9):
        """
        Agentic optimization loop: iteratively adjust weights/time threshold
        based on observed equity disparity until metrics are satisfied or max_iter reached.
        
        This implements an autonomous feedback loop:
        - Observe outcome via equity metric
        - Evaluate vs. tolerance
        - Adapt weights and constraints dynamically
        - Replan optimization model
        """

        # Initialize current parameters and history
        cur_dw, cur_ew, cur_tw = demand_weight, equity_weight, travel_weight
        cur_tt = time_threshold
        self.current_weights = {
            'demand_weight': cur_dw,
            'equity_weight': cur_ew,
            'travel_weight': cur_tw,
            'time_threshold': cur_tt
        }
        self.history = []

        for i in range(max_iter):
            # Build and solve model with current weights and constraints
            self.init_baseline_model(cur_dw, cur_ew, cur_tw, cur_tt)
            self.add_existing_hospitals_constraints(gdf_existing, supply_weight=cur_dw)
            self.add_neighbor_constraints(distance_threshold=7000)

            selected, assigns = self.predict()
            disparity = self.compute_equity_metric()

            # Collect metrics for monitoring and debugging
            max_cov = self.last_max_coverage
            min_cov = self.last_min_coverage

            self.history.append({
                'iter': i,
                'equity_weight': cur_ew,
                'demand_weight': cur_dw,
                'travel_weight': cur_tw,
                'time_threshold': cur_tt,
                'disparity': disparity,
                'max_coverage': max_cov,
                'min_coverage': min_cov,
            })

            # Stop if equity disparity within tolerance
            if disparity <= equity_tolerance:
                return {
                    'open_sites': selected,
                    'assignments': assigns,
                    'history': self.history,
                    'final_weights': self.current_weights
                }

            # --- Agentic adaptive adjustment ---
            # Increase equity emphasis proportionally to observed disparity
            cur_ew *= adaptive_factor

            # Adjust accessibility threshold inversely to equity emphasis
            cur_tt *= (1 / adaptive_factor)

            # Optional: modestly adapt demand and travel weights towards balancing objectives
            # For example, reduce travel_weight if equity is poor to prioritize equity more
            cur_tw *= decay_factor if disparity > equity_tolerance else 1.0
            # Increase demand weight slightly to keep demand coverage relevant
            cur_dw *= (1 + (disparity * 0.01))

            # Apply decay to prevent runaway weights
            cur_ew *= decay_factor
            cur_dw *= decay_factor
            cur_tw *= decay_factor

            # Update current_weights for external access or further logic
            self.current_weights = {
                'demand_weight': cur_dw,
                'equity_weight': cur_ew,
                'travel_weight': cur_tw,
                'time_threshold': cur_tt
            }

        # If no convergence after max_iter iterations
        raise RuntimeError(f"Agentic optimization did not converge after {max_iter} iterations.")
        