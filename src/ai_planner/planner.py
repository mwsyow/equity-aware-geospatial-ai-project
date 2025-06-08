import networkx as nx
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import osmnx as ox
import pulp

CRS_NUM = 4326
CRS = f"EPSG:{CRS_NUM}"
DEFAULT_GRAPH_LOCATION = "Saarland, Germany"

class HospitalPlanner:
    def __init__(self, districts_gdf: gpd.GeoDataFrame):
        self.districts_gdf = districts_gdf.to_crs(epsg=CRS_NUM)
        
    @staticmethod
    def generate_points(centroid, radius_km, n, bound_poly) -> list[Point]:
        r_deg = radius_km / 111.0  # Rough conversion
        points = []
        for _ in range(n):
            r = r_deg * np.sqrt(np.random.rand())
            theta = np.random.rand() * 2 * np.pi
            dx, dy = r * np.cos(theta), r * np.sin(theta)
            
            cand = Point(centroid.x + dx, centroid.y + dy)
            
            if bound_poly.contains(cand):
                points.append(cand)
        return points
    
    @staticmethod
    def normalize_metric(metric: dict) -> dict:
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
    
    def generate_candidates(self, n_samples_per_centroid: int,radius_km: int):
        #STEP2: Generate Candidate Hospital Locations Around Districts
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
        if graph_path is None:
            G = ox.graph_from_place(DEFAULT_GRAPH_LOCATION, network_type='drive')
        else:
            G = ox.load_graphml(graph_path)
            
        G = ox.project_graph(G, to_crs=CRS)

        # Add travel time
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
        ttm = {}
        for i, d_row in gdf1.iterrows():
            for j, c_row in gdf2.iterrows():
                try:
                    t = nx.shortest_path_length(G, d_row['node'], c_row['node'], weight=weight, method='dijkstra')
                except Exception as e:
                    t = float('inf')
                ttm[(i, j)] = t
        return ttm
    
    def init_baseline_model(self, 
                      demand_weight: float = 0.065,
                      equity_weight: float = 0.75,
                      travel_weight: float = 0.065,
                      time_threshold: int = 30) -> pulp.LpProblem:
        # Define problem data from your existing variables
        D = list(self.districts_gdf.index)  # districts
        P = list(self.candidates_gdf.index)  # candidate locations

        # Travel time matrix (already computed)
        c = HospitalPlanner.build_ttm(self.G, self.districts_gdf, self.candidates_gdf, 'travel_time') # (district, candidate) -> travel time
        c = {k: v / 60 for k, v in c.items()} # convert to minutes
        # Demand, equity, and supply data
        demand = dict(zip(self.districts_gdf.index, self.districts_gdf['demand']))
        equity_raw = dict(zip(self.districts_gdf.index, self.districts_gdf['equity_index']))

        equity_norm = self.normalize_metric(equity_raw)
        demand_norm = self.normalize_metric(demand)
        c_norm = self.normalize_metric(c)
        a = {(d, p): int(c[(d, p)] <= time_threshold) for d in D for p in P}
        
        # ---- 3. BUILD THE MODEL ----

        model = pulp.LpProblem("Equity_Supply_Travel_MaxCover", pulp.LpMaximize)

        # Decision vars
        x = pulp.LpVariable.dicts("open", P, cat="Binary")
        y = pulp.LpVariable.dicts("assign", (D, P), cat="Binary")
        
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
        
        for d in D:
            for p in P:
                model += y[d][p] <= x[p], f"Assign_if_open_{d}_{p}"
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
        candidates_gdf = self.candidates_gdf.copy()
        hospital_gdf = gdf.copy()
        hospital_gdf['node'] = ox.nearest_nodes(
            self.G,
            hospital_gdf.geometry.x,
            hospital_gdf.geometry.y
        )
        distance_existing_hospitals = HospitalPlanner.build_ttm(
            self.G,
            candidates_gdf,
            hospital_gdf,
            'length'
        )
        
        is_existing_hospitals_neighbors = {}
        for h in candidates_gdf.index: 
            is_existing_hospitals_neighbors[h] = []
            for (i, j), dist in distance_existing_hospitals.items():
                if dist <= distance_threshold and i == h and i != j:
                    is_existing_hospitals_neighbors[h].append(j)
        candidates_gdf['total_supply'] =[
            hospital_gdf.iloc[is_existing_hospitals_neighbors[h]]['MaxBeds'].sum() + 1 
            for h in candidates_gdf.index
        ]
        supply_raw = dict(zip(candidates_gdf.index, candidates_gdf['total_supply']))
        supply_norm = self.normalize_metric(supply_raw)
        
        self.model += (
            (1 - supply_weight * pulp.lpSum(
                supply_norm[p] * self.x[p] 
                for p in self.P 
            ))
        ), "Total_Supply"
        for p in self.P:
            if len(is_existing_hospitals_neighbors[p]) >= num_neighbors:
                self.model += self.x[p] == 0, f"Close_if_has_more_than_{num_neighbors}_existing_hospitals_neighbors_{p}"
    
    def add_neighbor_constraints(self, distance_threshold: int = 7_000) -> pulp.LpProblem:
        is_candidate_neighbors = HospitalPlanner.build_ttm(
            self.G,
            self.candidates_gdf,
            self.candidates_gdf,
            'length'
        )
        is_candidate_neighbors = {k: v for k, v in is_candidate_neighbors.items() if v <= distance_threshold}
        # No two open candidates can be neighbors
        for (p1, p2), is_neighbor in is_candidate_neighbors.items():
            if p1 != p2 and is_neighbor == 1:
                self.model += self.x[p1] + self.x[p2] <= 1, f"No_candidate_neighbors_{p1}_{p2}"
    
    def set_num_predictions(self, k: int) -> pulp.LpProblem:
        if "Open_k_Facilities" in self.model.constraints:
            self.model.constraints.pop("Open_k_Facilities")
        self.model += pulp.lpSum(self.x[p] for p in self.P) == k, "Open_k_Facilities"
        
    def predict(self) -> tuple[list[int], list[tuple[int, int]]]:
        self.model.solve(pulp.PULP_CBC_CMD(msg=False))
        selected = [p for p in self.P if self.x[p].value() > 0.5]
        assigns = [(d, p) for d in self.D for p in self.P if self.y[d][p].value() > 0.5]
        return selected, assigns
        