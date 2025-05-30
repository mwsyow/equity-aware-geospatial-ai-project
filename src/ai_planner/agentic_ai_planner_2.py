import geopandas as gpd
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

#EquityIndex = w1 x DemandForecastingIndex + w2 x GISD-Index + w3 x TravelTimeIndex + w4 x AccesibilityIndex

# where AccesibilityIndex = w5 x ElderlyShare + w6 x HospitalCapacityIndex
# This formula is applicable at a district level

#Higher value of EquityIndex should mean worse equity and vice versa
from shapely.geometry import Point
from bayes_opt import BayesianOptimization

COST_PER_BED = 1500  # Average cost per bed in EUR

class AgenticPlanner:
    def __init__(self, hospitals, districts, travel_time,
                 budget_beds, max_open_sites, alpha=0.7, max_travel=30):
        """
        Initialize planner with:
        - hospitals: DataFrame with Lon, Lat, CostPerBed, MaxBeds
        - districts: DataFrame with Demand, EquityIndex, centroid (geometry)
        - travel_time: dict of dicts: travel_time[site][district] in minutes
        - budget_beds: total beds available to allocate
        - max_open_sites: max hospitals that can be open simultaneously
        - alpha: weight for cost vs equity penalty in objective
        - max_travel: max travel time threshold (minutes)
        """
        self.hospitals  = hospitals
        self.districts  = districts
        self.tt         = travel_time
        self.budget     = budget_beds
        self.max_sites  = max_open_sites
        self.alpha      = alpha
        self.max_travel = max_travel

        # Current state: open sites and bed allocations per site
        self.open_sites = set()
        self.beds_alloc = {s: 0 for s in self.hospitals.index}

    def evaluate(self):
        """
        Evaluate current plan:
        - cost_term: sum over open sites of beds_alloc * cost_per_bed
        - penalty_term: weighted equity penalty for districts not covered
        Return combined objective and coverage dict.
        """
        # Cost: total cost for allocated beds
        cost = sum(self.beds_alloc[s] * COST_PER_BED
                   for s in self.open_sites)

        # Coverage per district: sum of reachable beds / demand capped at 1
        cov = {}
        for d, row in self.districts.iterrows():
            supply = sum(self.beds_alloc[s]
                         for s in self.open_sites
                         if self.tt[s][d] <= self.max_travel)
            cov[d] = min(1.0, supply / row["Demand"])

        # Penalty: weighted sum of unmet demand adjusted by equity index
        penalty = sum((1.0 - cov[d]) * self.districts.loc[d, "EquityIndex"]
                      for d in cov)

        return self.alpha * cost + (1 - self.alpha) * penalty, cov
    
    def optimize_with_bayesian(self):
        """Use Bayesian Optimization to find best bed allocation and open sites."""

        def objective_function(**kwargs):

            self.open_sites = set()
            self.beds_alloc = {s: 0 for s in self.hospitals.index}
            total_beds = 0

            sorted_sites = sorted(self.hospitals.index)
            for s in sorted_sites:
                key = f"beds_{s}"
                beds = int(round(kwargs.get(key, 0) / 50)) * 50
                if beds > 0:
                    self.open_sites.add(s)
                    beds = min(beds, self.hospitals.loc[s, "MaxBeds"])
                    self.beds_alloc[s] = beds
                    total_beds += beds

            if total_beds > self.budget or len(self.open_sites) > self.max_sites:
                return -1e9  # Penalty for violating constraints

            score, _ = self.evaluate()
            return -score  # Because BayesianOptimization maximizes
        
        print("Starting Bayesian Optimization...")
        # Define parameter bounds for each hospital's bed allocation

        pbounds = {}
        for s in self.hospitals.index:
            pbounds[f"beds_{s}"] = (0, self.hospitals.loc[s, "MaxBeds"])

        optimizer = BayesianOptimization(
            f=objective_function,
            pbounds=pbounds,
            random_state=42,
            verbose=2
        )

        optimizer.maximize(init_points=5, n_iter=15)

        best_params = optimizer.max["params"]
        for s in self.hospitals.index:
            val = int(round(best_params.get(f"beds_{s}", 0) / 50)) * 50
            if val > 0:
                self.open_sites.add(s)
                self.beds_alloc[s] = val

        final_score, final_cov = self.evaluate()
        return {
            "open_sites": self.open_sites,
            "beds_alloc": self.beds_alloc,
            "coverage": final_cov,
            "objective": final_score
        }

class AgenticPlannerWithPrediction(AgenticPlanner):
    def __init__(self, hospitals, districts, travel_time,
                 budget_beds, max_open_sites, alpha=0.7, max_travel=30,
                 predict_new=0):
        """
        Extends AgenticPlanner:
        - predict_new: number of new hospital locations to predict
        """
        super().__init__(hospitals, districts, travel_time,
                         budget_beds, max_open_sites, alpha, max_travel)
        self.predict_new = predict_new
        if predict_new > 0:
            self.predict_new_hospitals()
        
    @property    
    def predicted_hospitals(self) -> pd.DataFrame:
        """
        Returns DataFrame of predicted hospitals.
        Includes SiteID, Lon, Lat, CostPerBed, MaxBeds.
        """
        return self.hospitals.loc[self.hospitals.index.str.startswith("pred_")]

    def predict_new_hospitals(self):
        """
        Predict new hospital locations by clustering district centroids.
        Assign average bed capacity and cost.
        Estimate travel time by Euclidean distance.
        """
        # Extract district coordinates
        coords_array = self.districts["centroid"].values

        # Handle case where fewer districts exist than number of hospitals to predict
        n_samples = max(self.predict_new, len(coords_array)) * 3  # Oversample to give KMeans variety

        # Population-based weights for sampling
        weights = self.districts["Demand"] / self.districts["Demand"].sum()

        # Sample coordinates with population weighting (with replacement)
        sampled_coords = np.array([
            (coords_array[i].x, coords_array[i].y) for i in np.random.choice(len(coords_array), size=n_samples, p=weights)
        ])

        # Apply KMeans to sampled coordinates
        kmeans = KMeans(n_clusters=self.predict_new, random_state=42)
        kmeans.fit(sampled_coords)
        centers = kmeans.cluster_centers_

        # Calculate averages from existing hospitals
        max_beds_avg = int(self.hospitals["MaxBeds"].mean())
        cost_per_bed_avg = COST_PER_BED  # Assuming a fixed cost per bed for simplicity

        new_hospitals = []
        for i, (lon, lat) in enumerate(centers):
            site_id = f"pred_{i}"
            new_hospitals.append({
                "SiteID": site_id,
                "Lon": lon,
                "Lat": lat,
                "CostPerBed": cost_per_bed_avg,
                "MaxBeds": max_beds_avg
            })

        new_hospitals_df = pd.DataFrame(new_hospitals).set_index("SiteID")

        # Append new hospitals to existing data
        self.hospitals = pd.concat([self.hospitals, new_hospitals_df])

        # Initialize beds allocation for new hospitals
        for site_id in new_hospitals_df.index:
            self.beds_alloc[site_id] = 0

        # Estimate travel times for new hospitals (Euclidean distance scaled)
        for site_id, row in new_hospitals_df.iterrows():
            lon1, lat1 = row["Lon"], row["Lat"]
            self.tt[site_id] = {}
            for d, d_row in self.districts.iterrows():
                lon2, lat2 = d_row["centroid"].x, d_row["centroid"].y
                dist = np.sqrt((lon1 - lon2)**2 + (lat1 - lat2)**2)
                # Example scale: 1 degree approx 111 km, assume avg speed 50 km/h
                travel_time_estimate = (dist * 111) / 50 * 60  # minutes
                self.tt[site_id][d] = travel_time_estimate

    def plot_predicted_hospitals(self):
        """
        Plot districts demand heatmap + existing and predicted hospitals on a map.
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot district demand as colored polygons (lighter = low demand)
        self.districts.plot(column="Demand", cmap="Reds", legend=True,
                            alpha=0.5, ax=ax, edgecolor="k")

        # Plot existing hospitals as blue circles
        orig_hospitals = self.hospitals.loc[~self.hospitals.index.str.startswith("pred_")]
        gdf_orig = gpd.GeoDataFrame(orig_hospitals,
                                    geometry=gpd.points_from_xy(orig_hospitals.Lon, orig_hospitals.Lat))
        gdf_orig.plot(ax=ax, marker="o", color="blue", label="Existing Hospitals", markersize=80)

        # Plot predicted hospitals as green triangles
        pred_hospitals = self.hospitals.loc[self.hospitals.index.str.startswith("pred_")]
        gdf_pred = gpd.GeoDataFrame(pred_hospitals,
                                    geometry=gpd.points_from_xy(pred_hospitals.Lon, pred_hospitals.Lat))
        gdf_pred.plot(ax=ax, marker="^", color="green", label="Predicted Hospitals", markersize=120)

        plt.title("District Demand with Hospital Locations")
        plt.legend()
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.show()