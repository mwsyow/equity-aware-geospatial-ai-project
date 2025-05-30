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

COST_PER_BED = 1500  # Example cost per bed 

class AgenticPlanner:
    def __init__(self, hospitals, districts, travel_time,
                 budget_beds, max_open_sites, alpha=0.7, max_travel=30):
        """
        Initialize planner with:
        - hospitals: DataFrame with SiteID, Lon, Lat, CostPerBed, MaxBeds
        - districts: DataFrame with AGS_CODE, Demand, EquityIndex, centroid (geometry)
        - travel_time: dict of dicts: travel_time[site][district] in minutes
        - budget_beds: total beds available to allocate
        - max_open_sites: max hospitals that can be open simultaneously
        - alpha: weight for cost vs equity penalty in objective
        - max_travel: max travel time threshold (minutes)
        """
        self.hospitals  = hospitals.set_index("SiteID")
        self.districts  = districts.set_index("AGS_CODE")
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

    def propose_actions(self):
        """
        Generate all possible next actions:
        - Open new site if under max_sites
        - Add beds in increments (50,100,200) to open sites within budget and max
        """
        if len(self.open_sites) < self.max_sites:
            for s in self.hospitals.index:
                if s not in self.open_sites:
                    yield ("open", s, 0)

        for s in self.open_sites:
            max_more = min(
                self.hospitals.loc[s,"MaxBeds"] - self.beds_alloc[s],
                self.budget - sum(self.beds_alloc.values())
            )
            for delta in [50, 100, 200]:
                if delta <= max_more:
                    yield ("add_beds", s, delta)

    def run(self):
        """
        Greedy optimization loop:
        - Evaluate all possible single-step actions
        - Pick the one improving objective most
        - Stop if no improvement
        """
        best_score, _ = self.evaluate()
        improved = True

        while improved:
            improved = False
            candidates = []
            for act, site, delta in self.propose_actions():
                if act == "open":
                    self.open_sites.add(site)
                else:
                    self.beds_alloc[site] += delta

                score, _ = self.evaluate()
                candidates.append((score, act, site, delta))

                # Undo action
                if act == "open":
                    self.open_sites.remove(site)
                else:
                    self.beds_alloc[site] -= delta

            candidates.sort(key=lambda x: x[0])
            best_cand = candidates[0]
            if best_cand[0] < best_score:
                best_score, act, site, delta = best_cand
                print(f"Applying {act} on {site} (+{delta}) → score {best_score:.3f}")
                if act == "open":
                    self.open_sites.add(site)
                else:
                    self.beds_alloc[site] += delta
                improved = True

        final_score, final_cov = self.evaluate()
        return {
            "open_sites": self.open_sites,
            "beds_alloc": self.beds_alloc,
            "coverage": final_cov,
            "objective": final_score
        }
    
    
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

    def predict_new_hospitals(self):
        """
        Predict new hospital locations by clustering district centroids.
        Assign average bed capacity and cost.
        Estimate travel time by Euclidean distance.
        """
        # Extract coords (lon, lat) of district centroids
        coords = np.vstack(self.districts["centroid"].apply(lambda p: (p.x, p.y)))

        districts_dict = districts_df.to_dict('list')
        

        # Cluster district centroids to find new hospital locations
        kmeans = KMeans(n_clusters=min(self.predict_new, len(districts_dict["AGS_CODE"])), random_state=42)
        kmeans.fit(coords)
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


    """
        Initialize planner with:
        - hospitals: DataFrame with SiteID, Lon, Lat, CostPerBed, MaxBeds
        - districts: DataFrame with AGS_CODE, Demand, EquityIndex, centroid (geometry)
        - travel_time: dict of dicts: travel_time[site][district] in minutes
        - budget_beds: total beds available to allocate
        - max_open_sites: max hospitals that can be open simultaneously
        - alpha: weight for cost vs equity penalty in objective
        - max_travel: max travel time threshold (minutes)
    """


# Randomize MaxBeds for hospitals


#IMPORT THE DATA SETS AND REPLACE THE DUMMY BELOW 


# Create dummy hospital data
hospitals_df = pd.DataFrame({
    "SiteID": ["H1", "H2"],
    "Lon": [10.0, 10.5],
    "Lat": [50.0, 50.5],
})


hospitals_df_dict = hospitals_df.to_dict('list')

np.random.seed(42)
maxBeds = np.random.randint(200, 1001, size=len(hospitals_df_dict["SiteID"]))

hospitals_df["MaxBeds"] = maxBeds



# Create dummy district data with centroids
districts_df = pd.DataFrame({
    "AGS_CODE": ["D1", "D2", "D3"],
    "Demand": [300, 400, 500],
    "EquityIndex": [1.0, 0.8, 0.4],
    "Lon": [10.1, 10.3, 10.6],
    "Lat": [50.1, 50.2, 50.4]
})
districts_df["centroid"] = districts_df.apply(lambda row: Point(row["Lon"], row["Lat"]), axis=1)
districts_gdf = gpd.GeoDataFrame(districts_df, geometry="centroid")

# Create dummy travel_time dict: travel_time[hospital][district] in minutes
travel_time_dict = {
    "H1": {"D1": 15, "D2": 25, "D3": 35},
    "H2": {"D1": 20, "D2": 10, "D3": 30}
} 

#travel time planner 

# Run planner
planner = AgenticPlannerWithPrediction(
    hospitals_df,
    districts_gdf,
    travel_time_dict,
    budget_beds=1000,
    max_open_sites=2,
    alpha=0.6,
    max_travel=30,
    predict_new=3
)

plan = planner.optimize_with_bayesian()

print("Open sites:", plan["open_sites"])
print("Beds allocation:", plan["beds_alloc"])
print("Objective value:", plan["objective"])

# planner.plot_predicted_hospitals()

planner.plot_predicted_hospitals()