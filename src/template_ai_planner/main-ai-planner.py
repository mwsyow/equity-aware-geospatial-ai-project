
# Parameters
alpha = 0.7  # balance cost vs coverage penalty
MAX_BEDS_DEFAULT = 1000
total_beds_limit = 1500
max_hospitals_to_open = 2

# Max beds per site (could come from data)
max_beds_dict = {'S1': 500, 'S2': 600, 'S3': 450, 'S4': 700}

# Initialize model
model = LpProblem("Hospital_Planning", LpMinimize)

open_site = LpVariable.dicts("OpenSite", hospital_sites['SiteID'], 0, 1, LpBinary)
beds = LpVariable.dicts("Beds", hospital_sites['SiteID'], 0, None, LpInteger)
covered = LpVariable.dicts("Covered", district_data['AGS_CODE'], 0, 1)  # continuous coverage [0-1]

cost_term = lpSum([beds[s] * hospital_sites.loc[hospital_sites['SiteID'] == s, 'CostPerBed'].values[0] for s in hospital_sites['SiteID']])
uncovered_penalty = 1_000_000
penalty_term = lpSum([(1 - covered[d]) * district_data.loc[district_data['AGS_CODE'] == d, 'EquityIndex'].values[0] * uncovered_penalty for d in district_data['AGS_CODE']])

model += alpha * cost_term + (1 - alpha) * penalty_term, "WeightedCostCoverageObjective"

# Constraints
model += lpSum([open_site[s] for s in hospital_sites['SiteID']]) <= max_hospitals_to_open, "MaxHospitals"
model += lpSum([beds[s] for s in hospital_sites['SiteID']]) <= total_beds_limit, "TotalBedsLimit"

for s in hospital_sites['SiteID']:
    max_beds = max_beds_dict.get(s, MAX_BEDS_DEFAULT)
    model += beds[s] <= max_beds * open_site[s], f"MaxBeds_{s}"

for d in district_data['AGS_CODE']:
    covering_sites = [s for s in hospital_sites['SiteID'] if travel_time.at[s, d] <= MAX_TRAVEL_TIME]
    if covering_sites:
        model += lpSum([beds[s] for s in covering_sites]) >= district_data.loc[district_data['AGS_CODE'] == d, 'Demand'].values[0] * covered[d], f"Demand_{d}"
        model += covered[d] <= lpSum([open_site[s] for s in covering_sites]), f"CoverageLimit_{d}"
    else:
        model += covered[d] == 0, f"Uncoverable_{d}"

# Solve with time limit and gap for large problems
model.solve(pulp.PULP_CBC_CMD(timeLimit=300, gapRel=0.01))

print(f"Status: {LpStatus[model.status]}")

selected_sites = [s for s in hospital_sites['SiteID'] if open_site[s].varValue > 0.5]
print(f"Selected hospital sites: {selected_sites}")

total_beds = sum(int(beds[s].varValue) for s in selected_sites)
total_cost = sum(beds[s].varValue * hospital_sites.loc[hospital_sites['SiteID'] == s, 'CostPerBed'].values[0] for s in selected_sites)
weighted_coverage = sum(covered[d].varValue * district_data.loc[district_data['AGS_CODE'] == d, 'EquityIndex'].values[0] for d in district_data['AGS_CODE'])

print(f"Total beds assigned: {total_beds}")
print(f"Total estimated cost: €{total_cost:,.2f}")
print(f"Weighted coverage (EquityIndex): {weighted_coverage:.2f}")

# --- Prepare GeoDataFrame for hospital sites ---
hospital_sites_gdf = gpd.GeoDataFrame(
    hospital_sites,
    geometry=gpd.points_from_xy(hospital_sites['Longitude'], hospital_sites['Latitude']),
    crs="EPSG:4326"
)

# Filter only selected hospital sites
selected_hospitals_gdf = hospital_sites_gdf[hospital_sites_gdf['SiteID'].isin(selected_sites)].copy()

# Add beds assigned from optimization solution
selected_hospitals_gdf['Beds'] = selected_hospitals_gdf['SiteID'].apply(lambda s: int(beds[s].varValue))

# --- Plot ---

fig, ax = plt.subplots(1, 1, figsize=(12, 10))

# Plot districts as background
districts_gdf.boundary.plot(ax=ax, linewidth=0.7, edgecolor='gray')

# Plot hospital sites with circle size proportional to beds
selected_hospitals_gdf.plot(
    ax=ax,
    kind='scatter',
    x='Longitude',
    y='Latitude',
    s=selected_hospitals_gdf['Beds'] / 2,  # Scale size for visualization
    color='red',
    alpha=0.7,
    edgecolor='black',
    label='Selected Hospitals (size ~ beds)'
)

ax.set_title("Selected Hospital Locations with Assigned Bed Capacities")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.legend()

plt.show()