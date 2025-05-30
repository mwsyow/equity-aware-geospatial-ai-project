# Forecasting District‑Level Hospital Demand in Saarland

---

## 1. Introduction

This project forecasts hospital demand at the district level in Saarland, Germany, from 2000 through 2030. By combining historical inpatient counts, demographic data, and disease incidence, we construct a confidence‑weighted index to capture projected inpatient utilization for each district.

## 2. Defining Hospital Demand

We measure **hospital demand** as the number of inpatients. To control for population changes, we calculate demand on a per-capita basis before scaling back to absolute projections by region and district.

## 3. Project Scope

* **Region:** All six Saarland districts (Landkreise):

  * Stadtverband Saarbrücken (10041)
  * Merzig‑Wadern (10042)
  * Neunkirchen (10043)
  * Saarlouis (10044)
  * Saarpfalz‑Kreis (10045)
  * Sankt Wendel (10046)
* **Timeframe:**

  * Historical: 2000–2021
  * Forecast: 2022–2030

## 4. Key Assumptions & Limitations

* **Geographic granularity:** Inpatient and population data are available only at the state level; both must be apportioned to districts via proxy indicators, which may introduce estimation error.
* **Proxy metrics:** We use notifiable infectious‑disease counts to allocate demand, acknowledging that non‑infectious conditions and other factors also drive hospitalizations.

## 5. Data Sources

1. **Inpatient Counts:** Annual totals (Destatis).
2. **Population Data:** Annual Saarland population figures, both historical and projected (Destatis).
3. **Disease Incidence:** District‑level case counts under the German Infection Protection Act (IfSG).

## 6. Methodology

Our forecasting approach unfolds in three stages:

### 6.1 Regional Per‑Capita Demand

1. **Compute per‑capita rate** $D_t$:

   $$
   D_t = \frac{H_t}{P_t},
   $$

   where $H_t$ is total inpatients and $P_t$ is population in year $t$.
2. **Model** $D_t$ (2000–2021) and **forecast** through 2030 using ARIMA with optimal hyperparameters.
3. **Derive regional demand**:

   $$
   \widehat{H}^{\text{region}}_t = D^{\text{forecast}}_t \times P^{\text{proj}}_t.
   $$

### 6.2 District‑Level Disaggregation

1. **Forecast** district disease counts $C_{i,t}$ to 2030 via ARIMA, yielding 95% confidence intervals $\bigl[L_{i,t}, U_{i,t}\bigr]$, for each district $i$ in year $t$.
2. **Normalize**:

   $$
   \tilde{C}_{i,t} = \frac{C_{i,t}}{\sum_j C_{j,t}}.
   $$
3. **Allocate demand**:

   $$
   h_{i,t} = \tilde{C}_{i,t} \times \widehat{H}^{\text{region}}_t.
   $$

### 6.3 Confidence‑Weighted Index

To account for forecast uncertainty:

1. **Confidence weight**:

   $$
   w_{i,t} = \max\bigl(0,1 - \frac{U_{i,t} - L_{i,t}}{C_{i,t}}\bigr).
   $$
2. **Weighted aggregates**:

   $$
   S_t = \sum_i w_{i,t} h_{i,t}, \quad W_t = \sum_i w_{i,t}.
   $$
3. **Index**:

   $$
   I_i = \frac{S_i}{W_i}.
   $$
4. **Normalize and Invert** $I_i$ to \[0,1]:

   $$
   \tilde{I}_{i} = 1 - \frac{I_{i}}{\sum_j I_{j}}.
   $$

---
