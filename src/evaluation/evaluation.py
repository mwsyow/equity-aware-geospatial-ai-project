#!/usr/bin/env python3
"""
evaluate_and_plot_all_metrics.py

Given evaluation results in the form:
  results = {
    "ModelA": {
      "EquityIndex": {"D1": 0.2, "D2": 0.3, ...},
      "AvgTravelTime": {"D1": 15.2, ...},
      ...
      "CapacityToDemandRatio": {"D1": 0.8, "D2": 1.1, ...}  # for Lorenz/Gini
    },
    "ModelB": { ... },
    ...
  }
and a demand dict:
  demands = {"D1": 1200, "D2": 2500, ...}

This script:
  1. Aggregates per-district values into a summary (mean across districts) per model per metric.
  2. Normalizes each metric across models to [0,1], inverting those where lower raw is better.
  3. Plots a grouped bar chart: x-axis = metrics, bars for each model.
  4. Plots a heatmap of normalized scores: rows = models, columns = metrics.
  5. Computes demand-weighted Lorenz curves and Gini coefficients for capacity-to-demand ratios.
  6. Plots combined demand-weighted Lorenz curves across models.

Dependencies:
  pandas, numpy, matplotlib
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
from metrics.equity_index_metric import calculate_new_equity_index
from metrics.hdr_metric import calculate_hdr
from metrics.overserved_area_metric import compute_overserved_area_count
from metrics.hfdr_metric import calculate_hfdr
from metrics.accessibility_score_metric import accessibility_score

# Add the parent directory to sys.path so Python can find the sibling 'metrics' package
current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Now you can import functions from modules in the 'metrics' folder
# For example, if there is a file named 'metrics_utils.py' with a function 'compute_metric', do:



def compute_summary(results_dict, agg_func="mean"):
    """
    results_dict: dict of model -> dict of metric -> dict of district -> value
    agg_func: "mean" or "median"
    Returns: DataFrame summary_df with index=models, columns=metrics, values=aggregated metric.
    """
    models = list(results_dict.keys())
    metric_set = set()
    for m in models:
        metric_set.update(results_dict[m].keys())
    metrics = sorted(metric_set)

    summary = {}
    for model in models:
        summary[model] = {}
        for metric in metrics:
            per_district = results_dict[model].get(metric, {})
            if not per_district:
                summary[model][metric] = np.nan
            else:
                s = pd.Series(per_district)
                if agg_func == "mean":
                    summary[model][metric] = s.mean()
                elif agg_func == "median":
                    summary[model][metric] = s.median()
                else:
                    raise ValueError("Unsupported agg_func")
    summary_df = pd.DataFrame.from_dict(summary, orient="index")
    summary_df = summary_df.dropna(axis=1, how="all")
    return summary_df

def normalize_summary(summary_df, invert_metrics=None):
    """
    summary_df: DataFrame index=models, columns=metrics, values=aggregated scores.
    invert_metrics: list or set of metric names where lower raw value is better → invert after scaling.
    Returns: DataFrame norm_df of same shape, values in [0,1], where larger = better.
    """
    norm_df = summary_df.copy().astype(float)
    invert_metrics = set(invert_metrics) if invert_metrics is not None else set()
    for col in norm_df.columns:
        col_vals = norm_df[col].astype(float)
        if col_vals.isna().all():
            norm_df[col] = np.nan
            continue
        min_val = col_vals.min(skipna=True)
        max_val = col_vals.max(skipna=True)
        if max_val > min_val:
            scaled = (col_vals - min_val) / (max_val - min_val)
        else:
            # identical across models: neutral 0.5
            scaled = pd.Series(0.5, index=col_vals.index)
        if col in invert_metrics:
            scaled = 1 - scaled
        norm_df[col] = scaled
    return norm_df

def plot_grouped_bar(norm_df, title="Model comparison across metrics", figsize=(10,6), savepath=None, show=True):
    """
    norm_df: DataFrame index=models, columns=metrics, values in [0,1], larger = better.
    Plots a grouped bar chart: for each metric (x-axis), bars for each model.
    """
    models = list(norm_df.index)
    metrics = list(norm_df.columns)
    n_models = len(models)
    n_metrics = len(metrics)
    if n_metrics == 0 or n_models == 0:
        print("No data to plot grouped bar.")
        return

    x = np.arange(n_metrics)
    total_width = 0.8
    bar_width = total_width / n_models
    offsets = np.linspace(-total_width/2 + bar_width/2, total_width/2 - bar_width/2, n_models)

    fig, ax = plt.subplots(figsize=figsize)
    for i, model in enumerate(models):
        values = norm_df.loc[model].tolist()
        values = [0.5 if (isinstance(v, float) and np.isnan(v)) else v for v in values]
        ax.bar(x + offsets[i], values, width=bar_width, label=model)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=45, ha='right')
    ax.set_ylim(0, 1)
    ax.set_ylabel("Normalized Score (higher = better)")
    ax.set_title(title)
    ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1.0))
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath)
    if show:
        plt.show()
    plt.close()

def plot_heatmap(norm_df, title="Heatmap of normalized scores", figsize=(8,6), savepath=None, show=True):
    """
    norm_df: DataFrame index=models, columns=metrics, values in [0,1], larger = better.
    Plots a heatmap: rows=models, columns=metrics.
    """
    if norm_df.shape[0] == 0 or norm_df.shape[1] == 0:
        print("No data to plot heatmap.")
        return

    fig, ax = plt.subplots(figsize=figsize)
    data = norm_df.values.astype(float)
    im = ax.imshow(data, aspect='auto', cmap='viridis', vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(norm_df.columns)))
    ax.set_yticks(np.arange(len(norm_df.index)))
    ax.set_xticklabels(norm_df.columns, rotation=45, ha='right')
    ax.set_yticklabels(norm_df.index)
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Normalized Score")
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath)
    if show:
        plt.show()
    plt.close()

def compute_demand_weighted_lorenz(ratios, demands):
    """
    Compute the demand-weighted Lorenz curve for per-capita resource ratios.

    Parameters:
    - ratios: array-like of per-capita resource (capacity-to-demand ratio) per district
    - demands: array-like of demand/population per district
    Returns:
    - cum_pop_share: 1D array of cumulative population share [0..1]
    - cum_res_share: 1D array of cumulative resource share [0..1]
    """
    ratios = np.asarray(ratios, dtype=float)
    demands = np.asarray(demands, dtype=float)
    if ratios.shape != demands.shape:
        raise ValueError("ratios and demands must have the same shape")
    total_pop = demands.sum()
    resources = ratios * demands
    total_res = resources.sum()
    if total_pop <= 0 or total_res <= 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    sorted_idx = np.argsort(ratios)
    demands_sorted = demands[sorted_idx]
    resources_sorted = resources[sorted_idx]
    cum_pop = np.cumsum(demands_sorted)
    cum_res = np.cumsum(resources_sorted)
    cum_pop = np.concatenate(([0.0], cum_pop))
    cum_res = np.concatenate(([0.0], cum_res))
    cum_pop_share = cum_pop / total_pop
    cum_res_share = cum_res / total_res
    return cum_pop_share, cum_res_share

def compute_demand_weighted_gini(ratios, demands):
    """
    Compute the demand-weighted Gini coefficient for per-capita resource ratios.
    Gini = 1 - 2 * area under the demand-weighted Lorenz curve.
    Returns float in [0,1]: 0 = perfect equality, larger = more inequality.
    """
    cum_pop_share, cum_res_share = compute_demand_weighted_lorenz(ratios, demands)
    area = np.trapz(cum_res_share, cum_pop_share)
    gini = 1.0 - 2.0 * area
    return float(gini)

def plot_demand_weighted_lorenz(models_data, title="Demand-Weighted Lorenz Curves"):
    """
    Plot multiple demand-weighted Lorenz curves on one figure.
    models_data: dict model_name -> (ratios_array, demands_array)
    """
    plt.figure(figsize=(8,6))
    plt.plot([0,1],[0,1], linestyle='--', color='gray', label='Equality')
    for model_name, (ratios, demands) in models_data.items():
        cum_pop_share, cum_res_share = compute_demand_weighted_lorenz(ratios, demands)
        plt.step(cum_pop_share, cum_res_share, where='post', label=model_name)
    plt.xlabel("Cumulative share of population (demand)")
    plt.ylabel("Cumulative share of resource")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    plt.close()

def main():
    """model
    Example usage with dummy data. Replace `results` and `demands` with actual evaluation results.
    """
    rng = np.random.default_rng(42)
    # Example district list
    districts = [f"D{i}" for i in range(1, 11)]
    # Dummy demand per district
    demands = {d: rng.uniform(1000, 5000) for d in districts}
    # Dummy results for three models

    # Build the path to the experiments/results directory
    current_dir = os.path.dirname(__file__)
    experiments_results_dir = os.path.join(current_dir, "..", "experiments", "results")


    # from metrics.equity_index_metric import calculate_new_equity_index
    # from metrics.hdr_metric import calculate_hdr
    # from metrics.overserved_area_metric import compute_overserved_area_count
    # from metrics.hfdr_metric import calculate_hfdr
    # from metrics.accessibility_score_metric import accessibility_score

    # Define the mapping from model names to their Excel file names
    excel_files = {
        "main": "main.xlsx",
        "policy_maker_model": "policy_maker_model.xlsx",
        "deprivation_aware_model": "deprivation_aware_model.xlsx",
        "demand_based_model": "demand_based_model.xlsx",
        "status_quo_model": "status_quo_model.xlsx",
        "accessibility_model": "accessibility_based_model.xlsx",
    }

    # Read each Excel file into a results dictionary
    results = {}
    # In this example, we assume the Excel file has a column 'District' and several metric columns.
    for model, filename in excel_files.items():
        path = os.path.join(experiments_results_dir, filename)
        
        results[model] = {
            "EquityScore": calculate_new_equity_index(path, modelname=model),
            "HDR": calculate_hdr(path),
            "OverServedAreaCount": compute_overserved_area_count(path),
            "HFDR": calculate_hfdr(path),
        }
    

    accessibility_score_results = accessibility_score()

    for acc in accessibility_score_results:
        model_name = acc["model"]
        if model_name in results:
            results[model_name]["AccessibilityScore"] = {
                "mean_travel_time_mins": acc["mean_travel_time_mins"],
                "median_travel_time_mins": acc["median_travel_time_mins"],
                "p95_travel_time_mins": acc["p95_travel_time_mins"]
            }


    # 1) Aggregate per-district into summary (mean across districts)
    summary_df = compute_summary(results, agg_func="mean")
    print("Summary (mean) per model per metric:\n", summary_df)

    # 2) Decide which metrics to invert (lower raw is better)
    invert_metrics = {"EquityIndex", "AvgTravelTime", "HDR", "OverServedCount"}
    # If HSI needs special handling (e.g. deviation from ideal), precompute separately.

    # 3) Normalize so higher = better in [0,1]
    norm_df = normalize_summary(summary_df, invert_metrics=invert_metrics)
    print("Normalized scores:\n", norm_df)

    # 4) Plot grouped bar chart
    plot_grouped_bar(norm_df, title="Model comparison across metrics", savepath=None, show=True)

    # 5) Plot heatmap of normalized scores
    plot_heatmap(norm_df, title="Heatmap of normalized scores", savepath=None, show=True)

    # 6) Demand-weighted Lorenz & Gini for CapacityToDemandRatio
    # Build models_data: each model_name -> (ratios_array, demands_array)
    # Ensure districts ordering is consistent
    ratios_demands_data = {}
    # Create a list of demands in consistent order
    demands_array = np.array([demands[d] for d in districts], dtype=float)
    for model in models:
        # Extract per-district ratios in same order
        ratio_dict = results[model].get("CapacityToDemandRatio", {})
        ratios_array = np.array([ratio_dict.get(d, 0.0) for d in districts], dtype=float)
        ratios_demands_data[model] = (ratios_array, demands_array)
    # Compute and print demand-weighted Gini for each model
    for name, (ratios_array, demands_array) in ratios_demands_data.items():
        gini = compute_demand_weighted_gini(ratios_array, demands_array)
        print(f"Demand-weighted Gini for {name}: {gini:.4f}")
    # Plot combined Lorenz curves
    plot_demand_weighted_lorenz(ratios_demands_data, title="Demand-Weighted Lorenz Curves for Models")

if __name__ == "__main__":
    main()