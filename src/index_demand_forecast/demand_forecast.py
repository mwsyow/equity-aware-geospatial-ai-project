import os 
import numpy as np
import pandas as pd

from statsmodels.tsa.arima.model import ARIMA
from enum import StrEnum

DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')

CUT_OFF_YEAR = 2021
YEAR = 'time'
REGION_CODE = '1_variable_attribute_code'
VALUE = 'value'
ICD_VARIANT = '2_variable_attribute_label'
PROJECTION_VARIANT = '2_variable_attribute_code'
DISTRICT_CODE = 'district_code'

DF_FIELDS = [YEAR, REGION_CODE, VALUE]

class ProjectionVariant(StrEnum):
    """Population projection variants for demographic forecasting.
    
    Enumeration of different population projection scenarios used by the demographic model.
    """
    VAR01 = 'BEV-VARIANTE-01'
    VAR02 = 'BEV-VARIANTE-02'
    VAR03 = 'BEV-VARIANTE-03'
    VAR04 = 'BEV-VARIANTE-04'
    VAR05 = 'BEV-VARIANTE-05'
    
REGION_CODE_MAPPING = {
    11: 'Berlin',
    6: 'Hessen',
    10: 'Saarland',
    9: 'Bayern',
    5: 'Nordrhein-Westfalen',
    1: 'Schleswig-Holstein',
    16: 'Thüringen',
    13: 'Mecklenburg-Vorpommern',
    12: 'Brandenburg',
    15: 'Sachsen-Anhalt',
    8: 'Baden-Württemberg',
    2: 'Hamburg',
    3: 'Niedersachsen',
    4: 'Bremen',
    14: 'Sachsen',
    7: 'Rheinland-Pfalz'
}

SAARLAND_DISTRICT_MAPPING = {
    "Saarlouis County": 10044,
    "Sankt Wendel County": 10046,
    "Saarpfalz-Kreis County": 10045,
    "Merzig-Wadern County": 10042,
    "Neunkirchen County": 10043,
    "Stadtverband Saarbrücken County": 10041 
}

def df_diseases_history() -> pd.DataFrame:
    """Load and process historical disease data from CSV file.
    
    Returns
    -------
    pd.DataFrame
        Processed DataFrame containing historical disease data with district codes as rows
        and years as columns. Years after 2021 are dropped and numeric values are converted
        to proper data types.
    """
    df = pd.read_csv(os.path.join(DATA_PATH, 'diseases_history.csv'), sep='\t', encoding='utf-16')
    df = df.reset_index()
    col = df.iloc[0]
    col.iloc[0] = DISTRICT_CODE
    col.iloc[1:] = col.iloc[1:].map(lambda x: int(x.split('/')[0]))
    df.columns = col
    df = df.drop(df.index[0]).reset_index(drop=True)
    df = df.drop(columns=[col for col in df.columns if isinstance(col, int) and col > 2021])
    year_cols = [col for col in df.columns if isinstance(col, int)]
    df[year_cols] = df[year_cols].apply(pd.to_numeric, errors='coerce')
    
    return df

def df_saarland_diseases_history() -> pd.DataFrame:
    """Extract and process disease history data specifically for Saarland districts.
    
    Returns
    -------
    pd.DataFrame
        DataFrame containing disease history for Saarland districts only, with district
        codes mapped to standard identifiers and years as integer columns.
    """
    df = df_diseases_history()
    df = df[df[DISTRICT_CODE].isin(SAARLAND_DISTRICT_MAPPING)]
    df[DISTRICT_CODE] = df[DISTRICT_CODE].map(lambda x: SAARLAND_DISTRICT_MAPPING[x])
    df.index = df[DISTRICT_CODE]
    df=df.drop(columns=[DISTRICT_CODE])
    df.columns = df.columns.astype(int)
    return df

def df_elderly_population() -> pd.DataFrame:
    """Load and process elderly population data.
    
    Returns
    -------
    pd.DataFrame
        DataFrame containing elderly population statistics 
    """
    path = 'data/elderly_population.csv'
    df = pd.read_csv(path, sep=';')
    
    # in the dataset column `1_variable_attribute_code` is populated with district code not region code
    # unify the format with other dataset 
    df[DISTRICT_CODE] = df["1_variable_attribute_code"]
    df[REGION_CODE] = df["1_variable_attribute_code"].map(lambda x: int(x/1000))
    
    # transform date from `year-month-day` to `year` format
    df[YEAR] = df[YEAR].map(lambda x : int(x.split('-')[0]))
    
    # change elements on `value` column to integer
    df[VALUE] = pd.to_numeric(df[VALUE], errors='coerce')
    df = df[~df[VALUE].isna()]
    
    # only consider year before or equal than CUT_OFF_YEAR
    df = df[df[YEAR] <= CUT_OFF_YEAR].reset_index(drop=True)
    
    grouped_df = df.groupby([YEAR, REGION_CODE, DISTRICT_CODE], as_index=False)[VALUE].sum()
    return grouped_df

def df_elderly_population_per_region(region_code:int) -> pd.DataFrame:
    """Extract elderly population data for a specific region.
    
    Parameters
    ----------
    region_code : int
        Code identifying the region of interest

    Returns
    -------
    pd.DataFrame
        Pivoted DataFrame with district codes as index and years as columns,
        containing elderly population values.
    """
    dfep = df_elderly_population()
    elderly_population = dfep[dfep[REGION_CODE]==region_code]
    elderly_population = elderly_population.pivot(
        index=DISTRICT_CODE,
        columns=YEAR,
        values=VALUE
    )
    return elderly_population
    
def df_hospital_inpatients() -> pd.DataFrame:
    """Load and process hospital inpatient data.
    
    Returns
    -------
    pd.DataFrame
        Processed DataFrame containing hospital inpatient statistics up to CUT_OFF_YEAR,
        filtered for total values across all ICD variants.
    """
    path = os.path.join(DATA_PATH, 'hospital_inpatients.csv')
    df = pd.read_csv(path, sep=';')
    
    df = df[df[ICD_VARIANT]=='Total'].reset_index(drop=True)
    df = df[DF_FIELDS]
    
    # change elements on `value` column to integer
    df[VALUE] = pd.to_numeric(df[VALUE], errors='coerce')
    df = df[~df[VALUE].isna()]
    
    # only consider year before or equal than CUT_OFF_YEAR
    df = df[df[YEAR] <= CUT_OFF_YEAR].reset_index(drop=True)
    
    return df

def df_population_history() -> pd.DataFrame:
    """Load and process historical population data.
    
    Returns
    -------
    pd.DataFrame
        DataFrame containing historical population statistics up to CUT_OFF_YEAR
        with standardized year format.
    """
    path = os.path.join(DATA_PATH, 'population_history.csv')
    df = pd.read_csv(path, sep=';')
    
    df = df[DF_FIELDS]
    # transform date from `year-month-day` to `year` format
    df[YEAR] = df[YEAR].map(lambda x : int(x.split('-')[0]))
    
    # only consider year before or equal than CUT_OFF_YEAR
    df = df[df[YEAR] <= CUT_OFF_YEAR].reset_index(drop=True)
    
    return df
    
def df_population_projection(variant: ProjectionVariant) -> pd.DataFrame:
    """Load and process population projection data for a specific variant.
    
    Parameters
    ----------
    variant : ProjectionVariant
        The population projection variant to use

    Returns
    -------
    pd.DataFrame
        DataFrame containing population projections with values scaled from 
        per 1000 people to absolute numbers.
    """
    path = os.path.join(DATA_PATH, 'population_projection.csv')
    df = pd.read_csv(path, sep=';')
    
    df = df[df[PROJECTION_VARIANT]==variant].reset_index(drop=True)
    df = df[DF_FIELDS]
    
    # transform date from `year-month-day` to `year` format
    df[YEAR] = df[YEAR].map(lambda x : int(x.split('-')[0]))
    
    # transform value from per 1000 people
    df[VALUE] = df[VALUE].map(lambda x: x*1000)
    
    return df

def df_per_capita_demand() -> pd.DataFrame:
    """Calculate per capita demand based on hospital inpatients and population history.
    
    Returns
    -------
    pd.DataFrame
        DataFrame containing per capita demand values with regions as rows and years
        as columns.
    
    Raises
    ------
    AssertionError
        If years don't match between datasets or if number of datapoints is inconsistent.
    """
    df_inp = df_hospital_inpatients()
    df_pophist = df_population_history()
    df_inp = df_inp.sort_values(YEAR)
    df_pophist = df_pophist.sort_values(YEAR)

    assert np.array_equal(df_inp[YEAR].unique(), df_pophist[YEAR].unique()), (
        "years in hospital inpatients dataset doesn't match with years in population history dataset"
    )
    
    years = df_inp[YEAR].unique()
    
    per_capita_demand = {}
    for region_code in REGION_CODE_MAPPING.keys():
        
        df_temp_inp = df_inp[df_inp[REGION_CODE]==region_code].reset_index(drop=True)
        df_temp_pophist = df_pophist[df_pophist[REGION_CODE]==region_code].reset_index(drop=True)

        assert len(df_temp_inp) == len(df_temp_pophist), (
            f"Number of datapoints in hospital inpatients dataset {len(df_temp_inp)} doesn't match"
            f"Number of datapoints in population history dataset {len(df_temp_pophist)}"
            f"for region code {region_code}"
        )
        # per-capita demand = hospital inpatients / population history
        demand = df_temp_inp[VALUE]/df_temp_pophist[VALUE]
        per_capita_demand[region_code] = demand
        
    df = pd.DataFrame(per_capita_demand).transpose()
    df.columns = years

    return df

def grid_search_ARIMA(series: pd.Series, p_values: list, d_values: list, q_values: list):
    """Perform grid search to find optimal ARIMA parameters.
    
    Parameters
    ----------
    series : pd.Series
        Time series data to model
    p_values : list
        List of p values to try 
    d_values : list
        List of d values to try 
    q_values : list
        List of q values to try 

    Returns
    -------
    tuple
        Best fitting ARIMA model and its order (p,d,q)
    """
    # Grid search over (p,d,q)
    best_aic = float('inf')
    best_order = None
    best_model = None

    for p in p_values:
        for d in d_values:
            for q in q_values:
                try:
                    model = ARIMA(series, order=(p, d, q))
                    model_fit = model.fit()
                    aic = model_fit.aic
                    print(f'ARIMA({p},{d},{q}) AIC={aic:.2f}')
                    if aic < best_aic:
                        best_aic = aic
                        best_order = (p, d, q)
                        best_model = model_fit
                        print(f'Current best order ({p},{d},{q})')
                except Exception as e:
                    print(str(e))
                    continue
                
    return best_model, best_order

def forecast_ARIMA(model, period: int) -> tuple[pd.Series, pd.DataFrame]:
    """Generate forecasts using fitted ARIMA model.
    
    Parameters
    ----------
    model : ARIMA model
        Fitted ARIMA model
    period : int
        Number of periods to forecast

    Returns
    -------
    tuple
        Predicted mean values and confidence intervals
    """
    pred = model.get_forecast(period)
    conf_int = pred.conf_int()
    pred_mean = pred.predicted_mean
    pred_mean.index += 2000
    conf_int.index += 2000
    return pred_mean, conf_int

def forecast_demand(
    forecast: pd.Series, 
    region_code: int, 
    proj_variant: ProjectionVariant, 
    conf_int: pd.DataFrame=None
) -> pd.Series | tuple[pd.Series, pd.DataFrame]:
    """Calculate demand forecast based on population projections.
    
    Parameters
    ----------
    forecast : pd.Series
        Forecasted values
    region_code : int
        Region code for the forecast
    proj_variant : ProjectionVariant
        Population projection variant to use
    conf_int : pd.DataFrame, optional
        Confidence intervals for the forecast

    Returns
    -------
    pd.Series or tuple
        Demand forecast, and optionally confidence intervals
    
    Raises
    ------
    AssertionError
        If region code is invalid
    """
    assert region_code in REGION_CODE_MAPPING, f"code {region_code} is invalid"
    df = df_population_projection(proj_variant)
    df = df[
        (df[REGION_CODE]==region_code)&
        (df[YEAR] <= forecast.index.max())&
        (df[YEAR] >= forecast.index.min())
    ]
    df = df[[YEAR, VALUE]].set_index(YEAR).sort_index()
    
    demand = df[VALUE] * forecast
    
    if conf_int is not None and not conf_int.empty:
        demand_conf_int = conf_int.copy()
        lower = conf_int.iloc[:, 0]
        upper = conf_int.iloc[:, 1]
        lower_demand = df[VALUE] * lower
        upper_demand = df[VALUE] * upper
        demand_conf_int.iloc[:, 0] = lower_demand
        demand_conf_int.iloc[:, 1] = upper_demand
        return demand, demand_conf_int
    
    return demand

def forecast_diseases_history(df: pd.DataFrame, period: int, 
                            p_values: list, d_values: list, 
                            q_values: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate disease history forecasts using ARIMA models.
    
    Parameters
    ----------
    df : pd.DataFrame
        Historical disease data
    period : int
        Number of periods to forecast
    p_values : list
        List of p values for ARIMA
    d_values : list
        List of d values for ARIMA
    q_values : list
        List of q values for ARIMA

    Returns
    -------
    tuple
        Forecasted values and confidence intervals for each district
    """
    combined_dfs = []
    combined_conf_ints = []
    for district_code in df.index:
        disease_history = df.loc[district_code]
        best_model, _ = grid_search_ARIMA(
            disease_history, 
            p_values=p_values, 
            d_values=d_values, 
            q_values=q_values
        )
        forecast_disease_history, conf_int = forecast_ARIMA(best_model, period)
        # turns year index into a column
        df_forecast_disease_history = forecast_disease_history.to_frame().rename(columns={'predicted_mean': VALUE})
        df_forecast_disease_history[DISTRICT_CODE] = district_code
        combined_dfs.append(df_forecast_disease_history)
        
        conf_int.columns = ['lower', 'upper']
        conf_int[DISTRICT_CODE] = district_code
        combined_conf_ints.append(conf_int)

    final_df = pd.concat(combined_dfs)
    final_conf_int = pd.concat(combined_conf_ints)
    return final_df, final_conf_int

def forecast_demand_per_district_in_saarland() -> dict:
    """Generate demand forecasts for districts in Saarland.
    
    Calculates demand forecasts using ARIMA models and population projections,
    incorporating confidence-weighted indices for each district.

    Returns
    -------
    dict
        Dictionary mapping district codes to normalized demand indices
    """
    region_code = 10
    period = 9
    
    dfpcd = df_per_capita_demand()
    
    per_capita_demand = dfpcd.loc[region_code]

    best_model, _ = grid_search_ARIMA(
        per_capita_demand, 
        p_values=[3], 
        d_values=[0], 
        q_values=[0]
    )
    
    forecast, conf_int = forecast_ARIMA(best_model, period)
    demand, _ = forecast_demand(forecast, region_code, ProjectionVariant.VAR01, conf_int)
    
    df = df_saarland_diseases_history()
    forecast_diseases, diseases_conf_int = forecast_diseases_history(df, period, [1], [1], [0])
    forecast_diseases = forecast_diseases.reset_index().rename(columns={'index': YEAR})
    forecast_diseases = forecast_diseases.pivot(
        index=DISTRICT_CODE,
        columns=YEAR,
        values=VALUE
    )
    diseases_conf_int = diseases_conf_int.reset_index().rename(columns={'index': YEAR})
    upper_conf_int = diseases_conf_int.pivot(
        index=DISTRICT_CODE,
        columns=YEAR,
        values='upper'
    )
    lower_conf_int = diseases_conf_int.pivot(
        index=DISTRICT_CODE,
        columns=YEAR,
        values='lower'
    )
    forecast_diseases_standardized = forecast_diseases.div(forecast_diseases.sum(axis=0), axis=1)

    demand_per_district = forecast_diseases_standardized.mul(demand, axis=1)
    # Step 1: Compute relative width of interval
    interval_width = upper_conf_int - lower_conf_int
    relative_width = interval_width / forecast_diseases

    # Step 2: Convert to confidence (invert width)
    confidence_df = 1 - relative_width
    # Optional: clip negative values (in case intervals are huge)
    confidence_df = confidence_df.clip(lower=0)
    # Step 3: Compute confidence-weighted index
    weighted_values = demand_per_district * confidence_df
    weighted_sum = weighted_values.sum(axis=1)
    total_confidence = confidence_df.sum(axis=1)
    confidence_weighted_index = weighted_sum / total_confidence

    # # Step 4: Normalize to 0–1
    normalized_index = confidence_weighted_index.map(lambda x: 1 - x/confidence_weighted_index.sum())
    result = {str(k): v for k, v in normalized_index.to_dict().items()}
    return result
