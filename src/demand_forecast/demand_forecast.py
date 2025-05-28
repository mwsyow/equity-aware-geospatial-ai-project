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
    df = pd.read_csv(os.path.join(DATA_PATH, 'diseases_history.csv'), sep='\t', encoding='utf-16')
    df = df.reset_index()
    col = df.iloc[0]
    col.iloc[0] = DISTRICT_CODE
    col.iloc[1:] = col.iloc[1:].map(lambda x: int(x.split('/')[0]))
    df.columns = col
    df = df.drop(df.index[0]).reset_index(drop=True)
    df = df.drop(columns=[col for col in df.columns if isinstance(col, int) and col > 2021])
    return df

def df_saarland_diseases_history() -> pd.DataFrame:
    df = df_diseases_history()
    df = df[df[DISTRICT_CODE] in SAARLAND_DISTRICT_MAPPING]


def df_elderly_population() -> pd.DataFrame:
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
    dfep = df_elderly_population()
    elderly_population = dfep[dfep[REGION_CODE]==region_code]
    elderly_population = elderly_population.pivot(
        index=DISTRICT_CODE,
        columns=YEAR,
        values=VALUE
    )
    return elderly_population
    
def df_hospital_inpatients() -> pd.DataFrame:
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
    path = os.path.join(DATA_PATH, 'population_history.csv')
    df = pd.read_csv(path, sep=';')
    
    df = df[DF_FIELDS]
    # transform date from `year-month-day` to `year` format
    df[YEAR] = df[YEAR].map(lambda x : int(x.split('-')[0]))
    
    # only consider year before or equal than CUT_OFF_YEAR
    df = df[df[YEAR] <= CUT_OFF_YEAR].reset_index(drop=True)
    
    return df
    
def df_population_projection(variant: ProjectionVariant) -> pd.DataFrame:
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
                except:
                    continue
                
    return best_model, best_order

def forecast_ARIMA(model, period: int) -> tuple[pd.Series, pd.DataFrame]:
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
