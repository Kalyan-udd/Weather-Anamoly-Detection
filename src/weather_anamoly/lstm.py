import numpy as np
import pandas as pd
import requests


def inverse_transform_3D(windows: np.ndarray, window_size=24):
    n_windows, _, n_features = windows.shape
    total_len = n_windows + window_size - 1
    sums = np.zeros((total_len, n_features))
    counts = np.zeros((total_len, 1))

    for i in range(n_windows):
        sums[i:i+window_size] += windows[i]
        counts[i:i+window_size] += 1

    return sums / counts

def transform_3D(df: pd.DataFrame, window_size = 24):
    n = len(df)
    data_matrix = df.values
    window = []
    for i in range(n- window_size + 1):
        window.append(data_matrix[i:i+window_size])
    return np.array(window)

def lstm_data_encoding(df:pd.DataFrame):
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    df['month'] = df.index.month
    df['hour'] = df.index.hour
    df['mean_temp'] = df.groupby(['month', 'hour'])['temperature'].transform('mean')
    df['mean_press'] = df.groupby(['month', 'hour'])['pressure'].transform("mean")
    df['mean_humidity'] = df.groupby(['month', 'hour'])['humidity'].transform("mean")
    df['std_temp'] = df.groupby(['month', 'hour'])['temperature'].transform('std')
    df['std_press'] = df.groupby(['month', 'hour'])['pressure'].transform("std")
    df['std_humidity'] = df.groupby(['month', 'hour'])['humidity'].transform('std')
    df['z_temp'] = (df['temperature'] - df['mean_temp'])/df['std_temp']
    df['z_press'] = (df['pressure'] - df['mean_press'])/df['std_press']
    df['z_humidity'] = (df['humidity']- df['mean_humidity'])/df['std_humidity']
    df = df.reset_index()
    return df

def mean_std_calculation(df:pd.DataFrame):
    df_ = df.copy()
    if 'date' in df_.columns:
        df_['date'] = pd.to_datetime(df_['date'])
        df_.set_index("date", inplace=True)
    df_['month'] = df_.index.month
    df_['hour'] = df_.index.hour
    features = ['temperature', 'pressure', 'humidity']
    tbl = (df_.groupby(['month', 'hour'])[features].agg(['mean', 'std']))
    tbl.columns = [f"{col}_{stat}" for col, stat in tbl.columns]
    tbl.reset_index(inplace=True)
    return tbl
    
def forecat_raw(start_date: str, end_date: str, latitude: float, longitude: float)-> pd.DataFrame:
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": 'temperature_2m,surface_pressure,relative_humidity_2m',
        "timezone": "Asia/Kolkata",
    }
    data = requests.get(url=url, params=params)
    response = data.json()
    hourly = response.get("hourly")
    if not hourly or "time" not in hourly:
        error_msg = response.get(
            "reason", "No hourly weather data returned for the selected window.")
        raise ValueError(f"Open-Meteo extraction error: {error_msg}")
    df = pd.DataFrame(hourly)
    df = df.rename(
        columns={
            "time":"date",
            "temperature_2m": "temperature",
            "relative_humidity_2m": "humidity",
            "surface_pressure": "pressure",                
        }
    )
    ordered_columns = ["date","temperature", "humidity", "pressure"]
    df = df[ordered_columns]
    df['date'] = pd.to_datetime(df['date'])
    df = df.copy()
    return df

def table_combining(df:pd.DataFrame, table:pd.DataFrame)->pd.DataFrame:
    df_ = df.copy()
    if 'date' in df_.columns:
        df_ = df_.set_index('date')
    df_['month'] = df_.index.month
    df_['hour'] = df_.index.hour
    df_out = df_.merge(table, on=['month', 'hour'], how='left')
    if 'date' not in df.columns:
        df.reset_index()
    df_out['date'] = df['date']
    df_out = df_out[['date', 'temperature', 'humidity', 'pressure','month', 'hour', 'temperature_mean', 'temperature_std', 'pressure_mean',
       'pressure_std', 'humidity_mean', 'humidity_std']]
    return df_out

def zscore_calculation(df:pd.DataFrame) -> pd.DataFrame:
    df_out = df.copy()
    df_out['z_temp'] = (df_out['temperature'] - df_out['temperature_mean'])/df_out['temperature_std']
    df_out['z_humidity'] = (df_out['humidity']-df_out['humidity_mean'])/df_out['humidity_std']
    df_out['z_press'] = (df_out['pressure'] - df_out['pressure_mean'])/df_out['pressure_std']
    return df_out