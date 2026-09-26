import pandas as pd
import numpy as np
import requests_cache

cached = requests_cache.CachedSession("Production_cache", expire_after=3600)

rnd = np.random.default_rng(seed=42)

def build_multivariate_features(df:pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data['date'] = pd.to_datetime(df['date'])
    if 'date' in data.columns:
        data = data.set_index('date')
    variables = ['temperature', 'pressure', 'humidity']
    data['hour'] = data.index.hour
    data['month'] = data.index.month
    data['cos_hr'] = np.cos(2*np.pi*data['hour']/24)
    data['sin_hr'] = np.sin(2*np.pi*data['hour']/24)
    data['cos_mon'] = np.cos(2*np.pi*data['month']/12)
    data['sin_mon'] = np.sin(2*np.pi*data['month']/12)
    
    for col in variables:
        # 1. Lags (Past readings)
        data[f'{col}_lag1'] = data[col].shift(1)
        data[f'{col}_lag2'] = data[col].shift(2)
        data[f'{col}_lag24'] = data[col].shift(24)
        
        # 2. Gradients (Rate of change)
        data[f'{col}_grad_1h'] = data[f'{col}_lag1'] - data[f'{col}_lag2']
        data[f'{col}_grad_6h'] = data[f'{col}_lag1'] - data[col].shift(7)
        data[f'{col}_mean'] = data.groupby(['month','hour'])[col].transform('mean')
        data[f'{col}_std'] = data.groupby(['month', 'hour'])[col].transform('std')
    return data.dropna()


def spike_injection(x_test: pd.DataFrame, n_events: int) -> pd.DataFrame:
    df = x_test.copy()
    columns = ["temperature", "pressure", "humidity"]

    for col in columns:
         if f"{col}_label" not in df.columns:
            df[f'{col}_label'] = 0

    for _ in range(n_events):
        row_pos = int(rnd.integers(0, len(df)))
        row_index = df.index[row_pos]
        column = rnd.choice(columns)
        if df.at[row_index, f"{column}_label"] == 0:
            std_col = f"{column}_std"
            std_value = float(df.at[row_index, std_col])
            direction = rnd.choice([-1, 1])
            magnitude = float(rnd.uniform(3, 5) * std_value)
            current_value = float(df.at[row_index, column])
            if column == 'humidity':
                df.at[row_index, column] = int(current_value + (magnitude * direction))
                df.at[row_index, f"{column}_label"] = 1
            else:
                df.at[row_index, column] = current_value + (magnitude * direction)
                df.at[row_index, f"{column}_label"] = 1

    return df

def latest_spike_injection(df_: pd.DataFrame):
    df = df_.copy()
    columns = ['temperature', 'pressure', 'humidity']
    column = rnd.choice(columns)
    direction = rnd.choice([1, -1])
    std = df.loc[df.index[-1],f"{column}_std"]
    magnitude = np.float64(rnd.uniform(3, 5)*std)
    if column == 'humidity':
        df.loc[df.index[-1],f"{column}"] = int(df.loc[df.index[-1],f"{column}"]+ magnitude*direction)
    else:
        df.loc[df.index[-1],f"{column}"] = df.loc[df.index[-1],f"{column}"]+ magnitude*direction
    df.loc[df.index[-1], f"{column}_label"] = 1
    return df

def forecast_until_current_hour(latitude: float,longitude: float,start_hour,end_hour) -> pd.DataFrame:
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_hour": start_hour,
        "end_hour": end_hour,
        "hourly": 'temperature_2m,surface_pressure,relative_humidity_2m',
        "timezone": "Asia/Kolkata",
    }
    data = cached.get(url=url, params=params)
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