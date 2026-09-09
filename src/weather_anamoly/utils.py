from retry_requests import retry
import requests_cache
import openmeteo_requests
import pandas as pd
import numpy as np
from typing import Optional
from pathlib import Path
from weather_anamoly.logger import logger
import requests
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

ROOT = Path(__file__).resolve().parent.parent.parent
data_path = f"{ROOT}/model_training"


class ImportData:
    def __init__(self, cache_path : str = f"{data_path}/data/weather_cache"):
        self.Cached_session = requests_cache.CachedSession(cache_path, expire_after=-1)
        self.retry_session = retry(self.Cached_session, retries=5, backoff_factor=0.2)
        self.openmeteo = openmeteo_requests.Client(session = self.retry_session)
        self.dataframe = None
        self.hourly = None

    def extract(self, start_date: str, end_date: str, latitude: float, longitude: float) -> pd.DataFrame:
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ['temperature_2m', 'relative_humidity_2m', 'surface_pressure'],
            "timezone": "Asia/Kolkata"
        }
        try:
            response = self.openmeteo.weather_api(url=url, params=params)[0]
            hourly = response.Hourly()
            self.hourly = hourly
            self.dataframe = pd.DataFrame(
                {
                    "date": pd.date_range(
                        start=pd.to_datetime(self.hourly.Time(), unit="s", utc=True),
                        end=pd.to_datetime(self.hourly.TimeEnd(), unit="s", utc=True),
                        freq=pd.Timedelta(seconds=self.hourly.Interval()),
                        inclusive="left"
                    ),
                    "temperature": self.hourly.Variables(0).ValuesAsNumpy().astype('float64'),
                    "humidity": self.hourly.Variables(1).ValuesAsNumpy().astype('float64'),
                    "pressure": self.hourly.Variables(2).ValuesAsNumpy().astype('float64'),
                },
            )

            return self.dataframe
        except Exception as e:
            logger.info(f"failed fetching data from the api {e}")

class ImportCoordinates:
    def __init__(self, cache_path: str = f"{data_path}/data/coordinates/coordinates"):
        self.session = requests_cache.CachedSession(cache_path, expire_after=-1)
        self.retry = retry(self.session, retries=5, backoff_factor=0.2)
        self.latitude = None
        self.longitude = None
        self.time_zone = None
        self.elevation = None
        self.city = None

    def Fetch_coordinates(self, city:str ) -> tuple[float, float]:
        url= "https://geocoding-api.open-meteo.com/v1/search"
        params = {
            'name': f"{city}, India",
            'count': 1,
        }
     
        response = self.retry.get(url=url, params=params)
        data = response.json()
        result = data["results"][0]
        self.city = result['name']
        self.time_zone = result['timezone']
        self.longitude = result['longitude']
        self.latitude = result['latitude']
        self.elevation = result['elevation']
        return (self.latitude, self.longitude)

class AnomalyInjection:
    def __init__(self,columns = ('temperature', 'pressure', 'humidity'), seed = 42):
        self.rnd = np.random.default_rng(seed)
        self.columns = columns

    def inject_spike(self,n_events: int, df: pd.DataFrame, clean_std:dict) -> pd.DataFrame:
        n = len(df)
        if clean_std is None:
            clean_std = df[list(self.columns)].std().to_dict()
        for i in range(n_events):
            for tries in range(10):
                row_pos = self.rnd.integers(0,n)
                index_label = df.index[row_pos]
                if df.loc[index_label, "label"] == "genuine":
                    index = self.rnd.integers(0,n)
                    col = self.rnd.choice(self.columns)
                    std = clean_std[col]
                    direction = self.rnd.choice([-1, 1])
                    magnitude = self.rnd.uniform(3,12)*std
                    df.loc[index, col] =  df.loc[index, col] + direction*magnitude
                    df.loc[index, "label"] = "spike"
                    break 
        return df

    def inject_forzen(self,df: pd.DataFrame ,n_events: int) -> pd.DataFrame:
        n = len(df)
        for i in range(n_events):
            for attempt in range(10):
                duration = self.rnd.integers(3,13)
                start = self.rnd.integers(0, n - duration)
                target_indices = df.index[start: start+duration-1]
                if (df.loc[target_indices, "label"] == "genuine").all():
                    col = self.rnd.choice(self.columns)
                    stuck_value = df.loc[start, col]
                    df.loc[target_indices, col] = stuck_value
                    df.loc[target_indices, "label"] = "frozen"
                    break
        return df

    def inject_comm_error(self,df: pd.DataFrame ,n_events: int) -> pd.DataFrame:
            n = len(df)
            for i in range(n_events):
                for attempt in range(10):
                    duration = self.rnd.integers(1,6)
                    start = self.rnd.integers(0, n - duration)
                    target_indices = df.index[start:start+duration-1]
                    if (df.loc[target_indices, "label"] == "genuine").all():
                        col = self.rnd.choice(self.columns)
                        df.loc[target_indices, col] = np.nan
                        df.loc[target_indices, "label"] = "comm_error"
                        break
            return df

    def inject_drift_fault(self, df: pd.DataFrame, n_values:int, clean_std: dict) -> pd.DataFrame:
        n = len(df)
        if clean_std is None:
            clean_std = df[list(self.columns)].std().to_dict()
        for i in range(n_values):
            for attempt in range(10):
                duration = self.rnd.integers(24*14, 24*56)
                start = self.rnd.integers(0, n - duration)
                target_indices = df.index[start:start+duration-1]
                if (df.loc[target_indices, "label"] == "genuine").all():
                    col = self.rnd.choice(self.columns)
                    std = clean_std[col]
                    per_step_bias = self.rnd.uniform(0.02, 0.08)*std
                    direction = self.rnd.choice([-1, 1])
                    ramp = np.arange(1, duration)*per_step_bias*direction
                    df.loc[target_indices, col] = df.loc[target_indices, col].values + ramp
                    df.loc[target_indices, "label"] = "fault"
                    break
        return df

    def inject_anomalies(self, df:pd.DataFrame, spike_rate: int=1500, frozen_rate: int=2500, comm_rate:int =3000, drift_rate: int = 6000)-> pd.DataFrame:
        if "label" not in df.columns:
            df['label'] = "genuine"
        n_hours = len(df)

        clean_stds = df[list(self.columns)].std().to_dict()

        self.inject_spike(n_events=max(1, n_hours//spike_rate), df=df, clean_std=clean_stds)
        self.inject_forzen(n_events=max(1, n_hours//frozen_rate), df=df)
        #self.inject_comm_error(n_events=max(1, n_hours//comm_rate), df=df)
        self.inject_drift_fault(n_values=max(1, n_hours//drift_rate), df=df, clean_std=clean_stds)
        return df


def data_transformation(df:pd.DataFrame, elevation: Optional[float]=None, latitude: Optional[float]=None, longitude:Optional[float]=None, location_involvement: bool = False, )-> pd.DataFrame:
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    df['cos_hour'] = np.cos(2*np.pi*df.index.hour/24)
    df['sin_hour'] = np.sin(2*np.pi*df.index.hour/24)
    df['cos_month'] = np.cos(2*np.pi*df.index.month/12)
    df['sin_month'] = np.sin(2*np.pi*df.index.month/12)
    df['temp_gradient'] = df['temperature'].diff()
    df['humid_gradient'] = df['humidity'].diff()
    df['press_gradient'] = df['pressure'].diff()
    df["6hr_gradient_temp"] = df['temperature'].diff(periods=6)
    df["6hr_gradient_press"] = df['pressure'].diff(periods=6)
    df["6hr_gradient_humid"] = df['humidity'].diff(periods=6)
    if location_involvement:
        df['elevation'] = elevation
        df['latitude'] = latitude
        df['longitude'] = longitude
    df.dropna(axis=0, inplace=True)
    return df

def fetch_continuous_data(location:str):
    coordinates = ImportCoordinates()
    lat, long = coordinates.Fetch_coordinates(location)
    loc = coordinates.city
    weather_url = "https://api.open-meteo.com/v1/forecast"
    weather_params = {
        "latitude": lat,
        "longitude": long,
        "current": "temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,weather_code,surface_pressure,wind_speed_10m"
    }
    weather_data = requests.get(weather_url, weather_params, timeout=5)
    weather_json = weather_data.json()

    return{
        "city": loc,
        "latitude": lat,
        "longitude": long,
        "weather" : weather_json
    }

def Build_model(hp):
    model = tf.keras.Sequential()
    num_layers = hp.Int("num_layers", min_value=1, max_value=5, step=1)
    for i in range(num_layers):
        units = hp.Int(f"unit_layer_{i}", min_value=32, max_value=512, step=16)
        model.add(tf.keras.layers.Dense(units=units, activation=hp.Choice(f"activation_layer_{i}", ['relu', 'tanh', 'elu'])))

        if hp.Boolean(f'dropout_layer_{i}'):
            model.add(tf.keras.layers.Dropout(
                hp.Float(f"dropout_rate_{i}", 0.1, 0.5, step=0.1)
            ))
    model.add(tf.keras.layers.Dense(4, activation='softmax'))
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model.compile(
        optimizer=Adam(hp_learning_rate),
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics= ['accuracy']
    )
    return model