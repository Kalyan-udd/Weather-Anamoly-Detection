from retry_requests import retry
import requests_cache
import openmeteo_requests
import pandas as pd

class ImportData:
    def __init__(self, cache_path : str = "data/weather_cache"):
        self.Cached_session = requests_cache.CachedSession(cache_path, expire_after=-1)
        self.retry_session = retry(self.Cached_session, retries=5, backoff_factor=0.2)
        self.openmeteo = openmeteo_requests.Client(session=self.retry_session)
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
                "temperature": self.hourly.Variables(0).ValuesAsNumpy(),
                "humidity": self.hourly.Variables(1).ValuesAsNumpy(),
                "pressure": self.hourly.Variables(2).ValuesAsNumpy(),
            }
        ).set_index("date")

        return self.dataframe

class ImportCoordinates:
    def __init__(self, cache_path: str = "data/coordinates/coordinates"):
        self.session = requests_cache.CachedSession(cache_path, expire_after=-1)
        self.retry = retry(self.session, retries=5, backoff_factor=0.2)
        self.latitude = None
        self.longitude = None
        self.city = None
        self.district = None
        self.state = None
        self.time_zone = None

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
        self.district = result['admin2']
        self.state = result['admin1']
        self.time_zone = result['timezone']
        self.longitude = result['longitude']
        self.latitude = result['latitude']
        return (self.latitude, self.longitude)


class AnomalyAddition:
    def __init__(self):
        