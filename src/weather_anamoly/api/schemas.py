from datetime import datetime
from pydantic import BaseModel

class WeatherReading(BaseModel):
    station_id: str
    timestamp: datetime
    temperature: float 
    pressure: float 
    humidity: float

class WeatherRequest(BaseModel):
    location: str

class ModelTest(BaseModel):
    city: str
    start_time: str
    end_time: str