from datetime import datetime
from pydantic import BaseModel, Field

class WeatherReading(BaseModel):
    station_id: str
    timestamp: datetime
    temperature: float 
    pressure: float 
    humidity: float