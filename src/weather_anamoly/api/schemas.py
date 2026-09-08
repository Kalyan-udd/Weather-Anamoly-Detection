from datetime import datetime

from pydantic import BaseModel, Field


class WeatherReading(BaseModel):
    station_id: str

    timestamp: datetime

    temperature: float | None = Field(
        default=None,
        ge=-90,
        le=70
    )

    pressure: float | None = Field(
        default=None,
        ge=850,
        le=1100
    )

    humidity: float | None = Field(
        default=None,
        ge=0,
        le=100
    )