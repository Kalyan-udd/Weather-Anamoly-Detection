from fastapi import APIRouter
from .schemas import WeatherReading
from src.weather_anamoly.model import predict

router = APIRouter(
    prefix="/api",
    tags=["Weather"]
)


@router.post("/readings")
def receive_reading(reading: WeatherReading):
    prediction = predict(reading)

    return {
        "message": "Reading processed successfully",
        "reading": reading,
        "anomaly": prediction
    }