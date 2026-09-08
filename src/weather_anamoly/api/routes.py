from fastapi import APIRouter

from .schemas import WeatherReading

from src.weather_anamoly.services.anomaly_service import AnomalyService
from src.weather_anamoly.services.communication_service import CommunicationService

from src.weather_anamoly.database.database import (
    save_reading,
    save_alert
)


router = APIRouter(
    prefix="/api",
    tags=["Weather"]
)


anomaly_service = AnomalyService()
communication_service = CommunicationService()


@router.post("/readings")
def receive_reading(reading: WeatherReading):

    # --------------------------------
    # 1. Save reading
    # --------------------------------

    reading_id = save_reading(reading)


    # --------------------------------
    # 2. Run anomaly detection
    # --------------------------------

    prediction = anomaly_service.predict(reading)


    # --------------------------------
    # 3. Check communication
    # --------------------------------

    communication_result = communication_service.check(reading)


    # --------------------------------
    # 4. Save communication alert
    # --------------------------------

    if communication_result["is_error"]:

        save_alert(
            station_id=reading.station_id,
            timestamp=reading.timestamp.isoformat(),
            alert_type=communication_result["fault_type"],
            severity=communication_result["severity"],
            confidence=communication_result["confidence"],
            reason=communication_result["reason"]
        )


    # --------------------------------
    # 5. Return result
    # --------------------------------

    return {
        "message": "Reading processed successfully",

        "reading_id": reading_id,

        "reading": reading,

        "anomaly": prediction,

        "communication": communication_result
    }


@router.get("/alerts")
def get_all_alerts():

    from src.weather_anamoly.database.database import get_alerts

    return {
        "alerts": get_alerts()
    }