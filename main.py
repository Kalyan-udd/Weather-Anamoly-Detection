from fastapi import FastAPI

from src.weather_anamoly.api.routes import router
from src.weather_anamoly.database.database import initialize_database

app = FastAPI(
    title="Weather Anomaly Detection API",
    description="API for Automatic Weather Station monitoring",
    version="1.0.0"
)

initialize_database()
app.include_router(router)


@app.get("/")
def root():

    return {
        "message": "Weather Anomaly Detection API is running"
    }


@app.get("/health")
def health_check():

    return {
        "status": "healthy"
    }