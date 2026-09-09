from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from src.weather_anamoly.live_fetch import resolve_city, fetch_and_prepare
from src.weather_anamoly.model import predict

app = FastAPI(title="Weather Anomaly Detection API")
templates = Jinja2Templates(directory="templates")


@app.get("/")
def dashboard(request: Request, city: str = "Delhi"):
    latitude, longitude, resolved_city, state = resolve_city(city)
    latest_row = fetch_and_prepare(latitude, longitude)
    anomaly = predict(latest_row)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "city": resolved_city,
        "state": state,
        "reading": latest_row.to_dict(orient="records")[0],
        "anomaly": anomaly
    })


@app.get("/health")
def health_check():
    return {"status": "healthy"}