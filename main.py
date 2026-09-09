from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from weather_anamoly.api.schemas import WeatherRequest, ModelTest
from weather_anamoly.utils import fetch_continuous_data

app = FastAPI(title="Weather Anomaly Detection API")
templates = Jinja2Templates(directory="templates")

@app.get("/")
def index(request: Request):
    return templates.TemplateResponse(request=request, name='index.html')

@app.get("/model_test")
async def model_test_page(request: Request):
    return templates.TemplateResponse(request=request, name="model_test.html")

@app.post("/api/weather")
async def dashboard(req: WeatherRequest):
    meta_data = fetch_continuous_data(location=req.location)
    return meta_data

#app.post("/api/historical-weather")
#async def model_test(req: ModelTest):
