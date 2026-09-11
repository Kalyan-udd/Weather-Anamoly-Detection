from fastapi import FastAPI, Request, HTTPException
from fastapi.templating import Jinja2Templates
from weather_anamoly.api.schemas import WeatherRequest, ModelTest
from weather_anamoly.utils import fetch_continuous_data, extract, ImportCoordinates, data_transformation
import tensorflow as tf
from weather_anamoly.utils import AnomalyInjection 
from weather_anamoly.model import FEATURE_COLUMNS, LABEL_ENCODER, LABEL_DECODER
import numpy as np
from weather_anamoly.logger import logger
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


app = FastAPI(title="Weather Anomaly Detection API")
templates = Jinja2Templates(directory="templates")

model = tf.keras.models.load_model("artifacts/History_model.keras")
logger.info("Loading the model into the server.")

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

@app.post("/api/historical-weather")
async def model_test(req: ModelTest):
    city = req.location
    start = req.start_date
    end = req.end_date
    try:
        coor = ImportCoordinates()
        lat , long = coor.Fetch_coordinates(city)
        elevation = coor.elevation
        df = extract(latitude=lat, longitude=long, start_date=start, end_date=end)
        anomalies = AnomalyInjection()
        df = anomalies.inject_anomalies(df=df)
        df = data_transformation(df=df, elevation=elevation, location_involvement=True, latitude=lat, longitude=long)
    except Exception as e:
        logger.error(f"Extraction error: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Data extraction failed: {str(e)}")

    if df is None or df.empty:
        raise HTTPException(status_code=404, detail="No weather records returned for this date window.")
    x_test = df[FEATURE_COLUMNS]
    y_true = df['label'].map(LABEL_ENCODER).to_numpy()
    possibilities = model.predict(x_test)
    y_pred = np.argmax(possibilities, axis=1)
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_pred=y_pred, y_true=y_true, average='weighted', zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred).tolist()
    sample_inspection=[]
    for i in range(min(15, len(y_pred))):
        sample_inspection.append(
            {
                "sample_index": i+1,
                "actual": LABEL_DECODER.get(y_true[i], str(y_true[i])),
                "predicted": LABEL_DECODER.get(y_pred[i], str(y_pred[i])),
                "is_correct": bool(y_true[i]==y_pred[i]),
                "confidence": round(float(np.max(possibilities[i]))*100, 1)

            }
        )
    return {
        "city": city,
        "sample_count": len(y_true),
        "metrics":{
            "accuracy": round(float(acc) * 100, 2),
            "f1_score": round(float(f1) * 100, 2),
            "precision": round(float(precision) * 100, 2),
            "recall": round(float(recall) * 100, 2),
            "confusion_matrix": cm
        },
        "samples": sample_inspection
    }
    
