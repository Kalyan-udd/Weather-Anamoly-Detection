from fastapi import APIRouter, HTTPException
from datetime import datetime, timedelta
from tensorflow import keras
import numpy as np

from src.weather_anamoly.utils import ImportData, ImportCoordinates, data_transformation

router = APIRouter(
    prefix="/api",
    tags=["Weather"]
)

# Load model once when the app starts
model = keras.models.load_model("model_training/model.keras")

CLASS_LABELS = {
    0: "genuine",
    1: "spike",
    2: "frozen",
    3: "fault",
    4: "comm_error",
}  # TODO: confirm order matches your notebook's label_map output

FEATURE_COLUMNS = [
    "temperature", "humidity", "pressure", "hour", "month",
    "cos_hour", "sin_hour", "cos_month", "sin_month",
    "temp_gradient", "humid_gradient", "press_gradient",
]


@router.get("/predict")
def predict_for_city(city: str = "Delhi"):
    # 1. Turn city name into coordinates
    coords = ImportCoordinates()
    latitude, longitude = coords.Fetch_coordinates(city)

    # 2. Fetch recent weather data (need history for gradient features)
    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=4)

    importer = ImportData()
    df = importer.extract(
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
        latitude=latitude,
        longitude=longitude
    )

    # 3. Add derived features (matches training)
    df = data_transformation(df, latitude=latitude, longitude=longitude, location_involvement=False)
    df['hour'] = df.index.hour
    df['month'] = df.index.month
    df = df.dropna()

    if len(df) == 0:
        raise HTTPException(500, "Not enough data to compute features")

    latest_row = df.iloc[[-1]]

    # 4. Run the model
    X = latest_row[FEATURE_COLUMNS].to_numpy()
    probabilities = model.predict(X)
    predicted_class = int(np.argmax(probabilities, axis=1)[0])
    confidence = float(np.max(probabilities))

    # 5. Return result
    return {
        "city": coords.city,
        "state": coords.state,
        "reading": latest_row[["temperature", "humidity", "pressure"]].to_dict(orient="records")[0],
        "anomaly_type": CLASS_LABELS.get(predicted_class, "unknown"),
        "confidence": confidence
    }