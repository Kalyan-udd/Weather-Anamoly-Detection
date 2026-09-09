from tensorflow import keras
import numpy as np

model = keras.models.load_model("model_training/model.keras")

# TODO: confirm exact order from encoder.classes_ in the notebook
CLASS_LABELS = {
    0: "genuine",
    1: "spike",
    2: "frozen",
    3: "fault",
    4: "comm_error",
}

# TODO: must match X_train.columns order EXACTLY
FEATURE_COLUMNS = [
    "temperature", "humidity", "pressure", "hour", "month",
    "cos_hour", "sin_hour", "cos_month", "sin_month",
    "temp_gradient", "humid_gradient", "press_gradient",
]


def predict(latest_row):
    X = latest_row[FEATURE_COLUMNS].to_numpy()
    probabilities = model.predict(X)
    predicted_class = int(np.argmax(probabilities, axis=1)[0])
    confidence = float(np.max(probabilities))

    return {
        "anomaly_type": CLASS_LABELS.get(predicted_class, "unknown"),
        "confidence": confidence
    }