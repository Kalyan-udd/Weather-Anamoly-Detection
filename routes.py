from fastapi import APIRouter
from fastapi.templating import Jinja2Templates
from weather_anamoly.api.schemas import Xgboost
from fastapi import Request
from xgboost import XGBRegressor
from weather_anamoly.XGBOOST import forecast_until_current_hour
from weather_anamoly.XGBOOST import build_multivariate_features, latest_spike_injection
import pandas as pd
from datetime import datetime, timedelta

xgb_temp = XGBRegressor()
xgb_press = XGBRegressor()
xgb_humidity = XGBRegressor()
xgb_temp.load_model("artifacts/TempModelXGB.json")
xgb_press.load_model("artifacts/PressModelXGB.json")
xgb_humidity.load_model("artifacts/HumModelXGB.json")


router = APIRouter()


templates = Jinja2Templates(directory='templates')

@router.get("/xgboost")
async def xgboost(request: Request):
    return templates.TemplateResponse(request=request, name="xgboost.html")

@router.post("/api/xgboost")
async def xgboost_predict(req: Xgboost):
    anomaly = req.inject_anomaly
    start = datetime.now().replace(minute=0,second=0,microsecond=0)-timedelta(hours=168*2)
    start = start.isoformat()
    end = datetime.now().replace(minute=0, second=0, microsecond=0)+timedelta(hours=1)
    end = end.isoformat()
    df = forecast_until_current_hour(start_hour=start, end_hour=end, latitude=28.6448, longitude=77.2167)
    df_test = build_multivariate_features(df=df)
    df_test.reset_index(inplace=True)
    if anomaly == True:
        df_test = latest_spike_injection(df_test)
    x_temp = df_test[['hour', 'month','temperature_lag1', 'temperature_lag2',
       'temperature_lag24', 'temperature_grad_1h', 'temperature_grad_6h',
       'temperature_mean', 'temperature_std']]
    x_press = df_test[['hour', 'month','pressure_lag1', 'pressure_lag2',
        'pressure_lag24', 'pressure_grad_1h', 'pressure_grad_6h',
        'pressure_mean', 'pressure_std']]
    x_humidity = df_test[['hour', 'month','humidity_lag1', 'humidity_lag2',
        'humidity_lag24', 'humidity_grad_1h', 'humidity_grad_6h',
        'humidity_mean', 'humidity_std']]
    y_temp_pred = xgb_temp.predict(x_temp)
    y_press_pred = xgb_press.predict(x_press)
    y_humidity_pred = xgb_humidity.predict(x_humidity)
    df_pred = pd.DataFrame()
    df_pred['date'] = df_test['date']
    df_pred['temperature'] = y_temp_pred
    df_pred['pressure'] = y_press_pred
    df_pred['humidity'] = y_humidity_pred
    history = []
    for i in range(len(df_pred.tail(71))):
        temp_pred = round(float(df_pred.iloc[-i-2]['temperature']),1)
        press_pred = round(float(df_pred.iloc[-i-2]['pressure']), 2)
        hum_pred = int(df_pred.iloc[-i-2]['humidity'])
        temp_act = round(float(df_test.iloc[-i-2]['temperature']),1)
        press_act = round(float(df_test.iloc[-i-2]['pressure']),1)
        hum_act = int(df_test.iloc[-i-2]['humidity'])
        time_stamp = str(df_test.iloc[-i-2]['date'])
        temp_std = round(float(df_test.iloc[-i-2]['temperature_std']),1)
        press_std = round(float(df_test.iloc[-i-2]['pressure_std']),1)
        hum_std = int(df_test.iloc[-i-2]['humidity_std'])
        history.append({
            "timestamp": time_stamp,
            "actual": { "temperature": round(float(temp_act), 1), "pressure": round(float(press_act),1), "humidity": int(hum_act) },
            "predicted": { "temperature": round(float(temp_pred),1), "pressure": round(float(press_pred),1), "humidity": int(hum_pred) },
            "std": { "temperature": round(float(temp_std),1), "pressure": round(float(press_std),1), "humidity": int(hum_std) }
        })
    temp_pred = df_pred.loc[df_pred.index[-1], 'temperature']
    press_pred = df_pred.loc[df_pred.index[-1], 'pressure']
    hum_pred = df_pred.loc[df_pred.index[-1], 'humidity']
    temp_act = df_test.loc[df_test.index[-1], 'temperature']
    press_act = df_test.loc[df_test.index[-1], 'pressure']
    hum_act = df_test.loc[df_test.index[-1], 'humidity']
    time_stamp = df_test.loc[df_test.index[-1], 'date']
    temp_std = df_test.loc[df_test.index[-1], 'temperature_std']
    press_std = df_test.loc[df_test.index[-1], 'pressure_std']
    hum_std = df_test.loc[df_test.index[-1], 'humidity_std']
    return {
        "latest":{
            "timestamp": time_stamp,
            "actual": { "temperature": round(float(temp_act), 1), "pressure": round(float(press_act),1), "humidity": int(hum_act) },
            "predicted": { "temperature": round(float(temp_pred),1), "pressure": round(float(press_pred),1), "humidity": int(hum_pred) },
            "std": { "temperature": round(float(temp_std),1), "pressure": round(float(press_std),1), "humidity": int(hum_std) }
        },
        "history": history
    }