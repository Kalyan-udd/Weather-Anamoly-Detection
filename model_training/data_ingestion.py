from weather_anamoly.utils import ImportCoordinates, ImportData, AnomalyInjection, data_transformation

def training_data_extraction(city:str, start:str, end:str):
    coordinate = ImportCoordinates()
    result = coordinate.Fetch_coordinates(city=city)
    if result is None:
        raise ValueError(f"Could not fetch coordinates for city: {city}")
    latitude, longitude = result
    data = ImportData()
    df = data.extract(start, end, latitude, longitude)
    anomaly_injection = AnomalyInjection()
    df = anomaly_injection.inject_anomalies(df=df)
    elevation = coordinate.elevation
    df = data_transformation(df, location_involvement=True, latitude=latitude, longitude=longitude, elevation=elevation)
    df.to_csv(f"model_training/data/{city}_training_data.csv")

def testing_data_extraction(city:str, start:str, end:str):
    coordinate = ImportCoordinates()
    result = coordinate.Fetch_coordinates(city=city)
    if result is None:
        raise ValueError(f"Could not fetch coordinates for city: {city}")
    latitude, longitude = result
    data = ImportData()
    df = data.extract(start, end, latitude, longitude)
    anomaly_injection = AnomalyInjection()
    df = anomaly_injection.inject_anomalies(df=df)
    elevation = coordinate.elevation
    df = data_transformation(df, location_involvement=True, latitude=latitude, longitude=longitude, elevation=elevation)
    df.to_csv(f"model_training/data/{city}_testing_data.csv")