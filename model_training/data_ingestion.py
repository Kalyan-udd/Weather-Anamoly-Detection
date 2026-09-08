from weather_anamoly.utils import ImportCoordinates, ImportData, AnomalyInjection, data_transformation

city = "Kolkata"
start = "2024-01-01"
end = "2025-01-01"
coordinate = ImportCoordinates()
latitude, longitude = coordinate.Fetch_coordinates(city=city)
data = ImportData()
df = data.extract(start, end, latitude, longitude)
anomaly_injection = AnomalyInjection()
df = anomaly_injection.inject_anomalies(df=df)
elevation = coordinate.elevation
df = data_transformation(df, location_involvement=True, latitude=latitude, longitude=longitude, elevation=elevation)
df.to_csv("model_training/data/data_trail.csv")