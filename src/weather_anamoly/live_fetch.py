from datetime import datetime, timedelta
from src.weather_anamoly.utils import ImportData, ImportCoordinates, data_transformation
def resolve_city(city: str):
    coords = ImportCoordinates()
    latitude, longitude = coords.Fetch_coordinates(city)
    return latitude, longitude, coords.city, coords.state

def fetch_and_prepare(latitude: float, longitude: float, hours: int = 72):
    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=hours // 24 + 1)

    importer = ImportData()
    df = importer.extract(
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
        latitude=latitude,
        longitude=longitude
    )

    df = data_transformation(df, latitude=latitude, longitude=longitude, location_involvement=False)

    df['hour'] = df.index.hour
    df['month'] = df.index.month

    return df.dropna().iloc[[-1]]