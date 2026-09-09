from data_ingestion import training_data_extraction
from weather_anamoly.logger import logger

cities = ["Bengaluru"]
start = "2014-01-01"
end = "2024-01-01"

for city in cities:
    training_data_extraction(city=city, start=start, end=end)
    logger.info(f"Imported training data from {city} between {start} and {end}")

