from data_ingestion import testing_data_extraction
from weather_anamoly.logger import logger

cities = ["Delhi", "Chennai", "Leh", "Jaisalmer", "Kolkata", "Shimla", "Guwahati", "Bengaluru", "Visakhapatnam"]
start = "2024-01-01"
end = "2025-01-01"

for city in cities:
    testing_data_extraction(city=city, start=start, end=end)
    logger.info(f"importing test data from {city} between {start} and {end}")