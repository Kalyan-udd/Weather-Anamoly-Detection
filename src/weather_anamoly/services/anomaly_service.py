class AnomalyService:

    def __init__(self):
        pass

    def predict(self, reading):

        temperature = reading.temperature
        pressure = reading.pressure
        humidity = reading.humidity

        # -------------------------------------------------
        # 1. Missing value check
        # -------------------------------------------------

        if (
            temperature is None
            or pressure is None
            or humidity is None
        ):
            return {
                "is_anomaly": True,
                "anomaly_type": "MISSING_DATA",
                "score": 0.90,
                "confidence": 0.95,
                "severity": "HIGH",
                "reason": "One or more weather parameters are missing"
            }

        # -------------------------------------------------
        # 2. Physical range validation
        # -------------------------------------------------

        if temperature < -90 or temperature > 60:

            return {
                "is_anomaly": True,
                "anomaly_type": "TEMPERATURE_SENSOR_FAULT",
                "score": 0.98,
                "confidence": 0.98,
                "severity": "CRITICAL",
                "reason": f"Temperature value {temperature}°C is outside the expected physical range"
            }

        if pressure < 850 or pressure > 1100:

            return {
                "is_anomaly": True,
                "anomaly_type": "PRESSURE_SENSOR_FAULT",
                "score": 0.98,
                "confidence": 0.98,
                "severity": "CRITICAL",
                "reason": f"Pressure value {pressure} hPa is outside the expected physical range"
            }

        if humidity < 0 or humidity > 100:

            return {
                "is_anomaly": True,
                "anomaly_type": "HUMIDITY_SENSOR_FAULT",
                "score": 0.98,
                "confidence": 0.98,
                "severity": "CRITICAL",
                "reason": f"Humidity value {humidity}% is outside the valid range"
            }

        # -------------------------------------------------
        # 3. Suspicious temperature
        # -------------------------------------------------

        if temperature >= 50:

            return {
                "is_anomaly": True,
                "anomaly_type": "TEMPERATURE_ANOMALY",
                "score": 0.95,
                "confidence": 0.95,
                "severity": "CRITICAL",
                "reason": f"Extremely high temperature detected: {temperature}°C"
            }

        # -------------------------------------------------
        # 4. Suspicious humidity
        # -------------------------------------------------

        if humidity >= 99 and temperature >= 40:

            return {
                "is_anomaly": True,
                "anomaly_type": "MULTIVARIATE_INCONSISTENCY",
                "score": 0.92,
                "confidence": 0.90,
                "severity": "HIGH",
                "reason": (
                    f"Temperature ({temperature}°C) and humidity "
                    f"({humidity}%) show an unusual combination"
                )
            }

        # -------------------------------------------------
        # 5. Normal reading
        # -------------------------------------------------

        return {
            "is_anomaly": False,
            "anomaly_type": None,
            "score": 0.05,
            "confidence": 0.95,
            "severity": "NORMAL",
            "reason": "Weather reading is within the current validation limits"
        }