from src.weather_anamoly.database.database import save_alert


class AlertService:

    def create_alert(self, reading, communication_result, anomaly_result):
        """
        Create and store an alert if a fault is detected.
        """

        # Communication error has priority
        if communication_result["is_error"]:

            alert = {
                "station_id": reading.station_id,
                "timestamp": reading.timestamp.isoformat(),
                "alert_type": communication_result["fault_type"],
                "severity": communication_result["severity"],
                "confidence": communication_result["confidence"],
                "reason": communication_result["reason"]
            }

            alert_id = save_alert(
                alert["station_id"],
                alert["timestamp"],
                alert["alert_type"],
                alert["severity"],
                alert["confidence"],
                alert["reason"]
            )

            alert["id"] = alert_id

            return alert

        # Weather anomaly
        if anomaly_result["is_anomaly"]:

            alert = {
                "station_id": reading.station_id,
                "timestamp": reading.timestamp.isoformat(),
                "alert_type": anomaly_result["anomaly_type"],
                "severity": anomaly_result["severity"],
                "confidence": anomaly_result["confidence"],
                "reason": anomaly_result["reason"]
            }

            alert_id = save_alert(
                alert["station_id"],
                alert["timestamp"],
                alert["alert_type"],
                alert["severity"],
                alert["confidence"],
                alert["reason"]
            )

            alert["id"] = alert_id

            return alert

        return None