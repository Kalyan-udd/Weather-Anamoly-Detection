from datetime import datetime, timedelta


class CommunicationService:

    def __init__(self):
        # Store the last reading time for every station
        self.last_reading_time = {}

        # Consider communication failed
        # if no reading is received for 3 hours
        self.timeout_minutes = 180

    def check(self, reading):
        """
        Check whether communication with a weather station
        is working properly.
        """

        station_id = reading.station_id
        current_time = reading.timestamp

        # First reading from this station
        if station_id not in self.last_reading_time:

            self.last_reading_time[station_id] = current_time

            return {
                "is_error": False,
                "fault_type": None,
                "severity": "NORMAL",
                "confidence": 0.95,
                "reason": "First reading received from station"
            }

        # Get previous reading time
        previous_time = self.last_reading_time[station_id]

        # Calculate time difference
        time_difference = current_time - previous_time

        # Update last received reading
        self.last_reading_time[station_id] = current_time

        # Convert difference to minutes
        gap_minutes = time_difference.total_seconds() / 60

        # Communication failure
        if gap_minutes > self.timeout_minutes:

            return {
                "is_error": True,
                "fault_type": "COMMUNICATION_ERROR",
                "severity": "HIGH",
                "confidence": 0.99,
                "reason": f"No reading received for {int(gap_minutes)} minutes"
            }

        # Communication working normally
        return {
            "is_error": False,
            "fault_type": None,
            "severity": "NORMAL",
            "confidence": 0.95,
            "reason": "Communication with station is normal"
        }


# Create service object
communication_service = CommunicationService()