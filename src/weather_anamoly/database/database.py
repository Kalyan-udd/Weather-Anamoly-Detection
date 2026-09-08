import sqlite3
from pathlib import Path


# Project root directory
BASE_DIR = Path(__file__).resolve().parents[3]

# Database directory
DATA_DIR = BASE_DIR / "data"

# Create data directory if it doesn't exist
DATA_DIR.mkdir(exist_ok=True)

# SQLite database file
DATABASE_PATH = DATA_DIR / "weather.db"


def get_connection():
    """
    Create and return a connection to the SQLite database.
    """

    connection = sqlite3.connect(DATABASE_PATH)

    return connection


def initialize_database():
    """
    Create the required database tables if they don't already exist.
    """

    connection = get_connection()

    cursor = connection.cursor()

    # Weather readings table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            station_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            temperature REAL,
            pressure REAL,
            humidity REAL
        )
        """
    )

    # Alerts table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            station_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            alert_type TEXT NOT NULL,
            severity TEXT NOT NULL,
            confidence REAL,
            reason TEXT NOT NULL,
            status TEXT DEFAULT 'ACTIVE'
        )
        """
    )

    connection.commit()
    connection.close()


def save_reading(reading):
    """
    Save a weather reading into the database.
    """

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT INTO readings (
            station_id,
            timestamp,
            temperature,
            pressure,
            humidity
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            reading.station_id,
            reading.timestamp.isoformat(),
            reading.temperature,
            reading.pressure,
            reading.humidity
        )
    )

    connection.commit()

    reading_id = cursor.lastrowid

    connection.close()

    return reading_id


def save_alert(
    station_id,
    timestamp,
    alert_type,
    severity,
    confidence,
    reason
):
    """
    Save a detected alert into the database.
    """

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT INTO alerts (
            station_id,
            timestamp,
            alert_type,
            severity,
            confidence,
            reason
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            station_id,
            timestamp,
            alert_type,
            severity,
            confidence,
            reason
        )
    )

    connection.commit()

    alert_id = cursor.lastrowid

    connection.close()

    return alert_id


def get_alerts():
    """
    Return all alerts from the database.
    """

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT
            id,
            station_id,
            timestamp,
            alert_type,
            severity,
            confidence,
            reason,
            status
        FROM alerts
        ORDER BY id DESC
        """
    )

    rows = cursor.fetchall()

    connection.close()

    alerts = []

    for row in rows:
        alerts.append(
            {
                "id": row[0],
                "station_id": row[1],
                "timestamp": row[2],
                "alert_type": row[3],
                "severity": row[4],
                "confidence": row[5],
                "reason": row[6],
                "status": row[7]
            }
        )

    return alerts