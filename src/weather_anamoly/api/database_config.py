import sqlite3


class DataBase:
    def init_db(self):
        conn = sqlite3.connect("weather_anomaly.db")
        cursor = conn.cursor()
        cursor.execute('''
                CREATE TABLE IF NOT EXISTS weather_data(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date DATETIME DEFAULT CURRENT_TIMESTAMP,
                    temperature REAL,
                    humidity REAL,
                    pressure REAL,
                    temp_gradient REAL,
                    humid_gradient REAL,
                    press_gradient REAL,
                    six_hr_temp_gradient REAL,
                    six_hr_humid_gradient REAL,
                    six_hr_press_gradient REAL,
                    label TEXT
                );
        ''')
        conn.commit()
        conn.close()

    def inserting_values(self, temperature, pressure, humidity, temp_gradient, press_gradient, humid_gradient, six_hr_temp, six_hr_press, six_hr_humid, label):
        conn = sqlite3.connect("weather_anomaly.db")
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO weather_data (
            temperature,
            humidity,
            pressure,
            temp_gradient,
            humid_gradient,
            press_gradient,
            six_hr_temp_gradient,
            six_hr_humid_gradient,
            six_hr_press_gradient,
            label
            ) VALUES (?,?,?,?,?,?,?,?,?,?)
            """, (temperature, humidity, pressure, temp_gradient, humid_gradient, press_gradient, six_hr_temp, six_hr_humid, six_hr_press, label)
                       )
        conn.commit()
        conn.close()