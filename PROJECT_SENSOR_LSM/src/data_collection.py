import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

class SensorDataGenerator:
    """Generator data sensor IoT untuk simulasi"""
    
    def __init__(self, num_samples=1000):
        self.num_samples = num_samples
        np.random.seed(42)  # Untuk reproducibility
    
    def generate_sensor_data(self):
        """Generate data sensor suhu, kelembaban, dan cahaya"""
        start_time = datetime.now() - timedelta(days=30)
        timestamps = [start_time + timedelta(hours=i) 
                     for i in range(self.num_samples)]
        
        # Generate suhu
        base_temp = 25 + 5 * np.sin(np.linspace(0, 4*np.pi, self.num_samples))
        temperature = base_temp + np.random.normal(0, 0.5, self.num_samples)
        
        # Generate kelembaban
        humidity = 60 - 0.3 * (temperature - 25) + np.random.normal(0, 2, self.num_samples)
        humidity = np.clip(humidity, 30, 90)
        
        # Generate cahaya
        hour_of_day = np.array([ts.hour for ts in timestamps])
        light_intensity = 1000 * np.sin((hour_of_day - 6) * np.pi / 12)
        light_intensity[light_intensity < 0] = 0
        light_intensity += np.random.normal(0, 50, self.num_samples)
        
        # Buat DataFrame
        data = pd.DataFrame({
            'timestamp': timestamps,
            'temperature': np.round(temperature, 2),
            'humidity': np.round(humidity, 2),
            'light_intensity': np.round(light_intensity, 2),
            'sensor_id': 'IoT_Sensor_001',
            'location': 'Lab_Instrumentasi'
        })
        
        return data
    
    def add_calibration_data(self):
        """Generate data kalibrasi"""
        temp_calibration = pd.DataFrame({
            'reference_temp': [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            'sensor_reading': [0.1, 10.2, 20.1, 30.3, 40.2, 50.1, 60.3, 70.2, 80.1, 90.3, 100.2]
        })
        
        humidity_calibration = pd.DataFrame({
            'reference_humidity': [0, 25, 50, 75, 100],
            'sensor_reading': [1.2, 26.1, 51.3, 76.2, 101.1]
        })
        
        return temp_calibration, humidity_calibration
    
    def save_to_csv(self, filename='data/sensor_data.csv'):
        """Simpan data ke CSV"""
        data = self.generate_sensor_data()
        os.makedirs('data', exist_ok=True)
        data.to_csv(filename, index=False)
        print(f"Data saved to {filename}")
        print(f"Shape: {data.shape}")
        print(data.head())
        return data

if __name__ == "__main__":
    generator = SensorDataGenerator(num_samples=500)
    data = generator.save_to_csv()
    
    # Simpan data kalibrasi
    temp_cal, hum_cal = generator.add_calibration_data()
    temp_cal.to_csv('data/temperature_calibration.csv', index=False)
    hum_cal.to_csv('data/humidity_calibration.csv', index=False)
