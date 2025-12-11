import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
import json
import os

class LSMSensorFitter:
    """Implementasi Least Squares Method untuk fitting data sensor"""
    
    def __init__(self):
        self.models = {}
        self.coefficients = {}
        self.metrics = {}
    
    def linear_lsm(self, x, y):
        """Linear Least Squares Method manual"""
        n = len(x)
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        sum_x2 = np.sum(x**2)
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)
        intercept = (sum_y - slope * sum_x) / n
        
        y_pred = intercept + slope * x
        
        r2 = r2_score(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        
        return {
            'model': 'linear',
            'coefficients': {'intercept': intercept, 'slope': slope},
            'equation': f"y = {intercept:.4f} + {slope:.4f}x",
            'predictions': y_pred,
            'metrics': {'r2': r2, 'mse': mse, 'mae': mae}
        }
    
    def polynomial_lsm(self, x, y, degree=2):
        """Polynomial Least Squares"""
        coefficients = np.polyfit(x, y, degree)
        polynomial = np.poly1d(coefficients)
        
        y_pred = polynomial(x)
        
        r2 = r2_score(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        
        eq_parts = []
        for i, coef in enumerate(coefficients):
            power = degree - i
            if power == 0:
                eq_parts.append(f"{coef:.4f}")
            elif power == 1:
                eq_parts.append(f"{coef:.4f}x")
            else:
                eq_parts.append(f"{coef:.4f}x^{power}")
        
        equation = "y = " + " + ".join(eq_parts)
        
        return {
            'model': f'polynomial_degree_{degree}',
            'coefficients': coefficients.tolist(),
            'equation': equation,
            'predictions': y_pred,
            'metrics': {'r2': r2, 'mse': mse, 'mae': mae}
        }
    
    def exponential_lsm(self, x, y):
        """Exponential fitting"""
        try:
            y_log = np.log(y)
            result = self.linear_lsm(x, y_log)
            
            a = np.exp(result['coefficients']['intercept'])
            b = result['coefficients']['slope']
            c = 0
            
            y_pred = a * np.exp(b * x) + c
            
            r2 = r2_score(y, y_pred)
            mse = mean_squared_error(y, y_pred)
            mae = mean_absolute_error(y, y_pred)
            
            return {
                'model': 'exponential',
                'coefficients': {'a': a, 'b': b, 'c': c},
                'equation': f"y = {a:.4f} * exp({b:.4f}*x) + {c:.4f}",
                'predictions': y_pred,
                'metrics': {'r2': r2, 'mse': mse, 'mae': mae}
            }
        except:
            return None
    
    def analyze_sensor_relationships(self, data):
        """Analisis hubungan antar variabel sensor"""
        results = {}
        
        results['temp_vs_humidity'] = self.linear_lsm(
            data['temperature'].values, 
            data['humidity'].values
        )
        
        light_nonzero = data[data['light_intensity'] > 0]
        if len(light_nonzero) > 10:
            results['light_vs_temp'] = self.polynomial_lsm(
                light_nonzero['light_intensity'].values,
                light_nonzero['temperature'].values,
                degree=2
            )
        
        try:
            cal_data = pd.read_csv('data/temperature_calibration.csv')
            results['temp_calibration'] = self.linear_lsm(
                cal_data['sensor_reading'].values,
                cal_data['reference_temp'].values
            )
        except:
            pass
        
        self.models = results
        return results
    
    def plot_results(self, data, save_path='static/plots'):
        """Generate visualisasi hasil fitting"""
        os.makedirs(save_path, exist_ok=True)
        
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        x = data['temperature'].values
        y = data['humidity'].values
        
        if 'temp_vs_humidity' in self.models:
            model = self.models['temp_vs_humidity']
            plt.scatter(x, y, alpha=0.5, label='Data Aktual')
            
            x_fit = np.linspace(min(x), max(x), 100)
            intercept = model['coefficients']['intercept']
            slope = model['coefficients']['slope']
            y_fit = intercept + slope * x_fit
            
            plt.plot(x_fit, y_fit, 'r-', linewidth=2, 
                    label=f"Linear Fit\nR² = {model['metrics']['r2']:.3f}")
            
            plt.xlabel('Suhu (°C)')
            plt.ylabel('Kelembaban (%)')
            plt.title('Hubungan Suhu vs Kelembaban')
            plt.legend()
            plt.grid(alpha=0.3)
        
        plt.subplot(1, 2, 2)
        light_nonzero = data[data['light_intensity'] > 0]
        
        if len(light_nonzero) > 10 and 'light_vs_temp' in self.models:
            x_light = light_nonzero['light_intensity'].values
            y_temp = light_nonzero['temperature'].values
            model = self.models['light_vs_temp']
            
            plt.scatter(x_light, y_temp, alpha=0.5, label='Data Aktual')
            
            x_fit = np.linspace(min(x_light), max(x_light), 100)
            coeffs = model['coefficients']
            poly = np.poly1d(coeffs)
            y_fit = poly(x_fit)
            
            plt.plot(x_fit, y_fit, 'g-', linewidth=2,
                    label=f"Polynomial Fit\nR² = {model['metrics']['r2']:.3f}")
            
            plt.xlabel('Intensitas Cahaya (lux)')
            plt.ylabel('Suhu (°C)')
            plt.title('Hubungan Intensitas Cahaya vs Suhu')
            plt.legend()
            plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/sensor_relationships.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        axes[0].plot(data['timestamp'][:100], data['temperature'][:100], 
                    'b-', alpha=0.7, label='Suhu Aktual')
        axes[0].set_ylabel('Suhu (°C)')
        axes[0].set_title('Time Series Data Sensor')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        axes[1].plot(data['timestamp'][:100], data['humidity'][:100], 
                    'g-', alpha=0.7, label='Kelembaban Aktual')
        axes[1].set_ylabel('Kelembaban (%)')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        axes[2].plot(data['timestamp'][:100], data['light_intensity'][:100], 
                    'r-', alpha=0.7, label='Cahaya Aktual')
        axes[2].set_ylabel('Intensitas Cahaya (lux)')
        axes[2].set_xlabel('Waktu')
        axes[2].legend()
        axes[2].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/time_series.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        return ['sensor_relationships.png', 'time_series.png']
    
    def save_models(self):
        """Simpan model ke file"""
        os.makedirs('models', exist_ok=True)
        
        joblib.dump(self.models, 'models/sensor_models.pkl')
        
        metadata = {
            'generated_at': pd.Timestamp.now().isoformat(),
            'models_trained': list(self.models.keys()),
            'metrics': {k: v['metrics'] for k, v in self.models.items()},
            'equations': {k: v['equation'] for k, v in self.models.items()}
        }
        
        with open('models/models_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=4)
        
        print("Models saved successfully!")
    
    def load_models(self):
        """Load model dari file"""
        try:
            self.models = joblib.load('models/sensor_models.pkl')
            print("Models loaded successfully!")
            return True
        except:
            print("No saved models found!")
            return False

if __name__ == "__main__":
    data = pd.read_csv('data/sensor_data.csv')
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    
    fitter = LSMSensorFitter()
    results = fitter.analyze_sensor_relationships(data)
    
    print("\n" + "="*50)
    print("HASIL CURVE FITTING DATA SENSOR IoT")
    print("="*50)
    
    for name, result in results.items():
        print(f"\nModel: {name}")
        print(f"Equation: {result['equation']}")
        print(f"R-squared: {result['metrics']['r2']:.4f}")
        print(f"MSE: {result['metrics']['mse']:.4f}")
        print(f"MAE: {result['metrics']['mae']:.4f}")
    
    plots = fitter.plot_results(data)
    fitter.save_models()
