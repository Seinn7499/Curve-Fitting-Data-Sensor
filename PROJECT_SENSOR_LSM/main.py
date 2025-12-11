"""
Main file untuk menjalankan seluruh proyek LSM Sensor IoT
Urutan eksekusi:
1. Generate data sensor
2. LSM curve fitting
3. Jalankan dashboard
"""

import subprocess
import sys
import os

def install_requirements():
    """Install requirements.txt"""
    print("Menginstal dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def run_data_collection():
    """Jalankan data collection"""
    print("\n" + "="*50)
    print("1. GENERATING SENSOR DATA")
    print("="*50)
    
    # Import dan jalankan generator
    from data_collection import SensorDataGenerator
    generator = SensorDataGenerator(num_samples=500)
    data = generator.save_to_csv()
    
    # Simpan data kalibrasi
    temp_cal, hum_cal = generator.add_calibration_data()
    temp_cal.to_csv('data/temperature_calibration.csv', index=False)
    hum_cal.to_csv('data/humidity_calibration.csv', index=False)
    
    return data

def run_curve_fitting():
    """Jalankan curve fitting"""
    print("\n" + "="*50)
    print("2. CURVE FITTING DENGAN LSM")
    print("="*50)
    
    # Import dan jalankan fitter
    from curve_fitting import LSMSensorFitter
    import pandas as pd
    
    # Load data
    data = pd.read_csv('data/sensor_data.csv')
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    
    # Inisialisasi dan fitting
    fitter = LSMSensorFitter()
    results = fitter.analyze_sensor_relationships(data)
    
    # Print hasil
    print("\nHASIL CURVE FITTING:")
    for name, result in results.items():
        print(f"\nModel: {name}")
        print(f"Equation: {result['equation']}")
        print(f"R-squared: {result['metrics']['r2']:.4f}")
    
    # Generate plots
    plots = fitter.plot_results(data)
    print(f"\nPlots generated: {plots}")
    
    # Simpan model
    fitter.save_models()
    
    return fitter

def run_dashboard():
    """Jalankan dashboard web"""
    print("\n" + "="*50)
    print("3. STARTING WEB DASHBOARD")
    print("="*50)
    print("Dashboard akan berjalan di http://localhost:5000")
    print("Tekan Ctrl+C untuk menghentikan")
    
    # Jalankan dashboard
    subprocess.check_call([sys.executable, "dashboard_app.py"])

def main():
    """Fungsi utama"""
    print("="*60)
    print("PROYEK CURVE FITTING DATA SENSOR IoT MENGGUNAKAN LSM")
    print("="*60)
    print("Kelompok:")
    print("1. Muhammad Isra Dwi Firmansya (2401020143)")
    print("2. Nurfaizah Rasikha (2401020156)")
    print("3. Shalsabyla Finta Azalea (2401020167)")
    print("4. Teguh Hidayat (2401020168)")
    print("="*60)
    
    try:
        # Install requirements
        install_requirements()
        
        # Buat folder yang diperlukan
        os.makedirs('data', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        os.makedirs('static/plots', exist_ok=True)
        os.makedirs('templates', exist_ok=True)
        
        # Generate data
        data = run_data_collection()
        
        # Curve fitting
        fitter = run_curve_fitting()
        
        # Dashboard
        run_dashboard()
        
    except KeyboardInterrupt:
        print("\n\nProgram dihentikan oleh user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
