from flask import Flask, render_template, jsonify, request, send_file
import pandas as pd
import numpy as np
import json
import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from datetime import datetime
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from curve_fitting import LSMSensorFitter

app = Flask(__name__)

try:
    data = pd.read_csv('data/sensor_data.csv')
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    fitter = LSMSensorFitter()
    fitter.load_models()
    
    with open('models/models_metadata.json', 'r') as f:
        metadata = json.load(f)
except:
    data = None
    fitter = None
    metadata = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/data/overview')
def data_overview():
    if data is None:
        return jsonify({'error': 'Data tidak ditemukan'})
    
    overview = {
        'total_samples': len(data),
        'date_range': {
            'start': data['timestamp'].min().strftime('%Y-%m-%d %H:%M'),
            'end': data['timestamp'].max().strftime('%Y-%m-%d %H:%M')
        },
        'sensor_stats': {
            'temperature': {
                'min': float(data['temperature'].min()),
                'max': float(data['temperature'].max()),
                'mean': float(data['temperature'].mean()),
                'std': float(data['temperature'].std())
            },
            'humidity': {
                'min': float(data['humidity'].min()),
                'max': float(data['humidity'].max()),
                'mean': float(data['humidity'].mean()),
                'std': float(data['humidity'].std())
            },
            'light_intensity': {
                'min': float(data['light_intensity'].min()),
                'max': float(data['light_intensity'].max()),
                'mean': float(data['light_intensity'].mean()),
                'std': float(data['light_intensity'].std())
            }
        }
    }
    
    return jsonify(overview)

@app.route('/api/models')
def get_models():
    return jsonify(metadata)

@app.route('/api/plots/correlation')
def correlation_plot():
    if data is None:
        return jsonify({'error': 'Data tidak ditemukan'})
    
    corr_matrix = data[['temperature', 'humidity', 'light_intensity']].corr()
    
    fig = px.imshow(corr_matrix,
                   text_auto=True,
                   color_continuous_scale='RdBu',
                   title='Korelasi antar Variabel Sensor')
    
    return jsonify(json.loads(fig.to_json()))

@app.route('/api/plots/time-series')
def time_series_plot():
    if data is None:
        return jsonify({'error': 'Data tidak ditemukan'})
    
    sample_data = data.iloc[::10]
    
    fig = make_subplots(rows=3, cols=1,
                       subplot_titles=('Suhu vs Waktu', 
                                      'Kelembaban vs Waktu', 
                                      'Intensitas Cahaya vs Waktu'))
    
    fig.add_trace(
        go.Scatter(x=sample_data['timestamp'], 
                  y=sample_data['temperature'],
                  mode='lines',
                  name='Suhu',
                  line=dict(color='blue')),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=sample_data['timestamp'], 
                  y=sample_data['humidity'],
                  mode='lines',
                  name='Kelembaban',
                  line=dict(color='green')),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=sample_data['timestamp'], 
                  y=sample_data['light_intensity'],
                  mode='lines',
                  name='Cahaya',
                  line=dict(color='red')),
        row=3, col=1
    )
    
    fig.update_layout(height=800, showlegend=False)
    fig.update_xaxes(title_text="Waktu", row=3, col=1)
    
    return jsonify(json.loads(fig.to_json()))

@app.route('/api/plots/scatter-fitting')
def scatter_fitting_plot():
    if data is None or fitter.models == {}:
        return jsonify({'error': 'Model tidak ditemukan'})
    
    fig = make_subplots(rows=1, cols=2,
                       subplot_titles=('Suhu vs Kelembaban', 
                                      'Cahaya vs Suhu'))
    
    if 'temp_vs_humidity' in fitter.models:
        model = fitter.models['temp_vs_humidity']
        x = data['temperature'].values
        y = data['humidity'].values
        
        fig.add_trace(
            go.Scatter(x=x[::10], y=y[::10],
                      mode='markers',
                      name='Data',
                      marker=dict(color='blue', opacity=0.5)),
            row=1, col=1
        )
        
        x_fit = np.linspace(min(x), max(x), 100)
        intercept = model['coefficients']['intercept']
        slope = model['coefficients']['slope']
        y_fit = intercept + slope * x_fit
        
        fig.add_trace(
            go.Scatter(x=x_fit, y=y_fit,
                      mode='lines',
                      name=f"Linear Fit (R²={model['metrics']['r2']:.3f})",
                      line=dict(color='red', width=3)),
            row=1, col=1
        )
    
    light_nonzero = data[data['light_intensity'] > 0]
    if len(light_nonzero) > 10 and 'light_vs_temp' in fitter.models:
        model = fitter.models['light_vs_temp']
        x = light_nonzero['light_intensity'].values
        y = light_nonzero['temperature'].values
        
        fig.add_trace(
            go.Scatter(x=x[::10], y=y[::10],
                      mode='markers',
                      name='Data',
                      marker=dict(color='green', opacity=0.5)),
            row=1, col=2
        )
        
        x_fit = np.linspace(min(x), max(x), 100)
        coeffs = model['coefficients']
        poly = np.poly1d(coeffs)
        y_fit = poly(x_fit)
        
        fig.add_trace(
            go.Scatter(x=x_fit, y=y_fit,
                      mode='lines',
                      name=f"Poly Fit (R²={model['metrics']['r2']:.3f})",
                      line=dict(color='orange', width=3)),
            row=1, col=2
        )
    
    fig.update_layout(height=500, showlegend=True)
    fig.update_xaxes(title_text="Suhu (°C)", row=1, col=1)
    fig.update_yaxes(title_text="Kelembaban (%)", row=1, col=1)
    fig.update_xaxes(title_text="Intensitas Cahaya (lux)", row=1, col=2)
    fig.update_yaxes(title_text="Suhu (°C)", row=1, col=2)
    
    return jsonify(json.loads(fig.to_json()))

@app.route('/api/predict', methods=['POST'])
def predict():
    if fitter.models == {}:
        return jsonify({'error': 'Model tidak tersedia'})
    
    try:
        request_data = request.get_json()
        model_type = request_data.get('model_type', 'temp_vs_humidity')
        input_value = float(request_data.get('input_value', 25))
        
        if model_type not in fitter.models:
            return jsonify({'error': 'Model tidak ditemukan'})
        
        model = fitter.models[model_type]
        
        if model['model'] == 'linear':
            intercept = model['coefficients']['intercept']
            slope = model['coefficients']['slope']
            prediction = intercept + slope * input_value
        
        elif 'polynomial' in model['model']:
            coeffs = model['coefficients']
            poly = np.poly1d(coeffs)
            prediction = poly(input_value)
        
        elif model['model'] == 'exponential':
            a = model['coefficients']['a']
            b = model['coefficients']['b']
            c = model['coefficients']['c']
            prediction = a * np.exp(b * input_value) + c
        
        else:
            prediction = None
        
        return jsonify({
            'model': model_type,
            'equation': model['equation'],
            'input': input_value,
            'prediction': float(prediction) if prediction is not None else None,
            'r_squared': model['metrics']['r2']
        })
    
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/download/report')
def download_report():
    report_content = f"""
    LAPORAN ANALISIS DATA SENSOR IoT MENGGUNAKAN LEAST SQUARES METHOD
    ===============================================================
    
    Tanggal: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    
    DATA OVERVIEW:
    --------------
    Total sampel: {len(data) if data is not None else 'N/A'}
    
    HASIL CURVE FITTING:
    --------------------
    """
    
    if metadata:
        for model_name, metrics in metadata.get('metrics', {}).items():
            report_content += f"\nModel: {model_name}\n"
            report_content += f"Equation: {metadata.get('equations', {}).get(model_name, 'N/A')}\n"
            report_content += f"R-squared: {metrics.get('r2', 'N/A'):.4f}\n"
            report_content += f"MSE: {metrics.get('mse', 'N/A'):.4f}\n"
            report_content += f"MAE: {metrics.get('mae', 'N/A'):.4f}\n"
            report_content += "-" * 50 + "\n"
    
    buffer = io.BytesIO()
    buffer.write(report_content.encode('utf-8'))
    buffer.seek(0)
    
    return send_file(buffer,
                    as_attachment=True,
                    download_name=f'sensor_analysis_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt',
                    mimetype='text/plain')

if __name__ == '__main__':
    import os
    os.makedirs('templates', exist_ok=True)
    
    with open('templates/index.html', 'w') as f:
        f.write("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dashboard Analisis Sensor IoT - LSM</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { background-color: #f8f9fa; }
        .dashboard-header { 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
        }
        .card { 
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-bottom: 1rem;
            transition: transform 0.3s;
        }
        .card:hover { transform: translateY(-5px); }
        .plot-container { 
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .metric-card { text-align: center; }
        .metric-value { 
            font-size: 2.5rem;
            font-weight: bold;
            color: #667eea;
        }
        .team-info { 
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <div class="container-fluid">
        <div class="dashboard-header">
            <h1>📊 Dashboard Analisis Data Sensor IoT</h1>
            <p class="lead">Curve Fitting Menggunakan Least Squares Method (LSM)</p>
            <p class="mb-0">Kelompok: Muhammad Isra Dwi Firmansya (2401020143) | Nurfaizah Rasikha (2401020156) | Shalsabyla Finta Azalea (2401020167) | Teguh Hidayat (2401020168)</p>
        </div>
        
        <div class="row" id="metrics-overview"></div>
        
        <div class="row">
            <div class="col-md-12">
                <div class="plot-container">
                    <h4>📈 Korelasi antar Variabel Sensor</h4>
                    <div id="correlation-plot"></div>
                </div>
            </div>
        </div>
        
        <div class="row">
            <div class="col-md-12">
                <div class="plot-container">
                    <h4>⏰ Time Series Data Sensor</h4>
                    <div id="time-series-plot"></div>
                </div>
            </div>
        </div>
        
        <div class="row">
            <div class="col-md-12">
                <div class="plot-container">
                    <h4>📐 Hasil Curve Fitting</h4>
                    <div id="scatter-fitting-plot"></div>
                </div>
            </div>
        </div>
        
        <div class="row">
            <div class="col-md-12">
                <div class="plot-container">
                    <h4>🤖 Model Regression</h4>
                    <div id="model-info"></div>
                </div>
            </div>
        </div>
        
        <div class="row">
            <div class="col-md-6">
                <div class="card">
                    <div class="card-body">
                        <h5 class="card-title">🔮 Prediksi menggunakan Model</h5>
                        <div class="mb-3">
                            <label class="form-label">Pilih Model:</label>
                            <select class="form-select" id="model-select">
                                <option value="temp_vs_humidity">Suhu → Kelembaban</option>
                                <option value="light_vs_temp">Cahaya → Suhu</option>
                            </select>
                        </div>
                        <div class="mb-3">
                            <label class="form-label">Input Value:</label>
                            <input type="number" class="form-control" id="input-value" step="0.1" value="25">
                        </div>
                        <button class="btn btn-primary" onclick="predict()">Prediksi</button>
                        <div class="mt-3" id="prediction-result"></div>
                    </div>
                </div>
            </div>
            
            <div class="col-md-6">
                <div class="card">
                    <div class="card-body">
                        <h5 class="card-title">📥 Download</h5>
                        <p>Download laporan hasil analisis:</p>
                        <a href="/download/report" class="btn btn-success">📄 Download Report</a>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="team-info">
            <h5>👥 Informasi Kelompok</h5>
            <table class="table table-striped">
                <thead>
                    <tr>
                        <th>Nama</th>
                        <th>NIM</th>
                        <th>Peran</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Muhammad Isra Dwi Firmansya</td>
                        <td>2401020143</td>
                        <td>Data Collection & Analysis</td>
                    </tr>
                    <tr>
                        <td>Nurfaizah Rasikha</td>
                        <td>2401020156</td>
                        <td>LSM Implementation</td>
                    </tr>
                    <tr>
                        <td>Shalsabyla Finta Azalea</td>
                        <td>2401020167</td>
                        <td>Dashboard Development</td>
                    </tr>
                    <tr>
                        <td>Teguh Hidayat</td>
                        <td>2401020168</td>
                        <td>Testing & Documentation</td>
                    </tr>
                </tbody>
            </table>
        </div>
    </div>
    
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            loadOverview();
            loadCorrelationPlot();
            loadTimeSeriesPlot();
            loadScatterFittingPlot();
            loadModelInfo();
        });
        
        function loadOverview() {
            fetch('/api/data/overview')
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        document.getElementById('metrics-overview').innerHTML = 
                            `<div class="alert alert-warning">${data.error}</div>`;
                        return;
                    }
                    
                    const html = `
                        <div class="col-md-4">
                            <div class="card metric-card">
                                <div class="card-body">
                                    <h5 class="card-title">Sampel Data</h5>
                                    <div class="metric-value">${data.total_samples}</div>
                                    <p class="text-muted">Total pengukuran</p>
                                </div>
                            </div>
                        </div>
                    `;
                    
                    document.getElementById('metrics-overview').innerHTML = html;
                });
        }
        
        function loadCorrelationPlot() {
            fetch('/api/plots/correlation')
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        document.getElementById('correlation-plot').innerHTML = 
                            `<div class="alert alert-warning">${data.error}</div>`;
                        return;
                    }
                    
                    Plotly.newPlot('correlation-plot', data.data, data.layout);
                });
        }
        
        function loadTimeSeriesPlot() {
            fetch('/api/plots/time-series')
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        document.getElementById('time-series-plot').innerHTML = 
                            `<div class="alert alert-warning">${data.error}</div>`;
                        return;
                    }
                    
                    Plotly.newPlot('time-series-plot', data.data, data.layout);
                });
        }
        
        function loadScatterFittingPlot() {
            fetch('/api/plots/scatter-fitting')
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        document.getElementById('scatter-fitting-plot').innerHTML = 
                            `<div class="alert alert-warning">${data.error}</div>`;
                        return;
                    }
                    
                    Plotly.newPlot('scatter-fitting-plot', data.data, data.layout);
                });
        }
        
        function loadModelInfo() {
            fetch('/api/models')
                .then(response => response.json())
                .then(data => {
                    let html = '';
                    if (data.models_trained) {
                        data.models_trained.forEach(modelName => {
                            const metrics = data.metrics[modelName] || {};
                            const equation = data.equations[modelName] || '';
                            
                            html += `
                                <div class="card mb-3">
                                    <div class="card-body">
                                        <h5>${modelName}</h5>
                                        <p><strong>Equation:</strong> ${equation}</p>
                                        <p><strong>R²:</strong> ${metrics.r2 ? metrics.r2.toFixed(4) : 'N/A'}</p>
                                    </div>
                                </div>
                            `;
                        });
                    }
                    document.getElementById('model-info').innerHTML = html;
                });
        }
        
        function predict() {
            const modelType = document.getElementById('model-select').value;
            const inputValue = document.getElementById('input-value').value;
            
            fetch('/api/predict', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    model_type: modelType,
                    input_value: parseFloat(inputValue)
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    document.getElementById('prediction-result').innerHTML = 
                        `<div class="alert alert-danger">${data.error}</div>`;
                } else {
                    document.getElementById('prediction-result').innerHTML = `
                        <div class="alert alert-success">
                            <h6>Hasil Prediksi:</h6>
                            <p>Model: ${data.model}</p>
                            <p>Equation: ${data.equation}</p>
                            <p>Input: ${data.input}</p>
                            <p><strong>Prediction: ${data.prediction.toFixed(2)}</strong></p>
                            <p>R²: ${data.r_squared.toFixed(4)}</p>
                        </div>
                    `;
                }
            });
        }
    </script>
</body>
</html>
        """)
    
    app.run(debug=True, port=5000)
