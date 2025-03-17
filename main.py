"""
CLIMATE PREDICTOR - Versión Final Estable
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from flask import Flask, request, jsonify, render_template_string
import joblib
import plotly.express as px
from io import StringIO

# ---------- CONFIGURACIÓN FINAL ----------
app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Desactiva caché para desarrollo

# ---------- DATOS DE EJEMPLO OPTIMIZADOS ----------
def get_sample_data():
    return pd.DataFrame({
        'Year': range(2000, 2011),
        'co2': np.linspace(369, 390, 11),
        'temperature_anomaly': np.linspace(0.4, 0.72, 11)
    })

# ---------- MODELO FINAL ----------
def train_model():
    """Entrena modelo con datos sintéticos"""
    data = get_sample_data()
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(data[['co2']], data['temperature_anomaly'])
    joblib.dump(model, 'climate_model.pkl')
    return model

# Cargar modelo (versión robusta)
try:
    model = joblib.load('climate_model.pkl')
except (FileNotFoundError, EOFError):
    model = train_model()

# ---------- INTERFAZ WEB MEJORADA ----------
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Climate Predictor</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f0f4f8; }
        .container { max-width: 800px; margin: 0 auto; }
        .card { background: white; padding: 25px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        input { padding: 10px; margin: 10px 0; width: 200px; border: 1px solid #ddd; border-radius: 4px; }
        button { background: #2196F3; color: white; padding: 12px 24px; border: none; border-radius: 4px; cursor: pointer; }
        .result { margin-top: 20px; padding: 15px; background: #e8f5e9; border-radius: 4px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 Predictor Climático</h1>
        
        <div class="card">
            <h2>Simulador de CO2</h2>
            <form id="predictForm">
                <label>Concentración de CO2 (ppm):</label><br>
                <input type="number" step="0.1" name="co2" required><br>
                <button type="submit">Calcular Impacto</button>
            </form>
            <div id="predictionResult" class="result"></div>
        </div>

        <div class="card" style="margin-top: 30px;">
            <h2>Tendencia Histórica</h2>
            <div class="plot-container">{{ plot|safe }}</div>
        </div>
    </div>

    <script>
        async function handlePrediction(e) {
            e.preventDefault();
            const co2 = document.querySelector('input[name="co2"]').value;
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ co2: parseFloat(co2) })
                });
                
                const result = await response.json();
                document.getElementById('predictionResult').innerHTML = 
                    `Anomalía Térmica Estimada: <span style="font-size:1.4em; color:#2e7d32;">${result.prediction} °C</span>`;
            } catch (error) {
                console.error('Error:', error);
            }
        }
        
        document.getElementById('predictForm').addEventListener('submit', handlePrediction);
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    """Muestra la interfaz principal"""
    data = get_sample_data()
    fig = px.line(data, x='Year', y='temperature_anomaly',
                 title='Relación Histórica CO2 - Temperatura',
                 labels={'temperature_anomaly': 'Anomalía Térmica (°C)'},
                 template='ggplot2')
    return render_template_string(HTML_TEMPLATE, plot=fig.to_html())

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint de predicción mejorado"""
    try:
        co2_value = float(request.json.get('co2', 0))
        prediction = model.predict([[co2_value]])[0]
        return jsonify({'prediction': round(prediction, 2)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# ---------- CONFIGURACIÓN DE EJECUCIÓN ----------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)