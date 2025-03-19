"""
Climate app
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

# ---------- MODELO ----------
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

# ---------- WEB ----------
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Climate App</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        :root {
            --primary: #2A5C82;
            --secondary: #5DA271;
            --warning: #e67e22;
        }
        body { 
            font-family: 'Segoe UI', system-ui, sans-serif; 
            margin: 0; 
            padding: 20px; 
            background: #f8f9fa; 
            line-height: 1.6;
        }
        .container {
            max-width: 1000px;
            margin: 0 auto;
        }
        .card {
            background: white;
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            margin-bottom: 2rem;
        }
        h1 {
            color: var(--primary);
            text-align: center;
            margin: 2rem 0;
            font-size: 2.5rem;
        }
        h2 {
            color: var(--primary);
            margin-top: 0;
            font-size: 1.5rem;
        }
        .input-group {
            margin: 1.5rem 0;
        }
        label {
            display: block;
            margin-bottom: 0.5rem;
            font-weight: 500;
            color: #333;
        }
        input {
            width: 100%;
            padding: 12px;
            border: 2px solid #ddd;
            border-radius: 8px;
            font-size: 1.1rem;
            transition: border-color 0.3s;
        }
        input:focus {
            border-color: var(--primary);
            outline: none;
        }
        button {
            background: var(--primary);
            color: white;
            padding: 14px 28px;
            border: none;
            border-radius: 8px;
            font-size: 1.1rem;
            cursor: pointer;
            transition: transform 0.2s, background 0.3s;
            width: 100%;
        }
        button:hover {
            background: #1a4560;
            transform: translateY(-2px);
        }
        .result-box {
            padding: 1.5rem;
            background: #e8f4ff;
            border-radius: 8px;
            margin-top: 1.5rem;
            text-align: center;
        }
        .prediction-value {
            color: var(--primary);
            font-size: 2.2rem;
            font-weight: bold;
            margin: 0.5rem 0;
        }
        .info-text {
            color: #666;
            font-size: 0.95rem;
            margin: 0.5rem 0;
        }
        .visual-guide {
            display: flex;
            justify-content: space-between;
            margin: 1rem 0;
            padding: 1rem;
            background: #f3f8ff;
            border-radius: 8px;
        }
        .guide-item {
            text-align: center;
            padding: 0 1rem;
        }
        .guide-value {
            font-weight: bold;
            color: var(--primary);
        }
        .disclaimer {
            background: #fff3e0;
            padding: 1rem;
            border-radius: 8px;
            margin-top: 2rem;
            font-size: 0.9rem;
        }
        .plot-container {
            border-radius: 12px;
            overflow: hidden;
        }
        @media (min-width: 768px) {
            button {
                width: auto;
            }
            .input-group {
                max-width: 400px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Simulador de Impacto Climático</h1>
        
        <div class="card">
            <h2>Calculadora de Emisiones</h2>
            
            <div class="visual-guide">
                <div class="guide-item">
                    <div class="guide-value">417 ppm</div>
                    <div>Actual (2023)</div>
                </div>
                <div class="guide-item">
                    <div class="guide-value">300 ppm</div>
                    <div>Época preindustrial</div>
                </div>
            </div>

            <form id="predictForm">
                <div class="input-group">
                    <label for="co2-input">
                        Concentración de CO2 (partes por millón):
                        <span style="color: var(--warning);">*</span>
                    </label>
                    <input 
                        type="number" 
                        id="co2-input" 
                        name="co2" 
                        required
                        step="1"
                        min="200"
                        max="1000"
                        placeholder="Ej: 450">
                </div>
                
                <button type="submit">
                    📈 Calcular Impacto Climático
                </button>
            </form>

            <div id="predictionResult" class="result-box" style="display: none;">
                <div class="info-text">Incremento estimado de temperatura:</div>
                <div class="prediction-value" id="predictionValue">-- °C</div>
                <div class="info-text">
                    Comparado con el promedio histórico del siglo XX
                </div>
            </div>

            <div class="disclaimer">
                ⓘ Este modelo utiliza datos demostrativos. Los resultados son estimaciones 
                aproximadas con fines educativos.
            </div>
        </div>

        <div class="card">
            <h2>Tendencias Históricas</h2>
            <div class="plot-container">{{ plot|safe }}</div>
            <div class="info-text" style="margin-top: 1rem;">
                Datos de referencia: Muestra simulada para propósitos demostrativos
            </div>
        </div>
    </div>

    <script>
        async function handlePrediction(e) {
            e.preventDefault();
            const co2Input = document.getElementById('co2-input');
            const resultBox = document.getElementById('predictionResult');
            const predictionValue = document.getElementById('predictionValue');
            
            try {
                // Validación básica
                if(co2Input.value < 200 || co2Input.value > 1000) {
                    alert('Por favor ingresa un valor entre 200 y 1000 ppm');
                    return;
                }

                // Mostrar estado de carga
                resultBox.style.display = 'block';
                predictionValue.textContent = 'Calculando...';

                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ co2: parseFloat(co2Input.value) })
                });
                
                const result = await response.json();
                if(result.error) throw new Error(result.error);
                
                predictionValue.textContent = `${result.prediction} °C`;
                resultBox.style.backgroundColor = '#e8f5e9';

            } catch (error) {
                predictionValue.textContent = 'Error';
                resultBox.style.backgroundColor = '#ffeef0';
                console.error('Error:', error);
                alert('Ocurrió un error. Por favor intenta nuevamente.');
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