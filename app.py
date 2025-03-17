from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load('models/climate_model.pkl')

@app.route('/')
def home():
    return render_template('dashboard.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_df = pd.DataFrame([data])
    prediction = model.predict(input_df)[0]
    return jsonify({'prediction': round(prediction, 2)})

@app.route('/visualization')
def visualization():
    # Generar gráficos interactivos con Plotly
    return render_template('visualization.html')

if __name__ == '__main__':
    app.run(debug=True)