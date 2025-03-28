# Climate Predictor

## Overview
Climate_app is a web application that estimates temperature anomalies based on CO2 levels using a machine learning model. All based between 2000 - 2010

## Features
- Predicts temperature anomalies based on CO2 input.
- Interactive web interface.
- Simple API for predictions.
- Uses a pre-trained Random Forest model.

## Installation
1. Install Python 3.8+.
2. Install dependencies:
   ```bash
   pip install numpy pandas scikit-learn flask plotly joblib waitress
   ```

## Usage
### Run the App
```bash
python main.py
```
Access at: `http://localhost:5000`


### API Endpoint
- **POST /predict**
- Request JSON:
  ```json
  {"co2": 400}
  ```
- Response JSON:
  ```json
  {"prediction": 0.69}
  ```

## Deployment
For production, run:
```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

