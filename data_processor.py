import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

class ClimateDataProcessor:
    def __init__(self, config):
        self.data_urls = config['data_sources']
        self.features = config['features']
        
    def load_data(self):
        """Descarga y combina datos de múltiples fuentes"""
        temp_data = pd.read_csv(self.data_urls['temperature'])
        co2_data = pd.read_csv(self.data_urls['co2'])
        
        # Fusionar datasets
        merged = pd.merge(temp_data, co2_data, on='Year', how='inner')
        
        # Procesamiento avanzado
        processed = self._clean_data(merged)
        return processed
    
    def _clean_data(self, df):
        """Limpieza y transformación de datos"""
        # Manejo de valores faltantes
        imputer = SimpleImputer(strategy='ffill')
        df[self.features] = imputer.fit_transform(df[self.features])
        
        # Creación de nuevas características
        df['CO2_5yr_avg'] = df['CO2'].rolling(5).mean()
        df['Temp_anomaly_diff'] = df['Temp_anomaly'].diff()
        
        return df.dropna()