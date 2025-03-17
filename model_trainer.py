from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
import joblib

class ClimateModel:
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            random_state=42
        )
    
    def train(self, X, y):
        """Entrenamiento con validación temporal"""
        tscv = TimeSeriesSplit(n_splits=5)
        metrics = []
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            self.model.fit(X_train, y_train)
            preds = self.model.predict(X_test)
            mae = mean_absolute_error(y_test, preds)
            metrics.append(mae)
            
        print(f"MAE promedio: {np.mean(metrics):.2f}")
        joblib.dump(self.model, 'models/climate_model.pkl')