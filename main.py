from data_processor import ClimateDataProcessor
from model_trainer import ClimateModel
import yaml

def main():
    # Cargar configuración
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
    
    # Procesar datos
    processor = ClimateDataProcessor(config)
    data = processor.load_data()
    
    # Entrenar modelo
    model = ClimateModel()
    features = config['model']['features']
    target = config['model']['target']
    model.train(data[features], data[target])
    
    print("Pipeline completado! Modelo listo para predicciones.")

if __name__ == '__main__':
    main()