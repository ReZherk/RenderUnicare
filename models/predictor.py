"""
Módulo de predicción de depresión usando IA
Maneja la carga del modelo y las predicciones
"""

import tensorflow as tf
import numpy as np
import joblib
import os

class DepressionPredictor:
    def __init__(self, model_path="depression_model.h5", scaler_path="scaler.save"):
        """
        Inicializa el predictor de depresión
        
        Args:
            model_path: Ruta al archivo del modelo
            scaler_path: Ruta al archivo del scaler
        """
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.modelo = None
        self.scaler = None
        self.load_model()
    
    def load_model(self):
        """Carga el modelo y scaler"""
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
                self.modelo = tf.keras.models.load_model(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                print("✅ Modelo y scaler cargados exitosamente")
            else:
                print(f"❌ Archivos no encontrados: {self.model_path} o {self.scaler_path}")
        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            self.modelo = None
            self.scaler = None
    
    def is_ready(self):
        """Verifica si el modelo está listo para predicciones"""
        return self.modelo is not None and self.scaler is not None
    
    def validate_input(self, responses_array):
        """
        Valida que los datos de entrada sean correctos
        
        Args:
            responses_array: Lista de respuestas
            
        Returns:
            dict: Resultado de validación con success y message
        """
        if len(responses_array) != 9:
            return {
                'success': False,
                'message': 'Se necesitan exactamente 9 respuestas'
            }
        
        if not all(isinstance(val, (int, float)) and 0 <= val <= 3 for val in responses_array):
            return {
                'success': False,
                'message': 'Todas las respuestas deben ser valores entre 0 y 3'
            }
        
        return {'success': True, 'message': 'Datos válidos'}
    
    def predict(self, responses_array):
        """
        Realiza predicción de depresión
        
        Args:
            responses_array: Lista de 9 valores en escala 0-3
            
        Returns:
            dict: Resultado de predicción con percentage, result, interpretation
        """
        # Verificar que el modelo esté disponible
        if not self.is_ready():
            return {
                'success': False,
                'error': 'Modelo no disponible',
                'percentage': 0,
                'result': 0,
                'interpretation': 'Error: Modelo no cargado'
            }
        
        # Validar datos de entrada
        validation = self.validate_input(responses_array)
        if not validation['success']:
            return {
                'success': False,
                'error': validation['message'],
                'percentage': 0,
                'result': 0,
                'interpretation': 'Error en datos de entrada'
            }
        
        try:
         
            nuevos_datos = np.array([responses_array])
            
            # Escalar los datos
            nuevos_datos_scaled = self.scaler.transform(nuevos_datos)
            
            # Realizar predicción
            prediccion = self.modelo.predict(nuevos_datos_scaled, verbose=0)
            porcentaje = float(prediccion[0][0] * 100)
            
            # Interpretar resultado (umbral de 50%)
            resultado_binario = int((prediccion > 0.5).astype(int)[0][0])
            
            interpretacion = (
                "El estudiante TIENE indicadores de depresión" 
                if resultado_binario == 1 
                else "El estudiante NO tiene indicadores de depresión"
            )
            
            return {
                'success': True,
                'percentage': round(porcentaje, 2),
                'result': resultado_binario,
                'interpretation': interpretacion,
                'error': None
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Error en predicción: {str(e)}',
                'percentage': 0,
                'result': 0,
                'interpretation': 'Error en predicción'
            }
    
    def get_prediction_summary(self, responses_array):
        """
        Obtiene un resumen completo de la predicción
        
        Args:
            responses_array: Lista de respuestas
            
        Returns:
            dict: Resumen completo con datos de entrada y predicción
        """
        prediction = self.predict(responses_array)
        
        return {
            'input_data': {
                'responses': responses_array,
                'total_responses': len(responses_array),
                'response_sum': sum(responses_array),
                'response_avg': round(sum(responses_array) / len(responses_array), 2)
            },
            'prediction': prediction,
            'model_info': {
                'model_ready': self.is_ready(),
                'model_path': self.model_path,
                'scaler_path': self.scaler_path
            }
        }

# Instancia global del predictor
predictor = DepressionPredictor()

# Funciones de conveniencia para usar en otros módulos
def predict_depression(responses_array):
    """Función de conveniencia para realizar predicción"""
    return predictor.predict(responses_array)

def is_model_ready():
    """Función de conveniencia para verificar si el modelo está listo"""
    return predictor.is_ready()

def get_prediction_summary(responses_array):
    """Función de conveniencia para obtener resumen completo"""
    return predictor.get_prediction_summary(responses_array)