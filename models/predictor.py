"""
Módulo de predicción de depresión usando IA (versión TFLite)
Maneja la carga del modelo y las predicciones
"""

import tensorflow as tf
import numpy as np
import joblib
import os

class DepressionPredictor:
    def __init__(self, model_path="depression_model.tflite", scaler_path="scaler.save"):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.interpreter = None
        self.scaler = None
        self.input_details = None
        self.output_details = None
        self.load_model()
    
    def load_model(self):
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
                self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                self.scaler = joblib.load(self.scaler_path)
                print("✅ Modelo TFLite y scaler cargados exitosamente")
            else:
                print(f"❌ Archivos no encontrados: {self.model_path} o {self.scaler_path}")
        except Exception as e:
            print(f"❌ Error cargando modelo TFLite: {e}")
            self.interpreter = None
            self.scaler = None
    
    def is_ready(self):
        return self.interpreter is not None and self.scaler is not None

    def validate_input(self, responses_array):
        if len(responses_array) != 9:
            return {'success': False, 'message': 'Se necesitan exactamente 9 respuestas'}
        
        if not all(isinstance(val, (int, float)) and 0 <= val <= 3 for val in responses_array):
            return {'success': False, 'message': 'Todas las respuestas deben ser valores entre 0 y 3'}
        
        return {'success': True, 'message': 'Datos válidos'}

    def predict(self, responses_array):
        if not self.is_ready():
            return {
                'success': False,
                'error': 'Modelo no disponible',
                'percentage': 0,
                'result': 0,
                'interpretation': 'Error: Modelo no cargado'
            }

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
            input_data = np.array([responses_array], dtype=np.float32)
            input_scaled = self.scaler.transform(input_data).astype(np.float32)

            self.interpreter.set_tensor(self.input_details[0]['index'], input_scaled)
            self.interpreter.invoke()
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])

            prediction = float(output_data[0][0])
            porcentaje = prediction * 100
            resultado_binario = int(prediction >= 0.5)
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

def predict_depression(responses_array):
    return predictor.predict(responses_array)

def is_model_ready():
    return predictor.is_ready()

def get_prediction_summary(responses_array):
    return predictor.get_prediction_summary(responses_array)
