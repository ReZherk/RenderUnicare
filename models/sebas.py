import numpy as np
import joblib
import tensorflow as tf

# Cargar modelo y scaler previamente entrenados
modelo = tf.keras.models.load_model('depression_model.h5')
scaler = joblib.load('scaler.save')

def predecir_depresion(valores):
    """
    Recibe una lista con 9 valores escala Likert (de 0 a 3) y devuelve el porcentaje de probabilidad.
    """
    if len(valores) != 9:
        raise ValueError("Se requieren exactamente 9 valores para la predicción.")

    datos = np.array([valores])
    datos_escalados = scaler.transform(datos)
    prediccion = modelo.predict(datos_escalados, verbose=0)[0][0]
    porcentaje = prediccion * 100

    print("\n🧠 Resultado:")
    print(f"Probabilidad de depresión: {porcentaje:.2f}%")
    if porcentaje > 50:
        print("🔴 El modelo estima que la persona TIENE indicadores de depresión.")
    else:
        print("🟢 El modelo estima que la persona NO tiene indicadores de depresión.")

# Ejemplo de uso:
if __name__ == '__main__':
    # Puedes modificar estos valores entre 0 y 3
    entrada = [1, 1, 0, 1, 0, 1, 0, 1, 1]
    predecir_depresion(entrada)