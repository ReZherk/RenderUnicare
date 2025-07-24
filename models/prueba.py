import tensorflow as tf
import numpy as np
import joblib

# Cargar modelo y scaler
modelo = tf.keras.models.load_model("depression_model.h5")
scaler = joblib.load("scaler.save")

# Nuevos datos (puedes cambiar esto por cualquier combinación válida de 9 valores)
nuevos_datos = np.array([[1, 1, 1, 0, 0, 1, 0, 1, 1]])




# Escalar los datos nuevos con el mismo scaler usado en el entrenamiento
nuevos_datos_scaled = scaler.transform(nuevos_datos)

# Realizar la predicción
prediccion = modelo.predict(nuevos_datos_scaled)
porcentaje = prediccion[0][0] * 100

print(f"Probabilidad de depresión: {porcentaje:.2f}%")


# Interpretar el resultado
resultado = (prediccion > 0.5).astype(int)[0][0]

if resultado == 1:
    print("✅ Resultado: El estudiante TIENE depresión.")
else:
    print("✅ Resultado: El estudiante NO tiene depresión.")
