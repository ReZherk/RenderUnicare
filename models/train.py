import tensorflow as tf
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# Configurar para reproducibilidad
tf.random.set_seed(42)
np.random.seed(42)

df = pd.read_csv('trainer_data.csv')  # Carga el archivo CSV en un DataFrame de pandas
# Un dataframe es una estructura de datos bidimensional que se asemeja a una tabla, donde cada columna puede tener un tipo de dato diferente (números, cadenas, etc.).

print(f"Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
print(f"Distribución de clases: {df['depresion'].value_counts()}")

x = df.iloc[:, 0:9]  # Selecciona todas las filas y las columnas de la 0 a la 8
# iloc es un método de pandas que permite seleccionar filas y columnas por sus índices. La coma indica que se seleccionan todas las filas, y el rango 0:9 indica que se seleccionan las columnas desde la 0 hasta la 8 (sin incluir la 9).

y = df['depresion']  # Selecciona toda la columna 'depresion' del DataFrame df.

# Verificar que no haya valores nulos
if x.isnull().sum().sum() > 0 or y.isnull().sum() > 0:
    print("⚠️ Advertencia: Se encontraron valores nulos en los datos")
    # Opcional: rellenar o eliminar valores nulos
    x = x.fillna(0)
    y = y.fillna(0)

# Se usa StandardScaler cuando las variables tienen escalas muy diferentes (ej. Edad: 20-50 vs. Ingresos: 10,000-100,000)
# StandardScaler ajusta los datos para que tengan una media de 0 y una desviación estándar de 1,
# evitando que algunas variables influyan más que otras en el modelo.
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)  # Ajusta el escalador a los datos y transforma x

# Guarda el scaler para reutilizarlo y asegurar la misma normalización en futuras predicciones
joblib.dump(scaler, 'scaler.save')
print("✅ Scaler guardado exitosamente")

# train_test_split toma los datos y los divide en entrenamiento y prueba de manera aleatoria.
# Devuelve cuatro partes: X_train y X_test (datos de entrada) y y_train y y_test (resultados esperados).
X_train, X_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42, stratify=y)
# test_size=0.2: Porcentaje de datos reservados para prueba (20%)
# random_state fija la división para que siempre sea la misma en cada ejecución. Se usa para asegurar que los resultados sean reproducibles.
# stratify=y asegura que ambos conjuntos tengan la misma proporción de clases (balanceado)

print(f"Datos de entrenamiento: {X_train.shape[0]} muestras")
print(f"Datos de prueba: {X_test.shape[0]} muestras")

# Modelo con tres capas densas: 16 y 8 neuronas (ReLU) + 1 neurona (Sigmoid) para clasificación binaria
# Se usa ReLU (Rectified Linear Unit) como función de activación en las primeras capas para introducir no linealidades
# y Sigmoid en la última capa para obtener una probabilidad entre 0 y 1, adecuado para clasificación binaria.
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(X_train.shape[1],)),  # Capa de entrada con 16 neuronas y activación ReLU
    tf.keras.layers.Dropout(0.3),  # Dropout para evitar sobreajuste, desactiva aleatoriamente el 30% de las neuronas durante entrenamiento
    tf.keras.layers.Dense(8, activation='relu'),  # Capa oculta con 8 neuronas y activación ReLU
    tf.keras.layers.Dropout(0.2),  # Dropout adicional para regularización
    tf.keras.layers.Dense(1, activation='sigmoid')  # Capa de salida con 1 neurona y activación Sigmoid
])

# Compila el modelo con optimizador Adam, función de pérdida de entropía cruzada binaria y métricas de evaluación
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  # Optimizador Adam con tasa de aprendizaje específica
    loss='binary_crossentropy',  # Función de pérdida para clasificación binaria
    metrics=['accuracy', 'precision', 'recall']  # Métricas adicionales para mejor evaluación
)

print("📊 Resumen del modelo:")
model.summary()

# Callbacks para mejorar el entrenamiento
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',  # Monitorea la pérdida de validación
    patience=20,  # Espera 20 épocas sin mejora antes de detener
    restore_best_weights=True  # Restaura los mejores pesos encontrados
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',  # Monitorea la pérdida de validación
    factor=0.5,  # Reduce la tasa de aprendizaje a la mitad
    patience=10,  # Espera 10 épocas sin mejora antes de reducir
    min_lr=0.0001  # Tasa de aprendizaje mínima
)

# Entrena el modelo con callbacks para optimizar el proceso
history = model.fit(
    X_train, y_train,
    epochs=200,  # Más épocas con early stopping para encontrar el mejor punto
    batch_size=16,  # Tamaño de lote optimizado para este dataset
    validation_split=0.2,  # 20% de los datos de entrenamiento para validación
    callbacks=[early_stopping, reduce_lr],  # Callbacks para optimizar entrenamiento
    verbose=1
)

print("🎯 Entrenamiento completado")

# Evalúa el modelo con los datos de prueba y muestra métricas detalladas
loss, accuracy, precision, recall = model.evaluate(X_test, y_test, verbose=0)
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"\n📈 Métricas del modelo:")
print(f"Precisión (Accuracy): {accuracy:.4f}")
print(f"Precisión (Precision): {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1_score:.4f}")
print(f"Pérdida: {loss:.4f}")

# Predicciones para análisis detallado
y_pred_prob = model.predict(X_test, verbose=0)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

print(f"\n🔍 Reporte de clasificación:")
print(classification_report(y_test, y_pred, target_names=['No Depresión', 'Depresión']))

print(f"\n📊 Matriz de confusión:")
print(confusion_matrix(y_test, y_pred))

# Guarda el modelo entrenado
model.save('depression_model.h5')
print("✅ Modelo guardado exitosamente como 'depression_model.h5'")

# Prueba rápida del modelo guardado
print(f"\n🧪 Prueba del modelo guardado:")
modelo_cargado = tf.keras.models.load_model('depression_model.h5')
scaler_cargado = joblib.load('scaler.save')

# Ejemplo de predicción con datos de prueba (escala 0-3)
ejemplo_datos = np.array([[3, 1, 3, 3, 1, 2, 3, 1, 0]])  # Ejemplo con valores de escala Likert
ejemplo_escalado = scaler_cargado.transform(ejemplo_datos)
prediccion_ejemplo = modelo_cargado.predict(ejemplo_escalado, verbose=0)
porcentaje_ejemplo = prediccion_ejemplo[0][0] * 100

print(f"Ejemplo de predicción: {porcentaje_ejemplo:.2f}% probabilidad de depresión")
print(f"Resultado: {'TIENE indicadores de depresión' if porcentaje_ejemplo > 50 else 'NO tiene indicadores de depresión'}")