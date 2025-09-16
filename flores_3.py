# Importación de librerías necesarias
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns

# Carga del dataset Iris
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

print("Dataset cargado. Características:", feature_names)
print("Número de muestras:", X.shape[0])

# Lista para almacenar los valores de WCSS
wcss = []
# Rango de K a probar (de 1 a 10)
range_k = range(1, 11)

# Iterar para calcular WCSS para cada K
for i in range_k:
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_) # 'inertia_' es el atributo WCSS en scikit-learn

# Graficar los resultados
plt.figure(figsize=(10, 6))
plt.plot(range_k, wcss, marker='o', linestyle='--')
plt.title('Método del Codo para el Dataset Iris')
plt.xlabel('Número de Clústeres (K)')
plt.ylabel('WCSS')
plt.xticks(range_k)
plt.grid(True)
plt.show()

print("Análisis del Método del Codo: El valor óptimo de K es 3, que corresponde al número de especies.")

# Implementación de K-means con K=3
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
y_kmeans = kmeans.fit_predict(X)

# Visualización de los clústeres (usando las dos primeras características)
plt.figure(figsize=(10, 8))
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=100, c='purple', label='Clúster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=100, c='blue', label='Clúster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=100, c='green', label='Clúster 3')

# Visualización de los centroides de los clústeres
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', label='Centroides', marker='*')
plt.title('Clústeres de Iris (K-means)')
plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])
plt.legend()
plt.show()

# Análisis de la correspondencia entre clústeres y etiquetas reales
# (Esto requiere un mapeo manual, ya que K-means no aprende etiquetas)
from scipy.stats import mode

labels = np.zeros_like(y_kmeans)
for i in range(3):
    mask = (y_kmeans == i)
    labels[mask] = mode(y[mask])[0]

print("\nAnálisis del K-means:")
print("Etiquetas reales (0: setosa, 1: versicolor, 2: virginica)")
print("Clústeres asignados (0, 1, 2) después del mapeo:")
print(labels)

# Matriz de confusión para K-means (aproximada)
cm_kmeans = confusion_matrix(y, labels)
sns.heatmap(cm_kmeans, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.title('Matriz de Confusión para K-means')
plt.xlabel('Clústeres Asignados')
plt.ylabel('Etiquetas Reales')
plt.show()

accuracy_kmeans = accuracy_score(y, labels)
print(f"Precisión (Accuracy) de K-means: {accuracy_kmeans:.2f}")

# División de datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Escalar los datos para un mejor rendimiento del MLP
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Inicialización del MLP con hiperparámetros predeterminados
mlp_model = MLPClassifier(hidden_layer_sizes=(10, 5), # 2 capas ocultas con 10 y 5 neuronas
                          activation='relu', # Función de activación ReLU
                          solver='adam', # Optimizador Adam
                          alpha=0.0001, # L2 penalty (regularización)
                          learning_rate_init=0.001, # Tasa de aprendizaje inicial
                          max_iter=500, # Número máximo de épocas
                          random_state=42)

# Entrenamiento del modelo
mlp_model.fit(X_train_scaled, y_train)

# Predicción en el conjunto de prueba
y_pred = mlp_model.predict(X_test_scaled)

# Evaluación del modelo
print("\nResultados del Perceptrón Multicapa:")
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión (Accuracy) del modelo en el conjunto de prueba: {accuracy:.2f}")

# Matriz de Confusión
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.title('Matriz de Confusión para MLP')
plt.xlabel('Predicciones')
plt.ylabel('Etiquetas Reales')
plt.show()

# Reporte de clasificación (Precisión, Exhaustividad, F1-Score)
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred, target_names=target_names))

# Configuración de K-fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Modelo MLP a validar
mlp_cv = MLPClassifier(hidden_layer_sizes=(10, 5), activation='relu', solver='adam', max_iter=500, random_state=42)

# Realizar la validación cruzada
cv_scores = cross_val_score(mlp_cv, X, y, cv=kf, scoring='accuracy')

print("\nResultados de la Validación Cruzada (5-fold):")
print(f"Puntuaciones de precisión por fold: {cv_scores}")
print(f"Precisión media: {cv_scores.mean():.2f}")
print(f"Desviación estándar de la precisión: {cv_scores.std():.2f}")