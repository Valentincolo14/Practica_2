# Importación de librerías necesarias
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns

# 1. Carga y preparación del dataset Iris
print("--- 1. Carga y preparación del dataset Iris ---")
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f"Tamaño del conjunto de entrenamiento: {X_train.shape}")
print(f"Tamaño del conjunto de prueba: {X_test.shape}\n")

# ---

# 2. Implementación y optimización del Gradient Boosting Classifier
print("--- 2. Gradient Boosting Classifier (GBC) ---")

# Parámetros iniciales
gbc = GradientBoostingClassifier(random_state=42)

# Grilla de hiperparámetros para Grid Search
param_grid_gbc = {
    'n_estimators': [50, 100, 150, 200],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 4, 5]
}

# Búsqueda de la mejor combinación de hiperparámetros con validación cruzada
grid_search_gbc = GridSearchCV(gbc, param_grid_gbc, cv=5, scoring='accuracy', n_jobs=-1)
grid_search_gbc.fit(X_train, y_train)

# Mostrar los mejores hiperparámetros
print(f"Mejores hiperparámetros para GBC: {grid_search_gbc.best_params_}")

# Entrenar el modelo final con los mejores parámetros
best_gbc = grid_search_gbc.best_estimator_
best_gbc.fit(X_train, y_train)

# Predecir y evaluar el modelo en el conjunto de prueba
y_pred_gbc = best_gbc.predict(X_test)
accuracy_gbc = accuracy_score(y_test, y_pred_gbc)
print(f"Precisión del GBC en el conjunto de prueba: {accuracy_gbc:.4f}")

# Mostrar el reporte de clasificación y la matriz de confusión
print("\nReporte de clasificación GBC:\n", classification_report(y_test, y_pred_gbc, target_names=target_names))
cm_gbc = confusion_matrix(y_test, y_pred_gbc)
print("Matriz de Confusión GBC:\n", cm_gbc)

# Visualización de la matriz de confusión
plt.figure(figsize=(6, 5))
sns.heatmap(cm_gbc, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.title('Matriz de Confusión - Gradient Boosting')
plt.xlabel('Clase Predicha')
plt.ylabel('Clase Verdadera')
plt.show()

# ---

# 3. Implementación y optimización del Support Vector Machine (SVM)
print("\n--- 3. Support Vector Machine (SVM) ---")

# Parámetros iniciales
svm = SVC(random_state=42)

# Grilla de hiperparámetros para Grid Search
param_grid_svm = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.1, 1]
}

# Búsqueda de la mejor combinación de hiperparámetros con validación cruzada
grid_search_svm = GridSearchCV(svm, param_grid_svm, cv=5, scoring='accuracy', n_jobs=-1)
grid_search_svm.fit(X_train, y_train)

# Mostrar los mejores hiperparámetros
print(f"Mejores hiperparámetros para SVM: {grid_search_svm.best_params_}")

# Entrenar el modelo final con los mejores parámetros
best_svm = grid_search_svm.best_estimator_
best_svm.fit(X_train, y_train)

# Predecir y evaluar el modelo en el conjunto de prueba
y_pred_svm = best_svm.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(f"Precisión del SVM en el conjunto de prueba: {accuracy_svm:.4f}")

# Mostrar el reporte de clasificación y la matriz de confusión
print("\nReporte de clasificación SVM:\n", classification_report(y_test, y_pred_svm, target_names=target_names))
cm_svm = confusion_matrix(y_test, y_pred_svm)
print("Matriz de Confusión SVM:\n", cm_svm)

# Visualización de la matriz de confusión
plt.figure(figsize=(6, 5))
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Purples', xticklabels=target_names, yticklabels=target_names)
plt.title('Matriz de Confusión - SVM')
plt.xlabel('Clase Predicha')
plt.ylabel('Clase Verdadera')
plt.show()

# ---

# 4. Resumen de resultados
print("\n--- 4. Conclusiones y Comparación de Resultados ---")
print(f"Precisión final del Gradient Boosting Classifier: {accuracy_gbc:.4f}")
print(f"Precisión final del Support Vector Machine: {accuracy_svm:.4f}")

# Conclusión
if accuracy_gbc > accuracy_svm:
    print("\nEl Gradient Boosting Classifier tuvo un mejor rendimiento.")
elif accuracy_svm > accuracy_gbc:
    print("\nEl Support Vector Machine tuvo un mejor rendimiento.")
else:
    print("\nAmbos modelos tuvieron un rendimiento similar.")