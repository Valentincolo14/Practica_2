import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# ==============================================================================
# 1. Carga y Preparación del Dataset Iris
# ==============================================================================
print("1. Cargando y dividiendo el dataset Iris...")
iris = load_iris()
X = iris.data
y = iris.target

# División de los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print("Dataset dividido en entrenamiento (70%) y prueba (30%).")
print("-" * 50)

# ==============================================================================
# 2. Modelos SIN Optimizar (Parámetros por Defecto)
# ==============================================================================
print("2. Evaluación de modelos SIN optimizar (Naive Bayes y SVM).")

# Modelo Naive Bayes (GaussianNB)
print("\n--- Naive Bayes ---")
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred_gnb = gnb.predict(X_test)
precision_gnb = accuracy_score(y_test, y_pred_gnb)
print(f"Precisión de Naive Bayes: {precision_gnb:.4f}")
print(f"Reporte de Clasificación:\n{classification_report(y_test, y_pred_gnb, target_names=iris.target_names)}")

cm_gnb = confusion_matrix(y_test, y_pred_gnb)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_gnb, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.title('Matriz de Confusión - Naive Bayes (SIN Optimizar)')
plt.ylabel('Clase Verdadera')
plt.xlabel('Clase Predicha')
plt.show()

# Modelo SVM (SVC) con parámetros por defecto
print("\n--- SVM (Parámetros por defecto) ---")
svm_clf_unoptimized = SVC(random_state=42)
svm_clf_unoptimized.fit(X_train, y_train)
y_pred_svm_unoptimized = svm_clf_unoptimized.predict(X_test)
precision_svm_unoptimized = accuracy_score(y_test, y_pred_svm_unoptimized)
print(f"Precisión de SVM (sin optimizar): {precision_svm_unoptimized:.4f}")
print(f"Reporte de Clasificación:\n{classification_report(y_test, y_pred_svm_unoptimized, target_names=iris.target_names)}")

cm_svm_unoptimized = confusion_matrix(y_test, y_pred_svm_unoptimized)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_svm_unoptimized, annot=True, fmt='d', cmap='Reds', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.title('Matriz de Confusión - SVM (SIN Optimizar)')
plt.ylabel('Clase Verdadera')
plt.xlabel('Clase Predicha')
plt.show()
print("-" * 50)

# ==============================================================================
# 3. Optimización de Hiperparámetros con Grid Search
# ==============================================================================
print("3. Optimizando el modelo SVM con GridSearchCV...")

# Definición de la cuadrícula de hiperparámetros a probar
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.1, 0.01, 0.001],
    'kernel': ['rbf', 'linear']
}

# Configuración y ejecución de la búsqueda en malla
grid_search = GridSearchCV(SVC(random_state=42), param_grid, cv=5, verbose=1)
grid_search.fit(X_train, y_train)

# Impresión de los mejores resultados de la búsqueda
print(f"Mejores hiperparámetros encontrados: {grid_search.best_params_}")
print(f"Mejor precisión obtenida con validación cruzada: {grid_search.best_score_:.4f}")
print("-" * 50)

# ==============================================================================
# 4. Evaluación del Mejor Modelo SVM (Optimizado)
# ==============================================================================
print("4. Evaluación del mejor modelo SVM (Optimizado).")

# Obtener el mejor modelo
best_svm_clf = grid_search.best_estimator_

# Predicción y evaluación en el conjunto de prueba
y_pred_best_svm = best_svm_clf.predict(X_test)
precision_best_svm = accuracy_score(y_test, y_pred_best_svm)
print(f"Precisión del mejor modelo SVM (optimizado): {precision_best_svm:.4f}")
print(f"Reporte de Clasificación:\n{classification_report(y_test, y_pred_best_svm, target_names=iris.target_names)}")

# Matriz de Confusión para el SVM optimizado
cm_svm_optimized = confusion_matrix(y_test, y_pred_best_svm)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_svm_optimized, annot=True, fmt='d', cmap='Greens', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.title('Matriz de Confusión - SVM (OPTIMIZADO)')
plt.ylabel('Clase Verdadera')
plt.xlabel('Clase Predicha')
plt.show()

print("-" * 50)