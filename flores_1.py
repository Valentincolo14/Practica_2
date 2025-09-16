import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el dataset de Iris
iris = load_iris()
X = iris.data
y = iris.target

# Dividir el dataset en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("Tamaño del conjunto de entrenamiento:", X_train.shape)
print("Tamaño del conjunto de prueba:", X_test.shape)

print("--- Regresión Logística ---")
# Inicializar el modelo
log_reg = LogisticRegression(max_iter=200, solver='liblinear')

# Validación cruzada
scores_log_reg = cross_val_score(log_reg, X, y, cv=5, scoring='accuracy')
print(f"Precisión media de la validación cruzada: {scores_log_reg.mean():.4f}")

# Entrenar el modelo en el conjunto completo de entrenamiento y predecir
log_reg.fit(X_train, y_train)
y_pred_log_reg = log_reg.predict(X_test)
print("Precisión en el conjunto de prueba:", accuracy_score(y_test, y_pred_log_reg))
print("Reporte de clasificación:\n", classification_report(y_test, y_pred_log_reg))

# Matriz de Confusión
cm_log_reg = confusion_matrix(y_test, y_pred_log_reg)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_log_reg, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.title('Matriz de Confusión - Regresión Logística')
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.show()

print("--- Árbol de Decisión ---")
# Modelo inicial
dt_classifier = DecisionTreeClassifier(random_state=42)
scores_dt = cross_val_score(dt_classifier, X, y, cv=5, scoring='accuracy')
print(f"Precisión media de la validación cruzada (sin optimizar): {scores_dt.mean():.4f}")

# Optimización de hiperparámetros con GridSearchCV
param_grid_dt = {
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_leaf': [1, 2, 3, 5]
}

grid_search_dt = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid_dt, cv=5, scoring='accuracy')
grid_search_dt.fit(X_train, y_train)

best_dt = grid_search_dt.best_estimator_
print(f"Mejores hiperparámetros: {grid_search_dt.best_params_}")
print(f"Mejor precisión de validación: {grid_search_dt.best_score_:.4f}")

y_pred_dt = best_dt.predict(X_test)
print("Precisión en el conjunto de prueba:", accuracy_score(y_test, y_pred_dt))
print("Reporte de clasificación:\n", classification_report(y_test, y_pred_dt))

# Matriz de Confusión
cm_dt = confusion_matrix(y_test, y_pred_dt)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.title('Matriz de Confusión - Árbol de Decisión Optimizado')
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.show()

print("--- Bosques Aleatorios ---")
# Modelo inicial
rf_classifier = RandomForestClassifier(random_state=42)
scores_rf = cross_val_score(rf_classifier, X, y, cv=5, scoring='accuracy')
print(f"Precisión media de la validación cruzada (sin optimizar): {scores_rf.mean():.4f}")

# Optimización de hiperparámetros con GridSearchCV
param_grid_rf = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_leaf': [1, 2, 3]
}

grid_search_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=5, scoring='accuracy')
grid_search_rf.fit(X_train, y_train)

best_rf = grid_search_rf.best_estimator_
print(f"Mejores hiperparámetros: {grid_search_rf.best_params_}")
print(f"Mejor precisión de validación: {grid_search_rf.best_score_:.4f}")

y_pred_rf = best_rf.predict(X_test)
print("Precisión en el conjunto de prueba:", accuracy_score(y_test, y_pred_rf))
print("Reporte de clasificación:\n", classification_report(y_test, y_pred_rf))

# Matriz de Confusión
cm_rf = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.title('Matriz de Confusión - Bosques Aleatorios Optimizados')
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.show()