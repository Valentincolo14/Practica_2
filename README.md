Práctica 2 – Aprendizaje Automático con el Dataset Iris 

Este repositorio contiene la resolución de 4 retos aplicando diferentes algoritmos de aprendizaje automático sobre el dataset clásico Iris, ampliamente usado en clasificación y clustering.

El objetivo de esta práctica es comparar el rendimiento de distintos modelos, entender la fundamentación matemática detrás de cada técnica y analizar su capacidad de generalización.


Dataset: Iris

Número de muestras: 150

Número de características: 4 (longitud y ancho de sépalo y pétalo)

Número de clases: 3 (Setosa, Versicolor, Virginica)

Tarea: Clasificación supervisada (retos 1, 2, 4) y no supervisada + red neuronal (reto 3).


Reto 1 – Modelos Lineales, Árboles y Ensambles

Análisis

En este reto se compararon tres modelos supervisados:

- Regresión Logística (lineal, probabilístico).

- Árbol de Decisión (no lineal, basado en reglas jerárquicas).

- Random Forest (ensamble de árboles, reduce sobreajuste).

Se aplicó validación cruzada, optimización de hiperparámetros con GridSearchCV y evaluación mediante accuracy, matriz de confusión y classification report.

Fundamentación Matemática

- Regresión Logística:

<img width="329" height="126" alt="image" src="https://github.com/user-attachments/assets/081d44b6-6c8f-4517-ad9b-acdc88dd8594" />


Minimiza la función de pérdida log-loss.

- Árbol de Decisión: divide el espacio de atributos maximizando la ganancia de información o minimizando el índice de Gini.

- Random Forest: combina múltiples árboles con bagging y selección aleatoria de atributos para mejorar la generalización.


Conclusiones

- La regresión logística ofrece un buen rendimiento en datasets linealmente separables como Iris.

- Los árboles de decisión tienden a sobreajustar, pero con poda y optimización logran mayor estabilidad.

- El Random Forest superó en precisión a los modelos anteriores gracias a la diversidad de árboles.


🔹 Reto 2 – Naive Bayes y Support Vector Machines (SVM)

Análisis

Se probaron dos enfoques contrastantes:

- Naive Bayes (GaussianNB): simple y rápido, asume independencia condicional entre variables.

- SVM: busca un hiperplano óptimo de separación, con optimización de C, gamma y kernel vía GridSearch.

Fundamentación Matemática

Naive Bayes:


<img width="429" height="134" alt="image" src="https://github.com/user-attachments/assets/8004c08a-cf09-43f2-8ce7-acff49954ba8" />


Con 𝑥𝑖 modelado por una distribución Gaussiana.


- SVM: resuelve el problema de optimización:


<img width="423" height="71" alt="image" src="https://github.com/user-attachments/assets/2b8612b6-51cb-4b32-97c2-4898cb6b0caf" />

En la práctica, se extiende con kernel trick para separar datos no lineales.

Conclusiones

- Naive Bayes funciona sorprendentemente bien en Iris, pero sus supuestos limitan la capacidad predictiva.

- El SVM optimizado alcanzó una precisión superior, mostrando su poder en datasets con fronteras no lineales.

- El uso de GridSearchCV fue clave para encontrar hiperparámetros que maximizan el rendimiento.


🔹 Reto 3 – Clustering con K-Means y Redes Neuronales (MLP)
Análisis

Este reto combinó un método no supervisado (K-means) y un clasificador neuronal (MLP):

- Se aplicó el método del codo para determinar que 𝑘=3 es el número óptimo de clústeres.

- Se evaluó la correspondencia entre clústeres y clases reales, construyendo la matriz de confusión.

- Luego se entrenó un Perceptrón Multicapa (MLP) con datos escalados, usando 2 capas ocultas y validación cruzada.

Fundamentación Matemática

- K-means: minimiza la suma de distancias cuadradas dentro de los clústeres (WCSS):


<img width="325" height="104" alt="image" src="https://github.com/user-attachments/assets/01d8046a-affc-46d0-8f3d-8ff64fc897c2" />


donde 𝜇𝑖 es el centroide del clúster.

- MLP: aplica combinaciones lineales y funciones de activación (𝑅𝑒𝐿𝑈), entrenando con backpropagation:

<img width="278" height="85" alt="image" src="https://github.com/user-attachments/assets/b0d3aa5d-90f4-4363-9a42-0f2e3c371f1a" />

con actualización de pesos mediante el optimizador Adam.

Conclusiones

- K-means identificó correctamente los 3 grupos principales del dataset, aunque con cierta confusión entre Versicolor y Virginica.

- El MLP alcanzó alta precisión tras escalar los datos, confirmando la importancia del preprocesamiento en redes neuronales.

- La validación cruzada mostró un modelo consistente y con buena capacidad de generalización.


🔹 Reto 4 – Gradient Boosting y Support Vector Machine (Comparativa)

Análisis

Se implementaron y optimizaron dos modelos avanzados:

- Gradient Boosting Classifier (GBC): ensamble secuencial de árboles, optimizando n_estimators, learning_rate y max_depth.

- SVM: ajustado nuevamente, comparado con GBC.

Fundamentación Matemática

- Gradient Boosting: construye árboles secuenciales corrigiendo errores del anterior, minimizando una función de pérdida:


<img width="355" height="65" alt="image" src="https://github.com/user-attachments/assets/09cffada-8590-4125-bb36-2f138b054ebf" />

donde η es la learning rate y ℎ𝑚 es el nuevo árbol ajustado al residuo.

- SVM: mismo planteamiento matemático que en el reto 2, aquí afinado con grid search para una comparación justa.

Conclusiones

- Ambos modelos alcanzaron una alta precisión.

- En general, Gradient Boosting se destacó ligeramente al capturar relaciones más complejas entre atributos.

- SVM sigue siendo competitivo, especialmente con kernels adecuados.

- La comparación directa evidenció que los ensambles modernos (GBC, Random Forest) suelen superar a modelos individuales.


Conclusiones Generales

- El dataset Iris, aunque simple, permitió ilustrar claramente la diferencia entre modelos lineales, basados en reglas, ensambles, probabilísticos, de margen máximo y redes neuronales.

- Ensambladores (Random Forest, Gradient Boosting) mostraron mayor precisión y robustez.

- SVM y MLP confirmaron su potencial en problemas no lineales.

- Naive Bayes y Regresión Logística fueron rápidos y fáciles de entrenar, útiles como baseline.

- K-means demostró que incluso sin etiquetas, es posible encontrar estructuras naturales en los datos.
