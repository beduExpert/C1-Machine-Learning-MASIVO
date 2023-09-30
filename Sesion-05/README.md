[`Machine Learning`](../README.md) > `Sesión 05`

## Sesión 05: Árboles de decisión

<img src="https://github.com/beduExpert/Introduccion-a-Bases-de-Datos-Diciembre-2020/raw/master/imagenes/pizarron.png" align="right" height="100" width="100" hspace="10">
<div style="text-align: justify;">

### 1. Objetivos :dart: 
- Construir algoritmos que permitan separar datos por medio de clasificadores de árboles de decisión y random forests.

### 2. Contenido :blue_book:

---
#### <ins>Árboles de decisión</ins>
<img src="https://images.datacamp.com/image/upload/v1677504957/decision_tree_for_heart_attack_prevention_2140bd762d.png" align="right" height="300" width="400">

Un árbol de decisión es una representación gráfica de un proceso de toma de decisiones que se utiliza en inteligencia artificial y aprendizaje automático. Se compone de nodos que representan decisiones, condiciones o acciones, y de ramas que conectan estos nodos, mostrando las posibles secuencias de decisiones. Cada nodo interno representa una pregunta o una prueba sobre una característica de los datos, mientras que los nodos hoja representan las decisiones finales o resultados. Los árboles de decisión se utilizan para resolver problemas de clasificación y regresión al dividir los datos en subconjuntos basados en características, lo que permite tomar decisiones o predecir resultados. Son una herramienta poderosa para el análisis de datos y la toma de decisiones automatizada.

Un árbol de decisión funciona tomando decisiones basadas en características de los datos, y el índice de Gini es una medida que ayuda en este proceso. Aquí está cómo funciona:

1. **Selección de la característica:** El árbol de decisión comienza en la raíz con todos los datos y selecciona una característica (por ejemplo, una pregunta) que divide mejor los datos en subconjuntos más puros o homogéneos.

1. **Cálculo del índice de Gini:** Para evaluar la pureza de la división, se calcula el índice de Gini para la característica seleccionada. El índice de Gini mide la impureza de un conjunto de datos. Cuanto más bajo sea el índice de Gini, más puro o homogéneo es el conjunto de datos. Se calcula como 1 menos la suma de los cuadrados de las proporciones de cada clase en el conjunto de datos.

1. **División de los datos:** Se divide el conjunto de datos en función del valor de la característica seleccionada. Se crean ramas en el árbol, donde cada rama representa un posible valor o resultado de la característica.

1. **Repetición:** El proceso se repite en cada nodo interno del árbol, seleccionando la siguiente mejor característica que minimiza el índice de Gini para los subconjuntos resultantes.

1. **Nodos hoja:** Cuando se alcanza un criterio de parada, como una profundidad máxima o una pureza mínima, se detiene la subdivisión y los nodos finales se convierten en nodos hoja que representan decisiones o predicciones basadas en el valor mayoritario de la clase en ese subconjunto.

En resumen, un árbol de decisión utiliza el índice de Gini para medir la pureza de las divisiones de datos y selecciona las características que minimizan este índice en cada paso. Esto ayuda a construir un árbol que tome decisiones efectivas o haga predicciones basadas en las características de los datos.

- [**`EJEMPLO 1`**](Ejemplo01.ipynb)
- [**`RETO 1`**](Reto01.ipynb)

---
#### <ins>Random forests</ins>
<img src="https://serokell.io/files/vz/vz1f8191.Ensemble-of-decision-trees.png" align="right" height="250" width="400">

Un Random Forest es un modelo de aprendizaje automático que se basa en la combinación de múltiples árboles de decisión para mejorar la precisión y reducir el sobreajuste. Funciona de la siguiente manera:

1. **Construcción de múltiples árboles:** Se crea un conjunto (forest) de árboles de decisión, donde cada árbol se construye a partir de un subconjunto aleatorio de datos de entrenamiento y características (bootstrap y selección de características aleatorias).

1. **Votación o promedio:** Cuando se realiza una predicción, cada árbol en el Random Forest emite su propia predicción. En el caso de la clasificación, se lleva a cabo una votación para determinar la clase más frecuente entre los árboles. En la regresión, se promedian las predicciones de todos los árboles.

1. **Reducción del sobreajuste:** La diversidad de los árboles y la combinación de sus predicciones ayudan a reducir el sobreajuste (overfitting), lo que hace que el modelo sea más robusto y generalizable.

1. **Alta precisión:** Debido a la combinación de múltiples árboles, Random Forests tiende a ofrecer predicciones más precisas y estables en comparación con un solo árbol de decisión.

En resumen, un Random Forest es un modelo de ensamblaje que utiliza múltiples árboles de decisión para mejorar la precisión y la generalización de las predicciones, y es ampliamente utilizado en tareas de clasificación y regresión en aprendizaje automático.

- [**`EJEMPLO 2`**](Ejemplo02.ipynb)
- [**`RETO 2`**](Reto02.ipynb)

---


<br/>

[`Anterior`](../Sesion-04/README.md) | [`Siguiente`](../Sesion-06/README.md)      
