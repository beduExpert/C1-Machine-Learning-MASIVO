[`Machine Learning`](../README.md) > `Sesión 02`

## Sesión 02: Separación, validación y evaluación para algoritmos de ML

<img src="https://github.com/beduExpert/Introduccion-a-Bases-de-Datos-Diciembre-2020/raw/master/imagenes/pizarron.png" align="right" height="100" width="100" hspace="10">
<div style="text-align: justify;">

### 1. Objetivos :dart: 
- Construir algoritmos de separación de conjuntos de datos en entrenamiento / validación / prueba y algoritmos de evaluación de clasificadores de ML (clasificación binaria y multiclase).

### 2. Contenido :blue_book:

---
#### <ins>Separación de datos</ins>
<img src="https://assets-global.website-files.com/5d7b77b063a9066d83e1209c/627d12514852e122009eb71d_616b66004c27f02e81330769_data-training-needs-cover%2520(1).png" align="right" height="300" width="350">

La separación en entrenamiento, validación y prueba en Machine Learning implica dividir el conjunto de datos en tres partes:

1. **Entrenamiento:** Se utiliza para entrenar el modelo.
1. **Validación:** Se usa para ajustar hiperparámetros y evaluar el rendimiento del modelo durante el desarrollo.
1. **Prueba:** Se reserva para evaluar el rendimiento final del modelo después de que se haya ajustado y validado.

Esta separación ayuda a garantizar que el modelo funcione bien con datos nuevos y no vistos.

- [**`EJEMPLO 1`**](Ejemplo01.ipynb)
- [**`RETO 1`**](Reto01.ipynb)	

---
#### <ins>Validación cruzada</ins>
<img src="https://upload.wikimedia.org/wikipedia/commons/f/f2/K-fold_cross_validation.jpg" align="right" height="250" width="350">


La validación cruzada en Machine Learning es una técnica para evaluar el rendimiento de un modelo de manera más robusta. Consiste en dividir los datos en conjuntos de entrenamiento y prueba múltiples veces, permitiendo una evaluación más precisa de cómo el modelo generaliza a datos no vistos. Esto ayuda a evitar el sobreajuste y proporciona una medida más fiable del rendimiento del modelo.

- [**`EJEMPLO 2`**](Ejemplo02.ipynb)
- [**`RETO 2`**](Reto02.ipynb)

---

#### <ins>Leave-One-Out Cross Validation</ins>
<img src="https://miro.medium.com/v2/resize:fit:1400/0*v4goNAC_Ojb511a4.gif" align="right" height="250" width="350">



Leave-One-Out Cross-Validation (LOOCV) es una técnica de validación cruzada en la cual se entrena y evalúa un modelo de Machine Learning para cada punto de datos de forma individual, dejando uno fuera como conjunto de prueba. Es una validación exhaustiva pero costosa computacionalmente. Se utiliza para estimar el rendimiento del modelo y evaluar su capacidad de generalización.


- [**`EJEMPLO 3`**](Ejemplo03.ipynb)
- [**`RETO 3`**](Reto03.ipynb)

---

#### <ins>Matriz de confusión: Qué tan listo es tu algoritmo</ins>
<img src="https://miro.medium.com/v2/resize:fit:922/1*wEDbFRBl-je_ARYYai2_mg.gif" align="right" height="250" width="350">


Una matriz de confusión en Machine Learning es una tabla que muestra el desempeño de un modelo al comparar las predicciones con las etiquetas reales. Muestra cuántos casos se clasificaron correctamente (verdaderos positivos y verdaderos negativos) y cuántos se clasificaron incorrectamente (falsos positivos y falsos negativos), lo que ayuda a evaluar la precisión y el rendimiento del modelo en problemas de clasificación.

- [**`EJEMPLO 3`**](Ejemplo04.ipynb)
- [**`RETO 3`**](Reto04.ipynb)

---

#### <ins>Métricas de desempeño de Machine Learning</ins>
<img src="https://www.aprendemachinelearning.com/wp-content/uploads/2019/05/confusion_matix_example.png" align="right" height="400" width="350">

Las métricas de desempeño en Machine Learning son medidas que evalúan cuán bien un modelo se ajusta a los datos y hace predicciones precisas. Algunas métricas comunes incluyen:

1. **Precisión:** Mide la proporción de predicciones correctas sobre el total de predicciones.

1. **Recuperación o Sensibilidad:** Mide la proporción de verdaderos positivos respecto a todos los casos positivos reales.

1. **Especificidad:** Mide la proporción de verdaderos negativos respecto a todos los casos negativos reales.

1. **F1-score:** Combina precisión y recuperación en una sola métrica, útil cuando hay un desequilibrio en las clases.

1. **AUC-ROC:** Representa la capacidad del modelo para discriminar entre clases.

1. **Error cuadrático medio (MSE):** Mide el error promedio en problemas de regresión.

1. **Coeficiente de determinación (R²):** Evalúa cuánta variabilidad en los datos explica el modelo en problemas de regresión.

Estas métricas ayudan a cuantificar el rendimiento del modelo y a tomar decisiones sobre su uso o mejora.

- [**`EJEMPLO 5`**](Ejemplo04/README.md)
- [**`RETO 5`**](Reto04/README.md)

--- 

<br/>

[`Anterior`](../README.md) | [`Siguiente`](../Sesion02/README.md)      
