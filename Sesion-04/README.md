[`Machine Learning`](../README.md) > `Sesión 04`

## Sesión 04: Problemas de regresión

<img src="https://github.com/beduExpert/Introduccion-a-Bases-de-Datos-Diciembre-2020/raw/master/imagenes/pizarron.png" align="right" height="100" width="100" hspace="10">
<div style="text-align: justify;">

### 1. Objetivos :dart: 
- Construir algoritmos que permitan predecir comportamientos, tanto en series de tiempo como con características, por medio de regresores lineales y polinomiales.

### 2. Contenido :blue_book:

---
#### <ins>Correlación y regresión lineal</ins>
<img src="https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F62413fa0-3d80-411c-af93-ebd0f096a26a_1042x644.png" align="right" height="300" width="400">

##### Correlación:

La correlación es una medida estadística que evalúa la relación entre dos variables. Indica si existe una relación y qué tipo de relación (positiva o negativa) existe entre ellas, pero no implica causalidad. Se expresa mediante el coeficiente de correlación, que varía de -1 a 1. Un valor cercano a 1 indica una correlación positiva perfecta, un valor cercano a -1 indica una correlación negativa perfecta, y un valor cercano a 0 indica una correlación débil o nula.

##### Regresión Lineal:

La regresión lineal es un modelo estadístico que busca establecer una relación lineal entre una variable dependiente y una o más variables independientes. El objetivo es encontrar la ecuación de una línea recta que mejor se ajuste a los datos, permitiendo hacer predicciones basadas en esa relación. La regresión lineal busca encontrar los coeficientes (pendiente e intercepto) que minimizan la diferencia entre las predicciones del modelo y los valores reales.

##### Evaluación del Rendimiento en Machine Learning:

Para evaluar el rendimiento de modelos de correlación o regresión lineal en Machine Learning, puedes utilizar varias métricas:

1. **Error Cuadrático Medio (MSE):** Mide el promedio de los cuadrados de las diferencias entre las predicciones del modelo y los valores reales. Un MSE más bajo indica un mejor rendimiento.

1. **R-cuadrado (R^2):** Indica la proporción de la variabilidad en los datos que es explicada por el modelo. Un valor cercano a 1 es deseable, ya que significa que el modelo se ajusta bien a los datos.

1. **Gráficos de Dispersión:** Visualizar un gráfico de dispersión de las predicciones frente a los valores reales puede ayudarte a evaluar visualmente qué tan cerca están las predicciones del modelo de los datos reales.

1. **Validación Cruzada:** Utilizar técnicas de validación cruzada, como la validación cruzada k-fold, puede proporcionar una evaluación más robusta del rendimiento del modelo, especialmente si se trata de un conjunto de datos grande.

1. **Residuos:** Analizar los residuos (diferencias entre los valores reales y las predicciones) puede ayudarte a identificar patrones o tendencias en los errores del modelo.

La elección de la métrica dependerá de los objetivos específicos del problema y de la naturaleza de los datos. En general, se busca minimizar el error y maximizar el R^2 para obtener un buen rendimiento en correlación o regresión lineal.

- [**`EJEMPLO 1`**](Ejemplo01.ipynb)
- [**`RETO 1`**](Reto01.ipynb)

---
#### <ins>Regresión polinomial</ins>
<img src="https://gbhat.com/assets/gifs/polynomial_regression.gif" align="right" height="250" width="400">

La regresión polinomial es una extensión de la regresión lineal que permite modelar relaciones no lineales entre variables. En lugar de ajustar una línea recta a los datos, la regresión polinomial utiliza una ecuación polinómica (una curva) para describir la relación entre la variable dependiente y la variable independiente o variables independientes.

En términos simples, la regresión polinomial se basa en la idea de que una relación entre variables puede no ser lineal, y por lo tanto, se utiliza un polinomio (una suma de términos con diferentes potencias) para representar mejor esa relación. Los modelos de regresión polinomial pueden ser cuadráticos (potencia 2), cúbicos (potencia 3), o de grados superiores, dependiendo de la complejidad necesaria para ajustarse a los datos.

En resumen, la regresión polinomial es una técnica que se utiliza cuando se sospecha que la relación entre las variables no es lineal y se busca un ajuste más flexible que una línea recta, utilizando una curva polinómica para representar mejor dicha relación.

- [**`EJEMPLO 2`**](Ejemplo02.ipynb)
- [**`RETO 2`**](Reto02.ipynb)

---

#### <ins>Regresión lineal y neuronas artificiales</ins>
![imagen](https://d2f0ora2gkri0g.cloudfront.net/dd/db/dddb807b-a15b-457d-a21a-8a9e6f029a3e.png)

Las neuronas artificiales, a menudo llamadas "perceptrones" en el contexto de Machine Learning, son los componentes básicos de las redes neuronales. La relación entre regresión lineal y neuronas artificiales radica en que una neurona artificial puede realizar una operación similar a una regresión lineal.

Una neurona artificial toma una combinación lineal de las entradas ponderadas y luego aplica una función de activación no lineal a esa combinación lineal. Esto se asemeja a la ecuación de una regresión lineal, donde se ponderan las variables de entrada y se suman para obtener un valor. Sin embargo, la función de activación no lineal introduce no linealidades en el modelo, lo que permite a las neuronas artificiales aprender relaciones más complejas que las que se pueden capturar con una regresión lineal pura.

![img](https://github.com/beduExpert/C1-Machine-Learning-2020/raw/master/Sesion-04/imgassets/timeseriesregressor1.png)

En resumen, la relación entre regresión lineal y neuronas artificiales es que una neurona artificial puede realizar operaciones similares a la regresión lineal, pero con la capacidad de introducir no linealidades gracias a su función de activación, lo que las hace fundamentales en la construcción de redes neuronales más complejas para tareas de Machine Learning. 

Hablaremos de esto más adelante.

---

#### <ins>Series de tiempo</ins>
<img src="https://mathdatasimplified.com/wp-content/uploads/2022/01/Peek-2022-01-04-08-58.gif" align="right" height="450" width="400">

Una serie de tiempo es una secuencia de datos observados o medidos en intervalos de tiempo regulares. Estos datos pueden incluir valores como el precio de las acciones, la temperatura diaria, la demanda de productos, entre otros. El análisis de series de tiempo implica comprender y modelar patrones y tendencias a lo largo del tiempo.

La relación entre series de tiempo y regresión lineal en Machine Learning se da cuando se utiliza la regresión lineal para modelar y predecir valores futuros en una serie de tiempo. En este caso, el tiempo se usa como una variable independiente, y la regresión lineal busca encontrar una relación lineal entre el tiempo y la variable objetivo.

Por ejemplo, si tenemos datos de ventas mensuales durante varios años, podemos utilizar la regresión lineal para predecir las ventas futuras en función del tiempo. Sin embargo, en muchas aplicaciones de series de tiempo, es necesario considerar otros factores, como patrones estacionales o tendencias no lineales, por lo que es común utilizar técnicas más avanzadas, como modelos autoregresivos (AR), modelos de media móvil (MA) o modelos de suavización exponencial. Estos modelos tienen en cuenta la dependencia temporal inherente a las series de tiempo y pueden superar las limitaciones de la regresión lineal en este contexto.


Para lograr esto debemos transformar nuestro dataset:

![imagen](https://github.com/beduExpert/C1-Machine-Learning-2020/raw/master/Sesion-04/imgassets/inputstimeseries.png)

- [**`EJEMPLO 3`**](Ejemplo03.ipynb)
- [**`RETO 3`**](Reto03.ipynb)


<br/>

[`Anterior`](../Sesion-03/README.md) | [`Siguiente`](../Sesion-05/README.md)      
