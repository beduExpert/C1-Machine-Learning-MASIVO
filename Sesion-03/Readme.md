# Sesión 03: Algoritmos no supervisados  

#### Objetivo: Construir algoritmos que permitan hacer separación en conjuntos sin necesidad de suministrar datos de categorización.  

¡Bienvenido a la tercera sesión de Machine Learning! En esta sesión aprenderemos a generar algoritmos no supervisados y cómo nos pueden ayudar a resolver problemas específicos. 

Lo primero que necesitamos es conocer un poco de los fundamentos de los algoritmos no supervisados, y una vez teniendo los fundamentos fijos, podemos intentar unir problemas que se nos presentan con estos algoritmos. 

## Fundamentos de algoritmos no supervisados

Para entender qué son los algoritmos **_no supervisados_** primero debemos de partir de su contraparte, los algoritmos **_supervisados_**: Un algoritmo supervisado significa que sabemos exactamente cómo queremos que se comporte nuestro sistema: Sabemos qué entradas le entregamos, y también sabemos qué salidas esperamos que tenga. 

Un ejemplo interesante de un algoritmo supervisado (que nada tiene que ver con machine learning) es _andar en bicicleta_: Cuando aprendes a andar en bicicleta, tienes una clara noción de lo que quieres lograr (en este caso, desplazarte con la bicicleta) y las caídas y raspones que te das intentándolo se vuelven parte del entrenamiento: poco a poco tratas de caerte menos, hasta que andar en bicicleta se vuelve parte de tu conocimiento. 

![wiiiii](imgassets/robotbike.gif)

En cambio, un **_algoritmo no supervisado_** es el equivalente de una tarea que requiere creatividad: Como en casi todas las actividades creativas, no hay una manera (objetiva) de que alguien te diga que la actividad que estás realizando está mal hecha. Un ejemplo de ello es pintar artísticamente: tienes una idea vaga de lo que deseas obtener, pero hasta que estás pintando empieza a tomar forma el resultado. 

![ulala señor frances](imgassets/painter.jpg)

En resumen:
- Si tu problema tiene entradas y sabes exactamente cuál es la salida que esperas, debes implementar un algoritmo supervisado.
- Si tu problema tiene entradas, pero no tienes idea de qué deseas obtener como resultado de salida, debes implementar un algoritmo no supervisado.

### ¿De qué me sirve un algoritmo no supervisado?

Un algoritmo de ML no supervisado te puede dar información sobre datos parecidos. Por ejemplo:

- Contenidos similares: Dependiendo de lo que busques en Facebook o Instagram, o de lo que pases mucho tiempo viendo, los algoritmos de ML no supervisados procurarán mostrarte más contenido de ese tipo.
- Problemas parecidos: Si tienes muchos casos de estudio parecidos y tienen información mapeable, puedes presentar un caso de estudio y el sistema debe traerte varios parecidos y con ellos, la solución. Sistemas expertos pueden nacer de esta noción.
- Clientes que se comportan similarmente: Si tienes dos clientes con patrones de consumo similares, eso quiere decir que lo que compre un cliente el otro probablemente también lo desee.
- Detección de anomalías: Si tienes un cluster de comportamiento “normal”, y una acción está muy lejos del centroide (mas allá de cierto umbral) es probable que sea una anomalía.
- Detección de fraudes bancarios: Un fraude bancario puede detectarse de forma relativamente fácil: un patrón inusual en una cuenta bancaria probablemente es un fraude: Por ejemplo, una transacción en un lugar extraño, una cantidad extraña, una hora de operación extraña…

![otros usuarios compraron](imgassets/usuallyboughttogether.png)

Sin embargo, no todo es maravilloso con los algoritmos no supervisados: Los resultados podrían no ser lo que esperabas, ya que el algoritmo decide cómo va a categorizar los datos. Las clases (o clústers) podrían no ser como tú las defines, o bien, cada que ejecutes el algoritmo de entrenamiento puede que obtengas resultados diferentes. 

![chale](imgassets/sad.jpg)

### Algoritmo no supervisado: K-Means

Para empezar a definir el algoritmo de k-means, tenemos que partir del concepto de _centroide_: Un centroide es una representación promedio de cada uno de los grupos que quieres obtener. Por ejemplo, si quieres dividir a tus clientes en tres tipos (digamos: tipo rojo, tipo verde y tipo azul), el centroide será la representación "ideal" de cada uno de los tipos.

Lo primero que haremos será colocar los centroides (que en este caso los representaremos como diamantes) en cualquier lugar, algo así: 
![listos...](imgassets/kmeans_start.gif)

A partir de este paso inicial, haremos los siguientes pasos: 

1. Tomaremos todos los puntos, y por cada punto, calculamos cuál es el centroide más cercano. Cada punto pertenece al grupo del centroide mas cercano. 
2. Cuando todos los puntos tienen un grupo asociado, separamos los puntos en grupos (grupo rojo, grupo azul y grupo verde).
3. Sacamos el promedio de todos los datos del grupo rojo y movemos el centroide al valor promedio del grupo rojo. 
4. Sacamos el promedio de todos los datos del grupo verde y movemos el centroide al valor promedio del grupo verde. 
5. Sacamos el promedio de todos los datos del grupo azul y movemos el centroide al valor promedio del grupo azul. 
6. Repetimos todos los pasos hasta que todos los centroides dejen de moverse (o se muevan muy poco). 

Este proceso se ve más o menos así:
![kmeans](imgassets/kmeans_process.gif)

Una vez que el algoritmo termine, todos los datos tendrán una clase o grupo asociado, lo que quiere decir que todos los datos del mismo grupo son similares. Si un nuevo dato llega, puedes asociarlo a alguno de los K centroides, y a diferencia de K-nearest neighbors, sólo tienes que calcular K distancias, en lugar de calcular la distancia contra todos los datos.

Es importante elegir adecuadamente la cantidad de grupos (K) que necesitas: Si eliges *K = 1*, todos los datos pertenecerán al mismo conjunto y no servirá de nada el algoritmo. Si eliges *K = Número de datos*, todos los datos tendrán su propio centroide y tampoco servirá de nada. 

#### Reto 1

Entrena un modelo de K-Medias con el dataset MallCustomer.csv, clasifica tus datos y separalos por grupos.

### Análisis de grupos utilizando gráficas de densidad

Después de haber separado nuestros datos por grupos, lo más importante es entender qué tipo de datos se encuentran en cada grupo. Nuestra tarea es intentar caracterizar cada grupo para poder entenderlo y aprovecharlo. Con gráficas de densidad podemos hacer comparaciones entre los grupos y de esta manera entender cómo es que son iguales y cómo es que son distintos.

Tambíén podemos utilizar gráficas 3D para observar visualmente cómo están segmentados nuestros datos de forma espacial.

#### Reto 2

Utiliza gráficas de densidad para analizar las diferencias y similitudes de los grupos obtenidos en el Reto 1. Elabora una gráfica 3D para visualizar tus datos segmentados espacialmente.

### Método de Codo

Por ahora sólo hemos utilizado nuestra intuición para elegir el número de grupos en los que segmenteraremos nuestros datos. Vamos a ver cómo podemos elegir un K analíticamente con ayuda del Método de Codo.

#### Reto 3

Aplica el método de codo, encuentra el K ideal para tu dataset y repite el Reto 1 y 2 con el nuevo K para observar cómo cambian tus conclusiones.