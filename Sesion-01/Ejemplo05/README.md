[`Machine Learning`](../../README.md) > [`Sesión 01`](../README.md) > `Ejemplo 5`

## Ejemplo 5: Descriptores

<div style="text-align: justify;">

### 1. Objetivos :dart:

- Conocer ejemplos de aplicación de los descriptores

### 2. Desarrollo :rocket:

La función de un descriptor es esa misma: *describir* un objeto por medio de números y propiedades, de tal manera que los algoritmos de ML sean capaces de tratarlos para diferenciar entre un objeto y otro. La manera en la que estos objetos son descritos es por medio de sensores.

En esta imagen podemos ver varios descriptores: Una onda de sonido representada numéricamente, un espacio de color, o un algoritmo llamado *Local Binary Pattern* (LBP)
![Descriptores](../imgassets/Descriptor1.png)

Veamos como funcionan de manera práctica con el siguiente ejemplo: Imaginemos que tienes un separador de frutas (como manzanas). Los objetos rojos son manzanas maduras, los objetos verdes pueden ser frutas inmaduras, y los objetos de otro color podrían ser cualquier otra cosa que no es una fruta.

![Lleve sus frutas](../imgassets/Fruits.png)

Y bien, ¿qué es mejor? ¿Un descriptor o un algoritmo? Sorpresivamente es mejor tener un gran descriptor que un gran algoritmo.
- Un gran descriptor puede darte un excelente resultado, aún con un algoritmo mediocre.
- Un descriptor malo te da mucha ambiguedad y malos resultados. No importando que tan inteligente sea tu algoritmo. 

####  Problemas con los descriptores.

No siempre los descriptores que elegimos son los mejores para la tarea. Tenemos que ser cuidadosos a la hora de elegirlos porque puede que no sirvan para ello. Por ejemplo: El encontrar objetos por color...

![Donde esta el panda?](../imgassets/Panda.jpg)

Igualmente, hay propiedades que no funcionan de la misma manera en todo el mundo. Por ejemplo, supongamos que tratas de clasificar a personas por su estatura, y si dices:

> *Asumamos que los hombres son mas altos que las mujeres*

Esto puede ser cierto en casi todos los países del mundo, excepto en Holanda, ¡Donde las mujeres suelen ser mas altas que los hombres!


[`Anterior`](../README.md) | [`Siguiente`](../README.md)

</div>
