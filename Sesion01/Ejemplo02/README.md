[`Machine Learning`](../../README.md) > [`Sesión 01`](../README.md) > `Ejemplo 2`

## Ejemplo 2: Alien insolation

<div style="text-align: justify;">

### 1. Objetivos :dart:

- Entender cómo aprende una computadora por medio de un ejemplo práctico.

### 2. Desarrollo :rocket:

![On ta bebeeeee](../imgassets/AlienIsolation.jpg)

El juego de *Alien: Isolation* trata de escapar del clásico Alien de las películas. La inteligencia artificial del clásico monstruo espacial funciona con dos cerebros: Uno de los cerebros tiene el control del cuerpo del Alien, y otro cerebro tiene conocimiento de donde te encuentras en todo momento. Sin embargo, el cerebro que controla el movimiento no tiene idea de donde estás. 

¿Cómo le hace entonces el Alien para acecharte? Simple: El cerebro que sabe donde estás, le manda señales al cerebro que se mueve, en una especie de juego de *frio... frio... tibio... caliente... caliente...*

Cada vez que el cerebro del Alien recibe esas señales, lo que intenta compulsivamente es reducir la distancia entre tu y él, y ese indicador de frio-caliente es justamente *una derivada*.

![gradient1](../imgassets/gradientdescent1.gif)

Las matemáticas entonces no sirven <s>solamente</s> para torturar alumnos, sino que son una herramienta para traducir fenómenos del mundo real e insertarlos en las computadoras. Una derivada funciona entonces como esa función de frío - caliente, y los proyectos que fabricarás deberán intentar ajustarse para acercarse a un error cercano a 0. 

[`Anterior`](../README.md) | [`Siguiente`](../Ejemplo03/README.md)

</div>
