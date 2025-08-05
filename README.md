# PINNs - Physics Informed Neural Networks 

## Maestría en Inteligencia Artificial – 3er Bimestre 2025

Este proyecto aborda la **reconstrucción del campo de flujo en una cavidad cuadrada** utilizando *Redes Neuronales Informadas por Física* (PINNs), un enfoque moderno que combina ecuaciones diferenciales con redes neuronales profundas.

El problema consiste en modelar el comportamiento del fluido dentro de una cavidad con paredes fijas, excepto la superior que se mueve horizontalmente, generando un flujo característico. El objetivo es predecir los campos de **velocidad** y **presión** del fluido resolviendo las **ecuaciones de Navier-Stokes** de manera aproximada mediante redes neuronales.

<p align="center">
  <img src=image.png
  alt="Cavidad cuadrada" width="300"/>
</p>

> Imagen: Dominio del problema y condiciones de borde.  
> Flujo generado por movimiento de la tapa superior (“lid-driven cavity flow”).


Cada TP se encuentra documentado y ejecutado en su respectivo notebook.


### [TP N°1 – Puntos de colocación](TP1.ipynb)

* Definición de puntos de colocación para los residuos de PDE y condiciones de borde.
* Inclusión de puntos con datos rotulados.

---

### [TP N°2 – Modelado PINN](TP2.ipynb)

* Implementación completa de un modelo vanilla PINN.
* Entrenamiento sin datos rotulados y evaluación contra ground-truth.
* Búsqueda de hiperparametros usando Optuna.

---

### [TP N°3 – Estrategias de muestreo](TP3.ipynb)

* Comparación entre muestreo uniforme, LHS y RAD.
* Análisis de convergencia y rendimiento con distintos tamaños de dataset.

---

### [TP N°4 – Problema inverso](TP4.ipynb)

* Estimación del número de Reynolds a partir de datos parciales.
* Evaluación del impacto del ruido y de la región de muestreo.

---
## Autores
- Trinidad Monreal
- Jorge Ceferino Valdez
- Fabian Sarmiento
