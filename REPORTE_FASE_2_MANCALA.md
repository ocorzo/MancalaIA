# Reporte Final Fase 2 - Mancala IA

## Objetivo

Documentar la "fase dos" del proyecto de Mancala IA para que pueda retomarse en el futuro o presentarse como cierre formal antes de pasar a una fase tres.

La fase dos corresponde a una etapa de entrenamiento **supervisado iterativo** usando:

- jugadas ganadoras,
- jugadas perdedoras,
- reentrenamientos sucesivos,
- y evaluación contra jugadores base como `Azar` y `Codigo`.

No fue todavia una fase de aprendizaje por refuerzo puro. Fue una evolucion del enfoque supervisado original.

## Estado reconstruido del proyecto

La reconstruccion se hizo usando:

- el codigo historico del repositorio,
- el archivo de notas [Mancala IA Segundo intento.pdf](/Users/ocorzo/Codigo/Mancala/Mancala%20IA%20Segundo%20intento.pdf),
- la inspeccion de modelos `.h5` y `.keras`,
- benchmarks directos contra `Azar`,
- y un torneo todos-contra-todos entre los agentes mas fuertes disponibles.

## Como se entreno la fase 2

De acuerdo con las notas del PDF, la idea central de la fase dos fue esta:

1. Generar experiencia de juego con 10,000 partidas completamente aleatorias.
2. Extraer jugadas ganadoras desde los logs.
3. Barajar las jugadas ganadoras.
4. Seleccionar un subconjunto para entrenamiento, por ejemplo 80,000 jugadas.
5. Entrenar un modelo supervisado con esas jugadas ganadoras.
6. Medir el rendimiento del modelo resultante.
7. Extraer jugadas perdedoras desde los logs.
8. Barajar las jugadas perdedoras.
9. Seleccionar un subconjunto, por ejemplo 30,000 jugadas.
10. Agregar predicciones del modelo sobre esas jugadas.
11. Ajustar esas predicciones para reforzar negativamente malas decisiones.
12. Reentrenar el modelo con ese conjunto ajustado.
13. Volver a medir.
14. Repetir el ciclo alternando positivos y negativos.

En otras palabras, la fase dos intento salir del techo del entrenamiento solo con jugadas ganadoras agregando un esquema de:

- aprendizaje con ejemplos buenos,
- castigo indirecto a ejemplos malos,
- y reentrenamiento por iteraciones.

## Scripts y piezas del pipeline

Las notas del PDF mencionan este flujo de herramientas:

- `AprendeMAncala.py`
- `MancalaJugador_aleatorio.py`
- `AgregaPredicciones.py`
- `PrepararRefuerzoNegativo_V4.py`
- `ReentrenarConNegativos.py`
- `PrepararRefuerzoPositivo_V4.py`

No todos esos archivos estan hoy identificados de forma exacta en el directorio actual con ese mismo nombre, pero el flujo de trabajo si quedo claro en las notas.

## Reglas del juego usadas en la fase historica

Para mantener compatibilidad con las IAs previas, se tomo como referencia la logica legacy del proyecto:

- 4 semillas iniciales por hoyo.
- Siembra saltando la mancala rival.
- Turno extra si la ultima semilla cae en la mancala propia.
- Cierre cuando un lado se queda sin semillas.
- En versiones posteriores tambien existio una variante que termina si una mancala llega a 25 o mas semillas.
- La captura legacy del proyecto no coincide exactamente con la regla clasica de Kalah:
  - cuando la ultima semilla cae en un hoyo vacio propio, se suman a la mancala solo las semillas del hoyo opuesto;
  - la semilla que acaba de caer no se mueve a la mancala.

Ese detalle es importante porque cambia el comportamiento del juego y afecta cualquier comparacion entre modelos.

## Arquitectura de los modelos relevantes

Los modelos mas importantes inspeccionados en el repositorio fueron:

- `modelo_mancala.h5`
  - entrada: 15
  - ocultas: 30 y 30
  - salida: 12
  - activacion final: `softmax`

- `modelo_mancala_4000_mas_perdedoras.h5`
  - entrada: 15
  - ocultas: 30 y 30
  - salida: 12
  - activacion final: `softmax`

- `modelo_mancala_4000.h5`
  - entrada: 15
  - ocultas: 128 y 128
  - salida: 12
  - activacion final: `linear`

- `onLineModel_last.keras`
  - entrada: 15
  - ocultas: 64 y 64
  - salida: 12
  - activacion final: `linear`

Observacion:

- los modelos `30-30-softmax` se parecen a la familia supervisada tradicional;
- los modelos `64-64-linear` y `128-128-linear` parecen venir de experimentos posteriores mas cercanos a DQL.

## Modelo ganador de la fase 2

Tomando como criterio principal el torneo entre agentes disponibles del repositorio, el **modelo ganador** fue:

- `modelo_mancala_4000_mas_perdedoras.h5`

Este nombre sugiere, ademas, que forma parte directa del esquema de entrenamiento con **perdedoras**, lo cual encaja bien con la narrativa del PDF del "segundo intento".

## Importante sobre la trazabilidad historica

Hay una limitacion que conviene dejar documentada:

- no se pudo probar de forma absoluta que `modelo_mancala_4000_mas_perdedoras.h5` corresponda exactamente a una version concreta como "V8" o "V10" del PDF;
- pero por nombre, arquitectura y comportamiento, es el mejor candidato sobreviviente en el repositorio para representar la parte madura de la fase dos basada en ganadoras y perdedoras.

Por eso, este archivo debe tratarse como:

- **modelo ganador reconstruido de la fase dos**,
- no necesariamente como una identidad historica perfecta al 100%.

## Resultados contra Azar

Se construyo un benchmark reproducible para correr los modelos del directorio contra `Azar`.

Archivo usado:

- [benchmark_mancala_models.py](/Users/ocorzo/Codigo/Mancala/benchmark_mancala_models.py)

En una medicion de 1000 partidas por modelo contra `Azar`, con aleatoriedad tanto en el jugador inicial como en el lado del tablero, los cuatro modelos mas fuertes quedaron asi:

| Modelo | Ganadas | Perdidas | Empates | Win rate |
| --- | ---: | ---: | ---: | ---: |
| `modelo_mancala.h5` | 810 | 157 | 33 | 81.00% |
| `modelo_mancala_4000_mas_perdedoras.h5` | 770 | 196 | 34 | 77.00% |
| `modelo_mancala_4000.h5` | 766 | 202 | 32 | 76.60% |
| `onLineModel_last.keras` | 481 | 476 | 43 | 48.10% |

Lectura:

- el mejor rendimiento puro contra `Azar` lo dio `modelo_mancala.h5`;
- sin embargo, el modelo que mejor se comporto en el torneo global entre agentes fuertes fue `modelo_mancala_4000_mas_perdedoras.h5`.

Esto sugiere que:

- `modelo_mancala.h5` fue especialmente eficaz contra `Azar`,
- mientras que `modelo_mancala_4000_mas_perdedoras.h5` fue mas robusto en un ecosistema mixto con rivales mas fuertes.

## Torneo final de la fase 2

Se organizo un torneo todos-contra-todos entre:

- `modelo_mancala_4000_mas_perdedoras.h5`
- `modelo_mancala.h5`
- `modelo_mancala_4000.h5`
- `onLineModel_last.keras`
- `Codigo`
- `Azar`

Formato:

- 500 partidas por enfrentamiento
- puntuacion tipo futbol
  - victoria = 3 puntos
  - empate = 1 punto
  - derrota = 0 puntos

Archivo usado:

- [tournament_mancala.py](/Users/ocorzo/Codigo/Mancala/tournament_mancala.py)

Cada equipo jugo 2500 partidas en total, asi que el maximo posible era:

- `2500 x 3 = 7500 puntos`

Tabla final limpia:

| Equipo | Pts | PJ | G | E | P | % del maximo |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `modelo_mancala_4000_mas_perdedoras.h5` | 6435 | 2500 | 2138 | 21 | 341 | 85.80% |
| `Codigo` | 4325 | 2500 | 1437 | 14 | 1049 | 57.67% |
| `modelo_mancala.h5` | 4261 | 2500 | 1415 | 16 | 1069 | 56.81% |
| `modelo_mancala_4000.h5` | 4111 | 2500 | 1363 | 22 | 1115 | 54.81% |
| `Azar` | 1772 | 2500 | 560 | 92 | 1848 | 23.63% |
| `onLineModel_last.keras` | 1504 | 2500 | 495 | 19 | 1986 | 20.05% |

## Conclusiones de la fase 2

1. La fase dos si logro superar claramente el techo de la fase uno.
2. El esquema de alternar ganadoras y perdedoras fue util para mejorar el nivel competitivo de al menos una familia de modelos.
3. El modelo mejor posicionado al cierre reconstruido de la fase dos es `modelo_mancala_4000_mas_perdedoras.h5`.
4. En torneo global, ese modelo obtuvo `85.80%` del maximo de puntos posible.
5. Contra `Azar`, el mejor porcentaje directo observado en la reconstruccion fue `81.00%` con `modelo_mancala.h5`.
6. El repositorio no conserva trazabilidad perfecta entre cada version anotada en las notas y cada archivo final de modelo, por lo que parte de esta reconstruccion debe tratarse como inferencia informada.

## Recomendacion para la fase 3

La fase tres deberia tomar como baseline formal:

- `modelo_mancala_4000_mas_perdedoras.h5` como campeon reconstruido de la fase dos,
- `modelo_mancala.h5` como mejor modelo directo contra `Azar`,
- `Codigo` como baseline heuristico fuerte,
- `Azar` como baseline debil.

Esto permite comparar cualquier nuevo agente de aprendizaje por refuerzo contra referencias historicas reales del proyecto.

## Artefactos utiles ya preparados

- [benchmark_mancala_models.py](/Users/ocorzo/Codigo/Mancala/benchmark_mancala_models.py)
  - benchmark de modelos del directorio contra `Azar`

- [tournament_mancala.py](/Users/ocorzo/Codigo/Mancala/tournament_mancala.py)
  - torneo todos-contra-todos entre agentes seleccionados

- [mancala_rl.py](/Users/ocorzo/Codigo/Mancala/mancala_rl.py)
  - base nueva para la fase tres de aprendizaje por refuerzo

## Nota final

Este reporte cierra razonablemente la fase dos con la informacion hoy disponible en el repositorio. Si en el futuro aparecen:

- notebooks,
- scripts faltantes,
- logs originales de entrenamiento,
- o modelos adicionales,

conviene actualizar este documento para mejorar la trazabilidad historica.
