# Mancala IA

Proyecto experimental en Python para estudiar cómo aprende una inteligencia artificial a jugar Mancala. Reúne modelos supervisados, aprendizaje por refuerzo con DQN y Double DQN, una implementación inspirada en AlphaZero y jugadores de referencia basados en azar y búsqueda exhaustiva.

El objetivo es aprender de la experiencia de juego y comparar los agentes bajo las mismas reglas. El trabajo evolucionó desde aprender jugadas de partidas ganadas hasta entrenar mediante self-play, conservar experiencias y evaluar frente a distintos oponentes.

**Resultado más reciente:** el nuevo AlphaZero ganó un hexagonal de 1,500 partidas con **1,098 de 1,500 puntos posibles (73.20%)**. Frente al AlphaZero anterior obtuvo 75 victorias y 25 empates en 100 partidas. Ambos utilizan la misma arquitectura y 50 simulaciones MCTS: sobre las mismas posiciones, sus tiempos medios de decisión fueron prácticamente iguales, **3.782 ms y 3.745 ms**, respectivamente.

## Reglas implementadas

La compatibilidad con los modelos históricos determina la variante utilizada:

- Dos jugadores, seis cuencos por jugador y una mancala por lado.
- Cuatro semillas por cuenco al inicio: 48 semillas en total.
- El jugador 1 utiliza `A` a `F`; el jugador 2, `G` a `L`.
- La siembra sigue `A → B → C → D → E → F → 1 → L → K → J → I → H → G → 2 → A` y omite la mancala rival.
- Terminar la siembra en la mancala propia concede otro movimiento.
- Si la última semilla cae en un cuenco propio previamente vacío, se capturan las semillas del cuenco opuesto. **La semilla recién depositada permanece en su cuenco**. Esta regla difiere de la captura habitual de Kalah.
- Cuando un lado queda vacío, las semillas restantes del otro lado pasan a su mancala y se comparan los totales.
- En los scripts actuales de DQN, AlphaZero y torneo también se declara victoria al alcanzar 25 semillas en una mancala, después de comprobar el cierre por lado vacío.

Los scripts históricos contienen variantes. Para comparar resultados es necesario verificar las reglas de la versión que produjo cada medición.

## Evolución del proyecto

### Aprendizaje supervisado y fase 2

Las primeras redes aprendieron a elegir movimientos a partir de registros de partidas entre jugadores aleatorios, jugadores `Codigo` y redes anteriores. Cada registro incluye jugador, tablero, movimiento y resultado.

Posteriormente se incorporaron jugadas perdedoras: se ajustaron las predicciones para reducir el peso del movimiento realizado y se reentrenó alternando ejemplos ganadores y perdedores. Se trata de entrenamiento supervisado con objetivos modificados a partir del resultado de la partida.

La red histórica `modelo_mancala_4000_mas_perdedoras.h5` tiene 15 entradas, dos capas ocultas de 30 neuronas y 12 salidas con softmax. Las entradas representan el jugador y las 14 posiciones del tablero; las salidas corresponden a `A` a `L`. Al jugar se descartan los movimientos ilegales.

El [reporte de fase 2](REPORTE_FASE_2_MANCALA.md) documenta la reconstrucción del entrenamiento y sus límites de trazabilidad: no se conserva una correspondencia completa entre cada experimento histórico y cada modelo.

### DQN y Double DQN

La fase de aprendizaje por refuerzo utiliza una red que estima el valor de las acciones y una red target para construir los objetivos de entrenamiento. Estas dos redes cumplen funciones distintas; no equivalen por sí mismas a dos jugadores independientes.

La arquitectura DQN habitual es `15 → 64 → 64 → 12`, con activaciones ReLU en las capas ocultas y salida lineal. El agente aprende mediante partidas y recompensas de resultado, sin recompensas adicionales por capturas o turnos extra.

Durante las iteraciones se incorporaron Double DQN, actualizaciones de la red target, replay buffer, exploración epsilon, rivales de distintas generaciones, checkpoints y evaluaciones periódicas. El código actual también admite retornos de varios pasos, memoria de referencia, pérdida Huber, recorte de gradientes y selección de modelos por distintos criterios.

Las corridas extensas se ejecutaron en un servidor Linux. Se guardaron mejores modelos y modelos finales por separado porque el rendimiento puede degradarse durante el entrenamiento. Cargar únicamente los pesos no recupera toda la experiencia previa: la continuación también puede utilizar replay buffers guardados.

### Fase 2.5

Se exploró reentrenar la red supervisada con partidas frente al DQN, reutilizando la metodología de ganadoras y perdedoras. Se probaron cambios en la intensidad y el orden de los ajustes. Los scripts de generación de logs, preparación de datos y reentrenamiento permanecen disponibles; no se presenta aquí una mejora cuantificada porque falta un reporte consolidado local de esas pruebas.

### AlphaZero y búsqueda

`alphazero_mancala.py` implementa un entrenamiento inspirado en AlphaZero: self-play, búsqueda Monte Carlo de árboles (MCTS) y una red con dos salidas, política y valor. La política aprende de las visitas de MCTS y el valor aprende del resultado final.

La configuración predeterminada usa 15 entradas, dos capas ocultas de 128 neuronas, 12 salidas de política y una salida de valor con tanh. Puede evaluarse la política directamente o combinar la red con MCTS. El torneo documentado utilizó 50 simulaciones por movimiento.

La primera corrida completó **10,000 partidas de self-play**, organizadas en 200 iteraciones de 50 partidas, con 50 simulaciones MCTS por movimiento y un replay buffer de hasta 300,000 posiciones. Se guardaron por separado el mejor modelo de política, el mejor modelo con MCTS y el modelo final.

Después se continuó desde el mejor MCTS anterior durante **otras 10,000 partidas**, manteniendo la arquitectura. La nueva corrida reconstruyó su replay buffer con experiencia nueva, porque el entrenamiento anterior no había guardado esa memoria. Se evaluó el modelo inicial y cada 1,000 partidas nuevas frente a Azar, Codigo3, Codigo4 y Codigo5, con 1,000 partidas por rival. La selección del mejor checkpoint se hizo por puntuación mixta, incluyendo el modelo inicial como candidato.

El mejor checkpoint de esta continuación apareció en la **iteración 80, tras 4,000 partidas nuevas**. Aunque la corrida completó las 10,000, ese checkpoint representa el mejor resultado intermedio. No debe describirse como un modelo que necesariamente acumuló 20,000 partidas: parte de un checkpoint de la corrida anterior que también pudo guardarse antes de su final.

| Referencia | Archivo en el servidor, relativo al directorio del proyecto |
| --- | --- |
| AlphaZero anterior | `alphazero_mancala_10000_best_mcts.npz` |
| **AlphaZero nuevo, campeón del hexagonal** | **`alphazero_continue_20261006/best_mcts.npz`** |
| Final de la continuación | `alphazero_continue_20261006/final.npz` |

Los dos mejores modelos de AlphaZero están incluidos en las rutas indicadas para reproducir el hexagonal y la medición de tiempos. El modelo final y los demás artefactos de entrenamiento permanecen en el servidor; los modelos `smoke` corresponden a pruebas breves distintas.

También se implementaron `Codigo`, `Codigo3`, `Codigo4` y `Codigo5`. La familia actual explora exhaustivamente hasta 2, 3, 4 y 5 movimientos, respectivamente, y elige la rama con mayor diferencia entre la mancala propia y la rival. Conserva el jugador inicial como referencia incluso en movimientos del oponente. Es una búsqueda optimista: supone que puede alcanzarse la mejor rama encontrada. Aumentar la profundidad no garantiza más victorias. Los turnos extra también cuentan como movimientos dentro de esa profundidad.

El `Codigo` histórico utilizaba otra evaluación de dos niveles; sus resultados antiguos no deben atribuirse automáticamente a la implementación actual.

## Resultados obtenidos

### Hexagonal más reciente

Todos contra todos entre los dos AlphaZero, Codigo3, Codigo4, Codigo5 y Azar. Se jugaron **100 partidas por cruce**, 15 cruces y 1,500 partidas en total. Cada participante disputó 500 partidas y podía obtener 1,500 puntos. La puntuación utiliza victoria = 3, empate = 1 y derrota = 0; la diferencia de semillas no interviene en la clasificación.

Para cada cruce se equilibraron las cuatro combinaciones de lado e inicio: 25 partidas por combinación. Ambos AlphaZero utilizaron 50 simulaciones MCTS por movimiento.

| Posición | Jugador | Puntos | Partidas | Ganadas | Empates | Perdidas | % del máximo |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | **AlphaZero nuevo** | **1,098** | 500 | 349 | 51 | 100 | **73.20%** |
| 2 | Codigo5 | 1,050 | 500 | 350 | 0 | 150 | 70.00% |
| 3 | Codigo3 | 922 | 500 | 299 | 25 | 176 | 61.47% |
| 4 | Codigo4 | 875 | 500 | 275 | 50 | 175 | 58.33% |
| 5 | AlphaZero anterior | 469 | 500 | 147 | 28 | 325 | 31.27% |
| 6 | Azar | 7 | 500 | 1 | 4 | 495 | 0.47% |

El nuevo AlphaZero superó por 48 puntos a Codigo5. En su cruce contra el anterior ganó 75 y empató 25, sin derrotas: 250 puntos frente a 25. Contra Codigo4 obtuvo también 75 victorias y 25 empates; frente a Codigo3 y Codigo5 ganó 50 y perdió 50 en cada cruce.

Fuentes: [reporte del hexagonal](HEXAGONAL_100.md) y [resultados completos por cruce](hexagonal_100_20261006.json).

### Comparación durante la continuación de AlphaZero

Evaluaciones de 1,000 partidas por rival, con lado e inicio equilibrados. Estos resultados corresponden al modelo inicial, al mejor checkpoint y al modelo final de la nueva corrida:

| Métrica | Anterior, evaluación inicial | Nuevo mejor checkpoint | Final de la continuación |
| --- | ---: | ---: | ---: |
| Win% contra Azar | 99.1% | **99.9%** | 99.6% |
| Win% contra Codigo3 | 25% | 50% | **75%** |
| Win% contra Codigo4 | 0% | **75%** | 0% |
| Win% contra Codigo5 | 25% | **50%** | **50%** |
| Puntuación mixta | 37.33% | **70.82%** | 60.33% |

La puntuación mixta es el promedio del porcentaje de puntos posibles frente a los cuatro rivales. El mejor checkpoint ganó 750 y empató 250 partidas frente a Codigo4. El final empató 500 y perdió 500 frente a ese rival. Más experiencia permitió encontrar un modelo mejor, pero el rendimiento no aumentó de manera continua.

La comparación inicial/nuevo produjo 4,479 y 8,498 puntos, respectivamente, sobre un máximo de 12,000 por modelo. Esta evaluación no incluye un enfrentamiento entre ambos AlphaZero ni cruces entre los jugadores Codigo; esos enfrentamientos sí están incluidos en el hexagonal anterior.

Fuentes: [historial](results/alphazero_continue_20261006/history.csv), [log](results/alphazero_continue_20261006/training.log) y [comando ejecutado](results/alphazero_continue_20261006/command.json). El comando conserva rutas del servidor como registro histórico; para reproducirlo en otra máquina deben adaptarse.

### Costo de cálculo por movimiento

Se midieron 320 partidas y, por separado, una muestra de **200 posiciones distintas**, resuelta tres veces por cada jugador: 600 decisiones por participante. La tabla siguiente corresponde a esa muestra común, con las bibliotecas numéricas limitadas a un hilo.

Los tiempos incluyen elegir el movimiento y, en AlphaZero, construir y ejecutar MCTS. Excluyen carga del modelo, actualización del tablero, registro y comunicación con el servidor.

| Jugador | Tiempo medio (ms) | Mediana (ms) | Percentil 95 (ms) |
| --- | ---: | ---: | ---: |
| Azar | 0.004 | 0.004 | 0.005 |
| Codigo3 | 0.398 | 0.405 | 0.735 |
| Codigo4 | 1.686 | 1.621 | 3.547 |
| AlphaZero anterior | 3.745 | 4.407 | 4.971 |
| AlphaZero nuevo | 3.782 | 4.474 | 5.132 |
| Codigo5 | 7.199 | 6.520 | 15.752 |

El nuevo AlphaZero mejoró su resultado competitivo con un costo por movimiento prácticamente igual al anterior. La diferencia observada de aproximadamente 1% no demuestra un cambio significativo de costo. Codigo5 tardó aproximadamente 1.9 veces lo que tardó AlphaZero nuevo en la muestra común.

Estos valores miden tiempo en una máquina y muestra concretas; no equivalen a energía consumida, número de nodos o fuerza estratégica. En Azar, la sobrecarga de medición representa una proporción relevante del tiempo registrado.

Fuentes: [protocolo de medición](DECISION_TIMES.md) y [datos completos](decision_times_20261006.json).

### Reconstrucción de fase 2

Evaluación de 1,000 partidas contra Azar por modelo, con aleatoriedad en lado e inicio, según el reporte local:

| Modelo | Ganadas | Empates | Perdidas | Win% |
| --- | ---: | ---: | ---: | ---: |
| `modelo_mancala.h5` | 810 | 33 | 157 | 81.00% |
| `modelo_mancala_4000_mas_perdedoras.h5` | 770 | 34 | 196 | 77.00% |
| `modelo_mancala_4000.h5` | 766 | 32 | 202 | 76.60% |
| `onLineModel_last.keras` | 481 | 43 | 476 | 48.10% |

En el torneo de fase 2, con seis participantes y 500 partidas por cruce, ganó `modelo_mancala_4000_mas_perdedoras.h5`: 6,435 puntos de 7,500 posibles (85.80%), con 2,138 victorias, 21 empates y 341 derrotas. El mejor modelo contra Azar y el ganador del torneo fueron diferentes.

### Evaluaciones posteriores de DQN

Estos resultados fueron registrados durante el seguimiento remoto, con 10,000 partidas contra Azar por modelo. Los logs completos de estas evaluaciones no están presentes en este directorio al preparar el README:

| Modelo | Win% contra Azar |
| --- | ---: |
| `dqn_v3_best_100000.npz` | 92.90% |
| `dqn_v4_best_150000.npz` | 93.99% |
| `dqn_v6_best_250000.npz` | 95.03% |
| `dqn_v7_best_350000.npz` | 94.86% |

También se realizaron experimentos v8 y v9. El servidor conserva sus modelos, memorias, checkpoints, historiales y gráficas. No se agregan porcentajes finales robustos de esas versiones sin recuperar primero sus evaluaciones correspondientes. La [gráfica conservada de v8](dqn_v8_winrate_350000.png) permite consultar su evolución interna.

### Torneo de búsqueda exhaustiva y AlphaZero

Torneo remoto completado con seis participantes y 1,000 partidas por cruce: 15,000 partidas en total, 5,000 por participante. Se aleatorizó la asignación de lados; el jugador 1 inicia cada partida. Puntuación: victoria = 3, empate = 1, derrota = 0. Cada participante podía obtener 15,000 puntos.

| Jugador | Puntos | Partidas | Ganadas | Empates | Perdidas | % del máximo |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Codigo3 | 9,593 | 5,000 | 3,030 | 503 | 1,467 | 63.95% |
| Codigo5 | 9,447 | 5,000 | 2,985 | 492 | 1,523 | 62.98% |
| Codigo4 | 9,440 | 5,000 | 2,979 | 503 | 1,518 | 62.93% |
| AlphaZero | 9,092 | 5,000 | 3,029 | 5 | 1,966 | 60.61% |
| Codigo | 6,366 | 5,000 | 1,957 | 495 | 2,548 | 42.44% |
| Azar | 58 | 5,000 | 16 | 10 | 4,974 | 0.39% |

El modelo de AlphaZero fue `alphazero_mancala_10000_best_mcts.npz`. Obtuvo 990 victorias, cinco empates y cinco derrotas contra Azar (99.00%), ganó las 1,000 partidas frente a Codigo y perdió las 1,000 frente a Codigo4. Estos cruces muestran que vencer a Azar no caracteriza por completo la fuerza frente a otros estilos.

Fuente: tabla del log remoto `tournament_codigos_alphazero_1000.log`, recuperada y registrada en la conversación. Este log no está incluido localmente. Con agentes deterministas, muchas partidas entre los mismos lados repiten la misma trayectoria; 1,000 partidas no necesariamente representan 1,000 situaciones distintas.

`Win% = ganadas / partidas × 100`. Los empates no cuentan como victorias. El porcentaje de puntos es `puntos / (3 × partidas) × 100`; es una métrica distinta. Los porcentajes anteriores pertenecen a protocolos y versiones diferentes y no constituyen una comparación controlada única.

## Archivos principales

| Archivo | Función |
| --- | --- |
| `benchmark_mancala_models.py` | Carga de modelos históricos, reglas y jugadores Codigo actuales. |
| `mancala_dqn_selfplay.py` | Entrenamiento DQN/Double DQN, memoria y evaluaciones. |
| `evaluate_npz_vs_azar.py` | Evaluación de modelos DQN `.npz` contra Azar. |
| `alphazero_mancala.py` | Red de política/valor, MCTS y self-play. |
| `benchmark_alphazero_mancala.py` | Evaluación de AlphaZero. |
| `mancala_mcts.py` | Búsqueda MCTS para agentes DQN. |
| `tournament_mancala.py` | Torneos entre modelos históricos, Codigo, AlphaZero y Azar. |
| `tournament_hexagonal.py` | Hexagonal equilibrado entre ambos AlphaZero y cuatro rivales. |
| `benchmark_decision_time.py` | Medición de tiempos en partidas y posiciones comunes. |
| `mancala_web.py` | Interfaz web para jugar contra AlphaZero. |
| `generate_phase25_logs.py`, `prepare_phase25_dataset.py`, `train_phase25_supervised.py` | Experimentos de fase 2.5. |
| `MancalaIA_v0*.py`, `EntrenaLaRedNeurunal*.py` | Implementaciones históricas y entrenamiento supervisado. |

Los archivos `.h5` y `.keras` contienen modelos Keras. Los `.npz` contienen pesos o memorias de los agentes NumPy, según el archivo. Los modelos `smoke` de AlphaZero corresponden a pruebas cortas y no al modelo remoto del torneo.

## Ejecución

Desde la raíz del proyecto, se recomienda un entorno Python 3.10 o posterior:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Los scripts modernos utilizan NumPy; `h5py` permite cargar los modelos históricos compatibles. Algunos scripts antiguos requieren además TensorFlow y PyMySQL y contienen rutas locales que deben adaptarse.

Ejemplo de corrida DQN corta para comprobar el flujo, no para reproducir los resultados históricos:

```bash
python mancala_dqn_selfplay.py --episodes 1000 --eval-every 250 --eval-games 1000 --replay-warmup-episodes 0 --epsilon 1.0 --epsilon-min 0.1
```

Evaluar el modelo obtenido:

```bash
python evaluate_npz_vs_azar.py dqn_best_model.npz dqn_final_model.npz --games 10000
```

Prueba breve del entrenamiento AlphaZero:

```bash
python alphazero_mancala.py --iterations 2 --self-play-games 10 --simulations 25 --eval-games 20
```

Continuar desde el modelo anterior durante 10,000 partidas nuevas, desde un directorio que contenga ese modelo. Usar una carpeta de salida nueva para no mezclar historiales de distintas corridas:

```bash
mkdir -p alphazero_continue_20261006
python alphazero_mancala.py \
  --load-model alphazero_mancala_10000_best_mcts.npz \
  --iterations 200 --self-play-games 50 \
  --simulations 50 --eval-simulations 50 --replay-size 300000 \
  --eval-every 20 --eval-games 1000 \
  --eval-opponents azar codigo3 codigo4 codigo5 --seed 20261006 \
  --checkpoint-every 20 \
  --checkpoint-dir alphazero_continue_20261006/checkpoints \
  --output-model alphazero_continue_20261006/final.npz \
  --best-policy-model alphazero_continue_20261006/best_policy.npz \
  --best-mcts-model alphazero_continue_20261006/best_mcts.npz \
  --history alphazero_continue_20261006/history.csv
```

La ejecución realizada en el servidor utilizó una copia del script llamada `alphazero_continue_10000.py`. Las opciones de continuación también están incorporadas al `alphazero_mancala.py` de este directorio.

Repetir el hexagonal y la medición de tiempos requiere los dos modelos de AlphaZero en las rutas indicadas en la tabla de modelos:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tournament_hexagonal.py \
  --games-per-pair 100 --output hexagonal_100.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmark_decision_time.py \
  --output decision_times.json
```

Repetir el torneo documentado requiere disponer del modelo remoto `alphazero_mancala_10000_best_mcts.npz` y de los cuatro modelos históricos que carga `build_agents()`, incluso al filtrar participantes:

```bash
python tournament_mancala.py --games-per-pair 1000 --agents Codigo Codigo3 Codigo4 Codigo5 AlphaZero_alphazero_mancala_10000_best_mcts Azar
```

El torneo busca primero ese modelo AlphaZero y, si no existe, utiliza `alphazero_mancala_smoke_best_mcts.npz` cuando está disponible, con otro nombre de participante. La selección explícita del comando anterior falla si falta el modelo esperado.

Para jugar en el navegador con un modelo generado localmente:

```bash
python mancala_web.py --host 127.0.0.1 --port 8088 --model alphazero_mancala_best_mcts.npz
```

Abrir `http://127.0.0.1:8088`. Los scripts de entrenamiento admiten `--help` para consultar parámetros y destinos de los artefactos.

## Reproducibilidad y publicación

Para futuras comparaciones, conservar junto a cada resultado: modelo, versión del código, comando completo, semilla, reglas, rivales, asignación de lados e inicio y número de partidas. En AlphaZero también se necesita registrar las simulaciones MCTS por movimiento. Un checkpoint seleccionado por score mixto puede diferir del mejor contra Azar.

En los cruces entre agentes deterministas se repiten trayectorias: 100 o 1,000 partidas no necesariamente representan esa cantidad de situaciones distintas. El hexagonal reciente equilibra cuatro combinaciones de lado e inicio; el torneo histórico de 1,000 partidas por cruce siempre iniciaba con el jugador 1. Por eso sus clasificaciones no deben tratarse como una comparación directa bajo un protocolo idéntico. Una siguiente evaluación puede usar posiciones iniciales variadas para medir generalización.

Los scripts históricos de MySQL utilizan las variables `MANCALA_DB_HOST`, `MANCALA_DB_NAME`, `MANCALA_DB_USER` y `MANCALA_DB_PASSWORD`; [.env.example](.env.example) documenta su configuración. Deben exportarse al entorno: el código no carga ese archivo automáticamente. Las credenciales incrustadas se retiraron para preparar la publicación; si las anteriores siguen vigentes, deben renovarse. Algunos scripts históricos aún requieren adaptar rutas personales.

`.gitignore` excluye entornos virtuales, cachés, memorias de entrenamiento grandes, configuración privada y material ajeno a Mancala. Las dependencias de los scripts modernos se encuentran en [requirements.txt](requirements.txt). TensorFlow y PyMySQL son dependencias adicionales para ejecutar los scripts históricos correspondientes.

La implementación histórica indica que parte del juego se tomó de un ejemplo de SourceCodester. Antes de distribuir el código, revisar la licencia y atribución de ese origen y definir la licencia del proyecto.
