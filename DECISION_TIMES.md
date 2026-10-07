# Tiempo de decision de los jugadores

Medicion realizada en el servidor Linux del proyecto. Datos completos en
[decision_times_20261006.json](decision_times_20261006.json).

Se ejecutaron 320 partidas: 40 por cada cruce de AlphaZero anterior y nuevo
contra Azar, Codigo3, Codigo4 y Codigo5. Se alternaron lado e inicio en cuatro
combinaciones equilibradas. Ambos AlphaZero utilizaron 50 simulaciones MCTS.

Ademas, se seleccionaron 200 posiciones distintas de esas partidas y cada
jugador resolvio todas tres veces: 600 decisiones por jugador. Este segundo
escenario permite comparar costos sobre los mismos estados. Se aleatorizo
el orden de medicion y se limitaron las bibliotecas numericas a un hilo.

Se midio la seleccion de la jugada con perf_counter y process_time. Se excluyen
carga de modelos, siembra, registro y comunicacion SSH. Se incluye la creacion
del buscador MCTS. Los tiempos de Azar, del orden de microsegundos, incluyen
una proporcion relevante de sobrecarga del instrumento de medicion.

| Jugador | Media en partidas (ms) | Media en posiciones comunes (ms) | Mediana comun (ms) | P95 comun (ms) |
| --- | ---: | ---: | ---: | ---: |
| Azar | 0.00393 | 0.00396 | 0.00386 | 0.00474 |
| Codigo3 | 0.491 | 0.398 | 0.405 | 0.735 |
| Codigo4 | 1.671 | 1.686 | 1.621 | 3.547 |
| AlphaZero anterior | 4.000 | 3.745 | 4.407 | 4.971 |
| AlphaZero nuevo | 3.906 | 3.782 | 4.474 | 5.132 |
| Codigo5 | 9.355 | 7.199 | 6.520 | 15.752 |

Modelos:

- Anterior: alphazero_mancala_10000_best_mcts.npz.
- Nuevo: alphazero_continue_20261006/best_mcts.npz.

Sobre posiciones comunes, Codigo4 consume aproximadamente 4.24 veces el
tiempo de Codigo3 y Codigo5 4.27 veces el de Codigo4. AlphaZero nuevo consume
aproximadamente 1.01 veces el tiempo del anterior: la diferencia observada es
pequena y no demuestra un cambio de costo significativo. Codigo5 consume
aproximadamente 1.90 veces el tiempo de AlphaZero nuevo.

Estos tiempos son una medida empirica de costo computacional en esta maquina,
implementacion y muestra; no equivalen a numero de nodos, energia consumida o
capacidad estrategica. La muestra comun proviene de partidas de estos mismos
agentes. No es una muestra uniforme de todas las posiciones posibles.

Repetir en el servidor, desde la raiz del proyecto:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/ocorzo/anaconda3/envs/AI/bin/python benchmark_decision_time.py --output decision_times.json
```
