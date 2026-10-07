# Hexagonal: AlphaZero anterior y nuevo

Torneo ejecutado en el servidor: 100 partidas por cruce, 15 cruces,
1,500 partidas en total y 500 por participante. Victoria: 3 puntos;
empate: 1; derrota: 0. Maximo por participante: 1,500 puntos.

Cada cruce incluyo 25 partidas por combinacion de lado e inicio:
jugador A como jugador 1 o 2, con jugador 1 o 2 iniciando. Ambos AlphaZero
utilizaron 50 simulaciones MCTS. Semilla: 20261006. Reglas legacy con cierre
al alcanzar 25 semillas o vaciar un lado. Los empates de puntos en la
clasificacion se mantienen sin desempate por diferencia de semillas.

| Jugador | Puntos | Partidas | Ganadas | Empates | Perdidas | % del maximo |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AlphaZero nuevo | 1,098 | 500 | 349 | 51 | 100 | 73.20% |
| Codigo5 | 1,050 | 500 | 350 | 0 | 150 | 70.00% |
| Codigo3 | 922 | 500 | 299 | 25 | 176 | 61.47% |
| Codigo4 | 875 | 500 | 275 | 50 | 175 | 58.33% |
| AlphaZero anterior | 469 | 500 | 147 | 28 | 325 | 31.27% |
| Azar | 7 | 500 | 1 | 4 | 495 | 0.47% |

Cruce directo: AlphaZero nuevo gano 75 partidas y empato 25 contra el anterior.
Obtuvieron 250 y 25 puntos, respectivamente.

Modelo anterior: alphazero_mancala_10000_best_mcts.npz.
Modelo nuevo: alphazero_continue_20261006/best_mcts.npz.

Los agentes deterministas repiten trayectorias bajo las mismas condiciones.
La cantidad de partidas no equivale a la misma cantidad de situaciones
distintas. Este torneo equilibra los cuatro roles; el torneo historico de
1,000 partidas por cruce aleatorizaba lados pero siempre iniciaba el jugador 1.

Datos completos: [hexagonal_100_20261006.json](hexagonal_100_20261006.json).
Script: [tournament_hexagonal.py](tournament_hexagonal.py).
