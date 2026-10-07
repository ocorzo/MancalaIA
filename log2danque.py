import pandas as pd
from collections import deque

def process_file_to_memory(filename, target_player):
    """
    Procesar archivo y almacenar datos en formato deque (state, action, reward, next_state, done).

    Parámetros:
    - filename: Nombre del archivo que contiene las anotaciones.
    - target_player: Jugador objetivo (1 o 2) cuyos datos serán almacenados.

    Retorno:
    - memory: Memoria deque con la información procesada.
    """
    # Leer el archivo en un DataFrame
    df = pd.read_csv(filename, header=None)

    # Crear memoria deque
    memory = deque()

    # Iterar por las filas del DataFrame
    for i in range(len(df) - 1):  # Excluir la última línea (se procesará después)
        current_row = df.iloc[i]
        next_row = df.iloc[i + 1]

        # Procesar solo las filas del jugador objetivo
        if current_row[0] == target_player:
            # Estado actual
            state = current_row[1:15].tolist()
            # Acción
            action = current_row[15]
            # Reward
            reward = 0  # Recompensa predeterminada
            # Estado siguiente
            next_state = next_row[1:15].tolist()
            # ¿Es la última jugada del jugador objetivo?
            done = False
            #ganador = True if current_row[-1] == "winner" else False
            ganador = True if current_row.iloc[-1] == "winner" else False


            # Agregar la tupla a la memoria
            memory.append((state, action, reward, next_state, done))

    # Procesar la última línea del archivo
    last_row = df.iloc[-1]
    if last_row[0] == target_player and last_row.iloc[-1] == "winner":
        state = last_row[1:15].tolist()
        action = last_row[15]
        reward = 10
        next_state = [0] * 14  # No hay siguiente estado en la última jugada
        done = True  # Siempre es True en la última jugada
        memory.append((state, action, reward, next_state, done))

    elif last_row[0] == target_player and last_row.iloc[-1] == "loser":
        state = last_row[1:15].tolist()
        action = last_row[15]
        reward = -10
        next_state = [0] * 14
        done = True # Siempre es True en la última jugada
        memory.append((state, action, reward, next_state, done))

    else:
        state, action, reward, next_state, done = memory[-1]
        reward = 10 if ganador else -10
        done = True
        memory[-1] = (state, action, reward, next_state, done)

    return memory

# Ejemplo de uso
filename = "/Volumes/OctubreRojo/Borrar/log_juego_Azar_IA2_86926400_2.txt"  # Cambia por el nombre de tu archivo
target_player = 2  # Cambia según el jugador que desees procesar (1 o 2)
memory = process_file_to_memory(filename, target_player)

# Imprimir los primeros elementos de la memoria para ver el formato
for item in list(memory):  # Mostrar solo los primeros 5 elementos
    print(item)
