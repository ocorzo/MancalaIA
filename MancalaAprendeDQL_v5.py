import multiprocessing
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import subprocess
import os
import MancalaIA_v0_5_Para_DQL as MancalaIA
from collections import deque

workDir = '/Volumes/OctubreRojo/Borrar/MancalaLog'
partidasParaBatch = 10
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
gama = 0.95
memoria = deque(maxlen=2000)
batch_size = 128
epocas = 10
target_player = 1

action_map = {
    "A": 0, "B": 1, "C": 2, "D": 3,
    "E": 4, "F": 5, "G": 6, "H": 7,
    "I": 8, "J": 9, "K": 10, "L": 11
}

# función para crear el modelo de la red neuronal
def create_model(input_shape, output_shape):
    model = Sequential()
    model.add(Dense(64, input_dim=input_shape, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(output_shape, activation='linear'))
    model.compile(loss='mse', optimizer='adam')
    return model

# función para procesar los archivos de logs
def process_files(workDir):
    os.chdir(workDir)
    subprocess.run("touch /Volumes/OctubreRojo/Borrar/temp.txt", shell=True)
    with open("/Volumes/OctubreRojo/Borrar/temp.txt", "w+") as output_file:
        subprocess.run("cat * | grep e", shell=True, stdout=output_file, text=True)
    df = pd.read_csv("/Volumes/OctubreRojo/Borrar/temp.txt", header=None)
    subprocess.run("rm *", shell=True)
    subprocess.run("rm /Volumes/OctubreRojo/Borrar/temp.txt", shell=True)
    df = df.iloc[:, :-1]
    label_encoder = LabelEncoder()
    df.iloc[:, -1] = label_encoder.fit_transform(df.iloc[:, -1])
    df.iloc[:, -1] = pd.to_numeric(df.iloc[:, -1], errors='coerce')
    df = df[(df.iloc[:, -1] >= 1) & (df.iloc[:, -1] <= 12)]
    return df

# función para entrenar el modelo
def train_model(model, df):
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    y = to_categorical(y, num_classes=12)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    model.fit(X_train, y_train, epochs=15, batch_size=10, verbose=0)
    #model.save('/Users/ocorzo/Codigo/Mancala/onLineModel_last.keras')
    return model

def replay(batch_size):
    global epsilon; memoria
    minibatch = random.sample(memoria, batch_size)
    for state, action, reward, next_state, done in minibatch:
        target = reward
        next_state = [target_player] + next_state
        next_state = np.array(next_state)
        next_state = next_state.reshape(1, -1)
        state = [target_player] + state
        state = np.array(state)
        state = state.reshape(1, -1)
        action = action_map[action]
        if not done:
            target = reward + gama * np.amax(onLineModel.predict(next_state, verbose=0)[0])
        target_f = onLineModel.predict(state, verbose=0)
        target_f[0][action] = target
        onLineModel.fit(state, target_f, epochs=1, verbose=1)
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

# función para correr los juegos en batch
def runBatch(Jugador_Inf, Jugador_sup, partidasParaBatch, epsilon):
    # Asegúrate de que todos los parámetros se pasan directamente y no se usan variables globales
    # Si necesitas cargar modelos o hacer inicializaciones, hazlo aquí
    MancalaIA.correEnBatch(Jugador_Inf, Jugador_sup, partidasParaBatch, epsilon)

def run_batches_parallel(partidasParaBatch, epsilon):
    partidasPorBatch = partidasParaBatch // 4
    with multiprocessing.Pool(processes=4) as pool:
        # Crear el comando con los argumentos variables
        comando = f"/home/ocorzo/Mancala/oca1/bin/python ~/Mancala/MancalaIA_v0_5_Para_DQL.py 'IA2' 'Azar' {partidasPorBatch} {epsilon}"
        pool.apply_async(subprocess.run, args=(comando,), kwds={'shell': True})
        comando = f"/home/ocorzo/Mancala/oca2/bin/python ~/Mancala/MancalaIA_v0_5_Para_DQL.py 'IA2' 'Azar' {partidasPorBatch} {epsilon}"
        pool.apply_async(subprocess.run, args=(comando,), kwds={'shell': True})
        comando = f"/home/ocorzo/Mancala/oca3/bin/python ~/Mancala/MancalaIA_v0_5_Para_DQL.py 'Azar' 'IA2' {partidasPorBatch} {epsilon}"
        pool.apply_async(subprocess.run, args=(comando,), kwds={'shell': True})
        comando = f"/home/ocorzo/Mancala/oca4/bin/python ~/Mancala/MancalaIA_v0_5_Para_DQL.py 'Azar' 'IA2' {partidasPorBatch} {epsilon}"
        pool.apply_async(subprocess.run, args=(comando,), kwds={'shell': True})
        pool.close()
        pool.join()

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
    filename = f"{workDir}/{filename}"  # Cambia por la ruta completa de tu archivo
    df = pd.read_csv(filename, header=None)

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
            memoria.append((state, action, reward, next_state, done))

    # Procesar la última línea del archivo
    last_row = df.iloc[-1]
    if last_row[0] == target_player and last_row.iloc[-1] == "winner":
        state = last_row[1:15].tolist()
        action = last_row[15]
        reward = 10
        next_state = [0] * 14  # No hay siguiente estado en la última jugada
        done = True  # Siempre es True en la última jugada
        memoria.append((state, action, reward, next_state, done))

    elif last_row[0] == target_player and last_row.iloc[-1] == "loser":
        state = last_row[1:15].tolist()
        action = last_row[15]
        reward = -10
        next_state = [0] * 14
        done = True # Siempre es True en la última jugada
        memoria.append((state, action, reward, next_state, done))

    else:
        state, action, reward, next_state, done = memoria[-1]
        reward = 10 if ganador else -10
        done = True
        memoria[-1] = (state, action, reward, next_state, done)

    return memoria

# Ejemplo de uso
'''filename = "/Volumes/OctubreRojo/Borrar/log_juego_Azar_IA2_86926400_2.txt"  # Cambia por el nombre de tu archivo
target_player = 2  # Cambia según el jugador que desees procesar (1 o 2)
memory = process_file_to_memory(filename, target_player)'''

def procesarArchivos(workDir, target_player):
    global memoria
    if target_player == 1:
        other_player = 2
    else:
        other_player = 1
    ganados = 0
    perdidos = 0
    empates = 0
    for archivo in os.listdir(workDir):
        if archivo.endswith(".txt"):
            if archivo.endswith(f"_{target_player}.txt"):
                ganados += 1
            elif archivo.endswith(f"_{other_player}.txt"):
                perdidos += 1
            else:
                empates += 1
            memoria = process_file_to_memory(archivo, target_player)

        os.remove(f"{workDir}/{archivo}")
    replay(batch_size)

    return ganados, perdidos, empates

if __name__ == "__main__":
    #modeloInicio = load_model('modelo_mancala_4000.keras')
    onLineModel = create_model(15, 12)
    #onLineModel.set_weights(modeloInicio.get_weights())
    onLineModel.save('/Users/ocorzo/Codigo/Mancala/onLineModel.keras')
    #targetModel = create_model(15, 12)
    #targetModel.save('/home/ocorzo/Mancala/targetModel.keras')

    for i in range(epocas):
        #epsilon = epsilonInicial*(1 - i / epocas)
        #run_batches_parallel(partidasParaBatch, epsilon)
        MancalaIA.correEnBatch('IA2', 'Azar', partidasParaBatch, epsilon)
        ganados, perdidos, empatados = procesarArchivos(workDir, target_player)
        #df = process_files(workDir)
        #targetModel.set_weights(onLineModel.get_weights())
        #onLineModel = train_model(onLineModel, df)
        onLineModel.save('/Users/ocorzo/Codigo/Mancala/onLineModel_last.keras')
        print("Modelo entrenado en la iteración: ", i, " , epsilon de: ", str(epsilon), " , ganados: ", ganados, " , perdidos: ", perdidos, " , empatados: ", empatados)
