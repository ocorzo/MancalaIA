""" En este archivo vamos a implementar una sistema de aprendizaje por refuerzo para el juego de Mancala.
Este sistema se basará en un algoritmo de aprendizaje profundo llamado Deep Q-Learning (DQL).
El objetivo es entrenar una red neuronal para que aprenda a jugar Mancala de forma autónoma.
Deberá tener un par de redes neuronales, una denominada red principal y otra red objetivo.
la red principal es una versión actualizada de la red objetivo y tiene la función de verificar que el modelo valla evolucionandose correctamente."""

# Importar las librerías necesarias
import numpy as np
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import subprocess
import os
import MancalaIA_v0_5_Para_DQL as MancalaIA

workDir = '/Users/ocorzo/Borrar/MancalaLog_1/'
partidasParaBatch = 200 # número de partidas para procear un entrenamiento

# función para crear el modelo de la red neuronal
def create_model(input_shape, output_shape):
    model = Sequential()
    model.add(Dense(128, input_dim=input_shape, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(output_shape, activation='linear'))
    model.compile(loss='mse', optimizer='adam')

    # Compilar el modelo
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

# función para procesar los archivos de logs
def process_files(workDir):

    # cargar las líneas ganadoras en un dataframe y borrar todos los archivos de log
    os.chdir(workDir)
    with open("/Users/ocorzo/Borrar/temp.txt", "w") as output_file:
        subprocess.run("cat * | grep winner", shell=True, stdout=output_file, text=True)

    # Cargar el archivo en un DataFrame
    df = pd.read_csv("/Users/ocorzo/Borrar/temp.txt", header=None)

    subprocess.run("rm *", shell=True)
    subprocess.run("rm /Users/ocorzo/Borrar/temp.txt", shell=True)

    # nos deshacemos de la última columna, la que solo dice "winner"
    df = df.iloc[:, :-1]

    # Convertir las etiquetas categóricas a valores numéricos
    label_encoder = LabelEncoder()
    df.iloc[:, -1] = label_encoder.fit_transform(df.iloc[:, -1])

    # Convertir la última columna a enteros
    df.iloc[:, -1] = pd.to_numeric(df.iloc[:, -1], errors='coerce')

    # Filtrar las filas donde la última columna esté en el rango 1 a 12
    df = df[(df.iloc[:, -1] >= 1) & (df.iloc[:, -1] <= 12)]

    return df

# función para entrenar el modelo
def train_model(model, df):

    # Separar las columnas en X (características) y y (etiquetas)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Convertir etiquetas a formato categórico (one-hot encoding)
    y = to_categorical(y, num_classes=12)

    # Separar los datos en entrenamiento (90%) y prueba (10%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # Entrenar el modelo con 30 iteraciones (epochs)
    model.fit(X_train, y_train, epochs=30, batch_size=10, verbose=1)

    # Guardar el modelo entrenado en un archivo
    model.save('/Users/ocorzo/Codigo/Mancala/onLineModel.h5')

    return model

# funcion para correr los juegos en batch
def runBatch(Jugador_Inf, Jugador_sup, partidasParaBatch):
    MancalaIA.Jugador_Sup = Jugador_sup
    MancalaIA.Jugador_Inf = Jugador_Inf
    MancalaIA.partidas = partidasParaBatch
    MancalaIA.correEnBatch()

onLineModel = create_model(15, 12)
onLineModel.save('/Users/ocorzo/Codigo/Mancala/onLineModel.h5')
targetModel = create_model(15, 12)
targetModel.save('/Users/ocorzo/Codigo/Mancala/targetModel.h5')

for i in range(100):
    runBatch("IA2", "IA3", partidasParaBatch/2)
    runBatch("IA3", "IA2", partidasParaBatch/2)
    df = process_files(workDir)
    targetModel.set_weights(onLineModel.get_weights())
    onLineModel = train_model(onLineModel, df)
    print("Modelo entrenado en la iteración: ", i)
