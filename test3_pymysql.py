import os
import pymysql

def actualizar_jugadas(jugadas, host, database, username, password):
    # Configurar la conexión a la base de datos
    conexion = pymysql.connect(host=host, user=username, password=password, database=database)
    cursor = conexion.cursor()

    for jugada in jugadas:
        gameBoard = jugada['gameBoard']
        playerMove = jugada['playerMove']
        puntos = jugada['puntos']

        # Verificar si el gameBoard existe en la base de datos
        consulta_verificar = "SELECT * FROM Posicion WHERE pos = %s"
        cursor.execute(consulta_verificar, (gameBoard,))
        resultados = cursor.fetchall()

        # Si no existe, crear una nueva entrada
        if not resultados:
            print(f"No se encontró el gameBoard '{gameBoard}'. Creando una nueva entrada.")
            consulta_insertar = "INSERT INTO Posicion (pos, 1s, 2d, 3d, 4h, 5h, 6h, updates) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
            valores = (gameBoard, 1, 1, 1, 1, 1, 1, 0)  # Valores predeterminados
            cursor.execute(consulta_insertar, valores)
            conexion.commit()  # Confirmar la inserción

        # Determinar la columna a actualizar basada en playerMove
        columnas = {1: '1s', 2: '2d', 3: '3d', 4: '4h', 5: '5h', 6: '6h'}
        columna = columnas.get(playerMove)

        if columna:
            # Actualizar la columna específica con los puntos
            consulta_update = f"UPDATE Posicion SET {columna} = {columna} + %s, updates = updates + 1 WHERE pos = %s"
            cursor.execute(consulta_update, (puntos, gameBoard))

    # Confirmar los cambios y cerrar la conexión
    conexion.commit()
    cursor.close()
    conexion.close()

# Ejemplo de uso
jugadas = [
    {'gameBoard': 'DDDDDD@DDDDDD@', 'playerTurn': '1', 'playerMove': 1, 'puntos': 3},
    {'gameBoard': 'DDDDDD@@EEEED@', 'playerTurn': '2', 'playerMove': 1, 'puntos': 0},
    {'gameBoard': '@EEEED@@EEEED@', 'playerTurn': '1', 'playerMove': 2, 'puntos': 3},
    {'gameBoard': '@@FFFEA@EEEED@', 'playerTurn': '1', 'playerMove': 3, 'puntos': 3},
    {'gameBoard': 'AFEEED@@@@GGFB', 'playerTurn': '2', 'playerMove': 6, 'puntos': 0},
    {'gameBoard': 'AAAGGFBAFEEE@A', 'playerTurn': '1', 'playerMove': 1, 'puntos': 3},
    {'gameBoard': 'AFEEE@A@BAGGFB', 'playerTurn': '2', 'playerMove': 4, 'puntos': 0},
    {'gameBoard': 'ACAGGFBAFE@FAB', 'playerTurn': '1', 'playerMove': 4, 'puntos': 3}
]

host = os.environ["MANCALA_DB_HOST"]
database = os.environ["MANCALA_DB_NAME"]
username = os.environ["MANCALA_DB_USER"]
password = os.environ["MANCALA_DB_PASSWORD"]

# Llamar a la función para actualizar la base de datos
actualizar_jugadas(jugadas, host, database, username, password)
