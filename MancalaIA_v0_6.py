''' Notas de laversión:
V0_6 agregando interacción con la base de datos para IA4
V0_5 Se implementa finalización cuando alguno de los jugadores tenga 25 o mas semillas en su mancala y también
se implementa que se pueda jugar en tandas de "N" juegos
V0_3a Se implementa el nombre de los jugadores en el nombre del archivo log
V0_3 Se implementa IA2 entrenada con los resultados del triangular entre Azar Codigo e IA
V0_2 Se implementa IA en Code en la rutina CodigoMove y desactivación de selección de jugador para partidas en tandem comentando lína "Jugador_Sup, Jugador_Inf = selecciona_jugador()"
V0_1 se implementa selección de jugadores y rutina de juego al azar
V0, se agrega log de jugadas
Versión original tomada de https://www.sourcecodester.com/python/16790/simple-mancala-game-python-free-source-code.html#google_vignette'''

import sys, os, random
from tensorflow.keras.models import load_model
import numpy as np

PLAYER_1_PITS = ('A', 'B', 'C', 'D', 'E', 'F')
PLAYER_2_PITS = ('G', 'H', 'I', 'J', 'K', 'L')

OPPOSITE_PIT = {'A': 'G', 'B': 'H', 'C': 'I', 'D': 'J', 'E': 'K',
                   'F': 'L', 'G': 'A', 'H': 'B', 'I': 'C', 'J': 'D',
                   'K': 'E', 'L': 'F'}

NEXT_PIT = {'A': 'B', 'B': 'C', 'C': 'D', 'D': 'E', 'E': 'F', 'F': '1',
            '1': 'L', 'L': 'K', 'K': 'J', 'J': 'I', 'I': 'H', 'H': 'G',
            'G': '2', '2': 'A'}

PIT_LABELS = 'ABCDEF1LKJIHG2'

STARTING_NUMBER_OF_SEEDS = 4
log_file_name = "/Users/ocorzo/Borrar/MancalaLog/game_log.txt"
Jugador_Inf = "IA4" # Indica el tipo de juagador por lado, las opciones son: "Humano", "Azar", "Code", "IA", "IA2", "IA3" e "IA4"
Jugador_Sup = "Azar" # Indica el tipo de juagador por lado, las opciones son: "Humano", "Azar", "Code", "IA", "IA2", "IA3" e "IA4"

# Lista global para almacenar las jugadas
jugadas = []

# datos de conexión a la base de datos
host = os.environ["MANCALA_DB_HOST"]
database = os.environ["MANCALA_DB_NAME"]
username = os.environ["MANCALA_DB_USER"]
password = os.environ["MANCALA_DB_PASSWORD"]

import pymysql

def buscar_jugada(host, database, username, password, valor_pos):
    # Configurar la conexión a la base de datos
    conexion = pymysql.connect(host=host, user=username, password=password, database=database)

    cursor = conexion.cursor()

    # Consulta para buscar el valor específico en la tabla "Posicion"
    consulta = "SELECT * FROM Posicion WHERE Pos = %s"
    cursor.execute(consulta, (valor_pos,))

    # Obtener los resultados
    resultados = cursor.fetchall()

    # Si no se encuentran resultados, insertar un nuevo registro
    if not resultados:
        #print(f"No se encontró el valor '{valor_pos}'. Creando una nueva entrada.")
        insertar_nuevo(cursor, valor_pos)
        conexion.commit()
        # Ejecutar la consulta nuevamente para obtener el nuevo registro
        cursor.execute(consulta, (valor_pos,))
        resultados = cursor.fetchall()

    # Cerrar la conexión
    #cursor.close()
    #conexion.close()

    return resultados[0]

def insertar_nuevo(cursor, valor_pos):
    # Crear un nuevo registro con valores predeterminados
    consulta_insert = "INSERT INTO Posicion (Pos) VALUES (%s)"
    valores = (valor_pos)  # Cambia estos valores según sea necesario
    cursor.execute(consulta_insert, valores)

def buscar_jugada_old(host, database, username, password, posJugada):
    # Configurar la conexión a la base de datos
    conexion = pymysql.connect(host=host, user=username, password=password, database=database)

    cursor = conexion.cursor()

    # Consulta para buscar el valor específico en la tabla "Posicion"
    consulta = "SELECT * FROM Posicion WHERE Pos = %s"
    cursor.execute(consulta, (posJugada,))

    # Obtener los resultados
    resultados = cursor.fetchall()

    # Cerrar la conexión
    cursor.close()
    conexion.close()

    return resultados[0]

def reordenar_gameBoard(gameBoard, playerTurn):
    if playerTurn == '1':
        nuevo_orden = ['A', 'B', 'C', 'D', 'E', 'F', '1', 'L', 'K', 'J', 'I', 'H', 'G', '2']
    elif playerTurn == '2':
        #nuevo_orden = ['G', 'H', 'I', 'J', 'K', 'L', '2', 'A', 'B', 'C', 'D', 'E', 'F', '1']
        nuevo_orden = ['L', 'K', 'J', 'I', 'H', 'G', '2', 'A', 'B', 'C', 'D', 'E', 'F', '1']
    else:
        return gameBoard  # Si playerTurn no es '1' ni '2', no se reordena

    gameBoard_reordenado = {nuevo_orden[i]: gameBoard[nuevo_orden[i]] for i in range(14)}
    gameBoard_reordenado = ''.join(chr(64 + gameBoard_reordenado[clave]) for clave in gameBoard_reordenado)

    return gameBoard_reordenado

def almacenar_jugada(gameBoard, playerTurn, playerMove):
    # Modificar el orden de los valores en gameBoard dependiendo de playerTurn
    gameBoard = reordenar_gameBoard(gameBoard, playerTurn)

    # Convertir playerMove a un número entero
    mapa = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "L": 1, "K": 2, "J": 3, "I": 4, "H": 5, "G": 6}
    playerMove = mapa[playerMove]

    jugada = {
        'gameBoard': gameBoard,
        'playerTurn': playerTurn,
        'playerMove': playerMove
    }
    jugadas.append(jugada)
    print(jugada)

def agregar_puntos(jugadas, winner):
    for jugada in jugadas:
        if winner == "1":
            jugada["puntos"] = 3 if jugada["playerTurn"] == "1" else 0
        elif winner == "2":
            jugada["puntos"] = 3 if jugada["playerTurn"] == "2" else 0
        elif winner == "tie":
            jugada["puntos"] = 1
    return jugadas

def selecciona_jugador():
    Jugador_Sup = ""
    Jugador_Inf = ""
    sele_Sup = 0
    sele_Inf = 0
    print("Por favor selecciona el tipo de jugador para cada uno de los lados")

    while sele_Inf == 0:
        try:
            sele_Inf = int(input("Digite el número que representa al tipo de jugador INFERIOR: 1.- Humano, 2.- Azar, 3.- Code, 4.- IA ó 5.- IA2 "))  # Solicita el tipo de jugador para el lado de a bajor
            if sele_Inf not in [1, 2, 3, 4, 5]:
                print("Su selección debe de estar entre 1 y 5")
                sele_Inf = 0
        except ValueError:
            print("Por favor, digite un número válido.")

    if sele_Inf == 1: Jugador_Inf = "Humano"
    elif sele_Inf == 2: Jugador_Inf = "Azar"
    elif sele_Inf == 3: Jugador_Inf = "Code"
    elif sele_Inf == 4: Jugador_Inf = "IA"
    elif sele_Inf == 5: Jugador_Inf = "IA2"

    while sele_Sup == 0:
        try:
            sele_Sup = int(input("Digite el número que representa al tipo de jugador SUPERIOR: 1.- Humano, 2.- Azar, 3.- Code y 4.- IA "))  # Solicita el tipo de jugador para el lado de arriba
            if sele_Sup not in [1, 2, 3, 4, 5]:
                print("Su selección debe de estar entre 1 y 5")
                sele_Sup = 0
        except ValueError:
            print("Por favor, digite un número válido.")

    if sele_Sup == 1: Jugador_Sup = "Humano"
    elif sele_Sup == 2: Jugador_Sup = "Azar"
    elif sele_Sup == 3: Jugador_Sup = "Code"
    elif sele_Sup == 4: Jugador_Sup = "IA"
    elif sele_Inf == 5: Jugador_Inf = "IA2"

    return Jugador_Sup, Jugador_Inf

# Jugador_Sup, Jugador_Inf = selecciona_jugador()
hay_humano = True if Jugador_Sup == "Humano" or Jugador_Inf == "Humano" else False
if hay_humano: print('''======== Mancala Game ========\n\n\n''')

def main():
    global log_filename
    log_filename = crearArchivoLog()
    gameBoard = getNewBoard()
    playerTurn = '1'

    while True:
        if hay_humano:
            print('\n')
            print("###################################################################")
            print('\n')

        displayBoard(gameBoard)

        # Seleccionar la función de movimiento adecuada
        move_function = select_move_function(playerTurn, Jugador_Inf, Jugador_Sup)

        # Obtener el movimiento del jugador usando la función seleccionada
        playerMove = move_function(playerTurn, gameBoard)

        # Registrar el movimiento en el archivo de texto
        log_move(gameBoard, playerTurn, playerMove)

        almacenar_jugada(gameBoard, playerTurn, playerMove)

        playerTurn = makeMove(gameBoard, playerTurn, playerMove)

        winner = checkForWinner(gameBoard)
        if winner == '1' or winner == '2':
            displayBoard(gameBoard)
            print('Player ' + winner + ' has won!')
            update_log_with_winner(winner)
            renombrar_archivo(log_filename, winner)
            agregar_puntos(jugadas, winner)
            actualizar_jugadas(jugadas, host, database, username, password)
            break
        elif winner == 'tie':
            agregar_puntos(jugadas, winner)
            actualizar_jugadas(jugadas, host, database, username, password)
            displayBoard(gameBoard)
            print('There is a tie!')
            break
    return

def getNewBoard():
    """Return a dictionary representing a Mancala board in the starting
    state: 4 seeds in each pit and 0 in the stores."""

    s = STARTING_NUMBER_OF_SEEDS

    return {'1': 0, '2': 0, 'A': s, 'B': s, 'C': s, 'D': s, 'E': s,
            'F': s, 'G': s, 'H': s, 'I': s, 'J': s, 'K': s, 'L': s}

def displayBoard(board):
    """Displays the game board as ASCII-art based on the board
    dictionary."""

    seedAmounts = []

    for pit in 'GHIJKL21ABCDEF':
        numSeedsInThisPit = str(board[pit]).rjust(2)
        seedAmounts.append(numSeedsInThisPit)

    if hay_humano: print("""
+------+------+--<<<<<-Player 2----+------+------+------+
2      |G     |H     |I     |J     |K     |L     |      1
       |  {}  |  {}  |  {}  |  {}  |  {}  |  {}  |
S      |      |      |      |      |      |      |      S
T  {}  +------+------+------+------+------+------+  {}  T
O      |A     |B     |C     |D     |E     |F     |      O
R      |  {}  |  {}  |  {}  |  {}  |  {}  |  {}  |      R
E      |      |      |      |      |      |      |      E
+------+------+------+-Player 1->>>>>-----+------+------+

""".format(*seedAmounts))

def select_move_function(playerTurn, Jugador_Inf, Jugador_Sup):
    if playerTurn == '1':
        player_type = Jugador_Inf
    else:
        player_type = Jugador_Sup

    if player_type == 'Humano':
        return askForPlayerMove
    elif player_type == 'Azar':
        return AzarMove
    elif player_type == 'Code':
        return CodigoMove
    elif player_type == 'IA':
        return lambda playerTurn, board: IAmove(playerTurn, board, model)
    elif player_type == 'IA2':
        return lambda playerTurn, board: IA2move(playerTurn, board, model2)
    elif player_type == 'IA3':
        return lambda playerTurn, board: IA3move(playerTurn, board, model3)
    elif player_type == 'IA4':
        return IA4move
    else:
        raise ValueError("Tipo de jugador no reconocido")

def askForPlayerMove(playerTurn, board):
    """Asks the player which pit on their side of the board they
    select to sow seeds from. Returns the uppercase letter label of the
    selected pit as a string."""

    while True:

        if playerTurn == '1':
            print('Player 1, choose move: A-F (or QUIT)')
        elif playerTurn == '2':
            print('Player 2, choose move: G-L (or QUIT)')
        response = input('> ').upper().strip()

        if response == 'QUIT':
            print('Thanks for playing!')
            sys.exit()

        if (playerTurn == '1' and response not in PLAYER_1_PITS) or (
            playerTurn == '2' and response not in PLAYER_2_PITS
        ):
            print('Please pick a letter on your side of the board.')
            continue
        if board.get(response) == 0:
            print('Please pick a non-empty pit.')
            continue
        return response

# Selección del tiro de forma aleatoria
def AzarMove(playerTurn, board):
    if playerTurn == '1':
        pits = [pit for pit in PLAYER_1_PITS if board[pit] > 0]
    elif playerTurn == '2':
        pits = [pit for pit in PLAYER_2_PITS if board[pit] > 0]

    if pits:
        return random.choice(pits)
    else:
        return None  # Si no hay movimientos posibles

# Selección del tiro con una inteligencia en Code
def CodigoMove(playerTurn, board):
    best_move = AzarMove(playerTurn, board)
    max_mancala_seeds = -1

    if playerTurn == '1':
        player_pits = PLAYER_1_PITS
        player_mancala = '1'
        opponent_pits = PLAYER_2_PITS
        opponent_mancala = '2'
    else:
        player_pits = PLAYER_2_PITS
        player_mancala = '2'
        opponent_pits = PLAYER_1_PITS
        opponent_mancala = '1'

    for pit in player_pits:
        if board[pit] > 0:
            simulated_board = board.copy()
            player_seeds = simulate_move(simulated_board, playerTurn, pit)

            opponent_best_move, opponent_seeds = find_best_opponent_move(simulated_board, playerTurn)

            net_seeds = player_seeds - opponent_seeds
            if net_seeds > max_mancala_seeds:
                max_mancala_seeds = net_seeds
                best_move = pit

    return best_move

# Cargar el modelo entrenado para IA
def load_trained_model():
    model = load_model('/Users/ocorzo/Codigo/Mancala/modelo_mancala.h5')
    return model

# Cargar el modelo entrenado para IA2
def load_trained_model_IA2():
    model = load_model('/Users/ocorzo/Codigo/Mancala/modelo_mancala_9000.h5')
    return model

# Cargar el modelo entrenado para IA3
def load_trained_model_IA3():
    model3 = load_model('/Users/ocorzo/Codigo/Mancala/modelo_mancala_4000_mas_perdedoras.h5')
    return model3


hay_IA = True if Jugador_Sup == "IA" or Jugador_Inf == "IA" else False
# Cargar el modelo entrenado para IA
if hay_IA: model = load_trained_model()

hay_IA2 = True if Jugador_Sup == "IA2" or Jugador_Inf == "IA2" else False
# Cargar el modelo entrenado para IA2
if hay_IA2: model2 = load_trained_model_IA2()

hay_IA3 = True if Jugador_Sup == "IA3" or Jugador_Inf == "IA3" else False
# Cargar el modelo entrenado para IA3
if hay_IA3: model3 = load_trained_model_IA3()

import numpy as np
from tensorflow.keras.models import load_model

# Función para que la IA utilice el modelo y haga predicciones
def IAmove(playerTurn, board, model):
    # Asegurarnos de que todos los hoyos, mancalas y playerTurn están incluidos
    playerTurn_value = int(playerTurn)  # Convertir playerTurn a valor numérico 1 o 2
    X = [playerTurn_value] + [board[pit] for pit in PIT_LABELS]
    X = np.array([X])  # Convertir a matriz numpy y ajustar la dimensión

    # Predecir las probabilidades de cada posible movimiento
    y_prob = model.predict(X)[0]
    #print('Model_1')
    conversion_table = 'ABCDEFGHIJKL'

    # Ordenar los movimientos por probabilidad descendente
    y_sorted_indices = np.argsort(y_prob)[::-1]

    # Buscar el primer movimiento válido
    for idx in y_sorted_indices:
        move = conversion_table[idx]
        if (playerTurn == '1' and move in PLAYER_1_PITS and board[move] > 0) or (playerTurn == '2' and move in PLAYER_2_PITS and board[move] > 0):
            return move

    # Si no se encuentra un movimiento válido (no debería ocurrir), devolver None
    return None

# Función para que la IA2 utilice el modelo y haga predicciones
def IA2move(playerTurn, board, model2):
    # Asegurarnos de que todos los hoyos, mancalas y playerTurn están incluidos
    playerTurn_value = int(playerTurn)  # Convertir playerTurn a valor numérico 1 o 2
    X = [playerTurn_value] + [board[pit] for pit in PIT_LABELS]
    X = np.array([X])  # Convertir a matriz numpy y ajustar la dimensión

    # Predecir las probabilidades de cada posible movimiento
    y_prob = model2.predict(X)[0]
    #print('Model_2')
    conversion_table = 'ABCDEFGHIJKL'

    # Ordenar los movimientos por probabilidad descendente
    y_sorted_indices = np.argsort(y_prob)[::-1]

    # Buscar el primer movimiento válido
    for idx in y_sorted_indices:
        move = conversion_table[idx]
        if (playerTurn == '1' and move in PLAYER_1_PITS and board[move] > 0) or (playerTurn == '2' and move in PLAYER_2_PITS and board[move] > 0):
            return move

    # Si no se encuentra un movimiento válido (no debería ocurrir), devolver None
    return None

# Función para que la IA3 utilice el modelo y haga predicciones
def IA3move(playerTurn, board, model3):
    # Asegurarnos de que todos los hoyos, mancalas y playerTurn están incluidos
    playerTurn_value = int(playerTurn)  # Convertir playerTurn a valor numérico 1 o 2
    X = [playerTurn_value] + [board[pit] for pit in PIT_LABELS]
    X = np.array([X])  # Convertir a matriz numpy y ajustar la dimensión

    # Predecir las probabilidades de cada posible movimiento
    y_prob = model3.predict(X)[0]
    #print('Model_3')
    conversion_table = 'ABCDEFGHIJKL'

    # Ordenar los movimientos por probabilidad descendente
    y_sorted_indices = np.argsort(y_prob)[::-1]

    # Buscar el primer movimiento válido
    for idx in y_sorted_indices:
        move = conversion_table[idx]
        if (playerTurn == '1' and move in PLAYER_1_PITS and board[move] > 0) or (playerTurn == '2' and move in PLAYER_2_PITS and board[move] > 0):
            return move

    # Si no se encuentra un movimiento válido (no debería ocurrir), devolver None
    return None

def IA4move(playerTurn, board):
    Posicion = reordenar_gameBoard(board, playerTurn)
    _, a, b, c, d, e, f, _ = buscar_jugada(host, database, username, password, Posicion)
    pesos = [a, b, c, d, e, f]
    if playerTurn == '1':
        seleccion = random.choices(["A", "B", "C", "D", "E", "F"], weights=pesos, k=1)
    else:
        seleccion = random.choices(["L", "K", "J", "I", "H", "G"], weights=pesos, k=1)

    return seleccion[0]


def simulate_move(board, playerTurn, pit):
    seedsToSow = board[pit]
    board[pit] = 0
    player_mancala = '1' if playerTurn == '1' else '2'

    while seedsToSow > 0:
        pit = NEXT_PIT[pit]
        if (playerTurn == '1' and pit == '2') or (playerTurn == '2' and pit == '1'):
            continue
        board[pit] += 1
        seedsToSow -= 1

    return board[player_mancala]

def find_best_opponent_move(board, current_player):
    opponent = '2' if current_player == '1' else '1'
    best_move = None
    max_mancala_seeds = -1

    if opponent == '1':
        player_pits = PLAYER_1_PITS
        player_mancala = '1'
    else:
        player_pits = PLAYER_2_PITS
        player_mancala = '2'

    for pit in player_pits:
        if board[pit] > 0:
            simulated_board = board.copy()
            mancala_seeds = simulate_move(simulated_board, opponent, pit)
            if mancala_seeds > max_mancala_seeds:
                max_mancala_seeds = mancala_seeds
                best_move = pit

    return best_move, max_mancala_seeds

def makeMove(board, playerTurn, pit):
    """Modify the board data structure so that the player 1 or 2 in
    turn selected pit as their pit to sow seeds from. Returns either
    '1' or '2' for whose turn it is next."""

    seedsToSow = board[pit]
    board[pit] = 0

    while seedsToSow > 0:
        pit = NEXT_PIT[pit]
        if (playerTurn == '1' and pit == '2') or (
            playerTurn == '2' and pit == '1'
        ):
            continue
        board[pit] += 1
        seedsToSow -= 1

    if (pit == playerTurn == '1') or (pit == playerTurn == '2'):

        return playerTurn

    if playerTurn == '1' and pit in PLAYER_1_PITS and board[pit] == 1:
        oppositePit = OPPOSITE_PIT[pit]
        board['1'] += board[oppositePit]
        board[oppositePit] = 0
    elif playerTurn == '2' and pit in PLAYER_2_PITS and board[pit] == 1:
        oppositePit = OPPOSITE_PIT[pit]
        board['2'] += board[oppositePit]
        board[oppositePit] = 0

    if playerTurn == '1':
        return '2'
    elif playerTurn == '2':
        return '1'

def checkForWinner(board):

    player1Total = board['A'] + board['B'] + board['C']
    player1Total += board['D'] + board['E'] + board['F']
    player2Total = board['G'] + board['H'] + board['I']
    player2Total += board['J'] + board['K'] + board['L']

    if player1Total == 0:
        board['2'] += player2Total
        for pit in PLAYER_2_PITS:
            board[pit] = 0
    elif player2Total == 0:
        board['1'] += player1Total
        for pit in PLAYER_1_PITS:
            board[pit] = 0
    elif board['1'] >= 25:
        return '1'
    elif board['2'] >= 25:
        return '2'
    else:
        return 'no winner'

    if board['1'] > board['2']:
        return '1'
    elif board['2'] > board['1']:
        return '2'
    else:
        return 'tie'

# Directorio específico para guardar los archivos de log
log_directory = "/Users/ocorzo/Borrar/MancalaLog"
os.makedirs(log_directory, exist_ok=True) # NUEVO: Crea el directorio si no existe

# Inicializar el archivo de log
def crearArchivoLog():
    codigo_aleatorio = ''.join([str(random.randint(0, 9)) for _ in range(8)])
    log_filename = os.path.join(log_directory, f"log3_juego_{Jugador_Inf}_{Jugador_Sup}_{codigo_aleatorio}.txt")
    return log_filename

def log_move(board, player, pit):
    with open(log_filename, 'a+') as f:
        board_state = ','.join(str(board[p]) for p in PIT_LABELS)
        f.write(f"{player},{board_state},{pit}\n")

def update_log_with_winner(winner):
    with open(log_filename, 'r') as f:
        lines = f.readlines()

    with open(log_filename, 'w') as f:
        for line in lines:
            player = line.split(',')[0]
            if player == winner:
                f.write(line.strip() + ',winner\n')
            else:
                f.write(line.strip() + ',loser\n')
    f.close()

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
            #print(f"No se encontró el gameBoard '{gameBoard}'. Creando una nueva entrada.")
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

def renombrar_archivo(log_filename, ganador):
    # Separar el nombre del archivo y la extensión
    nombre, extension = os.path.splitext(log_filename)

    # Crear el nuevo nombre del archivo agregando el texto adicional
    nuevo_nombre = f"{nombre}_{ganador}{extension}"

    # Renombrar el archivo
    os.rename(log_filename, nuevo_nombre)

if __name__ == '__main__':
    for i in range(5):
        #pass
        main()
        jugadas = []
        pass

#for jugada in jugadas:
#    print(jugada)

#_ , a, b, c, d, e, f, _= buscar_jugada(host, database, username, password, "Prueba@333333@")
#print(a, b, c, d, e, f)
