''' Notas de laversión:
V0_2 Se implementa IA en código en la rutina CodigoMove y desactivación de selección de jugador para partidas en tandem comentando lína "Jugador_Sup, Jugador_Inf = selecciona_jugador()"
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
Jugador_Inf = "IA" # Indica el tipo de juagador por lado, las opciones son: "Humano", "Azar", "Código" e "IA"
Jugador_Sup = "IA" # Indica el tipo de juagador por lado, las opciones son: "Humano", "Azar", "Código" e "IA"

def selecciona_jugador():
    Jugador_Sup = ""
    Jugador_Inf = ""
    sele_Sup = 0
    sele_Inf = 0
    print("Por favor selecciona el tipo de jugador para cada uno de los lados")

    while sele_Inf == 0:
        try:
            sele_Inf = int(input("Digite el número que representa al tipo de jugador INFERIOR: 1.- Humano, 2.- Azar, 3.- Código y 4.- IA "))  # Solicita el tipo de jugador para el lado de a bajor
            if sele_Inf not in [1, 2, 3, 4]:
                print("Su selección debe de estar entre 1 y 4")
                sele_Inf = 0
        except ValueError:
            print("Por favor, digite un número válido.")

    if sele_Inf == 1: Jugador_Inf = "Humano"
    elif sele_Inf == 2: Jugador_Inf = "Azar"
    elif sele_Inf == 3: Jugador_Inf = "Código"
    elif sele_Inf == 4: Jugador_Inf = "IA"

    while sele_Sup == 0:
        try:
            sele_Sup = int(input("Digite el número que representa al tipo de jugador SUPERIOR: 1.- Humano, 2.- Azar, 3.- Código y 4.- IA "))  # Solicita el tipo de jugador para el lado de arriba
            if sele_Sup not in [1, 2, 3, 4]:
                print("Su selección debe de estar entre 1 y 4")
                sele_Sup = 0
        except ValueError:
            print("Por favor, digite un número válido.")

    if sele_Sup == 1: Jugador_Sup = "Humano"
    elif sele_Sup == 2: Jugador_Sup = "Azar"
    elif sele_Sup == 3: Jugador_Sup = "Código"
    elif sele_Sup == 4: Jugador_Sup = "IA"

    return Jugador_Sup, Jugador_Inf

# Jugador_Sup, Jugador_Inf = selecciona_jugador()
hay_humano = True if Jugador_Sup == "Humano" or Jugador_Inf == "Humano" else False
if hay_humano: print('''======== Mancala Game ========\n\n\n''')

def main():
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

        playerTurn = makeMove(gameBoard, playerTurn, playerMove)

        winner = checkForWinner(gameBoard)
        if winner == '1' or winner == '2':
            displayBoard(gameBoard)
            print('Player ' + winner + ' has won!')
            update_log_with_winner(winner)
            sys.exit()
        elif winner == 'tie':
            displayBoard(gameBoard)
            print('There is a tie!')
            sys.exit()

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
    elif player_type == 'Código':
        return CodigoMove
    elif player_type == 'IA':
        return lambda playerTurn, board: IAmove(playerTurn, board, model)
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

# Selección del tiro con una inteligencia en código
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

# Cargar el modelo entrenado
def load_trained_model():
    model = load_model('/Users/ocorzo/Codigo/Mancala/modelo_mancala.h5')
    return model

hay_IA = True if Jugador_Sup == "IA" or Jugador_Inf == "IA" else False
# Cargar el modelo entrenado para IA
if hay_IA: model = load_trained_model()

import numpy as np
from tensorflow.keras.models import load_model

# Cargar el modelo entrenado
def load_trained_model():
    model = load_model('modelo_mancala.h5')
    return model

# Función para que la IA utilice el modelo y haga predicciones
def IAmove(playerTurn, board, model):
    # Asegurarnos de que todos los hoyos, mancalas y playerTurn están incluidos
    playerTurn_value = int(playerTurn)  # Convertir playerTurn a valor numérico 1 o 2
    X = [playerTurn_value] + [board[pit] for pit in PIT_LABELS]
    X = np.array([X])  # Convertir a matriz numpy y ajustar la dimensión

    # Predecir las probabilidades de cada posible movimiento
    y_prob = model.predict(X)[0]
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
codigo_aleatorio = ''.join([str(random.randint(0, 9)) for _ in range(8)])
log_filename = os.path.join(log_directory, f"log_juego_{codigo_aleatorio}.txt")
#log_file = open(log_filename, "w+") # NUEVO: Abre el archivo de log para lectura y escritura, con un nombre único


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

if __name__ == '__main__':
    main()
