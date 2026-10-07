''' Notas de laversión:
V0, se agrega log de jugadas
Versión original tomada de https://www.sourcecodester.com/python/16790/simple-mancala-game-python-free-source-code.html#google_vignette'''

import sys, os, random

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

def main():
    print('''======== Mancala Game ========\n\n\n''')
    input('Press Enter to begin...')

    gameBoard = getNewBoard()
    playerTurn = '1'

    while True:

        print('\n')
        print("###################################################################")
        print('\n')

        displayBoard(gameBoard)
        playerMove = askForPlayerMove(playerTurn, gameBoard)

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

    print("""
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
