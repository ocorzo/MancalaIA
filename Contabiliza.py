# Codigo para contabilizar la salida de los tandes de Matatena

def count_results(file_name):
    player_1_wins = 0
    player_2_wins = 0
    ties = 0

    with open(file_name, 'r') as file:
        for line in file:
            line = line.strip()
            if "Player 1 has won" in line:
                player_1_wins += 1
            elif "Player 2 has won" in line:
                player_2_wins += 1
            elif "There is a tie" in line:
                ties += 1

    print(f"Player 1 wins: {player_1_wins}")
    print(f"Player 2 wins: {player_2_wins}")
    print(f"Ties: {ties}")

if __name__ == '__main__':
    count_results('/Users/ocorzo/Borrar/MancalaLog/MV4_I3_C.txt')
