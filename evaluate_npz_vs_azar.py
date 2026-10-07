from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np

from benchmark_mancala_models import (
    NEXT_PIT,
    OPPOSITE_PIT,
    PIT_LABELS,
    PLAYER_1_PITS,
    PLAYER_2_PITS,
    azar_move,
    get_new_board,
)


ACTION_LABELS = tuple("ABCDEFGHIJKL")


class NPZModel:
    def __init__(self, path: Path) -> None:
        data = np.load(path, allow_pickle=True)
        self.w1 = data["w1"]
        self.b1 = data["b1"]
        self.w2 = data["w2"]
        self.b2 = data["b2"]
        self.w3 = data["w3"]
        self.b3 = data["b3"]

    def predict(self, vector: np.ndarray) -> np.ndarray:
        x = vector.astype(np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        z1 = x @ self.w1 + self.b1
        a1 = np.maximum(z1, 0.0)
        z2 = a1 @ self.w2 + self.b2
        a2 = np.maximum(z2, 0.0)
        return a2 @ self.w3 + self.b3


def legal_moves(player_turn: str, board: dict[str, int]) -> list[int]:
    pits = PLAYER_1_PITS if player_turn == "1" else PLAYER_2_PITS
    offset = 0 if player_turn == "1" else 6
    return [offset + idx for idx, pit in enumerate(pits) if board[pit] > 0]


def board_to_vector(board: dict[str, int], player_turn: str) -> np.ndarray:
    return np.array(
        [float(int(player_turn))] + [float(board[pit]) for pit in PIT_LABELS],
        dtype=np.float64,
    )


def model_move(model: NPZModel, player_turn: str, board: dict[str, int]) -> str:
    q_values = model.predict(board_to_vector(board, player_turn))[0]
    legal = legal_moves(player_turn, board)
    best_score = max(q_values[action] for action in legal)
    best_actions = [action for action in legal if q_values[action] == best_score]
    return ACTION_LABELS[random.choice(best_actions)]


def make_move(board: dict[str, int], player_turn: str, pit: str) -> str:
    seeds_to_sow = board[pit]
    board[pit] = 0

    while seeds_to_sow > 0:
        pit = NEXT_PIT[pit]
        if (player_turn == "1" and pit == "2") or (player_turn == "2" and pit == "1"):
            continue
        board[pit] += 1
        seeds_to_sow -= 1

    if (pit == player_turn == "1") or (pit == player_turn == "2"):
        return player_turn

    if player_turn == "1" and pit in PLAYER_1_PITS and board[pit] == 1:
        opposite_pit = OPPOSITE_PIT[pit]
        board["1"] += board[opposite_pit]
        board[opposite_pit] = 0
    elif player_turn == "2" and pit in PLAYER_2_PITS and board[pit] == 1:
        opposite_pit = OPPOSITE_PIT[pit]
        board["2"] += board[opposite_pit]
        board[opposite_pit] = 0

    return "2" if player_turn == "1" else "1"


def check_for_winner(board: dict[str, int]) -> str:
    player1_total = sum(board[pit] for pit in PLAYER_1_PITS)
    player2_total = sum(board[pit] for pit in PLAYER_2_PITS)

    if player1_total == 0:
        board["2"] += player2_total
        for pit in PLAYER_2_PITS:
            board[pit] = 0
    elif player2_total == 0:
        board["1"] += player1_total
        for pit in PLAYER_1_PITS:
            board[pit] = 0
    else:
        return "no winner"

    if board["1"] > board["2"]:
        return "1"
    if board["2"] > board["1"]:
        return "2"
    return "tie"


def evaluate(model_path: Path, games: int, seed: int) -> dict[str, int | float | str]:
    random.seed(seed)
    model = NPZModel(model_path)
    wins = losses = draws = 0

    for _ in range(games):
        board = get_new_board()
        ai_player = random.choice(["1", "2"])
        player_turn = random.choice(["1", "2"])

        while True:
            if player_turn == ai_player:
                move = model_move(model, player_turn, board)
            else:
                move = azar_move(player_turn, board, random)

            player_turn = make_move(board, player_turn, move)
            winner = check_for_winner(board)
            if winner != "no winner":
                if winner == "tie":
                    draws += 1
                elif winner == ai_player:
                    wins += 1
                else:
                    losses += 1
                break

    return {
        "model": model_path.name,
        "games": games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / games,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evalua un modelo NPZ DQN contra Azar.")
    parser.add_argument("models", nargs="+", type=Path)
    parser.add_argument("--games", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260610)
    args = parser.parse_args()

    for model in args.models:
        print(evaluate(model, games=args.games, seed=args.seed))


if __name__ == "__main__":
    main()
