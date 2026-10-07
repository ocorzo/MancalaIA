from __future__ import annotations

import random
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np

from benchmark_mancala_models import (
    NEXT_PIT,
    OPPOSITE_PIT,
    PLAYER_1_PITS,
    PLAYER_2_PITS,
    azar_move,
    codigo3_move,
    codigo4_move,
    codigo5_move,
    codigo_move as codigo2_move,
    get_new_board,
    load_model_spec,
    model_move,
)


ACTION_LABELS = tuple("ABCDEFGHIJKL")


@dataclass
class Agent:
    name: str
    kind: str
    payload: object | None = None


class NPZModel:
    def __init__(self, path: Path) -> None:
        data = np.load(path, allow_pickle=True)
        self.w1 = data["w1"]
        self.b1 = data["b1"]
        self.w2 = data["w2"]
        self.b2 = data["b2"]
        self.w3 = data["w3"]
        self.b3 = data["b3"]
        self.input_size = int(self.w1.shape[0])

    def predict(self, vector: np.ndarray) -> np.ndarray:
        x = vector.astype(np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        z1 = x @ self.w1 + self.b1
        a1 = np.maximum(z1, 0.0)
        z2 = a1 @ self.w2 + self.b2
        a2 = np.maximum(z2, 0.0)
        return a2 @ self.w3 + self.b3


def legal_moves(player_turn: str, board: dict[str, int]) -> list[str]:
    pits = PLAYER_1_PITS if player_turn == "1" else PLAYER_2_PITS
    return [pit for pit in pits if board[pit] > 0]


def board_to_vector(board: dict[str, int], player_turn: str) -> np.ndarray:
    ordered = np.array(
        [float(int(player_turn))] + [float(board[p]) for p in "ABCDEF1LKJIHG2"],
        dtype=np.float64,
    )
    return ordered


def npz_model_move(model: NPZModel, player_turn: str, board: dict[str, int]) -> str:
    scores = model.predict(board_to_vector(board, player_turn))[0]
    valid = set(legal_moves(player_turn, board))
    for idx in np.argsort(scores)[::-1]:
        move = ACTION_LABELS[int(idx)]
        if move in valid:
            return move
    raise RuntimeError("No se encontro jugada valida para el modelo NPZ")


def simulate_move(board: dict[str, int], player_turn: str, pit: str) -> int:
    seeds_to_sow = board[pit]
    board[pit] = 0
    player_mancala = "1" if player_turn == "1" else "2"

    while seeds_to_sow > 0:
        pit = NEXT_PIT[pit]
        if (player_turn == "1" and pit == "2") or (player_turn == "2" and pit == "1"):
            continue
        board[pit] += 1
        seeds_to_sow -= 1

    return board[player_mancala]


def find_best_opponent_move(board: dict[str, int], current_player: str) -> tuple[str | None, int]:
    opponent = "2" if current_player == "1" else "1"
    pits = PLAYER_1_PITS if opponent == "1" else PLAYER_2_PITS
    best_move = None
    max_mancala_seeds = -1

    for pit in pits:
        if board[pit] > 0:
            simulated_board = board.copy()
            mancala_seeds = simulate_move(simulated_board, opponent, pit)
            if mancala_seeds > max_mancala_seeds:
                max_mancala_seeds = mancala_seeds
                best_move = pit

    return best_move, max_mancala_seeds


def codigo_move(player_turn: str, board: dict[str, int]) -> str:
    best_move = legal_moves(player_turn, board)[0]
    max_net = -1
    pits = PLAYER_1_PITS if player_turn == "1" else PLAYER_2_PITS

    for pit in pits:
        if board[pit] > 0:
            simulated_board = board.copy()
            player_seeds = simulate_move(simulated_board, player_turn, pit)
            _, opponent_seeds = find_best_opponent_move(simulated_board, player_turn)
            net = player_seeds - opponent_seeds
            if net > max_net:
                max_net = net
                best_move = pit

    return best_move


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


def agent_move(agent: Agent, player_turn: str, board: dict[str, int], rng: random.Random) -> str:
    if agent.kind == "benchmark_model":
        return model_move(agent.payload, player_turn, board)
    if agent.kind == "npz_model":
        return npz_model_move(agent.payload, player_turn, board)
    if agent.kind == "codigo":
        return codigo2_move(player_turn, board)
    if agent.kind == "codigo3":
        return codigo3_move(player_turn, board)
    if agent.kind == "codigo4":
        return codigo4_move(player_turn, board)
    if agent.kind == "codigo5":
        return codigo5_move(player_turn, board)
    if agent.kind == "azar":
        return azar_move(player_turn, board, rng)
    raise ValueError(f"Tipo de agente no soportado: {agent.kind}")


def play_game(agent_p1: Agent, agent_p2: Agent, seed: int) -> tuple[str, int, int]:
    rng = random.Random(seed)
    board = get_new_board()
    player_turn = rng.choice(["1", "2"])

    while True:
        current = agent_p1 if player_turn == "1" else agent_p2
        move = agent_move(current, player_turn, board, rng)
        player_turn = make_move(board, player_turn, move)
        winner = check_for_winner(board)
        if winner != "no winner":
            return winner, board["1"], board["2"]


def build_agents() -> list[Agent]:
    return [
        Agent(
            name="DQN_best_30000_epsmin02",
            kind="npz_model",
            payload=NPZModel(Path("dqn_best_remote_30000_epsmin02.npz")),
        ),
        Agent(
            name="Codigo",
            kind="codigo",
        ),
        Agent(
            name="modelo_mancala_4000_mas_perdedoras",
            kind="benchmark_model",
            payload=load_model_spec(Path("modelo_mancala_4000_mas_perdedoras.h5")),
        ),
        Agent(
            name="Azar",
            kind="azar",
        ),
    ]


def main() -> None:
    games_per_pair = 1000
    rng = random.Random(20260527)
    agents = build_agents()
    table = {
        agent.name: {"points": 0, "matches": 0, "wins": 0, "draws": 0, "losses": 0}
        for agent in agents
    }
    pair_rows: list[dict[str, object]] = []

    for idx, (agent_a, agent_b) in enumerate(combinations(agents, 2), start=1):
        pair = {
            agent_a.name: {"points": 0, "wins": 0, "draws": 0, "losses": 0},
            agent_b.name: {"points": 0, "wins": 0, "draws": 0, "losses": 0},
        }
        for game_idx in range(games_per_pair):
            if rng.random() < 0.5:
                p1, p2 = agent_a, agent_b
            else:
                p1, p2 = agent_b, agent_a
            winner, _, _ = play_game(p1, p2, seed=idx * 100000 + game_idx)
            table[p1.name]["matches"] += 1
            table[p2.name]["matches"] += 1

            if winner == "1":
                table[p1.name]["points"] += 3
                table[p1.name]["wins"] += 1
                table[p2.name]["losses"] += 1
                pair[p1.name]["points"] += 3
                pair[p1.name]["wins"] += 1
                pair[p2.name]["losses"] += 1
            elif winner == "2":
                table[p2.name]["points"] += 3
                table[p2.name]["wins"] += 1
                table[p1.name]["losses"] += 1
                pair[p2.name]["points"] += 3
                pair[p2.name]["wins"] += 1
                pair[p1.name]["losses"] += 1
            else:
                table[p1.name]["points"] += 1
                table[p2.name]["points"] += 1
                table[p1.name]["draws"] += 1
                table[p2.name]["draws"] += 1
                pair[p1.name]["points"] += 1
                pair[p2.name]["points"] += 1
                pair[p1.name]["draws"] += 1
                pair[p2.name]["draws"] += 1

        pair_rows.append(
            {
                "label": f"{agent_a.name} vs {agent_b.name}",
                "a_pts": pair[agent_a.name]["points"],
                "b_pts": pair[agent_b.name]["points"],
                "a_w": pair[agent_a.name]["wins"],
                "b_w": pair[agent_b.name]["wins"],
                "d": pair[agent_a.name]["draws"],
            }
        )

    standings = []
    max_points = (len(agents) - 1) * games_per_pair * 3
    for name, row in table.items():
        standings.append(
            {
                "name": name,
                **row,
                "pct": row["points"] / max_points,
            }
        )

    standings.sort(key=lambda row: (-row["points"], -row["wins"], row["name"].lower()))

    print(f"Cuadrangular a {games_per_pair} partidas por enfrentamiento")
    print("\nResumen por enfrentamiento")
    pair_header = f"{'Cruce':60} {'Pts A':>5} {'Pts B':>5} {'G A':>4} {'G B':>4} {'Emp':>4}"
    print(pair_header)
    print("-" * len(pair_header))
    for row in pair_rows:
        print(
            f"{row['label'][:60]:60} {row['a_pts']:5d} {row['b_pts']:5d} "
            f"{row['a_w']:4d} {row['b_w']:4d} {row['d']:4d}"
        )

    print("\nTabla final")
    header = f"{'Equipo':34} {'Pts':>4} {'PJ':>4} {'G':>4} {'E':>4} {'P':>4} {'% max':>8}"
    print(header)
    print("-" * len(header))
    for row in standings:
        print(
            f"{row['name'][:34]:34} {row['points']:4d} {row['matches']:4d} {row['wins']:4d} "
            f"{row['draws']:4d} {row['losses']:4d} {row['pct']*100:7.2f}%"
        )


if __name__ == "__main__":
    main()
