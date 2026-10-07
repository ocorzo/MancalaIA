from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
import random

from alphazero_mancala import AlphaZeroMCTS, AlphaZeroNetwork
from benchmark_mancala_models import (
    OPPOSITE_PIT,
    PIT_LABELS,
    PLAYER_1_PITS,
    PLAYER_2_PITS,
    NEXT_PIT,
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
ALPHAZERO_CANDIDATE_MODELS = (
    Path("alphazero_mancala_10000_best_mcts.npz"),
    Path("alphazero_mancala_smoke_best_mcts.npz"),
)
ALPHAZERO_MCTS_SIMULATIONS = 50


@dataclass
class Agent:
    name: str
    kind: str
    model: object | None = None


def legal_moves(player_turn: str, board: dict[str, int]) -> list[str]:
    pits = PLAYER_1_PITS if player_turn == "1" else PLAYER_2_PITS
    return [pit for pit in pits if board[pit] > 0]


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
    best_move = None
    max_mancala_seeds = -1

    player_pits = PLAYER_1_PITS if opponent == "1" else PLAYER_2_PITS
    for pit in player_pits:
        if board[pit] > 0:
            simulated_board = board.copy()
            mancala_seeds = simulate_move(simulated_board, opponent, pit)
            if mancala_seeds > max_mancala_seeds:
                max_mancala_seeds = mancala_seeds
                best_move = pit
    return best_move, max_mancala_seeds


def codigo_move(player_turn: str, board: dict[str, int]) -> str:
    best_move = legal_moves(player_turn, board)[0]
    max_mancala_seeds = -1
    player_pits = PLAYER_1_PITS if player_turn == "1" else PLAYER_2_PITS

    for pit in player_pits:
        if board[pit] > 0:
            simulated_board = board.copy()
            player_seeds = simulate_move(simulated_board, player_turn, pit)
            _, opponent_seeds = find_best_opponent_move(simulated_board, player_turn)
            net_seeds = player_seeds - opponent_seeds
            if net_seeds > max_mancala_seeds:
                max_mancala_seeds = net_seeds
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
    elif board["1"] >= 25:
        return "1"
    elif board["2"] >= 25:
        return "2"
    else:
        return "no winner"

    if board["1"] > board["2"]:
        return "1"
    if board["2"] > board["1"]:
        return "2"
    return "tie"


def agent_move(agent: Agent, player_turn: str, board: dict[str, int], seed: int) -> str:
    if agent.kind == "model":
        return model_move(agent.model, player_turn, board)
    if agent.kind == "codigo":
        return codigo2_move(player_turn, board)
    if agent.kind == "codigo3":
        return codigo3_move(player_turn, board)
    if agent.kind == "codigo4":
        return codigo4_move(player_turn, board)
    if agent.kind == "codigo5":
        return codigo5_move(player_turn, board)
    if agent.kind == "alphazero_mcts":
        mcts = AlphaZeroMCTS(
            network=agent.model,
            simulations=ALPHAZERO_MCTS_SIMULATIONS,
            rng=random.Random(seed),
        )
        action = mcts.search(board, player_turn, temperature=0.0, sample=False).action
        return ACTION_LABELS[action]
    if agent.kind == "azar":
        return azar_move(player_turn, board, random.Random(seed))
    raise ValueError(f"Tipo de agente no soportado: {agent.kind}")


def play_match(agent_p1: Agent, agent_p2: Agent, seed: int) -> tuple[str, int, int]:
    board = get_new_board()
    player_turn = "1"
    move_seed = seed

    while True:
        winner = check_for_winner(board)
        if winner != "no winner":
            return winner, board["1"], board["2"]

        if not legal_moves(player_turn, board):
            winner = check_for_winner(board)
            if winner == "no winner":
                winner = "tie"
            return winner, board["1"], board["2"]

        current = agent_p1 if player_turn == "1" else agent_p2
        move = agent_move(current, player_turn, board, move_seed)
        move_seed += 1
        player_turn = make_move(board, player_turn, move)


def build_agents() -> list[Agent]:
    model_names = [
        "modelo_mancala_4000_mas_perdedoras.h5",
        "modelo_mancala.h5",
        "modelo_mancala_4000.h5",
        "onLineModel_last.keras",
    ]
    agents = []
    for name in model_names:
        agents.append(Agent(name=name, kind="model", model=load_model_spec(Path(name))))
    agents.append(Agent(name="Codigo", kind="codigo"))
    agents.append(Agent(name="Codigo3", kind="codigo3"))
    agents.append(Agent(name="Codigo4", kind="codigo4"))
    agents.append(Agent(name="Codigo5", kind="codigo5"))
    alphazero_model = next((path for path in ALPHAZERO_CANDIDATE_MODELS if path.exists()), None)
    if alphazero_model is not None:
        alphazero_network = AlphaZeroNetwork()
        alphazero_network.load(alphazero_model)
        agents.append(
            Agent(
                name=f"AlphaZero_{alphazero_model.stem}",
                kind="alphazero_mcts",
                model=alphazero_network,
            )
        )
    agents.append(Agent(name="Azar", kind="azar"))
    return agents


def select_agents(all_agents: list[Agent], selected_names: list[str] | None) -> list[Agent]:
    if not selected_names:
        return all_agents

    name_map = {agent.name: agent for agent in all_agents}
    missing = [name for name in selected_names if name not in name_map]
    if missing:
        raise ValueError(f"Agentes no encontrados: {', '.join(missing)}")
    return [name_map[name] for name in selected_names]


def main() -> None:
    parser = argparse.ArgumentParser(description="Torneo todos contra todos para agentes de Mancala.")
    parser.add_argument("--games-per-pair", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260526)
    parser.add_argument("--agents", nargs="*", default=None)
    args = parser.parse_args()

    agents = select_agents(build_agents(), args.agents)
    games_per_pair = args.games_per_pair
    rng = random.Random(args.seed)
    table = {
        agent.name: {
            "points": 0,
            "wins": 0,
            "draws": 0,
            "losses": 0,
            "gf": 0,
            "ga": 0,
            "matches": 0,
        }
        for agent in agents
    }
    matchup_rows = []
    seed = args.seed

    for idx, (agent_a, agent_b) in enumerate(combinations(agents, 2), start=1):
        pair_summary = {
            agent_a.name: {"points": 0, "wins": 0, "draws": 0, "losses": 0},
            agent_b.name: {"points": 0, "wins": 0, "draws": 0, "losses": 0},
        }
        for game_idx in range(games_per_pair):
            if rng.random() < 0.5:
                home, away = agent_a, agent_b
            else:
                home, away = agent_b, agent_a
            match_seed = seed + idx * 10000 + game_idx
            winner, score1, score2 = play_match(home, away, match_seed)
            table[home.name]["matches"] += 1
            table[away.name]["matches"] += 1
            table[home.name]["gf"] += score1
            table[home.name]["ga"] += score2
            table[away.name]["gf"] += score2
            table[away.name]["ga"] += score1

            if winner == "1":
                table[home.name]["points"] += 3
                table[home.name]["wins"] += 1
                table[away.name]["losses"] += 1
                pair_summary[home.name]["points"] += 3
                pair_summary[home.name]["wins"] += 1
                pair_summary[away.name]["losses"] += 1
            elif winner == "2":
                table[away.name]["points"] += 3
                table[away.name]["wins"] += 1
                table[home.name]["losses"] += 1
                pair_summary[away.name]["points"] += 3
                pair_summary[away.name]["wins"] += 1
                pair_summary[home.name]["losses"] += 1
            else:
                table[home.name]["points"] += 1
                table[away.name]["points"] += 1
                table[home.name]["draws"] += 1
                table[away.name]["draws"] += 1
                pair_summary[home.name]["points"] += 1
                pair_summary[away.name]["points"] += 1
                pair_summary[home.name]["draws"] += 1
                pair_summary[away.name]["draws"] += 1

        matchup_rows.append(
            {
                "a": agent_a.name,
                "b": agent_b.name,
                "a_pts": pair_summary[agent_a.name]["points"],
                "b_pts": pair_summary[agent_b.name]["points"],
                "a_w": pair_summary[agent_a.name]["wins"],
                "b_w": pair_summary[agent_b.name]["wins"],
                "d": pair_summary[agent_a.name]["draws"],
            }
        )

    standings = []
    for name, row in table.items():
        standings.append(
            {
                "name": name,
                **row,
                "gd": row["gf"] - row["ga"],
            }
        )

    standings.sort(
        key=lambda row: (
            -row["points"],
            -row["gd"],
            -row["gf"],
            row["name"].lower(),
        )
    )

    print(f"Torneo todos contra todos a {games_per_pair} partidas por enfrentamiento")
    print("\nResumen por enfrentamiento")
    pair_header = f"{'Cruce':50} {'Pts A':>5} {'Pts B':>5} {'G A':>4} {'G B':>4} {'Emp':>4}"
    print(pair_header)
    print("-" * len(pair_header))
    for row in matchup_rows:
        label = f"{row['a']} vs {row['b']}"
        print(
            f"{label[:50]:50} {row['a_pts']:5d} {row['b_pts']:5d} "
            f"{row['a_w']:4d} {row['b_w']:4d} {row['d']:4d}"
        )

    print("\nTabla final")
    header = f"{'Equipo':34} {'Pts':>3} {'PJ':>3} {'G':>3} {'E':>3} {'P':>3} {'GF':>4} {'GC':>4} {'DG':>4}"
    print(header)
    print("-" * len(header))
    for row in standings:
        print(
            f"{row['name'][:34]:34} {row['points']:3d} {row['matches']:3d} {row['wins']:3d} "
            f"{row['draws']:3d} {row['losses']:3d} {row['gf']:4d} {row['ga']:4d} {row['gd']:4d}"
        )


if __name__ == "__main__":
    main()
