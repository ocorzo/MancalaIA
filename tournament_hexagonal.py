"""Balanced all-play-all tournament for the two AlphaZero models and baselines."""
from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path
import random

from alphazero_mancala import AlphaZeroMCTS, AlphaZeroNetwork
from benchmark_mancala_models import azar_move, codigo3_move, codigo4_move, codigo5_move
from mancala_dqn_selfplay import action_to_label, check_for_winner, get_new_board, make_move


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games-per-pair", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.games_per_pair <= 0 or args.games_per_pair % 4:
        parser.error("games-per-pair must be positive and divisible by four")
    rng = random.Random(args.seed)
    agents = {}
    for name, path in (
        ("AlphaZero anterior", "alphazero_mancala_10000_best_mcts.npz"),
        ("AlphaZero nuevo", "alphazero_continue_20261006/best_mcts.npz"),
    ):
        net = AlphaZeroNetwork()
        net.load(Path(path))

        def decide(board, player, network=net):
            mcts = AlphaZeroMCTS(network, simulations=50, rng=random.Random(123))
            return action_to_label(mcts.search(board, player, temperature=0, sample=False).action)

        agents[name] = decide
    agents.update({
        "Azar": lambda b, p: azar_move(p, b, rng),
        "Codigo3": lambda b, p: codigo3_move(p, b),
        "Codigo4": lambda b, p: codigo4_move(p, b),
        "Codigo5": lambda b, p: codigo5_move(p, b),
    })
    standings = {n: {"name": n, "points": 0, "games": 0, "wins": 0,
                     "draws": 0, "losses": 0} for n in agents}
    pairs = []
    roles = (("1", "1"), ("1", "2"), ("2", "1"), ("2", "2"))
    for a, b in combinations(agents, 2):
        wins_a = wins_b = draws = 0
        for game in range(args.games_per_pair):
            side_a, player = roles[game % 4]
            players = {side_a: a, "2" if side_a == "1" else "1": b}
            board = get_new_board()
            for _ in range(250):
                winner = check_for_winner(board)
                if winner != "no winner":
                    break
                move = agents[players[player]](board, player)
                player = make_move(board, player, move)
            else:
                winner = check_for_winner(board)
                if winner == "no winner":
                    raise RuntimeError(f"Unfinished match: {a} vs {b}")
            for name in (a, b):
                row = standings[name]
                row["games"] += 1
                if winner == "tie":
                    row["draws"] += 1
                    row["points"] += 1
                elif players[winner] == name:
                    row["wins"] += 1
                    row["points"] += 3
                else:
                    row["losses"] += 1
            if winner == "tie":
                draws += 1
            elif players[winner] == a:
                wins_a += 1
            else:
                wins_b += 1
        pair = {"a": a, "b": b, "wins_a": wins_a, "wins_b": wins_b, "draws": draws}
        pairs.append(pair)
        print(json.dumps(pair), flush=True)
    table = sorted(standings.values(), key=lambda r: (-r["points"], r["name"]))
    for row in table:
        assert row["games"] == row["wins"] + row["draws"] + row["losses"]
        assert row["points"] == 3 * row["wins"] + row["draws"]
        row["percent_max"] = 100 * row["points"] / (3 * row["games"])
    result = {"games_per_pair": args.games_per_pair, "seed": args.seed,
              "mcts_simulations": 50, "balanced_roles": True, "pairs": pairs, "standings": table}
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(table, indent=2), flush=True)


if __name__ == "__main__":
    main()
