"""Measure decision latency in matches and on a shared sample of positions."""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

from alphazero_mancala import AlphaZeroMCTS, AlphaZeroNetwork
from benchmark_mancala_models import azar_move, codigo3_move, codigo4_move, codigo5_move
from mancala_dqn_selfplay import action_to_label, check_for_winner, get_new_board, make_move


def stats(samples):
    wall = np.array([x[0] for x in samples]) * 1000
    cpu = np.array([x[1] for x in samples]) * 1000
    return {
        "decisions": len(samples), "mean_ms": float(wall.mean()),
        "median_ms": float(np.median(wall)), "p95_ms": float(np.percentile(wall, 95)),
        "cpu_mean_ms": float(cpu.mean()), "total_seconds": float(wall.sum() / 1000),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games-per-pair", type=int, default=40)
    parser.add_argument("--shared-positions", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rng = random.Random(20261006)
    random.seed(20261006)
    agents = {}
    for name, path in (
        ("AlphaZero_anterior", "alphazero_mancala_10000_best_mcts.npz"),
        ("AlphaZero_nuevo", "alphazero_continue_20261006/best_mcts.npz"),
    ):
        net = AlphaZeroNetwork()
        net.load(Path(path))

        def decide(board, player, network=net):
            search = AlphaZeroMCTS(network, simulations=50, rng=random.Random(123))
            return action_to_label(search.search(board, player, temperature=0, sample=False).action)

        agents[name] = decide
    agents.update({
        "Azar": lambda b, p: azar_move(p, b, rng),
        "Codigo3": lambda b, p: codigo3_move(p, b),
        "Codigo4": lambda b, p: codigo4_move(p, b),
        "Codigo5": lambda b, p: codigo5_move(p, b),
    })
    match_samples = {name: [] for name in agents}
    positions = []
    roles = (("1", "1"), ("1", "2"), ("2", "1"), ("2", "2"))
    for agent in ("AlphaZero_anterior", "AlphaZero_nuevo"):
        for rival in ("Azar", "Codigo3", "Codigo4", "Codigo5"):
            for game in range(args.games_per_pair):
                board = get_new_board()
                ai_side, player = roles[game % 4]
                for _ in range(250):
                    if check_for_winner(board) != "no winner":
                        break
                    positions.append((board.copy(), player))
                    name = agent if player == ai_side else rival
                    start_wall, start_cpu = time.perf_counter(), time.process_time()
                    move = agents[name](board, player)
                    elapsed_cpu = time.process_time() - start_cpu
                    elapsed_wall = time.perf_counter() - start_wall
                    match_samples[name].append((elapsed_wall, elapsed_cpu))
                    player = make_move(board, player, move)
                else:
                    raise RuntimeError("Match exceeded 250 moves")
            print(f"Completed {agent} vs {rival}", flush=True)
    # Deduplicate positions so deterministic trajectories do not dominate the sample.
    unique = {tuple([player] + [board[k] for k in sorted(board)]): (board, player)
              for board, player in positions}
    shared = rng.sample(list(unique.values()), min(args.shared_positions, len(unique)))
    same_samples = {name: [] for name in agents}
    for board, player in shared:
        for repeat in range(args.repeats):
            order = list(agents)
            rng.shuffle(order)
            for name in order:
                start_wall, start_cpu = time.perf_counter(), time.process_time()
                agents[name](board.copy(), player)
                elapsed_cpu = time.process_time() - start_cpu
                elapsed_wall = time.perf_counter() - start_wall
                same_samples[name].append((elapsed_wall, elapsed_cpu))
    result = {
        "protocol": {"games_per_pair": args.games_per_pair, "total_games": 8 * args.games_per_pair,
                     "shared_positions": len(shared), "unique_positions": len(unique),
                     "repeats": args.repeats, "mcts_simulations": 50, "seed": 20261006},
        "matches": {name: stats(values) for name, values in match_samples.items()},
        "shared_positions": {name: stats(values) for name, values in same_samples.items()},
    }
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
