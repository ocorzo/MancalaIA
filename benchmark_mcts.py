"""Compare a pure DDQN against the same DDQN guided by MCTS."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from mancala_dqn_selfplay import (
    NPZModel,
    action_to_label,
    board_to_vector,
    check_for_winner,
    get_new_board,
    legal_actions,
    make_move,
)
from mancala_mcts import MCTSPlayer


def model_action(model: NPZModel, board: dict[str, int], player_turn: str) -> str:
    q_values = model.predict(board_to_vector(board, player_turn))[0]
    actions = legal_actions(player_turn, board)
    best = max(q_values[action] for action in actions)
    candidates = [action for action in actions if q_values[action] == best]
    return action_to_label(candidates[0])


def play_game(model: NPZModel, simulations: int, seed: int, mcts_player_number: str) -> str:
    mcts_rng = random.Random(seed + 100_003)
    mcts = MCTSPlayer(
        model=model,
        simulations=simulations,
        rollout_randomness=0.08,
        rng=mcts_rng,
    )
    board = get_new_board()
    player_turn = "1" if seed % 2 == 0 else "2"
    while True:
        if player_turn == mcts_player_number:
            move = mcts.move(board, player_turn)
        else:
            move = model_action(model, board, player_turn)
        player_turn = make_move(board, player_turn, move)
        winner = check_for_winner(board)
        if winner != "no winner":
            return winner


def run_benchmark(model_path: Path, games: int, simulations: int, seed: int) -> dict[str, float | int]:
    model = NPZModel(model_path)
    wins = losses = draws = 0
    for game_index in range(games):
        mcts_player = "1" if game_index % 2 == 0 else "2"
        winner = play_game(model, simulations, seed + game_index, mcts_player)
        if winner == "tie":
            draws += 1
        elif winner == mcts_player:
            wins += 1
        else:
            losses += 1
    return {
        "games": games,
        "simulations": simulations,
        "mcts_wins": wins,
        "baseline_wins": losses,
        "draws": draws,
        "mcts_score": (wins + 0.5 * draws) / games,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("dqn_best_remote_30000_epsmin02.npz"))
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--simulations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260718)
    args = parser.parse_args()
    result = run_benchmark(args.model, args.games, args.simulations, args.seed)
    print("Benchmark DDQN puro vs DDQN + MCTS")
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
