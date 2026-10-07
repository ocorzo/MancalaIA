from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from benchmark_mancala_models import (
    PIT_LABELS,
    get_new_board,
    load_model_spec,
    model_move,
    make_move,
    check_for_winner,
)


ACTION_LABELS = tuple("ABCDEFGHIJKL")


@dataclass
class Agent:
    name: str
    kind: str
    payload: object


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


def board_to_vector(board: dict[str, int], player_turn: str) -> np.ndarray:
    return np.array(
        [float(int(player_turn))] + [float(board[pit]) for pit in PIT_LABELS],
        dtype=np.float64,
    )


def legal_moves(player_turn: str, board: dict[str, int]) -> list[str]:
    pits = "ABCDEF" if player_turn == "1" else "GHIJKL"
    return [pit for pit in pits if board[pit] > 0]


def npz_model_move(model: NPZModel, player_turn: str, board: dict[str, int]) -> str:
    scores = model.predict(board_to_vector(board, player_turn))[0]
    valid = set(legal_moves(player_turn, board))
    for idx in np.argsort(scores)[::-1]:
        move = ACTION_LABELS[int(idx)]
        if move in valid:
            return move
    raise RuntimeError("No se encontro jugada valida para el modelo NPZ")


def agent_move(agent: Agent, player_turn: str, board: dict[str, int]) -> str:
    if agent.kind == "npz":
        return npz_model_move(agent.payload, player_turn, board)
    if agent.kind == "h5":
        return model_move(agent.payload, player_turn, board)
    raise ValueError(f"Tipo de agente no soportado: {agent.kind}")


def annotate_lines(lines: list[str], winner: str) -> list[str]:
    annotated = []
    for line in lines:
        player = line.split(",")[0]
        if winner == "tie":
            outcome = "tie"
        elif player == winner:
            outcome = "winner"
        else:
            outcome = "loser"
        annotated.append(f"{line},{outcome}\n")
    return annotated


def play_logged_game(
    agent_p1: Agent,
    agent_p2: Agent,
    start_player: str,
) -> tuple[str, list[str], dict[str, int]]:
    board = get_new_board()
    player_turn = start_player
    raw_lines: list[str] = []

    while True:
        current_agent = agent_p1 if player_turn == "1" else agent_p2
        move = agent_move(current_agent, player_turn, board)
        board_state = ",".join(str(board[pit]) for pit in PIT_LABELS)
        raw_lines.append(f"{player_turn},{board_state},{move}")
        player_turn = make_move(board, player_turn, move)
        winner = check_for_winner(board)
        if winner != "no winner":
            return winner, annotate_lines(raw_lines, winner), board.copy()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Genera logs fase 2.5 entre fase 3 best y fase 2 best.")
    parser.add_argument("--games", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260527)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("logs_fase2_5/dqn_best_vs_fase2_10000"),
    )
    parser.add_argument(
        "--phase3-model",
        type=Path,
        default=Path("dqn_best_remote_30000_epsmin02.npz"),
    )
    parser.add_argument(
        "--phase2-model",
        type=Path,
        default=Path("modelo_mancala_4000_mas_perdedoras.h5"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rng = random.Random(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    phase3 = Agent(
        name="fase3_best",
        kind="npz",
        payload=NPZModel(args.phase3_model),
    )
    phase2 = Agent(
        name="fase2_best",
        kind="h5",
        payload=load_model_spec(args.phase2_model),
    )

    summary_rows: list[dict[str, object]] = []
    phase3_wins = 0
    phase2_wins = 0
    ties = 0

    for game_idx in range(1, args.games + 1):
        if game_idx <= args.games // 2:
            agent_p1 = phase3
            agent_p2 = phase2
        else:
            agent_p1 = phase2
            agent_p2 = phase3

        start_player = rng.choice(["1", "2"])
        winner, lines, final_board = play_logged_game(agent_p1, agent_p2, start_player)

        if winner == "1":
            winner_agent = agent_p1.name
        elif winner == "2":
            winner_agent = agent_p2.name
        else:
            winner_agent = "tie"

        if winner_agent == phase3.name:
            phase3_wins += 1
        elif winner_agent == phase2.name:
            phase2_wins += 1
        else:
            ties += 1

        filename = (
            f"log_fase25_{agent_p1.name}_vs_{agent_p2.name}_"
            f"{game_idx:05d}_{winner if winner != 'tie' else 'tie'}.txt"
        )
        (args.output_dir / filename).write_text("".join(lines), encoding="utf-8")

        summary_rows.append(
            {
                "game": game_idx,
                "agent_p1": agent_p1.name,
                "agent_p2": agent_p2.name,
                "start_player": start_player,
                "winner_player": winner,
                "winner_agent": winner_agent,
                "p1_store": final_board["1"],
                "p2_store": final_board["2"],
            }
        )

        if game_idx % 500 == 0 or game_idx == args.games:
            print(
                f"Partidas: {game_idx}/{args.games} | "
                f"fase3_best={phase3_wins} fase2_best={phase2_wins} empates={ties}"
            )

    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "game",
                "agent_p1",
                "agent_p2",
                "start_player",
                "winner_player",
                "winner_agent",
                "p1_store",
                "p2_store",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    metadata = {
        "games": args.games,
        "seed": args.seed,
        "phase3_model": str(args.phase3_model),
        "phase2_model": str(args.phase2_model),
        "phase3_wins": phase3_wins,
        "phase2_wins": phase2_wins,
        "ties": ties,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("Generacion de logs completada.")
    print(f"Directorio: {args.output_dir}")
    print(
        f"Resultado final | fase3_best={phase3_wins} "
        f"fase2_best={phase2_wins} empates={ties}"
    )


if __name__ == "__main__":
    main()
