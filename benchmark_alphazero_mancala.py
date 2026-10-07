from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from alphazero_mancala import AlphaZeroNetwork, evaluate_alphazero
from mancala_dqn_selfplay import build_eval_opponents, load_model_spec, load_npz_opponents


def load_phase2_model(path: Path | None) -> Any | None:
    if path is None or not path.exists():
        return None
    try:
        return load_model_spec(path)
    except ModuleNotFoundError as exc:
        print(f"Aviso: omito {path} porque falta dependencia opcional: {exc}")
        return None


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def flattened_rows(mode: str, result: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [
        {
            "mode": mode,
            "opponent": "summary",
            "wins": "",
            "losses": "",
            "draws": "",
            "win_rate": "",
            "score": result["mixed_score"],
            "worst_case_score": result["worst_case_score"],
        }
    ]
    opponent_names = sorted(
        key[: -len("_wins")]
        for key in result
        if key.endswith("_wins")
    )
    for opponent in opponent_names:
        rows.append(
            {
                "mode": mode,
                "opponent": opponent,
                "wins": result[f"{opponent}_wins"],
                "losses": result[f"{opponent}_losses"],
                "draws": result[f"{opponent}_draws"],
                "win_rate": result[f"{opponent}_win_rate"],
                "score": result[f"{opponent}_score"],
                "worst_case_score": "",
            }
        )
    return rows


def print_table(rows: list[dict[str, Any]]) -> None:
    print(f"{'Mode':8} {'Opponent':28} {'W':>5} {'L':>5} {'D':>5} {'Win%':>8} {'Score':>8} {'Worst':>8}")
    print("-" * 86)
    for row in rows:
        win_rate = row["win_rate"]
        score = row["score"]
        worst = row["worst_case_score"]
        print(
            f"{row['mode'][:8]:8} "
            f"{row['opponent'][:28]:28} "
            f"{str(row['wins']):>5} "
            f"{str(row['losses']):>5} "
            f"{str(row['draws']):>5} "
            f"{(f'{win_rate:.1%}' if isinstance(win_rate, float) else ''):>8} "
            f"{(f'{score:.2%}' if isinstance(score, float) else ''):>8} "
            f"{(f'{worst:.2%}' if isinstance(worst, float) else ''):>8}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compara un modelo AlphaZero Mancala contra baselines existentes.")
    parser.add_argument("--model", type=Path, default=Path("alphazero_mancala_10000_final.npz"))
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--mcts-simulations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--output-csv", type=Path, default=Path("alphazero_mancala_benchmark.csv"))
    parser.add_argument("--phase2-model", type=Path, default=Path("modelo_mancala_4000.h5"))
    parser.add_argument(
        "--fixed-dqn-model",
        type=Path,
        action="append",
        default=[Path("dqn_best_remote_30000_epsmin02.npz")],
    )
    args = parser.parse_args()

    network = AlphaZeroNetwork()
    network.load(args.model)
    fixed_dqn_models = load_npz_opponents(args.fixed_dqn_model)
    phase2_model = load_phase2_model(args.phase2_model)
    opponents = build_eval_opponents(
        fixed_dqn_models=fixed_dqn_models,
        phase2_model=phase2_model,
        include_azar=True,
        include_codigo=True,
    )

    policy_result = evaluate_alphazero(
        network,
        opponents=opponents,
        games=args.games,
        use_mcts=False,
        simulations=args.mcts_simulations,
        seed=args.seed,
    )
    mcts_result = evaluate_alphazero(
        network,
        opponents=opponents,
        games=args.games,
        use_mcts=True,
        simulations=args.mcts_simulations,
        seed=args.seed + 999_983,
    )
    rows = flattened_rows("policy", policy_result) + flattened_rows("mcts", mcts_result)
    print_table(rows)
    write_csv(args.output_csv, rows)
    print(f"\nCSV guardado en {args.output_csv}")


if __name__ == "__main__":
    main()
