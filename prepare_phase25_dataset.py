from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


COLUMN_NAMES = [
    "player_turn",
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "1",
    "L",
    "K",
    "J",
    "I",
    "H",
    "G",
    "2",
    "move",
    "outcome",
]

MOVE_TO_INDEX = {label: idx for idx, label in enumerate("ABCDEFGHIJKL")}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Consolida logs fase 2.5 en datasets de ganadoras, perdedoras y empates."
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path("logs_fase2_5/dqn_best_vs_fase2_10000"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("phase25_dataset/dqn_best_vs_fase2_10000"),
    )
    return parser


def load_logs(logs_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(logs_dir.glob("*.txt")):
        if path.name == "run.log":
            continue
        df = pd.read_csv(path, header=None, names=COLUMN_NAMES)
        df["source_file"] = path.name
        frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No se encontraron logs .txt en {logs_dir}")

    full_df = pd.concat(frames, ignore_index=True)
    full_df["player_turn"] = full_df["player_turn"].astype(int)
    for column in COLUMN_NAMES[1:15]:
        full_df[column] = full_df[column].astype(int)
    full_df["move_idx"] = full_df["move"].map(MOVE_TO_INDEX)
    if full_df["move_idx"].isna().any():
        bad_moves = sorted(full_df.loc[full_df["move_idx"].isna(), "move"].unique())
        raise ValueError(f"Se encontraron movimientos no reconocidos: {bad_moves}")
    full_df["move_idx"] = full_df["move_idx"].astype(int)
    return full_df


def save_dataset(df: pd.DataFrame, output_dir: Path, name: str) -> int:
    path = output_dir / f"{name}.csv"
    df.to_csv(path, index=False)
    return len(df)


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    full_df = load_logs(args.logs_dir)
    winners_df = full_df[full_df["outcome"] == "winner"].copy()
    losers_df = full_df[full_df["outcome"] == "loser"].copy()
    ties_df = full_df[full_df["outcome"] == "tie"].copy()

    counts = {
        "all_rows": save_dataset(full_df, args.output_dir, "all_rows"),
        "winners": save_dataset(winners_df, args.output_dir, "winners"),
        "losers": save_dataset(losers_df, args.output_dir, "losers"),
        "ties": save_dataset(ties_df, args.output_dir, "ties"),
        "games": len({name for name in full_df["source_file"]}),
    }

    metadata = {
        "logs_dir": str(args.logs_dir),
        "output_dir": str(args.output_dir),
        "counts": counts,
        "columns": COLUMN_NAMES + ["source_file", "move_idx"],
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("Dataset fase 2.5 preparado.")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
