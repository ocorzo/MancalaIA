from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical


FEATURE_COLUMNS = [
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
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reentrena el mejor modelo de fase 2 con ganadoras y perdedoras generadas por fase 3."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("phase25_dataset/dqn_best_vs_fase2_10000"),
    )
    parser.add_argument(
        "--base-model",
        type=Path,
        default=Path("modelo_mancala_4000_mas_perdedoras.h5"),
    )
    parser.add_argument(
        "--output-model",
        type=Path,
        default=Path("modelo_mancala_fase25.h5"),
    )
    parser.add_argument(
        "--metrics-json",
        type=Path,
        default=Path("modelo_mancala_fase25_metrics.json"),
    )
    parser.add_argument("--winner-epochs", type=int, default=30)
    parser.add_argument("--loser-epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260527)
    parser.add_argument(
        "--winner-train-fraction",
        type=float,
        default=1.0,
        help="Fraccion de filas ganadoras de entrenamiento que se usara, entre 0.0 y 1.0.",
    )
    parser.add_argument(
        "--training-order",
        choices=("winners_first", "losers_first"),
        default="winners_first",
        help="Orden de las fases de entrenamiento supervisado.",
    )
    parser.add_argument(
        "--winner-label-smoothing",
        type=float,
        default=0.0,
        help="Suavizado para las etiquetas de ganadoras, entre 0.0 y 1.0.",
    )
    parser.add_argument(
        "--loser-action-scale",
        type=float,
        default=0.10,
        help="Escala aplicada a la probabilidad del movimiento perdedor antes de renormalizar.",
    )
    return parser


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [column for column in FEATURE_COLUMNS + ["move_idx", "source_file"] if column not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas en {path}: {missing}")
    return df


def split_games(
    winners_df: pd.DataFrame,
    losers_df: pd.DataFrame,
    test_size: float,
    seed: int,
    winner_train_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_games = np.array(sorted(set(winners_df["source_file"]) | set(losers_df["source_file"])))
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(all_games)
    test_count = max(1, int(len(shuffled) * test_size))
    test_games = set(shuffled[:test_count].tolist())
    train_games = set(shuffled[test_count:].tolist())

    winners_train = winners_df[winners_df["source_file"].isin(train_games)].copy()
    winners_test = winners_df[winners_df["source_file"].isin(test_games)].copy()
    losers_train = losers_df[losers_df["source_file"].isin(train_games)].copy()
    losers_test = losers_df[losers_df["source_file"].isin(test_games)].copy()

    if 0 < winner_train_fraction < 1.0 and not winners_train.empty:
        sample_size = max(1, int(len(winners_train) * winner_train_fraction))
        sampled_idx = rng.choice(winners_train.index.to_numpy(), size=sample_size, replace=False)
        winners_train = winners_train.loc[np.sort(sampled_idx)].copy()

    return winners_train, winners_test, losers_train, losers_test


def to_supervised_arrays(
    df: pd.DataFrame,
    label_smoothing: float,
) -> tuple[np.ndarray, np.ndarray]:
    x = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y_idx = df["move_idx"].to_numpy(dtype=np.int64)
    y = to_categorical(y_idx, num_classes=12)
    if label_smoothing > 0:
        smooth = float(label_smoothing)
        y = y * (1.0 - smooth) + (smooth / y.shape[1])
    return x, y


def penalized_targets(model, df: pd.DataFrame, loser_action_scale: float) -> tuple[np.ndarray, np.ndarray]:
    x = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    move_idx = df["move_idx"].to_numpy(dtype=np.int64)
    y_pred = model.predict(x, verbose=0)

    adjusted = y_pred.copy()
    for row_idx, action_idx in enumerate(move_idx):
        adjusted[row_idx, action_idx] *= loser_action_scale
        total = adjusted[row_idx].sum()
        if total <= 0:
            adjusted[row_idx] = np.full(12, 1.0 / 12.0, dtype=np.float32)
        else:
            adjusted[row_idx] /= total
    return x, adjusted


def evaluate_model(model, x_test: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    return {"loss": float(loss), "accuracy": float(accuracy)}


def load_and_compile_model(path: Path):
    model = load_model(path, compile=False)
    model.compile(
        loss=CategoricalCrossentropy(),
        optimizer=Adam(),
        metrics=["accuracy"],
    )
    return model


def main() -> None:
    args = build_parser().parse_args()

    winners_df = load_dataset(args.dataset_dir / "winners.csv")
    losers_df = load_dataset(args.dataset_dir / "losers.csv")

    winners_train_df, winners_test_df, losers_train_df, losers_test_df = split_games(
        winners_df,
        losers_df,
        test_size=args.test_size,
        seed=args.seed,
        winner_train_fraction=args.winner_train_fraction,
    )

    x_train, y_train = to_supervised_arrays(
        winners_train_df,
        label_smoothing=args.winner_label_smoothing,
    )
    x_test, y_test = to_supervised_arrays(winners_test_df, label_smoothing=0.0)

    model = load_and_compile_model(args.base_model)
    base_metrics = evaluate_model(model, x_test, y_test)

    loser_x, loser_y = penalized_targets(
        model,
        losers_train_df,
        loser_action_scale=args.loser_action_scale,
    )

    histories: dict[str, object] = {}
    metrics_after_stage: dict[str, dict[str, float]] = {}

    def train_winners() -> None:
        histories["winners"] = model.fit(
            x_train,
            y_train,
            epochs=args.winner_epochs,
            batch_size=args.batch_size,
            verbose=1,
            validation_data=(x_test, y_test),
        )
        metrics_after_stage["winners"] = evaluate_model(model, x_test, y_test)

    def train_losers() -> None:
        histories["losers"] = model.fit(
            loser_x,
            loser_y,
            epochs=args.loser_epochs,
            batch_size=args.batch_size,
            verbose=1,
        )
        metrics_after_stage["losers"] = evaluate_model(model, x_test, y_test)

    if args.training_order == "winners_first":
        train_winners()
        train_losers()
    else:
        train_losers()
        train_winners()

    model.save(args.output_model)

    winner_history = histories["winners"]
    loser_history = histories["losers"]
    post_winner_metrics = metrics_after_stage["winners"]
    post_loser_metrics = metrics_after_stage["losers"]

    metrics = {
        "dataset_dir": str(args.dataset_dir),
        "base_model": str(args.base_model),
        "output_model": str(args.output_model),
        "winner_rows": int(len(winners_df)),
        "loser_rows": int(len(losers_df)),
        "winner_train_rows": int(len(winners_train_df)),
        "winner_test_rows": int(len(winners_test_df)),
        "loser_train_rows": int(len(losers_train_df)),
        "loser_test_rows": int(len(losers_test_df)),
        "winner_epochs": args.winner_epochs,
        "loser_epochs": args.loser_epochs,
        "batch_size": args.batch_size,
        "training_order": args.training_order,
        "winner_train_fraction": args.winner_train_fraction,
        "winner_label_smoothing": args.winner_label_smoothing,
        "loser_action_scale": args.loser_action_scale,
        "base_metrics": base_metrics,
        "post_winner_metrics": post_winner_metrics,
        "post_loser_metrics": post_loser_metrics,
        "winner_history_last": {
            key: float(values[-1]) for key, values in winner_history.history.items()
        },
        "loser_history_last": {
            key: float(values[-1]) for key, values in loser_history.history.items()
        },
    }

    args.metrics_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("Entrenamiento fase 2.5 completado.")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
