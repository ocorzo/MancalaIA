from __future__ import annotations

import json
import math
import random
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import h5py
except ModuleNotFoundError:  # Optional when evaluating NumPy/NPZ models only.
    h5py = None


PIT_LABELS = "ABCDEF1LKJIHG2"
PLAYER_1_PITS = ("A", "B", "C", "D", "E", "F")
PLAYER_2_PITS = ("G", "H", "I", "J", "K", "L")
NEXT_PIT = {
    "A": "B",
    "B": "C",
    "C": "D",
    "D": "E",
    "E": "F",
    "F": "1",
    "1": "L",
    "L": "K",
    "K": "J",
    "J": "I",
    "I": "H",
    "H": "G",
    "G": "2",
    "2": "A",
}
OPPOSITE_PIT = {
    "A": "G",
    "B": "H",
    "C": "I",
    "D": "J",
    "E": "K",
    "F": "L",
    "G": "A",
    "H": "B",
    "I": "C",
    "J": "D",
    "K": "E",
    "L": "F",
}


@dataclass
class DenseLayer:
    units: int
    activation: str
    kernel: np.ndarray
    bias: np.ndarray


@dataclass
class ModelSpec:
    path: Path
    input_size: int
    layers: list[DenseLayer]

    def predict(self, vector: np.ndarray) -> np.ndarray:
        x = vector.astype(np.float64)
        for layer in self.layers:
            x = x @ layer.kernel + layer.bias
            if layer.activation == "relu":
                x = np.maximum(x, 0.0)
            elif layer.activation == "softmax":
                x = _softmax(x)
            elif layer.activation == "linear":
                pass
            else:
                raise ValueError(f"Activacion no soportada: {layer.activation}")
        return x


def _softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def _extract_json_from_h5(path: Path) -> dict:
    raw = path.read_bytes()
    marker = b'{"class_name": "Sequential"'
    start = raw.find(marker)
    if start == -1:
        marker = b'{"class_name":"Sequential"'
        start = raw.find(marker)
    if start == -1:
        raise ValueError(f"No encontre config JSON en {path.name}")
    text = raw[start:].decode("utf-8", errors="ignore")
    decoder = json.JSONDecoder()
    obj, _ = decoder.raw_decode(text)
    return obj


def _load_keras_config(path: Path) -> dict:
    with zipfile.ZipFile(path) as zf:
        return json.loads(zf.read("config.json"))


def _layer_defs_from_config(config: dict) -> tuple[int, list[tuple[str, int, str]]]:
    layers = config["config"]["layers"]
    input_size = None
    dense_defs: list[tuple[str, int, str]] = []
    for layer in layers:
        class_name = layer["class_name"]
        layer_config = layer["config"]
        if class_name == "InputLayer":
            batch_shape = layer_config.get("batch_shape") or layer_config.get(
                "batch_input_shape"
            )
            input_size = int(batch_shape[-1])
        elif class_name == "Dense":
            dense_defs.append(
                (
                    layer_config["name"],
                    int(layer_config["units"]),
                    layer_config["activation"],
                )
            )
    if input_size is None:
        raise ValueError("No pude determinar el tamano de entrada")
    return input_size, dense_defs


def _load_weights_from_keras(path: Path, dense_defs: list[tuple[str, int, str]]) -> list[DenseLayer]:
    temp_path = Path(tempfile.gettempdir()) / f"{path.stem}_weights.h5"
    with zipfile.ZipFile(path) as zf:
        temp_path.write_bytes(zf.read("model.weights.h5"))
    layers: list[DenseLayer] = []
    with h5py.File(temp_path, "r") as h5:
        weight_layer_names = sorted(
            h5["layers"].keys(),
            key=lambda name: (0 if name == "dense" else int(name.split("_")[-1]) + 1),
        )
        for (config_name, units, activation), weight_name in zip(dense_defs, weight_layer_names):
            kernel = np.array(h5[f"layers/{weight_name}/vars/0"])
            bias = np.array(h5[f"layers/{weight_name}/vars/1"])
            layers.append(DenseLayer(units=units, activation=activation, kernel=kernel, bias=bias))
    return layers


def _find_dataset(group: h5py.Group, dataset_name: str) -> np.ndarray:
    found = None

    def visitor(name: str, obj: h5py.Dataset | h5py.Group) -> None:
        nonlocal found
        if found is not None:
            return
        if isinstance(obj, h5py.Dataset) and name.endswith(f"/{dataset_name}"):
            found = np.array(obj)

    group.visititems(visitor)
    if found is None:
        raise KeyError(f"No encontre dataset {dataset_name}")
    return found


def _load_weights_from_h5(path: Path, dense_defs: list[tuple[str, int, str]]) -> list[DenseLayer]:
    if h5py is None:
        raise ModuleNotFoundError("h5py es necesario para cargar modelos .h5")
    layers: list[DenseLayer] = []
    with h5py.File(path, "r") as h5:
        model_weights = h5["model_weights"]
        for layer_name, units, activation in dense_defs:
            group = model_weights[layer_name]
            kernel = _find_dataset(group, "kernel")
            bias = _find_dataset(group, "bias")
            layers.append(DenseLayer(units=units, activation=activation, kernel=kernel, bias=bias))
    return layers


def load_model_spec(path: Path) -> ModelSpec:
    if path.suffix == ".keras":
        config = _load_keras_config(path)
        input_size, dense_defs = _layer_defs_from_config(config)
        layers = _load_weights_from_keras(path, dense_defs)
    elif path.suffix == ".h5":
        config = _extract_json_from_h5(path)
        input_size, dense_defs = _layer_defs_from_config(config)
        layers = _load_weights_from_h5(path, dense_defs)
    else:
        raise ValueError(f"Formato no soportado: {path.suffix}")
    return ModelSpec(path=path, input_size=input_size, layers=layers)


def get_new_board() -> dict[str, int]:
    return {
        "1": 0,
        "2": 0,
        "A": 4,
        "B": 4,
        "C": 4,
        "D": 4,
        "E": 4,
        "F": 4,
        "G": 4,
        "H": 4,
        "I": 4,
        "J": 4,
        "K": 4,
        "L": 4,
    }


def legal_moves(player_turn: str, board: dict[str, int]) -> list[str]:
    pits = PLAYER_1_PITS if player_turn == "1" else PLAYER_2_PITS
    return [pit for pit in pits if board[pit] > 0]


def opponent_of(player_turn: str) -> str:
    return "2" if player_turn == "1" else "1"


def azar_move(player_turn: str, board: dict[str, int], rng: random.Random) -> str:
    return rng.choice(legal_moves(player_turn, board))


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


def store_diff_for_player(board: dict[str, int], player_turn: str) -> int:
    opponent = opponent_of(player_turn)
    return int(board[player_turn]) - int(board[opponent])


def simulate_full_move(
    board: dict[str, int],
    player_turn: str,
    pit: str,
) -> tuple[dict[str, int], str, str]:
    simulated = board.copy()
    next_player = make_move(simulated, player_turn, pit)
    winner = check_for_winner(simulated)
    return simulated, next_player, winner


def codigo_search_score(
    board: dict[str, int],
    current_player: str,
    root_player: str,
    depth: int,
) -> float:
    winner = check_for_winner(board.copy())
    if winner != "no winner" or depth <= 0:
        return float(store_diff_for_player(board, root_player))

    moves = legal_moves(current_player, board)
    if not moves:
        return float(store_diff_for_player(board, root_player))

    best_score = float("-inf")

    for pit in moves:
        simulated, next_player, winner = simulate_full_move(board, current_player, pit)
        if winner != "no winner":
            score = float(store_diff_for_player(simulated, root_player))
        else:
            score = codigo_search_score(simulated, next_player, root_player, depth - 1)
        best_score = max(best_score, score)

    return best_score


def codigo_move_depth(player_turn: str, board: dict[str, int], depth: int = 2) -> str:
    moves = legal_moves(player_turn, board)
    if not moves:
        raise RuntimeError(f"El jugador {player_turn} no tiene jugadas legales")

    best_move = moves[0]
    best_score = float("-inf")
    best_immediate = float("-inf")

    for pit in moves:
        simulated, next_player, winner = simulate_full_move(board, player_turn, pit)
        immediate = float(store_diff_for_player(simulated, player_turn))
        if winner != "no winner":
            score = float(store_diff_for_player(simulated, player_turn))
        else:
            score = codigo_search_score(simulated, next_player, player_turn, max(0, depth - 1))

        if score > best_score or (score == best_score and immediate > best_immediate):
            best_score = score
            best_immediate = immediate
            best_move = pit

    return best_move


def codigo_move(player_turn: str, board: dict[str, int]) -> str:
    return codigo_move_depth(player_turn, board, depth=2)


def codigo3_move(player_turn: str, board: dict[str, int]) -> str:
    return codigo_move_depth(player_turn, board, depth=3)


def codigo4_move(player_turn: str, board: dict[str, int]) -> str:
    return codigo_move_depth(player_turn, board, depth=4)


def codigo5_move(player_turn: str, board: dict[str, int]) -> str:
    return codigo_move_depth(player_turn, board, depth=5)


def board_vector_for_model(board: dict[str, int], player_turn: str, input_size: int) -> np.ndarray:
    ordered = np.array([board[pit] for pit in PIT_LABELS], dtype=np.float64)
    if input_size == 15:
        return np.concatenate(([float(int(player_turn))], ordered))
    if input_size == 14:
        return ordered
    raise ValueError(f"Tamano de entrada no soportado: {input_size}")


def model_move(model: ModelSpec, player_turn: str, board: dict[str, int]) -> str:
    vector = board_vector_for_model(board, player_turn, model.input_size)
    scores = model.predict(vector)
    sorted_indices = list(np.argsort(scores)[::-1])
    conversion_table = "ABCDEFGHIJKL"
    valid = set(legal_moves(player_turn, board))
    for idx in sorted_indices:
        move = conversion_table[int(idx)]
        if move in valid:
            return move
    raise RuntimeError(f"{model.path.name} no encontro jugada valida")


def play_game(model: ModelSpec, rng: random.Random) -> dict[str, object]:
    board = get_new_board()
    ai_player = rng.choice(["1", "2"])
    player_turn = rng.choice(["1", "2"])
    first_player = player_turn

    while True:
        if player_turn == ai_player:
            move = model_move(model, player_turn, board)
        else:
            move = azar_move(player_turn, board, rng)
        player_turn = make_move(board, player_turn, move)
        winner = check_for_winner(board)
        if winner != "no winner":
            return {
                "winner": winner,
                "ai_player": ai_player,
                "first_player": first_player,
                "ai_store": board[ai_player],
                "azar_store": board["1" if ai_player == "2" else "2"],
            }


def evaluate_model(model: ModelSpec, games: int, seed: int) -> dict[str, object]:
    rng = random.Random(seed)
    wins = 0
    losses = 0
    draws = 0
    ai_started = 0
    ai_as_p1 = 0
    score_diff_total = 0

    for _ in range(games):
        result = play_game(model, rng)
        ai_as_p1 += 1 if result["ai_player"] == "1" else 0
        ai_started += 1 if result["first_player"] == result["ai_player"] else 0
        score_diff_total += int(result["ai_store"]) - int(result["azar_store"])
        if result["winner"] == "tie":
            draws += 1
        elif result["winner"] == result["ai_player"]:
            wins += 1
        else:
            losses += 1

    return {
        "model": model.path.name,
        "input": model.input_size,
        "hidden": "-".join(str(layer.units) for layer in model.layers[:-1]),
        "output_activation": model.layers[-1].activation,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / games,
        "ai_started": ai_started,
        "ai_as_p1": ai_as_p1,
        "avg_store_diff": score_diff_total / games,
    }


def main() -> None:
    model_paths = sorted(Path(".").glob("*.h5")) + sorted(Path(".").glob("*.keras"))
    seed = 20260526
    games = 10
    results = []
    for path in model_paths:
        try:
            spec = load_model_spec(path)
            results.append(evaluate_model(spec, games=games, seed=seed))
        except Exception as exc:
            results.append(
                {
                    "model": path.name,
                    "input": "ERR",
                    "hidden": "ERR",
                    "output_activation": "ERR",
                    "wins": "-",
                    "losses": "-",
                    "draws": "-",
                    "win_rate": math.nan,
                    "ai_started": "-",
                    "ai_as_p1": "-",
                    "avg_store_diff": f"ERROR: {type(exc).__name__}",
                }
            )

    results.sort(key=lambda row: (-(row["win_rate"] if isinstance(row["win_rate"], float) else -1), row["model"]))

    header = (
        f"{'Model':34} {'In':>3} {'Hidden':>9} {'Out':>8} "
        f"{'W':>2} {'L':>2} {'D':>2} {'Win%':>7} {'AI1st':>5} {'AsP1':>5} {'AvgDiff':>8}"
    )
    print(header)
    print("-" * len(header))
    for row in results:
        win_pct = f"{row['win_rate'] * 100:6.1f}%" if isinstance(row["win_rate"], float) else "   n/a"
        avg_diff = (
            f"{row['avg_store_diff']:8.2f}"
            if isinstance(row["avg_store_diff"], (int, float))
            else str(row["avg_store_diff"])
        )
        print(
            f"{row['model'][:34]:34} {str(row['input']):>3} {str(row['hidden']):>9} "
            f"{str(row['output_activation'])[:8]:>8} {str(row['wins']):>2} {str(row['losses']):>2} "
            f"{str(row['draws']):>2} {win_pct:>7} {str(row['ai_started']):>5} {str(row['ai_as_p1']):>5} {avg_diff:>8}"
        )


if __name__ == "__main__":
    main()
