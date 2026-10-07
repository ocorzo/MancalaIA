from __future__ import annotations

import argparse
import csv
import math
import random
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from mancala_dqn_selfplay import (
    ACTION_LABELS,
    Opponent,
    action_to_label,
    board_to_vector,
    build_eval_opponents,
    check_for_winner,
    get_new_board,
    legal_actions,
    load_model_spec,
    load_npz_opponents,
    make_move,
    opponent_move as base_opponent_move,
)
from benchmark_mancala_models import codigo3_move, codigo4_move, codigo5_move


def opponent_move(opponent: Opponent, player_turn: str, board: dict[str, int]) -> str:
    searches = {"codigo3": codigo3_move, "codigo4": codigo4_move, "codigo5": codigo5_move}
    if opponent.kind in searches:
        return searches[opponent.kind](player_turn, board)
    return base_opponent_move(opponent, player_turn, board)


ACTION_SIZE = len(ACTION_LABELS)
STATE_SIZE = 15


def softmax_masked(logits: np.ndarray, legal: list[int], temperature: float = 1.0) -> np.ndarray:
    probabilities = np.zeros(ACTION_SIZE, dtype=np.float64)
    if not legal:
        return probabilities
    temperature = max(float(temperature), 1e-6)
    legal_logits = logits[np.array(legal, dtype=np.int64)] / temperature
    legal_logits = legal_logits - np.max(legal_logits)
    exp = np.exp(legal_logits)
    total = float(np.sum(exp))
    if not math.isfinite(total) or total <= 0.0:
        probabilities[legal] = 1.0 / float(len(legal))
    else:
        probabilities[legal] = exp / total
    return probabilities


def visits_to_policy(visits: dict[int, int], temperature: float) -> np.ndarray:
    policy = np.zeros(ACTION_SIZE, dtype=np.float64)
    if not visits:
        return policy
    if temperature <= 1e-8:
        best_action = max(visits, key=lambda action: (visits[action], -action))
        policy[best_action] = 1.0
        return policy
    actions = list(visits)
    counts = np.array([visits[action] for action in actions], dtype=np.float64)
    counts = counts ** (1.0 / temperature)
    total = float(np.sum(counts))
    if not math.isfinite(total) or total <= 0.0:
        policy[actions] = 1.0 / float(len(actions))
    else:
        policy[actions] = counts / total
    return policy


def winner_value(winner: str, player_turn: str) -> float:
    if winner == "tie" or winner == "no winner":
        return 0.0
    return 1.0 if winner == player_turn else -1.0


def sample_action(policy: np.ndarray, rng: random.Random) -> int:
    total = float(np.sum(policy))
    if total <= 0.0:
        legal = np.flatnonzero(policy >= 0.0).tolist()
        return rng.choice(legal)
    threshold = rng.random() * total
    running = 0.0
    for action, probability in enumerate(policy):
        running += float(probability)
        if running >= threshold:
            return action
    return int(np.argmax(policy))


@dataclass
class AlphaZeroExample:
    state: np.ndarray
    policy: np.ndarray
    player_turn: str
    value: float = 0.0


class AlphaZeroNetwork:
    def __init__(self, input_size: int = STATE_SIZE, hidden1: int = 128, hidden2: int = 128) -> None:
        self.w1 = np.random.randn(input_size, hidden1) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros(hidden1)
        self.w2 = np.random.randn(hidden1, hidden2) * np.sqrt(2.0 / hidden1)
        self.b2 = np.zeros(hidden2)
        self.wp = np.random.randn(hidden2, ACTION_SIZE) * np.sqrt(2.0 / hidden2)
        self.bp = np.zeros(ACTION_SIZE)
        self.wv = np.random.randn(hidden2, 1) * np.sqrt(2.0 / hidden2)
        self.bv = np.zeros(1)
        self._init_optimizer_state()

    def _parameter_names(self) -> tuple[str, ...]:
        return ("w1", "b1", "w2", "b2", "wp", "bp", "wv", "bv")

    def _init_optimizer_state(self) -> None:
        for name in self._parameter_names():
            setattr(self, f"m_{name}", np.zeros_like(getattr(self, name)))
            setattr(self, f"v_{name}", np.zeros_like(getattr(self, name)))
        self.optimizer_step = 0

    def clone(self) -> "AlphaZeroNetwork":
        cloned = AlphaZeroNetwork(self.w1.shape[0], self.w1.shape[1], self.w2.shape[1])
        cloned.copy_from(self)
        return cloned

    def copy_from(self, other: "AlphaZeroNetwork") -> None:
        for name in self._parameter_names():
            setattr(self, name, getattr(other, name).copy())
            setattr(self, f"m_{name}", getattr(other, f"m_{name}").copy())
            setattr(self, f"v_{name}", getattr(other, f"v_{name}").copy())
        self.optimizer_step = other.optimizer_step

    def _forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        if x.ndim == 1:
            x = x.reshape(1, -1)
        z1 = x @ self.w1 + self.b1
        a1 = np.maximum(z1, 0.0)
        z2 = a1 @ self.w2 + self.b2
        a2 = np.maximum(z2, 0.0)
        logits = a2 @ self.wp + self.bp
        value_raw = a2 @ self.wv + self.bv
        value = np.tanh(value_raw)
        cache = {"x": x, "z1": z1, "a1": a1, "z2": z2, "a2": a2, "value_raw": value_raw}
        return logits, value, cache

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        logits, value, _ = self._forward(x.astype(np.float64, copy=False))
        return logits, value.reshape(-1)

    def _apply_adam(
        self,
        param_name: str,
        grad: np.ndarray,
        lr: float,
        beta1: float,
        beta2: float,
        adam_eps: float,
    ) -> None:
        param = getattr(self, param_name)
        m_name = f"m_{param_name}"
        v_name = f"v_{param_name}"
        m = beta1 * getattr(self, m_name) + (1.0 - beta1) * grad
        v = beta2 * getattr(self, v_name) + (1.0 - beta2) * (grad * grad)
        m_hat = m / (1.0 - beta1 ** self.optimizer_step)
        v_hat = v / (1.0 - beta2 ** self.optimizer_step)
        param -= lr * m_hat / (np.sqrt(v_hat) + adam_eps)
        setattr(self, param_name, param)
        setattr(self, m_name, m)
        setattr(self, v_name, v)

    def train_batch(
        self,
        states: np.ndarray,
        target_policies: np.ndarray,
        target_values: np.ndarray,
        lr: float,
        l2: float = 1e-4,
        clip_norm: float = 5.0,
        beta1: float = 0.9,
        beta2: float = 0.999,
        adam_eps: float = 1e-8,
    ) -> dict[str, float]:
        logits, values, cache = self._forward(states.astype(np.float64, copy=False))
        batch_size = states.shape[0]

        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp = np.exp(shifted)
        pred_policies = exp / np.sum(exp, axis=1, keepdims=True)
        target_values = target_values.reshape(-1, 1).astype(np.float64, copy=False)

        policy_loss = -float(np.mean(np.sum(target_policies * np.log(pred_policies + 1e-12), axis=1)))
        value_error = values - target_values
        value_loss = float(np.mean(value_error * value_error))
        l2_loss = 0.5 * l2 * float(sum(np.sum(getattr(self, name) ** 2) for name in ("w1", "w2", "wp", "wv")))
        loss = policy_loss + value_loss + l2_loss

        dlogits = (pred_policies - target_policies) / float(batch_size)
        dvalue = (2.0 * value_error / float(batch_size)) * (1.0 - values * values)

        a2 = cache["a2"]
        dwp = a2.T @ dlogits + l2 * self.wp
        dbp = np.sum(dlogits, axis=0)
        dwv = a2.T @ dvalue + l2 * self.wv
        dbv = np.sum(dvalue, axis=0)

        da2 = dlogits @ self.wp.T + dvalue @ self.wv.T
        dz2 = da2 * (cache["z2"] > 0.0)
        dw2 = cache["a1"].T @ dz2 + l2 * self.w2
        db2 = np.sum(dz2, axis=0)
        da1 = dz2 @ self.w2.T
        dz1 = da1 * (cache["z1"] > 0.0)
        dw1 = cache["x"].T @ dz1 + l2 * self.w1
        db1 = np.sum(dz1, axis=0)

        grads = [dw1, db1, dw2, db2, dwp, dbp, dwv, dbv]
        global_norm = float(np.sqrt(sum(np.sum(grad * grad) for grad in grads)))
        if global_norm > clip_norm > 0.0:
            scale = clip_norm / (global_norm + 1e-12)
            grads = [grad * scale for grad in grads]
            dw1, db1, dw2, db2, dwp, dbp, dwv, dbv = grads

        self.optimizer_step += 1
        for name, grad in zip(self._parameter_names(), (dw1, db1, dw2, db2, dwp, dbp, dwv, dbv)):
            self._apply_adam(name, grad, lr, beta1, beta2, adam_eps)

        return {"loss": loss, "policy_loss": policy_loss, "value_loss": value_loss, "l2_loss": l2_loss}

    def save(self, path: Path, metadata: dict[str, Any] | None = None) -> None:
        payload = {name: getattr(self, name) for name in self._parameter_names()}
        for name in self._parameter_names():
            payload[f"m_{name}"] = getattr(self, f"m_{name}")
            payload[f"v_{name}"] = getattr(self, f"v_{name}")
        payload["optimizer_step"] = np.array([self.optimizer_step], dtype=np.int64)
        if metadata:
            payload["metadata"] = np.array([repr(metadata)], dtype=object)
        np.savez(path, **payload)

    def load(self, path: Path) -> None:
        data = np.load(path, allow_pickle=True)
        for name in self._parameter_names():
            setattr(self, name, data[name].copy())
        self._init_optimizer_state()
        for name in self._parameter_names():
            m_name = f"m_{name}"
            v_name = f"v_{name}"
            if m_name in data:
                setattr(self, m_name, data[m_name].copy())
            if v_name in data:
                setattr(self, v_name, data[v_name].copy())
        if "optimizer_step" in data:
            self.optimizer_step = int(data["optimizer_step"][0])


@dataclass
class AZNode:
    board: dict[str, int]
    player_turn: str
    parent: "AZNode | None" = None
    action: int | None = None
    prior: float = 0.0
    visits: int = 0
    value_sum: float = 0.0
    children: dict[int, "AZNode"] = field(default_factory=dict)
    terminal_value: float | None = None

    @property
    def mean_value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


@dataclass(frozen=True)
class AZSearchResult:
    action: int
    policy: np.ndarray
    visits: dict[int, int]
    values: dict[int, float]


class AlphaZeroMCTS:
    def __init__(
        self,
        network: AlphaZeroNetwork,
        simulations: int = 100,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.35,
        dirichlet_epsilon: float = 0.25,
        rng: random.Random | None = None,
    ) -> None:
        if simulations <= 0:
            raise ValueError("simulations debe ser mayor que cero")
        self.network = network
        self.simulations = int(simulations)
        self.c_puct = float(c_puct)
        self.dirichlet_alpha = float(dirichlet_alpha)
        self.dirichlet_epsilon = float(dirichlet_epsilon)
        self.rng = rng or random.Random()

    def _terminal(self, board: dict[str, int], player_turn: str) -> float | None:
        checked = board.copy()
        winner = check_for_winner(checked)
        if winner == "no winner":
            return None
        board.update(checked)
        return winner_value(winner, player_turn)

    def _network_policy_value(self, board: dict[str, int], player_turn: str) -> tuple[np.ndarray, float]:
        logits, values = self.network.predict(board_to_vector(board, player_turn))
        legal = legal_actions(player_turn, board)
        return softmax_masked(logits[0], legal), float(values[0])

    def _expand(self, node: AZNode, add_root_noise: bool = False) -> float:
        terminal_value = self._terminal(node.board, node.player_turn)
        if terminal_value is not None:
            node.terminal_value = terminal_value
            return terminal_value

        priors, value = self._network_policy_value(node.board, node.player_turn)
        actions = legal_actions(node.player_turn, node.board)
        if add_root_noise and actions and self.dirichlet_epsilon > 0.0:
            noise = np.random.default_rng(self.rng.randrange(2**32)).dirichlet(
                np.full(len(actions), self.dirichlet_alpha, dtype=np.float64)
            )
            for idx, action in enumerate(actions):
                priors[action] = (1.0 - self.dirichlet_epsilon) * priors[action] + self.dirichlet_epsilon * noise[idx]

        for action in actions:
            child_board = node.board.copy()
            next_player = make_move(child_board, node.player_turn, action_to_label(action))
            child_terminal = self._terminal(child_board, next_player)
            node.children[action] = AZNode(
                board=child_board,
                player_turn=next_player,
                parent=node,
                action=action,
                prior=float(priors[action]),
                terminal_value=child_terminal,
            )
        return value

    def _child_value_for_parent(self, parent: AZNode, child: AZNode) -> float:
        value = child.mean_value
        return value if child.player_turn == parent.player_turn else -value

    def _select_child(self, node: AZNode) -> AZNode:
        parent_visits = max(1, node.visits)

        def score(child: AZNode) -> tuple[float, float, int]:
            q_value = self._child_value_for_parent(node, child)
            exploration = self.c_puct * child.prior * math.sqrt(parent_visits) / (1 + child.visits)
            return q_value + exploration, q_value, -int(child.action or 0)

        return max(node.children.values(), key=score)

    def _backpropagate(self, node: AZNode, value: float) -> None:
        current: AZNode | None = node
        current_value = value
        while current is not None:
            current.visits += 1
            current.value_sum += current_value
            parent = current.parent
            if parent is not None and parent.player_turn != current.player_turn:
                current_value = -current_value
            current = parent

    def search(
        self,
        board: dict[str, int],
        player_turn: str,
        temperature: float = 1.0,
        add_root_noise: bool = False,
        sample: bool = True,
    ) -> AZSearchResult:
        root = AZNode(board=board.copy(), player_turn=player_turn)
        root_value = self._expand(root, add_root_noise=add_root_noise)
        self._backpropagate(root, root_value)
        if not root.children:
            raise ValueError("No hay jugadas legales para iniciar MCTS")

        for _ in range(self.simulations):
            node = root
            while node.children and node.terminal_value is None:
                node = self._select_child(node)
            if node.terminal_value is not None:
                value = node.terminal_value
            else:
                value = self._expand(node)
            self._backpropagate(node, value)

        visits = {action: child.visits for action, child in root.children.items()}
        values = {action: self._child_value_for_parent(root, child) for action, child in root.children.items()}
        policy = visits_to_policy(visits, temperature)
        if sample:
            action = sample_action(policy, self.rng)
        else:
            action = max(visits, key=lambda candidate: (visits[candidate], values[candidate], -candidate))
        return AZSearchResult(action=action, policy=policy, visits=visits, values=values)


class AlphaZeroAgent:
    def __init__(self, network: AlphaZeroNetwork) -> None:
        self.network = network

    def act(self, state: np.ndarray, legal: list[int], explore: bool = False) -> int:
        logits, _ = self.network.predict(state)
        scores = logits[0]
        best_score = max(scores[action] for action in legal)
        best_actions = [action for action in legal if scores[action] == best_score]
        return random.choice(best_actions)


def play_self_play_game(
    network: AlphaZeroNetwork,
    simulations: int,
    rng: random.Random,
    temperature_moves: int,
    c_puct: float,
    max_moves: int,
) -> list[AlphaZeroExample]:
    board = get_new_board()
    player_turn = rng.choice(["1", "2"])
    examples: list[AlphaZeroExample] = []

    for move_index in range(max_moves):
        legal = legal_actions(player_turn, board)
        if not legal:
            break
        temperature = 1.0 if move_index < temperature_moves else 0.0
        mcts = AlphaZeroMCTS(
            network=network,
            simulations=simulations,
            c_puct=c_puct,
            rng=random.Random(rng.randrange(2**63)),
        )
        result = mcts.search(
            board,
            player_turn,
            temperature=temperature,
            add_root_noise=True,
            sample=True,
        )
        examples.append(
            AlphaZeroExample(
                state=board_to_vector(board, player_turn),
                policy=result.policy.copy(),
                player_turn=player_turn,
            )
        )
        player_turn = make_move(board, player_turn, action_to_label(result.action))
        winner = check_for_winner(board)
        if winner != "no winner":
            for example in examples:
                example.value = winner_value(winner, example.player_turn)
            return examples

    material_winner = "tie"
    if board["1"] > board["2"]:
        material_winner = "1"
    elif board["2"] > board["1"]:
        material_winner = "2"
    for example in examples:
        example.value = winner_value(material_winner, example.player_turn)
    return examples


def train_from_replay(
    network: AlphaZeroNetwork,
    replay: deque[AlphaZeroExample],
    steps: int,
    batch_size: int,
    lr: float,
    rng: random.Random,
) -> dict[str, float]:
    if not replay:
        return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "l2_loss": 0.0}
    losses: list[dict[str, float]] = []
    batch_size = min(batch_size, len(replay))
    replay_list = list(replay)
    for _ in range(max(1, steps)):
        batch = rng.sample(replay_list, batch_size)
        states = np.vstack([example.state for example in batch])
        policies = np.vstack([example.policy for example in batch])
        values = np.array([example.value for example in batch], dtype=np.float64)
        losses.append(network.train_batch(states, policies, values, lr=lr))
    return {key: float(np.mean([loss[key] for loss in losses])) for key in losses[0]}


def play_vs_opponent_policy(
    network: AlphaZeroNetwork,
    opponent: Opponent,
    games: int,
    use_mcts: bool,
    simulations: int,
    seed: int,
) -> tuple[int, int, int]:
    wins = losses = draws = 0
    rng = random.Random(seed)
    role_cycle = (("1", "1"), ("1", "2"), ("2", "1"), ("2", "2"))

    for game_index in range(games):
        board = get_new_board()
        ai_player, player_turn = role_cycle[game_index % len(role_cycle)]
        while True:
            if player_turn == ai_player:
                if use_mcts:
                    mcts = AlphaZeroMCTS(
                        network=network,
                        simulations=simulations,
                        rng=random.Random(rng.randrange(2**63)),
                    )
                    action = mcts.search(board, player_turn, temperature=0.0, sample=False).action
                else:
                    logits, _ = network.predict(board_to_vector(board, player_turn))
                    legal = legal_actions(player_turn, board)
                    best = max(logits[0][action] for action in legal)
                    candidates = [action for action in legal if logits[0][action] == best]
                    action = candidates[0]
                move = action_to_label(action)
            else:
                move = opponent_move(opponent, player_turn, board)
            player_turn = make_move(board, player_turn, move)
            winner = check_for_winner(board)
            if winner != "no winner":
                if winner == "tie":
                    draws += 1
                elif winner == ai_player:
                    wins += 1
                else:
                    losses += 1
                break
    return wins, losses, draws


def evaluate_alphazero(
    network: AlphaZeroNetwork,
    opponents: list[Opponent],
    games: int,
    use_mcts: bool,
    simulations: int,
    seed: int,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    scores = []
    for index, opponent in enumerate(opponents):
        wins, losses, draws = play_vs_opponent_policy(
            network,
            opponent,
            games=games,
            use_mcts=use_mcts,
            simulations=simulations,
            seed=seed + index * 10_000,
        )
        score = (3.0 * wins + draws) / (3.0 * games)
        scores.append(score)
        prefix = opponent.name
        results[f"{prefix}_wins"] = wins
        results[f"{prefix}_losses"] = losses
        results[f"{prefix}_draws"] = draws
        results[f"{prefix}_win_rate"] = wins / float(games)
        results[f"{prefix}_score"] = score
    results["mixed_score"] = float(np.mean(scores)) if scores else 0.0
    results["worst_case_score"] = float(np.min(scores)) if scores else 0.0
    return results


def append_history(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    fieldnames = list(row)
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def run_training(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    rng = random.Random(args.seed)
    network = AlphaZeroNetwork(hidden1=args.hidden_size, hidden2=args.hidden_size)
    if args.load_model and args.load_model.exists():
        network.load(args.load_model)

    fixed_dqn_models = load_npz_opponents(args.fixed_dqn_model)
    phase2_model = None
    if args.phase2_model and args.phase2_model.exists():
        try:
            phase2_model = load_model_spec(args.phase2_model)
        except ModuleNotFoundError as exc:
            print(f"Aviso: omito {args.phase2_model} porque falta dependencia opcional: {exc}")
    eval_opponents = build_eval_opponents(
        fixed_dqn_models=fixed_dqn_models,
        phase2_model=phase2_model,
        include_azar=True,
        include_codigo=True,
    )
    if not eval_opponents:
        eval_opponents = [Opponent(name="azar", kind="azar")]
    if args.eval_opponents:
        eval_opponents = [Opponent(name=name, kind=name) for name in args.eval_opponents]

    replay: deque[AlphaZeroExample] = deque(maxlen=args.replay_size)
    best_mcts_score = -1.0
    best_policy_score = -1.0

    if args.load_model:
        baseline = evaluate_alphazero(
            network, eval_opponents, args.eval_games, True,
            args.eval_simulations, args.seed,
        )
        best_mcts_score = float(baseline["mixed_score"])
        network.save(args.best_mcts_model, metadata={"iteration": 0, **baseline})
        append_history(args.history, {
            "iteration": 0, "generated_examples": 0, "replay_size": 0,
            **{f"train_{key}": math.nan for key in ("loss", "policy_loss", "value_loss", "l2_loss")},
            "policy_mixed_score": math.nan, "policy_worst_case_score": math.nan,
            "mcts_mixed_score": baseline["mixed_score"],
            "mcts_worst_case_score": baseline["worst_case_score"],
            "policy_azar_win_rate": math.nan,
            "mcts_azar_win_rate": baseline.get("azar_win_rate", math.nan),
            **{f"mcts_{key}": value for key, value in baseline.items()
               if key not in ("mixed_score", "worst_case_score", "azar_win_rate")},
        })
        print(f"Baseline MCTS: {baseline}", flush=True)

    print("Entrenamiento AlphaZero Mancala")
    print(
        f"iteraciones={args.iterations} self_play_games={args.self_play_games} "
        f"simulaciones={args.simulations} replay={args.replay_size}"
    )

    for iteration in range(1, args.iterations + 1):
        generated = 0
        for _ in range(args.self_play_games):
            examples = play_self_play_game(
                network=network,
                simulations=args.simulations,
                rng=rng,
                temperature_moves=args.temperature_moves,
                c_puct=args.c_puct,
                max_moves=args.max_moves,
            )
            replay.extend(examples)
            generated += len(examples)

        losses = train_from_replay(
            network=network,
            replay=replay,
            steps=args.train_steps,
            batch_size=args.batch_size,
            lr=args.lr,
            rng=rng,
        )

        should_evaluate = iteration % args.eval_every == 0 or iteration == args.iterations
        if should_evaluate:
            policy_eval = evaluate_alphazero(
                network,
                opponents=eval_opponents,
                games=args.eval_games,
                use_mcts=False,
                simulations=args.eval_simulations,
                seed=args.seed + iteration * 1000,
            )
            mcts_eval = evaluate_alphazero(
                network,
                opponents=eval_opponents,
                games=args.eval_games,
                use_mcts=True,
                simulations=args.eval_simulations,
                seed=args.seed + iteration * 2000,
            )
        else:
            policy_eval = {"mixed_score": math.nan, "worst_case_score": math.nan, "azar_win_rate": math.nan}
            mcts_eval = {"mixed_score": math.nan, "worst_case_score": math.nan, "azar_win_rate": math.nan}

        row = {
            "iteration": iteration,
            "generated_examples": generated,
            "replay_size": len(replay),
            **{f"train_{key}": value for key, value in losses.items()},
            "policy_mixed_score": policy_eval["mixed_score"],
            "policy_worst_case_score": policy_eval["worst_case_score"],
            "mcts_mixed_score": mcts_eval["mixed_score"],
            "mcts_worst_case_score": mcts_eval["worst_case_score"],
            "policy_azar_win_rate": policy_eval.get("azar_win_rate", 0.0),
            "mcts_azar_win_rate": mcts_eval.get("azar_win_rate", 0.0),
        }
        for opponent in eval_opponents:
            for metric in ("wins", "losses", "draws", "win_rate", "score"):
                key = f"{opponent.name}_{metric}"
                if key != "azar_win_rate":
                    row[f"mcts_{key}"] = mcts_eval.get(key, math.nan)
        append_history(args.history, row)

        print(
            f"Iter {iteration:4d} | nuevos {generated:5d} | replay {len(replay):6d} | "
            f"loss {losses['loss']:.4f} | "
            f"policy mixed {policy_eval['mixed_score']:.2%} | "
            f"MCTS mixed {mcts_eval['mixed_score']:.2%} | "
            f"Azar policy {policy_eval.get('azar_win_rate', math.nan):.1%} | "
            f"Azar MCTS {mcts_eval.get('azar_win_rate', math.nan):.1%}"
        )

        if should_evaluate and mcts_eval["mixed_score"] > best_mcts_score:
            best_mcts_score = float(mcts_eval["mixed_score"])
            network.save(
                args.best_mcts_model,
                metadata={"iteration": iteration, "mcts_mixed_score": best_mcts_score, "kind": "alphazero_mcts_best"},
            )
        if should_evaluate and policy_eval["mixed_score"] > best_policy_score:
            best_policy_score = float(policy_eval["mixed_score"])
            network.save(
                args.best_policy_model,
                metadata={"iteration": iteration, "policy_mixed_score": best_policy_score, "kind": "alphazero_policy_best"},
            )
        if args.checkpoint_dir and iteration % args.checkpoint_every == 0:
            args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            network.save(
                args.checkpoint_dir / f"alphazero_checkpoint_iter{iteration}.npz",
                metadata={"iteration": iteration, "kind": "alphazero_checkpoint"},
            )

    network.save(args.output_model, metadata={"iterations": args.iterations, "kind": "alphazero_final"})
    print(f"Modelo final guardado en {args.output_model}")
    print(f"Mejor policy guardado en {args.best_policy_model} con mixed={best_policy_score:.2%}")
    print(f"Mejor MCTS guardado en {args.best_mcts_model} con mixed={best_mcts_score:.2%}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Entrenamiento estilo AlphaZero para Mancala.")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--self-play-games", type=int, default=20)
    parser.add_argument("--simulations", type=int, default=50)
    parser.add_argument("--eval-simulations", type=int, default=50)
    parser.add_argument("--train-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--replay-size", type=int, default=100000)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--temperature-moves", type=int, default=10)
    parser.add_argument("--max-moves", type=int, default=250)
    parser.add_argument("--eval-games", type=int, default=40)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--load-model", type=Path)
    parser.add_argument("--eval-opponents", nargs="+", choices=("azar", "codigo", "codigo3", "codigo4", "codigo5"))
    parser.add_argument("--output-model", type=Path, default=Path("alphazero_mancala_final.npz"))
    parser.add_argument("--best-policy-model", type=Path, default=Path("alphazero_mancala_best_policy.npz"))
    parser.add_argument("--best-mcts-model", type=Path, default=Path("alphazero_mancala_best_mcts.npz"))
    parser.add_argument("--history", type=Path, default=Path("alphazero_mancala_history.csv"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("alphazero_mancala_checkpoints"))
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--phase2-model", type=Path, default=Path("modelo_mancala_4000.h5"))
    parser.add_argument(
        "--fixed-dqn-model",
        type=Path,
        action="append",
        default=[Path("dqn_best_remote_30000_epsmin02.npz")],
    )
    return parser


def main() -> None:
    run_training(build_parser().parse_args())


if __name__ == "__main__":
    main()
