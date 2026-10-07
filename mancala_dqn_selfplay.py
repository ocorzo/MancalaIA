from __future__ import annotations

import argparse
import csv
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from benchmark_mancala_models import (
    NEXT_PIT,
    OPPOSITE_PIT,
    PIT_LABELS,
    PLAYER_1_PITS,
    PLAYER_2_PITS,
    azar_move,
    codigo3_move as benchmark_codigo3_move,
    codigo4_move as benchmark_codigo4_move,
    codigo5_move as benchmark_codigo5_move,
    codigo_move as benchmark_codigo2_move,
    get_new_board,
    load_model_spec,
    model_move,
)


ACTION_LABELS = tuple("ABCDEFGHIJKL")
PLAYER_ACTIONS = {
    "1": tuple(range(0, 6)),
    "2": tuple(range(6, 12)),
}


def legal_actions(player_turn: str, board: dict[str, int]) -> list[int]:
    pits = PLAYER_1_PITS if player_turn == "1" else PLAYER_2_PITS
    offset = 0 if player_turn == "1" else 6
    return [offset + idx for idx, pit in enumerate(pits) if board[pit] > 0]


def action_to_label(action_index: int) -> str:
    return ACTION_LABELS[action_index]


def board_to_vector(board: dict[str, int], player_turn: str) -> np.ndarray:
    return np.array([float(int(player_turn))] + [float(board[p]) for p in PIT_LABELS], dtype=np.float64)


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


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    next_legal_actions: list[int]
    n_steps: int = 1


@dataclass
class PendingTransition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    next_legal_actions: list[int]


class NumpyMLP:
    def __init__(self, input_size: int = 15, hidden1: int = 64, hidden2: int = 64, output_size: int = 12) -> None:
        self.w1 = np.random.randn(input_size, hidden1) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros(hidden1)
        self.w2 = np.random.randn(hidden1, hidden2) * np.sqrt(2.0 / hidden1)
        self.b2 = np.zeros(hidden2)
        self.w3 = np.random.randn(hidden2, output_size) * np.sqrt(2.0 / hidden2)
        self.b3 = np.zeros(output_size)
        self._init_optimizer_state()

    def _init_optimizer_state(self) -> None:
        self.m_w1 = np.zeros_like(self.w1)
        self.m_b1 = np.zeros_like(self.b1)
        self.m_w2 = np.zeros_like(self.w2)
        self.m_b2 = np.zeros_like(self.b2)
        self.m_w3 = np.zeros_like(self.w3)
        self.m_b3 = np.zeros_like(self.b3)
        self.v_w1 = np.zeros_like(self.w1)
        self.v_b1 = np.zeros_like(self.b1)
        self.v_w2 = np.zeros_like(self.w2)
        self.v_b2 = np.zeros_like(self.b2)
        self.v_w3 = np.zeros_like(self.w3)
        self.v_b3 = np.zeros_like(self.b3)
        self.optimizer_step = 0

    def copy_from(self, other: "NumpyMLP") -> None:
        self.w1 = other.w1.copy()
        self.b1 = other.b1.copy()
        self.w2 = other.w2.copy()
        self.b2 = other.b2.copy()
        self.w3 = other.w3.copy()
        self.b3 = other.b3.copy()
        self.m_w1 = other.m_w1.copy()
        self.m_b1 = other.m_b1.copy()
        self.m_w2 = other.m_w2.copy()
        self.m_b2 = other.m_b2.copy()
        self.m_w3 = other.m_w3.copy()
        self.m_b3 = other.m_b3.copy()
        self.v_w1 = other.v_w1.copy()
        self.v_b1 = other.v_b1.copy()
        self.v_w2 = other.v_w2.copy()
        self.v_b2 = other.v_b2.copy()
        self.v_w3 = other.v_w3.copy()
        self.v_b3 = other.v_b3.copy()
        self.optimizer_step = other.optimizer_step

    def load_from_npz(self, path: Path) -> None:
        data = np.load(path, allow_pickle=True)
        self.w1 = data["w1"].copy()
        self.b1 = data["b1"].copy()
        self.w2 = data["w2"].copy()
        self.b2 = data["b2"].copy()
        self.w3 = data["w3"].copy()
        self.b3 = data["b3"].copy()
        self._init_optimizer_state()
        for name in (
            "m_w1",
            "m_b1",
            "m_w2",
            "m_b2",
            "m_w3",
            "m_b3",
            "v_w1",
            "v_b1",
            "v_w2",
            "v_b2",
            "v_w3",
            "v_b3",
        ):
            if name in data:
                setattr(self, name, data[name].copy())
        if "optimizer_step" in data:
            self.optimizer_step = int(data["optimizer_step"][0])

    def clone(self) -> "NumpyMLP":
        cloned = NumpyMLP(
            input_size=self.w1.shape[0],
            hidden1=self.w1.shape[1],
            hidden2=self.w2.shape[1],
            output_size=self.w3.shape[1],
        )
        cloned.copy_from(self)
        return cloned

    def soft_update_from(self, other: "NumpyMLP", tau: float) -> None:
        self.w1 = (1.0 - tau) * self.w1 + tau * other.w1
        self.b1 = (1.0 - tau) * self.b1 + tau * other.b1
        self.w2 = (1.0 - tau) * self.w2 + tau * other.w2
        self.b2 = (1.0 - tau) * self.b2 + tau * other.b2
        self.w3 = (1.0 - tau) * self.w3 + tau * other.w3
        self.b3 = (1.0 - tau) * self.b3 + tau * other.b3

    def predict(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            x = x.reshape(1, -1)
        z1 = x @ self.w1 + self.b1
        a1 = np.maximum(z1, 0.0)
        z2 = a1 @ self.w2 + self.b2
        a2 = np.maximum(z2, 0.0)
        return a2 @ self.w3 + self.b3

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
        m = getattr(self, m_name)
        v = getattr(self, v_name)
        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * (grad * grad)
        m_hat = m / (1.0 - (beta1 ** self.optimizer_step))
        v_hat = v / (1.0 - (beta2 ** self.optimizer_step))
        param -= lr * m_hat / (np.sqrt(v_hat) + adam_eps)
        setattr(self, param_name, param)
        setattr(self, m_name, m)
        setattr(self, v_name, v)

    def train_batch(
        self,
        x: np.ndarray,
        actions: np.ndarray,
        target_values: np.ndarray,
        lr: float,
        huber_delta: float = 1.0,
        clip_norm: float = 5.0,
        beta1: float = 0.9,
        beta2: float = 0.999,
        adam_eps: float = 1e-8,
    ) -> float:
        if x.ndim == 1:
            x = x.reshape(1, -1)

        z1 = x @ self.w1 + self.b1
        a1 = np.maximum(z1, 0.0)
        z2 = a1 @ self.w2 + self.b2
        a2 = np.maximum(z2, 0.0)
        preds = a2 @ self.w3 + self.b3

        row_indexes = np.arange(x.shape[0])
        chosen_q = preds[row_indexes, actions]
        td_error = chosen_q - target_values
        abs_error = np.abs(td_error)
        quadratic = np.minimum(abs_error, huber_delta)
        linear = abs_error - quadratic
        loss = float(np.mean(0.5 * quadratic * quadratic + huber_delta * linear))

        dchosen = np.where(abs_error <= huber_delta, td_error, huber_delta * np.sign(td_error))
        dchosen /= float(x.shape[0])
        dpreds = np.zeros_like(preds)
        dpreds[row_indexes, actions] = dchosen

        dw3 = a2.T @ dpreds
        db3 = np.sum(dpreds, axis=0)

        da2 = dpreds @ self.w3.T
        dz2 = da2 * (z2 > 0)
        dw2 = a1.T @ dz2
        db2 = np.sum(dz2, axis=0)

        da1 = dz2 @ self.w2.T
        dz1 = da1 * (z1 > 0)
        dw1 = x.T @ dz1
        db1 = np.sum(dz1, axis=0)

        grads = [dw1, db1, dw2, db2, dw3, db3]
        global_norm = float(np.sqrt(sum(np.sum(grad * grad) for grad in grads)))
        if global_norm > clip_norm > 0.0:
            scale = clip_norm / (global_norm + 1e-12)
            grads = [grad * scale for grad in grads]
            dw1, db1, dw2, db2, dw3, db3 = grads

        self.optimizer_step += 1
        self._apply_adam("w1", dw1, lr, beta1, beta2, adam_eps)
        self._apply_adam("b1", db1, lr, beta1, beta2, adam_eps)
        self._apply_adam("w2", dw2, lr, beta1, beta2, adam_eps)
        self._apply_adam("b2", db2, lr, beta1, beta2, adam_eps)
        self._apply_adam("w3", dw3, lr, beta1, beta2, adam_eps)
        self._apply_adam("b3", db3, lr, beta1, beta2, adam_eps)
        return loss

    def save(self, path: Path, metadata: dict[str, float | int | str] | None = None) -> None:
        payload = {
            "w1": self.w1,
            "b1": self.b1,
            "w2": self.w2,
            "b2": self.b2,
            "w3": self.w3,
            "b3": self.b3,
            "m_w1": self.m_w1,
            "m_b1": self.m_b1,
            "m_w2": self.m_w2,
            "m_b2": self.m_b2,
            "m_w3": self.m_w3,
            "m_b3": self.m_b3,
            "v_w1": self.v_w1,
            "v_b1": self.v_b1,
            "v_w2": self.v_w2,
            "v_b2": self.v_b2,
            "v_w3": self.v_w3,
            "v_b3": self.v_b3,
            "optimizer_step": np.array([self.optimizer_step], dtype=np.int64),
        }
        if metadata:
            payload["metadata"] = np.array([repr(metadata)], dtype=object)
        np.savez(path, **payload)


class NPZModel:
    def __init__(self, path: Path) -> None:
        data = np.load(path, allow_pickle=True)
        self.w1 = data["w1"]
        self.b1 = data["b1"]
        self.w2 = data["w2"]
        self.b2 = data["b2"]
        self.w3 = data["w3"]
        self.b3 = data["b3"]

    def predict(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            x = x.reshape(1, -1)
        z1 = x @ self.w1 + self.b1
        a1 = np.maximum(z1, 0.0)
        z2 = a1 @ self.w2 + self.b2
        a2 = np.maximum(z2, 0.0)
        return a2 @ self.w3 + self.b3


@dataclass
class Opponent:
    name: str
    kind: str
    payload: object | None = None
    weight: float = 1.0
    noise: float = 0.0


@dataclass
class EliteCheckpoint:
    score: float
    path: Path
    episode: int


class DQNAgent:
    def __init__(
        self,
        gamma: float = 0.98,
        epsilon: float = 0.02,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.9995,
        epsilon_decay_episodes: int = 100000,
        lr: float = 0.0003,
        batch_size: int = 128,
        memory_size: int = 600000,
        anchor_memory_size: int = 200000,
        anchor_batch_ratio: float = 0.25,
        n_step: int = 3,
        target_sync_every: int = 1000,
        target_update_mode: str = "hard",
        target_tau: float = 0.005,
        huber_delta: float = 1.0,
        grad_clip_norm: float = 5.0,
    ) -> None:
        self.online = NumpyMLP()
        self.target = NumpyMLP()
        self.target.copy_from(self.online)
        self.gamma = gamma
        self.epsilon_start = epsilon
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.epsilon_decay_episodes = max(1, epsilon_decay_episodes)
        self.lr = lr
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.memory: deque[Transition] = deque(maxlen=memory_size)
        self.anchor_memory_size = anchor_memory_size
        self.anchor_memory: deque[Transition] = deque(maxlen=anchor_memory_size) if anchor_memory_size > 0 else deque()
        self.anchor_batch_ratio = anchor_batch_ratio
        self.n_step = max(1, n_step)
        self.target_sync_every = target_sync_every
        self.target_update_mode = target_update_mode
        self.target_tau = target_tau
        self.huber_delta = huber_delta
        self.grad_clip_norm = grad_clip_norm
        self.training_steps = 0

    def load_model(self, path: Path) -> None:
        self.online.load_from_npz(path)
        self.target.copy_from(self.online)

    def act(self, state: np.ndarray, legal: list[int], explore: bool = True) -> int:
        if explore and random.random() < self.epsilon:
            return random.choice(legal)
        q_values = self.online.predict(state)[0]
        legal_scores = [(action, q_values[action]) for action in legal]
        best_score = max(score for _, score in legal_scores)
        best_actions = [action for action, score in legal_scores if score == best_score]
        return random.choice(best_actions)

    def set_episode_epsilon(self, episode_index: int) -> None:
        progress = min(1.0, max(0.0, float(episode_index) / float(self.epsilon_decay_episodes)))
        linear_value = self.epsilon_start + (self.epsilon_min - self.epsilon_start) * progress
        if self.epsilon_decay > 0.0:
            exp_value = self.epsilon_start * (self.epsilon_decay ** episode_index)
            self.epsilon = max(self.epsilon_min, min(linear_value, exp_value))
        else:
            self.epsilon = max(self.epsilon_min, linear_value)

    def remember(self, transition: Transition, to_anchor: bool = False) -> None:
        self.memory.append(transition)
        if to_anchor and self.anchor_memory_size > 0 and len(self.anchor_memory) < self.anchor_memory_size:
            self.anchor_memory.append(transition)

    def _serialize_buffer(self, buffer_name: str) -> deque[Transition]:
        if buffer_name == "anchor":
            return self.anchor_memory
        return self.memory

    def save_replay_buffer(self, path: Path, buffer_name: str = "recent") -> None:
        buffer = self._serialize_buffer(buffer_name)
        states = np.stack([item.state for item in buffer]) if buffer else np.zeros((0, 15), dtype=np.float64)
        next_states = (
            np.stack([item.next_state for item in buffer]) if buffer else np.zeros((0, 15), dtype=np.float64)
        )
        actions = np.array([item.action for item in buffer], dtype=np.int16)
        rewards = np.array([item.reward for item in buffer], dtype=np.float32)
        dones = np.array([item.done for item in buffer], dtype=np.bool_)
        n_steps = np.array([item.n_steps for item in buffer], dtype=np.int8)
        next_legal = np.full((len(buffer), 12), -1, dtype=np.int8)
        for idx, item in enumerate(buffer):
            legal = item.next_legal_actions[:12]
            if legal:
                next_legal[idx, : len(legal)] = np.array(legal, dtype=np.int8)

        np.savez_compressed(
            path,
            states=states,
            next_states=next_states,
            actions=actions,
            rewards=rewards,
            dones=dones,
            n_steps=n_steps,
            next_legal=next_legal,
            memory_size=np.array([len(buffer)], dtype=np.int32),
            buffer_capacity=np.array([buffer.maxlen or len(buffer)], dtype=np.int32),
            buffer_name=np.array([buffer_name], dtype="<U16"),
            epsilon=np.array([self.epsilon], dtype=np.float64),
            training_steps=np.array([self.training_steps], dtype=np.int64),
        )

    def load_replay_buffer(self, path: Path, buffer_name: str = "recent", restore_epsilon: bool = True) -> None:
        data = np.load(path, allow_pickle=False)
        loaded_size = int(data["buffer_capacity"][0]) if "buffer_capacity" in data else int(data["memory_size"][0])
        if buffer_name == "anchor":
            self.anchor_memory = deque(maxlen=max(self.anchor_memory_size, loaded_size))
            target_buffer = self.anchor_memory
        else:
            self.memory = deque(maxlen=max(self.memory_size, loaded_size))
            target_buffer = self.memory
        states = data["states"]
        next_states = data["next_states"]
        actions = data["actions"]
        rewards = data["rewards"]
        dones = data["dones"]
        n_steps = data["n_steps"] if "n_steps" in data else np.ones(len(actions), dtype=np.int8)
        next_legal = data["next_legal"]
        for idx in range(len(actions)):
            legal = [int(value) for value in next_legal[idx].tolist() if int(value) >= 0]
            target_buffer.append(
                Transition(
                    state=states[idx].astype(np.float64, copy=True),
                    action=int(actions[idx]),
                    reward=float(rewards[idx]),
                    next_state=next_states[idx].astype(np.float64, copy=True),
                    done=bool(dones[idx]),
                    next_legal_actions=legal,
                    n_steps=int(n_steps[idx]),
                )
            )
        if restore_epsilon and "epsilon" in data:
            self.epsilon = float(data["epsilon"][0])
        if "training_steps" in data:
            self.training_steps = int(data["training_steps"][0])

    def sample_batch(self) -> list[Transition] | None:
        available_recent = len(self.memory)
        available_anchor = len(self.anchor_memory)
        total_available = available_recent + available_anchor
        if total_available < self.batch_size:
            return None

        desired_anchor = int(round(self.batch_size * self.anchor_batch_ratio)) if available_anchor > 0 else 0
        desired_anchor = min(desired_anchor, available_anchor)
        desired_recent = self.batch_size - desired_anchor
        desired_recent = min(desired_recent, available_recent)

        remaining = self.batch_size - (desired_anchor + desired_recent)
        if remaining > 0 and available_recent - desired_recent > 0:
            extra_recent = min(remaining, available_recent - desired_recent)
            desired_recent += extra_recent
            remaining -= extra_recent
        if remaining > 0 and available_anchor - desired_anchor > 0:
            extra_anchor = min(remaining, available_anchor - desired_anchor)
            desired_anchor += extra_anchor
            remaining -= extra_anchor
        if remaining > 0:
            return None

        batch: list[Transition] = []
        if desired_anchor > 0:
            batch.extend(random.sample(list(self.anchor_memory), desired_anchor))
        if desired_recent > 0:
            batch.extend(random.sample(list(self.memory), desired_recent))
        random.shuffle(batch)
        return batch

    def train_step(self) -> float | None:
        batch = self.sample_batch()
        if batch is None:
            return None

        states = np.vstack([item.state for item in batch])
        actions = np.array([item.action for item in batch], dtype=np.int64)
        target_values = np.array([item.reward for item in batch], dtype=np.float64)

        for idx, item in enumerate(batch):
            if not item.done and item.next_legal_actions:
                next_online = self.online.predict(item.next_state)[0]
                best_next_action = max(
                    item.next_legal_actions,
                    key=lambda action: next_online[action],
                )
                next_target = self.target.predict(item.next_state)[0]
                target_values[idx] += (self.gamma ** item.n_steps) * next_target[best_next_action]

        loss = self.online.train_batch(
            states,
            actions,
            target_values,
            self.lr,
            huber_delta=self.huber_delta,
            clip_norm=self.grad_clip_norm,
        )
        self.training_steps += 1
        if self.target_update_mode == "soft":
            self.target.soft_update_from(self.online, self.target_tau)
        elif self.training_steps % self.target_sync_every == 0:
            self.target.copy_from(self.online)
        return loss


def reward_for_move(player_turn: str, winner: str) -> float:
    reward = 0.0
    if winner == player_turn:
        reward += 10.0
    elif winner == "tie":
        reward += 1.0
    elif winner != "no winner":
        reward -= 10.0
    return float(reward)


def codigo_move(player_turn: str, board: dict[str, int]) -> str:
    return benchmark_codigo2_move(player_turn, board)


def model_move_from_numpy(model: NumpyMLP | NPZModel, player_turn: str, board: dict[str, int]) -> str:
    q_values = model.predict(board_to_vector(board, player_turn))[0]
    legal = legal_actions(player_turn, board)
    best_score = max(q_values[action] for action in legal)
    best_actions = [action for action in legal if q_values[action] == best_score]
    return action_to_label(random.choice(best_actions))


def opponent_move(opponent: Opponent, player_turn: str, board: dict[str, int]) -> str:
    if opponent.noise > 0.0 and random.random() < opponent.noise:
        return azar_move(player_turn, board, random)
    if opponent.kind == "azar":
        return azar_move(player_turn, board, random)
    if opponent.kind == "codigo":
        return codigo_move(player_turn, board)
    if opponent.kind == "codigo3":
        return benchmark_codigo3_move(player_turn, board)
    if opponent.kind == "codigo4":
        return benchmark_codigo4_move(player_turn, board)
    if opponent.kind == "codigo5":
        return benchmark_codigo5_move(player_turn, board)
    if opponent.kind == "benchmark_model":
        return model_move(opponent.payload, player_turn, board)
    if opponent.kind in {"npz_model", "numpy_model"}:
        return model_move_from_numpy(opponent.payload, player_turn, board)
    raise ValueError(f"Tipo de oponente no soportado: {opponent.kind}")


def choose_training_opponent(opponents: list[Opponent]) -> Opponent:
    weights = [opponent.weight for opponent in opponents]
    return random.choices(opponents, weights=weights, k=1)[0]


def adaptive_weight_from_win_rate(win_rate: float | None, floor: float = 0.25) -> float:
    if win_rate is None:
        return 1.0
    distance = min(1.0, abs(win_rate - 0.5) / 0.5)
    return floor + (1.0 - distance)


def build_training_opponents(
    agent: DQNAgent,
    snapshot_pool: list[NumpyMLP],
    fixed_dqn_models: list[Opponent],
    phase2_model,
    opponent_stats: dict[str, float] | None = None,
) -> list[Opponent]:
    opponent_stats = opponent_stats or {}
    opponents = [
        Opponent(name="self_current", kind="numpy_model", payload=agent.online.clone(), weight=0.30),
        Opponent(name="codigo", kind="codigo", weight=0.10),
        Opponent(name="codigo_noise_02", kind="codigo", weight=0.08, noise=0.02),
        Opponent(name="azar", kind="azar", weight=0.04),
    ]
    if snapshot_pool:
        shared_snapshot_weight = 0.18 / float(len(snapshot_pool))
        for idx, snapshot in enumerate(snapshot_pool):
            opponents.append(
                Opponent(
                    name=f"self_snapshot_{idx}",
                    kind="numpy_model",
                    payload=snapshot.clone(),
                    weight=shared_snapshot_weight,
                )
            )
    if fixed_dqn_models:
        shared_weight = 0.22 / float(len(fixed_dqn_models))
        for fixed_model in fixed_dqn_models:
            opponents.append(
                Opponent(
                    name=fixed_model.name,
                    kind=fixed_model.kind,
                    payload=fixed_model.payload,
                    weight=shared_weight,
                )
            )
    if phase2_model is not None:
        opponents.append(
            Opponent(
                name="fase2_best",
                kind="benchmark_model",
                payload=phase2_model,
                weight=0.10,
            )
        )
        opponents.append(
            Opponent(
                name="fase2_best_noise_02",
                kind="benchmark_model",
                payload=phase2_model,
                weight=0.06,
                noise=0.02,
            )
        )
    for opponent in opponents:
        opponent.weight *= adaptive_weight_from_win_rate(opponent_stats.get(opponent.name))
    total = sum(opponent.weight for opponent in opponents)
    for opponent in opponents:
        opponent.weight /= total
    return opponents


def finalize_n_step_transition(
    pending: deque[PendingTransition],
    n_step: int,
    gamma: float,
) -> Transition:
    usable = list(pending)[: max(1, min(n_step, len(pending)))]
    reward = 0.0
    for idx, step in enumerate(usable):
        reward += (gamma ** idx) * step.reward
    first = usable[0]
    last = usable[-1]
    return Transition(
        state=first.state,
        action=first.action,
        reward=reward,
        next_state=last.next_state,
        done=last.done,
        next_legal_actions=last.next_legal_actions,
        n_steps=len(usable),
    )


def play_training_episode(
    agent: DQNAgent,
    opponents: list[Opponent],
    train: bool = True,
    fill_anchor: bool = False,
    mcts_simulations: int = 0,
) -> tuple[str, int, float, str]:
    board = get_new_board()
    opponent = choose_training_opponent(opponents)
    ai_player = random.choice(["1", "2"])
    player_turn = random.choice(["1", "2"])
    done = False
    losses = []
    winner = "tie"
    pending_steps: deque[PendingTransition] = deque()
    mcts = None
    if mcts_simulations > 0:
        # Import locally to keep this module usable as the MCTS rules backend.
        from mancala_mcts import MCTSPlayer

        mcts = MCTSPlayer(
            model=agent.online,
            simulations=mcts_simulations,
            rng=random.Random(random.randrange(2**63)),
        )

    while not done:
        if player_turn == ai_player:
            state = board_to_vector(board, player_turn)
            legal = legal_actions(player_turn, board)
            if mcts is not None and random.random() >= agent.epsilon:
                action = mcts.act(board, player_turn)
            else:
                action = agent.act(state, legal, explore=True)
            move = action_to_label(action)
        else:
            state = None
            action = None
            move = opponent_move(opponent, player_turn, board)

        next_player = make_move(board, player_turn, move)
        winner = check_for_winner(board)
        done = winner != "no winner"

        if player_turn == ai_player:
            if done:
                next_state = np.zeros(15, dtype=np.float64)
                next_legal = []
            else:
                current_turn = next_player
                while current_turn != ai_player and not done:
                    opp_move = opponent_move(opponent, current_turn, board)
                    current_turn = make_move(board, current_turn, opp_move)
                    winner = check_for_winner(board)
                    done = winner != "no winner"
                if done:
                    next_state = np.zeros(15, dtype=np.float64)
                    next_legal = []
                else:
                    next_state = board_to_vector(board, current_turn)
                    next_legal = legal_actions(current_turn, board)
                    next_player = current_turn

            pending_steps.append(
                PendingTransition(
                    state=state,
                    action=action,
                    reward=reward_for_move(ai_player, winner),
                    next_state=next_state,
                    done=done,
                    next_legal_actions=next_legal,
                )
            )
            if len(pending_steps) >= agent.n_step:
                agent.remember(
                    finalize_n_step_transition(pending_steps, agent.n_step, agent.gamma),
                    to_anchor=fill_anchor,
                )
                pending_steps.popleft()
            if train:
                loss = agent.train_step()
                if loss is not None:
                    losses.append(loss)
        player_turn = next_player

    while pending_steps:
        agent.remember(
            finalize_n_step_transition(pending_steps, agent.n_step, agent.gamma),
            to_anchor=fill_anchor,
        )
        pending_steps.popleft()

    avg_loss = float(np.mean(losses)) if losses else 0.0
    return winner, len(losses), avg_loss, opponent.name


def populate_replay_buffer(
    agent: DQNAgent,
    episodes: int,
    seed: int,
    fixed_dqn_paths: list[Path],
    phase2_path: Path,
    opponent_stats: dict[str, float] | None = None,
) -> None:
    if episodes <= 0:
        return

    random.seed(seed)
    np.random.seed(seed)
    snapshot_pool: list[NumpyMLP] = []
    fixed_dqn_models = load_npz_opponents(fixed_dqn_paths)
    phase2_model = load_model_spec(phase2_path) if phase2_path.exists() else None

    print()
    print(f"Prellenando replay buffer con {episodes} episodios antes del entrenamiento...")
    report_every = max(1, min(1000, episodes))

    for episode in range(1, episodes + 1):
        opponents = build_training_opponents(agent, snapshot_pool, fixed_dqn_models, phase2_model, opponent_stats)
        play_training_episode(agent, opponents, train=False, fill_anchor=True)
        if episode % report_every == 0 or episode == episodes:
            print(
                "Warmup "
                f"{episode:5d}/{episodes:5d} | recientes: {len(agent.memory):6d} | ancla: {len(agent.anchor_memory):6d}"
            )

    print("Replay buffer listo.")


def load_npz_opponents(paths: list[Path]) -> list[Opponent]:
    opponents: list[Opponent] = []
    for path in paths:
        if path.exists():
            opponents.append(Opponent(name=path.stem, kind="npz_model", payload=NPZModel(path)))
    return opponents


def play_vs_opponent(agent: DQNAgent, opponent: Opponent, games: int = 100) -> tuple[int, int, int]:
    wins = 0
    losses = 0
    draws = 0
    role_cycle = (("1", "1"), ("1", "2"), ("2", "1"), ("2", "2"))

    for game_index in range(games):
        board = get_new_board()
        ai_player, player_turn = role_cycle[game_index % len(role_cycle)]

        while True:
            if player_turn == ai_player:
                state = board_to_vector(board, player_turn)
                action = agent.act(state, legal_actions(player_turn, board), explore=False)
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


def build_eval_opponents(
    fixed_dqn_models: list[Opponent],
    phase2_model,
    include_azar: bool = True,
    include_codigo: bool = True,
) -> list[Opponent]:
    opponents: list[Opponent] = []
    if include_azar:
        opponents.append(Opponent(name="azar", kind="azar"))
    if include_codigo:
        opponents.append(Opponent(name="codigo", kind="codigo"))
        opponents.append(Opponent(name="codigo_noise_02", kind="codigo", noise=0.02))
        opponents.append(Opponent(name="codigo_noise_05", kind="codigo", noise=0.05))
    if phase2_model is not None:
        opponents.append(Opponent(name="fase2_best", kind="benchmark_model", payload=phase2_model))
        opponents.append(Opponent(name="fase2_best_noise_02", kind="benchmark_model", payload=phase2_model, noise=0.02))
    opponents.extend(
        Opponent(name=model.name, kind=model.kind, payload=model.payload)
        for model in fixed_dqn_models
    )
    return opponents


def evaluate_agent(
    agent: DQNAgent,
    eval_games: int,
    eval_opponents: list[Opponent],
) -> dict[str, Any]:
    if not eval_opponents:
        wins, losses, draws = play_vs_opponent(agent, Opponent(name="azar", kind="azar"), games=eval_games)
        score = (3.0 * wins + draws) / (3.0 * eval_games)
        return {
            "mixed_score": score,
            "worst_case_score": score,
            "azar_wins": wins,
            "azar_losses": losses,
            "azar_draws": draws,
            "azar_win_rate": wins / float(eval_games),
            "azar_score": score,
        }

    results: dict[str, Any] = {}
    score_total = 0.0
    score_values: list[float] = []
    for opponent in eval_opponents:
        wins, losses, draws = play_vs_opponent(agent, opponent, games=eval_games)
        score = (3.0 * wins + draws) / (3.0 * eval_games)
        win_rate = wins / float(eval_games)
        score_total += score
        score_values.append(score)
        results[f"{opponent.name}_wins"] = wins
        results[f"{opponent.name}_losses"] = losses
        results[f"{opponent.name}_draws"] = draws
        results[f"{opponent.name}_win_rate"] = win_rate
        results[f"{opponent.name}_score"] = score
    results["mixed_score"] = score_total / float(len(eval_opponents))
    results["worst_case_score"] = min(score_values) if score_values else 0.0
    return results


def save_model_snapshot(model: NumpyMLP, path: Path, metadata: dict[str, Any]) -> None:
    model.save(path, metadata=metadata)


def update_elite_checkpoints(
    elite_entries: list[EliteCheckpoint],
    elite_dir: Path | None,
    elite_size: int,
    model: NumpyMLP,
    episode: int,
    score: float,
    metadata: dict[str, Any],
) -> list[EliteCheckpoint]:
    if elite_dir is None or elite_size <= 0:
        return elite_entries

    elite_dir.mkdir(parents=True, exist_ok=True)
    candidate = EliteCheckpoint(
        score=score,
        episode=episode,
        path=elite_dir / f"elite_ep{episode}_score{score:.4f}.npz",
    )

    if len(elite_entries) < elite_size or score > min(entry.score for entry in elite_entries):
        save_model_snapshot(model, candidate.path, metadata)
        elite_entries.append(candidate)
        elite_entries.sort(key=lambda entry: (entry.score, entry.episode), reverse=True)
        while len(elite_entries) > elite_size:
            removed = elite_entries.pop()
            if removed.path.exists():
                removed.path.unlink()
    return elite_entries


def run_training(
    episodes: int,
    eval_every: int,
    eval_games: int,
    seed: int,
    gamma: float,
    epsilon: float,
    epsilon_min: float,
    epsilon_decay: float,
    epsilon_decay_episodes: int,
    lr: float,
    batch_size: int,
    memory_size: int,
    anchor_memory_size: int,
    anchor_batch_ratio: float,
    n_step: int,
    target_sync_every: int,
    target_update_mode: str,
    target_tau: float,
    huber_delta: float,
    grad_clip_norm: float,
    fixed_dqn_paths: list[Path],
    phase2_path: Path,
    load_model_path: Path | None,
    load_replay_buffer_path: Path | None,
    load_anchor_replay_buffer_path: Path | None,
    reset_epsilon_on_load: bool,
    checkpoint_dir: Path | None,
    checkpoint_every: int,
    episode_offset: int,
    best_mixed_model_path: Path | None,
    best_azar_model_path: Path | None,
    best_worst_model_path: Path | None,
    best_mixed_replay_buffer_path: Path | None,
    best_azar_replay_buffer_path: Path | None,
    best_worst_replay_buffer_path: Path | None,
    final_replay_buffer_path: Path | None,
    final_anchor_replay_buffer_path: Path | None,
    elite_dir: Path | None,
    elite_size: int,
    replay_warmup_episodes: int,
    mcts_simulations: int = 0,
) -> tuple[DQNAgent, list[dict[str, float | int]]]:
    random.seed(seed)
    np.random.seed(seed)
    agent = DQNAgent(
        gamma=gamma,
        epsilon=epsilon,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        epsilon_decay_episodes=epsilon_decay_episodes,
        lr=lr,
        batch_size=batch_size,
        memory_size=memory_size,
        anchor_memory_size=anchor_memory_size,
        anchor_batch_ratio=anchor_batch_ratio,
        n_step=n_step,
        target_sync_every=target_sync_every,
        target_update_mode=target_update_mode,
        target_tau=target_tau,
        huber_delta=huber_delta,
        grad_clip_norm=grad_clip_norm,
    )
    if load_model_path is not None:
        if not load_model_path.exists():
            raise FileNotFoundError(f"No existe el modelo inicial: {load_model_path}")
        agent.load_model(load_model_path)
    if load_replay_buffer_path is not None:
        if not load_replay_buffer_path.exists():
            raise FileNotFoundError(f"No existe el replay buffer inicial: {load_replay_buffer_path}")
        agent.load_replay_buffer(load_replay_buffer_path, buffer_name="recent", restore_epsilon=not reset_epsilon_on_load)
    if load_anchor_replay_buffer_path is not None:
        if not load_anchor_replay_buffer_path.exists():
            raise FileNotFoundError(f"No existe el replay buffer ancla inicial: {load_anchor_replay_buffer_path}")
        agent.load_replay_buffer(
            load_anchor_replay_buffer_path,
            buffer_name="anchor",
            restore_epsilon=False,
        )
    if reset_epsilon_on_load:
        agent.epsilon = agent.epsilon_start

    opponent_stats: dict[str, float] = {}
    if replay_warmup_episodes > 0:
        populate_replay_buffer(
            agent=agent,
            episodes=replay_warmup_episodes,
            seed=seed + 1,
            fixed_dqn_paths=fixed_dqn_paths,
            phase2_path=phase2_path,
            opponent_stats=opponent_stats,
        )

    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    snapshot_pool: deque[NumpyMLP] = deque(maxlen=24)
    fixed_dqn_models = load_npz_opponents(fixed_dqn_paths)
    phase2_model = load_model_spec(phase2_path) if phase2_path.exists() else None
    eval_opponents = build_eval_opponents(fixed_dqn_models, phase2_model)
    print("Entrenando DQN por autojuego con arquitectura 15 -> 64 -> 64 -> 12")
    print("Double DQN, replay dual, Huber+Adam, n-step=3 y poblacion mixta adaptativa")
    if mcts_simulations > 0:
        print(f"MCTS activo para las jugadas del agente con {mcts_simulations} simulaciones por turno")
    if load_model_path is not None:
        print(f"Continuando desde modelo inicial: {load_model_path}")
    if load_replay_buffer_path is not None:
        print(f"Replay buffer restaurado desde: {load_replay_buffer_path}")
    if load_anchor_replay_buffer_path is not None:
        print(f"Replay buffer ancla restaurado desde: {load_anchor_replay_buffer_path}")
    print()
    print(f"{'Ep':>7} {'Epsilon':>8} {'Mix%':>7} {'Azar%':>7} {'Worst%':>8} {'AvgLoss':>10}")
    print("-" * 64)

    recent_losses: list[float] = []
    history: list[dict[str, float | int]] = []
    elite_entries: list[EliteCheckpoint] = []
    best_scores = {
        "mixed": float("-inf"),
        "azar": float("-inf"),
        "worst": float("-inf"),
    }
    for episode in range(1, episodes + 1):
        agent.set_episode_epsilon(episode - 1)
        opponents = build_training_opponents(
            agent,
            list(snapshot_pool),
            fixed_dqn_models,
            phase2_model,
            opponent_stats,
        )
        _, _, avg_loss, _ = play_training_episode(
            agent,
            opponents,
            mcts_simulations=mcts_simulations,
        )
        if avg_loss > 0:
            recent_losses.append(avg_loss)
        if len(recent_losses) > eval_every:
            recent_losses.pop(0)

        if episode % eval_every == 0:
            eval_results = evaluate_agent(agent, eval_games=eval_games, eval_opponents=eval_opponents)
            loss_window = float(np.mean(recent_losses)) if recent_losses else 0.0
            snapshot_pool.append(agent.online.clone())
            absolute_episode = episode_offset + episode
            row = {
                "episode": absolute_episode,
                "epsilon": agent.epsilon,
                "avg_loss": loss_window,
            }
            row.update(eval_results)
            history.append(row)
            mixed_score = float(row["mixed_score"])
            azar_win_rate = float(row.get("azar_win_rate", 0.0))
            worst_score = float(row.get("worst_case_score", 0.0))
            opponent_stats = {
                opponent.name: float(row.get(f"{opponent.name}_win_rate", 0.5))
                for opponent in eval_opponents
            }
            print(
                f"{absolute_episode:7d} {agent.epsilon:8.4f} {mixed_score*100:6.1f}% "
                f"{azar_win_rate*100:6.1f}% {worst_score*100:7.1f}% {loss_window:10.6f}"
            )
            if checkpoint_dir is not None and checkpoint_every > 0 and episode % checkpoint_every == 0:
                checkpoint_path = checkpoint_dir / f"dqn_checkpoint_ep{absolute_episode}.npz"
                checkpoint_metadata = dict(row)
                checkpoint_metadata["type"] = "periodic_checkpoint"
                save_model_snapshot(agent.online, checkpoint_path, checkpoint_metadata)
                print(f"Checkpoint periodico guardado: {checkpoint_path}")
            elite_entries = update_elite_checkpoints(
                elite_entries=elite_entries,
                elite_dir=elite_dir,
                elite_size=elite_size,
                model=agent.online,
                episode=absolute_episode,
                score=mixed_score,
                metadata=dict(row, type="elite"),
            )
            if best_mixed_model_path is not None and mixed_score > best_scores["mixed"]:
                best_scores["mixed"] = mixed_score
                save_model_snapshot(agent.online, best_mixed_model_path, dict(row, selection="best_mixed"))
                if best_mixed_replay_buffer_path is not None:
                    agent.save_replay_buffer(best_mixed_replay_buffer_path, buffer_name="recent")
                print(
                    "Nuevo mejor modelo mixto guardado en episodio "
                    f"{absolute_episode} con score de {mixed_score*100:.2f}%: {best_mixed_model_path}"
                )
            if best_azar_model_path is not None and azar_win_rate > best_scores["azar"]:
                best_scores["azar"] = azar_win_rate
                save_model_snapshot(agent.online, best_azar_model_path, dict(row, selection="best_azar"))
                if best_azar_replay_buffer_path is not None:
                    agent.save_replay_buffer(best_azar_replay_buffer_path, buffer_name="recent")
                print(
                    "Nuevo mejor modelo vs Azar guardado en episodio "
                    f"{absolute_episode} con Win% de {azar_win_rate*100:.2f}%: {best_azar_model_path}"
                )
            if best_worst_model_path is not None and worst_score > best_scores["worst"]:
                best_scores["worst"] = worst_score
                save_model_snapshot(agent.online, best_worst_model_path, dict(row, selection="best_worst_case"))
                if best_worst_replay_buffer_path is not None:
                    agent.save_replay_buffer(best_worst_replay_buffer_path, buffer_name="recent")
                print(
                    "Nuevo mejor modelo worst-case guardado en episodio "
                    f"{absolute_episode} con score de {worst_score*100:.2f}%: {best_worst_model_path}"
                )
    if final_replay_buffer_path is not None:
        agent.save_replay_buffer(final_replay_buffer_path, buffer_name="recent")
    if final_anchor_replay_buffer_path is not None:
        agent.save_replay_buffer(final_anchor_replay_buffer_path, buffer_name="anchor")
    return agent, history


def save_history_csv(history: list[dict[str, float | int]], path: Path) -> None:
    if not history:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(history[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def plot_history(history: list[dict[str, float | int]], path: Path) -> None:
    import matplotlib.pyplot as plt

    episodes = [int(row["episode"]) for row in history]
    mixed_scores = [float(row["mixed_score"]) * 100.0 for row in history]
    azar_scores = [float(row.get("azar_win_rate", 0.0)) * 100.0 for row in history]
    worst_scores = [float(row.get("worst_case_score", 0.0)) * 100.0 for row in history]

    plt.figure(figsize=(10, 5))
    plt.plot(episodes, mixed_scores, marker="o", linewidth=2, color="#1f5aa6", label="Score mixto")
    if any(score > 0 for score in azar_scores):
        plt.plot(episodes, azar_scores, linewidth=1.5, color="#d97706", alpha=0.85, label="Win% vs Azar")
    if any(score > 0 for score in worst_scores):
        plt.plot(episodes, worst_scores, linewidth=1.5, color="#15803d", alpha=0.85, label="Worst-case score")
    plt.title("DQN Self-Play Evaluation")
    plt.xlabel("Episodios de entrenamiento")
    plt.ylabel("Porcentaje")
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Entrenamiento DQN por autojuego para Mancala.")
    parser.add_argument("--episodes", type=int, default=100000)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--eval-games", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260526)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--epsilon", type=float, default=0.02)
    parser.add_argument("--epsilon-min", type=float, default=0.01)
    parser.add_argument("--epsilon-decay", type=float, default=0.9995)
    parser.add_argument("--epsilon-decay-episodes", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--memory-size", type=int, default=600000)
    parser.add_argument("--anchor-memory-size", type=int, default=200000)
    parser.add_argument("--anchor-batch-ratio", type=float, default=0.25)
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument("--target-sync-every", type=int, default=1000)
    parser.add_argument("--target-update-mode", choices=("hard", "soft"), default="soft")
    parser.add_argument("--target-tau", type=float, default=0.005)
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--grad-clip-norm", type=float, default=5.0)
    parser.add_argument("--history-csv", type=Path, default=Path("dqn_history.csv"))
    parser.add_argument("--plot", type=Path, default=Path("dqn_winrate.png"))
    parser.add_argument("--best-model", type=Path, default=Path("dqn_best_model.npz"))
    parser.add_argument("--best-azar-model", type=Path, default=None)
    parser.add_argument("--best-worst-model", type=Path, default=None)
    parser.add_argument("--final-model", type=Path, default=Path("dqn_final_model.npz"))
    parser.add_argument("--load-model", type=Path, default=None)
    parser.add_argument("--load-replay-buffer", type=Path, default=None)
    parser.add_argument("--load-anchor-replay-buffer", type=Path, default=None)
    parser.add_argument("--reset-epsilon-on-load", action="store_true")
    parser.add_argument("--episode-offset", type=int, default=0)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--replay-warmup-episodes", type=int, default=50000)
    parser.add_argument("--best-replay-buffer", type=Path, default=None)
    parser.add_argument("--best-azar-replay-buffer", type=Path, default=None)
    parser.add_argument("--best-worst-replay-buffer", type=Path, default=None)
    parser.add_argument("--final-replay-buffer", type=Path, default=None)
    parser.add_argument("--final-anchor-replay-buffer", type=Path, default=None)
    parser.add_argument("--elite-dir", type=Path, default=None)
    parser.add_argument("--elite-size", type=int, default=3)
    parser.add_argument(
        "--mcts-simulations",
        type=int,
        default=0,
        help="Simulaciones MCTS por jugada del agente; cero conserva DDQN puro.",
    )
    parser.add_argument(
        "--fixed-dqn-path",
        type=Path,
        default=Path("dqn_best_remote_30000_epsmin02.npz"),
    )
    parser.add_argument(
        "--training-model-paths",
        nargs="*",
        type=Path,
        default=[],
    )
    parser.add_argument(
        "--phase2-model-path",
        type=Path,
        default=Path("modelo_mancala_4000_mas_perdedoras.h5"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    fixed_dqn_paths = [args.fixed_dqn_path] + list(args.training_model_paths)
    agent, history = run_training(
        episodes=args.episodes,
        eval_every=args.eval_every,
        eval_games=args.eval_games,
        seed=args.seed,
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        epsilon_decay_episodes=args.epsilon_decay_episodes,
        lr=args.lr,
        batch_size=args.batch_size,
        memory_size=args.memory_size,
        anchor_memory_size=args.anchor_memory_size,
        anchor_batch_ratio=args.anchor_batch_ratio,
        n_step=args.n_step,
        target_sync_every=args.target_sync_every,
        target_update_mode=args.target_update_mode,
        target_tau=args.target_tau,
        huber_delta=args.huber_delta,
        grad_clip_norm=args.grad_clip_norm,
        fixed_dqn_paths=fixed_dqn_paths,
        phase2_path=args.phase2_model_path,
        load_model_path=args.load_model,
        load_replay_buffer_path=args.load_replay_buffer,
        load_anchor_replay_buffer_path=args.load_anchor_replay_buffer,
        reset_epsilon_on_load=args.reset_epsilon_on_load,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_every=args.checkpoint_every,
        episode_offset=args.episode_offset,
        best_mixed_model_path=args.best_model,
        best_azar_model_path=args.best_azar_model,
        best_worst_model_path=args.best_worst_model,
        best_mixed_replay_buffer_path=args.best_replay_buffer,
        best_azar_replay_buffer_path=args.best_azar_replay_buffer,
        best_worst_replay_buffer_path=args.best_worst_replay_buffer,
        final_replay_buffer_path=args.final_replay_buffer,
        final_anchor_replay_buffer_path=args.final_anchor_replay_buffer,
        elite_dir=args.elite_dir,
        elite_size=args.elite_size,
        replay_warmup_episodes=args.replay_warmup_episodes,
        mcts_simulations=args.mcts_simulations,
    )
    save_history_csv(history, args.history_csv)
    plot_history(history, args.plot)
    final_episode = history[-1]["episode"] if history else 0
    final_metadata = dict(history[-1]) if history else {"episode": final_episode, "type": "final"}
    final_metadata["type"] = "final"
    save_model_snapshot(agent.online, args.final_model, final_metadata)
    print()
    print(f"Historial guardado en: {args.history_csv}")
    print(f"Grafica guardada en: {args.plot}")
    print(f"Mejor modelo mixto guardado en: {args.best_model}")
    if args.best_azar_model is not None:
        print(f"Mejor modelo vs Azar guardado en: {args.best_azar_model}")
    if args.best_worst_model is not None:
        print(f"Mejor modelo worst-case guardado en: {args.best_worst_model}")
    print(f"Modelo final guardado en: {args.final_model}")
    if args.best_replay_buffer is not None:
        print(f"Replay buffer del mejor modelo mixto guardado en: {args.best_replay_buffer}")
    if args.best_azar_replay_buffer is not None:
        print(f"Replay buffer del mejor modelo vs Azar guardado en: {args.best_azar_replay_buffer}")
    if args.best_worst_replay_buffer is not None:
        print(f"Replay buffer del mejor modelo worst-case guardado en: {args.best_worst_replay_buffer}")
    if args.final_replay_buffer is not None:
        print(f"Replay buffer final guardado en: {args.final_replay_buffer}")
    if args.final_anchor_replay_buffer is not None:
        print(f"Replay buffer ancla final guardado en: {args.final_anchor_replay_buffer}")


if __name__ == "__main__":
    main()
