"""Monte Carlo Tree Search guided by the current Mancala DDQN.

The tree stores values from the root player's perspective. This makes the
player switch explicit and, importantly for Mancala, preserves extra turns:
the value is only negated at opponent nodes, not after every move.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from mancala_dqn_selfplay import (
    ACTION_LABELS,
    action_to_label,
    board_to_vector,
    check_for_winner,
    get_new_board,
    legal_actions,
    make_move,
)


class QModel(Protocol):
    def predict(self, state: np.ndarray) -> np.ndarray:
        """Return a batch with twelve action values."""


def _softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    temperature = max(float(temperature), 1e-6)
    scaled = values / temperature
    scaled -= np.max(scaled)
    probabilities = np.exp(scaled)
    total = float(np.sum(probabilities))
    if not math.isfinite(total) or total <= 0.0:
        return np.full(len(values), 1.0 / len(values), dtype=np.float64)
    return probabilities / total


def _winner_value(winner: str, root_player: str) -> float:
    if winner == "tie" or winner == "no winner":
        return 0.0
    return 1.0 if winner == root_player else -1.0


@dataclass
class MCTSNode:
    board: dict[str, int]
    player_turn: str
    root_player: str
    parent: "MCTSNode | None" = None
    action: int | None = None
    prior: float = 0.0
    visits: int = 0
    value_sum: float = 0.0
    children: dict[int, "MCTSNode"] = field(default_factory=dict)
    untried_actions: list[int] = field(default_factory=list)
    terminal_value: float | None = None

    @property
    def mean_value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


@dataclass(frozen=True)
class MCTSResult:
    action: int
    visits: dict[int, int]
    values: dict[int, float]
    simulations: int

    @property
    def move(self) -> str:
        return action_to_label(self.action)


class MCTSPlayer:
    """MCTS player that uses a DDQN for priors and rollout policy."""

    def __init__(
        self,
        model: QModel | None = None,
        simulations: int = 200,
        c_puct: float = 1.4,
        prior_temperature: float = 1.0,
        rollout_temperature: float = 0.7,
        rollout_randomness: float = 0.08,
        max_rollout_moves: int = 200,
        rng: random.Random | None = None,
    ) -> None:
        if simulations <= 0:
            raise ValueError("simulations debe ser mayor que cero")
        if max_rollout_moves <= 0:
            raise ValueError("max_rollout_moves debe ser mayor que cero")
        if not 0.0 <= rollout_randomness <= 1.0:
            raise ValueError("rollout_randomness debe estar entre cero y uno")

        self.model = model
        self.simulations = int(simulations)
        self.c_puct = float(c_puct)
        self.prior_temperature = float(prior_temperature)
        self.rollout_temperature = float(rollout_temperature)
        self.rollout_randomness = float(rollout_randomness)
        self.max_rollout_moves = int(max_rollout_moves)
        self.rng = rng or random.Random()

    def _priors(self, board: dict[str, int], player_turn: str, actions: list[int]) -> dict[int, float]:
        if not actions:
            return {}
        if self.model is None:
            probability = 1.0 / len(actions)
            return {action: probability for action in actions}

        state = board_to_vector(board, player_turn)
        q_values = np.asarray(self.model.predict(state)[0], dtype=np.float64)
        if q_values.shape[0] != len(ACTION_LABELS):
            raise ValueError("El modelo debe devolver doce valores de accion")
        probabilities = _softmax(q_values[actions], self.prior_temperature)
        return {action: float(probabilities[index]) for index, action in enumerate(actions)}

    def _new_root(self, board: dict[str, int], player_turn: str) -> MCTSNode:
        root_board = board.copy()
        terminal = check_for_winner(root_board)
        actions = [] if terminal != "no winner" else legal_actions(player_turn, root_board)
        return MCTSNode(
            board=root_board,
            player_turn=player_turn,
            root_player=player_turn,
            untried_actions=list(actions),
            terminal_value=_winner_value(terminal, player_turn) if terminal != "no winner" else None,
        )

    def _expand(self, node: MCTSNode) -> MCTSNode:
        priors = self._priors(node.board, node.player_turn, node.untried_actions)
        action = max(node.untried_actions, key=lambda candidate: (priors[candidate], -candidate))
        node.untried_actions.remove(action)

        child_board = node.board.copy()
        move = action_to_label(action)
        next_player = make_move(child_board, node.player_turn, move)
        winner = check_for_winner(child_board)
        child_actions = [] if winner != "no winner" else legal_actions(next_player, child_board)
        child = MCTSNode(
            board=child_board,
            player_turn=next_player,
            root_player=node.root_player,
            parent=node,
            action=action,
            prior=priors[action],
            untried_actions=list(child_actions),
            terminal_value=_winner_value(winner, node.root_player) if winner != "no winner" else None,
        )
        node.children[action] = child
        return child

    def _select_child(self, node: MCTSNode) -> MCTSNode:
        parent_visits = max(1, node.visits)
        maximizing = node.player_turn == node.root_player

        def score(child: MCTSNode) -> tuple[float, float, int]:
            exploitation = child.mean_value if maximizing else -child.mean_value
            exploration = self.c_puct * child.prior * math.sqrt(parent_visits) / (1 + child.visits)
            return exploitation + exploration, child.mean_value, -int(child.action or 0)

        return max(node.children.values(), key=score)

    def _rollout(self, node: MCTSNode) -> float:
        if node.terminal_value is not None:
            return node.terminal_value

        board = node.board.copy()
        player_turn = node.player_turn
        for _ in range(self.max_rollout_moves):
            actions = legal_actions(player_turn, board)
            if not actions:
                terminal_board = board.copy()
                winner = check_for_winner(terminal_board)
                return _winner_value(winner, node.root_player)

            if self.model is not None and self.rng.random() >= self.rollout_randomness:
                priors = self._priors(board, player_turn, actions)
                weights = [priors[action] for action in actions]
                action = self.rng.choices(actions, weights=weights, k=1)[0]
            else:
                action = self.rng.choice(actions)

            player_turn = make_move(board, player_turn, action_to_label(action))
            winner = check_for_winner(board)
            if winner != "no winner":
                return _winner_value(winner, node.root_player)

        # A legal Mancala game should finish before this limit. A material
        # fallback keeps a malformed/custom ruleset from hanging evaluation.
        return float(np.sign(board[node.root_player] - board["2" if node.root_player == "1" else "1"]))

    def _backpropagate(self, node: MCTSNode, value: float) -> None:
        current: MCTSNode | None = node
        while current is not None:
            current.visits += 1
            current.value_sum += value
            current = current.parent

    def search(self, board: dict[str, int], player_turn: str) -> MCTSResult:
        root = self._new_root(board, player_turn)
        if not root.untried_actions:
            raise ValueError("No hay jugadas legales para iniciar MCTS")

        for _ in range(self.simulations):
            node = root
            while node.terminal_value is None and not node.untried_actions and node.children:
                node = self._select_child(node)
            if node.terminal_value is None and node.untried_actions:
                node = self._expand(node)
            value = self._rollout(node)
            self._backpropagate(node, value)

        visits = {action: child.visits for action, child in root.children.items()}
        values = {action: child.mean_value for action, child in root.children.items()}
        action = max(root.children, key=lambda candidate: (visits[candidate], values[candidate], -candidate))
        return MCTSResult(action=action, visits=visits, values=values, simulations=self.simulations)

    def act(self, board: dict[str, int], player_turn: str) -> int:
        return self.search(board, player_turn).action

    def move(self, board: dict[str, int], player_turn: str) -> str:
        return self.search(board, player_turn).move


def initial_board() -> dict[str, int]:
    """Small convenience wrapper for scripts and interactive experiments."""

    return get_new_board()
