"""Entrenamiento por autojuego para Mancala usando Q-learning tabular.

Este módulo evita dependencias externas y ofrece un punto de partida limpio
para entrenar un agente que aprenda jugando contra sí mismo.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


PLAYER_0_PITS = tuple(range(0, 6))
PLAYER_0_STORE = 6
PLAYER_1_PITS = tuple(range(7, 13))
PLAYER_1_STORE = 13
TOTAL_SEEDS = 48
INITIAL_BOARD = (4, 4, 4, 4, 4, 4, 0, 4, 4, 4, 4, 4, 4, 0)


def opponent(player: int) -> int:
    return 1 - player


def player_pits(player: int) -> Tuple[int, ...]:
    return PLAYER_0_PITS if player == 0 else PLAYER_1_PITS


def player_store(player: int) -> int:
    return PLAYER_0_STORE if player == 0 else PLAYER_1_STORE


def action_to_board_index(player: int, action: int) -> int:
    if not 0 <= action <= 5:
        raise ValueError(f"Accion invalida: {action}")
    return action if player == 0 else 7 + action


def board_index_to_label(index: int) -> str:
    labels = "ABCDEF"
    if index in PLAYER_0_PITS:
        return labels[index]
    if index in PLAYER_1_PITS:
        return labels[index - 7]
    raise ValueError(f"Indice sin etiqueta de hoyo: {index}")


def label_to_action(label: str) -> int:
    label = label.upper().strip()
    labels = "ABCDEF"
    if label not in labels:
        raise ValueError("La jugada debe ser una letra entre A y F.")
    return labels.index(label)


def canonical_state(board: Sequence[int], player: int) -> Tuple[int, ...]:
    """Normaliza el tablero desde la perspectiva del jugador activo."""
    if player == 0:
        own_pits = [board[i] for i in PLAYER_0_PITS]
        own_store = board[PLAYER_0_STORE]
        opp_pits = [board[i] for i in PLAYER_1_PITS]
        opp_store = board[PLAYER_1_STORE]
    else:
        own_pits = [board[i] for i in PLAYER_1_PITS]
        own_store = board[PLAYER_1_STORE]
        opp_pits = [board[i] for i in PLAYER_0_PITS]
        opp_store = board[PLAYER_0_STORE]
    return tuple(own_pits + [own_store] + opp_pits + [opp_store])


def serialize_state(state: Sequence[int]) -> str:
    return ",".join(str(value) for value in state)


def render_board(board: Sequence[int]) -> str:
    top = " ".join(f"{board[i]:2d}" for i in range(12, 6, -1))
    bottom = " ".join(f"{board[i]:2d}" for i in range(0, 6))
    return (
        "\n"
        "          F  E  D  C  B  A\n"
        f"          {top}\n"
        f" P2 [{board[13]:2d}]                  [{board[6]:2d}] P1\n"
        f"          {bottom}\n"
        "          A  B  C  D  E  F\n"
    )


@dataclass
class StepResult:
    next_player: int
    reward: float
    done: bool
    winner: int | None
    extra_turn: bool


class MancalaEnv:
    def __init__(self, legacy_rules: bool = True) -> None:
        self.board: List[int] = list(INITIAL_BOARD)
        self.current_player = 0
        self.legacy_rules = legacy_rules

    def reset(self) -> Tuple[int, ...]:
        self.board = list(INITIAL_BOARD)
        self.current_player = 0
        return canonical_state(self.board, self.current_player)

    def legal_actions(self, player: int | None = None) -> List[int]:
        player = self.current_player if player is None else player
        return [
            action
            for action in range(6)
            if self.board[action_to_board_index(player, action)] > 0
        ]

    def is_terminal(self) -> bool:
        return self._side_empty(0) or self._side_empty(1)

    def winner(self) -> int | None:
        if self.board[PLAYER_0_STORE] > self.board[PLAYER_1_STORE]:
            return 0
        if self.board[PLAYER_1_STORE] > self.board[PLAYER_0_STORE]:
            return 1
        return None

    def step(self, action: int) -> StepResult:
        player = self.current_player
        legal = self.legal_actions(player)
        if action not in legal:
            raise ValueError(f"Accion ilegal {action} para el jugador {player}.")

        own_store = player_store(player)
        opp_store = player_store(opponent(player))
        own_store_before = self.board[own_store]
        opp_store_before = self.board[opp_store]

        pit_index = action_to_board_index(player, action)
        seeds = self.board[pit_index]
        self.board[pit_index] = 0
        index = pit_index

        while seeds > 0:
            index = (index + 1) % 14
            if index == opp_store:
                continue
            self.board[index] += 1
            seeds -= 1

        extra_turn = index == own_store
        own_pit_indices = player_pits(player)
        if index in own_pit_indices and self.board[index] == 1:
            opposite = 12 - index
            captured = self.board[opposite]
            if captured > 0:
                if self.legacy_rules:
                    # Replica el comportamiento historico del proyecto:
                    # solo mueve las semillas del hoyo opuesto a la mancala
                    # y deja la ultima semilla en el hoyo donde cayo.
                    self.board[own_store] += captured
                else:
                    self.board[own_store] += captured + 1
                    self.board[index] = 0
                self.board[opposite] = 0

        done = False
        winner = None
        if self._side_empty(0) or self._side_empty(1):
            self._collect_remaining_seeds()
            done = True
            winner = self.winner()
        elif self.legacy_rules and self.board[PLAYER_0_STORE] >= 25:
            done = True
            winner = 0
        elif self.legacy_rules and self.board[PLAYER_1_STORE] >= 25:
            done = True
            winner = 1

        own_gain = self.board[own_store] - own_store_before
        opp_gain = self.board[opp_store] - opp_store_before
        reward = float(own_gain - opp_gain * 0.2)

        if done:
            if winner == player:
                reward += 20.0
            elif winner is None:
                reward += 5.0
            else:
                reward -= 20.0

        self.current_player = player if extra_turn and not done else opponent(player)
        return StepResult(
            next_player=self.current_player,
            reward=reward,
            done=done,
            winner=winner,
            extra_turn=extra_turn,
        )

    def _side_empty(self, player: int) -> bool:
        return all(self.board[index] == 0 for index in player_pits(player))

    def _collect_remaining_seeds(self) -> None:
        if not self._side_empty(0):
            remaining = sum(self.board[index] for index in PLAYER_0_PITS)
            for index in PLAYER_0_PITS:
                self.board[index] = 0
            self.board[PLAYER_0_STORE] += remaining
        if not self._side_empty(1):
            remaining = sum(self.board[index] for index in PLAYER_1_PITS)
            for index in PLAYER_1_PITS:
                self.board[index] = 0
            self.board[PLAYER_1_STORE] += remaining


class QLearningAgent:
    def __init__(
        self,
        alpha: float = 0.15,
        gamma: float = 0.98,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.9995,
    ) -> None:
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table: Dict[str, List[float]] = {}

    def q_values(self, state: Sequence[int]) -> List[float]:
        key = serialize_state(state)
        if key not in self.q_table:
            self.q_table[key] = [0.0] * 6
        return self.q_table[key]

    def choose_action(
        self,
        state: Sequence[int],
        legal_actions: Sequence[int],
        explore: bool = True,
    ) -> int:
        if not legal_actions:
            raise ValueError("No hay acciones legales disponibles.")
        if explore and random.random() < self.epsilon:
            return random.choice(list(legal_actions))

        q_values = self.q_values(state)
        best_value = max(q_values[action] for action in legal_actions)
        best_actions = [
            action for action in legal_actions if q_values[action] == best_value
        ]
        return random.choice(best_actions)

    def update(
        self,
        state: Sequence[int],
        action: int,
        reward: float,
        next_state: Sequence[int] | None,
        next_legal_actions: Sequence[int],
        done: bool,
    ) -> None:
        q_values = self.q_values(state)
        target = reward
        if not done and next_state is not None and next_legal_actions:
            next_q_values = self.q_values(next_state)
            target += self.gamma * max(
                next_q_values[next_action] for next_action in next_legal_actions
            )
        q_values[action] += self.alpha * (target - q_values[action])

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: Path) -> None:
        payload = {
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "q_table": self.q_table,
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "QLearningAgent":
        payload = json.loads(path.read_text(encoding="utf-8"))
        agent = cls(
            alpha=payload["alpha"],
            gamma=payload["gamma"],
            epsilon=payload["epsilon"],
            epsilon_min=payload["epsilon_min"],
            epsilon_decay=payload["epsilon_decay"],
        )
        agent.q_table = {
            key: [float(value) for value in values]
            for key, values in payload["q_table"].items()
        }
        return agent


def train_self_play(
    episodes: int,
    model_path: Path,
    alpha: float,
    gamma: float,
    epsilon: float,
    epsilon_min: float,
    epsilon_decay: float,
    report_every: int,
) -> None:
    env = MancalaEnv()
    agent = (
        QLearningAgent.load(model_path)
        if model_path.exists()
        else QLearningAgent(
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
            epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay,
        )
    )

    wins = [0, 0]
    draws = 0

    for episode in range(1, episodes + 1):
        state = env.reset()
        done = False

        while not done:
            player = env.current_player
            legal_actions = env.legal_actions(player)
            action = agent.choose_action(state, legal_actions, explore=True)
            result = env.step(action)
            next_state = None if result.done else canonical_state(env.board, result.next_player)
            next_legal_actions = [] if result.done else env.legal_actions(result.next_player)

            agent.update(
                state=state,
                action=action,
                reward=result.reward,
                next_state=next_state,
                next_legal_actions=next_legal_actions,
                done=result.done,
            )

            state = next_state if next_state is not None else ()
            done = result.done

        agent.decay_epsilon()
        if result.winner is None:
            draws += 1
        else:
            wins[result.winner] += 1

        if episode % report_every == 0 or episode == episodes:
            print(
                f"Episodios: {episode:6d} | "
                f"epsilon={agent.epsilon:.4f} | "
                f"victorias P1={wins[0]} P2={wins[1]} empates={draws} | "
                f"estados aprendidos={len(agent.q_table)}"
            )
            wins = [0, 0]
            draws = 0

    agent.save(model_path)
    print(f"Modelo guardado en: {model_path}")


def evaluate_agent(model_path: Path, games: int) -> None:
    agent = QLearningAgent.load(model_path)
    agent.epsilon = 0.0
    env = MancalaEnv()
    wins = [0, 0]
    draws = 0

    for _ in range(games):
        env.reset()
        done = False
        while not done:
            player = env.current_player
            state = canonical_state(env.board, player)
            action = agent.choose_action(state, env.legal_actions(player), explore=False)
            result = env.step(action)
            done = result.done
        if result.winner is None:
            draws += 1
        else:
            wins[result.winner] += 1

    print(
        f"Evaluacion en {games} partidas | "
        f"victorias P1={wins[0]} P2={wins[1]} empates={draws}"
    )


def play_against_agent(model_path: Path, human_first: bool) -> None:
    agent = QLearningAgent.load(model_path)
    agent.epsilon = 0.0
    env = MancalaEnv()
    env.reset()
    human_player = 0 if human_first else 1

    while True:
        print(render_board(env.board))
        player = env.current_player
        legal_actions = env.legal_actions(player)

        if player == human_player:
            print(f"Te toca. Jugadas legales: {[board_index_to_label(action_to_board_index(player, action)) for action in legal_actions]}")
            while True:
                try:
                    action = label_to_action(input("Elige A-F: "))
                except ValueError as error:
                    print(error)
                    continue
                if action in legal_actions:
                    break
                print("Esa jugada no es legal en este turno.")
        else:
            state = canonical_state(env.board, player)
            action = agent.choose_action(state, legal_actions, explore=False)
            label = board_index_to_label(action_to_board_index(player, action))
            print(f"La IA juega: {label}")

        result = env.step(action)
        if result.done:
            print(render_board(env.board))
            if result.winner is None:
                print("La partida termino en empate.")
            elif result.winner == human_player:
                print("Ganaste.")
            else:
                print("La IA gano.")
            return


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aprendizaje por refuerzo para Mancala.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Entrenar agente por autojuego.")
    train_parser.add_argument("--episodes", type=int, default=50000)
    train_parser.add_argument("--model", type=Path, default=Path("mancala_q_table.json"))
    train_parser.add_argument("--alpha", type=float, default=0.15)
    train_parser.add_argument("--gamma", type=float, default=0.98)
    train_parser.add_argument("--epsilon", type=float, default=1.0)
    train_parser.add_argument("--epsilon-min", type=float, default=0.05)
    train_parser.add_argument("--epsilon-decay", type=float, default=0.9995)
    train_parser.add_argument("--report-every", type=int, default=1000)

    eval_parser = subparsers.add_parser("evaluate", help="Evaluar el agente entrenado.")
    eval_parser.add_argument("--games", type=int, default=1000)
    eval_parser.add_argument("--model", type=Path, default=Path("mancala_q_table.json"))

    play_parser = subparsers.add_parser("play", help="Jugar contra el agente entrenado.")
    play_parser.add_argument("--model", type=Path, default=Path("mancala_q_table.json"))
    play_parser.add_argument(
        "--human-first",
        action="store_true",
        help="Si se indica, el humano juega primero.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        train_self_play(
            episodes=args.episodes,
            model_path=args.model,
            alpha=args.alpha,
            gamma=args.gamma,
            epsilon=args.epsilon,
            epsilon_min=args.epsilon_min,
            epsilon_decay=args.epsilon_decay,
            report_every=args.report_every,
        )
    elif args.command == "evaluate":
        evaluate_agent(model_path=args.model, games=args.games)
    elif args.command == "play":
        play_against_agent(model_path=args.model, human_first=args.human_first)


if __name__ == "__main__":
    main()
