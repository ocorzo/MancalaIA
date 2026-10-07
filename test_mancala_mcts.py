from __future__ import annotations

import random
import unittest

import numpy as np

from mancala_dqn_selfplay import get_new_board, legal_actions, make_move
from mancala_mcts import MCTSPlayer


class FlatModel:
    def predict(self, state: np.ndarray) -> np.ndarray:
        # Deliberately prefer the first legal action after masking in MCTS.
        return np.zeros((1, 12), dtype=np.float64)


class MancalaMCTSTest(unittest.TestCase):
    def test_search_returns_legal_action_and_does_not_mutate_board(self) -> None:
        board = get_new_board()
        original = board.copy()
        result = MCTSPlayer(model=FlatModel(), simulations=30, rng=random.Random(7)).search(board, "1")

        self.assertIn(result.action, legal_actions("1", board))
        self.assertEqual(board, original)
        self.assertEqual(result.simulations, 30)
        self.assertEqual(sum(result.visits.values()), 30)
        self.assertEqual(set(result.visits), set(legal_actions("1", board)))

    def test_search_is_reproducible_with_seed(self) -> None:
        board = get_new_board()
        first = MCTSPlayer(model=FlatModel(), simulations=40, rng=random.Random(123)).search(board, "1")
        second = MCTSPlayer(model=FlatModel(), simulations=40, rng=random.Random(123)).search(board, "1")

        self.assertEqual(first, second)

    def test_extra_turn_is_preserved_by_rules(self) -> None:
        board = {"1": 0, "2": 0}
        board.update({pit: 0 for pit in "ABCDEFGHIJKL"})
        board["F"] = 1
        board["G"] = 1

        next_player = make_move(board, "1", "F")

        self.assertEqual(next_player, "1")
        self.assertEqual(board["1"], 1)
        self.assertEqual(board["G"], 1)

    def test_invalid_configuration_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            MCTSPlayer(simulations=0)
        with self.assertRaises(ValueError):
            MCTSPlayer(rollout_randomness=1.1)


if __name__ == "__main__":
    unittest.main()
