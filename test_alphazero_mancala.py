from __future__ import annotations

import random
import tempfile
import unittest
from collections import deque
from pathlib import Path

import numpy as np

from alphazero_mancala import (
    ACTION_SIZE,
    AlphaZeroMCTS,
    AlphaZeroNetwork,
    softmax_masked,
    train_from_replay,
)
from alphazero_mancala import AlphaZeroExample
from mancala_dqn_selfplay import get_new_board, legal_actions


class AlphaZeroMancalaTest(unittest.TestCase):
    def test_mcts_policy_is_legal_distribution(self) -> None:
        random.seed(11)
        np.random.seed(11)
        board = get_new_board()
        network = AlphaZeroNetwork(hidden1=16, hidden2=16)
        result = AlphaZeroMCTS(network, simulations=12, rng=random.Random(11)).search(board, "1")

        legal = legal_actions("1", board)
        self.assertIn(result.action, legal)
        self.assertAlmostEqual(float(np.sum(result.policy)), 1.0)
        self.assertEqual(set(result.visits), set(legal))
        self.assertTrue(all(result.policy[action] == 0.0 for action in range(ACTION_SIZE) if action not in legal))

    def test_softmax_masks_illegal_actions(self) -> None:
        logits = np.arange(ACTION_SIZE, dtype=np.float64)
        policy = softmax_masked(logits, [1, 3, 5])

        self.assertAlmostEqual(float(np.sum(policy)), 1.0)
        self.assertEqual(float(policy[0]), 0.0)
        self.assertGreater(float(policy[5]), float(policy[3]))

    def test_train_batch_and_save_load(self) -> None:
        random.seed(17)
        np.random.seed(17)
        network = AlphaZeroNetwork(hidden1=16, hidden2=16)
        state = np.ones(15, dtype=np.float64)
        policy = np.zeros(ACTION_SIZE, dtype=np.float64)
        policy[2] = 1.0
        replay = [
            AlphaZeroExample(state=state, policy=policy, player_turn="1", value=1.0),
            AlphaZeroExample(state=state * 0.5, policy=policy, player_turn="2", value=-1.0),
        ]
        before_logits, before_value = network.predict(state)
        losses = train_from_replay(network, replay=deque(replay), steps=3, batch_size=2, lr=0.001, rng=random.Random(17))
        after_logits, after_value = network.predict(state)

        self.assertTrue(np.isfinite(losses["loss"]))
        self.assertFalse(np.allclose(before_logits, after_logits))
        self.assertFalse(np.allclose(before_value, after_value))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "az.npz"
            network.save(path)
            loaded = AlphaZeroNetwork(hidden1=16, hidden2=16)
            loaded.load(path)
            loaded_logits, loaded_value = loaded.predict(state)

        self.assertTrue(np.allclose(after_logits, loaded_logits))
        self.assertTrue(np.allclose(after_value, loaded_value))


if __name__ == "__main__":
    unittest.main()
