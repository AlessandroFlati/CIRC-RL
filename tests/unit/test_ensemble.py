"""Tests for circ_rl.evaluation.ensemble."""

import numpy as np
import pytest
import torch

from circ_rl.evaluation.ensemble import EnsemblePolicy
from circ_rl.evaluation.mdl_scorer import MDLScore
from circ_rl.policy.causal_policy import CausalPolicy
from circ_rl.training.trajectory_buffer import Trajectory


def _make_policy() -> CausalPolicy:
    mask = np.ones(4, dtype=bool)
    return CausalPolicy(
        full_state_dim=4, action_dim=2, feature_mask=mask, hidden_dims=(16,)
    )


def _make_trajectory(length: int = 50) -> Trajectory:
    return Trajectory(
        states=torch.randn(length, 4),
        actions=torch.randint(0, 2, (length,)),
        rewards=torch.randn(length),
        log_probs=torch.randn(length),
        values=torch.randn(length),
        next_states=torch.randn(length, 4),
        dones=torch.zeros(length),
        env_id=0,
    )


class TestEnsemblePolicy:
    def test_weights_sum_to_one(self) -> None:
        policies = [_make_policy(), _make_policy()]
        scores = [
            MDLScore(total=1.0, data_fit=0.8, complexity=0.2, n_parameters=100),
            MDLScore(total=2.0, data_fit=1.5, complexity=0.5, n_parameters=100),
        ]
        weights = np.array([0.7, 0.3])
        ensemble = EnsemblePolicy(policies, weights, scores)
        assert abs(ensemble.weights.sum() - 1.0) < 1e-6

    def test_lower_mdl_gets_higher_weight(self) -> None:
        policies = [_make_policy(), _make_policy(), _make_policy()]
        traj = _make_trajectory(100)
        ensemble = EnsemblePolicy.from_mdl_scores(policies, traj)

        # The policy with lowest MDL should get highest weight
        best_idx = int(np.argmax(ensemble.weights))
        best_score = ensemble.scores[best_idx].total
        for i, s in enumerate(ensemble.scores):
            if i != best_idx:
                assert s.total >= best_score

    def test_get_action_returns_valid(self) -> None:
        policies = [_make_policy(), _make_policy()]
        scores = [
            MDLScore(total=1.0, data_fit=0.8, complexity=0.2, n_parameters=100),
            MDLScore(total=2.0, data_fit=1.5, complexity=0.5, n_parameters=100),
        ]
        weights = np.array([0.5, 0.5])
        ensemble = EnsemblePolicy(policies, weights, scores)

        state = torch.randn(4)
        action = ensemble.get_action(state)
        assert 0 <= action < 2

    def test_deterministic_action(self) -> None:
        policies = [_make_policy(), _make_policy()]
        weights = np.array([0.9, 0.1])
        scores = [
            MDLScore(total=1.0, data_fit=0.8, complexity=0.2, n_parameters=100),
            MDLScore(total=2.0, data_fit=1.5, complexity=0.5, n_parameters=100),
        ]
        ensemble = EnsemblePolicy(policies, weights, scores)

        state = torch.randn(4)
        a1 = ensemble.get_action(state, deterministic=True)
        a2 = ensemble.get_action(state, deterministic=True)
        assert a1 == a2

    def test_evaluate_returns_log_probs(self) -> None:
        policies = [_make_policy(), _make_policy()]
        weights = np.array([0.5, 0.5])
        scores = [
            MDLScore(total=1.0, data_fit=0.8, complexity=0.2, n_parameters=100),
            MDLScore(total=1.0, data_fit=0.8, complexity=0.2, n_parameters=100),
        ]
        ensemble = EnsemblePolicy(policies, weights, scores)

        states = torch.randn(8, 4)
        actions = torch.randint(0, 2, (8,))
        log_probs = ensemble.evaluate(states, actions)
        assert log_probs.shape == (8,)
        assert (log_probs <= 0.0).all()

    def test_rejects_empty_policies(self) -> None:
        with pytest.raises(ValueError, match="zero policies"):
            EnsemblePolicy([], np.array([]), [])

    def test_rejects_mismatched_lengths(self) -> None:
        policies = [_make_policy()]
        weights = np.array([0.5, 0.5])
        with pytest.raises(ValueError, match="must match"):
            EnsemblePolicy(policies, weights, [])
