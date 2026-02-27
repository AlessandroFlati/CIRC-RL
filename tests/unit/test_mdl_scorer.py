"""Tests for circ_rl.evaluation.mdl_scorer."""

import numpy as np
import torch

from circ_rl.evaluation.mdl_scorer import MDLScorer
from circ_rl.policy.causal_policy import CausalPolicy
from circ_rl.training.trajectory_buffer import Trajectory


def _make_policy(hidden: tuple[int, ...] = (32, 32)) -> CausalPolicy:
    mask = np.ones(4, dtype=bool)
    return CausalPolicy(
        full_state_dim=4, action_dim=2, feature_mask=mask, hidden_dims=hidden
    )


def _make_trajectory(length: int = 100) -> Trajectory:
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


class TestMDLScorer:
    def test_score_is_positive(self) -> None:
        scorer = MDLScorer()
        policy = _make_policy()
        traj = _make_trajectory()
        score = scorer.score(policy, traj)
        assert score.total > 0.0

    def test_simpler_policy_lower_complexity(self) -> None:
        scorer = MDLScorer(complexity_weight=1.0)
        small = _make_policy(hidden=(8,))
        large = _make_policy(hidden=(128, 128))
        traj = _make_trajectory()
        s_small = scorer.score(small, traj)
        s_large = scorer.score(large, traj)
        assert s_small.complexity < s_large.complexity
        assert s_small.n_parameters < s_large.n_parameters

    def test_rank_policies(self) -> None:
        scorer = MDLScorer()
        policies = [_make_policy() for _ in range(3)]
        traj = _make_trajectory()
        ranked = scorer.rank_policies(policies, traj)
        assert len(ranked) == 3
        # Should be sorted by total score
        for i in range(len(ranked) - 1):
            assert ranked[i][1].total <= ranked[i + 1][1].total

    def test_score_breakdown(self) -> None:
        scorer = MDLScorer()
        policy = _make_policy()
        traj = _make_trajectory()
        score = scorer.score(policy, traj)
        assert abs(score.total - (score.data_fit + score.complexity)) < 1e-5
