"""Tests for circ_rl.evaluation.const_verifier."""

import numpy as np
import torch

from circ_rl.constraints.const_definition import StateBoundConstraint
from circ_rl.constraints.const_set import ConstraintSet
from circ_rl.evaluation.const_verifier import ConstraintVerifier
from circ_rl.policy.causal_policy import CausalPolicy
from circ_rl.training.trajectory_buffer import Trajectory


def _make_policy() -> CausalPolicy:
    mask = np.ones(4, dtype=bool)
    return CausalPolicy(
        full_state_dim=4, action_dim=2, feature_mask=mask, hidden_dims=(16,)
    )


def _make_trajectory_in_bounds() -> Trajectory:
    """Trajectory where state dim 0 is in [-1, 1]."""
    states = torch.rand(100, 4) * 2 - 1  # Uniform [-1, 1]
    return Trajectory(
        states=states,
        actions=torch.randint(0, 2, (100,)),
        rewards=torch.randn(100),
        log_probs=torch.randn(100),
        values=torch.randn(100),
        next_states=states,
        dones=torch.zeros(100),
        env_id=0,
    )


def _make_trajectory_out_of_bounds() -> Trajectory:
    """Trajectory where state dim 0 is in [5, 10] -- violating upper=2."""
    states = torch.rand(100, 4) * 5 + 5
    return Trajectory(
        states=states,
        actions=torch.randint(0, 2, (100,)),
        rewards=torch.randn(100),
        log_probs=torch.randn(100),
        values=torch.randn(100),
        next_states=states,
        dones=torch.zeros(100),
        env_id=0,
    )


class TestConstraintVerifier:
    def test_feasible_policy_accepted(self) -> None:
        constraint = StateBoundConstraint(
            "bound", threshold=0.1, state_dim_idx=0, lower=-2.0, upper=2.0
        )
        cs = ConstraintSet([constraint])
        verifier = ConstraintVerifier(cs, confidence_margin_k=2.0)

        policy = _make_policy()
        traj = _make_trajectory_in_bounds()
        result = verifier.verify(policy, traj)
        assert result.feasible
        assert len(result.margin_violations) == 0

    def test_infeasible_policy_rejected(self) -> None:
        constraint = StateBoundConstraint(
            "bound", threshold=0.0, state_dim_idx=0, upper=2.0
        )
        cs = ConstraintSet([constraint])
        verifier = ConstraintVerifier(cs, confidence_margin_k=2.0)

        policy = _make_policy()
        traj = _make_trajectory_out_of_bounds()
        result = verifier.verify(policy, traj)
        assert not result.feasible
        assert "bound" in result.margin_violations

    def test_filter_feasible(self) -> None:
        constraint = StateBoundConstraint(
            "bound", threshold=0.1, state_dim_idx=0, lower=-2.0, upper=2.0
        )
        cs = ConstraintSet([constraint])
        verifier = ConstraintVerifier(cs, confidence_margin_k=2.0)

        policies = [_make_policy() for _ in range(3)]
        traj = _make_trajectory_in_bounds()
        feasible = verifier.filter_feasible(policies, traj)
        # All should be feasible with in-bounds trajectory
        assert len(feasible) == 3

    def test_filter_with_violations(self) -> None:
        constraint = StateBoundConstraint(
            "bound", threshold=0.0, state_dim_idx=0, upper=2.0
        )
        cs = ConstraintSet([constraint])
        verifier = ConstraintVerifier(cs, confidence_margin_k=0.0)

        policies = [_make_policy(), _make_policy()]
        # First trajectory in bounds, second out of bounds
        traj_in = _make_trajectory_in_bounds()
        traj_out = _make_trajectory_out_of_bounds()

        # Both should be feasible with in-bounds data
        feasible_in = verifier.filter_feasible(policies, traj_in)
        assert len(feasible_in) == 2

        # Both should fail with out-of-bounds data
        feasible_out = verifier.filter_feasible(policies, traj_out)
        assert len(feasible_out) == 0
