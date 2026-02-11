"""Tests for circ_rl.constraints components."""

import torch
import pytest

from circ_rl.constraints.const_definition import (
    ExpectedCostConstraint,
    StateBoundConstraint,
)
from circ_rl.constraints.const_set import ConstraintSet
from circ_rl.constraints.const_lagrange import LagrangeMultiplierManager


def _dummy_tensors(
    batch: int = 16, state_dim: int = 4
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    states = torch.randn(batch, state_dim)
    actions = torch.randint(0, 2, (batch,))
    rewards = torch.randn(batch)
    next_states = torch.randn(batch, state_dim)
    return states, actions, rewards, next_states


class TestStateBoundConstraint:
    def test_within_bounds_zero_cost(self) -> None:
        c = StateBoundConstraint("test", threshold=0.0, state_dim_idx=0, lower=-10.0, upper=10.0)
        states = torch.zeros(8, 4)
        s, a, r, ns = _dummy_tensors(8)
        cost = c.evaluate(states, a, r, ns)
        assert (cost == 0.0).all()

    def test_upper_violation(self) -> None:
        c = StateBoundConstraint("test", threshold=0.0, state_dim_idx=0, upper=1.0)
        states = torch.tensor([[2.0, 0.0], [0.5, 0.0]])
        a = torch.zeros(2, dtype=torch.long)
        r = torch.zeros(2)
        ns = states
        cost = c.evaluate(states, a, r, ns)
        assert cost[0].item() == pytest.approx(1.0)
        assert cost[1].item() == 0.0

    def test_lower_violation(self) -> None:
        c = StateBoundConstraint("test", threshold=0.0, state_dim_idx=1, lower=0.0)
        states = torch.tensor([[0.0, -0.5], [0.0, 0.5]])
        a = torch.zeros(2, dtype=torch.long)
        r = torch.zeros(2)
        ns = states
        cost = c.evaluate(states, a, r, ns)
        assert cost[0].item() == pytest.approx(0.5)
        assert cost[1].item() == 0.0

    def test_requires_at_least_one_bound(self) -> None:
        with pytest.raises(ValueError, match="At least one"):
            StateBoundConstraint("test", threshold=0.0, state_dim_idx=0)

    def test_is_satisfied(self) -> None:
        c = StateBoundConstraint("test", threshold=0.0, state_dim_idx=0, upper=5.0)
        states = torch.zeros(8, 2)
        s, a, r, ns = _dummy_tensors(8, 2)
        assert c.is_satisfied(states, a, r, ns)


class TestExpectedCostConstraint:
    def test_custom_cost_fn(self) -> None:
        def cost_fn(
            states: torch.Tensor,
            actions: torch.Tensor,
            rewards: torch.Tensor,
            next_states: torch.Tensor,
        ) -> torch.Tensor:
            return rewards.abs()

        c = ExpectedCostConstraint("abs_reward", threshold=1.0, cost_fn=cost_fn)
        s, a, r, ns = _dummy_tensors()
        cost = c.evaluate(s, a, r, ns)
        assert cost.shape == r.shape


class TestConstraintSet:
    def test_evaluate_all(self) -> None:
        c1 = StateBoundConstraint("c1", threshold=0.0, state_dim_idx=0, upper=5.0)
        c2 = StateBoundConstraint("c2", threshold=0.0, state_dim_idx=1, lower=-5.0)
        cs = ConstraintSet([c1, c2])

        assert cs.n_constraints == 2
        assert cs.names == ["c1", "c2"]

        s, a, r, ns = _dummy_tensors()
        costs = cs.evaluate_all(s, a, r, ns)
        assert len(costs) == 2

    def test_all_satisfied(self) -> None:
        c = StateBoundConstraint("test", threshold=100.0, state_dim_idx=0, upper=100.0)
        cs = ConstraintSet([c])
        states = torch.zeros(8, 4)
        a = torch.zeros(8, dtype=torch.long)
        r = torch.zeros(8)
        assert cs.all_satisfied(states, a, r, states)

    def test_violation_vector(self) -> None:
        c = StateBoundConstraint("test", threshold=0.0, state_dim_idx=0, upper=0.0)
        cs = ConstraintSet([c])
        states = torch.ones(8, 2)  # All violating
        a = torch.zeros(8, dtype=torch.long)
        r = torch.zeros(8)
        violations = cs.violation_vector(states, a, r, states)
        assert violations.shape == (1,)
        assert violations[0].item() > 0.0


class TestLagrangeMultiplierManager:
    def test_initial_multipliers_are_initial_value(self) -> None:
        c = StateBoundConstraint("test", threshold=0.0, state_dim_idx=0, upper=5.0)
        cs = ConstraintSet([c])
        lm = LagrangeMultiplierManager(cs, initial_value=0.5)
        assert lm.multipliers[0].item() == pytest.approx(0.5)

    def test_dual_update_increases_on_violation(self) -> None:
        c = StateBoundConstraint("test", threshold=0.0, state_dim_idx=0, upper=0.0)
        cs = ConstraintSet([c])
        lm = LagrangeMultiplierManager(cs, initial_value=0.0, learning_rate=0.1)

        states = torch.ones(8, 2)  # Violating
        a = torch.zeros(8, dtype=torch.long)
        r = torch.zeros(8)
        lm.dual_update(states, a, r, states)

        assert lm.multipliers[0].item() > 0.0

    def test_multipliers_stay_nonnegative(self) -> None:
        c = StateBoundConstraint("test", threshold=100.0, state_dim_idx=0, upper=100.0)
        cs = ConstraintSet([c])
        lm = LagrangeMultiplierManager(cs, initial_value=0.0, learning_rate=0.1)

        states = torch.zeros(8, 2)
        a = torch.zeros(8, dtype=torch.long)
        r = torch.zeros(8)
        lm.dual_update(states, a, r, states)

        assert lm.multipliers[0].item() >= 0.0

    def test_lagrangian_penalty(self) -> None:
        c = StateBoundConstraint("test", threshold=0.0, state_dim_idx=0, upper=0.0)
        cs = ConstraintSet([c])
        lm = LagrangeMultiplierManager(cs, initial_value=1.0)

        costs = [torch.ones(8) * 2.0]
        penalty = lm.compute_lagrangian_penalty(costs)
        assert penalty.item() == pytest.approx(2.0)  # lambda=1 * mean(cost)=2

    def test_zero_multiplier_zero_penalty(self) -> None:
        c = StateBoundConstraint("test", threshold=0.0, state_dim_idx=0, upper=0.0)
        cs = ConstraintSet([c])
        lm = LagrangeMultiplierManager(cs, initial_value=0.0)

        costs = [torch.ones(8) * 5.0]
        penalty = lm.compute_lagrangian_penalty(costs)
        assert penalty.item() == pytest.approx(0.0)
