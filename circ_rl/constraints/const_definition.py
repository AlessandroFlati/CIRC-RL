"""Constraint definitions for constrained MDP optimization.

Implements constraint types from ``CIRC-RL_Framework.md`` Section 3.5:

    E_{tau ~ rho_pi}[C_i(tau)] <= delta_i  for all i in {1, ..., m}

Provides abstract base class and concrete implementations for
expected-cost and state-bound constraints.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import torch

# Type alias for cost functions
CostCallable = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    torch.Tensor,
]


class ConstraintFunction(ABC):
    """Abstract base class for constraint functions.

    A constraint C_i is satisfied when E[C_i(tau)] <= threshold.

    :param name: Human-readable name for the constraint.
    :param threshold: Maximum allowed expected cost (delta_i).
    """

    def __init__(self, name: str, threshold: float) -> None:
        self._name = name
        self._threshold = threshold

    @property
    def name(self) -> str:
        """Name of this constraint."""
        return self._name

    @property
    def threshold(self) -> float:
        """Maximum allowed expected cost."""
        return self._threshold

    @abstractmethod
    def evaluate(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-step constraint cost.

        :param states: States, shape ``(batch, state_dim)``.
        :param actions: Actions, shape ``(batch,)`` or ``(batch, action_dim)``.
        :param rewards: Rewards, shape ``(batch,)``.
        :param next_states: Next states, shape ``(batch, state_dim)``.
        :returns: Per-step cost, shape ``(batch,)``.
        """

    def violation(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
    ) -> float:
        """Compute constraint violation: E[C_i] - threshold.

        Positive values indicate violation.

        :returns: Scalar violation value.
        """
        costs = self.evaluate(states, actions, rewards, next_states)
        return float(costs.mean().item() - self._threshold)

    def is_satisfied(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
    ) -> bool:
        """Check if the constraint is satisfied.

        :returns: True if E[C_i] <= threshold.
        """
        return self.violation(states, actions, rewards, next_states) <= 0.0

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name='{self._name}', threshold={self._threshold})"


class ExpectedCostConstraint(ConstraintFunction):
    """Constraint on expected cumulative cost.

    The cost is a user-defined function of (state, action, reward, next_state).

    :param name: Constraint name.
    :param threshold: Maximum allowed expected cost.
    :param cost_fn: Function computing per-step cost.
    """

    def __init__(
        self,
        name: str,
        threshold: float,
        cost_fn: CostCallable,
    ) -> None:
        super().__init__(name, threshold)
        self._cost_fn = cost_fn

    def evaluate(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-step cost via the user-provided function."""
        return self._cost_fn(states, actions, rewards, next_states)


class StateBoundConstraint(ConstraintFunction):
    """Constraint that a specific state dimension stays within bounds.

    Cost = max(0, state[dim] - upper) + max(0, lower - state[dim]).

    :param name: Constraint name.
    :param threshold: Maximum allowed expected bound violation (typically 0).
    :param state_dim_idx: Index of the state dimension to constrain.
    :param lower: Lower bound (None for no lower bound).
    :param upper: Upper bound (None for no upper bound).
    """

    def __init__(
        self,
        name: str,
        threshold: float,
        state_dim_idx: int,
        lower: float | None = None,
        upper: float | None = None,
    ) -> None:
        super().__init__(name, threshold)
        if lower is None and upper is None:
            raise ValueError("At least one of lower or upper must be specified")
        self._dim_idx = state_dim_idx
        self._lower = lower
        self._upper = upper

    def evaluate(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-step bound violation cost."""
        val = states[:, self._dim_idx]  # (batch,)
        cost = torch.zeros_like(val)

        if self._upper is not None:
            cost = cost + torch.clamp(val - self._upper, min=0.0)
        if self._lower is not None:
            cost = cost + torch.clamp(self._lower - val, min=0.0)

        return cost
