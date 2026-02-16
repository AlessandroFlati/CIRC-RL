"""Collection of constraints with batch evaluation.

Groups multiple ConstraintFunction instances for convenient evaluation
during training and verification.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from circ_rl.constraints.const_definition import ConstraintFunction


class ConstraintSet:
    """A collection of constraint functions.

    :param constraints: List of constraint functions.
    """

    def __init__(self, constraints: list[ConstraintFunction]) -> None:
        self._constraints = list(constraints)

    @property
    def n_constraints(self) -> int:
        """Number of constraints in the set."""
        return len(self._constraints)

    @property
    def constraints(self) -> list[ConstraintFunction]:
        """The list of constraints."""
        return list(self._constraints)

    @property
    def names(self) -> list[str]:
        """Names of all constraints."""
        return [c.name for c in self._constraints]

    def evaluate_all(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Evaluate all constraints and return per-step costs.

        :returns: List of cost tensors, each of shape ``(batch,)``.
        """
        return [
            c.evaluate(states, actions, rewards, next_states)
            for c in self._constraints
        ]

    def violation_vector(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
    ) -> torch.Tensor:
        """Compute violation (E[C_i] - threshold) for each constraint.

        :returns: Violation tensor of shape ``(n_constraints,)``.
            Positive values indicate violation.
        """
        violations = [
            c.violation(states, actions, rewards, next_states)
            for c in self._constraints
        ]
        return torch.tensor(violations)

    def all_satisfied(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
    ) -> bool:
        """Check if all constraints are satisfied.

        :returns: True if every constraint has E[C_i] <= threshold.
        """
        return all(
            c.is_satisfied(states, actions, rewards, next_states)
            for c in self._constraints
        )

    def __len__(self) -> int:
        return len(self._constraints)

    def __repr__(self) -> str:
        return f"ConstraintSet(n_constraints={len(self._constraints)})"
