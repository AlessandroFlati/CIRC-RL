"""Lagrange multiplier management for constrained optimization.

Implements the dual update from ``CIRC-RL_Framework.md`` Section 3.5:

    lambda_i <- max(0, lambda_i + eta * (E[C_i(tau)] - delta_i))

and the Lagrangian penalty:

    L(pi, {lambda_i}) = E[R(tau) - sum_i lambda_i * C_i(tau)]
"""

from __future__ import annotations

import torch
from loguru import logger

from circ_rl.constraints.const_set import ConstraintSet


class LagrangeMultiplierManager:
    """Manages Lagrange multipliers for constrained optimization.

    :param constraint_set: The set of constraints.
    :param initial_value: Initial value for all multipliers.
    :param learning_rate: Learning rate for dual updates (eta).
    """

    def __init__(
        self,
        constraint_set: ConstraintSet,
        initial_value: float = 0.0,
        learning_rate: float = 0.01,
    ) -> None:
        self._constraint_set = constraint_set
        self._lr = learning_rate
        self._multipliers = torch.full(
            (constraint_set.n_constraints,),
            initial_value,
            dtype=torch.float32,
        )

    @property
    def multipliers(self) -> torch.Tensor:
        """Current Lagrange multipliers, shape ``(n_constraints,)``."""
        return self._multipliers.clone()

    @property
    def learning_rate(self) -> float:
        """Dual update learning rate."""
        return self._lr

    def dual_update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
    ) -> None:
        r"""Update multipliers via projected gradient ascent on the dual.

        .. math::

            \lambda_i \leftarrow \max(0, \lambda_i + \eta (E[C_i] - \delta_i))

        :param states: States from the current batch.
        :param actions: Actions from the current batch.
        :param rewards: Rewards from the current batch.
        :param next_states: Next states from the current batch.
        """
        violations = self._constraint_set.violation_vector(
            states, actions, rewards, next_states
        )  # (n_constraints,)

        new_multipliers = torch.clamp(
            self._multipliers + self._lr * violations, min=0.0
        )

        for i, (old, new) in enumerate(
            zip(self._multipliers.tolist(), new_multipliers.tolist())
        ):
            if abs(old - new) > 1e-6:
                logger.debug(
                    "Lagrange multiplier '{}': {:.4f} -> {:.4f} (violation={:.4f})",
                    self._constraint_set.names[i],
                    old,
                    new,
                    violations[i].item(),
                )

        self._multipliers = new_multipliers

    def compute_lagrangian_penalty(
        self,
        constraint_costs: list[torch.Tensor],
    ) -> torch.Tensor:
        r"""Compute the Lagrangian penalty: :math:`\sum_i \lambda_i \cdot E[C_i]`.

        :param constraint_costs: Per-step costs for each constraint,
            each of shape ``(batch,)``.
        :returns: Scalar penalty to add to the loss.
        """
        penalty = torch.tensor(0.0)
        if len(constraint_costs) > 0:
            penalty = penalty.to(constraint_costs[0].device)

        for i, costs in enumerate(constraint_costs):
            penalty = penalty + self._multipliers[i] * costs.mean()

        return penalty

    def __repr__(self) -> str:
        return (
            f"LagrangeMultiplierManager("
            f"n_constraints={self._constraint_set.n_constraints}, "
            f"multipliers={self._multipliers.tolist()}, "
            f"lr={self._lr})"
        )
