"""Worst-case optimization for robust policy learning.

Implements the min-max formulation from ``CIRC-RL_Framework.md`` Section 3.3:

    pi* = argmax_pi min_{e in E} E_{tau ~ rho_pi^e}[R(tau)]

Uses a soft-min approximation with temperature parameter for differentiability,
plus a variance penalty for stability.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class WorstCaseOptimizer(nn.Module):
    """Soft worst-case return optimization with variance penalty.

    Combines two objectives:
    1. Soft-min over per-environment returns (maximizes worst-case)
    2. Variance penalty (reduces performance spread)

    :param temperature: Temperature for soft-min (lower = harder min).
    :param variance_weight: Weight of the variance penalty.
    """

    def __init__(
        self,
        temperature: float = 1.0,
        variance_weight: float = 0.1,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self._temperature = temperature
        self._variance_weight = variance_weight

    @property
    def temperature(self) -> float:
        """Current soft-min temperature."""
        return self._temperature

    def soft_min(self, env_returns: torch.Tensor) -> torch.Tensor:
        r"""Compute differentiable soft-min of per-environment returns.

        .. math::

            \text{soft\_min}(r_1, \ldots, r_n) =
                \frac{\sum_e r_e \exp(-r_e / \tau)}{\sum_e \exp(-r_e / \tau)}

        :param env_returns: Per-environment expected returns, shape ``(n_envs,)``.
        :returns: Scalar soft-min value.
        """
        weights = torch.softmax(-env_returns / self._temperature, dim=0)
        return (weights * env_returns).sum()

    def compute_loss(self, env_returns: torch.Tensor) -> torch.Tensor:
        """Compute the worst-case loss (to be minimized).

        Returns negative soft-min (since we maximize returns but minimize loss)
        plus variance penalty.

        :param env_returns: Per-environment expected returns, shape ``(n_envs,)``.
        :returns: Scalar loss.
        """
        soft_min_val = self.soft_min(env_returns)
        variance = env_returns.var()

        # Negate soft-min because optimizer minimizes, and we want max worst-case
        return -soft_min_val + self._variance_weight * variance

    def forward(self, env_returns: torch.Tensor) -> torch.Tensor:
        """Compute worst-case loss (alias for ``compute_loss``).

        :param env_returns: Per-environment expected returns.
        :returns: Scalar loss.
        """
        return self.compute_loss(env_returns)
