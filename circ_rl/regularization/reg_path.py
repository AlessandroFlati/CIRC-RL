"""Path complexity: C_path = E[sum d(a_t, a_{t+1})].

Implements the path complexity penalty from ``CIRC-RL_Framework.md``
Section 2.3. Penalizes policies that produce erratic action sequences,
encouraging smooth temporal behavior.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PathComplexity(nn.Module):
    """Action smoothness penalty over trajectories.

    For discrete actions, uses indicator distance (different action = 1).
    For continuous actions, uses L2 distance between consecutive actions.

    .. math::

        C_{\\text{path}}(\\pi) = \\mathbb{E}\\left[
            \\sum_{t=0}^{T-1} d(a_t, a_{t+1})
        \\right]

    :param weight: Scaling factor for the penalty.
    """

    def __init__(self, weight: float = 0.001) -> None:
        super().__init__()
        self._weight = weight

    def compute_discrete(self, actions: torch.Tensor) -> torch.Tensor:
        """Compute path complexity for discrete action sequences.

        :param actions: Action indices of shape ``(batch, timesteps)``.
        :returns: Scalar path complexity penalty.
        """
        if actions.shape[1] < 2:
            return torch.tensor(0.0, device=actions.device)

        # Count action changes between consecutive timesteps
        changes = (actions[:, 1:] != actions[:, :-1]).float()  # (batch, T-1)
        return self._weight * changes.mean()

    def compute_continuous(self, actions: torch.Tensor) -> torch.Tensor:
        """Compute path complexity for continuous action sequences.

        :param actions: Action tensor of shape ``(batch, timesteps, action_dim)``.
        :returns: Scalar path complexity penalty.
        """
        if actions.shape[1] < 2:
            return torch.tensor(0.0, device=actions.device)

        diffs = actions[:, 1:] - actions[:, :-1]  # (batch, T-1, action_dim)
        distances = diffs.pow(2).sum(dim=-1).sqrt()  # (batch, T-1)
        return self._weight * distances.mean()

    def forward(self, actions: torch.Tensor, discrete: bool = True) -> torch.Tensor:
        """Compute path complexity.

        :param actions: Action sequence tensor.
        :param discrete: Whether actions are discrete indices.
        :returns: Scalar penalty.
        """
        if discrete:
            return self.compute_discrete(actions)
        return self.compute_continuous(actions)
