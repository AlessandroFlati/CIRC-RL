"""Causal Q-network restricted to causal features.

Implements the causal action-value function from ``CIRC-RL_Framework.md``
Section 3.2:

    Q_causal(s, a) = E[R_t + gamma V_causal(S_{t+1}) | S_t = s, do(A_t = a)]

The network only receives features identified as causal ancestors of reward
(Phase 2 output), discarding spurious correlations.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np


class CausalQNetwork(nn.Module):
    """Q-network that operates on causal features only.

    :param full_state_dim: Dimensionality of the full state space.
    :param action_dim: Number of discrete actions.
    :param feature_mask: Boolean array of shape ``(full_state_dim,)``
        indicating which state dimensions are causal features.
    :param hidden_dims: Sizes of hidden layers.
    :param activation: Activation function class.
    """

    def __init__(
        self,
        full_state_dim: int,
        action_dim: int,
        feature_mask: np.ndarray,
        hidden_dims: tuple[int, ...] = (256, 256),
        activation: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()

        if feature_mask.shape != (full_state_dim,):
            raise ValueError(
                f"feature_mask shape {feature_mask.shape} does not match "
                f"full_state_dim ({full_state_dim},)"
            )

        self.register_buffer(
            "_feature_mask",
            torch.from_numpy(feature_mask.astype(np.bool_)),
        )
        self._causal_dim = int(feature_mask.sum())
        self._action_dim = action_dim

        if self._causal_dim == 0:
            raise ValueError("feature_mask selects zero features")

        # Build MLP: causal_dim -> hidden -> ... -> action_dim
        layers: list[nn.Module] = []
        in_dim = self._causal_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(activation())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, action_dim))

        self._network = nn.Sequential(*layers)

    @property
    def causal_dim(self) -> int:
        """Number of causal features used as input."""
        return self._causal_dim

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Compute Q-values for all actions given a full state.

        :param state: Full state tensor of shape ``(batch, full_state_dim)``.
        :returns: Q-values of shape ``(batch, action_dim)``.
        """
        # Apply causal feature mask
        causal_features = state[:, self._feature_mask]  # (batch, causal_dim)
        return self._network(causal_features)  # (batch, action_dim)

    def q_value(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute Q-value for specific actions.

        :param state: Full state tensor of shape ``(batch, full_state_dim)``.
        :param action: Action indices of shape ``(batch,)``.
        :returns: Q-values of shape ``(batch,)``.
        """
        q_all = self.forward(state)  # (batch, action_dim)
        return q_all.gather(1, action.unsqueeze(1).long()).squeeze(1)  # (batch,)
