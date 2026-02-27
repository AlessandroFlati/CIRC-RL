"""Bounded residual correction policy.

Learns a small correction to the analytic policy for unexplained
variance, with magnitude bounded proportionally to the analytic
action magnitude.

See ``CIRC-RL_Framework.md`` Section 3.7 (Phase 6: Bounded
Residual Learning).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class ResidualPolicyOutput:
    """Output from a residual policy forward pass.

    :param delta_action: Bounded correction, shape ``(batch, action_dim)``.
    :param value: State value estimate, shape ``(batch,)``.
    :param log_prob: Log-probability of the sampled correction, shape ``(batch,)``.
    :param entropy: Entropy of the correction distribution, shape ``(batch,)``.
    :param raw_output: Unbounded raw network output before bounding,
        shape ``(batch, action_dim)``.
    """

    delta_action: torch.Tensor
    value: torch.Tensor
    log_prob: torch.Tensor
    entropy: torch.Tensor
    raw_output: torch.Tensor


class ResidualPolicy(nn.Module):
    r"""Bounded residual correction policy.

    Produces a correction :math:`\delta a` bounded by:

    .. math::

        \delta a = \eta_{\max} \cdot \tanh(\text{raw}) \cdot |a_{\text{analytic}}|

    Key constraints (from ``CIRC-RL_Framework.md`` Section 3.7):

    1. **Bounded magnitude**: correction cannot exceed
       :math:`\eta_{\max}` fraction of the analytic action.
    2. **No environment parameter access**: the residual depends only
       on state, not on :math:`\theta_e`. All environment-dependent
       adaptation is handled by the analytic component.
    3. **Complexity penalty**: standard MDL regularization on the network.

    :param state_dim: State dimensionality.
    :param action_dim: Action dimensionality.
    :param hidden_dims: Hidden layer sizes.
    :param eta_max: Maximum correction fraction. Default 0.1.
    """

    _LOG_STD_MIN = -5.0
    _LOG_STD_MAX = 0.0

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: tuple[int, ...] = (64, 64),
        eta_max: float = 0.1,
    ) -> None:
        super().__init__()
        if eta_max <= 0 or eta_max > 1:
            raise ValueError(f"eta_max must be in (0, 1], got {eta_max}")

        self._state_dim = state_dim
        self._action_dim = action_dim
        self._eta_max = eta_max

        # Shared trunk (state only -- no env params)
        layers: list[nn.Module] = []
        in_dim = state_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.Tanh())
            in_dim = h_dim
        self._trunk = nn.Sequential(*layers)

        # Policy head: mean and log_std for the correction
        self._mean_head = nn.Linear(in_dim, action_dim)
        self._log_std = nn.Parameter(
            torch.zeros(action_dim) + self._LOG_STD_MIN
        )

        # Value head
        self._value_head = nn.Linear(in_dim, 1)

    @property
    def eta_max(self) -> float:
        """Maximum correction fraction."""
        return self._eta_max

    def forward(
        self,
        state: torch.Tensor,
        analytic_action: torch.Tensor,
    ) -> ResidualPolicyOutput:
        """Compute bounded residual correction.

        :param state: Current states, shape ``(batch, state_dim)``.
        :param analytic_action: Analytic policy actions, shape
            ``(batch, action_dim)``. Used to bound the correction.
        :returns: ResidualPolicyOutput.
        """
        assert state.shape[-1] == self._state_dim, (
            f"Expected state_dim={self._state_dim}, got {state.shape[-1]}"
        )

        features = self._trunk(state)  # (batch, hidden_dim)

        # Policy: sample correction direction
        raw_mean = self._mean_head(features)  # (batch, action_dim)
        log_std = self._log_std.clamp(self._LOG_STD_MIN, self._LOG_STD_MAX)
        std = log_std.exp().expand_as(raw_mean)

        dist = torch.distributions.Normal(raw_mean, std)
        raw_sample = dist.rsample()  # (batch, action_dim)

        # Bound: delta_a = eta_max * tanh(raw) * |a_analytic|
        bounded = (
            self._eta_max
            * torch.tanh(raw_sample)
            * analytic_action.detach().abs()
        )
        # (batch, action_dim)

        log_prob = dist.log_prob(raw_sample).sum(dim=-1)  # (batch,)
        entropy = dist.entropy().sum(dim=-1)  # (batch,)

        # Value
        value = self._value_head(features).squeeze(-1)  # (batch,)

        return ResidualPolicyOutput(
            delta_action=bounded,
            value=value,
            log_prob=log_prob,
            entropy=entropy,
            raw_output=raw_sample,
        )

    def evaluate_actions(
        self,
        state: torch.Tensor,
        analytic_action: torch.Tensor,
        taken_raw: torch.Tensor,
    ) -> ResidualPolicyOutput:
        """Evaluate log-prob and value for taken (raw) actions.

        :param state: States, shape ``(batch, state_dim)``.
        :param analytic_action: Analytic actions, shape ``(batch, action_dim)``.
        :param taken_raw: The raw (unbounded) actions that were taken,
            shape ``(batch, action_dim)``.
        :returns: ResidualPolicyOutput with log_prob for the taken actions.
        """
        features = self._trunk(state)
        raw_mean = self._mean_head(features)
        log_std = self._log_std.clamp(self._LOG_STD_MIN, self._LOG_STD_MAX)
        std = log_std.exp().expand_as(raw_mean)

        dist = torch.distributions.Normal(raw_mean, std)
        log_prob = dist.log_prob(taken_raw).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        bounded = self._eta_max * torch.tanh(taken_raw) * analytic_action.detach().abs()
        value = self._value_head(features).squeeze(-1)

        return ResidualPolicyOutput(
            delta_action=bounded,
            value=value,
            log_prob=log_prob,
            entropy=entropy,
            raw_output=taken_raw,
        )
