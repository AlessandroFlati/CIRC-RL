"""Causal policy combining feature masking, optional IB encoder, and policy head.

Implements the full policy architecture from ``CIRC-RL_Framework.md``:
feature mask (Phase 2) -> optional IB encoder (Section 3.4) -> policy head.

Supports both discrete (Categorical) and continuous (tanh-squashed Normal)
action spaces.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal

from circ_rl.policy.encoder import InformationBottleneckEncoder


@dataclass
class PolicyOutput:
    """Output from a policy forward pass.

    :param action: Sampled action. Shape ``(batch,)`` for discrete,
        ``(batch, action_dim)`` for continuous.
    :param log_prob: Log-probability of the sampled action, shape ``(batch,)``.
    :param entropy: Entropy of the action distribution, shape ``(batch,)``.
    :param value: State value estimate, shape ``(batch,)``.
    :param kl_divergence: KL divergence from IB encoder (0 if no encoder), shape ``(batch,)``.
    """

    action: torch.Tensor
    log_prob: torch.Tensor
    entropy: torch.Tensor
    value: torch.Tensor
    kl_divergence: torch.Tensor


class CausalPolicy(nn.Module):
    """Actor-critic policy with causal feature masking and optional IB encoder.

    Architecture:
        full_state -> causal_mask -> [IB_encoder] -> shared_trunk -> policy_head
                                                                  -> value_head

    :param full_state_dim: Dimensionality of the full state space.
    :param action_dim: Number of discrete actions, or continuous action dimensions.
    :param feature_mask: Boolean array indicating causal features.
    :param hidden_dims: Sizes of hidden layers in the shared trunk.
    :param activation: Activation function class.
    :param use_info_bottleneck: Whether to use the IB encoder.
    :param latent_dim: Latent dimension for the IB encoder (if used).
    :param continuous: If True, use Normal distribution with tanh squashing.
    :param action_low: Lower bounds for continuous actions, shape ``(action_dim,)``.
    :param action_high: Upper bounds for continuous actions, shape ``(action_dim,)``.
    """

    _LOG_STD_MIN = -20.0
    _LOG_STD_MAX = 2.0

    def __init__(
        self,
        full_state_dim: int,
        action_dim: int,
        feature_mask: np.ndarray,
        hidden_dims: tuple[int, ...] = (256, 256),
        activation: type[nn.Module] = nn.ReLU,
        use_info_bottleneck: bool = False,
        latent_dim: int = 32,
        continuous: bool = False,
        action_low: np.ndarray | None = None,
        action_high: np.ndarray | None = None,
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
        causal_dim = int(feature_mask.sum())
        if causal_dim == 0:
            raise ValueError("feature_mask selects zero features")

        self._continuous = continuous

        if continuous:
            if action_low is None or action_high is None:
                raise ValueError(
                    "action_low and action_high are required for continuous policies"
                )
            if action_low.shape != (action_dim,) or action_high.shape != (action_dim,):
                raise ValueError(
                    f"action bounds must have shape ({action_dim},), "
                    f"got low={action_low.shape}, high={action_high.shape}"
                )
            self.register_buffer(
                "_action_low",
                torch.from_numpy(action_low.astype(np.float32)),
            )
            self.register_buffer(
                "_action_high",
                torch.from_numpy(action_high.astype(np.float32)),
            )

        self._use_ib = use_info_bottleneck
        self._encoder: InformationBottleneckEncoder | None = None

        if use_info_bottleneck:
            self._encoder = InformationBottleneckEncoder(
                input_dim=causal_dim,
                latent_dim=latent_dim,
                hidden_dims=hidden_dims[:1] if hidden_dims else (64,),
                activation=activation,
            )
            trunk_input_dim = latent_dim
        else:
            trunk_input_dim = causal_dim

        # Shared trunk
        trunk_layers: list[nn.Module] = []
        in_dim = trunk_input_dim
        for h_dim in hidden_dims:
            trunk_layers.append(nn.Linear(in_dim, h_dim))
            trunk_layers.append(activation())
            in_dim = h_dim

        self._trunk = nn.Sequential(*trunk_layers)

        # Policy head
        if continuous:
            # Output mean and log_std for each action dimension
            self._policy_head = nn.Linear(in_dim, 2 * action_dim)
        else:
            # Output logits for Categorical distribution
            self._policy_head = nn.Linear(in_dim, action_dim)

        # Value head
        self._value_head = nn.Linear(in_dim, 1)

    @property
    def continuous(self) -> bool:
        """Whether this policy uses continuous actions."""
        return self._continuous

    def _compute_trunk_features(
        self, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Shared computation: mask -> encoder -> trunk -> features + value + kl.

        :returns: (features, value, kl) tuple.
        """
        causal_features = state[:, self._feature_mask]
        kl = torch.zeros(state.shape[0], device=state.device)

        if self._use_ib and self._encoder is not None:
            z, mu, logvar = self._encoder(causal_features)
            kl = InformationBottleneckEncoder.kl_divergence(mu, logvar)
            trunk_input = z
        else:
            trunk_input = causal_features

        features = self._trunk(trunk_input)
        value = self._value_head(features).squeeze(-1)
        return features, value, kl

    def _continuous_distribution(
        self, features: torch.Tensor
    ) -> tuple[Normal, torch.Tensor]:
        """Build a Normal distribution from trunk features.

        :param features: Trunk output, shape ``(batch, hidden_dim)``.
        :returns: (Normal distribution, mean) tuple.
        """
        raw = self._policy_head(features)  # (batch, 2 * action_dim)
        mean, log_std = raw.chunk(2, dim=-1)  # each (batch, action_dim)
        log_std = log_std.clamp(self._LOG_STD_MIN, self._LOG_STD_MAX)
        std = log_std.exp()
        return Normal(mean, std), mean

    def _squash_action(self, raw_action: torch.Tensor) -> torch.Tensor:
        """Apply tanh squashing and rescale to [action_low, action_high].

        :param raw_action: Unbounded action, shape ``(batch, action_dim)``.
        :returns: Bounded action, shape ``(batch, action_dim)``.
        """
        squashed = torch.tanh(raw_action)
        return self._action_low + (squashed + 1.0) * 0.5 * (self._action_high - self._action_low)

    def _squash_log_prob(
        self, log_prob: torch.Tensor, raw_action: torch.Tensor
    ) -> torch.Tensor:
        """Correct log-probability for tanh squashing and affine rescaling.

        :param log_prob: Gaussian log-prob summed over action dims, shape ``(batch,)``.
        :param raw_action: Pre-squash action, shape ``(batch, action_dim)``.
        :returns: Corrected log-prob, shape ``(batch,)``.
        """
        # Jacobian correction for tanh: -sum(log(1 - tanh^2(x) + eps))
        correction = torch.log(1.0 - torch.tanh(raw_action).pow(2) + 1e-6)
        return log_prob - correction.sum(dim=-1)

    def _unsquash_action(self, action: torch.Tensor) -> torch.Tensor:
        """Invert squash to recover raw (pre-tanh) action.

        :param action: Bounded action, shape ``(batch, action_dim)``.
        :returns: Unbounded raw action, shape ``(batch, action_dim)``.
        """
        # action = low + (tanh(raw) + 1) * 0.5 * (high - low)
        # => tanh(raw) = 2 * (action - low) / (high - low) - 1
        normalized = 2.0 * (action - self._action_low) / (self._action_high - self._action_low) - 1.0
        normalized = normalized.clamp(-0.999999, 0.999999)
        return torch.atanh(normalized)

    def forward(self, state: torch.Tensor) -> PolicyOutput:
        """Sample an action and compute value, log_prob, entropy, and KL.

        :param state: Full state tensor of shape ``(batch, full_state_dim)``.
        :returns: PolicyOutput with all training signals.
        """
        features, value, kl = self._compute_trunk_features(state)

        if self._continuous:
            dist, _mean = self._continuous_distribution(features)
            raw_action = dist.rsample()  # (batch, action_dim)
            action = self._squash_action(raw_action)
            log_prob = dist.log_prob(raw_action).sum(dim=-1)  # (batch,)
            log_prob = self._squash_log_prob(log_prob, raw_action)
            entropy = dist.entropy().sum(dim=-1)  # (batch,)
        else:
            logits = self._policy_head(features)
            dist_d = Categorical(logits=logits)
            action = dist_d.sample()
            log_prob = dist_d.log_prob(action)
            entropy = dist_d.entropy()

        return PolicyOutput(
            action=action,
            log_prob=log_prob,
            entropy=entropy,
            value=value,
            kl_divergence=kl,
        )

    def evaluate_actions(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> PolicyOutput:
        """Evaluate log_prob and value for given state-action pairs.

        :param state: Full state tensor of shape ``(batch, full_state_dim)``.
        :param action: Action tensor. Shape ``(batch,)`` for discrete,
            ``(batch, action_dim)`` for continuous.
        :returns: PolicyOutput with log_prob and value for the given actions.
        """
        features, value, kl = self._compute_trunk_features(state)

        if self._continuous:
            dist, _mean = self._continuous_distribution(features)
            raw_action = self._unsquash_action(action)
            log_prob = dist.log_prob(raw_action).sum(dim=-1)
            log_prob = self._squash_log_prob(log_prob, raw_action)
            entropy = dist.entropy().sum(dim=-1)
        else:
            logits = self._policy_head(features)
            dist_d = Categorical(logits=logits)
            log_prob = dist_d.log_prob(action)
            entropy = dist_d.entropy()

        return PolicyOutput(
            action=action,
            log_prob=log_prob,
            entropy=entropy,
            value=value,
            kl_divergence=kl,
        )

    def get_action(
        self, state: torch.Tensor, deterministic: bool = False
    ) -> int | np.ndarray:
        """Get a single action for inference.

        :param state: Single state tensor of shape ``(state_dim,)`` or ``(1, state_dim)``.
        :param deterministic: If True, use mean (continuous) or argmax (discrete).
        :returns: Integer action for discrete, numpy array for continuous.
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)

        with torch.no_grad():
            causal_features = state[:, self._feature_mask]

            if self._use_ib and self._encoder is not None:
                trunk_input = self._encoder.encode_deterministic(causal_features)
            else:
                trunk_input = causal_features

            features = self._trunk(trunk_input)

            if self._continuous:
                dist, mean = self._continuous_distribution(features)
                if deterministic:
                    raw_action = mean
                else:
                    raw_action = dist.sample()
                action = self._squash_action(raw_action)
                return action.squeeze(0).cpu().numpy()
            else:
                logits = self._policy_head(features)
                if deterministic:
                    action = logits.argmax(dim=-1)
                else:
                    dist_d = Categorical(logits=logits)
                    action = dist_d.sample()
                return int(action.item())
