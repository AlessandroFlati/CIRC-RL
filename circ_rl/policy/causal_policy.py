"""Causal policy combining feature masking, optional IB encoder, and policy head.

Implements the full policy architecture from ``CIRC-RL_Framework.md``:
feature mask (Phase 2) -> optional IB encoder (Section 3.4) -> policy head.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from circ_rl.policy.encoder import InformationBottleneckEncoder


@dataclass
class PolicyOutput:
    """Output from a policy forward pass.

    :param action: Sampled action of shape ``(batch,)``.
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
    :param action_dim: Number of discrete actions.
    :param feature_mask: Boolean array indicating causal features.
    :param hidden_dims: Sizes of hidden layers in the shared trunk.
    :param activation: Activation function class.
    :param use_info_bottleneck: Whether to use the IB encoder.
    :param latent_dim: Latent dimension for the IB encoder (if used).
    """

    def __init__(
        self,
        full_state_dim: int,
        action_dim: int,
        feature_mask: np.ndarray,
        hidden_dims: tuple[int, ...] = (256, 256),
        activation: type[nn.Module] = nn.ReLU,
        use_info_bottleneck: bool = False,
        latent_dim: int = 32,
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

        # Policy head (logits for Categorical)
        self._policy_head = nn.Linear(in_dim, action_dim)

        # Value head
        self._value_head = nn.Linear(in_dim, 1)

    def forward(self, state: torch.Tensor) -> PolicyOutput:
        """Sample an action and compute value, log_prob, entropy, and KL.

        :param state: Full state tensor of shape ``(batch, full_state_dim)``.
        :returns: PolicyOutput with all training signals.
        """
        causal_features = state[:, self._feature_mask]  # (batch, causal_dim)

        kl = torch.zeros(state.shape[0], device=state.device)

        if self._use_ib and self._encoder is not None:
            z, mu, logvar = self._encoder(causal_features)
            kl = InformationBottleneckEncoder.kl_divergence(mu, logvar)
            trunk_input = z
        else:
            trunk_input = causal_features

        features = self._trunk(trunk_input)  # (batch, hidden_dim)

        logits = self._policy_head(features)  # (batch, action_dim)
        value = self._value_head(features).squeeze(-1)  # (batch,)

        dist = Categorical(logits=logits)
        action = dist.sample()  # (batch,)
        log_prob = dist.log_prob(action)  # (batch,)
        entropy = dist.entropy()  # (batch,)

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

        Used during training when actions are already determined by rollout.

        :param state: Full state tensor of shape ``(batch, full_state_dim)``.
        :param action: Action tensor of shape ``(batch,)``.
        :returns: PolicyOutput with log_prob and value for the given actions.
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
        logits = self._policy_head(features)
        value = self._value_head(features).squeeze(-1)

        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return PolicyOutput(
            action=action,
            log_prob=log_prob,
            entropy=entropy,
            value=value,
            kl_divergence=kl,
        )

    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> int:
        """Get a single action for inference.

        :param state: Single state tensor of shape ``(state_dim,)`` or ``(1, state_dim)``.
        :param deterministic: If True, take the argmax action.
        :returns: Integer action.
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
            logits = self._policy_head(features)

            if deterministic:
                action = logits.argmax(dim=-1)
            else:
                dist = Categorical(logits=logits)
                action = dist.sample()

        return int(action.item())
