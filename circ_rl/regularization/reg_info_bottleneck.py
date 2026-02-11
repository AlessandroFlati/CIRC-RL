"""Information bottleneck loss for state representation compression.

Implements the IB loss from ``CIRC-RL_Framework.md`` Section 3.4:

    L_IB = KL(q_phi(z|s) || p(z)) - beta * E_q[log pi(a|z)]

This forces the latent representation to compress irrelevant state
information while retaining action-relevant features.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class InformationBottleneckLoss(nn.Module):
    """Information bottleneck loss combining KL and reconstruction terms.

    :param beta: Trade-off between compression (KL) and informativeness.
        Higher beta = more compression.
    """

    def __init__(self, beta: float = 0.01) -> None:
        super().__init__()
        if beta < 0:
            raise ValueError(f"beta must be >= 0, got {beta}")
        self._beta = beta

    @property
    def beta(self) -> float:
        """Current IB beta weight."""
        return self._beta

    @beta.setter
    def beta(self, value: float) -> None:
        if value < 0:
            raise ValueError(f"beta must be >= 0, got {value}")
        self._beta = value

    def compute(
        self,
        kl_divergence: torch.Tensor,
        log_prob: torch.Tensor,
    ) -> torch.Tensor:
        r"""Compute the information bottleneck loss.

        .. math::

            \mathcal{L}_{\text{IB}} = \text{KL}(q_\phi(z|s) \| p(z))
                - \beta \cdot \mathbb{E}_{q_\phi}[\log \pi(a|z)]

        :param kl_divergence: KL divergence per sample, shape ``(batch,)``.
        :param log_prob: Log-probability of actions, shape ``(batch,)``.
        :returns: Scalar IB loss.
        """
        kl_term = kl_divergence.mean()
        reconstruction_term = -log_prob.mean()
        return kl_term + self._beta * reconstruction_term

    def forward(
        self,
        kl_divergence: torch.Tensor,
        log_prob: torch.Tensor,
    ) -> torch.Tensor:
        """Compute IB loss (alias for ``compute``)."""
        return self.compute(kl_divergence, log_prob)
