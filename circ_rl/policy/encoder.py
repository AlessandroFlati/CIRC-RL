"""Information bottleneck encoder for state compression.

Implements the variational encoder q_phi(z|s) from ``CIRC-RL_Framework.md``
Section 3.4:

    L_IB = KL(q_phi(z|s) || p(z)) - beta * E_q[log pi(a|z)]

where p(z) = N(0, I) is the standard Gaussian prior.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class InformationBottleneckEncoder(nn.Module):
    """Variational encoder that compresses states into a latent representation.

    Uses the reparameterization trick: z = mu + sigma * eps, eps ~ N(0, I).

    :param input_dim: Dimensionality of input features.
    :param latent_dim: Dimensionality of the latent space.
    :param hidden_dims: Sizes of hidden layers in the encoder network.
    :param activation: Activation function class.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: tuple[int, ...] = (256, 256),
        activation: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()

        self._latent_dim = latent_dim

        # Shared trunk
        layers: list[nn.Module] = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(activation())
            in_dim = h_dim

        self._trunk = nn.Sequential(*layers)

        # Mean and log-variance heads
        self._mu_head = nn.Linear(in_dim, latent_dim)
        self._logvar_head = nn.Linear(in_dim, latent_dim)

    @property
    def latent_dim(self) -> int:
        """Dimensionality of the latent space."""
        return self._latent_dim

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode input and sample latent via reparameterization.

        :param x: Input tensor of shape ``(batch, input_dim)``.
        :returns: Tuple of (z, mu, logvar) where:
            - z: sampled latent of shape ``(batch, latent_dim)``
            - mu: mean of shape ``(batch, latent_dim)``
            - logvar: log-variance of shape ``(batch, latent_dim)``
        """
        h = self._trunk(x)  # (batch, hidden_dim)
        mu = self._mu_head(h)  # (batch, latent_dim)
        logvar = self._logvar_head(h)  # (batch, latent_dim)

        z = self._reparameterize(mu, logvar)  # (batch, latent_dim)
        return z, mu, logvar

    def encode_deterministic(self, x: torch.Tensor) -> torch.Tensor:
        """Encode without sampling (use mean).

        :param x: Input tensor of shape ``(batch, input_dim)``.
        :returns: Mean latent of shape ``(batch, latent_dim)``.
        """
        h = self._trunk(x)
        return self._mu_head(h)

    @staticmethod
    def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        r"""Compute KL divergence from posterior to standard Gaussian prior.

        .. math::

            \text{KL}(q_\phi(z|s) \| \mathcal{N}(0, I)) =
                -\frac{1}{2} \sum_{j=1}^{d} (1 + \log\sigma_j^2 - \mu_j^2 - \sigma_j^2)

        :param mu: Mean of shape ``(batch, latent_dim)``.
        :param logvar: Log-variance of shape ``(batch, latent_dim)``.
        :returns: KL divergence per sample of shape ``(batch,)``.
        """
        # (batch, latent_dim) -> sum over latent dims -> (batch,)
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

    @staticmethod
    def _reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = mu + std * eps."""
        std = torch.exp(0.5 * logvar)  # (batch, latent_dim)
        eps = torch.randn_like(std)  # (batch, latent_dim)
        return mu + std * eps
