"""Invariant Risk Minimization (IRM) penalty for RL.

Implements the IRM gradient penalty from ``CIRC-RL_Framework.md`` Section 3.3:

    L_IRM(pi) = sum_e R^e(pi) + lambda * sum_e ||grad_{w|w=1} R^e(w * pi)||^2

The penalty is zero iff the policy is simultaneously optimal (up to scaling)
in all environments, forcing invariant representations.

Reference: Arjovsky et al. (2019), "Invariant Risk Minimization".
"""

from __future__ import annotations

import torch
import torch.nn as nn


class IRMPenalty(nn.Module):
    """IRM gradient penalty applied per-environment.

    Computes the squared gradient of the environment-specific loss with respect
    to a dummy scalar multiplier ``w = 1``. When the penalty is zero, the
    learned representation is invariant across environments.

    :param lambda_irm: Weight of the IRM penalty term.
    """

    def __init__(self, lambda_irm: float = 1.0) -> None:
        super().__init__()
        self._lambda_irm = lambda_irm

    @property
    def lambda_irm(self) -> float:
        """Current IRM penalty weight."""
        return self._lambda_irm

    @lambda_irm.setter
    def lambda_irm(self, value: float) -> None:
        self._lambda_irm = value

    def compute_penalty(self, env_losses: list[torch.Tensor]) -> torch.Tensor:
        r"""Compute the IRM gradient penalty.

        For each environment loss :math:`L_e`, compute:

        .. math::

            \|\nabla_{w|_{w=1}} w \cdot L_e\|^2

        which simplifies to :math:`L_e^2` when :math:`L_e` is a scalar.
        For the full version with representation gradients, the penalty
        measures whether the optimal predictor on top of the representation
        is the same across environments.

        :param env_losses: List of per-environment loss tensors (each scalar).
        :returns: Scalar IRM penalty.
        """
        if len(env_losses) == 0:
            return torch.tensor(0.0)

        penalty = torch.tensor(0.0, device=env_losses[0].device)
        for loss_e in env_losses:
            # Create dummy scalar w = 1.0 requiring grad
            w = torch.tensor(1.0, device=loss_e.device, requires_grad=True)
            scaled_loss = w * loss_e

            # Compute gradient of scaled loss w.r.t. w
            grad = torch.autograd.grad(
                scaled_loss,
                w,
                create_graph=True,
                retain_graph=True,
            )[0]

            penalty = penalty + grad.pow(2)

        return self._lambda_irm * penalty

    def forward(self, env_losses: list[torch.Tensor]) -> torch.Tensor:
        """Compute the IRM penalty (alias for ``compute_penalty``).

        :param env_losses: Per-environment losses.
        :returns: Scalar IRM penalty.
        """
        return self.compute_penalty(env_losses)
