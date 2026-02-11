"""Parametric complexity: C_param = ||theta||.

Implements the parametric complexity penalty from ``CIRC-RL_Framework.md``
Section 2.3. Penalizes the total number of effective parameters
(approximated by L2 norm of weights).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ParametricComplexity(nn.Module):
    """L2 norm of model parameters as a complexity measure.

    .. math::

        C_{\\text{param}}(\\pi) = \\sum_i \\|\\theta_i\\|_2^2

    :param weight: Scaling factor for the penalty.
    """

    def __init__(self, weight: float = 1e-4) -> None:
        super().__init__()
        self._weight = weight

    def compute(self, model: nn.Module) -> torch.Tensor:
        """Compute the parametric complexity of a model.

        :param model: The neural network whose parameters to measure.
        :returns: Scalar complexity penalty.
        """
        total = torch.tensor(0.0, device=next(model.parameters()).device)
        for param in model.parameters():
            total = total + param.pow(2).sum()
        return self._weight * total

    def forward(self, model: nn.Module) -> torch.Tensor:
        """Compute parametric complexity (alias for ``compute``)."""
        return self.compute(model)
