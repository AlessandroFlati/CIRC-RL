"""Composite regularizer aggregating all complexity penalties.

Combines parametric, functional, path, and information bottleneck
complexity terms with configurable weights.

See ``CIRC-RL_Framework.md`` Section 3.4:

    V_reg(s) = V(s) - beta_1 * C_param - beta_2 * C_func - beta_3 * C_path
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from circ_rl.regularization.reg_functional import FunctionalComplexity
from circ_rl.regularization.reg_info_bottleneck import InformationBottleneckLoss
from circ_rl.regularization.reg_parametric import ParametricComplexity
from circ_rl.regularization.reg_path import PathComplexity


@dataclass
class RegularizationBreakdown:
    """Breakdown of individual regularization terms for logging.

    :param parametric: Parametric complexity (L2 norm).
    :param functional: Functional complexity (negative entropy).
    :param path: Path complexity (action smoothness).
    :param info_bottleneck: Information bottleneck loss (KL + reconstruction).
    :param total: Weighted sum of all terms.
    """

    parametric: float
    functional: float
    path: float
    info_bottleneck: float
    total: float


class CompositeRegularizer(nn.Module):
    """Weighted sum of all regularization components.

    :param parametric_weight: Weight for parametric complexity.
    :param functional_weight: Weight for functional complexity.
    :param path_weight: Weight for path complexity.
    :param ib_beta: Beta for information bottleneck.
    """

    def __init__(
        self,
        parametric_weight: float = 1e-4,
        functional_weight: float = 0.01,
        path_weight: float = 0.001,
        ib_beta: float = 0.01,
    ) -> None:
        super().__init__()

        self._parametric = ParametricComplexity(weight=parametric_weight)
        self._functional = FunctionalComplexity(weight=functional_weight)
        self._path = PathComplexity(weight=path_weight)
        self._ib = InformationBottleneckLoss(beta=ib_beta)

    def compute(
        self,
        model: nn.Module,
        entropy: torch.Tensor,
        actions: torch.Tensor | None = None,
        kl_divergence: torch.Tensor | None = None,
        log_prob: torch.Tensor | None = None,
        discrete_actions: bool = True,
    ) -> tuple[torch.Tensor, RegularizationBreakdown]:
        """Compute the total regularization penalty.

        :param model: The policy network (for parametric complexity).
        :param entropy: Per-sample entropy of shape ``(batch,)``.
        :param actions: Action sequence of shape ``(batch, T)`` for path.
        :param kl_divergence: KL divergence for IB, shape ``(batch,)``.
        :param log_prob: Log-probability for IB, shape ``(batch,)``.
        :param discrete_actions: Whether actions are discrete.
        :returns: Tuple of (total_penalty, breakdown).
        """
        param_loss = self._parametric(model)
        func_loss = self._functional(entropy)

        if actions is not None and actions.dim() >= 2 and actions.shape[1] >= 2:
            path_loss = self._path(actions, discrete=discrete_actions)
        else:
            path_loss = torch.tensor(0.0, device=entropy.device)

        if kl_divergence is not None and log_prob is not None:
            ib_loss = self._ib(kl_divergence, log_prob)
        else:
            ib_loss = torch.tensor(0.0, device=entropy.device)

        total = param_loss + func_loss + path_loss + ib_loss

        breakdown = RegularizationBreakdown(
            parametric=float(param_loss.item()),
            functional=float(func_loss.item()),
            path=float(path_loss.item()),
            info_bottleneck=float(ib_loss.item()),
            total=float(total.item()),
        )

        return total, breakdown

    def forward(
        self,
        model: nn.Module,
        entropy: torch.Tensor,
        actions: torch.Tensor | None = None,
        kl_divergence: torch.Tensor | None = None,
        log_prob: torch.Tensor | None = None,
        discrete_actions: bool = True,
    ) -> tuple[torch.Tensor, RegularizationBreakdown]:
        """Compute total regularization (alias for ``compute``)."""
        return self.compute(
            model, entropy, actions, kl_divergence, log_prob, discrete_actions
        )
