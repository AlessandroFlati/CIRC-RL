"""Minimum Description Length scoring for policy evaluation.

Implements the MDL score from ``CIRC-RL_Framework.md`` Section 2.3, Phase 4:

    MDL(pi) = -log P(D|pi) + C(pi)

where the first term measures data fit and the second measures complexity.
Lower MDL = better trade-off between fit and simplicity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from loguru import logger

if TYPE_CHECKING:
    from circ_rl.policy.causal_policy import CausalPolicy
    from circ_rl.training.trajectory_buffer import Trajectory


@dataclass(frozen=True)
class MDLScore:
    """MDL score for a single policy.

    :param total: Total MDL score (lower = better).
    :param data_fit: Negative log-likelihood of data under the policy.
    :param complexity: Parametric complexity of the policy.
    :param n_parameters: Number of trainable parameters.
    """

    total: float
    data_fit: float
    complexity: float
    n_parameters: int


class MDLScorer:
    r"""Score policies using the Minimum Description Length principle.

    .. math::

        \text{MDL}(\pi) = -\log P(D|\pi) + C(\pi)

    Data fit is estimated as negative mean log-likelihood of observed actions
    under the policy. Complexity is the normalized L2 norm of parameters.

    :param complexity_weight: Weight for the complexity term.
    """

    def __init__(self, complexity_weight: float = 0.01) -> None:
        self._complexity_weight = complexity_weight

    def score(
        self,
        policy: CausalPolicy,
        trajectory: Trajectory,
    ) -> MDLScore:
        """Compute the MDL score of a policy on evaluation data.

        :param policy: The policy to score.
        :param trajectory: Evaluation trajectory.
        :returns: MDLScore with breakdown.
        """
        policy.eval()

        with torch.no_grad():
            output = policy.evaluate_actions(trajectory.states, trajectory.actions)

        # Data fit: negative mean log-likelihood
        data_fit = float(-output.log_prob.mean().item())

        # Complexity: log of parameter count + weighted parameter norm
        # MDL penalizes both the number and magnitude of parameters
        n_params = sum(p.numel() for p in policy.parameters())
        param_norm = sum(
            float(p.pow(2).sum().item()) for p in policy.parameters()
        )
        complexity = self._complexity_weight * (np.log(max(n_params, 1)) + param_norm)

        total = data_fit + complexity

        policy.train()

        logger.debug(
            "MDL score: total={:.4f} (fit={:.4f}, complexity={:.4f}, "
            "n_params={})",
            total,
            data_fit,
            complexity,
            n_params,
        )

        return MDLScore(
            total=total,
            data_fit=data_fit,
            complexity=complexity,
            n_parameters=n_params,
        )

    def rank_policies(
        self,
        policies: list[CausalPolicy],
        trajectory: Trajectory,
    ) -> list[tuple[int, MDLScore]]:
        """Score and rank multiple policies by MDL (lower = better).

        :param policies: List of policies to rank.
        :param trajectory: Evaluation trajectory.
        :returns: List of (policy_index, MDLScore) sorted by total score.
        """
        scored = [
            (i, self.score(policy, trajectory))
            for i, policy in enumerate(policies)
        ]
        scored.sort(key=lambda x: x[1].total)
        return scored
