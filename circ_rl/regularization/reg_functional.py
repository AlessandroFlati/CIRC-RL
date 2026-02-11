"""Functional complexity: C_func = E[H(pi(.|s))].

Implements the functional complexity penalty from ``CIRC-RL_Framework.md``
Section 2.3. Penalizes low-entropy policies (overly deterministic policies
are more likely to overfit).

Note: this is the *negative* entropy, so minimizing this term encourages
higher-entropy (simpler) policies.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FunctionalComplexity(nn.Module):
    """Negative policy entropy as a complexity measure.

    .. math::

        C_{\\text{func}}(\\pi) = -\\mathbb{E}_s[H(\\pi(\\cdot|s))]

    Lower entropy = higher complexity = larger penalty.

    :param weight: Scaling factor for the penalty.
    """

    def __init__(self, weight: float = 0.01) -> None:
        super().__init__()
        self._weight = weight

    def compute(self, entropy: torch.Tensor) -> torch.Tensor:
        """Compute functional complexity from entropy values.

        :param entropy: Per-sample entropy of shape ``(batch,)``.
        :returns: Scalar complexity penalty (negative mean entropy).
        """
        return -self._weight * entropy.mean()

    def forward(self, entropy: torch.Tensor) -> torch.Tensor:
        """Compute functional complexity (alias for ``compute``)."""
        return self.compute(entropy)
