"""Symbolic MDL scoring for hypothesis selection.

Scores hypotheses using the Minimum Description Length principle
with symbolic complexity as the complexity measure.

See ``CIRC-RL_Framework.md`` Section 3.5.4 (Selection Among
Surviving Hypotheses).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from circ_rl.environments.data_collector import ExploratoryDataset
    from circ_rl.hypothesis.expression import SymbolicExpression


@dataclass(frozen=True)
class SymbolicMDLScore:
    r"""MDL score for a symbolic hypothesis.

    .. math::

        \text{MDL}(h) = -\log P(D|h) + C_{\text{sym}}(h) \cdot \log(n)

    :param total: Total MDL score (lower is better).
    :param data_fit: Negative log-likelihood term :math:`-\log P(D|h)`.
    :param complexity_penalty: Complexity penalty
        :math:`C_{\text{sym}}(h) \cdot \log(n)`.
    :param symbolic_complexity: Raw symbolic complexity :math:`C_{\text{sym}}`.
    :param n_samples: Number of data samples :math:`n`.
    """

    total: float
    data_fit: float
    complexity_penalty: float
    symbolic_complexity: int
    n_samples: int


class SymbolicMDLScorer:
    r"""Score symbolic hypotheses using MDL.

    .. math::

        \text{MDL}(h) = -\log P(D|h) + C_{\text{sym}}(h) \cdot \log(n)

    The data fit term assumes Gaussian residuals:

    .. math::

        -\log P(D|h) = \frac{n}{2} \log(2\pi\hat{\sigma}^2) + \frac{n}{2}

    where :math:`\hat{\sigma}^2` is the residual variance.

    See ``CIRC-RL_Framework.md`` Section 3.5.4.
    """

    def score(
        self,
        expression: SymbolicExpression,
        dataset: ExploratoryDataset,
        target_dim_idx: int,
        variable_names: list[str],
        derived_columns: dict[str, np.ndarray] | None = None,
        wrap_angular: bool = False,
    ) -> SymbolicMDLScore:
        """Compute the MDL score of a hypothesis on data.

        :param expression: The symbolic expression.
        :param dataset: Multi-environment data.
        :param target_dim_idx: Target dimension index (-1 for reward).
        :param variable_names: Variable names for expression evaluation.
        :param wrap_angular: If True, wrap target delta via atan2(sin, cos).
        :returns: SymbolicMDLScore.
        """
        # Build targets
        if target_dim_idx >= 0:
            targets = (
                dataset.next_states[:, target_dim_idx]
                - dataset.states[:, target_dim_idx]
            )
            if wrap_angular:
                targets = np.arctan2(np.sin(targets), np.cos(targets))
        else:
            targets = dataset.rewards

        n = len(targets)

        # Build features
        from circ_rl.hypothesis.structural_consistency import (
            StructuralConsistencyTest,
        )

        x = StructuralConsistencyTest._build_features(
            dataset, variable_names, derived_columns,
        )

        # Evaluate hypothesis
        try:
            func = expression.to_callable(variable_names)
            y_pred = func(x)
            residuals = targets - y_pred
            residual_var = float(np.var(residuals))
        except Exception as exc:
            logger.warning("Failed to evaluate expression: {}", exc)
            residual_var = float(np.var(targets))

        # Avoid log(0) for perfect fit
        residual_var = max(residual_var, 1e-15)

        # Gaussian log-likelihood: -n/2 * log(2*pi*sigma^2) - n/2
        data_fit = (n / 2.0) * np.log(2.0 * np.pi * residual_var) + n / 2.0

        # Complexity penalty: C_sym * log(n)
        complexity_penalty = expression.complexity * np.log(max(n, 2))

        total = data_fit + complexity_penalty

        logger.debug(
            "SymbolicMDL: total={:.2f} (fit={:.2f}, penalty={:.2f}, "
            "C_sym={}, n={})",
            total, data_fit, complexity_penalty, expression.complexity, n,
        )

        return SymbolicMDLScore(
            total=total,
            data_fit=data_fit,
            complexity_penalty=complexity_penalty,
            symbolic_complexity=expression.complexity,
            n_samples=n,
        )
