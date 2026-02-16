"""Causal effect estimation via the back-door adjustment.

Estimates the Average Treatment Effect (ATE) of a cause variable on an
effect variable, adjusting for confounders identified from a causal graph.

Reference: Pearl (2009), *Causality*, Section 3.3.1 -- the back-door criterion.
See also ``CIRC-RL_Framework.md`` Section 3.2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger
from scipy import stats

if TYPE_CHECKING:
    from circ_rl.causal_discovery.causal_graph import CausalGraph


@dataclass(frozen=True)
class CausalEffectResult:
    """Result of causal effect estimation.

    :param cause: Name of the cause variable.
    :param effect: Name of the effect variable.
    :param ate: Estimated Average Treatment Effect.
    :param adjustment_set: Variables conditioned on for back-door adjustment.
    :param coefficients: Regression coefficients (intercept, cause, adj_1, ...).
    """

    cause: str
    effect: str
    ate: float
    adjustment_set: frozenset[str]
    coefficients: np.ndarray


class CausalEffectEstimator:
    r"""Estimate causal effects using the back-door adjustment formula.

    For a cause-effect pair :math:`(X, Y)` in a causal graph, identifies
    a valid adjustment set :math:`Z` satisfying the back-door criterion
    and estimates:

    .. math::

        E[Y \mid do(X = x)] = \sum_z E[Y \mid X = x, Z = z] \cdot P(Z = z)

    In practice, this is implemented via linear regression of Y on (X, Z),
    where the coefficient of X is the ATE.

    See ``CIRC-RL_Framework.md`` Section 3.6, Phase 2, step 2.
    """

    @staticmethod
    def find_adjustment_set(
        graph: CausalGraph,
        cause: str,
        effect: str,
    ) -> frozenset[str]:
        r"""Find a valid back-door adjustment set for estimating the effect of
        ``cause`` on ``effect``.

        The back-door criterion requires a set :math:`Z` such that:
        1. No node in :math:`Z` is a descendant of ``cause``.
        2. :math:`Z` blocks every back-door path from ``cause`` to ``effect``.

        A sufficient choice: the parents of ``cause`` (always satisfies
        back-door when the graph is a DAG and there are no hidden variables).

        :param graph: The causal graph.
        :param cause: The treatment / cause variable.
        :param effect: The outcome / effect variable.
        :returns: A valid adjustment set.
        """
        # Parents of cause always satisfy the back-door criterion in a DAG
        # (they block all non-causal paths without introducing bias).
        parents_of_cause = graph.parents(cause)

        # Remove the effect node itself if it appears (shouldn't in a DAG)
        adjustment = parents_of_cause - {effect}

        logger.debug(
            "Adjustment set for {} -> {}: {}",
            cause,
            effect,
            sorted(adjustment),
        )
        return adjustment

    @staticmethod
    def estimate_ate(
        data: np.ndarray,
        node_names: list[str],
        cause: str,
        effect: str,
        adjustment_set: frozenset[str],
    ) -> CausalEffectResult:
        r"""Estimate the Average Treatment Effect via linear regression.

        Regresses ``effect`` on ``(cause, adjustment_set)`` and extracts
        the coefficient of ``cause`` as the ATE estimate.

        :param data: Data matrix of shape ``(n_samples, n_variables)``.
        :param node_names: Variable names corresponding to data columns.
        :param cause: Name of the cause variable.
        :param effect: Name of the effect variable.
        :param adjustment_set: Variables to adjust for.
        :returns: CausalEffectResult with the ATE estimate.
        :raises ValueError: If variable names are not found in node_names.
        """
        name_to_idx = {name: i for i, name in enumerate(node_names)}

        for var in [cause, effect, *sorted(adjustment_set)]:
            if var not in name_to_idx:
                raise ValueError(
                    f"Variable '{var}' not found in node_names: {node_names}"
                )

        cause_idx = name_to_idx[cause]
        effect_idx = name_to_idx[effect]
        adj_idxs = [name_to_idx[a] for a in sorted(adjustment_set)]

        y = data[:, effect_idx]  # (n_samples,)

        # Build design matrix: [intercept, cause, adj_1, adj_2, ...]
        x_cols = [np.ones(data.shape[0]), data[:, cause_idx]]
        for idx in adj_idxs:
            x_cols.append(data[:, idx])
        x_matrix = np.column_stack(x_cols)  # (n_samples, 1 + 1 + len(adj))

        coeffs, _, _, _ = np.linalg.lstsq(x_matrix, y, rcond=None)

        ate = float(coeffs[1])  # Coefficient of the cause variable

        logger.debug(
            "ATE of {} -> {}: {:.4f} (adjustment set: {})",
            cause,
            effect,
            ate,
            sorted(adjustment_set),
        )

        return CausalEffectResult(
            cause=cause,
            effect=effect,
            ate=ate,
            adjustment_set=adjustment_set,
            coefficients=coeffs,
        )

    @staticmethod
    def test_mechanism_invariance(
        data_a: np.ndarray,
        data_b: np.ndarray,
        target_idx: int,
        predictor_idxs: list[int],
        poly_degree: int = 2,
    ) -> float:
        r"""Chow test with polynomial basis for mechanism invariance.

        Tests :math:`H_0`: the regression surface
        :math:`\text{target} \sim \text{poly}(\text{predictors})` is the
        same in both environments. Polynomial expansion captures non-linear
        mechanisms (e.g., quadratic and interaction effects) that a linear
        Chow test would miss.

        Uses the F-statistic:

        .. math::

            F = \frac{(\text{SSR}_{\text{pooled}} - \text{SSR}_a - \text{SSR}_b) / k}
                     {(\text{SSR}_a + \text{SSR}_b) / (n_a + n_b - 2k)}

        where :math:`k` is the number of polynomial regression parameters
        (including intercept) and SSR is the sum of squared residuals.

        :param data_a: Data from environment A, shape ``(n_a, n_vars)``.
        :param data_b: Data from environment B, shape ``(n_b, n_vars)``.
        :param target_idx: Column index of the target variable (e.g., reward).
        :param predictor_idxs: Column indices of predictor variables
            (treatment + adjustment set).
        :param poly_degree: Degree of polynomial expansion (default 2).
        :returns: p-value of the Chow F-test (high = mechanism invariant).
        """
        from sklearn.preprocessing import PolynomialFeatures

        n_a = data_a.shape[0]
        n_b = data_b.shape[0]
        min_n = min(n_a, n_b)

        if len(predictor_idxs) == 0:
            return 1.0

        # Safeguard: reduce poly degree if too many features for sample size
        degree = poly_degree
        pf_probe = PolynomialFeatures(degree=degree, include_bias=False)
        n_raw = len(predictor_idxs)
        probe_input = np.zeros((1, n_raw))
        n_poly = pf_probe.fit_transform(probe_input).shape[1]
        k_with_intercept = n_poly + 1  # +1 for intercept

        while degree > 1 and k_with_intercept > min_n / 5:
            degree -= 1
            pf_probe = PolynomialFeatures(degree=degree, include_bias=False)
            n_poly = pf_probe.fit_transform(probe_input).shape[1]
            k_with_intercept = n_poly + 1

        pf = PolynomialFeatures(degree=degree, include_bias=False)

        def _poly_residuals(data: np.ndarray) -> np.ndarray:
            """Compute residuals of target regressed on poly(predictors)."""
            n = data.shape[0]
            target = data[:, target_idx]  # (n,)
            predictors = data[:, predictor_idxs]  # (n, n_predictors)
            poly_features = pf.fit_transform(predictors)  # (n, n_poly)
            design = np.column_stack(
                [np.ones(n), poly_features]
            )  # (n, k)
            coeffs, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
            return target - design @ coeffs  # (n,)

        pooled = np.concatenate([data_a, data_b], axis=0)

        res_pooled = _poly_residuals(pooled)
        res_a = _poly_residuals(data_a)
        res_b = _poly_residuals(data_b)

        ssr_pooled = float(np.sum(res_pooled**2))
        ssr_a = float(np.sum(res_a**2))
        ssr_b = float(np.sum(res_b**2))

        k = k_with_intercept
        dof_denom = n_a + n_b - 2 * k
        if dof_denom <= 0:
            return 1.0

        numerator = (ssr_pooled - ssr_a - ssr_b) / k
        denominator = (ssr_a + ssr_b) / dof_denom

        if denominator < 1e-15:
            return 1.0

        f_stat = numerator / denominator
        if f_stat < 0:
            return 1.0

        return float(1.0 - stats.f.cdf(f_stat, k, dof_denom))

    def estimate(
        self,
        data: np.ndarray,
        node_names: list[str],
        graph: CausalGraph,
        cause: str,
        effect: str,
    ) -> CausalEffectResult:
        """Estimate the causal effect of ``cause`` on ``effect``.

        Combines adjustment set finding with ATE estimation.

        :param data: Data matrix of shape ``(n_samples, n_variables)``.
        :param node_names: Variable names corresponding to data columns.
        :param graph: The causal graph.
        :param cause: Name of the cause variable.
        :param effect: Name of the effect variable.
        :returns: CausalEffectResult with the ATE estimate.
        """
        adjustment = self.find_adjustment_set(graph, cause, effect)
        return self.estimate_ate(data, node_names, cause, effect, adjustment)
