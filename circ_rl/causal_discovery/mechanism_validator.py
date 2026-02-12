"""MechanismValidator: test causal mechanism invariance across environments.

Implements Definition 2.2 from ``CIRC-RL_Framework.md`` Section 2.2:
a mechanism is causally invariant if P_e(Y | do(X)) = P_e'(Y | do(X))
for all environment pairs.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from loguru import logger
from scipy import stats

from circ_rl.causal_discovery.causal_graph import CausalGraph
from circ_rl.environments.data_collector import ExploratoryDataset


@dataclass(frozen=True)
class MechanismValidationResult:
    """Result of mechanism invariance validation.

    :param is_invariant: True if all mechanisms pass the invariance test.
    :param p_values: Mapping of (parent, child) edge to the minimum
        p-value across all environment pairs.
    :param unstable_mechanisms: List of (parent, child) edges that failed
        the invariance test.
    """

    is_invariant: bool
    p_values: dict[tuple[str, str], float]
    unstable_mechanisms: list[tuple[str, str]]


class MechanismValidator:
    r"""Validate that causal mechanisms are invariant across environments.

    For each edge :math:`X \to Y` in the causal graph, tests whether the
    conditional distribution :math:`P_e(Y \mid X, \text{Pa}(Y))` is
    the same across all environments. This is a proxy for testing the
    interventional invariance :math:`P_e(Y \mid do(X)) = P_{e'}(Y \mid do(X))`.

    Uses a two-sample test (Kolmogorov-Smirnov) on the regression
    residuals from each environment pair.

    :param alpha: Significance level for the invariance tests.

    See ``CIRC-RL_Framework.md`` Section 2.2, Definition 2.2 and
    Section 3.6, Phase 1, step 4.
    """

    def __init__(self, alpha: float = 0.05) -> None:
        self._alpha = alpha

    def validate_invariance(
        self,
        dataset: ExploratoryDataset,
        graph: CausalGraph,
        node_names: list[str],
        target_node: str = "reward",
    ) -> MechanismValidationResult:
        """Test mechanism invariance for all edges pointing to the target.

        :param dataset: Exploratory data with env_ids.
        :param graph: The inferred causal graph.
        :param node_names: Variable names matching dataset columns.
        :param target_node: Node whose incoming mechanisms to validate.
        :returns: MechanismValidationResult.
        """
        # Use augmented data when env params are present so column indices
        # match node_names that include ep_ nodes.
        if dataset.env_params is not None:
            data = dataset.to_flat_array_with_env_params()  # (N, n_vars + n_ep)
        else:
            data = dataset.to_flat_array()  # (N, n_vars)
        name_to_idx = {name: i for i, name in enumerate(node_names)}
        env_ids = dataset.env_ids
        unique_envs = sorted(set(env_ids.tolist()))

        if len(unique_envs) < 2:
            logger.warning(
                "Only {} environment(s) in dataset; "
                "invariance testing requires at least 2",
                len(unique_envs),
            )
            return MechanismValidationResult(
                is_invariant=True,
                p_values={},
                unstable_mechanisms=[],
            )

        # Exclude env-param parents: they are exogenous (constant within each
        # env) so testing mechanism invariance for them is not meaningful.
        all_parents = graph.parents(target_node)
        parents = frozenset(
            p for p in all_parents if not graph.is_env_param_node(p)
        )
        if not parents:
            logger.info("Target '{}' has no parents; nothing to validate", target_node)
            return MechanismValidationResult(
                is_invariant=True,
                p_values={},
                unstable_mechanisms=[],
            )

        p_values: dict[tuple[str, str], float] = {}
        unstable: list[tuple[str, str]] = []

        target_idx = name_to_idx[target_node]

        for parent in sorted(parents):
            parent_idx = name_to_idx[parent]

            # Get all parent indices for regression (include ep parents
            # as covariates but only iterate over non-ep parents above)
            all_parent_idxs = [name_to_idx[p] for p in sorted(all_parents)]

            min_p = 1.0

            for env_a, env_b in combinations(unique_envs, 2):
                mask_a = env_ids == env_a
                mask_b = env_ids == env_b

                p_val = self._test_mechanism_stability(
                    data[mask_a], data[mask_b], target_idx, all_parent_idxs
                )
                min_p = min(min_p, p_val)

            p_values[(parent, target_node)] = min_p

            if min_p < self._alpha:
                unstable.append((parent, target_node))
                logger.warning(
                    "Unstable mechanism: {} -> {} (min p-value={:.4f})",
                    parent,
                    target_node,
                    min_p,
                )

        is_invariant = len(unstable) == 0
        if is_invariant:
            logger.info(
                "All mechanisms to '{}' are invariant (alpha={})",
                target_node,
                self._alpha,
            )

        return MechanismValidationResult(
            is_invariant=is_invariant,
            p_values=p_values,
            unstable_mechanisms=unstable,
        )

    @staticmethod
    def _test_mechanism_stability(
        data_a: np.ndarray,
        data_b: np.ndarray,
        target_idx: int,
        parent_idxs: list[int],
    ) -> float:
        r"""Chow test for mechanism stability across two environments.

        Tests :math:`H_0`: the regression coefficients of target on parents
        are the same in both environments. Uses the F-statistic:

        .. math::

            F = \frac{(\text{SSR}_{\text{pooled}} - \text{SSR}_1 - \text{SSR}_2) / k}
                     {(\text{SSR}_1 + \text{SSR}_2) / (n_1 + n_2 - 2k)}

        where :math:`k` is the number of regression parameters (including
        intercept) and :math:`\text{SSR}` is the sum of squared residuals.

        :param data_a: Data from environment A, shape ``(n_a, n_vars)``.
        :param data_b: Data from environment B, shape ``(n_b, n_vars)``.
        :param target_idx: Column index of the target variable.
        :param parent_idxs: Column indices of parent variables.
        :returns: p-value of the Chow F-test.
        """
        n_a = data_a.shape[0]
        n_b = data_b.shape[0]
        k = len(parent_idxs) + 1  # parents + intercept

        pooled = np.concatenate([data_a, data_b], axis=0)

        res_pooled = _compute_residuals(pooled, target_idx, parent_idxs)
        res_a = _compute_residuals(data_a, target_idx, parent_idxs)
        res_b = _compute_residuals(data_b, target_idx, parent_idxs)

        ssr_pooled = float(np.sum(res_pooled**2))
        ssr_a = float(np.sum(res_a**2))
        ssr_b = float(np.sum(res_b**2))

        dof_denom = n_a + n_b - 2 * k
        if dof_denom <= 0:
            return 1.0

        numerator = (ssr_pooled - ssr_a - ssr_b) / k
        denominator = (ssr_a + ssr_b) / dof_denom

        if denominator < 1e-15:
            return 1.0

        f_stat = numerator / denominator
        p_value = float(1.0 - stats.f.cdf(f_stat, k, dof_denom))
        return p_value


def _compute_residuals(
    data: np.ndarray,
    target_idx: int,
    parent_idxs: list[int],
) -> np.ndarray:
    """Compute regression residuals of target on parents.

    :param data: Data matrix of shape ``(n_samples, n_vars)``.
    :param target_idx: Column index of target.
    :param parent_idxs: Column indices of parents.
    :returns: Residuals of shape ``(n_samples,)``.
    """
    target = data[:, target_idx]  # (n_samples,)

    if len(parent_idxs) == 0:
        return target - np.mean(target)

    parents = data[:, parent_idxs]  # (n_samples, n_parents)
    parents_aug = np.column_stack(
        [np.ones(parents.shape[0]), parents]
    )  # (n_samples, n_parents+1)

    coeffs, _, _, _ = np.linalg.lstsq(parents_aug, target, rcond=None)
    return target - parents_aug @ coeffs
