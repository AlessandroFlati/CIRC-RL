"""Structural consistency test for hypothesis falsification.

Tests whether a hypothesis's predicted parametric relationship between
environment parameters and dynamics coefficients holds independently
in each environment.

See ``CIRC-RL_Framework.md`` Section 3.5.1 (Cross-Environment Structural
Consistency).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger
from scipy import stats

if TYPE_CHECKING:
    from circ_rl.environments.data_collector import ExploratoryDataset
    from circ_rl.hypothesis.expression import SymbolicExpression


@dataclass(frozen=True)
class StructuralConsistencyResult:
    """Result of structural consistency test.

    :param passed: Whether the hypothesis passed the test.
    :param f_statistic: The F-statistic from the pooled vs. per-env test.
    :param p_value: The p-value of the F-test.
    :param per_env_mse: Per-environment mean squared error.
    :param pooled_mse: MSE from the pooled (hypothesis-constrained) model.
    :param relative_improvement: Relative MSE improvement from per-env
        calibration. Values near 0 indicate the functional form is correct.
    """

    passed: bool
    f_statistic: float
    p_value: float
    per_env_mse: dict[int, float]
    pooled_mse: float
    relative_improvement: float = 0.0


class StructuralConsistencyTest:
    r"""Test cross-environment structural consistency of a hypothesis.

    For a hypothesis :math:`h_i`, estimates its coefficients independently
    in each environment and tests whether they are consistent with the
    structural prediction via an F-test comparing pooled vs. per-environment
    residuals.

    See ``CIRC-RL_Framework.md`` Section 3.5.1.

    :param p_threshold: Falsification threshold. If the F-test p-value is
        below this threshold, the hypothesis is falsified. Default 0.01.
    :param min_relative_improvement: Practical significance threshold.
        If per-env calibration reduces MSE by less than this fraction
        relative to pooled MSE, the hypothesis passes regardless of
        p-value. This prevents falsifying expressions with the correct
        functional form but slightly imprecise numerical coefficients
        when sample sizes are large. Default 0.01 (1%).
    """

    def __init__(
        self,
        p_threshold: float = 0.01,
        min_relative_improvement: float = 0.01,
    ) -> None:
        if p_threshold <= 0 or p_threshold >= 1:
            raise ValueError(
                f"p_threshold must be in (0, 1), got {p_threshold}"
            )
        self._p_threshold = p_threshold
        self._min_relative_improvement = min_relative_improvement

    def test(
        self,
        expression: SymbolicExpression,
        dataset: ExploratoryDataset,
        target_dim_idx: int,
        variable_names: list[str],
        derived_columns: dict[str, np.ndarray] | None = None,
        wrap_angular: bool = False,
    ) -> StructuralConsistencyResult:
        r"""Test structural consistency of a hypothesis across environments.

        Compares pooled-model residuals (hypothesis applied to all data)
        vs. per-environment residuals (independent fit per env) using a
        Chow-style F-test.

        **Falsification criterion:** If the pooled model fits significantly
        worse than per-environment models (p < threshold), the hypothesis
        imposes a structural relationship that does not hold.

        :param expression: The symbolic expression to test.
        :param dataset: Multi-environment data.
        :param target_dim_idx: Index of the target state dimension
            (for dynamics hypotheses) or -1 for reward.
        :param variable_names: Variable names matching the expression's
            free symbols.
        :param wrap_angular: If True, wrap target delta via atan2(sin, cos).
        :returns: StructuralConsistencyResult.
        """
        unique_envs = sorted(set(dataset.env_ids.tolist()))
        n_envs = len(unique_envs)

        if n_envs < 2:
            logger.warning(
                "Structural consistency test requires >= 2 envs, got {}",
                n_envs,
            )
            return StructuralConsistencyResult(
                passed=True, f_statistic=0.0, p_value=1.0,
                per_env_mse={}, pooled_mse=0.0,
            )

        # Build targets
        if target_dim_idx >= 0:
            targets = (
                dataset.next_states[:, target_dim_idx]
                - dataset.states[:, target_dim_idx]
            )  # (N,)
            if wrap_angular:
                targets = np.arctan2(np.sin(targets), np.cos(targets))
        else:
            targets = dataset.rewards  # (N,)

        # Build feature matrix
        x = self._build_features(
            dataset, variable_names, derived_columns,
        )  # (N, n_vars)

        n_total = len(targets)

        # Evaluate hypothesis on all data to get h(x) feature
        try:
            func = expression.to_callable(variable_names)
            h_x = np.asarray(func(x), dtype=np.float64).ravel()  # (N,)
            if h_x.shape[0] == 1:
                h_x = np.broadcast_to(h_x, (n_total,)).copy()
        except (ValueError, Exception) as exc:
            logger.warning(
                "Failed to evaluate expression: {}", exc,
            )
            return StructuralConsistencyResult(
                passed=False, f_statistic=float("inf"), p_value=0.0,
                per_env_mse={}, pooled_mse=float("inf"),
            )

        # Reject expressions producing non-finite values
        if not np.all(np.isfinite(h_x)):
            n_bad = int(np.sum(~np.isfinite(h_x)))
            logger.warning(
                "Expression produces {} non-finite values out of {}",
                n_bad, n_total,
            )
            return StructuralConsistencyResult(
                passed=False, f_statistic=float("inf"), p_value=0.0,
                per_env_mse={}, pooled_mse=float("inf"),
            )

        # Pooled model: y = alpha * h(x) + beta (calibrated across all envs)
        # This allows PySR's numerical coefficients to be rescaled,
        # testing the FUNCTIONAL FORM rather than exact coefficients.
        design_pooled = np.column_stack(
            [np.ones(n_total), h_x],
        )  # (N, 2)
        try:
            beta_pooled = np.linalg.lstsq(
                design_pooled, targets, rcond=None,
            )[0]
        except np.linalg.LinAlgError:
            logger.warning(
                "Pooled lstsq failed (SVD did not converge)",
            )
            return StructuralConsistencyResult(
                passed=False, f_statistic=float("inf"), p_value=0.0,
                per_env_mse={}, pooled_mse=float("inf"),
            )
        y_pred_pooled = design_pooled @ beta_pooled  # (N,)
        ss_pooled = float(np.sum((targets - y_pred_pooled) ** 2))
        n_pooled_params = 2  # alpha + beta

        # Per-environment: y = alpha_e * h(x) + beta_e (per-env calibration)
        ss_per_env = 0.0
        per_env_mse: dict[int, float] = {}
        total_per_env_params = 0

        for env_id in unique_envs:
            mask = dataset.env_ids == env_id
            h_env = h_x[mask]  # (n_e,)
            y_env = targets[mask]  # (n_e,)
            n_e = int(mask.sum())

            # Fit calibrated model per env: y = alpha_e * h(x) + beta_e
            design_env = np.column_stack(
                [np.ones(n_e), h_env],
            )  # (n_e, 2)
            n_params = 2

            if n_e <= n_params:
                per_env_mse[env_id] = 0.0
                continue

            try:
                beta_env = np.linalg.lstsq(
                    design_env, y_env, rcond=None,
                )[0]
            except np.linalg.LinAlgError:
                per_env_mse[env_id] = float("inf")
                continue
            y_pred_env = design_env @ beta_env
            ss_env = float(np.sum((y_env - y_pred_env) ** 2))
            ss_per_env += ss_env
            per_env_mse[env_id] = ss_env / n_e
            total_per_env_params += n_params

        pooled_mse = ss_pooled / n_total if n_total > 0 else 0.0

        # F-test: does the per-env calibration explain significantly
        # more variance than the pooled calibration?
        # H0: one (alpha, beta) is sufficient for all envs.
        # H1: each env needs its own (alpha_e, beta_e).
        extra_params = total_per_env_params - n_pooled_params
        denom_df = n_total - total_per_env_params

        if extra_params <= 0 or denom_df <= 0 or ss_per_env <= 0:
            return StructuralConsistencyResult(
                passed=True, f_statistic=0.0, p_value=1.0,
                per_env_mse=per_env_mse, pooled_mse=pooled_mse,
            )

        f_stat = ((ss_pooled - ss_per_env) / extra_params) / (
            ss_per_env / denom_df
        )
        p_value = 1.0 - float(stats.f.cdf(f_stat, extra_params, denom_df))

        # Practical significance: relative MSE improvement from per-env
        # calibration. If negligible, the functional form is correct even
        # if the F-test is significant (large-sample power issue).
        relative_improvement = (
            (ss_pooled - ss_per_env) / ss_pooled if ss_pooled > 0 else 0.0
        )

        statistically_significant = p_value < self._p_threshold
        practically_significant = (
            relative_improvement >= self._min_relative_improvement
        )

        # Pass if: not statistically significant, OR statistically
        # significant but not practically significant (tiny effect).
        passed = not statistically_significant or not practically_significant

        logger.debug(
            "Structural consistency: F={:.4f}, p={:.6f}, "
            "rel_improvement={:.6f}, passed={}",
            f_stat, p_value, relative_improvement, passed,
        )

        return StructuralConsistencyResult(
            passed=passed,
            f_statistic=f_stat,
            p_value=p_value,
            per_env_mse=per_env_mse,
            pooled_mse=pooled_mse,
            relative_improvement=relative_improvement,
        )

    @staticmethod
    def _build_features(
        dataset: ExploratoryDataset,
        variable_names: list[str],
        derived_columns: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        """Build feature matrix matching variable_names from dataset.

        Assembles columns from states, actions, env_params, and
        derived features based on variable name conventions
        (s0..sN, action/action_0..N, raw param names, derived names).

        :param derived_columns: Pre-computed derived feature arrays,
            keyed by variable name. Each value has shape ``(N,)``.
        :returns: Array of shape ``(N, len(variable_names))``.
        """
        n = dataset.states.shape[0]
        state_dim = dataset.states.shape[1]
        actions_2d = (
            dataset.actions if dataset.actions.ndim == 2
            else dataset.actions[:, np.newaxis]
        )
        action_dim = actions_2d.shape[1]

        columns: list[np.ndarray] = []
        for name in variable_names:
            # Derived features (checked first for priority)
            if derived_columns is not None and name in derived_columns:
                columns.append(derived_columns[name])
            # State features: s0, s1, ...
            elif name.startswith("s") and name[1:].isdigit():
                idx = int(name[1:])
                columns.append(dataset.states[:, idx])
            # Action features
            elif name == "action":
                columns.append(actions_2d[:, 0])
            elif name.startswith("action_") and name[7:].isdigit():
                idx = int(name[7:])
                columns.append(actions_2d[:, idx])
            # Env params (raw names)
            elif dataset.env_params is not None:
                # Try to find as env param column
                # env_params columns follow the order of param_names
                # We don't have param_names here, so use positional
                # This is a fallback -- proper matching would need
                # param_names
                found = False
                if dataset.env_params is not None:
                    n_ep = dataset.env_params.shape[1]
                    for _ep_idx in range(n_ep):
                        state_action_count = state_dim + action_dim
                        var_idx = variable_names.index(name)
                        if var_idx >= state_action_count:
                            ep_col = var_idx - state_action_count
                            if ep_col < n_ep:
                                columns.append(
                                    dataset.env_params[:, ep_col],
                                )
                                found = True
                                break
                if not found:
                    columns.append(np.zeros(n))
            else:
                columns.append(np.zeros(n))

        return np.column_stack(columns)  # (N, len(variable_names))
