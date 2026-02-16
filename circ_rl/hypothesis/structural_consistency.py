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
    """

    passed: bool
    f_statistic: float
    p_value: float
    per_env_mse: dict[int, float]
    pooled_mse: float


class StructuralConsistencyTest:
    r"""Test cross-environment structural consistency of a hypothesis.

    For a hypothesis :math:`h_i`, estimates its coefficients independently
    in each environment and tests whether they are consistent with the
    structural prediction via an F-test comparing pooled vs. per-environment
    residuals.

    See ``CIRC-RL_Framework.md`` Section 3.5.1.

    :param p_threshold: Falsification threshold. If the F-test p-value is
        below this threshold, the hypothesis is falsified. Default 0.01.
    """

    def __init__(self, p_threshold: float = 0.01) -> None:
        if p_threshold <= 0 or p_threshold >= 1:
            raise ValueError(
                f"p_threshold must be in (0, 1), got {p_threshold}"
            )
        self._p_threshold = p_threshold

    def test(
        self,
        expression: SymbolicExpression,
        dataset: ExploratoryDataset,
        target_dim_idx: int,
        variable_names: list[str],
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
        else:
            targets = dataset.rewards  # (N,)

        # Build feature matrix
        x = self._build_features(dataset, variable_names)  # (N, n_vars)

        # Pooled model: evaluate hypothesis on all data
        try:
            func = expression.to_callable(variable_names)
            y_pred_pooled = func(x)  # (N,)
        except (ValueError, Exception) as exc:
            logger.warning(
                "Failed to evaluate expression: {}", exc,
            )
            return StructuralConsistencyResult(
                passed=False, f_statistic=float("inf"), p_value=0.0,
                per_env_mse={}, pooled_mse=float("inf"),
            )

        ss_pooled = float(np.sum((targets - y_pred_pooled) ** 2))
        n_total = len(targets)

        # Per-environment: independent linear regression on each env
        ss_per_env = 0.0
        per_env_mse: dict[int, float] = {}
        total_per_env_params = 0

        for env_id in unique_envs:
            mask = dataset.env_ids == env_id
            x_env = x[mask]  # (n_e, n_vars)
            y_env = targets[mask]  # (n_e,)
            n_e = int(mask.sum())

            # Fit independent linear model per env
            design = np.column_stack([np.ones(n_e), x_env])  # (n_e, 1 + n_vars)
            n_params = design.shape[1]

            if n_e <= n_params:
                per_env_mse[env_id] = 0.0
                continue

            beta = np.linalg.lstsq(design, y_env, rcond=None)[0]
            y_pred_env = design @ beta
            ss_env = float(np.sum((y_env - y_pred_env) ** 2))
            ss_per_env += ss_env
            per_env_mse[env_id] = ss_env / n_e
            total_per_env_params += n_params

        pooled_mse = ss_pooled / n_total if n_total > 0 else 0.0

        # F-test: (SS_pooled - SS_per_env) / extra_params
        #         vs. SS_per_env / (N - total_per_env_params)
        n_pooled_params = len(variable_names) + 1  # intercept + features
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

        passed = p_value >= self._p_threshold

        logger.debug(
            "Structural consistency: F={:.4f}, p={:.6f}, passed={}",
            f_stat, p_value, passed,
        )

        return StructuralConsistencyResult(
            passed=passed,
            f_statistic=f_stat,
            p_value=p_value,
            per_env_mse=per_env_mse,
            pooled_mse=pooled_mse,
        )

    @staticmethod
    def _build_features(
        dataset: ExploratoryDataset,
        variable_names: list[str],
    ) -> np.ndarray:
        """Build feature matrix matching variable_names from dataset.

        Assembles columns from states, actions, and env_params based
        on variable name conventions (s0..sN, action/action_0..N,
        raw param names).

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
            # State features: s0, s1, ...
            if name.startswith("s") and name[1:].isdigit():
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
                # This is a fallback -- proper matching would need param_names
                found = False
                if dataset.env_params is not None:
                    # Assume env_params columns are in alphabetical order
                    # of param names -- match by position
                    n_ep = dataset.env_params.shape[1]
                    for _ep_idx in range(n_ep):
                        # Heuristic: variable_names may contain raw param names
                        # after state and action names
                        state_action_count = state_dim + action_dim
                        var_idx = variable_names.index(name)
                        if var_idx >= state_action_count:
                            ep_col = var_idx - state_action_count
                            if ep_col < n_ep:
                                columns.append(dataset.env_params[:, ep_col])
                                found = True
                                break
                if not found:
                    columns.append(np.zeros(n))
            else:
                columns.append(np.zeros(n))

        return np.column_stack(columns)  # (N, len(variable_names))
