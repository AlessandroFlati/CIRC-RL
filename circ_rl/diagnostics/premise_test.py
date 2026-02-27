"""Premise test: validate dynamics hypothesis in test environments.

Tests whether the dynamics observed in held-out test environments
are compatible with the validated hypothesis.

See ``CIRC-RL_Framework.md`` Section 3.9.1.
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
class PremiseTestResult:
    """Result of the premise test.

    :param passed: Whether the dynamics hypothesis is valid in test envs.
    :param per_env_r2: R^2 of the hypothesis predictions per test env.
    :param overall_r2: Aggregate R^2 across all test environments.
    :param per_env_rmse: RMSE per test environment.
    """

    passed: bool
    per_env_r2: dict[int, float]
    overall_r2: float
    per_env_rmse: dict[int, float]


class PremiseTest:
    r"""Test whether the dynamics hypothesis is valid in test environments.

    Evaluates prediction error of :math:`\Delta s` given :math:`(s, a, \theta_e)`
    on held-out environments.

    See ``CIRC-RL_Framework.md`` Section 3.9.1.

    :param r2_threshold: Minimum R^2 for the hypothesis to pass.
        Default 0.5 (the hypothesis should explain at least half
        the variance in test environments).
    """

    def __init__(self, r2_threshold: float = 0.5) -> None:
        self._r2_threshold = r2_threshold

    def test(
        self,
        dynamics_expressions: dict[int, SymbolicExpression],
        dataset: ExploratoryDataset,
        state_feature_names: list[str],
        variable_names: list[str],
        test_env_ids: list[int],
    ) -> PremiseTestResult:
        """Run the premise test on test environments.

        :param dynamics_expressions: Mapping from state dimension index
            to validated SymbolicExpression.
        :param dataset: Data from test environments.
        :param state_feature_names: Names of state features.
        :param variable_names: Variable names for expression evaluation.
        :param test_env_ids: Environment IDs to test on.
        :returns: PremiseTestResult.
        """
        from circ_rl.hypothesis.structural_consistency import (
            StructuralConsistencyTest,
        )

        x = StructuralConsistencyTest._build_features(dataset, variable_names)

        # Compile callables
        callables: dict[int, object] = {}
        for dim_idx, expr in dynamics_expressions.items():
            try:
                callables[dim_idx] = expr.to_callable(variable_names)
            except ValueError as exc:
                logger.warning("Cannot compile dim {}: {}", dim_idx, exc)

        per_env_r2: dict[int, float] = {}
        per_env_rmse: dict[int, float] = {}
        all_ss_res = 0.0
        all_ss_tot = 0.0

        for env_id in test_env_ids:
            mask = dataset.env_ids == env_id
            if not np.any(mask):
                continue

            x_env = x[mask]
            env_ss_res = 0.0
            env_ss_tot = 0.0
            env_n = 0

            for dim_idx, func in callables.items():
                y_true = (
                    dataset.next_states[mask, dim_idx]
                    - dataset.states[mask, dim_idx]
                )
                try:
                    y_pred = func(x_env)  # type: ignore[operator]
                except Exception:
                    y_pred = np.zeros_like(y_true)

                ss_res = float(np.sum((y_true - y_pred) ** 2))
                ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
                env_ss_res += ss_res
                env_ss_tot += ss_tot
                env_n += len(y_true)

            all_ss_res += env_ss_res
            all_ss_tot += env_ss_tot

            env_r2 = 1.0 - env_ss_res / env_ss_tot if env_ss_tot > 0 else 0.0
            env_rmse = float(np.sqrt(env_ss_res / env_n)) if env_n > 0 else 0.0
            per_env_r2[env_id] = env_r2
            per_env_rmse[env_id] = env_rmse

        overall_r2 = 1.0 - all_ss_res / all_ss_tot if all_ss_tot > 0 else 0.0
        passed = overall_r2 >= self._r2_threshold

        logger.info(
            "Premise test: overall_r2={:.4f}, threshold={:.4f}, passed={}",
            overall_r2, self._r2_threshold, passed,
        )

        return PremiseTestResult(
            passed=passed,
            per_env_r2=per_env_r2,
            overall_r2=overall_r2,
            per_env_rmse=per_env_rmse,
        )
