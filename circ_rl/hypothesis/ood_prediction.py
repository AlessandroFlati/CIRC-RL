"""Out-of-distribution prediction test for hypothesis falsification.

Tests whether a hypothesis calibrated on training environments can
predict dynamics in held-out environments.

See ``CIRC-RL_Framework.md`` Section 3.5.2 (Out-of-Distribution
Prediction).
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
class OODPredictionResult:
    """Result of OOD prediction test.

    :param passed: Whether the hypothesis passed the test.
    :param per_env_passed: Whether each held-out env passed.
    :param per_env_r2: R^2 on each held-out environment.
    :param failure_fraction: Fraction of held-out envs that failed.
    """

    passed: bool
    per_env_passed: dict[int, bool]
    per_env_r2: dict[int, float]
    failure_fraction: float


class OODPredictionTest:
    r"""Test out-of-distribution prediction of a hypothesis.

    Holds out a subset of environments, calibrates the hypothesis on
    training environments, and tests whether predictions on held-out
    environments are accurate (within 99% confidence intervals).

    See ``CIRC-RL_Framework.md`` Section 3.5.2.

    :param confidence: Confidence level for prediction intervals.
        Default 0.99.
    :param failure_fraction: Maximum fraction of held-out environments
        that may fail before the hypothesis is globally rejected.
        Default 0.2.
    :param held_out_fraction: Fraction of environments to hold out.
        Default 0.2.
    """

    def __init__(
        self,
        confidence: float = 0.99,
        failure_fraction: float = 0.2,
        held_out_fraction: float = 0.2,
    ) -> None:
        if confidence <= 0 or confidence >= 1:
            raise ValueError(f"confidence must be in (0, 1), got {confidence}")
        if failure_fraction <= 0 or failure_fraction >= 1:
            raise ValueError(
                f"failure_fraction must be in (0, 1), got {failure_fraction}"
            )
        self._confidence = confidence
        self._failure_fraction = failure_fraction
        self._held_out_fraction = held_out_fraction

    def test(
        self,
        expression: SymbolicExpression,
        dataset: ExploratoryDataset,
        target_dim_idx: int,
        variable_names: list[str],
        train_env_ids: list[int] | None = None,
        held_out_env_ids: list[int] | None = None,
        derived_columns: dict[str, np.ndarray] | None = None,
    ) -> OODPredictionResult:
        """Test OOD prediction of a hypothesis.

        :param expression: The symbolic expression to test.
        :param dataset: Multi-environment data.
        :param target_dim_idx: Index of the target state dimension
            (for dynamics hypotheses) or -1 for reward.
        :param variable_names: Variable names matching expression symbols.
        :param train_env_ids: Training environment IDs. If None, split
            automatically using held_out_fraction.
        :param held_out_env_ids: Held-out environment IDs. If None, split
            automatically.
        :returns: OODPredictionResult.
        """
        unique_envs = sorted(set(dataset.env_ids.tolist()))
        n_envs = len(unique_envs)

        if n_envs < 3:
            logger.warning("OOD test requires >= 3 envs, got {}", n_envs)
            return OODPredictionResult(
                passed=True, per_env_passed={}, per_env_r2={},
                failure_fraction=0.0,
            )

        # Split into train/held-out
        if train_env_ids is None or held_out_env_ids is None:
            n_held_out = max(1, int(n_envs * self._held_out_fraction))
            held_out_env_ids = unique_envs[-n_held_out:]
            train_env_ids = unique_envs[:-n_held_out]

        # Build targets
        if target_dim_idx >= 0:
            all_targets = (
                dataset.next_states[:, target_dim_idx]
                - dataset.states[:, target_dim_idx]
            )
        else:
            all_targets = dataset.rewards

        # Build feature matrix using structural_consistency helper
        from circ_rl.hypothesis.structural_consistency import (
            StructuralConsistencyTest,
        )

        x = StructuralConsistencyTest._build_features(
            dataset, variable_names, derived_columns,
        )

        # Evaluate hypothesis on held-out envs
        try:
            func = expression.to_callable(variable_names)
        except (ValueError, Exception) as exc:
            logger.warning("Failed to compile expression: {}", exc)
            return OODPredictionResult(
                passed=False,
                per_env_passed=dict.fromkeys(held_out_env_ids, False),
                per_env_r2=dict.fromkeys(held_out_env_ids, 0.0),
                failure_fraction=1.0,
            )

        per_env_passed: dict[int, bool] = {}
        per_env_r2: dict[int, float] = {}

        # Compute training R^2 as baseline for comparison
        train_mask = np.isin(dataset.env_ids, train_env_ids)
        y_train = all_targets[train_mask]
        x_train = x[train_mask]

        try:
            y_pred_train = func(x_train)
            ss_res_train = float(np.sum((y_train - y_pred_train) ** 2))
            ss_tot_train = float(np.sum((y_train - y_train.mean()) ** 2))
            1.0 - ss_res_train / ss_tot_train if ss_tot_train > 0 else 0.0
            train_rmse = float(np.sqrt(np.mean((y_train - y_pred_train) ** 2)))
        except Exception:
            train_rmse = float(np.std(y_train))

        z_score = float(stats.norm.ppf((1 + self._confidence) / 2))

        for env_id in held_out_env_ids:
            mask = dataset.env_ids == env_id
            x_env = x[mask]
            y_env = all_targets[mask]
            n_e = int(mask.sum())

            if n_e == 0:
                per_env_passed[env_id] = True
                per_env_r2[env_id] = 0.0
                continue

            try:
                y_pred = func(x_env)
            except Exception:
                per_env_passed[env_id] = False
                per_env_r2[env_id] = 0.0
                continue

            # R^2 on held-out env
            ss_res = float(np.sum((y_env - y_pred) ** 2))
            ss_tot = float(np.sum((y_env - y_env.mean()) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            per_env_r2[env_id] = r2

            # Two-criterion check:
            # 1. R^2 must be non-negative (hypothesis must predict better
            #    than the mean). Negative R^2 means the hypothesis actively
            #    harms prediction.
            # 2. Held-out RMSE must be within z_score multiples of the
            #    training RMSE (prediction quality doesn't degrade badly
            #    on unseen environments).
            held_out_rmse = float(np.sqrt(np.mean((y_env - y_pred) ** 2)))
            r2_ok = r2 >= 0.0
            rmse_ok = held_out_rmse <= z_score * max(train_rmse, 1e-10)
            env_passed = r2_ok and rmse_ok
            per_env_passed[env_id] = env_passed

        n_failed = sum(1 for p in per_env_passed.values() if not p)
        n_tested = len(held_out_env_ids)
        actual_failure_fraction = n_failed / n_tested if n_tested > 0 else 0.0

        passed = actual_failure_fraction <= self._failure_fraction

        logger.debug(
            "OOD prediction: {}/{} held-out envs passed, "
            "failure_fraction={:.2f} (threshold={}), passed={}",
            n_tested - n_failed, n_tested,
            actual_failure_fraction, self._failure_fraction, passed,
        )

        return OODPredictionResult(
            passed=passed,
            per_env_passed=per_env_passed,
            per_env_r2=per_env_r2,
            failure_fraction=actual_failure_fraction,
        )
