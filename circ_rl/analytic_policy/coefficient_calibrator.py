"""Online and offline coefficient calibration for transferred functional forms.

Given a validated functional form h(x), re-estimates the linear
calibration coefficients (alpha, beta) such that:

    delta = alpha * h(x) + beta

This enables rapid adaptation to new environments without re-running
symbolic regression. The structure (functional form) is the invariant;
the coefficients are re-estimated per-environment.

See ``CIRC-RL_Framework.md`` Section 7.2 (Online Coefficient Calibration).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from circ_rl.environments.data_collector import ExploratoryDataset
    from circ_rl.hypothesis.expression import SymbolicExpression


@dataclass(frozen=True)
class CalibrationResult:
    """Per-environment coefficient calibration with uncertainty.

    :param env_id: Environment index.
    :param alpha: Scale coefficient from ``y = alpha * h(x) + beta``.
    :param beta: Offset coefficient (intercept).
    :param alpha_se: Standard error of alpha.
    :param beta_se: Standard error of beta.
    :param covariance: 2x2 covariance matrix of ``(beta, alpha)``,
        matching the OLS design matrix ``[1, h(x)]``.
    :param mse: Residual MSE for this environment.
    :param n_samples: Number of samples used for the fit.
    """

    env_id: int
    alpha: float
    beta: float
    alpha_se: float
    beta_se: float
    covariance: np.ndarray  # (2, 2)
    mse: float
    n_samples: int


@dataclass(frozen=True)
class CoefficientUncertainty:
    """Aggregated coefficient uncertainty across environments.

    :param per_env: Per-environment calibration results.
    :param pooled_alpha: Pooled (global) alpha.
    :param pooled_beta: Pooled (global) beta.
    :param pooled_covariance: Pooled 2x2 covariance of
        ``(beta, alpha)``.
    """

    per_env: dict[int, CalibrationResult]
    pooled_alpha: float
    pooled_beta: float
    pooled_covariance: np.ndarray  # (2, 2)


class CoefficientCalibrator:
    """Extract coefficient confidence sets from calibration regressions.

    Fits ``y = alpha * h(x) + beta`` per environment and pooled,
    computing standard errors and covariance matrices from the OLS
    residuals. The covariance enables scenario sampling for robust
    MPC.

    See ``CIRC-RL_Framework.md`` Section 7.2.

    :param confidence_level: Confidence level for the coefficient
        confidence region. Default 0.95.
    """

    def __init__(self, confidence_level: float = 0.95) -> None:
        if confidence_level <= 0 or confidence_level >= 1:
            raise ValueError(
                f"confidence_level must be in (0, 1), "
                f"got {confidence_level}"
            )
        self._confidence_level = confidence_level

    def calibrate(
        self,
        expression: SymbolicExpression,
        dataset: ExploratoryDataset,
        target_dim_idx: int,
        variable_names: list[str],
        derived_columns: dict[str, np.ndarray] | None = None,
        wrap_angular: bool = False,
    ) -> CoefficientUncertainty:
        """Run calibration regression and extract uncertainty.

        :param expression: The validated symbolic expression.
        :param dataset: Multi-environment data.
        :param target_dim_idx: Target state dimension (>=0 for
            dynamics) or -1 for reward.
        :param variable_names: Variable names for expression evaluation.
        :param derived_columns: Pre-computed derived feature arrays.
        :param wrap_angular: If True, wrap target deltas via atan2.
        :returns: CoefficientUncertainty with per-env and pooled results.
        """
        from circ_rl.hypothesis.structural_consistency import (
            StructuralConsistencyTest,
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
        features = StructuralConsistencyTest._build_features(
            dataset, variable_names, derived_columns,
        )  # (N, n_vars)

        n_total = len(targets)

        # Evaluate h(x) on all data
        func = expression.to_callable(variable_names)
        h_x = np.asarray(func(features), dtype=np.float64).ravel()  # (N,)
        if h_x.shape[0] == 1:
            h_x = np.broadcast_to(h_x, (n_total,)).copy()

        # Pooled fit: y = alpha * h(x) + beta
        design_pooled = np.column_stack(
            [np.ones(n_total), h_x],
        )  # (N, 2)
        pooled_alpha, pooled_beta, pooled_cov = self._ols_with_covariance(
            design_pooled, targets,
        )

        # Per-environment fits
        unique_envs = sorted(set(dataset.env_ids.tolist()))
        per_env: dict[int, CalibrationResult] = {}

        for env_id in unique_envs:
            mask = dataset.env_ids == env_id
            h_env = h_x[mask]  # (n_e,)
            y_env = targets[mask]  # (n_e,)
            n_e = int(mask.sum())

            if n_e <= 2:
                continue

            design_env = np.column_stack(
                [np.ones(n_e), h_env],
            )  # (n_e, 2)

            alpha_e, beta_e, cov_e = self._ols_with_covariance(
                design_env, y_env,
            )

            # Standard errors from diagonal of covariance
            beta_se = float(np.sqrt(max(cov_e[0, 0], 0.0)))
            alpha_se = float(np.sqrt(max(cov_e[1, 1], 0.0)))

            # Residual MSE
            y_pred = design_env @ np.array([beta_e, alpha_e])
            mse_e = float(np.mean((y_env - y_pred) ** 2))

            per_env[env_id] = CalibrationResult(
                env_id=env_id,
                alpha=alpha_e,
                beta=beta_e,
                alpha_se=alpha_se,
                beta_se=beta_se,
                covariance=cov_e,
                mse=mse_e,
                n_samples=n_e,
            )

        logger.debug(
            "Coefficient calibration: {} envs, pooled alpha={:.4f}, "
            "beta={:.4f}",
            len(per_env), pooled_alpha, pooled_beta,
        )

        return CoefficientUncertainty(
            per_env=per_env,
            pooled_alpha=pooled_alpha,
            pooled_beta=pooled_beta,
            pooled_covariance=pooled_cov,
        )

    @staticmethod
    def _ols_with_covariance(
        design: np.ndarray,
        targets: np.ndarray,
    ) -> tuple[float, float, np.ndarray]:
        """Fit OLS and return (alpha, beta, covariance).

        Design matrix is ``[1, h(x)]`` (intercept first), so
        coefficients are ``[beta, alpha]``.

        :returns: Tuple of (alpha, beta, covariance_2x2).
        """
        n = design.shape[0]
        try:
            xtx = design.T @ design  # (2, 2)
            xtx_inv = np.linalg.inv(xtx)  # (2, 2)
            coeffs = xtx_inv @ design.T @ targets  # (2,)
        except np.linalg.LinAlgError:
            return 1.0, 0.0, np.eye(2) * 1e6

        beta_val = float(coeffs[0])
        alpha_val = float(coeffs[1])

        residuals = targets - design @ coeffs  # (n,)
        dof = max(n - 2, 1)
        residual_var = float(np.sum(residuals ** 2)) / dof
        covariance = residual_var * xtx_inv  # (2, 2)

        return alpha_val, beta_val, covariance


class OnlineCoefficientCalibrator:
    """Incrementally calibrate alpha, beta from observed transitions.

    Collects state-action-next_state tuples, evaluates the fixed
    functional form h(x) on them, and fits ``(alpha, beta)`` via
    OLS when enough samples are accumulated. Operates on a rolling
    window of the most recent transitions.

    See ``CIRC-RL_Framework.md`` Section 7.2.

    :param expression: The validated symbolic expression (fixed form).
    :param variable_names: Variable names for expression evaluation.
    :param target_dim_idx: Which state dimension this expression predicts.
    :param min_samples: Minimum transitions before calibration. Default 10.
    :param max_samples: Maximum buffer size (rolling window). Default 100.
    """

    def __init__(
        self,
        expression: SymbolicExpression,
        variable_names: list[str],
        target_dim_idx: int,
        min_samples: int = 10,
        max_samples: int = 100,
    ) -> None:
        if min_samples < 2:
            raise ValueError(
                f"min_samples must be >= 2, got {min_samples}"
            )
        if max_samples < min_samples:
            raise ValueError(
                f"max_samples ({max_samples}) must be >= "
                f"min_samples ({min_samples})"
            )
        self._expression = expression
        self._variable_names = variable_names
        self._target_dim_idx = target_dim_idx
        self._min_samples = min_samples
        self._max_samples = max_samples
        self._fn = expression.to_callable(variable_names)

        # Rolling buffer
        self._h_values: list[float] = []
        self._y_values: list[float] = []
        self._alpha: float = 1.0
        self._beta: float = 0.0
        self._r2: float | None = None

    @property
    def is_calibrated(self) -> bool:
        """Whether enough samples have been collected for calibration."""
        return len(self._h_values) >= self._min_samples

    @property
    def r2(self) -> float | None:
        """R2 of the calibrated fit, or None if not yet calibrated."""
        return self._r2

    def observe(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
    ) -> None:
        """Record a transition for calibration.

        :param state: Current state, shape ``(state_dim,)``.
        :param action: Action taken, shape ``(action_dim,)``.
        :param next_state: Resulting state, shape ``(state_dim,)``.
        """
        # Compute h(x) using the fixed functional form
        x = np.concatenate([state, action]).reshape(1, -1)  # (1, n_vars)
        h_val = float(np.asarray(self._fn(x)).ravel()[0])

        # Compute observed delta
        y_val = float(
            next_state[self._target_dim_idx]
            - state[self._target_dim_idx]
        )

        self._h_values.append(h_val)
        self._y_values.append(y_val)

        # Trim to max_samples (rolling window)
        if len(self._h_values) > self._max_samples:
            self._h_values = self._h_values[-self._max_samples:]
            self._y_values = self._y_values[-self._max_samples:]

        # Recalibrate if we have enough
        if len(self._h_values) >= self._min_samples:
            self._fit()

    def _fit(self) -> None:
        """Fit alpha, beta via OLS: y = alpha * h(x) + beta."""
        h = np.array(self._h_values)  # (N,)
        y = np.array(self._y_values)  # (N,)
        n = len(h)

        design = np.column_stack([np.ones(n), h])  # (N, 2)
        try:
            coeffs = np.linalg.lstsq(design, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            return

        self._beta = float(coeffs[0])
        self._alpha = float(coeffs[1])

        # R2
        y_pred = design @ coeffs
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        self._r2 = 1.0 - ss_res / max(ss_tot, 1e-12)

    def get_coefficients(self) -> tuple[float, float]:
        """Return ``(alpha, beta)``, defaulting to ``(1.0, 0.0)``."""
        return (self._alpha, self._beta)

    def reset(self) -> None:
        """Clear all buffered observations and calibration."""
        self._h_values.clear()
        self._y_values.clear()
        self._alpha = 1.0
        self._beta = 0.0
        self._r2 = None
