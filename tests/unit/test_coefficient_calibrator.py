# ruff: noqa: ANN001 ANN201

"""Unit tests for coefficient calibration."""

from __future__ import annotations

import numpy as np
import pytest

from circ_rl.analytic_policy.coefficient_calibrator import (
    CalibrationResult,
    CoefficientCalibrator,
    CoefficientUncertainty,
    OnlineCoefficientCalibrator,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeExpression:
    """Minimal expression mock for testing."""

    def __init__(self, fn):
        self._fn = fn

    def to_callable(self, variable_names):
        return self._fn


class _FakeDataset:
    """Minimal dataset mock for testing."""

    def __init__(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        next_states: np.ndarray,
        rewards: np.ndarray,
        env_ids: np.ndarray,
    ):
        self.states = states
        self.actions = actions
        self.next_states = next_states
        self.rewards = rewards
        self.env_ids = env_ids


def _make_linear_dataset(
    n_per_env: int = 100,
    n_envs: int = 3,
    alpha: float = 2.0,
    beta: float = 0.5,
    noise_std: float = 0.01,
    seed: int = 42,
) -> tuple[_FakeDataset, _FakeExpression]:
    """Create a dataset where delta_s0 = alpha * h(x) + beta + noise.

    h(x) = x0 + x1 (simple sum of features).
    """
    rng = np.random.default_rng(seed)
    n_total = n_per_env * n_envs

    states = rng.standard_normal((n_total, 2))
    actions = rng.standard_normal((n_total, 1))
    env_ids = np.repeat(np.arange(n_envs), n_per_env)

    # h(x) = states[:, 0] + states[:, 1]
    h_x = states[:, 0] + states[:, 1]

    # Target: delta_s0 = alpha * h(x) + beta + noise
    deltas = alpha * h_x + beta + rng.normal(0, noise_std, n_total)
    next_states = states.copy()
    next_states[:, 0] += deltas

    dataset = _FakeDataset(
        states=states,
        actions=actions,
        next_states=next_states,
        rewards=rng.standard_normal(n_total),
        env_ids=env_ids,
    )

    def eval_fn(features: np.ndarray) -> np.ndarray:
        return features[:, 0] + features[:, 1]

    expr = _FakeExpression(eval_fn)
    return dataset, expr


# ---------------------------------------------------------------------------
# CoefficientCalibrator tests
# ---------------------------------------------------------------------------

class TestCoefficientCalibrator:
    def test_calibrate_recovers_known_coefficients(self):
        """Calibration should recover alpha and beta from synthetic data."""
        alpha_true = 2.0
        beta_true = 0.5
        dataset, expr = _make_linear_dataset(
            alpha=alpha_true, beta=beta_true,
        )

        calibrator = CoefficientCalibrator(confidence_level=0.95)
        result = calibrator.calibrate(
            expression=expr,
            dataset=dataset,
            target_dim_idx=0,
            variable_names=["s0", "s1"],
        )

        assert isinstance(result, CoefficientUncertainty)
        assert abs(result.pooled_alpha - alpha_true) < 0.1
        assert abs(result.pooled_beta - beta_true) < 0.1

    def test_per_env_results_populated(self):
        """Each environment should have its own calibration result."""
        dataset, expr = _make_linear_dataset(n_envs=4)

        calibrator = CoefficientCalibrator()
        result = calibrator.calibrate(
            expression=expr,
            dataset=dataset,
            target_dim_idx=0,
            variable_names=["s0", "s1"],
        )

        assert len(result.per_env) == 4
        for env_id, cal in result.per_env.items():
            assert isinstance(cal, CalibrationResult)
            assert cal.env_id == env_id
            assert cal.n_samples > 0
            assert cal.covariance.shape == (2, 2)

    def test_covariance_is_symmetric(self):
        """OLS covariance should be symmetric."""
        dataset, expr = _make_linear_dataset()

        calibrator = CoefficientCalibrator()
        result = calibrator.calibrate(
            expression=expr,
            dataset=dataset,
            target_dim_idx=0,
            variable_names=["s0", "s1"],
        )

        np.testing.assert_allclose(
            result.pooled_covariance,
            result.pooled_covariance.T,
            atol=1e-12,
        )

    def test_standard_errors_positive(self):
        """Standard errors should be positive."""
        dataset, expr = _make_linear_dataset()

        calibrator = CoefficientCalibrator()
        result = calibrator.calibrate(
            expression=expr,
            dataset=dataset,
            target_dim_idx=0,
            variable_names=["s0", "s1"],
        )

        for cal in result.per_env.values():
            assert cal.alpha_se > 0
            assert cal.beta_se > 0

    def test_low_noise_gives_small_covariance(self):
        """Low noise data should yield small covariance."""
        dataset, expr = _make_linear_dataset(noise_std=1e-6)

        calibrator = CoefficientCalibrator()
        result = calibrator.calibrate(
            expression=expr,
            dataset=dataset,
            target_dim_idx=0,
            variable_names=["s0", "s1"],
        )

        assert np.trace(result.pooled_covariance) < 1e-6

    def test_invalid_confidence_level_raises(self):
        with pytest.raises(ValueError, match="confidence_level"):
            CoefficientCalibrator(confidence_level=0.0)
        with pytest.raises(ValueError, match="confidence_level"):
            CoefficientCalibrator(confidence_level=1.0)

    def test_reward_target(self):
        """Calibration should work with target_dim_idx=-1 (reward)."""
        rng = np.random.default_rng(42)
        n = 200
        states = rng.standard_normal((n, 2))
        actions = rng.standard_normal((n, 1))
        # reward = 3.0 * h(x) + 1.0
        rewards = 3.0 * (states[:, 0] + states[:, 1]) + 1.0

        dataset = _FakeDataset(
            states=states,
            actions=actions,
            next_states=states,
            rewards=rewards,
            env_ids=np.zeros(n, dtype=int),
        )

        def eval_fn(features: np.ndarray) -> np.ndarray:
            return features[:, 0] + features[:, 1]

        expr = _FakeExpression(eval_fn)
        calibrator = CoefficientCalibrator()
        result = calibrator.calibrate(
            expression=expr,
            dataset=dataset,
            target_dim_idx=-1,
            variable_names=["s0", "s1"],
        )

        assert abs(result.pooled_alpha - 3.0) < 0.1
        assert abs(result.pooled_beta - 1.0) < 0.1


# ---------------------------------------------------------------------------
# OnlineCoefficientCalibrator tests
# ---------------------------------------------------------------------------

class TestOnlineCoefficientCalibrator:
    def test_not_calibrated_initially(self):
        def eval_fn(features):
            return features[:, 0]

        expr = _FakeExpression(eval_fn)
        cal = OnlineCoefficientCalibrator(
            expression=expr,
            variable_names=["s0", "a0"],
            target_dim_idx=0,
            min_samples=5,
        )
        assert not cal.is_calibrated
        assert cal.r2 is None

    def test_calibrated_after_min_samples(self):
        def eval_fn(features):
            return features[:, 0]

        expr = _FakeExpression(eval_fn)
        cal = OnlineCoefficientCalibrator(
            expression=expr,
            variable_names=["s0", "a0"],
            target_dim_idx=0,
            min_samples=5,
        )

        # Observe 5 transitions where delta = 2 * s0
        for i in range(5):
            state = np.array([float(i)])
            action = np.array([0.0])
            next_state = np.array([float(i) + 2.0 * float(i)])
            cal.observe(state, action, next_state)

        assert cal.is_calibrated
        alpha, beta = cal.get_coefficients()
        assert abs(alpha - 2.0) < 0.3

    def test_rolling_window_trims_buffer(self):
        def eval_fn(features):
            return features[:, 0]

        expr = _FakeExpression(eval_fn)
        cal = OnlineCoefficientCalibrator(
            expression=expr,
            variable_names=["s0", "a0"],
            target_dim_idx=0,
            min_samples=3,
            max_samples=5,
        )

        for i in range(10):
            state = np.array([float(i)])
            action = np.array([0.0])
            next_state = np.array([float(i) + float(i)])
            cal.observe(state, action, next_state)

        assert len(cal._h_values) == 5

    def test_reset_clears_state(self):
        def eval_fn(features):
            return features[:, 0]

        expr = _FakeExpression(eval_fn)
        cal = OnlineCoefficientCalibrator(
            expression=expr,
            variable_names=["s0", "a0"],
            target_dim_idx=0,
            min_samples=3,
        )

        for i in range(5):
            state = np.array([float(i)])
            action = np.array([0.0])
            next_state = np.array([float(i) + float(i)])
            cal.observe(state, action, next_state)

        assert cal.is_calibrated
        cal.reset()
        assert not cal.is_calibrated
        assert cal.get_coefficients() == (1.0, 0.0)

    def test_default_coefficients(self):
        def eval_fn(features):
            return features[:, 0]

        expr = _FakeExpression(eval_fn)
        cal = OnlineCoefficientCalibrator(
            expression=expr,
            variable_names=["s0", "a0"],
            target_dim_idx=0,
        )
        assert cal.get_coefficients() == (1.0, 0.0)

    def test_invalid_min_samples_raises(self):
        def eval_fn(features):
            return features[:, 0]

        expr = _FakeExpression(eval_fn)
        with pytest.raises(ValueError, match="min_samples"):
            OnlineCoefficientCalibrator(
                expression=expr,
                variable_names=["s0", "a0"],
                target_dim_idx=0,
                min_samples=1,
            )

    def test_max_less_than_min_raises(self):
        def eval_fn(features):
            return features[:, 0]

        expr = _FakeExpression(eval_fn)
        with pytest.raises(ValueError, match="max_samples"):
            OnlineCoefficientCalibrator(
                expression=expr,
                variable_names=["s0", "a0"],
                target_dim_idx=0,
                min_samples=10,
                max_samples=5,
            )
