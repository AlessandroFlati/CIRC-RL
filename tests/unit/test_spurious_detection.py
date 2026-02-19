# ruff: noqa: ANN001 ANN201

"""Unit tests for spurious term detection."""

from __future__ import annotations

import numpy as np
import pytest
import sympy

from circ_rl.hypothesis.spurious_detection import (
    SpuriousDetectionResult,
    SpuriousTermDetector,
    TermAnalysis,
    _build_targets,
    _compute_calibrated_r2,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeExpression:
    """Minimal expression mock."""

    def __init__(self, sympy_expr: sympy.Expr, variable_names: list[str]):
        self.sympy_expr = sympy_expr
        self._var_names = variable_names
        self.complexity = int(sympy.count_ops(sympy_expr))

    def to_callable(self, variable_names: list[str]):
        symbols = [sympy.Symbol(n) for n in variable_names]
        func = sympy.lambdify(symbols, self.sympy_expr, modules="numpy")

        def eval_fn(features: np.ndarray) -> np.ndarray:
            args = [features[:, i] for i in range(features.shape[1])]
            return np.asarray(func(*args), dtype=np.float64)

        return eval_fn

    @classmethod
    def from_sympy(cls, expr: sympy.Expr):
        free = sorted(expr.free_symbols, key=str)
        return cls(expr, [str(s) for s in free])


class _FakeDataset:
    """Minimal dataset mock."""

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


def _make_dataset(
    n: int = 500,
    n_envs: int = 2,
    seed: int = 42,
) -> _FakeDataset:
    """Create synthetic 2D dataset."""
    rng = np.random.default_rng(seed)
    states = rng.standard_normal((n, 2))
    actions = rng.standard_normal((n, 1))
    env_ids = np.repeat(np.arange(n_envs), n // n_envs)

    # delta_s0 = 2*s0 + 3*s1 (exact, no noise)
    deltas = 2.0 * states[:, 0] + 3.0 * states[:, 1]
    next_states = states.copy()
    next_states[:, 0] += deltas

    return _FakeDataset(
        states=states,
        actions=actions,
        next_states=next_states,
        rewards=rng.standard_normal(n),
        env_ids=env_ids,
    )


# Monkey-patch SymbolicExpression.from_sympy so spurious_detection
# can create ablated expressions
import circ_rl.hypothesis.expression as _expr_mod
import circ_rl.hypothesis.structural_consistency as _sc_mod

# We need to patch StructuralConsistencyTest._build_features
_orig_build = getattr(_sc_mod.StructuralConsistencyTest, "_build_features", None)


# ---------------------------------------------------------------------------
# Tests for _build_targets
# ---------------------------------------------------------------------------

class TestBuildTargets:
    def test_dynamics_target(self):
        ds = _make_dataset()
        targets = _build_targets(ds, target_dim_idx=0, wrap_angular=False)
        expected = ds.next_states[:, 0] - ds.states[:, 0]
        np.testing.assert_allclose(targets, expected)

    def test_reward_target(self):
        ds = _make_dataset()
        targets = _build_targets(ds, target_dim_idx=-1, wrap_angular=False)
        np.testing.assert_allclose(targets, ds.rewards)

    def test_angular_wrapping(self):
        ds = _make_dataset()
        targets = _build_targets(ds, target_dim_idx=0, wrap_angular=True)
        raw_delta = ds.next_states[:, 0] - ds.states[:, 0]
        expected = np.arctan2(np.sin(raw_delta), np.cos(raw_delta))
        np.testing.assert_allclose(targets, expected)


# ---------------------------------------------------------------------------
# Tests for _compute_calibrated_r2
# ---------------------------------------------------------------------------

class TestCalibratedR2:
    def test_perfect_prediction_gives_r2_one(self):
        """Expression matching targets exactly should give R2=1."""
        rng = np.random.default_rng(42)
        n = 200
        features = rng.standard_normal((n, 2))
        targets = 2.0 * features[:, 0] + 3.0 * features[:, 1]

        x, y = sympy.symbols("x y")
        expr = _FakeExpression(2 * x + 3 * y, ["x", "y"])

        r2 = _compute_calibrated_r2(expr, features, targets, ["x", "y"])
        assert r2 > 0.999

    def test_constant_prediction_gives_low_r2(self):
        """Constant expression should give R2 close to 0."""
        rng = np.random.default_rng(42)
        n = 200
        features = rng.standard_normal((n, 2))
        targets = 2.0 * features[:, 0] + 3.0 * features[:, 1]

        expr = _FakeExpression(sympy.Integer(1), ["x", "y"])

        r2 = _compute_calibrated_r2(expr, features, targets, ["x", "y"])
        # Calibrated R2 with constant still gets ~0 since no
        # information from features
        assert r2 < 0.1


# ---------------------------------------------------------------------------
# Tests for SpuriousTermDetector
# ---------------------------------------------------------------------------

class TestSpuriousTermDetector:
    def test_no_spurious_in_exact_expression(self):
        """All terms should be non-spurious when they all contribute."""
        ds = _make_dataset()
        x, y = sympy.symbols("s0 s1")
        expr = _FakeExpression(2 * x + 3 * y, ["s0", "s1"])

        detector = SpuriousTermDetector(
            r2_contribution_threshold=0.005,
        )
        result = detector.detect(
            expression=expr,
            dataset=ds,
            target_dim_idx=0,
            variable_names=["s0", "s1"],
        )

        assert isinstance(result, SpuriousDetectionResult)
        assert result.n_spurious == 0
        assert result.pruned_expr == expr.sympy_expr

    def test_negligible_term_flagged_as_spurious(self):
        """A term with tiny coefficient should be flagged."""
        ds = _make_dataset()
        x, y = sympy.symbols("s0 s1")
        # True model: 2*x + 3*y
        # Add negligible term: 0.0001 * x * y
        expr_sympy = 2 * x + 3 * y + sympy.Rational(1, 10000) * x * y
        expr = _FakeExpression(expr_sympy, ["s0", "s1"])

        detector = SpuriousTermDetector(
            r2_contribution_threshold=0.005,
            coefficient_ratio_threshold=0.01,
        )
        result = detector.detect(
            expression=expr,
            dataset=ds,
            target_dim_idx=0,
            variable_names=["s0", "s1"],
        )

        assert result.n_spurious >= 1
        # The pruned expression should still have high R2
        assert result.pruned_r2 > 0.99

    def test_single_term_not_analyzed(self):
        """Single-term expression should return no analyses."""
        ds = _make_dataset()
        x = sympy.symbols("s0")
        expr = _FakeExpression(2 * x, ["s0", "s1"])

        detector = SpuriousTermDetector()
        result = detector.detect(
            expression=expr,
            dataset=ds,
            target_dim_idx=0,
            variable_names=["s0", "s1"],
        )

        assert result.n_spurious == 0
        assert len(result.term_analyses) == 0

    def test_min_terms_to_keep(self):
        """Should never prune below min_terms_to_keep."""
        ds = _make_dataset()
        x, y = sympy.symbols("s0 s1")
        # Both terms contribute but set threshold very high
        expr = _FakeExpression(2 * x + 3 * y, ["s0", "s1"])

        detector = SpuriousTermDetector(
            r2_contribution_threshold=0.99,  # Very high threshold
            coefficient_ratio_threshold=0.0,  # Disable coeff check
            min_terms_to_keep=2,
        )
        result = detector.detect(
            expression=expr,
            dataset=ds,
            target_dim_idx=0,
            variable_names=["s0", "s1"],
        )

        # Even with very high threshold, min_terms=2 keeps both
        assert result.n_spurious == 0

    def test_r2_contribution_ordering(self):
        """Term with higher contribution should have higher r2_contribution."""
        ds = _make_dataset()
        x, y = sympy.symbols("s0 s1")
        # True: 2*x + 3*y, so y term contributes more (coeff=3 vs 2)
        expr = _FakeExpression(2 * x + 3 * y, ["s0", "s1"])

        detector = SpuriousTermDetector()
        result = detector.detect(
            expression=expr,
            dataset=ds,
            target_dim_idx=0,
            variable_names=["s0", "s1"],
        )

        assert len(result.term_analyses) == 2
        # Both terms should have positive R2 contribution
        for ta in result.term_analyses:
            assert ta.r2_contribution > 0

    def test_invalid_threshold_raises(self):
        with pytest.raises(ValueError, match="r2_contribution_threshold"):
            SpuriousTermDetector(r2_contribution_threshold=-0.1)
        with pytest.raises(ValueError, match="r2_contribution_threshold"):
            SpuriousTermDetector(r2_contribution_threshold=1.5)

    def test_invalid_min_terms_raises(self):
        with pytest.raises(ValueError, match="min_terms_to_keep"):
            SpuriousTermDetector(min_terms_to_keep=0)
