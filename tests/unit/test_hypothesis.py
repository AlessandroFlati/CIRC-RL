"""Tests for circ_rl.hypothesis module (M6: Hypothesis Infrastructure)."""

from __future__ import annotations

import numpy as np
import pytest
import sympy

from circ_rl.hypothesis.expression import SymbolicExpression
from circ_rl.hypothesis.hypothesis_register import (
    HypothesisEntry,
    HypothesisRegister,
    HypothesisStatus,
)


# ------------------------------------------------------------------
# SymbolicExpression tests
# ------------------------------------------------------------------


class TestSymbolicExpression:
    def test_complexity_counts_tree_nodes(self):
        """Complexity = number of nodes in the expression tree."""
        x, y = sympy.symbols("x y")

        # Single symbol: 1 node
        assert SymbolicExpression.count_tree_nodes(x) == 1

        # x + y: 3 nodes (Add, x, y)
        assert SymbolicExpression.count_tree_nodes(x + y) == 3

        # x * y + 1: 5 nodes (Add, Mul, x, y, 1)
        expr = x * y + 1
        assert SymbolicExpression.count_tree_nodes(expr) == 5

        # sin(x): 2 nodes (sin, x)
        assert SymbolicExpression.count_tree_nodes(sympy.sin(x)) == 2

        # sin(x) + cos(y): 5 nodes (Add, sin, x, cos, y)
        expr = sympy.sin(x) + sympy.cos(y)
        assert SymbolicExpression.count_tree_nodes(expr) == 5

    def test_from_sympy_creates_correct_expression(self):
        x, y = sympy.symbols("x y")
        expr = x**2 + 3 * y
        se = SymbolicExpression.from_sympy(expr)
        assert se.free_symbols == frozenset({"x", "y"})
        assert se.complexity > 0
        assert se.expression_str == str(expr)

    def test_from_sympy_with_constant_symbols(self):
        x, c1 = sympy.symbols("x c1")
        expr = c1 * x + 2
        se = SymbolicExpression.from_sympy(
            expr, constant_symbols=frozenset({"c1"})
        )
        assert se.n_constants == 1
        assert "c1" in se.free_symbols
        assert "x" in se.free_symbols

    def test_expression_to_callable_evaluates_correctly(self):
        """Compiled callable matches manual evaluation."""
        x, y = sympy.symbols("x y")
        expr = 2 * x + 3 * y**2
        se = SymbolicExpression.from_sympy(expr)

        func = se.to_callable(["x", "y"])

        data = np.array([[1.0, 2.0], [3.0, 4.0], [0.0, 0.0]])
        result = func(data)
        expected = 2.0 * data[:, 0] + 3.0 * data[:, 1] ** 2
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_callable_raises_on_missing_symbol(self):
        x, y = sympy.symbols("x y")
        expr = x + y
        se = SymbolicExpression.from_sympy(expr)

        with pytest.raises(ValueError, match="free symbols"):
            se.to_callable(["x"])

    def test_callable_raises_on_wrong_shape(self):
        x = sympy.Symbol("x")
        se = SymbolicExpression.from_sympy(x)
        func = se.to_callable(["x"])

        with pytest.raises(AssertionError):
            func(np.array([[1.0, 2.0]]))  # 2 columns but 1 expected


# ------------------------------------------------------------------
# HypothesisRegister tests
# ------------------------------------------------------------------


def _make_entry(
    hid: str = "h1",
    target: str = "delta_s0",
    complexity: int = 5,
    r2: float = 0.9,
    mse: float = 0.1,
    status: HypothesisStatus = HypothesisStatus.UNTESTED,
) -> HypothesisEntry:
    x = sympy.Symbol("x")
    expr = SymbolicExpression.from_sympy(x)
    return HypothesisEntry(
        hypothesis_id=hid,
        target_variable=target,
        expression=expr,
        complexity=complexity,
        training_r2=r2,
        training_mse=mse,
        status=status,
    )


class TestHypothesisRegister:
    def test_register_and_retrieve(self):
        reg = HypothesisRegister()
        entry = _make_entry("h1")
        reg.register(entry)
        assert reg.get("h1") is entry
        assert reg.n_hypotheses == 1

    def test_register_duplicate_raises(self):
        reg = HypothesisRegister()
        reg.register(_make_entry("h1"))
        with pytest.raises(ValueError, match="already registered"):
            reg.register(_make_entry("h1"))

    def test_status_transitions(self):
        """Valid transitions: UNTESTED -> VALIDATED, UNTESTED -> FALSIFIED.
        Invalid: FALSIFIED -> anything, VALIDATED -> UNTESTED."""
        reg = HypothesisRegister()

        # UNTESTED -> VALIDATED
        reg.register(_make_entry("h1"))
        reg.update_status("h1", HypothesisStatus.VALIDATED)
        assert reg.get("h1").status == HypothesisStatus.VALIDATED

        # UNTESTED -> FALSIFIED
        reg.register(_make_entry("h2"))
        reg.update_status("h2", HypothesisStatus.FALSIFIED, reason="bad fit")
        assert reg.get("h2").status == HypothesisStatus.FALSIFIED
        assert reg.get("h2").falsification_reason == "bad fit"

        # FALSIFIED -> anything raises
        with pytest.raises(ValueError, match="falsified"):
            reg.update_status("h2", HypothesisStatus.VALIDATED)

        # VALIDATED -> UNTESTED raises
        with pytest.raises(ValueError, match="revert"):
            reg.update_status("h1", HypothesisStatus.UNTESTED)

    def test_pareto_front_ordering(self):
        """Pareto front returns non-dominated hypotheses sorted by complexity."""
        reg = HypothesisRegister()
        # h1: low complexity, low R2
        reg.register(_make_entry("h1", complexity=3, r2=0.5))
        # h2: medium complexity, high R2 (dominates h3)
        reg.register(_make_entry("h2", complexity=5, r2=0.95))
        # h3: high complexity, medium R2 (dominated by h2)
        reg.register(_make_entry("h3", complexity=10, r2=0.8))
        # h4: highest complexity, highest R2
        reg.register(_make_entry("h4", complexity=15, r2=0.99))

        front = reg.pareto_front("delta_s0")
        ids = [e.hypothesis_id for e in front]
        # h1 (3, 0.5) - not dominated by anything simpler
        # h2 (5, 0.95) - improves on h1's R2
        # h4 (15, 0.99) - improves on h2's R2
        # h3 is dominated by h2 (lower complexity AND higher R2)
        assert ids == ["h1", "h2", "h4"]

    def test_mdl_selection(self):
        """select_best returns the validated hypothesis with lowest MDL."""
        reg = HypothesisRegister()

        reg.register(_make_entry("h1", complexity=3, r2=0.5))
        reg.register(_make_entry("h2", complexity=5, r2=0.95))
        reg.register(_make_entry("h3", complexity=10, r2=0.99))

        # Validate h1 and h2
        reg.update_status("h1", HypothesisStatus.VALIDATED)
        reg.update_status("h2", HypothesisStatus.VALIDATED)
        reg.update_status("h3", HypothesisStatus.FALSIFIED, reason="OOD fail")

        # Set MDL scores (lower is better)
        reg.set_mdl_score("h1", 10.0)
        reg.set_mdl_score("h2", 5.0)

        best = reg.select_best("delta_s0")
        assert best is not None
        assert best.hypothesis_id == "h2"

    def test_select_best_fallback_no_mdl(self):
        """Without MDL scores, falls back to Pareto front selection."""
        reg = HypothesisRegister()
        reg.register(_make_entry("h1", complexity=3, r2=0.5))
        reg.register(_make_entry("h2", complexity=5, r2=0.95))

        reg.update_status("h1", HypothesisStatus.VALIDATED)
        reg.update_status("h2", HypothesisStatus.VALIDATED)

        best = reg.select_best("delta_s0")
        assert best is not None
        # Should pick from Pareto front (h1 first as simplest that's non-dominated)
        assert best.hypothesis_id == "h1"

    def test_select_best_returns_none_when_no_validated(self):
        reg = HypothesisRegister()
        reg.register(_make_entry("h1"))
        assert reg.select_best("delta_s0") is None

    def test_get_by_target(self):
        reg = HypothesisRegister()
        reg.register(_make_entry("h1", target="delta_s0"))
        reg.register(_make_entry("h2", target="reward"))
        reg.register(_make_entry("h3", target="delta_s0"))

        s0_entries = reg.get_by_target("delta_s0")
        assert len(s0_entries) == 2
        reward_entries = reg.get_by_target("reward")
        assert len(reward_entries) == 1

    def test_get_by_status(self):
        reg = HypothesisRegister()
        reg.register(_make_entry("h1"))
        reg.register(_make_entry("h2"))
        reg.update_status("h1", HypothesisStatus.VALIDATED)

        untested = reg.get_by_status(HypothesisStatus.UNTESTED)
        assert len(untested) == 1
        assert untested[0].hypothesis_id == "h2"

        validated = reg.get_by_status(HypothesisStatus.VALIDATED)
        assert len(validated) == 1
        assert validated[0].hypothesis_id == "h1"

    def test_select_best_pareto_threshold_met_by_simplest(self):
        """When simplest Pareto entry meets R2 threshold, select it."""
        reg = HypothesisRegister()
        reg.register(_make_entry("h1", complexity=3, r2=0.999))
        reg.register(_make_entry("h2", complexity=7, r2=0.9995))
        reg.register(_make_entry("h3", complexity=15, r2=0.99999))

        for hid in ("h1", "h2", "h3"):
            reg.update_status(hid, HypothesisStatus.VALIDATED)

        best = reg.select_best_pareto("delta_s0", r2_threshold=0.999)
        assert best is not None
        assert best.hypothesis_id == "h1"

    def test_select_best_pareto_threshold_met_by_middle(self):
        """When simplest doesn't meet threshold, skip to next on Pareto front."""
        reg = HypothesisRegister()
        reg.register(_make_entry("h1", complexity=3, r2=0.99))
        reg.register(_make_entry("h2", complexity=7, r2=0.9995))
        reg.register(_make_entry("h3", complexity=15, r2=0.99999))

        for hid in ("h1", "h2", "h3"):
            reg.update_status(hid, HypothesisStatus.VALIDATED)

        best = reg.select_best_pareto("delta_s0", r2_threshold=0.999)
        assert best is not None
        assert best.hypothesis_id == "h2"

    def test_select_best_pareto_no_threshold_met_fallback(self):
        """When no Pareto entry meets threshold, fall back to highest R2."""
        reg = HypothesisRegister()
        reg.register(_make_entry("h1", complexity=3, r2=0.9))
        reg.register(_make_entry("h2", complexity=7, r2=0.95))
        reg.register(_make_entry("h3", complexity=15, r2=0.97))

        for hid in ("h1", "h2", "h3"):
            reg.update_status(hid, HypothesisStatus.VALIDATED)

        best = reg.select_best_pareto("delta_s0", r2_threshold=0.999)
        assert best is not None
        assert best.hypothesis_id == "h3"

    def test_select_best_pareto_no_validated(self):
        """Returns None when no validated hypotheses exist."""
        reg = HypothesisRegister()
        reg.register(_make_entry("h1", complexity=3, r2=0.999))
        assert reg.select_best_pareto("delta_s0") is None

    def test_select_best_pareto_single_validated(self):
        """Single validated hypothesis is returned regardless of threshold."""
        reg = HypothesisRegister()
        reg.register(_make_entry("h1", complexity=5, r2=0.8))
        reg.update_status("h1", HypothesisStatus.VALIDATED)

        # R2=0.8 < threshold=0.999, but it's the only one -> fallback
        best = reg.select_best_pareto("delta_s0", r2_threshold=0.999)
        assert best is not None
        assert best.hypothesis_id == "h1"
