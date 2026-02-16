"""Tests for hypothesis falsification (M7)."""

from __future__ import annotations

import numpy as np
import pytest
import sympy

from circ_rl.environments.data_collector import ExploratoryDataset
from circ_rl.hypothesis.expression import SymbolicExpression
from circ_rl.hypothesis.falsification_engine import (
    FalsificationConfig,
    FalsificationEngine,
)
from circ_rl.hypothesis.hypothesis_register import (
    HypothesisEntry,
    HypothesisRegister,
    HypothesisStatus,
)
from circ_rl.hypothesis.mdl_symbolic import SymbolicMDLScorer
from circ_rl.hypothesis.ood_prediction import OODPredictionTest
from circ_rl.hypothesis.structural_consistency import StructuralConsistencyTest
from circ_rl.hypothesis.trajectory_prediction import TrajectoryPredictionTest


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_linear_dataset(
    n_envs: int = 5,
    n_per_env: int = 200,
    state_dim: int = 2,
    coeff: float = 2.0,
    noise_std: float = 0.1,
    wrong_coeff: float | None = None,
    seed: int = 42,
) -> tuple[ExploratoryDataset, list[str]]:
    """Create a dataset where delta_s0 = coeff * action + noise.

    If wrong_coeff is given, use a different coefficient per env
    (simulating a wrong hypothesis).
    """
    rng = np.random.RandomState(seed)

    all_states = []
    all_actions = []
    all_next_states = []
    all_rewards = []
    all_env_ids = []

    for env_id in range(n_envs):
        states = rng.randn(n_per_env, state_dim).astype(np.float32)
        actions = rng.randn(n_per_env, 1).astype(np.float32)

        c = coeff if wrong_coeff is None else (coeff + wrong_coeff * env_id)
        delta_s0 = c * actions[:, 0] + noise_std * rng.randn(n_per_env)
        delta_s1 = 0.5 * states[:, 0] + noise_std * rng.randn(n_per_env)

        next_states = states.copy()
        next_states[:, 0] += delta_s0.astype(np.float32)
        next_states[:, 1] += delta_s1.astype(np.float32)

        rewards = (states[:, 0] + actions[:, 0]).astype(np.float32)

        all_states.append(states)
        all_actions.append(actions)
        all_next_states.append(next_states)
        all_rewards.append(rewards)
        all_env_ids.append(np.full(n_per_env, env_id, dtype=np.int32))

    state_names = [f"s{i}" for i in range(state_dim)]

    return (
        ExploratoryDataset(
            states=np.concatenate(all_states),
            actions=np.concatenate(all_actions),
            next_states=np.concatenate(all_next_states),
            rewards=np.concatenate(all_rewards),
            env_ids=np.concatenate(all_env_ids),
        ),
        state_names,
    )


def _make_expression(formula: str) -> SymbolicExpression:
    """Parse a string formula into a SymbolicExpression."""
    s0, s1, action = sympy.symbols("s0 s1 action")
    expr = sympy.sympify(formula, locals={"s0": s0, "s1": s1, "action": action})
    return SymbolicExpression.from_sympy(expr)


# ------------------------------------------------------------------
# Structural Consistency tests
# ------------------------------------------------------------------


class TestStructuralConsistency:
    def test_passes_correct_relationship(self):
        """A correct hypothesis (same linear relationship across envs) passes."""
        dataset, state_names = _make_linear_dataset(coeff=2.0, noise_std=0.1)
        expr = _make_expression("2.0 * action")
        var_names = ["s0", "s1", "action"]

        test = StructuralConsistencyTest(p_threshold=0.01)
        result = test.test(expr, dataset, target_dim_idx=0, variable_names=var_names)

        assert result.passed

    def test_detects_wrong_relationship(self):
        """A wrong hypothesis (coeff varies by env) is detected."""
        # wrong_coeff causes delta_s0 coefficient to vary per env
        dataset, state_names = _make_linear_dataset(
            coeff=2.0, wrong_coeff=2.0, noise_std=0.01,
        )
        expr = _make_expression("2.0 * action")
        var_names = ["s0", "s1", "action"]

        test = StructuralConsistencyTest(p_threshold=0.01)
        result = test.test(expr, dataset, target_dim_idx=0, variable_names=var_names)

        assert not result.passed


# ------------------------------------------------------------------
# OOD Prediction tests
# ------------------------------------------------------------------


class TestOODPrediction:
    def test_accepts_valid_hypothesis(self):
        """A correct hypothesis passes OOD prediction on held-out envs."""
        dataset, state_names = _make_linear_dataset(
            n_envs=6, coeff=2.0, noise_std=0.1,
        )
        expr = _make_expression("2.0 * action")
        var_names = ["s0", "s1", "action"]

        test = OODPredictionTest(
            confidence=0.99, failure_fraction=0.5, held_out_fraction=0.3,
        )
        result = test.test(expr, dataset, target_dim_idx=0, variable_names=var_names)

        assert result.passed

    def test_rejects_overfitting_hypothesis(self):
        """A wildly wrong hypothesis fails OOD prediction."""
        dataset, state_names = _make_linear_dataset(
            n_envs=6, coeff=2.0, noise_std=0.01,
        )
        # Completely wrong expression
        expr = _make_expression("100.0 * s0 * action")
        var_names = ["s0", "s1", "action"]

        test = OODPredictionTest(
            confidence=0.99, failure_fraction=0.2, held_out_fraction=0.3,
        )
        result = test.test(expr, dataset, target_dim_idx=0, variable_names=var_names)

        assert not result.passed


# ------------------------------------------------------------------
# Trajectory Prediction tests
# ------------------------------------------------------------------


class TestTrajectoryPrediction:
    def test_passes_accurate_model(self):
        """Accurate dynamics hypothesis predicts trajectories well."""
        dataset, state_names = _make_linear_dataset(
            n_envs=3, n_per_env=500, coeff=0.5, noise_std=0.01,
        )
        # Correct expression for dim 0
        expr = _make_expression("0.5 * action")
        var_names = ["s0", "s1", "action"]

        test = TrajectoryPredictionTest(
            max_horizon=10,
            divergence_threshold_factor=1.0,
            failure_fraction=0.5,
            n_trajectories=5,
        )
        result = test.test(
            dynamics_expressions={0: expr},
            dataset=dataset,
            state_feature_names=state_names,
            variable_names=var_names,
        )

        assert result.passed

    def test_detects_divergence(self):
        """Wrong dynamics hypothesis causes trajectory divergence."""
        dataset, state_names = _make_linear_dataset(
            n_envs=3, n_per_env=500, coeff=0.5, noise_std=0.01,
        )
        # Wrong expression (10x too strong)
        expr = _make_expression("5.0 * action")
        var_names = ["s0", "s1", "action"]

        test = TrajectoryPredictionTest(
            max_horizon=20,
            divergence_threshold_factor=0.1,
            failure_fraction=0.1,
            n_trajectories=10,
        )
        result = test.test(
            dynamics_expressions={0: expr},
            dataset=dataset,
            state_feature_names=state_names,
            variable_names=var_names,
        )

        assert not result.passed


# ------------------------------------------------------------------
# Symbolic MDL tests
# ------------------------------------------------------------------


class TestSymbolicMDL:
    def test_prefers_simpler_expression(self):
        """MDL prefers simpler expression when both fit similarly."""
        dataset, state_names = _make_linear_dataset(
            n_envs=3, coeff=2.0, noise_std=0.1,
        )
        var_names = ["s0", "s1", "action"]

        # Simple expression (correct)
        simple = _make_expression("2.0 * action")
        # Complex expression (also fits but more complex)
        complex_expr = _make_expression(
            "2.0 * action + 0.001 * s0 * s1"
        )

        scorer = SymbolicMDLScorer()
        score_simple = scorer.score(simple, dataset, target_dim_idx=0, variable_names=var_names)
        score_complex = scorer.score(complex_expr, dataset, target_dim_idx=0, variable_names=var_names)

        # Simpler should have lower (better) MDL score
        assert score_simple.total < score_complex.total
        assert score_simple.symbolic_complexity < score_complex.symbolic_complexity


# ------------------------------------------------------------------
# FalsificationEngine tests
# ------------------------------------------------------------------


class TestFalsificationEngine:
    def test_full_pipeline(self):
        """End-to-end: correct hypothesis is validated, wrong is falsified."""
        dataset, state_names = _make_linear_dataset(
            n_envs=6, n_per_env=300, coeff=2.0, noise_std=0.05,
        )
        var_names = ["s0", "s1", "action"]

        register = HypothesisRegister()

        # Good hypothesis
        good_expr = _make_expression("2.0 * action")
        register.register(HypothesisEntry(
            hypothesis_id="dyn_delta_s0_good",
            target_variable="delta_s0",
            expression=good_expr,
            complexity=good_expr.complexity,
            training_r2=0.99,
            training_mse=0.01,
        ))

        # Bad hypothesis (completely wrong)
        bad_expr = _make_expression("50.0 * s0")
        register.register(HypothesisEntry(
            hypothesis_id="dyn_delta_s0_bad",
            target_variable="delta_s0",
            expression=bad_expr,
            complexity=bad_expr.complexity,
            training_r2=0.1,
            training_mse=10.0,
        ))

        config = FalsificationConfig(
            structural_p_threshold=0.01,
            ood_confidence=0.99,
            ood_failure_fraction=0.5,
            trajectory_failure_fraction=0.5,
            held_out_fraction=0.2,
            trajectory_horizon=10,
            trajectory_divergence_factor=1.0,
            n_test_trajectories=5,
        )

        engine = FalsificationEngine(config)
        result = engine.run(register, dataset, state_names, var_names)

        assert result.n_tested == 2
        assert result.n_validated >= 1

        # Good hypothesis should be validated
        good_entry = register.get("dyn_delta_s0_good")
        assert good_entry.status == HypothesisStatus.VALIDATED
        assert good_entry.mdl_score is not None

        # Best for delta_s0 should be the good one
        best = register.select_best("delta_s0")
        assert best is not None
        assert best.hypothesis_id == "dyn_delta_s0_good"
