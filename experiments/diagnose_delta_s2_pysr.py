# ruff: noqa: T201
"""Test PySR with different configurations to find the correct delta_s2 expression.

Tests several PySR configurations to identify which settings allow
discovery of the true expression: 0.075*g*sin(phi_0)/l + 0.15*action/(m*l^2)
"""
from __future__ import annotations

import time

import numpy as np

from circ_rl.environments.data_collector import DataCollector, ExploratoryDataset
from circ_rl.environments.env_family import EnvironmentFamily
from circ_rl.hypothesis.symbolic_regressor import (
    SymbolicRegressionConfig,
    SymbolicRegressor,
)
from circ_rl.observation_analysis.observation_analyzer import ObservationAnalyzer
from circ_rl.utils.seeding import seed_everything


def main() -> None:
    seed_everything(42)

    # Create same data as pendulum_compare
    train_family = EnvironmentFamily.from_gymnasium(
        base_env="Pendulum-v1",
        param_distributions={
            "g": (8.0, 12.0),
            "m": (0.8, 1.5),
            "l": (0.7, 1.3),
        },
        n_envs=25,
        seed=42,
    )

    collector = DataCollector(train_family, include_env_params=True)
    dataset = collector.collect(n_transitions_per_env=5000, seed=42)

    # Canonical coordinates
    analyzer = ObservationAnalyzer()
    oa_result = analyzer.analyze(dataset, ["s0", "s1", "s2"])

    canonical_dataset = ExploratoryDataset(
        states=oa_result.canonical_states,
        actions=dataset.actions,
        next_states=oa_result.canonical_next_states,
        rewards=dataset.rewards,
        env_ids=dataset.env_ids,
        env_params=dataset.env_params,
    )

    # Build features and target
    phi_0 = canonical_dataset.states[:, 0]
    s2 = canonical_dataset.states[:, 1]
    action = dataset.actions.ravel()

    assert dataset.env_params is not None
    g_arr = dataset.env_params[:, 0]
    m_arr = dataset.env_params[:, 1]
    l_arr = dataset.env_params[:, 2]

    delta_s2 = canonical_dataset.next_states[:, 1] - canonical_dataset.states[:, 1]

    features = np.column_stack([phi_0, s2, action, g_arr, m_arr, l_arr])
    var_names = ["phi_0", "s2", "action", "g", "m", "l"]

    # Ground truth R2 for reference
    accel = -3.0 * g_arr / (2.0 * l_arr) * np.sin(phi_0 + np.pi) + \
            3.0 / (m_arr * l_arr**2) * action
    pred_true = accel * 0.05
    r2_true = 1.0 - np.var(delta_s2 - pred_true) / np.var(delta_s2)
    print(f"Ground truth R2 (no clip): {r2_true:.6f}")

    # ======================================================================
    # Configuration A: Current (pendulum_compare.py settings)
    # ======================================================================
    configs = {
        "A: Current settings": SymbolicRegressionConfig(
            max_complexity=30,
            n_iterations=80,
            populations=25,
            binary_operators=("+", "-", "*", "/"),
            unary_operators=("sin", "cos", "square", "sqrt", "abs"),
            parsimony=0.0005,
            timeout_seconds=600,
            deterministic=True,
            random_state=42,
            nested_constraints={
                "*": {"+": 0, "-": 0},
                "/": {"+": 0, "-": 0},
            },
            complexity_of_operators={"square": 1, "sin": 2, "cos": 2},
            max_samples=10000,
        ),
        "B: +constraints on sin nesting, more iterations": SymbolicRegressionConfig(
            max_complexity=30,
            n_iterations=200,
            populations=30,
            binary_operators=("+", "-", "*", "/"),
            unary_operators=("sin", "cos", "square", "sqrt", "abs"),
            parsimony=0.0005,
            timeout_seconds=600,
            deterministic=True,
            random_state=42,
            nested_constraints={
                "*": {"+": 0, "-": 0},
                "/": {"+": 0, "-": 0},
                "sin": {"sin": 0, "cos": 0},
                "cos": {"sin": 0, "cos": 0},
            },
            complexity_of_operators={"square": 1, "sin": 2, "cos": 2},
            constraints={"sin": 5, "cos": 5},  # max 5 nodes inside sin/cos
            max_samples=10000,
        ),
        "C: Like B + more samples + multiple runs": SymbolicRegressionConfig(
            max_complexity=30,
            n_iterations=200,
            populations=30,
            binary_operators=("+", "-", "*", "/"),
            unary_operators=("sin", "cos", "square", "sqrt", "abs"),
            parsimony=0.0005,
            timeout_seconds=600,
            deterministic=True,
            random_state=42,
            nested_constraints={
                "*": {"+": 0, "-": 0},
                "/": {"+": 0, "-": 0},
                "sin": {"sin": 0, "cos": 0},
                "cos": {"sin": 0, "cos": 0},
            },
            complexity_of_operators={"square": 1, "sin": 2, "cos": 2},
            constraints={"sin": 5, "cos": 5},
            max_samples=30000,
            n_sr_runs=3,
        ),
    }

    for name, config in configs.items():
        print(f"\n{'='*70}")
        print(f"Config {name}")
        print(f"  iterations={config.n_iterations}, populations={config.populations}, "
              f"max_samples={config.max_samples}, n_sr_runs={config.n_sr_runs}")
        print(f"{'='*70}")

        regressor = SymbolicRegressor(config)
        t0 = time.time()
        expressions = regressor.fit(features, delta_s2, var_names)
        elapsed = time.time() - t0
        print(f"  Time: {elapsed:.1f}s, Found: {len(expressions)} expressions")

        for expr in expressions:
            func = expr.to_callable(var_names)
            y_pred = func(features[:10000])
            rng = np.random.default_rng(42)
            idx = rng.choice(len(delta_s2), 10000, replace=False)
            y_pred_full = func(features)
            r2 = 1.0 - np.var(delta_s2 - y_pred_full) / np.var(delta_s2)
            print(f"    C={expr.complexity:2d}  R2={r2:.6f}  {expr.expression_str}")

        # Check if any expression has R2 > 0.99
        best_r2 = max(
            1.0 - np.var(delta_s2 - expr.to_callable(var_names)(features)) / np.var(delta_s2)
            for expr in expressions
        )
        print(f"\n  Best R2: {best_r2:.6f} (target: {r2_true:.6f})")

        if best_r2 >= 0.99:
            print("  --> SUCCESS: found expression with R2 >= 0.99")
        else:
            print(f"  --> FAILED: best R2={best_r2:.6f} < 0.99")


if __name__ == "__main__":
    main()
