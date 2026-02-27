# ruff: noqa: T201
"""Verify: does running delta_phi_0 SR before delta_s2 SR affect results?

Tests whether PySR's Julia state leaks between sequential SR calls,
causing different results when delta_s2 is run alone vs after delta_phi_0.
"""
from __future__ import annotations

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

    train_family = EnvironmentFamily.from_gymnasium(
        base_env="Pendulum-v1",
        param_distributions={"g": (8.0, 12.0), "m": (0.8, 1.5), "l": (0.7, 1.3)},
        n_envs=25,
        seed=42,
    )

    collector = DataCollector(train_family, include_env_params=True)
    dataset = collector.collect(n_transitions_per_env=5000, seed=42)

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

    assert dataset.env_params is not None

    # Build features (same for both dims since env params are included)
    phi_0 = canonical_dataset.states[:, 0]
    s2 = canonical_dataset.states[:, 1]
    action = dataset.actions.ravel()
    g_arr = dataset.env_params[:, 0]
    m_arr = dataset.env_params[:, 1]
    l_arr = dataset.env_params[:, 2]

    features = np.column_stack([phi_0, s2, action, g_arr, m_arr, l_arr])
    var_names = ["phi_0", "s2", "action", "g", "m", "l"]

    # Target delta_phi_0
    delta_phi_0 = canonical_dataset.next_states[:, 0] - canonical_dataset.states[:, 0]
    delta_phi_0 = np.arctan2(np.sin(delta_phi_0), np.cos(delta_phi_0))

    # Target delta_s2
    delta_s2 = canonical_dataset.next_states[:, 1] - canonical_dataset.states[:, 1]

    sr_config = SymbolicRegressionConfig(
        max_complexity=30,
        n_iterations=80,
        populations=25,
        binary_operators=("+", "-", "*", "/"),
        unary_operators=("sin", "cos", "square", "sqrt", "abs"),
        parsimony=0.0005,
        timeout_seconds=600,
        deterministic=True,
        random_state=42,
        nested_constraints={"*": {"+": 0, "-": 0}, "/": {"+": 0, "-": 0}},
        complexity_of_operators={"square": 1, "sin": 2, "cos": 2},
        max_samples=10000,
    )

    regressor = SymbolicRegressor(sr_config)

    # ======================================================================
    # Test A: Run delta_s2 ALONE (like the diagnostic)
    # ======================================================================
    print("=" * 70)
    print("Test A: delta_s2 SR alone")
    print("=" * 70)
    exprs_a = regressor.fit(features, delta_s2, var_names)
    print(f"  Found {len(exprs_a)} expressions")
    for expr in exprs_a:
        y_pred = expr.to_callable(var_names)(features)
        r2 = 1.0 - np.var(delta_s2 - y_pred) / np.var(delta_s2)
        print(f"    C={expr.complexity:2d}  R2={r2:.6f}  {expr.expression_str}")

    best_a = max(
        1.0 - np.var(delta_s2 - e.to_callable(var_names)(features)) / np.var(delta_s2)
        for e in exprs_a
    )
    print(f"\n  Best R2: {best_a:.6f}")

    # ======================================================================
    # Test B: Run delta_phi_0 first, then delta_s2 (like the experiment)
    # ======================================================================
    print(f"\n{'=' * 70}")
    print("Test B: delta_phi_0 first, THEN delta_s2")
    print("=" * 70)

    print("  Running delta_phi_0 SR first...")
    exprs_phi = regressor.fit(features, delta_phi_0, var_names)
    print(f"  delta_phi_0: found {len(exprs_phi)} expressions")

    print("  Now running delta_s2 SR...")
    exprs_b = regressor.fit(features, delta_s2, var_names)
    print(f"  Found {len(exprs_b)} expressions")
    for expr in exprs_b:
        y_pred = expr.to_callable(var_names)(features)
        r2 = 1.0 - np.var(delta_s2 - y_pred) / np.var(delta_s2)
        print(f"    C={expr.complexity:2d}  R2={r2:.6f}  {expr.expression_str}")

    best_b = max(
        1.0 - np.var(delta_s2 - e.to_callable(var_names)(features)) / np.var(delta_s2)
        for e in exprs_b
    )
    print(f"\n  Best R2: {best_b:.6f}")

    # ======================================================================
    # Comparison
    # ======================================================================
    print(f"\n{'='*70}")
    print("COMPARISON")
    print(f"{'='*70}")
    print(f"  delta_s2 alone:            best R2={best_a:.6f} ({len(exprs_a)} exprs)")
    print(f"  delta_s2 after delta_phi_0: best R2={best_b:.6f} ({len(exprs_b)} exprs)")

    if abs(best_a - best_b) > 0.001:
        print("  --> DIFFERENT results! Julia state leaks between PySR calls.")
    else:
        print("  --> Same results. Julia state does NOT leak.")


if __name__ == "__main__":
    main()
