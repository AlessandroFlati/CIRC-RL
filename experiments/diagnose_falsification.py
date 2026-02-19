"""Diagnostic: why do simple delta_phi_0 expressions get falsified?

Checks ground truth R2 and runs the structural consistency test on
manually-constructed expressions of varying complexity.
"""
from __future__ import annotations

import numpy as np
import sympy

from circ_rl.environments.data_collector import DataCollector
from circ_rl.environments.env_family import EnvironmentFamily
from circ_rl.hypothesis.expression import SymbolicExpression
from circ_rl.hypothesis.structural_consistency import StructuralConsistencyTest
from circ_rl.observation_analysis.observation_analyzer import ObservationAnalyzer
from circ_rl.utils.seeding import seed_everything


def main() -> None:
    seed_everything(42)

    # Create training family (same as pendulum_compare.py)
    train_family = EnvironmentFamily.from_gymnasium(
        base_env="Pendulum-v1",
        param_distributions={
            "g": (8.0, 12.0),
            "m": (0.8, 1.5),
            "l": (0.7, 1.3),
        },
        n_envs=25,
        seed=42,
        fixed_params={"max_torque": 100.0},
    )

    # Collect data
    collector = DataCollector(train_family, include_env_params=True)
    dataset = collector.collect(n_transitions_per_env=5000, seed=42)
    env_param_names = ["g", "m", "l"]

    # Observation analysis
    analyzer = ObservationAnalyzer()
    oa_result = analyzer.analyze(dataset, ["s0", "s1", "s2"])
    print(f"Canonical coords: {oa_result.canonical_state_names}")

    # Build canonical dataset
    from circ_rl.environments.data_collector import ExploratoryDataset

    canonical_dataset = ExploratoryDataset(
        states=oa_result.canonical_states,
        actions=dataset.actions,
        next_states=oa_result.canonical_next_states,
        rewards=dataset.rewards,
        env_ids=dataset.env_ids,
        env_params=dataset.env_params,
    )
    canonical_names = oa_result.canonical_state_names

    # Build derived columns
    canonical_derived: dict[str, np.ndarray] = {}
    for i, cname in enumerate(canonical_names):
        canonical_derived[cname] = canonical_dataset.states[:, i]

    variable_names = list(canonical_names) + ["action"] + env_param_names

    # Ground truth analysis
    phi_0 = canonical_dataset.states[:, 0]
    phi_0_next = canonical_dataset.next_states[:, 0]
    s2 = canonical_dataset.states[:, 1]
    action = dataset.actions.ravel()
    delta_phi_0 = np.arctan2(
        np.sin(phi_0_next - phi_0), np.cos(phi_0_next - phi_0),
    )

    # Pendulum-v1 dynamics (semi-implicit Euler):
    #   newthdot = thdot + (-3g/(2l)*sin(th+pi) + 3/(ml^2)*u) * dt
    #   newth = th + newthdot * dt
    # So delta_theta = newthdot * dt = thdot*dt + accel*dt^2
    # = s2*0.05 + (-3g/(2l)*sin(phi_0+pi) + 3/(ml^2)*action) * 0.0025
    dt = 0.05
    dt2 = dt * dt  # 0.0025

    print("\n=== Ground truth analysis ===")

    # Model 1: simple kinematic
    pred1 = s2 * dt
    r2_1 = 1.0 - np.var(delta_phi_0 - pred1) / np.var(delta_phi_0)
    print(f"Model 1 (s2*dt):             R2={r2_1:.6f}")

    # Model 2: include gravity acceleration
    if dataset.env_params is not None:
        g_arr = dataset.env_params[:, 0]
        m_arr = dataset.env_params[:, 1]
        l_arr = dataset.env_params[:, 2]

        accel = -3.0 * g_arr / (2.0 * l_arr) * np.sin(phi_0 + np.pi) + \
                3.0 / (m_arr * l_arr**2) * action
        pred2 = s2 * dt + accel * dt2
        r2_2 = 1.0 - np.var(delta_phi_0 - pred2) / np.var(delta_phi_0)
        print(f"Model 2 (s2*dt + accel*dt2): R2={r2_2:.6f}")

        # Model 3: include velocity clipping effect (max_speed=8)
        newthdot = s2 + accel * dt
        newthdot_clipped = np.clip(newthdot, -8.0, 8.0)
        pred3 = newthdot_clipped * dt
        r2_3 = 1.0 - np.var(delta_phi_0 - pred3) / np.var(delta_phi_0)
        print(f"Model 3 (clipped full):      R2={r2_3:.6f}")

        # Check how much residual varies per env for each model
        unique_envs = sorted(set(dataset.env_ids.tolist()))
        print(f"\n  Per-env residual std:")
        print(f"  {'Env':>4s}  {'g':>6s}  {'m':>6s}  {'l':>6s}  "
              f"{'resid1':>10s}  {'resid2':>10s}")
        for env_id in unique_envs:
            mask = dataset.env_ids == env_id
            ep = dataset.env_params[mask][0]
            r1 = np.std(delta_phi_0[mask] - pred1[mask])
            r2 = np.std(delta_phi_0[mask] - pred2[mask])
            print(f"  {env_id:4d}  {ep[0]:6.2f}  {ep[1]:6.2f}  {ep[2]:6.2f}  "
                  f"{r1:10.6f}  {r2:10.6f}")

    # Now test structural consistency on manually-built expressions
    print(f"\n=== Structural consistency tests ===")

    struct_test = StructuralConsistencyTest(
        p_threshold=0.01,
        min_relative_improvement=0.01,
    )

    # Sympy symbols
    phi_0_sym, s2_sym = sympy.symbols("phi_0 s2")
    action_sym = sympy.Symbol("action")
    g_sym, m_sym, l_sym = sympy.symbols("g m l")

    test_exprs = [
        ("0.05*s2", s2_sym * 0.05),
        ("0.05*s2 + 0.0375*sin(phi_0)*g/l (approx true)",
         s2_sym * 0.05 - 0.0025 * 3 * g_sym / (2 * l_sym) * sympy.sin(phi_0_sym + sympy.pi)
         + 0.0025 * 3 / (m_sym * l_sym**2) * action_sym),
    ]

    for name, sympy_expr in test_exprs:
        se = SymbolicExpression.from_sympy(sympy_expr)
        result = struct_test.test(
            se,
            canonical_dataset,
            target_dim_idx=0,
            variable_names=variable_names,
            derived_columns=canonical_derived,
            wrap_angular=True,
        )

        # Compute R2 manually
        func = se.to_callable(variable_names)
        x = StructuralConsistencyTest._build_features(
            canonical_dataset, variable_names, canonical_derived,
        )
        y_pred = func(x)
        r2 = 1.0 - np.var(delta_phi_0 - y_pred) / np.var(delta_phi_0)

        print(f"\n  [{name}]")
        print(f"    Complexity: {se.complexity}")
        print(f"    R2: {r2:.6f}")
        print(f"    F-stat={result.f_statistic:.4f}, p={result.p_value:.8f}")
        print(f"    Pooled MSE={result.pooled_mse:.8f}")
        print(f"    Relative improvement={result.relative_improvement:.6f}")
        print(f"    PASSED={result.passed}")


if __name__ == "__main__":
    main()
