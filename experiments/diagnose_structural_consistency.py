# ruff: noqa: T201
"""Verify: does the true delta_s2 expression get falsified by structural consistency?

Computes the exact structural consistency metrics (F-stat, p-value,
relative_improvement, pooled R2) for the ground truth expression to
confirm that velocity clipping is the cause of falsification.
"""
from __future__ import annotations

import numpy as np
import sympy

from circ_rl.environments.data_collector import DataCollector, ExploratoryDataset
from circ_rl.environments.env_family import EnvironmentFamily
from circ_rl.hypothesis.expression import SymbolicExpression
from circ_rl.hypothesis.structural_consistency import StructuralConsistencyTest
from circ_rl.observation_analysis.observation_analyzer import ObservationAnalyzer
from circ_rl.utils.seeding import seed_everything


def main() -> None:
    seed_everything(42)

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

    # Canonical coords
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

    canonical_names = oa_result.canonical_state_names
    canonical_derived: dict[str, np.ndarray] = {}
    for i, cname in enumerate(canonical_names):
        canonical_derived[cname] = canonical_dataset.states[:, i]

    variable_names = list(canonical_names) + ["action"] + ["g", "m", "l"]

    # Build expressions to test
    phi_0_sym, s2_sym = sympy.symbols("phi_0 s2")
    action_sym = sympy.Symbol("action")
    g_sym, m_sym, l_sym = sympy.symbols("g m l")

    test_exprs = [
        (
            "true delta_s2 (exact)",
            0.075 * g_sym * sympy.sin(phi_0_sym) / l_sym
            + 0.15 * action_sym / (m_sym * l_sym**2),
        ),
        (
            "PySR-found (C=20, R2=0.997)",
            0.14847983 * action_sym / (m_sym * l_sym**2)
            + 0.07478791 * g_sym * sympy.sin(phi_0_sym) / l_sym,
        ),
    ]

    # Test with different min_relative_improvement thresholds
    for thresh in [0.01, 0.05, 0.10, 0.50]:
        print(f"\n{'='*70}")
        print(f"Structural consistency test (min_relative_improvement={thresh})")
        print(f"{'='*70}")

        struct_test = StructuralConsistencyTest(
            p_threshold=0.01,
            min_relative_improvement=thresh,
        )

        for name, sympy_expr in test_exprs:
            se = SymbolicExpression.from_sympy(sympy_expr)
            result = struct_test.test(
                se,
                canonical_dataset,
                target_dim_idx=1,  # s2 dimension
                variable_names=variable_names,
                derived_columns=canonical_derived,
                wrap_angular=False,
            )

            # Also compute pooled R2
            func = se.to_callable(variable_names)
            x = StructuralConsistencyTest._build_features(
                canonical_dataset, variable_names, canonical_derived,
            )
            y = canonical_dataset.next_states[:, 1] - canonical_dataset.states[:, 1]
            y_pred = func(x)
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            ss_res = float(np.sum((y - y_pred) ** 2))
            raw_r2 = 1.0 - ss_res / ss_tot

            # Calibrated R2 (what pooled model achieves)
            design = np.column_stack([np.ones(len(y)), y_pred])
            beta = np.linalg.lstsq(design, y, rcond=None)[0]
            y_cal = design @ beta
            ss_cal = float(np.sum((y - y_cal) ** 2))
            cal_r2 = 1.0 - ss_cal / ss_tot

            print(f"\n  [{name}]")
            print(f"    Raw R2: {raw_r2:.6f}")
            print(f"    Calibrated R2: {cal_r2:.6f}")
            print(f"    F-stat: {result.f_statistic:.2f}")
            print(f"    p-value: {result.p_value:.8f}")
            print(f"    Pooled MSE: {result.pooled_mse:.8f}")
            print(f"    Relative improvement: {result.relative_improvement:.6f}")
            print(f"    PASSED: {result.passed}")


if __name__ == "__main__":
    main()
