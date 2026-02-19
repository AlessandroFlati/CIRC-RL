# ruff: noqa: T201
"""Quick run of Trials 4-5 from delta_s2_iterate.py.

Trial 4: Constrain sin/cos to single-variable arguments.
Trial 5: Feature engineering with g/l and 1/(m*l^2).
"""

from __future__ import annotations

import sys
import time

import numpy as np
from loguru import logger

from circ_rl.environments.data_collector import (
    DataCollector,
    ExploratoryDataset,
)
from circ_rl.environments.env_family import EnvironmentFamily
from circ_rl.hypothesis.structural_consistency import StructuralConsistencyTest
from circ_rl.hypothesis.symbolic_regressor import (
    SymbolicRegressionConfig,
    SymbolicRegressor,
)


def _run_trial(
    name: str,
    sr_config: SymbolicRegressionConfig,
    x: np.ndarray,
    y: np.ndarray,
    variable_names: list[str],
    dataset: ExploratoryDataset,
    dim_idx: int,
    structural_test: StructuralConsistencyTest,
) -> bool:
    """Run one SR + falsification trial for delta_s2."""
    print(f"\n{'='*60}")
    print(f"TRIAL: {name}")
    print(f"  max_complexity={sr_config.max_complexity}, "
          f"n_iterations={sr_config.n_iterations}, "
          f"parsimony={sr_config.parsimony}")
    if sr_config.nested_constraints:
        print(f"  nested_constraints={sr_config.nested_constraints}")
    if sr_config.complexity_of_operators:
        print(f"  complexity_of_operators={sr_config.complexity_of_operators}")
    if sr_config.constraints:
        print(f"  constraints={sr_config.constraints}")
    print(f"  x.shape={x.shape}, variable_names={variable_names}")
    print(f"{'='*60}")

    regressor = SymbolicRegressor(sr_config)

    t0 = time.time()
    expressions = regressor.fit(x, y, variable_names)
    elapsed = time.time() - t0

    print(f"\n  SR complete: {len(expressions)} expressions in {elapsed:.1f}s")

    n_passed = 0
    for expr in expressions:
        try:
            func = expr.to_callable(variable_names)
            y_pred = func(x)
            ss_res = float(np.sum((y - y_pred) ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        except Exception:
            r2 = 0.0

        result = structural_test.test(
            expression=expr,
            dataset=dataset,
            target_dim_idx=dim_idx,
            variable_names=variable_names,
        )

        status = "PASS" if result.passed else "FAIL"
        if result.passed:
            n_passed += 1

        print(
            f"  [{status}] {expr.expression_str}"
            f"  (R2={r2:.4f}, F={result.f_statistic:.1f}, "
            f"p={result.p_value:.6f}, rel_impr={result.relative_improvement:.4f})"
        )

    print(f"\n  Result: {n_passed}/{len(expressions)} passed structural consistency")
    if n_passed > 0:
        print("  *** SUCCESS: Found validated delta_s2 expression! ***")
    return n_passed > 0


def main() -> None:
    """Run Trials 4-5 for delta_s2."""
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format=(
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
    )

    seed = 42
    n_envs = 12
    n_transitions = 3000

    print("=" * 60)
    print("DELTA_S2 TRIALS 4-5")
    print(f"  True form: dt*(-3g/(2l)*sin(theta) + 3/(m*l^2)*action)")
    print(f"  = -0.075*g*s1/l + 0.15*action/(m*l^2)")
    print("=" * 60)

    env_family = EnvironmentFamily.from_gymnasium(
        base_env="Pendulum-v1",
        param_distributions={
            "g": (6.0, 14.0),
            "m": (0.5, 2.0),
            "l": (0.5, 1.5),
        },
        n_envs=n_envs,
        seed=seed,
    )

    for i in range(n_envs):
        p = env_family.get_env_params(i)
        print(f"  Env {i:2d}: g={p['g']:.2f}, m={p['m']:.2f}, l={p['l']:.2f}")

    print(f"\nCollecting {n_envs * n_transitions} transitions...")
    collector = DataCollector(env_family, include_env_params=True)
    dataset = collector.collect(
        n_transitions_per_env=n_transitions,
        seed=seed,
    )

    state_names = ["s0", "s1", "s2"]
    env_param_names = ["g", "m", "l"]
    variable_names = state_names + ["action"] + env_param_names

    dim_idx = 2
    states = dataset.states
    actions = dataset.actions[:, np.newaxis] if dataset.actions.ndim == 1 else dataset.actions
    next_states = dataset.next_states

    y_full = next_states[:, dim_idx] - states[:, dim_idx]
    x_full = np.column_stack([states, actions, dataset.env_params])

    # Subsample
    rng = np.random.default_rng(seed)
    n_sub = min(8000, x_full.shape[0])
    idx = rng.choice(x_full.shape[0], n_sub, replace=False)
    x_sub, y_sub = x_full[idx], y_full[idx]
    print(f"  Subsampled: {n_sub}/{x_full.shape[0]} points")

    structural_test = StructuralConsistencyTest(p_threshold=0.01)

    # ================================================================
    # Trial 4: Constrain sin/cos to single-variable arguments
    # ================================================================
    _run_trial(
        name="Constrained sin/cos (max 1 node inside trig)",
        sr_config=SymbolicRegressionConfig(
            max_complexity=35,
            n_iterations=60,
            populations=20,
            binary_operators=("+", "-", "*", "/"),
            unary_operators=("sin", "cos", "square"),
            parsimony=0.001,
            timeout_seconds=300,
            deterministic=True,
            random_state=seed,
            constraints={"sin": 1, "cos": 1},
            nested_constraints={
                "/": {"+": 0, "-": 0},
                "sin": {"sin": 0, "cos": 0},
                "cos": {"sin": 0, "cos": 0},
            },
            complexity_of_operators={"square": 1},
        ),
        x=x_sub,
        y=y_sub,
        variable_names=variable_names,
        dataset=dataset,
        dim_idx=dim_idx,
        structural_test=structural_test,
    )

    # ================================================================
    # Trial 5: Feature engineering with g/l and 1/(m*l^2)
    # ================================================================
    ep = dataset.env_params
    ep_aug = np.column_stack([
        ep,
        ep[:, 0] / ep[:, 2],
        1.0 / (ep[:, 1] * ep[:, 2]**2),
    ])
    dataset_aug = ExploratoryDataset(
        states=dataset.states,
        actions=dataset.actions,
        next_states=dataset.next_states,
        rewards=dataset.rewards,
        env_ids=dataset.env_ids,
        env_params=ep_aug,
    )
    var_names_aug = variable_names + ["g_over_l", "inv_ml2"]

    x_aug = np.column_stack([
        x_sub,
        x_sub[:, 4] / x_sub[:, 6],
        1.0 / (x_sub[:, 5] * x_sub[:, 6]**2),
    ])

    _run_trial(
        name="Feature engineering (g/l, 1/(m*l^2) added)",
        sr_config=SymbolicRegressionConfig(
            max_complexity=20,
            n_iterations=40,
            populations=15,
            binary_operators=("+", "-", "*", "/"),
            unary_operators=("sin", "cos", "square"),
            parsimony=0.005,
            timeout_seconds=180,
            deterministic=True,
            random_state=seed,
        ),
        x=x_aug,
        y=y_sub,
        variable_names=var_names_aug,
        dataset=dataset_aug,
        dim_idx=dim_idx,
        structural_test=structural_test,
    )

    # ================================================================
    # Trial 6: No trig operators at all - pure algebraic
    # Forces g*s1/l form instead of sin(g*c)/l
    # ================================================================
    _run_trial(
        name="Pure algebraic (no sin/cos, just +,-,*,/,square)",
        sr_config=SymbolicRegressionConfig(
            max_complexity=30,
            n_iterations=60,
            populations=20,
            binary_operators=("+", "-", "*", "/"),
            unary_operators=("square",),
            parsimony=0.001,
            timeout_seconds=300,
            deterministic=True,
            random_state=seed,
            nested_constraints={"/": {"+": 0, "-": 0}},
            complexity_of_operators={"square": 1},
        ),
        x=x_sub,
        y=y_sub,
        variable_names=variable_names,
        dataset=dataset,
        dim_idx=dim_idx,
        structural_test=structural_test,
    )


if __name__ == "__main__":
    main()
