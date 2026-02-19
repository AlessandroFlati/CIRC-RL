# ruff: noqa: T201
"""Final targeted delta_s2 trials with sum-of-products constraints.

Key insight: preventing + inside / isn't enough because PySR factors via *:
  (g + c) * (gravity_term + torque_term)
We also need to prevent + inside * to force sum-of-products form:
  c1*g*s1/l + c2*action/(m*l^2)

Usage::

    uv run python experiments/delta_s2_final.py
"""

from __future__ import annotations

import sys
import time

import numpy as np
from loguru import logger

from circ_rl.environments.data_collector import DataCollector, ExploratoryDataset
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
        if result.passed and r2 > 0.5:
            n_passed += 1

        print(
            f"  [{status}] {expr.expression_str}"
            f"  (R2={r2:.4f}, F={result.f_statistic:.1f}, "
            f"p={result.p_value:.6f}, rel_impr={result.relative_improvement:.4f})"
        )

    print(f"\n  Result: {n_passed}/{len(expressions)} non-trivial passed")
    if n_passed > 0:
        print("  *** SUCCESS ***")
    return n_passed > 0


def main() -> None:
    """Run final delta_s2 trials."""
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
    print("DELTA_S2 FINAL TRIALS")
    print(f"  True form: 0.075*g*s1/l + 0.15*action/(m*l^2)")
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

    collector = DataCollector(env_family, include_env_params=True)
    dataset = collector.collect(n_transitions_per_env=n_transitions, seed=seed)

    state_names = ["s0", "s1", "s2"]
    env_param_names = ["g", "m", "l"]
    variable_names = state_names + ["action"] + env_param_names

    dim_idx = 2
    states = dataset.states
    actions = dataset.actions[:, np.newaxis] if dataset.actions.ndim == 1 else dataset.actions
    next_states = dataset.next_states

    y_full = next_states[:, dim_idx] - states[:, dim_idx]
    x_full = np.column_stack([states, actions, dataset.env_params])

    rng = np.random.default_rng(seed)
    n_sub = min(8000, x_full.shape[0])
    idx = rng.choice(x_full.shape[0], n_sub, replace=False)
    x_sub, y_sub = x_full[idx], y_full[idx]
    print(f"\n  Subsampled: {n_sub}/{x_full.shape[0]} points")

    structural_test = StructuralConsistencyTest(p_threshold=0.01)

    # ================================================================
    # Trial A: Sum-of-products constraint (no + or - inside * or /)
    # Forces expressions like: A*B*C + D*E*F
    # Not: (A + B) * C or A * (B + C)
    # ================================================================
    _run_trial(
        name="Sum-of-products (no +/- inside * or /)",
        sr_config=SymbolicRegressionConfig(
            max_complexity=35,
            n_iterations=80,
            populations=25,
            binary_operators=("+", "-", "*", "/"),
            unary_operators=("sin", "cos", "square"),
            parsimony=0.0005,
            timeout_seconds=600,
            deterministic=True,
            random_state=seed,
            nested_constraints={
                "*": {"+": 0, "-": 0},
                "/": {"+": 0, "-": 0},
                "sin": {"sin": 0, "cos": 0},
                "cos": {"sin": 0, "cos": 0},
            },
            constraints={"sin": 1, "cos": 1},
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
    # Trial B: Same but pure algebraic (no trig)
    # ================================================================
    _run_trial(
        name="Sum-of-products, pure algebraic",
        sr_config=SymbolicRegressionConfig(
            max_complexity=30,
            n_iterations=80,
            populations=25,
            binary_operators=("+", "-", "*", "/"),
            unary_operators=("square",),
            parsimony=0.0005,
            timeout_seconds=600,
            deterministic=True,
            random_state=seed,
            nested_constraints={
                "*": {"+": 0, "-": 0},
                "/": {"+": 0, "-": 0},
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
    # Trial C: Feature engineering with sum-of-products
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
        name="Feature engineering + sum-of-products",
        sr_config=SymbolicRegressionConfig(
            max_complexity=15,
            n_iterations=40,
            populations=15,
            binary_operators=("+", "-", "*", "/"),
            unary_operators=("sin", "square"),
            parsimony=0.01,
            timeout_seconds=120,
            deterministic=True,
            random_state=seed,
            nested_constraints={
                "*": {"+": 0, "-": 0},
                "/": {"+": 0, "-": 0},
            },
            complexity_of_operators={"square": 1, "sin": 2},
        ),
        x=x_aug,
        y=y_sub,
        variable_names=var_names_aug,
        dataset=dataset_aug,
        dim_idx=dim_idx,
        structural_test=structural_test,
    )


if __name__ == "__main__":
    main()
