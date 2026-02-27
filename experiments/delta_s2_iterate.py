# ruff: noqa: T201
"""Focused iteration on delta_s2 symbolic regression.

Targets the angular velocity dynamics dimension which is the hardest for
PySR to discover because the true form has two terms with DIFFERENT
parametric dependencies:

    delta_s2 = dt * (-g * sin(theta) / l + action / (m * l^2))

The gravity term depends on g/l while the torque term depends on 1/(m*l^2).
PySR tends to factor these into a single parametric pathway (e.g., g/l),
making the expression fail the structural consistency falsification test.

This script iterates on PySR configurations to find a delta_s2 expression
that passes falsification.

Usage::

    uv run python experiments/delta_s2_iterate.py
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
    """Run one SR + falsification trial for delta_s2.

    :returns: True if at least one expression passed falsification.
    """
    print(f"\n{'='*60}")
    print(f"TRIAL: {name}")
    print(f"  max_complexity={sr_config.max_complexity}, "
          f"n_iterations={sr_config.n_iterations}, "
          f"parsimony={sr_config.parsimony}")
    if sr_config.nested_constraints:
        print(f"  nested_constraints={sr_config.nested_constraints}")
    if sr_config.complexity_of_operators:
        print(f"  complexity_of_operators={sr_config.complexity_of_operators}")
    print(f"  x.shape={x.shape}, variable_names={variable_names}")
    print(f"{'='*60}")

    regressor = SymbolicRegressor(sr_config)

    t0 = time.time()
    expressions = regressor.fit(x, y, variable_names)
    elapsed = time.time() - t0

    print(f"\n  SR complete: {len(expressions)} expressions in {elapsed:.1f}s")

    # Test each expression with structural consistency
    n_passed = 0
    for expr in expressions:
        # Compute R2
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
            f"p={result.p_value:.6f})"
        )

    print(f"\n  Result: {n_passed}/{len(expressions)} passed structural consistency")
    if n_passed > 0:
        print("  *** SUCCESS: Found validated delta_s2 expression! ***")
    return n_passed > 0


def main() -> None:
    """Run delta_s2 iteration experiments."""
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

    # Use MORE environments with WIDER parameter ranges to make
    # parametric differences more pronounced
    n_envs = 12
    n_transitions = 3000

    print("=" * 60)
    print("DELTA_S2 ITERATION: Discovering angular velocity dynamics")
    print(f"  True form: dt*(-g*sin(theta)/l + action/(m*l^2))")
    print(f"  Environments: {n_envs} with wide parameter ranges")
    print(f"  Transitions per env: {n_transitions}")
    print("=" * 60)

    # Create environment family with wider parameter ranges
    env_family = EnvironmentFamily.from_gymnasium(
        base_env="Pendulum-v1",
        param_distributions={
            "g": (6.0, 14.0),   # wider than (8, 12)
            "m": (0.5, 2.0),    # wider than (0.8, 1.5)
            "l": (0.5, 1.5),    # wider than (0.7, 1.3)
        },
        n_envs=n_envs,
        seed=seed,
    )

    for i in range(n_envs):
        p = env_family.get_env_params(i)
        print(f"  Env {i:2d}: g={p['g']:.2f}, m={p['m']:.2f}, l={p['l']:.2f}")

    # Collect data WITH environment parameters
    print(f"\nCollecting {n_envs * n_transitions} transitions...")
    collector = DataCollector(env_family, include_env_params=True)
    dataset = collector.collect(
        n_transitions_per_env=n_transitions,
        seed=seed,
    )

    state_names = ["s0", "s1", "s2"]
    action_names = ["action"]
    env_param_names = ["g", "m", "l"]
    variable_names = state_names + action_names + env_param_names

    # Build regression data for delta_s2 ONLY
    dim_idx = 2  # s2 = theta_dot
    states = dataset.states  # (N, 3)
    actions = dataset.actions[:, np.newaxis] if dataset.actions.ndim == 1 else dataset.actions
    next_states = dataset.next_states  # (N, 3)

    y = next_states[:, dim_idx] - states[:, dim_idx]  # (N,) delta_s2
    x = np.column_stack([states, actions, dataset.env_params])  # (N, 7)

    print(f"\n  SR input: x.shape={x.shape}, y.shape={y.shape}")
    print(f"  variable_names={variable_names}")
    print(f"  y range: [{y.min():.4f}, {y.max():.4f}], std={y.std():.4f}")

    # Subsample to 8000 points (PySR warns at >10k)
    rng = np.random.default_rng(seed)
    n_sub = min(8000, x.shape[0])
    idx = rng.choice(x.shape[0], n_sub, replace=False)
    x_sub, y_sub = x[idx], y[idx]
    print(f"  Subsampled: {n_sub}/{x.shape[0]} points")

    structural_test = StructuralConsistencyTest(p_threshold=0.01)

    common_kwargs = {
        "x": x_sub,
        "y": y_sub,
        "variable_names": variable_names,
        "dataset": dataset,
        "dim_idx": dim_idx,
        "structural_test": structural_test,
    }

    # ================================================================
    # Trial 1: Baseline with env params included
    # ================================================================
    _run_trial(
        name="Baseline (env params as features)",
        sr_config=SymbolicRegressionConfig(
            max_complexity=25,
            n_iterations=40,
            populations=15,
            binary_operators=("+", "-", "*", "/"),
            unary_operators=("sin", "cos", "square"),
            parsimony=0.003,
            timeout_seconds=180,
            deterministic=True,
            random_state=seed,
        ),
        **common_kwargs,
    )

    # ================================================================
    # Trial 2: Nested constraints - prevent + inside /
    # Forces PySR to discover SEPARATE additive terms instead of
    # factored forms like g/l * (action/m + c*s1)
    # ================================================================
    _run_trial(
        name="Nested constraints (no + inside /)",
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
            nested_constraints={"/": {"+": 0, "-": 0}},
            complexity_of_operators={"square": 1},
        ),
        **common_kwargs,
    )

    # ================================================================
    # Trial 3: Aggressive + nested constraints + cheap square
    # ================================================================
    _run_trial(
        name="Aggressive (high complexity + constrained)",
        sr_config=SymbolicRegressionConfig(
            max_complexity=40,
            n_iterations=80,
            populations=25,
            binary_operators=("+", "-", "*", "/"),
            unary_operators=("sin", "cos", "square"),
            parsimony=0.0005,
            timeout_seconds=600,
            deterministic=True,
            random_state=seed,
            nested_constraints={"/": {"+": 0, "-": 0}},
            complexity_of_operators={"square": 1, "sin": 2, "cos": 2},
        ),
        **common_kwargs,
    )

    # ================================================================
    # Trial 4: Constrain sin/cos to single-variable arguments
    # Prevents sin(g*c) but allows sin(s1), g*sin(s1)/l, etc.
    # Also constrain sin/cos nesting depth.
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
        **common_kwargs,
    )

    # ================================================================
    # Trial 5: Feature engineering - add physics-informed ratios
    # g/l and 1/(m*l^2) as additional features.
    # True form: -0.075*g_over_l*s1 + 0.15*inv_ml2*action
    # ================================================================
    # Augment the full dataset env_params with derived columns
    ep = dataset.env_params  # (N, 3) = [g, m, l]
    ep_aug = np.column_stack([
        ep,
        ep[:, 0] / ep[:, 2],                      # g/l
        1.0 / (ep[:, 1] * ep[:, 2]**2),           # 1/(m*l^2)
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

    # Subsample augmented data
    x_aug = np.column_stack([
        x_sub,
        x_sub[:, 4] / x_sub[:, 6],                # g/l
        1.0 / (x_sub[:, 5] * x_sub[:, 6]**2),    # 1/(m*l^2)
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


if __name__ == "__main__":
    main()
