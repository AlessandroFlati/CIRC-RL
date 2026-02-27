# ruff: noqa: T201
"""Unified CIRC-RL v2 benchmark on classic control environments.

Runs the full v2 pipeline (causal discovery -> hypothesis falsification ->
analytic policy derivation) on each classic control environment and reports
performance on OOD test environments.

Supported environments:
    - Pendulum-v1 (continuous action, iLQR)
    - CartPole-v1 (discrete action, iLQR + discretization)
    - MountainCarContinuous-v0 (continuous action, MPPI)
    - MountainCar-v0 (discrete action, MPPI + discretization)

Usage::

    # Run single environment
    uv run python experiments/classic_control_benchmark.py --env pendulum

    # Run all environments
    uv run python experiments/classic_control_benchmark.py --all

    # Fast mode (moderate SR, same envs/data)
    uv run python experiments/classic_control_benchmark.py --all --fast

    # Quick mode (fewer envs, minimal SR -- for rapid iteration)
    uv run python experiments/classic_control_benchmark.py --all --quick

Requires PySR: ``uv sync --extra symbolic``
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import gymnasium as gym
import numpy as np
from loguru import logger

from circ_rl.environments.env_family import EnvironmentFamily
from circ_rl.hypothesis.symbolic_regressor import SymbolicRegressionConfig
from circ_rl.orchestration.stages import (
    AnalyticPolicyDerivationStage,
    CausalDiscoveryStage,
    FeatureSelectionStage,
    HypothesisFalsificationStage,
    HypothesisGenerationStage,
    ObservationAnalysisStage,
    TransitionAnalysisStage,
)

if TYPE_CHECKING:
    from circ_rl.hypothesis.derived_features import DerivedFeatureSpec
    from circ_rl.hypothesis.hypothesis_register import HypothesisEntry
    from circ_rl.orchestration.stages import _ILQRAnalyticPolicy


# ---------------------------------------------------------------------------
# Environment configs
# ---------------------------------------------------------------------------

ENV_NAMES = ["pendulum", "cartpole", "mountaincar", "mountaincar_continuous"]


@dataclass
class EnvConfig:
    """Per-environment benchmark configuration.

    :param name: Short name for the environment.
    :param gym_id: Gymnasium environment ID.
    :param param_distributions: Parameter ranges for environment family.
    :param test_param_distributions: Wider ranges for OOD test family.
    :param continuous_action: Whether the environment accepts continuous actions.
    :param max_steps: Maximum steps per evaluation episode.
    :param solver: Solver type to use ("ilqr" or "mppi").
    :param sr_unary_operators: PySR unary operators for this environment.
    :param sr_max_complexity: PySR max expression complexity.
    :param reward_derived_features: Derived features for reward SR.
    :param goal_is_survival: If True, "goal" = surviving max_steps (CartPole).
        If False, "goal" = terminated=True (MountainCar reaching summit).
    """

    name: str
    gym_id: str
    param_distributions: dict[str, tuple[float, float]]
    test_param_distributions: dict[str, tuple[float, float]]
    continuous_action: bool
    max_steps: int
    solver: str
    sr_unary_operators: tuple[str, ...]
    sr_max_complexity: int
    reward_derived_features: list[DerivedFeatureSpec]
    goal_is_survival: bool = False


def _make_env_configs() -> dict[str, EnvConfig]:
    """Build per-environment configurations."""
    from circ_rl.hypothesis.derived_features import DerivedFeatureSpec

    return {
        "pendulum": EnvConfig(
            name="pendulum",
            gym_id="Pendulum-v1",
            param_distributions={
                "g": (8.0, 12.0),
                "m": (0.8, 1.5),
                "l": (0.7, 1.3),
            },
            test_param_distributions={
                "g": (6.0, 14.0),
                "m": (0.5, 2.0),
                "l": (0.5, 1.5),
            },
            continuous_action=True,
            max_steps=200,
            solver="ilqr",
            sr_unary_operators=("square",),
            sr_max_complexity=30,
            reward_derived_features=[
                DerivedFeatureSpec(
                    name="theta",
                    source_names=("s1", "s0"),
                    compute_fn=np.arctan2,
                ),
            ],
        ),
        "cartpole": EnvConfig(
            name="cartpole",
            gym_id="CartPole-v1",
            param_distributions={
                "gravity": (7.0, 13.0),
                "masscart": (0.5, 2.0),
                "length": (0.3, 0.8),
                "masspole": (0.05, 0.2),
            },
            test_param_distributions={
                "gravity": (6.0, 14.0),
                "masscart": (0.3, 3.0),
                "length": (0.2, 1.0),
                "masspole": (0.03, 0.3),
            },
            continuous_action=False,
            max_steps=500,
            solver="cem",
            sr_unary_operators=("sin", "cos", "square"),
            sr_max_complexity=30,
            reward_derived_features=[],
            goal_is_survival=True,
        ),
        "mountaincar": EnvConfig(
            name="mountaincar",
            gym_id="MountainCar-v0",
            param_distributions={
                "gravity": (0.0015, 0.0040),
                "force": (0.0005, 0.0020),
            },
            test_param_distributions={
                "gravity": (0.0010, 0.0050),
                "force": (0.0003, 0.0025),
            },
            continuous_action=False,
            max_steps=200,
            solver="cem",
            sr_unary_operators=("cos", "square"),
            sr_max_complexity=25,
            reward_derived_features=[],
        ),
        "mountaincar_continuous": EnvConfig(
            name="mountaincar_continuous",
            gym_id="MountainCarContinuous-v0",
            param_distributions={
                "power": (0.0008, 0.0030),
            },
            test_param_distributions={
                "power": (0.0005, 0.0035),
            },
            continuous_action=True,
            max_steps=999,
            solver="mppi",
            sr_unary_operators=("cos", "square"),
            sr_max_complexity=25,
            reward_derived_features=[],
        ),
    }


# ---------------------------------------------------------------------------
# Pipeline runner (stages 1-6)
# ---------------------------------------------------------------------------


def _make_sr_config(
    env_cfg: EnvConfig,
    seed: int,
    speed: str = "normal",
) -> SymbolicRegressionConfig:
    """Create PySR configuration for the given environment.

    :param env_cfg: Environment configuration.
    :param seed: Random seed.
    :param speed: One of "normal", "fast", "quick".
        - normal: full SR budget (80 iters, 25 pops, 10k samples, 600s).
        - fast: reduced SR budget (60 iters, 20 pops, 8k samples, 400s).
        - quick: minimal SR budget (50 iters, 15 pops, 5k samples, 300s).
    :returns: A SymbolicRegressionConfig.
    """
    sr_params = {
        "normal": (80, 25, 10000, 600),
        "fast": (60, 20, 8000, 400),
        "quick": (50, 15, 5000, 300),
    }
    n_iterations, populations, max_samples, timeout = sr_params[speed]

    return SymbolicRegressionConfig(
        max_complexity=env_cfg.sr_max_complexity,
        n_iterations=n_iterations,
        populations=populations,
        binary_operators=("+", "-", "*", "/"),
        unary_operators=env_cfg.sr_unary_operators,
        parsimony=0.0005,
        timeout_seconds=timeout,
        deterministic=True,
        random_state=seed,
        nested_constraints={
            "*": {"+": 0, "-": 0},
            "/": {"+": 0, "-": 0},
        },
        complexity_of_operators={
            op: (1 if op == "square" else 2)
            for op in env_cfg.sr_unary_operators
        },
        max_samples=max_samples,
    )


def _run_pipeline(
    env_cfg: EnvConfig,
    train_family: EnvironmentFamily,
    n_transitions: int,
    seed: int,
    speed: str = "normal",
) -> dict[str, Any]:
    """Run v2 pipeline stages 1-6 on training environments.

    :returns: Dict with all stage outputs needed for solver construction
        and evaluation.
    """
    sr_config = _make_sr_config(env_cfg, seed, speed)

    # Stage 1: Causal discovery
    print("  [1/6] Causal discovery...")
    cd_stage = CausalDiscoveryStage(
        env_family=train_family,
        n_transitions_per_env=n_transitions,
        discovery_method="pc",
        alpha=0.05,
        seed=seed,
        include_env_params=True,
        ep_correlation_threshold=0.05,
    )
    cd_output = cd_stage.run({})
    state_names = cd_output["state_names"]
    print(f"    State names: {state_names}")

    # Stage 2: Feature selection
    print("  [2/6] Feature selection...")
    fs_stage = FeatureSelectionStage(
        epsilon=0.15,
        min_ate=0.01,
        use_mechanism_invariance=True,
    )
    fs_output = fs_stage.run({"causal_discovery": cd_output})

    # Stage 3: Transition analysis
    print("  [3/6] Transition analysis...")
    ta_stage = TransitionAnalysisStage()
    ta_output = ta_stage.run({
        "causal_discovery": cd_output,
        "feature_selection": fs_output,
    })
    ta_result = ta_output["transition_result"]
    print(f"    Variant dims: {ta_result.variant_dims}")

    # Stage 4: Observation analysis (detect canonical coords)
    print("  [4/6] Observation analysis...")
    oa_stage = ObservationAnalysisStage()
    oa_output = oa_stage.run({"causal_discovery": cd_output})
    analysis_result = oa_output.get("analysis_result")
    if analysis_result is not None:
        print(f"    Canonical coords: {oa_output['canonical_state_names']}")
    else:
        print("    No canonical mappings (dynamics in obs space)")

    # Stage 5: Hypothesis generation (PySR)
    print("  [5/6] Symbolic regression (PySR)...")
    print(
        f"    Config: complexity<={sr_config.max_complexity}, "
        f"iters={sr_config.n_iterations}, "
        f"timeout={sr_config.timeout_seconds}s"
    )
    hg_stage = HypothesisGenerationStage(
        include_env_params=True,
        sr_config=sr_config,
        reward_sr_config=sr_config,
        reward_derived_features=env_cfg.reward_derived_features,
    )
    hg_output = hg_stage.run({
        "causal_discovery": cd_output,
        "feature_selection": fs_output,
        "transition_analysis": ta_output,
        "observation_analysis": oa_output,
    })
    register = hg_output["register"]
    print(f"    Discovered {len(register.entries)} hypotheses")

    # Stage 6: Hypothesis falsification
    print("  [6/6] Hypothesis falsification...")
    hf_stage = HypothesisFalsificationStage(
        structural_p_threshold=0.01,
        structural_min_relative_improvement=0.01,
        ood_confidence=0.99,
        held_out_fraction=0.2,
    )
    hf_output = hf_stage.run({
        "hypothesis_generation": hg_output,
        "causal_discovery": cd_output,
        "observation_analysis": oa_output,
    })

    hf_result = hf_output["falsification_result"]
    best_dynamics = hf_output["best_dynamics"]
    best_reward = hf_output["best_reward"]
    print(
        f"    Tested: {hf_result.n_tested}, "
        f"Validated: {hf_result.n_validated}, "
        f"Falsified: {hf_result.n_falsified}"
    )

    if best_dynamics:
        for target, entry in best_dynamics.items():
            print(
                f"    [{target}] {entry.expression.expression_str}"
                f"  (R2={entry.training_r2:.4f})"
            )
    else:
        print("    WARNING: No validated dynamics hypotheses!")

    if best_reward:
        print(
            f"    [reward] {best_reward.expression.expression_str}"
            f"  (R2={best_reward.training_r2:.4f})"
        )

    return {
        "cd_output": cd_output,
        "fs_output": fs_output,
        "ta_output": ta_output,
        "oa_output": oa_output,
        "hg_output": hg_output,
        "hf_output": hf_output,
        "best_dynamics": best_dynamics,
        "best_reward": best_reward,
        "state_names": state_names,
    }


# ---------------------------------------------------------------------------
# Solver builders
# ---------------------------------------------------------------------------


def _build_ilqr_solver(
    env_cfg: EnvConfig,
    pipeline: dict[str, Any],
    test_family: EnvironmentFamily,
) -> _ILQRAnalyticPolicy:
    """Build iLQR policy from pipeline output.

    Uses the AnalyticPolicyDerivationStage for consistent construction.

    :returns: An ``_ILQRAnalyticPolicy`` with per-env solvers.
    """
    apd_stage = AnalyticPolicyDerivationStage(
        env_family=test_family,
        gamma=0.99,
        adaptive_replan_multiplier=3.0,
        min_replan_interval=3,
    )
    apd_output = apd_stage.run({
        "hypothesis_falsification": pipeline["hf_output"],
        "transition_analysis": pipeline["ta_output"],
        "causal_discovery": pipeline["cd_output"],
        "observation_analysis": pipeline["oa_output"],
    })
    print(f"    Solver: {apd_output['solver_type']}")
    print(f"    Explained variance: {apd_output['explained_variance']:.4f}")
    return apd_output["analytic_policy"]


def _build_mppi_solver(
    env_cfg: EnvConfig,
    pipeline: dict[str, Any],
    test_family: EnvironmentFamily,
) -> _ILQRAnalyticPolicy:
    """Build MPPI policy from pipeline output.

    Uses validated symbolic dynamics for forward simulation in MPPI
    sampling. Reward function uses position-based shaping appropriate
    for MountainCar.

    :returns: An ``_ILQRAnalyticPolicy`` wrapping MPPI solvers.
    """
    from circ_rl.analytic_policy.fast_dynamics import build_batched_dynamics_fn
    from circ_rl.analytic_policy.mppi_solver import MPPIConfig, MPPISolver
    from circ_rl.hypothesis.expression import SymbolicExpression
    from circ_rl.orchestration.stages import (
        _build_dynamics_fn,
        _ILQRAnalyticPolicy,
    )

    hf_output = pipeline["hf_output"]
    cd_output = pipeline["cd_output"]
    oa_output = pipeline["oa_output"]
    best_dynamics = pipeline["best_dynamics"]
    state_names = pipeline["state_names"]

    # Canonical space
    obs_to_canonical_fn = None
    angular_dims: tuple[int, ...] = ()
    analysis_result = oa_output.get("analysis_result")
    if analysis_result is not None:
        obs_to_canonical_fn = analysis_result.obs_to_canonical_fn
        angular_dims = analysis_result.angular_dims

    # Parse dynamics expressions by dim index
    state_dim = len(state_names)
    dynamics_expressions: dict[int, SymbolicExpression] = {}
    for target, entry in best_dynamics.items():
        dim_name = target.removeprefix("delta_")
        dim_idx = state_names.index(dim_name)
        dynamics_expressions[dim_idx] = entry.expression

    dynamics_state_names = hf_output.get("dynamics_state_names", state_names)
    action_names = hf_output.get("action_names", ["action"])

    # Calibration coefficients (pooled)
    pooled_cal: dict[int, tuple[float, float]] = {}
    for target, entry in best_dynamics.items():
        dim_name = target.removeprefix("delta_")
        dim_idx = state_names.index(dim_name)
        if entry.pooled_calibration is not None:
            pooled_cal[dim_idx] = entry.pooled_calibration
    cal_arg = pooled_cal or None

    # Adaptive replanning threshold
    dynamics_mse_sum = sum(e.training_mse for e in best_dynamics.values())
    adaptive_tau: float | None = None
    if dynamics_mse_sum > 0:
        adaptive_tau = 3.0 * np.sqrt(dynamics_mse_sum)

    # Choose reward functions based on environment
    if env_cfg.name in ("mountaincar", "mountaincar_continuous"):
        # Try energy-based reward from discovered dynamics
        terrain = _extract_terrain_from_dynamics(best_dynamics)
        goal_pos = 0.45 if "Continuous" in env_cfg.gym_id else 0.5
        if terrain is not None:
            c_pos, grav_coeff = terrain
            reward_fn, batched_reward_fn = _dynamics_energy_reward_fns(
                c_pos, grav_coeff, goal_pos,
            )
            print("    Reward: energy-based (from discovered dynamics)")
        else:
            reward_fn, batched_reward_fn = _mountaincar_reward_fns(
                env_cfg.gym_id,
            )
            print("    Reward: height-based (fallback)")
    elif env_cfg.name == "cartpole":
        reward_fn, batched_reward_fn = _cartpole_reward_fns()
    else:
        raise ValueError(f"No MPPI reward function for {env_cfg.name}")

    # Determine action bounds
    sample_env = gym.make(env_cfg.gym_id)
    action_space = sample_env.action_space
    sample_env.close()

    if isinstance(action_space, gym.spaces.Box):
        max_action = float(action_space.high[0])
        action_dim = action_space.shape[0]
    else:
        # Discrete action -> continuous proxy
        max_action = 1.0
        action_dim = 1

    # Per-environment MPPI tuning
    if env_cfg.name == "cartpole":
        mppi_config = MPPIConfig(
            horizon=30,
            n_samples=256,
            temperature=0.3,
            noise_sigma=0.5,
            n_iterations=3,
            gamma=0.99,
            max_action=max_action,
            colored_noise_beta=0.5,
            replan_interval=1,
            adaptive_replan_threshold=adaptive_tau,
            min_replan_interval=1,
        )
    else:
        mppi_config = MPPIConfig(
            horizon=50,
            n_samples=256,
            temperature=0.5,
            noise_sigma=0.3,
            n_iterations=3,
            gamma=0.99,
            max_action=max_action,
            colored_noise_beta=1.0,
            replan_interval=3,
            adaptive_replan_threshold=adaptive_tau,
            min_replan_interval=2,
        )

    # Base horizon for adaptive scaling
    base_horizon = mppi_config.horizon

    mppi_solvers: dict[int, MPPISolver] = {}
    for env_idx in range(test_family.n_envs):
        env_params = test_family.get_env_params(env_idx)

        # Per-env adaptive horizon
        horizon = _compute_adaptive_horizon(env_params, base_horizon)
        if horizon != base_horizon:
            env_mppi_config = MPPIConfig(
                horizon=horizon,
                n_samples=mppi_config.n_samples,
                temperature=mppi_config.temperature,
                noise_sigma=mppi_config.noise_sigma,
                n_iterations=mppi_config.n_iterations,
                gamma=mppi_config.gamma,
                max_action=mppi_config.max_action,
                colored_noise_beta=mppi_config.colored_noise_beta,
                replan_interval=mppi_config.replan_interval,
                adaptive_replan_threshold=mppi_config.adaptive_replan_threshold,
                min_replan_interval=mppi_config.min_replan_interval,
            )
        else:
            env_mppi_config = mppi_config

        dynamics_fn = _build_dynamics_fn(
            dynamics_expressions,
            dynamics_state_names,
            action_names,
            state_dim,
            env_params,
            obs_low=None,
            obs_high=None,
            angular_dims=angular_dims,
            calibration_coefficients=cal_arg,
        )

        batched_dyn_fn = build_batched_dynamics_fn(
            dynamics_expressions,
            dynamics_state_names,
            action_names,
            state_dim,
            env_params,
            angular_dims=angular_dims,
            calibration_coefficients=cal_arg,
        )

        mppi_solvers[env_idx] = MPPISolver(
            config=env_mppi_config,
            dynamics_fn=dynamics_fn,
            reward_fn=reward_fn,
            batched_dynamics_fn=batched_dyn_fn,
            batched_reward_fn=batched_reward_fn,
        )

    first_entry = next(iter(best_dynamics.values()))

    policy = _ILQRAnalyticPolicy(
        dynamics_hypothesis=first_entry,
        reward_hypothesis=None,
        state_dim=state_dim,
        action_dim=action_dim,
        n_envs=test_family.n_envs,
        ilqr_solvers=mppi_solvers,  # type: ignore[arg-type]
        action_low=-max_action * np.ones(action_dim),
        action_high=max_action * np.ones(action_dim),
        obs_to_canonical_fn=obs_to_canonical_fn,
    )
    horizons = [mppi_solvers[i].config.horizon for i in range(test_family.n_envs)]
    print(f"    Solver: MPPI (horizon={min(horizons)}-{max(horizons)}, "
          f"samples={mppi_config.n_samples})")
    return policy


def _compute_adaptive_horizon(
    env_params: dict[str, float],
    base_horizon: int,
    min_horizon: int = 50,
    max_horizon: int = 200,
) -> int:
    """Scale planning horizon based on environment force/gravity ratio.

    When gravity is large relative to actuator force, the agent needs
    more timesteps to build momentum. This scales the horizon accordingly
    using only parameters from the environment family (not hardcoded
    domain knowledge).

    :param env_params: Per-env parameter dict from EnvironmentFamily.
    :param base_horizon: Default planning horizon.
    :param min_horizon: Minimum horizon.
    :param max_horizon: Maximum horizon.
    :returns: Scaled horizon clamped to [min_horizon, max_horizon].
    """
    force_keys = {"force", "power", "force_mag"}
    gravity_keys = {"gravity"}

    forces = [v for k, v in env_params.items() if k in force_keys and v > 0]
    gravities = [v for k, v in env_params.items() if k in gravity_keys and v > 0]

    if not forces or not gravities:
        return base_horizon

    # Ratio > 1 means gravity dominates force -> need longer horizon
    ratio = max(gravities) / min(forces)
    scaled = int(base_horizon * np.sqrt(ratio))
    return min(max_horizon, max(min_horizon, scaled))


def _build_cem_solver(
    env_cfg: EnvConfig,
    pipeline: dict[str, Any],
    test_family: EnvironmentFamily,
) -> _ILQRAnalyticPolicy:
    """Build CEM policy from pipeline output for discrete action envs.

    Uses validated symbolic dynamics for forward simulation in CEM
    sampling. Actions are sampled directly from categorical distributions
    over the discrete action set, avoiding the continuous-to-discrete
    mismatch inherent in MPPI-based planning.

    :returns: An ``_ILQRAnalyticPolicy`` wrapping CEM solvers.
    """
    from circ_rl.analytic_policy.cem_solver import CEMConfig, CEMSolver
    from circ_rl.analytic_policy.fast_dynamics import build_batched_dynamics_fn
    from circ_rl.hypothesis.expression import SymbolicExpression
    from circ_rl.orchestration.stages import (
        _build_dynamics_fn,
        _ILQRAnalyticPolicy,
    )

    hf_output = pipeline["hf_output"]
    cd_output = pipeline["cd_output"]
    oa_output = pipeline["oa_output"]
    best_dynamics = pipeline["best_dynamics"]
    state_names = pipeline["state_names"]

    # Canonical space
    obs_to_canonical_fn = None
    angular_dims: tuple[int, ...] = ()
    analysis_result = oa_output.get("analysis_result")
    if analysis_result is not None:
        obs_to_canonical_fn = analysis_result.obs_to_canonical_fn
        angular_dims = analysis_result.angular_dims

    # Parse dynamics expressions by dim index
    state_dim = len(state_names)
    dynamics_expressions: dict[int, SymbolicExpression] = {}
    for target, entry in best_dynamics.items():
        dim_name = target.removeprefix("delta_")
        dim_idx = state_names.index(dim_name)
        dynamics_expressions[dim_idx] = entry.expression

    dynamics_state_names = hf_output.get("dynamics_state_names", state_names)
    action_names = hf_output.get("action_names", ["action"])

    # Calibration coefficients (pooled)
    pooled_cal: dict[int, tuple[float, float]] = {}
    for target, entry in best_dynamics.items():
        dim_name = target.removeprefix("delta_")
        dim_idx = state_names.index(dim_name)
        if entry.pooled_calibration is not None:
            pooled_cal[dim_idx] = entry.pooled_calibration
    cal_arg = pooled_cal or None

    # Determine action mapping for this environment
    sample_env = gym.make(env_cfg.gym_id)
    action_space = sample_env.action_space
    sample_env.close()

    if isinstance(action_space, gym.spaces.Discrete):
        n_actions = int(action_space.n)
        # action_values: what the dynamics model was trained on (integer indices)
        action_values = [float(i) for i in range(n_actions)]
        # discretization_values: what _discretize_action maps from [-1, 1]
        if n_actions == 2:
            # CartPole: action 0 -> -1.0, action 1 -> +1.0
            discretization_values = [-1.0, 1.0]
        elif n_actions == 3:
            # MountainCar: action 0 -> -1.0, action 1 -> 0.0, action 2 -> +1.0
            discretization_values = [-1.0, 0.0, 1.0]
        else:
            # General case: evenly spaced from -1 to 1
            discretization_values = list(
                np.linspace(-1.0, 1.0, n_actions),
            )
    else:
        raise ValueError(
            f"CEM solver requires discrete action space, "
            f"got {type(action_space).__name__}",
        )

    # Choose reward functions based on environment
    if env_cfg.name in ("mountaincar",):
        # Try energy-based reward from discovered dynamics
        terrain = _extract_terrain_from_dynamics(best_dynamics)
        goal_pos = 0.5
        if terrain is not None:
            c_pos, grav_coeff = terrain
            reward_fn, batched_reward_fn = _dynamics_energy_reward_fns(
                c_pos, grav_coeff, goal_pos,
            )
            print("    Reward: energy-based (from discovered dynamics)")
        else:
            reward_fn, batched_reward_fn = _mountaincar_reward_fns(
                env_cfg.gym_id,
            )
            print("    Reward: height-based (fallback)")
    elif env_cfg.name == "cartpole":
        reward_fn, batched_reward_fn = _cartpole_reward_fns()
    else:
        raise ValueError(f"No CEM reward function for {env_cfg.name}")

    # Base CEM horizon
    base_horizon = 50 if env_cfg.name == "mountaincar" else 30

    cem_solvers: dict[int, CEMSolver] = {}
    for env_idx in range(test_family.n_envs):
        env_params = test_family.get_env_params(env_idx)

        # Per-env adaptive horizon
        horizon = _compute_adaptive_horizon(
            env_params, base_horizon,
        )

        cem_config = CEMConfig(
            horizon=horizon,
            n_samples=256,
            n_iterations=5,
            n_actions=n_actions,
            elite_fraction=0.2,
            gamma=0.99,
            smoothing_alpha=0.8,
        )

        dynamics_fn = _build_dynamics_fn(
            dynamics_expressions,
            dynamics_state_names,
            action_names,
            state_dim,
            env_params,
            obs_low=None,
            obs_high=None,
            angular_dims=angular_dims,
            calibration_coefficients=cal_arg,
        )

        batched_dyn_fn = build_batched_dynamics_fn(
            dynamics_expressions,
            dynamics_state_names,
            action_names,
            state_dim,
            env_params,
            angular_dims=angular_dims,
            calibration_coefficients=cal_arg,
        )

        cem_solvers[env_idx] = CEMSolver(
            config=cem_config,
            dynamics_fn=dynamics_fn,
            reward_fn=reward_fn,
            action_values=action_values,
            discretization_values=discretization_values,
            batched_dynamics_fn=batched_dyn_fn,
            batched_reward_fn=batched_reward_fn,
        )

    first_entry = next(iter(best_dynamics.values()))

    policy = _ILQRAnalyticPolicy(
        dynamics_hypothesis=first_entry,
        reward_hypothesis=None,
        state_dim=state_dim,
        action_dim=1,
        n_envs=test_family.n_envs,
        ilqr_solvers=cem_solvers,  # type: ignore[arg-type]
        action_low=-1.0 * np.ones(1),
        action_high=1.0 * np.ones(1),
        obs_to_canonical_fn=obs_to_canonical_fn,
    )
    horizons = [cem_solvers[i].config.horizon for i in range(test_family.n_envs)]
    print(f"    Solver: CEM (horizon={min(horizons)}-{max(horizons)}, "
          f"samples=256, actions={n_actions})")
    return policy


def _extract_terrain_from_dynamics(
    best_dynamics: dict[str, HypothesisEntry],
) -> tuple[float, float] | None:
    """Extract cos(c*x) terrain coefficients from discovered velocity dynamics.

    Walks the sympy expression tree of the velocity dynamics (delta_s1),
    looking for ``cos(c * s0)`` terms. Extracts ``c`` (frequency) and the
    multiplying coefficient.

    :param best_dynamics: Validated dynamics hypotheses keyed by target name.
    :returns: ``(c_position, gravity_coeff)`` or ``None`` if no terrain term found.
    """
    import sympy

    velocity_entry = best_dynamics.get("delta_s1")
    if velocity_entry is None:
        return None

    expr = velocity_entry.expression.sympy_expr
    s0 = sympy.Symbol("s0")

    # Expand the expression and look for cos(c*s0) terms
    expanded = sympy.expand(expr)

    # Walk through additive terms
    for term in sympy.Add.make_args(expanded):
        # Check if term contains cos
        cos_atoms = [a for a in term.atoms(sympy.cos)]
        for cos_term in cos_atoms:
            arg = cos_term.args[0]
            # Check if the argument is c*s0
            if arg.has(s0):
                # Extract the coefficient of s0 inside cos
                coeff_s0 = arg.coeff(s0)
                if coeff_s0 != 0 and coeff_s0.is_number:
                    c_position = float(abs(coeff_s0))
                    # Extract the coefficient multiplying cos(c*s0)
                    # This may contain symbols (ep_gravity etc.), so check
                    coeff_expr = term.coeff(cos_term)
                    if coeff_expr.is_number and coeff_expr != 0:
                        gravity_coeff = float(coeff_expr)
                    else:
                        # Coefficient contains symbols; use sign heuristic
                        # In MountainCar, gravity opposes motion (negative)
                        gravity_coeff = -1.0
                    logger.info(
                        "Extracted terrain: cos({:.2f}*x) with coeff={:.6f}",
                        c_position, gravity_coeff,
                    )
                    return (c_position, gravity_coeff)

    return None


def _dynamics_energy_reward_fns(
    c_position: float,
    gravity_coeff: float,
    goal_pos: float,
) -> tuple[Any, Any]:
    """Build energy-based reward from discovered dynamics terrain coefficients.

    Potential energy: ``PE = -gravity_coeff * sin(c*x) / c``
    Kinetic energy: ``KE = 0.5 * v^2``
    Total energy serves as a progress measure for hill-climbing.

    :param c_position: Position frequency from ``cos(c*x)`` in dynamics.
    :param gravity_coeff: Coefficient multiplying the terrain term.
    :param goal_pos: Goal x-position.
    :returns: ``(scalar_reward_fn, batched_reward_fn)``.
    """
    # Potential energy: integral of gravity_coeff * cos(c*x) dx
    # = gravity_coeff * sin(c*x) / c
    # We want higher position -> higher reward, so sign matters.
    # In MountainCar, the gravity term is negative (opposing motion),
    # so PE = -gravity_coeff * sin(c*x) / c gives increasing PE with height.
    pe_scale = -gravity_coeff / c_position

    def scalar_reward_fn(
        state: np.ndarray, action: np.ndarray,
    ) -> float:
        x, v = state[0], state[1]
        pe = pe_scale * np.sin(c_position * x)
        ke = 0.5 * v**2
        energy = pe + ke
        goal_bonus = 10.0 if x >= goal_pos else 0.0
        velocity_bonus = 0.1 * abs(v)
        return float(energy + goal_bonus + velocity_bonus)

    def batched_reward_fn(
        states: np.ndarray, actions: np.ndarray,
    ) -> np.ndarray:
        x = states[:, 0]  # (K,)
        v = states[:, 1]  # (K,)
        pe = pe_scale * np.sin(c_position * x)  # (K,)
        ke = 0.5 * v**2  # (K,)
        energy = pe + ke  # (K,)
        goal_bonus = np.where(x >= goal_pos, 10.0, 0.0)  # (K,)
        velocity_bonus = 0.1 * np.abs(v)  # (K,)
        return energy + goal_bonus + velocity_bonus  # (K,)

    return scalar_reward_fn, batched_reward_fn


def _mountaincar_reward_fns(
    gym_id: str,
) -> tuple[
    Any,  # scalar reward fn
    Any,  # batched reward fn
]:
    """Create MountainCar reward functions for MPPI.

    Uses height-based + velocity shaping to guide the car uphill.
    The reward encourages reaching the goal position (x=0.5 for
    MountainCar-v0, x=0.45 for MountainCarContinuous-v0).

    :returns: ``(scalar_reward_fn, batched_reward_fn)``.
    """
    goal_pos = 0.45 if "Continuous" in gym_id else 0.5

    def scalar_reward_fn(
        state: np.ndarray, action: np.ndarray,
    ) -> float:
        x, v = state[0], state[1]
        # Height = sin(3*x) gives potential energy proxy
        height = np.sin(3.0 * x)
        # Bonus for reaching goal
        goal_bonus = 10.0 if x >= goal_pos else 0.0
        # Velocity in right direction is good
        velocity_bonus = 0.1 * abs(v)
        return float(height + goal_bonus + velocity_bonus - 0.01 * action[0] ** 2)

    def batched_reward_fn(
        states: np.ndarray, actions: np.ndarray,
    ) -> np.ndarray:
        x = states[:, 0]  # (K,)
        v = states[:, 1]  # (K,)
        u = actions[:, 0]  # (K,)
        height = np.sin(3.0 * x)  # (K,)
        goal_bonus = np.where(x >= goal_pos, 10.0, 0.0)  # (K,)
        velocity_bonus = 0.1 * np.abs(v)  # (K,)
        return height + goal_bonus + velocity_bonus - 0.01 * u**2  # (K,)

    return scalar_reward_fn, batched_reward_fn


def _cartpole_reward_fns() -> tuple[Any, Any]:
    """Create CartPole reward functions for MPPI.

    Uses angle-based shaping to keep the pole upright and the cart centered.
    CartPole terminates when ``|theta| > 0.2095 rad`` or ``|x| > 2.4``.
    The reward encourages staying alive (small angle, centered cart).

    :returns: ``(scalar_reward_fn, batched_reward_fn)``.
    """
    # CartPole state: [x, x_dot, theta, theta_dot]
    # Termination: |theta| > 12 deg (0.2095 rad), |x| > 2.4
    theta_limit = 0.2095
    x_limit = 2.4

    def scalar_reward_fn(
        state: np.ndarray, action: np.ndarray,
    ) -> float:
        x, x_dot, theta, theta_dot = state[0], state[1], state[2], state[3]
        # Heavy penalty near terminal boundaries
        if abs(theta) > theta_limit or abs(x) > x_limit:
            return -10.0
        # Reward for staying upright and centered
        angle_cost = (theta / theta_limit) ** 2
        position_cost = (x / x_limit) ** 2
        return float(1.0 - 0.5 * angle_cost - 0.3 * position_cost)

    def batched_reward_fn(
        states: np.ndarray, actions: np.ndarray,
    ) -> np.ndarray:
        x = states[:, 0]  # (K,)
        theta = states[:, 2]  # (K,)
        terminal = (np.abs(theta) > theta_limit) | (np.abs(x) > x_limit)
        angle_cost = (theta / theta_limit) ** 2  # (K,)
        position_cost = (x / x_limit) ** 2  # (K,)
        reward = 1.0 - 0.5 * angle_cost - 0.3 * position_cost  # (K,)
        return np.where(terminal, -10.0, reward)  # (K,)

    return scalar_reward_fn, batched_reward_fn


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@dataclass
class EvalResult:
    """Evaluation result for one environment config.

    :param env_name: Environment short name.
    :param pipeline_time: Time for pipeline stages 1-6.
    :param solver_build_time: Time to build the solver.
    :param eval_time: Time for evaluation rollouts.
    :param mean_return: Mean return across test environments.
    :param median_return: Median return.
    :param worst_return: Worst (min) return.
    :param std_return: Standard deviation.
    :param goal_rate: Fraction of episodes reaching goal state.
    :param mean_steps: Mean steps per episode.
    :param n_test_envs: Number of test environments evaluated.
    :param n_dynamics_validated: Number of dynamics dims with validated hypotheses.
    :param n_dynamics_total: Total dynamics dimensions.
    """

    env_name: str
    pipeline_time: float
    solver_build_time: float
    eval_time: float
    mean_return: float
    median_return: float
    worst_return: float
    std_return: float
    goal_rate: float
    mean_steps: float
    n_test_envs: int
    n_dynamics_validated: int
    n_dynamics_total: int


def _evaluate_policy(
    env_cfg: EnvConfig,
    policy: _ILQRAnalyticPolicy,
    test_family: EnvironmentFamily,
    max_steps: int | None = None,
) -> EvalResult:
    """Evaluate policy on test environments.

    For discrete-action environments, maps continuous policy output
    to discrete actions via threshold discretization.

    Environments are evaluated serially because the iLQR/MPPI
    solvers already use ThreadPoolExecutor for restart parallelism,
    saturating available cores. Adding outer env-level threads
    causes GIL contention and slows execution.

    :returns: EvalResult with aggregated statistics.
    """
    if max_steps is None:
        max_steps = env_cfg.max_steps

    _max_steps = max_steps  # capture for closure

    def _eval_single_env(
        env_idx: int,
    ) -> tuple[int, float, bool, float, dict[str, float]]:
        env = test_family.make_env(env_idx)
        obs, _ = env.reset(seed=42)
        policy.reset(env_idx)

        total_reward = 0.0
        reached_goal = False
        n_steps = float(_max_steps)

        for step in range(_max_steps):
            action = policy.get_action(obs, env_idx)

            if env_cfg.continuous_action:
                action = np.clip(action, env.action_space.low, env.action_space.high)
                obs, reward, terminated, truncated, _ = env.step(action)
            else:
                discrete_action = _discretize_action(
                    action, env_cfg.gym_id,
                )
                obs, reward, terminated, truncated, _ = env.step(discrete_action)

            total_reward += float(reward)
            if terminated:
                if not env_cfg.goal_is_survival:
                    reached_goal = True
                n_steps = float(step + 1)
                break
            if truncated:
                if env_cfg.goal_is_survival:
                    reached_goal = True
                n_steps = float(step + 1)
                break
        else:
            if env_cfg.goal_is_survival:
                reached_goal = True

        env.close()
        params = test_family.get_env_params(env_idx)
        return env_idx, total_reward, reached_goal, n_steps, params

    n_envs = test_family.n_envs
    raw_results = [_eval_single_env(i) for i in range(n_envs)]

    returns: list[float] = []
    steps_list: list[float] = []
    goals: list[float] = []

    for env_idx, total_reward, reached_goal, n_steps, params in raw_results:
        returns.append(total_reward)
        steps_list.append(n_steps)
        goals.append(1.0 if reached_goal else 0.0)
        params_str = "  ".join(f"{k}={v:.4f}" for k, v in params.items())
        goal_str = "GOAL" if reached_goal else "----"
        print(
            f"  [{env_idx:3d}] R={total_reward:8.1f} "
            f"steps={n_steps:5.0f} "
            f"{goal_str}  {params_str}"
        )

    returns_arr = np.array(returns)
    steps_arr = np.array(steps_list)

    return EvalResult(
        env_name=env_cfg.name,
        pipeline_time=0.0,  # filled by caller
        solver_build_time=0.0,
        eval_time=0.0,
        mean_return=float(returns_arr.mean()),
        median_return=float(np.median(returns_arr)),
        worst_return=float(returns_arr.min()),
        std_return=float(returns_arr.std()),
        goal_rate=float(np.mean(goals)),
        mean_steps=float(steps_arr.mean()),
        n_test_envs=test_family.n_envs,
        n_dynamics_validated=0,
        n_dynamics_total=0,
    )


def _discretize_action(
    continuous_action: np.ndarray,
    gym_id: str,
) -> int:
    """Map continuous policy output to discrete action.

    CartPole-v1: 2 actions (0=left, 1=right)
    MountainCar-v0: 3 actions (0=left, 1=noop, 2=right)

    :param continuous_action: Shape ``(1,)`` or scalar, in ``[-1, 1]``.
    :param gym_id: Gymnasium environment ID.
    :returns: Discrete action index.
    """
    u = float(continuous_action[0]) if continuous_action.ndim > 0 else float(
        continuous_action
    )

    if gym_id == "CartPole-v1":
        return 1 if u > 0.0 else 0
    elif gym_id == "MountainCar-v0":
        if u < -0.33:
            return 0
        elif u > 0.33:
            return 2
        else:
            return 1
    else:
        raise ValueError(f"No discretization for {gym_id}")


def _evaluate_baseline(
    env_cfg: EnvConfig,
    test_family: EnvironmentFamily,
    baseline: str = "random",
    max_steps: int | None = None,
) -> tuple[float, float, float]:
    """Evaluate a baseline policy on test environments.

    :param baseline: "random" or "zero" (no-op action).
    :returns: ``(mean_return, goal_rate, mean_steps)``.
    """
    if max_steps is None:
        max_steps = env_cfg.max_steps

    _max_steps = max_steps
    _baseline = baseline

    def _eval_baseline_env(env_idx: int) -> tuple[float, float, float]:
        env = test_family.make_env(env_idx)
        obs, _ = env.reset(seed=42)

        total_reward = 0.0
        reached_goal = False
        n_steps = float(_max_steps)

        for step in range(_max_steps):
            if _baseline == "random":
                action = env.action_space.sample()
            else:
                if isinstance(env.action_space, gym.spaces.Box):
                    action = np.zeros(env.action_space.shape)
                else:
                    action = env.action_space.n // 2

            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            if terminated:
                if not env_cfg.goal_is_survival:
                    reached_goal = True
                n_steps = float(step + 1)
                break
            if truncated:
                if env_cfg.goal_is_survival:
                    reached_goal = True
                n_steps = float(step + 1)
                break
        else:
            if env_cfg.goal_is_survival:
                reached_goal = True

        env.close()
        return total_reward, 1.0 if reached_goal else 0.0, n_steps

    n_envs = test_family.n_envs
    results = [_eval_baseline_env(i) for i in range(n_envs)]

    returns = [r[0] for r in results]
    goals = [r[1] for r in results]
    steps_list = [r[2] for r in results]

    return (
        float(np.mean(returns)),
        float(np.mean(goals)),
        float(np.mean(steps_list)),
    )


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------


def run_benchmark(
    env_name: str,
    n_train: int = 25,
    n_test: int = 20,
    n_transitions: int = 5000,
    seed: int = 42,
    speed: str = "normal",
) -> EvalResult | None:
    """Run the full CIRC-RL v2 benchmark for one environment.

    :param env_name: One of "pendulum", "cartpole", "mountaincar",
        "mountaincar_continuous".
    :param n_train: Number of training environments.
    :param n_test: Number of OOD test environments.
    :param n_transitions: Transitions per training environment.
    :param seed: Random seed.
    :param speed: One of "normal", "fast", "quick" for SR budget.
    :returns: EvalResult, or None if pipeline failed.
    """
    configs = _make_env_configs()
    env_cfg = configs[env_name]

    print(f"\n{'=' * 70}")
    print(f"BENCHMARK: {env_cfg.gym_id}")
    print(f"  {n_train} train envs, {n_test} OOD test envs")
    print(f"  Params: {list(env_cfg.param_distributions.keys())}")
    print(f"  Solver: {env_cfg.solver.upper()}")
    print(f"  Speed: {speed}")
    print(f"{'=' * 70}")

    # Create environment families
    print("\n  Creating environment families...")
    train_family = EnvironmentFamily.from_gymnasium(
        base_env=env_cfg.gym_id,
        param_distributions=env_cfg.param_distributions,
        n_envs=n_train,
        seed=seed,
    )
    test_family = EnvironmentFamily.from_gymnasium(
        base_env=env_cfg.gym_id,
        param_distributions=env_cfg.test_param_distributions,
        n_envs=n_test,
        seed=seed + 1000,
        max_episode_steps=env_cfg.max_steps,
    )

    # Run pipeline
    print("\n  Running v2 pipeline...")
    t_pipeline = time.time()
    pipeline = _run_pipeline(
        env_cfg, train_family, n_transitions, seed, speed,
    )
    pipeline_time = time.time() - t_pipeline
    print(f"  Pipeline time: {pipeline_time:.1f}s")

    best_dynamics = pipeline["best_dynamics"]
    state_names = pipeline["state_names"]

    if not best_dynamics:
        print("\n  FAILED: No validated dynamics hypotheses.")
        print("  PySR may need different config or more data.")
        return None

    n_dynamics_total = len(state_names)
    n_dynamics_validated = len(best_dynamics)

    # Build solver
    print(f"\n  Building {env_cfg.solver.upper()} policy...")
    t_build = time.time()
    if env_cfg.solver == "ilqr":
        policy = _build_ilqr_solver(env_cfg, pipeline, test_family)
    elif env_cfg.solver == "cem":
        policy = _build_cem_solver(env_cfg, pipeline, test_family)
    elif env_cfg.solver == "mppi":
        policy = _build_mppi_solver(env_cfg, pipeline, test_family)
    else:
        raise ValueError(f"Unknown solver: {env_cfg.solver}")
    build_time = time.time() - t_build
    print(f"  Build time: {build_time:.1f}s")

    # Evaluate
    print(f"\n  Evaluating on {n_test} OOD test environments...")
    t_eval = time.time()
    result = _evaluate_policy(env_cfg, policy, test_family)
    eval_time = time.time() - t_eval

    # Fill timing fields
    result.pipeline_time = pipeline_time
    result.solver_build_time = build_time
    result.eval_time = eval_time
    result.n_dynamics_validated = n_dynamics_validated
    result.n_dynamics_total = n_dynamics_total

    # Baselines
    print("\n  Evaluating baselines...")
    random_ret, random_goal, random_steps = _evaluate_baseline(
        env_cfg, test_family, "random",
    )

    # Print summary
    total_time = pipeline_time + build_time + eval_time
    print(f"\n  --- {env_cfg.gym_id} Results ---")
    print(f"  Dynamics: {n_dynamics_validated}/{n_dynamics_total} validated")
    print(
        f"  CIRC-RL:  mean={result.mean_return:8.1f}  "
        f"goal={result.goal_rate * 100:5.1f}%  "
        f"steps={result.mean_steps:5.0f}"
    )
    print(
        f"  Random:   mean={random_ret:8.1f}  "
        f"goal={random_goal * 100:5.1f}%  "
        f"steps={random_steps:5.0f}"
    )
    print(f"  Total time: {total_time:.1f}s")

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="CIRC-RL v2 classic control benchmark",
    )
    parser.add_argument(
        "--env",
        choices=ENV_NAMES,
        help="Environment to benchmark (default: all).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all environments.",
    )
    parser.add_argument(
        "--n-train", type=int, default=25,
        help="Number of training environments (default: 25).",
    )
    parser.add_argument(
        "--n-test", type=int, default=20,
        help="Number of OOD test environments (default: 20).",
    )
    parser.add_argument(
        "--n-transitions", type=int, default=5000,
        help="Transitions per training environment (default: 5000).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42).",
    )
    speed_group = parser.add_mutually_exclusive_group()
    speed_group.add_argument(
        "--fast", action="store_true",
        help="Fast mode: moderate SR (60 iters, 20 pops, 8k samples, 400s).",
    )
    speed_group.add_argument(
        "--quick", action="store_true",
        help="Quick mode: 8 train, 10 test envs, heavily reduced SR.",
    )
    return parser.parse_args()


def _print_summary_table(results: list[EvalResult]) -> None:
    """Print a formatted summary table of all benchmark results."""
    print(f"\n{'=' * 90}")
    print("CIRC-RL v2 CLASSIC CONTROL BENCHMARK SUMMARY")
    print(f"{'=' * 90}")

    header = (
        f"{'Environment':<30} "
        f"{'Mean R':>8} "
        f"{'Goal%':>6} "
        f"{'Steps':>6} "
        f"{'Dyn':>5} "
        f"{'Pipeline':>9} "
        f"{'Total':>8}"
    )
    print(header)
    print("-" * 90)

    for r in results:
        total = r.pipeline_time + r.solver_build_time + r.eval_time
        print(
            f"{r.env_name:<30} "
            f"{r.mean_return:8.1f} "
            f"{r.goal_rate * 100:5.1f}% "
            f"{r.mean_steps:6.0f} "
            f"{r.n_dynamics_validated}/{r.n_dynamics_total:<3} "
            f"{r.pipeline_time:7.1f}s "
            f"{total:6.1f}s"
        )

    print(f"{'=' * 90}")


def main() -> None:
    """Run classic control benchmarks."""
    args = _parse_args()

    # Determine which environments to run
    if args.all:
        envs_to_run = ENV_NAMES
    elif args.env:
        envs_to_run = [args.env]
    else:
        print("Specify --env <name> or --all. Available envs:")
        for name in ENV_NAMES:
            print(f"  {name}")
        return

    # Speed mode overrides
    if args.quick:
        speed = "quick"
        args.n_train = 8
        args.n_test = 10
        args.n_transitions = 3000
    elif args.fast:
        speed = "fast"
    else:
        speed = "normal"

    logger.remove()
    logger.add(sys.stderr, level="INFO")
    os.environ["SDL_VIDEODRIVER"] = "dummy"

    t0 = time.time()
    results: list[EvalResult] = []

    for env_name in envs_to_run:
        result = run_benchmark(
            env_name=env_name,
            n_train=args.n_train,
            n_test=args.n_test,
            n_transitions=args.n_transitions,
            seed=args.seed,
            speed=speed,
        )
        if result is not None:
            results.append(result)

    total_elapsed = time.time() - t0

    if results:
        _print_summary_table(results)
        print(f"\nTotal elapsed: {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()
