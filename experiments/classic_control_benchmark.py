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

    # Fast mode (fewer envs, quick SR)
    uv run python experiments/classic_control_benchmark.py --all --fast

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
            },
            test_param_distributions={
                "gravity": (6.0, 14.0),
                "masscart": (0.3, 3.0),
                "length": (0.2, 1.0),
            },
            continuous_action=False,
            max_steps=500,
            solver="mppi",
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
            solver="mppi",
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
    fast: bool = False,
) -> SymbolicRegressionConfig:
    """Create PySR configuration for the given environment.

    :param env_cfg: Environment configuration.
    :param seed: Random seed.
    :param fast: If True, use reduced iterations/populations.
    :returns: A SymbolicRegressionConfig.
    """
    n_iterations = 30 if fast else 80
    populations = 15 if fast else 25
    max_samples = 5000 if fast else 10000

    return SymbolicRegressionConfig(
        max_complexity=env_cfg.sr_max_complexity,
        n_iterations=n_iterations,
        populations=populations,
        binary_operators=("+", "-", "*", "/"),
        unary_operators=env_cfg.sr_unary_operators,
        parsimony=0.0005,
        timeout_seconds=300 if fast else 600,
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
    fast: bool = False,
) -> dict[str, Any]:
    """Run v2 pipeline stages 1-6 on training environments.

    :returns: Dict with all stage outputs needed for solver construction
        and evaluation.
    """
    sr_config = _make_sr_config(env_cfg, seed, fast)

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
        reward_fn, batched_reward_fn = _mountaincar_reward_fns(
            env_cfg.gym_id,
        )
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

    mppi_solvers: dict[int, MPPISolver] = {}
    for env_idx in range(test_family.n_envs):
        env_params = test_family.get_env_params(env_idx)

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
            config=mppi_config,
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
    print(f"    Solver: MPPI (horizon={mppi_config.horizon}, "
          f"samples={mppi_config.n_samples})")
    return policy


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

    :returns: EvalResult with aggregated statistics.
    """
    if max_steps is None:
        max_steps = env_cfg.max_steps

    returns: list[float] = []
    steps_list: list[float] = []
    goals: list[float] = []

    for env_idx in range(test_family.n_envs):
        env = test_family.make_env(env_idx)
        obs, _ = env.reset(seed=42)
        policy.reset(env_idx)

        total_reward = 0.0
        reached_goal = False

        for step in range(max_steps):
            action = policy.get_action(obs, env_idx)

            if env_cfg.continuous_action:
                # Clip to action bounds
                action = np.clip(action, env.action_space.low, env.action_space.high)
                obs, reward, terminated, truncated, _ = env.step(action)
            else:
                # Discrete: map continuous -> discrete
                discrete_action = _discretize_action(
                    action, env_cfg.gym_id,
                )
                obs, reward, terminated, truncated, _ = env.step(discrete_action)

            total_reward += float(reward)
            if terminated:
                if not env_cfg.goal_is_survival:
                    reached_goal = True
                steps_list.append(float(step + 1))
                break
            if truncated:
                if env_cfg.goal_is_survival:
                    reached_goal = True
                steps_list.append(float(step + 1))
                break
        else:
            steps_list.append(float(max_steps))
            if env_cfg.goal_is_survival:
                reached_goal = True

        returns.append(total_reward)
        goals.append(1.0 if reached_goal else 0.0)

        params = test_family.get_env_params(env_idx)
        params_str = "  ".join(f"{k}={v:.4f}" for k, v in params.items())
        goal_str = "GOAL" if reached_goal else "----"
        print(
            f"  [{env_idx:3d}] R={total_reward:8.1f} "
            f"steps={steps_list[-1]:5.0f} "
            f"{goal_str}  {params_str}"
        )
        env.close()

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

    returns: list[float] = []
    goals: list[float] = []
    steps_list: list[float] = []

    for env_idx in range(test_family.n_envs):
        env = test_family.make_env(env_idx)
        obs, _ = env.reset(seed=42)
        rng = np.random.default_rng(env_idx)

        total_reward = 0.0
        reached_goal = False

        for step in range(max_steps):
            if baseline == "random":
                action = env.action_space.sample()
            else:
                # Zero/no-op action
                if isinstance(env.action_space, gym.spaces.Box):
                    action = np.zeros(env.action_space.shape)
                else:
                    # Discrete: middle action (noop-like)
                    action = env.action_space.n // 2

            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            if terminated:
                if not env_cfg.goal_is_survival:
                    reached_goal = True
                steps_list.append(float(step + 1))
                break
            if truncated:
                if env_cfg.goal_is_survival:
                    reached_goal = True
                steps_list.append(float(step + 1))
                break
        else:
            steps_list.append(float(max_steps))
            if env_cfg.goal_is_survival:
                reached_goal = True

        returns.append(total_reward)
        goals.append(1.0 if reached_goal else 0.0)
        env.close()

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
    n_train: int = 15,
    n_test: int = 20,
    n_transitions: int = 5000,
    seed: int = 42,
    fast: bool = False,
) -> EvalResult | None:
    """Run the full CIRC-RL v2 benchmark for one environment.

    :param env_name: One of "pendulum", "cartpole", "mountaincar",
        "mountaincar_continuous".
    :param n_train: Number of training environments.
    :param n_test: Number of OOD test environments.
    :param n_transitions: Transitions per training environment.
    :param seed: Random seed.
    :param fast: If True, use reduced SR config.
    :returns: EvalResult, or None if pipeline failed.
    """
    configs = _make_env_configs()
    env_cfg = configs[env_name]

    print(f"\n{'=' * 70}")
    print(f"BENCHMARK: {env_cfg.gym_id}")
    print(f"  {n_train} train envs, {n_test} OOD test envs")
    print(f"  Params: {list(env_cfg.param_distributions.keys())}")
    print(f"  Solver: {env_cfg.solver.upper()}")
    print(f"  Fast: {fast}")
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
    )

    # Run pipeline
    print("\n  Running v2 pipeline...")
    t_pipeline = time.time()
    pipeline = _run_pipeline(
        env_cfg, train_family, n_transitions, seed, fast,
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
    else:
        policy = _build_mppi_solver(env_cfg, pipeline, test_family)
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
        "--n-train", type=int, default=15,
        help="Number of training environments (default: 15).",
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
    parser.add_argument(
        "--fast", action="store_true",
        help="Fast mode: 8 train, 10 test envs, reduced SR.",
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

    # Fast mode overrides
    if args.fast:
        args.n_train = 8
        args.n_test = 10
        args.n_transitions = 3000

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
            fast=args.fast,
        )
        if result is not None:
            results.append(result)

    total_elapsed = time.time() - t0

    if results:
        _print_summary_table(results)
        print(f"\nTotal elapsed: {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()
