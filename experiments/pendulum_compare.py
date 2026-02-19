# ruff: noqa: T201
"""Head-to-head comparison: v1 (neural) vs v2 (analytic) on 50 diverse envs.

Evaluates generalization of both approaches to out-of-distribution
physics parameters (g, m, l) on 50 test environments:

- **v1 (neural)**: CausalPolicy with context conditioning, trained via PPO.
- **v2 (analytic)**: Symbolic dynamics + iLQR, derived from PySR expressions
  discovered on 25 training environments.

Produces two 10x5 grid videos and prints comparison statistics.

Usage::

    uv run python experiments/pendulum_compare.py

Requires PySR: ``pip install 'circ-rl[symbolic]'``
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import gymnasium as gym
import numpy as np
import torch
from loguru import logger

from circ_rl.analytic_policy.ilqr_solver import (
    ILQRConfig,
    ILQRSolver,
    make_quadratic_terminal_cost,
)
from circ_rl.environments.data_collector import DataCollector
from circ_rl.environments.env_family import EnvironmentFamily
from circ_rl.feature_selection.transition_analyzer import TransitionAnalyzer
from circ_rl.hypothesis.derived_features import DerivedFeatureSpec
from circ_rl.hypothesis.symbolic_regressor import SymbolicRegressionConfig
from circ_rl.orchestration.stages import (
    CausalDiscoveryStage,
    FeatureSelectionStage,
    HypothesisFalsificationStage,
    HypothesisGenerationStage,
    ObservationAnalysisStage,
    TransitionAnalysisStage,
)
from circ_rl.policy.causal_policy import CausalPolicy

if TYPE_CHECKING:
    from collections.abc import Callable

    from circ_rl.hypothesis.expression import SymbolicExpression
    from circ_rl.hypothesis.hypothesis_register import HypothesisEntry
    from circ_rl.orchestration.stages import _ILQRAnalyticPolicy

# -- Video recording helpers (adapted from pendulum_record_50.py) --


def _get_bitmap_font() -> dict[str, list[int]]:
    """Return a minimal 5x7 bitmap font for common characters."""
    return {
        "0": [0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E],
        "1": [0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E],
        "2": [0x0E, 0x11, 0x01, 0x06, 0x08, 0x10, 0x1F],
        "3": [0x0E, 0x11, 0x01, 0x06, 0x01, 0x11, 0x0E],
        "4": [0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02],
        "5": [0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E],
        "6": [0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E],
        "7": [0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08],
        "8": [0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E],
        "9": [0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C],
        ".": [0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C],
        "-": [0x00, 0x00, 0x00, 0x1F, 0x00, 0x00, 0x00],
        "=": [0x00, 0x00, 0x1F, 0x00, 0x1F, 0x00, 0x00],
        " ": [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
        ",": [0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x08],
        "g": [0x00, 0x00, 0x0F, 0x11, 0x0F, 0x01, 0x0E],
        "m": [0x00, 0x00, 0x1A, 0x15, 0x15, 0x11, 0x11],
        "l": [0x0C, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E],
        "R": [0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11],
        "G": [0x0E, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0F],
        "M": [0x11, 0x1B, 0x15, 0x15, 0x11, 0x11, 0x11],
        "L": [0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F],
        ":": [0x00, 0x0C, 0x0C, 0x00, 0x0C, 0x0C, 0x00],
        "/": [0x01, 0x02, 0x02, 0x04, 0x08, 0x08, 0x10],
        "v": [0x00, 0x00, 0x11, 0x11, 0x11, 0x0A, 0x04],
    }


def _add_text_to_frame(frame: np.ndarray, text: str) -> None:
    """Burn text label into top-left corner of an RGB frame."""
    bar_height = 16
    frame[:bar_height, :] = (frame[:bar_height, :].astype(np.float32) * 0.3).astype(
        np.uint8
    )

    font = _get_bitmap_font()
    x_offset = 4
    y_offset = 2
    scale = 1

    for ch in text:
        glyph = font.get(ch, font.get("?", []))
        if not glyph:
            x_offset += 6 * scale
            continue
        for row_idx, row_bits in enumerate(glyph):
            for col_idx in range(5):
                if row_bits & (1 << (4 - col_idx)):
                    y = y_offset + row_idx * scale
                    x = x_offset + col_idx * scale
                    if y < frame.shape[0] and x < frame.shape[1]:
                        frame[y : y + scale, x : x + scale] = 255
        x_offset += 6 * scale


def _make_grid_video(
    all_frames: list[list[np.ndarray]],
    labels: list[str],
    output_path: str,
    fps: int = 20,
    n_cols: int = 10,
) -> None:
    """Combine multiple episode frame lists into a grid video."""
    import imageio.v3 as iio

    max_len = max(len(f) for f in all_frames)
    for frames in all_frames:
        while len(frames) < max_len:
            frames.append(frames[-1].copy())

    n = len(all_frames)
    n_rows = (n + n_cols - 1) // n_cols

    grid_frames = []
    for t in range(max_len):
        row_images = []
        for r in range(n_rows):
            col_images = []
            for c in range(n_cols):
                idx = r * n_cols + c
                if idx < n:
                    frame = all_frames[idx][t].copy()
                    _add_text_to_frame(frame, labels[idx])
                    col_images.append(frame)
                else:
                    col_images.append(np.zeros_like(all_frames[0][0]))
            row_images.append(np.hstack(col_images))
        grid_frames.append(np.vstack(row_images))

    iio.imwrite(
        output_path,
        np.stack(grid_frames),
        fps=fps,
        codec="libx264",
    )


def _record_episode_generic(
    env_family: EnvironmentFamily,
    env_idx: int,
    get_action_fn: Callable[[np.ndarray, int], np.ndarray],
    max_steps: int = 400,
    fixed_params: dict[str, float] | None = None,
) -> tuple[list[np.ndarray], float]:
    """Record one episode with a generic action function.

    :param get_action_fn: Callable ``(obs_np, env_idx) -> action_np``.
    """
    params = env_family.get_env_params(env_idx)
    env = gym.make(
        "Pendulum-v1",
        render_mode="rgb_array",
        max_episode_steps=max_steps,
    )
    unwrapped = env.unwrapped
    if fixed_params:
        for attr, value in fixed_params.items():
            setattr(unwrapped, attr, value)
    for attr, value in params.items():
        setattr(unwrapped, attr, value)

    obs, _ = env.reset(seed=42)
    frames: list[np.ndarray] = []
    total_reward = 0.0

    for _ in range(max_steps):
        frame = env.render()
        assert isinstance(frame, np.ndarray)
        frame = frame[::2, ::2, :].copy()  # 2x downsample
        frames.append(frame)

        action_np = get_action_fn(obs, env_idx)

        obs, reward, terminated, _truncated, _ = env.step(action_np)
        total_reward += float(reward)

        if terminated:
            break

    env.close()
    return frames, total_reward


def _quick_eval_env(
    env_family: EnvironmentFamily,
    env_idx: int,
    get_action_fn: Callable[[np.ndarray, int], np.ndarray],
    max_steps: int = 100,
    fixed_params: dict[str, float] | None = None,
) -> float:
    """Quickly evaluate one episode without video recording.

    Creates a gym env without render mode, runs the episode, and
    returns only the total reward. Suitable for fast screening before
    full evaluation with video recording.

    :param max_steps: Number of steps for quick evaluation (default 100).
    :returns: Total reward for the episode.
    """
    params = env_family.get_env_params(env_idx)
    env = gym.make(
        "Pendulum-v1",
        max_episode_steps=max_steps,
    )
    unwrapped = env.unwrapped
    if fixed_params:
        for attr, value in fixed_params.items():
            setattr(unwrapped, attr, value)
    for attr, value in params.items():
        setattr(unwrapped, attr, value)

    obs, _ = env.reset(seed=42)
    total_reward = 0.0

    for _ in range(max_steps):
        action_np = get_action_fn(obs, env_idx)
        obs, reward, terminated, _truncated, _ = env.step(action_np)
        total_reward += float(reward)
        if terminated:
            break

    env.close()
    return total_reward


def _quick_evaluate_parallel(
    env_family: EnvironmentFamily,
    env_indices: list[int],
    get_action_fn: Callable[[np.ndarray, int], np.ndarray],
    reset_fn: Callable[[int], None] | None,
    max_steps: int = 100,
    fixed_params: dict[str, float] | None = None,
    max_workers: int | None = None,
) -> dict[int, float]:
    """Evaluate multiple environments in parallel without video.

    Uses ThreadPoolExecutor for concurrent evaluation. Each thread
    creates its own gym environment instance.

    :param env_indices: Which environment indices to evaluate.
    :param get_action_fn: ``(obs, env_idx) -> action``.
    :param reset_fn: Optional callable to reset policy state for an env
        (e.g., ``v2_policy.reset(env_idx)``).
    :param max_workers: Thread pool size. Defaults to ``min(len(env_indices), cpu_count)``.
    :returns: Dict mapping ``env_idx -> total_reward``.
    """
    if max_workers is None:
        max_workers = min(len(env_indices), os.cpu_count() or 4)

    def _worker(env_idx: int) -> tuple[int, float]:
        if reset_fn is not None:
            reset_fn(env_idx)
        ret = _quick_eval_env(
            env_family, env_idx, get_action_fn,
            max_steps=max_steps, fixed_params=fixed_params,
        )
        return env_idx, ret

    results: dict[int, float] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for env_idx, reward in pool.map(_worker, env_indices):
            results[env_idx] = reward

    return results


# -- v2 pipeline runner --


def _run_v2_pipeline(
    seed: int = 42,
) -> tuple[
    dict[str, HypothesisEntry],
    HypothesisEntry | None,
    list[str],
    list[str],
    list[DerivedFeatureSpec],
    dict[str, Any],
    dict[str, Any],
    list[str],
    dict[str, Any],
]:
    """Run v2 pipeline stages 1-7 on 25 training envs.

    :returns: (best_dynamics, best_reward, dynamics_state_names,
        action_names, reward_derived_features, hf_output, oa_output).
    """
    n_envs = 25
    n_transitions = 5000

    sr_config = SymbolicRegressionConfig(
        max_complexity=40,
        n_iterations=120,
        populations=30,
        binary_operators=("+", "-", "*", "/"),
        unary_operators=("sin", "cos", "square", "sqrt", "abs"),
        parsimony=0.0002,
        timeout_seconds=600,
        deterministic=True,
        random_state=seed,
        nested_constraints={
            "*": {"+": 0, "-": 0},
            "/": {"+": 0, "-": 0},
            "sin": {"sin": 0, "cos": 0},
            "cos": {"sin": 0, "cos": 0},
        },
        complexity_of_operators={"square": 1, "sin": 2, "cos": 2},
        constraints={"sin": 5, "cos": 5},
        max_samples=25000,
        n_sr_runs=3,
    )

    import sympy as sp

    reward_derived_specs = [
        DerivedFeatureSpec(
            name="theta",
            source_names=("s1", "s0"),
            compute_fn=np.arctan2,
            sympy_fn=sp.atan2,
        ),
    ]

    # Stage 1: Environment family
    print("\n  [v2 1/7] Creating training environment family...")
    train_family = EnvironmentFamily.from_gymnasium(
        base_env="Pendulum-v1",
        param_distributions={
            "g": (8.0, 12.0),
            "m": (0.8, 1.5),
            "l": (0.7, 1.3),
        },
        n_envs=n_envs,
        seed=seed,
    )

    # Stage 2: Causal discovery
    print("  [v2 2/7] Running causal discovery...")
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

    # Stage 3: Feature selection
    print("  [v2 3/7] Running feature selection...")
    fs_stage = FeatureSelectionStage(
        epsilon=0.15,
        min_ate=0.01,
        use_mechanism_invariance=True,
    )
    fs_output = fs_stage.run({"causal_discovery": cd_output})

    # Stage 4: Transition analysis
    print("  [v2 4/7] Running transition analysis...")
    ta_stage = TransitionAnalysisStage()
    ta_output = ta_stage.run(
        {
            "causal_discovery": cd_output,
            "feature_selection": fs_output,
        }
    )

    # Stage 5: Observation analysis (constraint detection + canonical coords)
    print("  [v2 5/7] Running observation analysis...")
    oa_stage = ObservationAnalysisStage()
    oa_output = oa_stage.run({"causal_discovery": cd_output})
    analysis_result = oa_output.get("analysis_result")
    if analysis_result is not None:
        print(f"    Canonical coords: {oa_output['canonical_state_names']}")
        for c in analysis_result.constraints:
            print(f"    Constraint: {c.constraint_type} on dims {c.involved_dims}")
    else:
        print("    No canonical mappings found (dynamics in obs space)")

    # Stage 6: Hypothesis generation (PySR)
    print("  [v2 6/7] Running symbolic regression (PySR)...")
    hg_stage = HypothesisGenerationStage(
        include_env_params=True,
        sr_config=sr_config,
        reward_sr_config=sr_config,
        reward_derived_features=reward_derived_specs,
    )
    hg_output = hg_stage.run(
        {
            "causal_discovery": cd_output,
            "feature_selection": fs_output,
            "transition_analysis": ta_output,
            "observation_analysis": oa_output,
        }
    )

    register = hg_output["register"]
    print(f"    Discovered {len(register.entries)} hypotheses")

    # Stage 7: Hypothesis falsification
    print("  [v2 7/7] Running hypothesis falsification...")
    hf_stage = HypothesisFalsificationStage(
        structural_p_threshold=0.01,
        structural_min_relative_improvement=0.01,
        ood_confidence=0.99,
        held_out_fraction=0.2,
    )
    hf_output = hf_stage.run(
        {
            "hypothesis_generation": hg_output,
            "causal_discovery": cd_output,
            "observation_analysis": oa_output,
        }
    )

    best_dynamics = hf_output["best_dynamics"]
    best_reward = hf_output["best_reward"]
    reward_derived_features = hf_output.get("reward_derived_features", [])

    hf_result = hf_output["falsification_result"]
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
    if best_reward:
        print(
            f"    [reward] {best_reward.expression.expression_str}"
            f"  (R2={best_reward.training_r2:.4f})"
        )

    action_names = hg_output.get("action_names", ["action"])
    dynamics_state_names = hf_output.get("dynamics_state_names", state_names)

    return (
        best_dynamics,
        best_reward,
        dynamics_state_names,
        action_names,
        reward_derived_features,
        hf_output,
        oa_output,
        state_names,  # observation-space state names (for reward eval)
        cd_output,  # needed for spurious detection dataset
    )


def _build_v2_ilqr_policy(
    test_family: EnvironmentFamily,
    best_dynamics: dict[str, HypothesisEntry],
    best_reward: HypothesisEntry | None,
    state_names: list[str],
    action_names: list[str],
    reward_derived_features: list[DerivedFeatureSpec],
    oa_output: dict[str, Any] | None = None,
    gamma: float = 0.99,
    max_action: float = 2.0,
    obs_state_names: list[str] | None = None,
    cd_output: dict[str, Any] | None = None,
    adaptive_replan_multiplier: float = 3.0,
    min_replan_interval: int = 3,
) -> _ILQRAnalyticPolicy:
    """Build a stateful iLQR policy for test environments.

    Substitutes each test env's (g, m, l) into the validated symbolic
    expressions to produce parametrically adapted iLQR controllers.

    Includes Section 7.2 improvements:
    - Spurious term detection and pruning (via ``cd_output`` dataset)
    - Coefficient calibration (pooled alpha/beta from structural consistency)
    - Adaptive replanning (MSE-based prediction-error threshold)

    When canonical coordinates are available (from observation analysis),
    iLQR plans in canonical space and obs_to_canonical_fn converts gym
    observations before planning.

    :param oa_output: Observation analysis stage output (optional).
    :param obs_state_names: Observation-space state names, used for
        reward evaluation when canonical coordinates are active.
    :param cd_output: Causal discovery output (for spurious detection).
    :param adaptive_replan_multiplier: Multiplier for adaptive replan
        threshold (tau = multiplier * sqrt(sum_mse)).
    :param min_replan_interval: Minimum steps between adaptive replans.
    :returns: An _ILQRAnalyticPolicy with get_action/reset interface.
    """
    from circ_rl.orchestration.stages import (
        _build_dynamics_fn,
        _build_dynamics_jacobian_fns,
        _build_reward_derivative_fns,
        _build_reward_fn,
        _default_reward,
        _ILQRAnalyticPolicy,
    )

    # Check for canonical space from observation analysis
    analysis_result = None
    obs_to_canonical_fn = None
    canonical_to_obs_fn = None
    angular_dims: tuple[int, ...] = ()
    if oa_output is not None:
        analysis_result = oa_output.get("analysis_result")
        if analysis_result is not None:
            obs_to_canonical_fn = analysis_result.obs_to_canonical_fn
            canonical_to_obs_fn = analysis_result.canonical_to_obs_fn
            angular_dims = analysis_result.angular_dims

    # When in canonical space, don't clamp to obs bounds
    # (canonical coords are unbounded, e.g. theta in [-pi, pi])
    obs_low = None
    obs_high = None
    if analysis_result is None:
        ref_env = test_family.make_env(0)
        obs_low = np.asarray(
            ref_env.observation_space.low,  # type: ignore[attr-defined]
            dtype=np.float64,
        )
        obs_high = np.asarray(
            ref_env.observation_space.high,  # type: ignore[attr-defined]
            dtype=np.float64,
        )
        ref_env.close()

    # Parse dynamics expressions by dim index
    state_dim = len(state_names)
    dynamics_expressions: dict[int, SymbolicExpression] = {}
    for target, entry in best_dynamics.items():
        dim_name = target.removeprefix("delta_")
        dim_idx = state_names.index(dim_name)
        dynamics_expressions[dim_idx] = entry.expression

    # -- Section 7.2: Spurious term detection --
    spurious_dataset = cd_output.get("dataset") if cd_output is not None else None
    if spurious_dataset is not None:
        from circ_rl.hypothesis.spurious_detection import SpuriousTermDetector

        detector = SpuriousTermDetector(r2_contribution_threshold=0.005)
        spvar_names = list(state_names) + list(action_names)
        if test_family.param_names:
            spvar_names.extend(test_family.param_names)

        sp_dataset = spurious_dataset
        if oa_output is not None:
            oa_canonical = oa_output.get("canonical_dataset")
            if oa_canonical is not None:
                sp_dataset = oa_canonical

        for dim_idx, expr in list(dynamics_expressions.items()):
            try:
                sp_result = detector.detect(
                    expr, sp_dataset, dim_idx, spvar_names,
                    wrap_angular=(dim_idx in angular_dims),
                )
                if sp_result.n_spurious > 0:
                    from circ_rl.hypothesis.expression import (
                        SymbolicExpression as _SPE,
                    )
                    dynamics_expressions[dim_idx] = _SPE.from_sympy(
                        sp_result.pruned_expr,
                    )
                    print(
                        f"    Pruned {sp_result.n_spurious} spurious terms "
                        f"from dim {dim_idx} "
                        f"(R2: {sp_result.original_r2:.6f} -> "
                        f"{sp_result.pruned_r2:.6f})"
                    )
            except Exception:
                logger.debug(
                    "Spurious detection skipped for dim {} "
                    "(evaluation error)",
                    dim_idx,
                )

    # -- Section 7.2: Coefficient calibration (pooled) --
    # For test envs we use pooled (alpha, beta) since per-env
    # calibration is only available for training env indices.
    pooled_cal: dict[int, tuple[float, float]] = {}
    for target, entry in best_dynamics.items():
        dim_name = target.removeprefix("delta_")
        dim_idx = state_names.index(dim_name)
        if entry.pooled_calibration is not None:
            pooled_cal[dim_idx] = entry.pooled_calibration
    cal_arg = pooled_cal or None
    if cal_arg:
        print(f"    Calibration: {len(cal_arg)} dims with pooled (alpha, beta)")

    # -- Section 7.2: Adaptive replanning threshold --
    dynamics_mse_sum = sum(e.training_mse for e in best_dynamics.values())
    adaptive_tau: float | None = None
    if dynamics_mse_sum > 0:
        adaptive_tau = adaptive_replan_multiplier * np.sqrt(dynamics_mse_sum)
    if adaptive_tau is not None:
        print(f"    Adaptive replan threshold: tau={adaptive_tau:.4f}")

    ilqr_config = ILQRConfig(
        horizon=100,
        gamma=gamma,
        max_action=max_action,
        n_random_restarts=8,
        restart_scale=0.3,
        replan_interval=10,
        adaptive_replan_threshold=adaptive_tau,
        min_replan_interval=min_replan_interval,
    )

    ilqr_solvers: dict[int, ILQRSolver] = {}
    for env_idx in range(test_family.n_envs):
        env_params = test_family.get_env_params(env_idx)

        dynamics_fn = _build_dynamics_fn(
            dynamics_expressions,
            state_names,
            action_names,
            state_dim,
            env_params,
            obs_low=obs_low,
            obs_high=obs_high,
            angular_dims=angular_dims,
            calibration_coefficients=cal_arg,
        )

        reward_fn = None
        if best_reward is not None:
            reward_fn = _build_reward_fn(
                best_reward.expression,
                state_names,
                action_names,
                env_params,
                reward_derived_features if reward_derived_features else None,
                canonical_to_obs_fn=canonical_to_obs_fn,
                obs_state_names=obs_state_names,
            )

        jac_state_fn, jac_action_fn = _build_dynamics_jacobian_fns(
            dynamics_expressions,
            state_names,
            action_names,
            state_dim,
            env_params,
            obs_low=obs_low,
            obs_high=obs_high,
            calibration_coefficients=cal_arg,
        )

        # Build analytic reward derivatives if possible
        reward_deriv_fn = None
        if best_reward is not None:
            reward_deriv_fn = _build_reward_derivative_fns(
                best_reward.expression,
                state_names,
                action_names,
                env_params,
                reward_derived_features or None,
                obs_state_names=obs_state_names,
            )

        effective_reward = reward_fn or _default_reward
        terminal_cost_fn = make_quadratic_terminal_cost(
            reward_fn=effective_reward,
            action_dim=1,
            gamma=gamma,
            state_dim=state_dim,
            scale_override=10.0,
        )

        ilqr_solvers[env_idx] = ILQRSolver(
            config=ilqr_config,
            dynamics_fn=dynamics_fn,
            reward_fn=effective_reward,
            dynamics_jac_state_fn=jac_state_fn,
            dynamics_jac_action_fn=jac_action_fn,
            terminal_cost_fn=terminal_cost_fn,
            reward_derivatives_fn=reward_deriv_fn,
        )

    first_entry = next(iter(best_dynamics.values()))
    action_dim = 1

    return _ILQRAnalyticPolicy(
        dynamics_hypothesis=first_entry,
        reward_hypothesis=best_reward,
        state_dim=state_dim,
        action_dim=action_dim,
        n_envs=test_family.n_envs,
        ilqr_solvers=ilqr_solvers,
        action_low=-max_action * np.ones(action_dim),
        action_high=max_action * np.ones(action_dim),
        obs_to_canonical_fn=obs_to_canonical_fn,
    )


# -- v1 policy loader --


def _load_v1_policy(
    test_family: EnvironmentFamily,
    checkpoint_path: str,
    max_torque: float = 100.0,
) -> tuple[CausalPolicy, list[str]]:
    """Load v1 CausalPolicy from checkpoint.

    :returns: (policy, env_param_names).
    """
    env_param_names = ["g", "m", "l"]

    # Compute dynamics reference scale from training family
    train_family = EnvironmentFamily.from_gymnasium(
        base_env="Pendulum-v1",
        param_distributions={
            "g": (7.0, 13.0),
            "m": (0.5, 2.0),
            "l": (0.5, 1.5),
        },
        n_envs=16,
        seed=42,
        fixed_params={"max_torque": max_torque},
    )
    collector = DataCollector(train_family, include_env_params=True)
    discovery_data = collector.collect(n_transitions_per_env=2000, seed=42)
    state_names = [f"s{i}" for i in range(3)]
    ta_result = TransitionAnalyzer().analyze(discovery_data, state_names, 1)
    print(f"  v1 reference dynamics scale: {ta_result.reference_scale:.4f}")

    action_low = np.array([-max_torque], dtype=np.float32)
    action_high = np.array([max_torque], dtype=np.float32)

    policy = CausalPolicy(
        full_state_dim=3,
        feature_mask=np.ones(3, dtype=bool),
        action_dim=1,
        hidden_dims=(64, 64),
        continuous=True,
        action_low=action_low,
        action_high=action_high,
        context_dim=len(env_param_names),
        use_dynamics_normalization=True,
        dynamics_reference_scale=ta_result.reference_scale,
    )

    state_dict = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    policy.load_state_dict(state_dict)
    policy.eval()

    return policy, env_param_names


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="v1 (neural) vs v2 (analytic) comparison on 50 pendulum envs",
    )
    parser.add_argument(
        "--eval-mode",
        choices=["full", "adaptive"],
        default="full",
        help=(
            "'full': record video for all 50 envs (original behavior). "
            "'adaptive': quick-screen all envs, then full video on worst "
            "15 + random 10 (faster)."
        ),
    )
    parser.add_argument(
        "--quick-steps",
        type=int,
        default=100,
        help="Steps per env in adaptive quick-screening phase (default: 100).",
    )
    parser.add_argument(
        "--n-video-worst",
        type=int,
        default=15,
        help="Number of worst-performing envs to record full video (default: 15).",
    )
    parser.add_argument(
        "--n-video-random",
        type=int,
        default=10,
        help="Number of random easy envs to record full video (default: 10).",
    )
    return parser.parse_args()


def main() -> None:
    """Run v1 vs v2 comparison on 50 diverse environments."""
    args = _parse_args()

    logger.remove()
    logger.add(sys.stderr, level="INFO")
    os.environ["SDL_VIDEODRIVER"] = "dummy"

    output_dir = "experiments/outputs"
    os.makedirs(output_dir, exist_ok=True)

    n_test_envs = 50
    max_steps = 600

    # Both policies use max_torque=100 for fair comparison
    v1_max_torque = 100.0
    v2_max_torque = 100.0

    eval_mode = args.eval_mode

    print("=" * 70)
    print("CIRC-RL: v1 (NEURAL) vs v2 (ANALYTIC) COMPARISON")
    print("  50 test environments with OOD physics (g, m, l)")
    print(f"  v1: CausalPolicy + PPO (max_torque={v1_max_torque})")
    print(f"  v2: PySR + iLQR (max_torque={v2_max_torque})")
    print(f"  Eval mode: {eval_mode}")
    print("=" * 70)

    t0 = time.time()

    # ================================================================
    # Create 50 TEST environments (OOD parameters)
    # ================================================================
    print("\n[1/5] Creating 50 test environments...")
    test_family = EnvironmentFamily.from_gymnasium(
        base_env="Pendulum-v1",
        param_distributions={
            "g": (5.0, 15.0),
            "m": (0.3, 3.0),
            "l": (0.3, 2.0),
        },
        n_envs=n_test_envs,
        seed=123,
    )

    # ================================================================
    # Run v2 pipeline on training envs
    # ================================================================
    print("\n[2/5] Running v2 pipeline (PySR + falsification)...")
    t_v2 = time.time()
    (
        best_dynamics,
        best_reward,
        dynamics_state_names,
        action_names,
        reward_derived_features,
        _hf_output,
        oa_output,
        obs_state_names,
        cd_output,
    ) = _run_v2_pipeline(seed=42)

    if not best_dynamics:
        print("\n  ERROR: No validated dynamics. Cannot build v2 policy.")
        print("  Try re-running -- PySR is stochastic.")
        return

    # Build iLQR policy for all 50 test envs (with Section 7.2 improvements)
    print("\n  Building iLQR solvers for 50 test envs...")
    v2_policy = _build_v2_ilqr_policy(
        test_family,
        best_dynamics,
        best_reward,
        dynamics_state_names,
        action_names,
        reward_derived_features,
        oa_output=oa_output,
        gamma=0.99,
        max_action=v2_max_torque,
        obs_state_names=obs_state_names,
        cd_output=cd_output,
    )
    t_v2_total = time.time() - t_v2
    print(f"  v2 pipeline + iLQR build: {t_v2_total:.1f}s")

    # v2 action function (uses iLQR policy with feedback gains)
    def v2_get_action(obs: np.ndarray, env_idx: int) -> np.ndarray:
        action = v2_policy.get_action(obs, env_idx)
        return np.clip(action, -v2_max_torque, v2_max_torque)

    # ================================================================
    # Load v1 policy
    # ================================================================
    print("\n[3/5] Loading v1 neural policy...")
    checkpoint_path = os.path.join(
        output_dir,
        "pendulum_policy_best.pt",
    )
    if not os.path.exists(checkpoint_path):
        print(f"  WARNING: Checkpoint not found at {checkpoint_path}")
        print("  Skipping v1. Only v2 will be recorded.")
        v1_policy = None
        v1_param_names: list[str] = []
    else:
        v1_policy, v1_param_names = _load_v1_policy(
            test_family,
            checkpoint_path,
            max_torque=v1_max_torque,
        )
        print(f"  Loaded v1 policy from {checkpoint_path}")

    # v1 action function
    def v1_get_action(obs: np.ndarray, env_idx: int) -> np.ndarray:
        assert v1_policy is not None
        params = test_family.get_env_params(env_idx)
        context = torch.tensor(
            [params[name] for name in v1_param_names],
            dtype=torch.float32,
        )
        state_tensor = torch.tensor(
            obs,
            dtype=torch.float32,
        ).unsqueeze(0)
        action_np = v1_policy.get_action(
            state_tensor,
            deterministic=True,
            context=context,
        )
        assert isinstance(action_np, np.ndarray)
        return action_np

    # ================================================================
    # Record episodes
    # ================================================================
    v1_returns: list[float] = []
    v2_returns: list[float] = []
    v1_all_frames: list[list[np.ndarray]] = []
    v2_all_frames: list[list[np.ndarray]] = []
    v1_labels: list[str] = []
    v2_labels: list[str] = []

    # Determine which envs get full video recording
    if eval_mode == "adaptive":
        # Phase 1: Quick parallel screening of all envs
        print(f"\n[4/6] Quick screening {n_test_envs} envs "
              f"({args.quick_steps} steps, parallel)...")
        t_quick = time.time()
        quick_returns = _quick_evaluate_parallel(
            test_family,
            list(range(n_test_envs)),
            v2_get_action,
            reset_fn=v2_policy.reset,
            max_steps=args.quick_steps,
            fixed_params={"max_torque": v2_max_torque},
        )
        t_quick_elapsed = time.time() - t_quick
        print(f"  Quick screening done in {t_quick_elapsed:.1f}s")

        # Rank envs by return (worst first)
        sorted_envs = sorted(quick_returns.keys(), key=lambda i: quick_returns[i])
        quick_arr = np.array([quick_returns[i] for i in range(n_test_envs)])
        print(f"  Quick returns: mean={quick_arr.mean():.1f}, "
              f"min={quick_arr.min():.1f}, max={quick_arr.max():.1f}")

        # Select subset for full video: worst N + random from the rest
        n_worst = min(args.n_video_worst, n_test_envs)
        worst_envs = set(sorted_envs[:n_worst])
        remaining = [i for i in sorted_envs[n_worst:]]
        rng = np.random.default_rng(42)
        n_rand = min(args.n_video_random, len(remaining))
        random_envs = set(rng.choice(remaining, size=n_rand, replace=False))
        video_envs = sorted(worst_envs | random_envs)

        print(f"  Full video: {len(video_envs)} envs "
              f"({n_worst} worst + {n_rand} random)")

        # Store quick returns for all envs (used in final summary)
        for env_idx in range(n_test_envs):
            v2_returns.append(quick_returns[env_idx])

        # Phase 2: Full recording on selected subset
        step_label = "[5/6]"
    else:
        # Full mode: record all envs
        video_envs = list(range(n_test_envs))
        step_label = "[4/5]"

    print(f"\n{step_label} Recording {len(video_envs)} episodes per policy...")

    for i, env_idx in enumerate(video_envs):
        params = test_family.get_env_params(env_idx)
        param_str = f"g={params['g']:.1f} m={params['m']:.1f} l={params['l']:.1f}"
        print(f"  [{i + 1:2d}/{len(video_envs)}] env {env_idx} {param_str}",
              end="", flush=True)

        # v2 (analytic) -- reset iLQR plan for new episode
        v2_policy.reset(env_idx)
        v2_frames, v2_ret = _record_episode_generic(
            test_family,
            env_idx,
            v2_get_action,
            max_steps=max_steps,
            fixed_params={"max_torque": v2_max_torque},
        )
        if eval_mode == "full":
            v2_returns.append(v2_ret)
        v2_all_frames.append(v2_frames)
        v2_labels.append(f"R={v2_ret:.0f}")
        print(f"  v2={v2_ret:.0f}", end="", flush=True)

        # v1 (neural)
        if v1_policy is not None:
            v1_frames, v1_ret = _record_episode_generic(
                test_family,
                env_idx,
                v1_get_action,
                max_steps=max_steps,
                fixed_params={"max_torque": v1_max_torque},
            )
            v1_returns.append(v1_ret)
            v1_all_frames.append(v1_frames)
            v1_labels.append(f"R={v1_ret:.0f}")
            print(f"  v1={v1_ret:.0f}")
        else:
            print()

    # ================================================================
    # Summary + videos
    # ================================================================
    summary_step = "[6/6]" if eval_mode == "adaptive" else "[5/5]"
    print(f"\n{summary_step} Generating videos...")

    v2_arr = np.array(v2_returns)
    v2_video_path = os.path.join(output_dir, "pendulum_v2_50env.mp4")
    _make_grid_video(
        v2_all_frames,
        v2_labels,
        v2_video_path,
        fps=60,
        n_cols=10,
    )
    print(f"  v2 video: {v2_video_path}")

    if v1_policy is not None and v1_all_frames:
        v1_arr = np.array(v1_returns)
        v1_video_path = os.path.join(output_dir, "pendulum_v1_50env.mp4")
        _make_grid_video(
            v1_all_frames,
            v1_labels,
            v1_video_path,
            fps=60,
            n_cols=10,
        )
        print(f"  v1 video: {v1_video_path}")

    elapsed = time.time() - t0

    # Print comparison table
    n_v2_eval = len(v2_returns)
    print(f"\n{'=' * 70}")
    print("COMPARISON RESULTS")
    if eval_mode == "adaptive":
        print(f"  (v2 stats from {args.quick_steps}-step quick screening "
              f"of {n_v2_eval} envs; video on {len(video_envs)} envs)")
    print(f"{'=' * 70}")
    print(f"  {'Metric':<20} {'v2 (analytic)':>15}", end="")
    if v1_policy is not None and v1_returns:
        print(f" {'v1 (neural)':>15}", end="")
    print()
    print(f"  {'-' * 20} {'-' * 15}", end="")
    if v1_policy is not None and v1_returns:
        print(f" {'-' * 15}", end="")
    print()
    print(f"  {'Mean return':<20} {v2_arr.mean():>15.1f}", end="")
    if v1_policy is not None and v1_returns:
        print(f" {v1_arr.mean():>15.1f}", end="")
    print()
    print(f"  {'Median return':<20} {np.median(v2_arr):>15.1f}", end="")
    if v1_policy is not None and v1_returns:
        print(f" {np.median(v1_arr):>15.1f}", end="")
    print()
    print(f"  {'Best return':<20} {v2_arr.max():>15.1f}", end="")
    if v1_policy is not None and v1_returns:
        print(f" {v1_arr.max():>15.1f}", end="")
    print()
    print(f"  {'Worst return':<20} {v2_arr.min():>15.1f}", end="")
    if v1_policy is not None and v1_returns:
        print(f" {v1_arr.min():>15.1f}", end="")
    print()
    print(f"  {'Std return':<20} {v2_arr.std():>15.1f}", end="")
    if v1_policy is not None and v1_returns:
        print(f" {v1_arr.std():>15.1f}", end="")
    print()
    print(f"  {'Max torque':<20} {v2_max_torque:>15.1f}", end="")
    if v1_policy is not None:
        print(f" {v1_max_torque:>15.1f}", end="")
    print()
    print(f"\n  Total time: {elapsed:.1f}s")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
