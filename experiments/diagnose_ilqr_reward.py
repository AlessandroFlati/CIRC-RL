# ruff: noqa: T201
"""Diagnose: is the wrong reward function causing iLQR failure on high-inertia envs?

Compares iLQR performance with:
  (a) Default reward: -(phi_0^2 + s2^2 + 0.01*action^2)
  (b) True Pendulum-v1 reward: -(phi_0^2 + 0.1*s2^2 + 0.001*action^2)
  (c) True reward + multiple random action initializations

Tests on 6 environments: 3 where v2 fails badly, 3 where v2 succeeds.
"""
from __future__ import annotations

import time

import gymnasium as gym
import numpy as np
from loguru import logger

from circ_rl.analytic_policy.ilqr_solver import ILQRConfig, ILQRSolver
from circ_rl.environments.env_family import EnvironmentFamily
from circ_rl.utils.seeding import seed_everything


def _true_pendulum_dynamics(
    state: np.ndarray,
    action: np.ndarray,
    g: float,
    m: float,
    l: float,
    dt: float = 0.05,
) -> np.ndarray:
    """True Pendulum-v1 dynamics in canonical (phi_0, s2) coordinates.

    phi_0 = angle (0 = upright), s2 = angular velocity.
    """
    phi_0 = state[0]
    s2 = state[1]
    u = float(action[0])

    # True physics: theta_dot_dot = -3g/(2l)*sin(theta+pi) + 3u/(m*l^2)
    # In canonical coords: phi_0 is theta, so sin(phi_0 + pi) = -sin(phi_0)
    accel = 3.0 * g / (2.0 * l) * np.sin(phi_0) + 3.0 * u / (m * l**2)

    # Semi-implicit Euler with velocity clipping
    new_s2 = s2 + accel * dt
    new_s2 = np.clip(new_s2, -8.0, 8.0)
    new_phi_0 = phi_0 + new_s2 * dt

    # Wrap angle to [-pi, pi]
    new_phi_0 = float(np.arctan2(np.sin(new_phi_0), np.cos(new_phi_0)))

    return np.array([new_phi_0, new_s2])


def _approx_pendulum_dynamics(
    state: np.ndarray,
    action: np.ndarray,
    g: float,
    m: float,
    l: float,
    dt: float = 0.05,
) -> np.ndarray:
    """Approximate dynamics from PySR expressions (no velocity clipping).

    Uses the discovered expressions:
      delta_s2 = 0.149*action/(m*l^2) + 0.0748*g*sin(phi_0)/l
      delta_phi_0 ~ 0.05*s2 + small corrections
    """
    phi_0 = state[0]
    s2 = state[1]
    u = float(action[0])

    # delta_s2 from PySR (R2=0.9969)
    delta_s2 = 0.149 * u / (m * l**2) + 0.0748 * g * np.sin(phi_0) / l

    # delta_phi_0: simplified version (s2*dt is dominant term)
    delta_phi_0 = 0.05 * s2 + 0.0077 * u + 0.037 * np.sin(phi_0) / l

    new_s2 = s2 + delta_s2
    new_phi_0 = phi_0 + delta_phi_0

    # Wrap angle
    new_phi_0 = float(np.arctan2(np.sin(new_phi_0), np.cos(new_phi_0)))

    return np.array([new_phi_0, new_s2])


def _default_reward(state: np.ndarray, action: np.ndarray) -> float:
    """Default (wrong) reward: -(phi_0^2 + s2^2 + 0.01*action^2)."""
    return -float(np.sum(state**2) + 0.01 * np.sum(action**2))


def _true_reward(state: np.ndarray, action: np.ndarray) -> float:
    """True Pendulum-v1 reward: -(phi_0^2 + 0.1*s2^2 + 0.001*action^2)."""
    phi_0 = state[0]
    s2 = state[1]
    u = float(action[0])
    return -(phi_0**2 + 0.1 * s2**2 + 0.001 * u**2)


def _run_episode_with_ilqr(
    solver: ILQRSolver,
    env_params: dict[str, float],
    max_steps: int = 200,
    seed: int = 42,
    n_random_restarts: int = 0,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Run one episode using iLQR with the TRUE environment dynamics.

    The iLQR plans using its internal dynamics model, but the episode
    is rolled out using the TRUE Pendulum-v1 dynamics to measure actual
    performance.

    :returns: (total_return, states_trajectory, actions_trajectory)
    """
    g = env_params["g"]
    m = env_params["m"]
    l_val = env_params["l"]

    # Random initial state (same as Pendulum-v1)
    rng = np.random.default_rng(seed)
    theta = rng.uniform(-np.pi, np.pi)
    thetadot = rng.uniform(-1.0, 1.0)
    state = np.array([theta, thetadot])

    # Plan from initial state
    if n_random_restarts > 0:
        # Try multiple random action initializations, pick best
        best_sol = None
        best_reward = -np.inf
        for restart_idx in range(n_random_restarts + 1):
            if restart_idx == 0:
                warm_start = None  # zero init
            else:
                warm_start = rng.uniform(
                    -100.0, 100.0,
                    size=(solver.config.horizon, 1),
                )
            sol = solver.plan(state, 1, warm_start_actions=warm_start)
            if sol.total_reward > best_reward:
                best_reward = sol.total_reward
                best_sol = sol
        sol = best_sol
    else:
        sol = solver.plan(state, 1)

    assert sol is not None

    # Execute plan in TRUE dynamics
    total_return = 0.0
    states_list = [state.copy()]
    actions_list = []

    current_state = state.copy()
    for t in range(min(max_steps, sol.nominal_states.shape[0] - 1)):
        # Feedback control
        dx = current_state - sol.nominal_states[t]
        action = sol.nominal_actions[t] + sol.feedback_gains[t] @ dx
        action = np.clip(action, -100.0, 100.0)
        actions_list.append(action.copy())

        # True reward
        r = _true_reward(current_state, action)
        total_return += r

        # Step with TRUE dynamics
        current_state = _true_pendulum_dynamics(
            current_state, action, g, m, l_val,
        )
        states_list.append(current_state.copy())

    return total_return, np.array(states_list), np.array(actions_list)


def main() -> None:
    seed_everything(42)
    logger.disable("circ_rl")  # Reduce noise

    # Test environments (from experiment results)
    test_envs = {
        # FAILING: v2 loses badly to v1
        "FAIL env13 (m=2.7, l=1.9)": {"g": 9.3, "m": 2.7, "l": 1.9},
        "FAIL env42 (m=2.7, l=2.0)": {"g": 10.2, "m": 2.7, "l": 2.0},
        "FAIL env49 (m=2.3, l=1.6)": {"g": 9.6, "m": 2.3, "l": 1.6},
        # SUCCEEDING: v2 beats v1
        "GOOD env05 (m=0.5, l=1.0)": {"g": 9.4, "m": 0.5, "l": 1.0},
        "GOOD env48 (m=2.8, l=1.3)": {"g": 8.0, "m": 2.8, "l": 1.3},
        "GOOD env34 (m=1.7, l=1.4)": {"g": 8.4, "m": 1.7, "l": 1.4},
    }

    # Use true dynamics for all solvers (isolate the reward function effect)
    ilqr_config = ILQRConfig(
        horizon=200,
        gamma=0.99,
        max_action=100.0,
    )

    print("=" * 80)
    print("DIAGNOSTIC: iLQR reward function impact")
    print("  Comparing default reward vs true Pendulum-v1 reward")
    print("  All tests use TRUE dynamics (not PySR approximations)")
    print("=" * 80)

    # Test each reward configuration
    configs = [
        ("A: Default reward", _default_reward, 0),
        ("B: True Pendulum reward", _true_reward, 0),
        ("C: True reward + 5 restarts", _true_reward, 5),
    ]

    # Summary storage
    results: dict[str, dict[str, float]] = {}

    for config_name, reward_fn, n_restarts in configs:
        print(f"\n{'='*70}")
        print(f"Config: {config_name}")
        print(f"{'='*70}")

        results[config_name] = {}

        for env_name, params in test_envs.items():
            g, m, l_val = params["g"], params["m"], params["l"]

            def make_dynamics(
                g: float = g, m: float = m, l: float = l_val,
            ) -> Any:
                def dyn(state: np.ndarray, action: np.ndarray) -> np.ndarray:
                    return _true_pendulum_dynamics(state, action, g, m, l)
                return dyn

            solver = ILQRSolver(
                config=ilqr_config,
                dynamics_fn=make_dynamics(),
                reward_fn=reward_fn,
            )

            # Run episode
            t0 = time.time()
            total_ret, states, actions = _run_episode_with_ilqr(
                solver, params, max_steps=200, seed=42,
                n_random_restarts=n_restarts,
            )
            elapsed = time.time() - t0

            # Compute diagnostics
            max_vel = float(np.max(np.abs(states[:, 1])))
            max_action_used = float(np.max(np.abs(actions)))
            # Final state
            final_phi = states[-1, 0]
            final_s2 = states[-1, 1]
            # Time at top (|phi_0| < 0.5)
            at_top = np.sum(np.abs(states[:, 0]) < 0.5)

            results[config_name][env_name] = total_ret

            print(
                f"  {env_name}: "
                f"R={total_ret:8.1f}  "
                f"max_vel={max_vel:5.1f}  "
                f"max_u={max_action_used:6.1f}  "
                f"at_top={at_top:3d}/200  "
                f"final_phi={final_phi:+5.2f}  "
                f"({elapsed:.1f}s)"
            )

    # Summary comparison
    print(f"\n\n{'='*80}")
    print("SUMMARY: True reward (in actual env) for each configuration")
    print(f"{'='*80}")
    print(f"{'Environment':<35}", end="")
    for config_name, _, _ in configs:
        label = config_name.split(":")[0]
        print(f" {label:>12}", end="")
    print()
    print("-" * 80)

    for env_name in test_envs:
        print(f"  {env_name:<33}", end="")
        for config_name, _, _ in configs:
            val = results[config_name][env_name]
            print(f" {val:12.1f}", end="")
        print()

    # Averages
    print("-" * 80)
    for group_prefix, group_label in [("FAIL", "FAILING mean"), ("GOOD", "SUCCESS mean")]:
        print(f"  {group_label:<33}", end="")
        for config_name, _, _ in configs:
            vals = [
                v for k, v in results[config_name].items()
                if k.startswith(group_prefix)
            ]
            print(f" {np.mean(vals):12.1f}", end="")
        print()

    print(f"  {'OVERALL mean':<33}", end="")
    for config_name, _, _ in configs:
        vals = list(results[config_name].values())
        print(f" {np.mean(vals):12.1f}", end="")
    print()


# Need this for the type annotation in the closure
from typing import Any  # noqa: E402


if __name__ == "__main__":
    main()
