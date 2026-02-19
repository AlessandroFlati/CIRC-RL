# ruff: noqa: T201
"""Diagnose: how much does unmodeled velocity clipping hurt iLQR?

For OOD test environments, the PySR-discovered dynamics don't model
the velocity clip at [-8, 8]. This script measures the impact by:
1. Computing what fraction of steps hit the clip limit per env
2. Comparing iLQR with clipped vs unclipped dynamics on affected envs
"""
from __future__ import annotations

import numpy as np

from circ_rl.analytic_policy.ilqr_solver import ILQRConfig, ILQRSolver
from circ_rl.utils.seeding import seed_everything


def _dynamics_with_clip(
    state: np.ndarray,
    action: np.ndarray,
    g: float,
    m: float,
    l: float,
    dt: float = 0.05,
) -> np.ndarray:
    """True Pendulum-v1 dynamics WITH velocity clipping."""
    phi_0, s2 = state[0], state[1]
    u = float(action[0])
    accel = 3.0 * g / (2.0 * l) * np.sin(phi_0) + 3.0 * u / (m * l**2)
    new_s2 = np.clip(s2 + accel * dt, -8.0, 8.0)
    new_phi_0 = float(np.arctan2(
        np.sin(phi_0 + new_s2 * dt),
        np.cos(phi_0 + new_s2 * dt),
    ))
    return np.array([new_phi_0, new_s2])


def _dynamics_no_clip(
    state: np.ndarray,
    action: np.ndarray,
    g: float,
    m: float,
    l: float,
    dt: float = 0.05,
) -> np.ndarray:
    """True Pendulum-v1 dynamics WITHOUT velocity clipping."""
    phi_0, s2 = state[0], state[1]
    u = float(action[0])
    accel = 3.0 * g / (2.0 * l) * np.sin(phi_0) + 3.0 * u / (m * l**2)
    new_s2 = s2 + accel * dt
    new_phi_0 = float(np.arctan2(
        np.sin(phi_0 + new_s2 * dt),
        np.cos(phi_0 + new_s2 * dt),
    ))
    return np.array([new_phi_0, new_s2])


def _true_reward(state: np.ndarray, action: np.ndarray) -> float:
    phi_0, s2 = state[0], state[1]
    u = float(action[0])
    return -(phi_0**2 + 0.1 * s2**2 + 0.001 * u**2)


def _run_episode(
    solver: ILQRSolver,
    g: float,
    m: float,
    l: float,
    seed: int = 42,
    max_steps: int = 200,
    n_restarts: int = 5,
) -> tuple[float, float, float]:
    """Run episode. Returns (return, max_vel, clip_fraction)."""
    rng = np.random.default_rng(seed)
    theta = rng.uniform(-np.pi, np.pi)
    thetadot = rng.uniform(-1.0, 1.0)
    state = np.array([theta, thetadot])

    # Multi-start
    best_sol = None
    best_reward = -np.inf
    for i in range(n_restarts + 1):
        warm = (
            None if i == 0
            else rng.uniform(-100.0, 100.0, size=(solver.config.horizon, 1))
        )
        sol = solver.plan(state, 1, warm_start_actions=warm)
        if sol.total_reward > best_reward:
            best_reward = sol.total_reward
            best_sol = sol
    assert best_sol is not None

    # Execute with TRUE (clipped) dynamics
    total_return = 0.0
    n_clipped = 0
    max_vel = 0.0
    current = state.copy()
    for t in range(min(max_steps, best_sol.nominal_states.shape[0] - 1)):
        dx = current - best_sol.nominal_states[t]
        action = best_sol.nominal_actions[t] + best_sol.feedback_gains[t] @ dx
        action = np.clip(action, -100.0, 100.0)
        total_return += _true_reward(current, action)

        # Step with TRUE dynamics (with clipping)
        current = _dynamics_with_clip(current, action, g, m, l)
        vel = abs(current[1])
        max_vel = max(max_vel, vel)
        if vel >= 7.99:  # effectively clipped
            n_clipped += 1

    clip_frac = n_clipped / max_steps
    return total_return, max_vel, clip_frac


def main() -> None:
    seed_everything(42)

    # Test range of environments spanning the OOD distribution
    test_envs = [
        # Short l: high gravity acceleration -> velocity clips often
        ("g=15.0, m=1.0, l=0.3", {"g": 15.0, "m": 1.0, "l": 0.3}),
        ("g=12.0, m=0.8, l=0.5", {"g": 12.0, "m": 0.8, "l": 0.5}),
        ("g=10.0, m=2.0, l=0.5", {"g": 10.0, "m": 2.0, "l": 0.5}),
        # Medium l
        ("g=10.0, m=1.0, l=1.0", {"g": 10.0, "m": 1.0, "l": 1.0}),
        ("g=10.0, m=2.7, l=1.0", {"g": 10.0, "m": 2.7, "l": 1.0}),
        # Long l: low gravity acceleration -> velocity rarely clips
        ("g= 9.3, m=2.7, l=1.9", {"g":  9.3, "m": 2.7, "l": 1.9}),
        ("g=10.2, m=2.7, l=2.0", {"g": 10.2, "m": 2.7, "l": 2.0}),
        ("g= 8.0, m=2.8, l=1.3", {"g":  8.0, "m": 2.8, "l": 1.3}),
    ]

    ilqr_config = ILQRConfig(
        horizon=200,
        gamma=0.99,
        max_action=100.0,
    )

    print("=" * 80)
    print("DIAGNOSTIC: Velocity clipping impact on iLQR")
    print("=" * 80)
    print(f"{'Environment':<30} {'clip_dyn R':>10} {'noclip_dyn R':>12} "
          f"{'delta':>8} {'max_vel':>8} {'clip%':>6}")
    print("-" * 80)

    from loguru import logger as _logger
    _logger.disable("circ_rl")

    for env_name, params in test_envs:
        g, m, l_val = params["g"], params["m"], params["l"]

        # Solver A: plans with clipped dynamics (true model)
        def make_clip_dyn(
            g: float = g, m: float = m, l: float = l_val,
        ) -> Any:
            def dyn(s: np.ndarray, a: np.ndarray) -> np.ndarray:
                return _dynamics_with_clip(s, a, g, m, l)
            return dyn

        # Solver B: plans with unclipped dynamics (PySR-like model)
        def make_noclip_dyn(
            g: float = g, m: float = m, l: float = l_val,
        ) -> Any:
            def dyn(s: np.ndarray, a: np.ndarray) -> np.ndarray:
                return _dynamics_no_clip(s, a, g, m, l)
            return dyn

        solver_clip = ILQRSolver(
            config=ilqr_config,
            dynamics_fn=make_clip_dyn(),
            reward_fn=_true_reward,
        )
        solver_noclip = ILQRSolver(
            config=ilqr_config,
            dynamics_fn=make_noclip_dyn(),
            reward_fn=_true_reward,
        )

        ret_clip, max_v_clip, clip_frac_clip = _run_episode(
            solver_clip, g, m, l_val, n_restarts=5,
        )
        ret_noclip, max_v_noclip, clip_frac_noclip = _run_episode(
            solver_noclip, g, m, l_val, n_restarts=5,
        )

        delta = ret_noclip - ret_clip
        print(
            f"  {env_name:<28} {ret_clip:10.1f} {ret_noclip:12.1f} "
            f"{delta:+8.1f} {max_v_clip:8.1f} {clip_frac_clip*100:5.1f}%"
        )


from typing import Any  # noqa: E402


if __name__ == "__main__":
    main()
