"""Pendulum benchmark environment family.

Creates a parametrically varied Pendulum family for evaluating CIRC-RL.
Parameters randomized: gravity, mass, length, max_torque.

Typical usage::

    family = make_pendulum_family(n_envs=10, seed=42)
    held_out = make_pendulum_family(n_envs=5, seed=999)
"""

from __future__ import annotations

from circ_rl.environments.env_family import EnvironmentFamily


def make_pendulum_family(
    n_envs: int = 10,
    seed: int = 42,
    gravity_range: tuple[float, float] = (7.0, 13.0),
    mass_range: tuple[float, float] = (0.5, 2.0),
    length_range: tuple[float, float] = (0.5, 1.5),
    max_torque_range: tuple[float, float] = (1.0, 3.0),
) -> EnvironmentFamily:
    """Create a Pendulum environment family with randomized physics.

    Default ranges are chosen to produce meaningful variation while
    keeping the environment controllable:
    - gravity (g): 7.0-13.0 (default 10.0)
    - mass (m): 0.5-2.0 (default 1.0)
    - length (l): 0.5-1.5 (default 1.0)
    - max_torque: 1.0-3.0 (default 2.0)

    :param n_envs: Number of environment instances.
    :param seed: Random seed for parameter sampling.
    :param gravity_range: (low, high) for gravitational acceleration.
    :param mass_range: (low, high) for pendulum mass.
    :param length_range: (low, high) for pendulum length.
    :param max_torque_range: (low, high) for maximum torque.
    :returns: An EnvironmentFamily of varied Pendulum instances.
    """
    return EnvironmentFamily.from_gymnasium(
        base_env="Pendulum-v1",
        param_distributions={
            "g": gravity_range,
            "m": mass_range,
            "l": length_range,
            "max_torque": max_torque_range,
        },
        n_envs=n_envs,
        seed=seed,
    )


def make_pendulum_family_minimal(
    n_envs: int = 3,
    seed: int = 42,
) -> EnvironmentFamily:
    """Create a small Pendulum family for quick testing.

    Only varies gravity with a narrow range.

    :param n_envs: Number of environment instances.
    :param seed: Random seed.
    :returns: An EnvironmentFamily.
    """
    return EnvironmentFamily.from_gymnasium(
        base_env="Pendulum-v1",
        param_distributions={"g": (9.0, 11.0)},
        n_envs=n_envs,
        seed=seed,
    )
