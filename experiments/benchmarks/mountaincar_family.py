"""MountainCar benchmark environment families.

Creates parametrically varied MountainCar and MountainCarContinuous families
for evaluating CIRC-RL.

MountainCar-v0 parameters randomized: gravity, force.
MountainCarContinuous-v0 parameters randomized: gravity, power.

Typical usage::

    family = make_mountaincar_family(n_envs=10, seed=42)
    family_c = make_mountaincar_continuous_family(n_envs=10, seed=42)
"""

from __future__ import annotations

from circ_rl.environments.env_family import EnvironmentFamily


def make_mountaincar_family(
    n_envs: int = 10,
    seed: int = 42,
    gravity_range: tuple[float, float] = (0.0015, 0.0040),
    force_range: tuple[float, float] = (0.0005, 0.0020),
) -> EnvironmentFamily:
    """Create a MountainCar-v0 environment family with randomized physics.

    Default ranges are chosen to produce meaningful variation while
    keeping the environment solvable:
    - gravity: 0.0015-0.0040 (default 0.0025)
    - force: 0.0005-0.0020 (default 0.001)

    :param n_envs: Number of environment instances.
    :param seed: Random seed for parameter sampling.
    :param gravity_range: (low, high) for gravitational acceleration scale.
    :param force_range: (low, high) for discrete action force.
    :returns: An EnvironmentFamily of varied MountainCar instances.
    """
    return EnvironmentFamily.from_gymnasium(
        base_env="MountainCar-v0",
        param_distributions={
            "gravity": gravity_range,
            "force": force_range,
        },
        n_envs=n_envs,
        seed=seed,
    )


def make_mountaincar_continuous_family(
    n_envs: int = 10,
    seed: int = 42,
    power_range: tuple[float, float] = (0.0008, 0.0030),
) -> EnvironmentFamily:
    """Create a MountainCarContinuous-v0 family with randomized physics.

    Default ranges are chosen to produce meaningful variation while
    keeping the environment solvable:
    - power: 0.0008-0.0030 (default 0.0015)

    Note: MountainCarContinuous-v0 hardcodes gravity (0.0025) in its
    step method, so only ``power`` can be varied parametrically.

    :param n_envs: Number of environment instances.
    :param seed: Random seed for parameter sampling.
    :param power_range: (low, high) for continuous action power.
    :returns: An EnvironmentFamily of varied MountainCarContinuous instances.
    """
    return EnvironmentFamily.from_gymnasium(
        base_env="MountainCarContinuous-v0",
        param_distributions={
            "power": power_range,
        },
        n_envs=n_envs,
        seed=seed,
    )
