"""CartPole benchmark environment family.

Creates a parametrically varied CartPole family for evaluating CIRC-RL.
Parameters randomized: gravity, cart mass (masscart), pole length (length).

Typical usage::

    family = make_cartpole_family(n_envs=10, seed=42)
    held_out = make_cartpole_family(n_envs=5, seed=999)
"""

from __future__ import annotations

from circ_rl.environments.env_family import EnvironmentFamily


def make_cartpole_family(
    n_envs: int = 10,
    seed: int = 42,
    gravity_range: tuple[float, float] = (7.0, 13.0),
    masscart_range: tuple[float, float] = (0.5, 2.0),
    length_range: tuple[float, float] = (0.3, 0.8),
) -> EnvironmentFamily:
    """Create a CartPole environment family with randomized physics.

    Default ranges are chosen to produce meaningful variation while
    keeping the environment solvable:
    - gravity: 7.0-13.0 (default 9.8)
    - masscart: 0.5-2.0 (default 1.0)
    - length: 0.3-0.8 (default 0.5, half the pole length)

    :param n_envs: Number of environment instances.
    :param seed: Random seed for parameter sampling.
    :param gravity_range: (low, high) for gravitational acceleration.
    :param masscart_range: (low, high) for cart mass.
    :param length_range: (low, high) for pole half-length.
    :returns: An EnvironmentFamily of varied CartPole instances.
    """
    return EnvironmentFamily.from_gymnasium(
        base_env="CartPole-v1",
        param_distributions={
            "gravity": gravity_range,
            "masscart": masscart_range,
            "length": length_range,
        },
        n_envs=n_envs,
        seed=seed,
    )


def make_cartpole_family_minimal(
    n_envs: int = 3,
    seed: int = 42,
) -> EnvironmentFamily:
    """Create a small CartPole family for quick testing.

    Only varies gravity with a narrow range.

    :param n_envs: Number of environment instances.
    :param seed: Random seed.
    :returns: An EnvironmentFamily.
    """
    return EnvironmentFamily.from_gymnasium(
        base_env="CartPole-v1",
        param_distributions={"gravity": (9.0, 11.0)},
        n_envs=n_envs,
        seed=seed,
    )
