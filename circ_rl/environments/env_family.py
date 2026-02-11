"""EnvironmentFamily: a collection of parametrically varied MDPs.

Implements Definition 2.1 from ``CIRC-RL_Framework.md`` Section 2.2.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from loguru import logger


class EnvironmentFamily:
    r"""A family of MDPs sharing state/action spaces and causal graph.

    An environment family :math:`\mathcal{E}` is a set of MDPs
    :math:`\{M_e = (\mathcal{S}, \mathcal{A}, P_e, R_e, \gamma) : e \in \mathcal{E}\}`
    that share the same state space, action space, and causal graph, but may
    differ in dynamics parameters or exogenous noise distributions.

    See ``CIRC-RL_Framework.md`` Section 2.2, Definition 2.1.

    :param env_factory: Callable that takes a parameter dict and returns a Gymnasium env.
    :param param_distributions: Mapping of parameter names to (low, high) uniform ranges.
    :param n_envs: Number of environment instances in the family.
    :param seed: Random seed for parameter sampling.
    """

    def __init__(
        self,
        env_factory: Any,
        param_distributions: dict[str, tuple[float, float]],
        n_envs: int,
        seed: int = 42,
    ) -> None:
        if n_envs < 1:
            raise ValueError(f"n_envs must be >= 1, got {n_envs}")

        self._env_factory = env_factory
        self._param_distributions = param_distributions
        self._n_envs = n_envs
        self._rng = np.random.RandomState(seed)

        self._env_params: list[dict[str, float]] = []
        for _ in range(n_envs):
            params: dict[str, float] = {}
            for name, (low, high) in param_distributions.items():
                params[name] = float(self._rng.uniform(low, high))
            self._env_params.append(params)

        probe_env = self._env_factory(self._env_params[0])
        self._observation_space = probe_env.observation_space
        self._action_space = probe_env.action_space
        probe_env.close()

        logger.info(
            "EnvironmentFamily created: {} envs, params={}",
            n_envs,
            list(param_distributions.keys()),
        )

    @classmethod
    def from_gymnasium(
        cls,
        base_env: str,
        param_distributions: dict[str, tuple[float, float]],
        n_envs: int,
        seed: int = 42,
    ) -> EnvironmentFamily:
        """Create an EnvironmentFamily from a Gymnasium environment ID.

        Parameters are applied by modifying the environment's internal
        attributes after construction. Only works for environments whose
        underlying model exposes modifiable attributes.

        :param base_env: Gymnasium environment ID (e.g., ``"CartPole-v1"``).
        :param param_distributions: Mapping of attribute names to (low, high) ranges.
        :param n_envs: Number of environment instances.
        :param seed: Random seed.
        :returns: A new EnvironmentFamily instance.
        """

        def factory(params: dict[str, float]) -> gym.Env[Any, Any]:
            env = gym.make(base_env)
            unwrapped = env.unwrapped
            for attr, value in params.items():
                if not hasattr(unwrapped, attr):
                    raise AttributeError(
                        f"Environment {base_env} does not have attribute '{attr}'. "
                        f"Available: {[a for a in dir(unwrapped) if not a.startswith('_')]}"
                    )
                setattr(unwrapped, attr, value)
            return env

        return cls(
            env_factory=factory,
            param_distributions=param_distributions,
            n_envs=n_envs,
            seed=seed,
        )

    def make_env(self, env_idx: int) -> gym.Env[Any, Any]:
        """Create a single environment instance by index.

        :param env_idx: Index into the family (0 to n_envs-1).
        :returns: A Gymnasium environment with the sampled parameters applied.
        :raises IndexError: If env_idx is out of range.
        """
        if env_idx < 0 or env_idx >= self._n_envs:
            raise IndexError(
                f"env_idx {env_idx} out of range [0, {self._n_envs})"
            )
        return self._env_factory(self._env_params[env_idx])

    def sample_envs(self, n: int) -> list[tuple[int, gym.Env[Any, Any]]]:
        """Sample n environment instances (with replacement).

        :param n: Number of environments to sample.
        :returns: List of (env_idx, env) tuples.
        """
        indices = self._rng.choice(self._n_envs, size=n, replace=True)
        return [(int(idx), self.make_env(int(idx))) for idx in indices]

    def get_env_params(self, env_idx: int) -> dict[str, float]:
        """Return the parameter dict for a specific environment instance.

        :param env_idx: Index into the family.
        :returns: Dictionary of parameter name to sampled value.
        :raises IndexError: If env_idx is out of range.
        """
        if env_idx < 0 or env_idx >= self._n_envs:
            raise IndexError(
                f"env_idx {env_idx} out of range [0, {self._n_envs})"
            )
        return dict(self._env_params[env_idx])

    @property
    def n_envs(self) -> int:
        """Number of environment instances in the family."""
        return self._n_envs

    @property
    def observation_space(self) -> gym.Space[Any]:
        """Shared observation space across all environments."""
        return self._observation_space

    @property
    def action_space(self) -> gym.Space[Any]:
        """Shared action space across all environments."""
        return self._action_space

    @property
    def param_names(self) -> list[str]:
        """Names of the varied parameters."""
        return list(self._param_distributions.keys())

    def __repr__(self) -> str:
        return (
            f"EnvironmentFamily(n_envs={self._n_envs}, "
            f"params={self.param_names})"
        )
