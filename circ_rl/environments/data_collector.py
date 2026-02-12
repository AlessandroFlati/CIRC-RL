"""Data collection from environment families for causal discovery."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from circ_rl.environments.env_family import EnvironmentFamily


@dataclass(frozen=True)
class ExploratoryDataset:
    """Transition data collected from multiple environments for causal discovery.

    :param states: State observations, shape ``(N, state_dim)``.
    :param actions: Actions taken, shape ``(N,)`` for discrete or ``(N, action_dim)`` for continuous.
    :param next_states: Next state observations, shape ``(N, state_dim)``.
    :param rewards: Rewards received, shape ``(N,)``.
    :param env_ids: Environment index for each transition, shape ``(N,)``.
    :param env_params: Environment parameters per transition, shape ``(N, n_env_params)``.
        Each row contains the parameter values of the environment that generated
        that transition. None when env-param discovery is disabled.
    """

    states: np.ndarray
    actions: np.ndarray
    next_states: np.ndarray
    rewards: np.ndarray
    env_ids: np.ndarray
    env_params: np.ndarray | None = None

    @property
    def n_transitions(self) -> int:
        """Total number of transitions across all environments."""
        return int(self.states.shape[0])

    @property
    def n_environments(self) -> int:
        """Number of unique environments in the dataset."""
        return int(np.unique(self.env_ids).shape[0])

    @property
    def state_dim(self) -> int:
        """Dimensionality of the state space."""
        return int(self.states.shape[1])

    @property
    def n_env_params(self) -> int:
        """Number of environment parameters (0 if not tracked)."""
        if self.env_params is None:
            return 0
        return int(self.env_params.shape[1])

    def get_env_data(self, env_id: int) -> ExploratoryDataset:
        """Return a subset of data from a specific environment.

        :param env_id: Environment index to filter by.
        :returns: A new ExploratoryDataset with only transitions from env_id.
        """
        mask = self.env_ids == env_id
        return ExploratoryDataset(
            states=self.states[mask],
            actions=self.actions[mask],
            next_states=self.next_states[mask],
            rewards=self.rewards[mask],
            env_ids=self.env_ids[mask],
            env_params=self.env_params[mask] if self.env_params is not None else None,
        )

    def to_flat_array(self) -> np.ndarray:
        """Concatenate states, actions, rewards, next_states into a flat array.

        Useful for causal discovery algorithms that operate on a single data matrix.
        Actions are reshaped to 2D if needed.

        :returns: Array of shape ``(N, state_dim + action_dim + 1 + state_dim)``.
        """
        actions_2d = self.actions if self.actions.ndim == 2 else self.actions[:, np.newaxis]
        return np.hstack([
            self.states,
            actions_2d,
            self.rewards[:, np.newaxis],
            self.next_states,
        ])

    def to_flat_array_with_env_params(self) -> np.ndarray:
        """Concatenate states, actions, rewards, next_states, and env_params.

        Like :meth:`to_flat_array` but appends environment parameter columns
        for causal discovery that includes env-param nodes.

        :returns: Array of shape
            ``(N, state_dim + action_dim + 1 + state_dim + n_env_params)``.
        :raises ValueError: If env_params is None.
        """
        if self.env_params is None:
            raise ValueError(
                "Cannot build flat array with env params: env_params is None. "
                "Collect data with include_env_params=True."
            )
        base = self.to_flat_array()  # (N, state_dim + action_dim + 1 + state_dim)
        return np.hstack([base, self.env_params])  # (N, ... + n_env_params)


class DataCollector:
    """Collect exploratory transition data from an EnvironmentFamily.

    Uses a random policy to gather diverse transitions for causal discovery.

    :param env_family: The environment family to collect data from.
    :param include_env_params: If True, record environment parameter values
        alongside each transition for env-param causal discovery.
    """

    def __init__(
        self,
        env_family: EnvironmentFamily,
        include_env_params: bool = False,
    ) -> None:
        self._env_family = env_family
        self._include_env_params = include_env_params

    def collect(
        self,
        n_transitions_per_env: int,
        seed: int = 42,
    ) -> ExploratoryDataset:
        """Collect transitions from all environments in the family.

        :param n_transitions_per_env: Number of transitions to collect per environment.
        :param seed: Random seed for action sampling.
        :returns: ExploratoryDataset with data from all environments.
        :raises ValueError: If n_transitions_per_env < 1.
        """
        if n_transitions_per_env < 1:
            raise ValueError(
                f"n_transitions_per_env must be >= 1, got {n_transitions_per_env}"
            )

        all_states: list[np.ndarray] = []
        all_actions: list[np.ndarray] = []
        all_next_states: list[np.ndarray] = []
        all_rewards: list[np.ndarray] = []
        all_env_ids: list[np.ndarray] = []
        all_env_params: list[np.ndarray] = []

        for env_idx in range(self._env_family.n_envs):
            env = self._env_family.make_env(env_idx)
            rng = np.random.RandomState(seed + env_idx)

            states, actions, next_states, rewards = self._collect_from_env(
                env, n_transitions_per_env, rng
            )

            all_states.append(states)
            all_actions.append(actions)
            all_next_states.append(next_states)
            all_rewards.append(rewards)
            all_env_ids.append(
                np.full(states.shape[0], env_idx, dtype=np.int32)
            )

            if self._include_env_params:
                params = self._env_family.get_env_params(env_idx)
                param_values = np.array(
                    [params[name] for name in self._env_family.param_names],
                    dtype=np.float32,
                )  # (n_env_params,)
                all_env_params.append(
                    np.tile(param_values, (states.shape[0], 1))
                )  # (n_transitions, n_env_params)

            env.close()

        env_params_array: np.ndarray | None = None
        if self._include_env_params and all_env_params:
            env_params_array = np.concatenate(all_env_params)

        dataset = ExploratoryDataset(
            states=np.concatenate(all_states),
            actions=np.concatenate(all_actions),
            next_states=np.concatenate(all_next_states),
            rewards=np.concatenate(all_rewards),
            env_ids=np.concatenate(all_env_ids),
            env_params=env_params_array,
        )

        logger.info(
            "Collected {} transitions from {} environments ({} per env)",
            dataset.n_transitions,
            self._env_family.n_envs,
            n_transitions_per_env,
        )
        return dataset

    @staticmethod
    def _collect_from_env(
        env: object,
        n_transitions: int,
        rng: np.random.RandomState,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Collect transitions from a single environment using random actions.

        :param env: A Gymnasium environment.
        :param n_transitions: Number of transitions to collect.
        :param rng: Random state for action sampling.
        :returns: Tuple of (states, actions, next_states, rewards).
        """
        import gymnasium as gym

        assert isinstance(env, gym.Env)

        states_list: list[np.ndarray] = []
        actions_list: list[np.ndarray] = []
        next_states_list: list[np.ndarray] = []
        rewards_list: list[float] = []

        obs, _ = env.reset(seed=int(rng.randint(0, 2**31)))
        collected = 0

        while collected < n_transitions:
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = env.step(action)

            states_list.append(np.asarray(obs, dtype=np.float32))
            actions_list.append(np.asarray(action))
            next_states_list.append(np.asarray(next_obs, dtype=np.float32))
            rewards_list.append(float(reward))

            collected += 1

            if terminated or truncated:
                obs, _ = env.reset()
            else:
                obs = next_obs

        return (
            np.stack(states_list),
            np.stack(actions_list),
            np.stack(next_states_list),
            np.array(rewards_list, dtype=np.float32),
        )
