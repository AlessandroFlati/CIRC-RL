"""Rollout worker for collecting trajectories from environments.

Collects transitions using the current policy across multiple environments,
storing them in the trajectory buffer for training.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import gymnasium as gym
import numpy as np
import torch
from loguru import logger

from circ_rl.training.trajectory_buffer import (
    MultiEnvTrajectoryBuffer,
    Trajectory,
)

if TYPE_CHECKING:
    from circ_rl.environments.env_family import EnvironmentFamily
    from circ_rl.policy.causal_policy import CausalPolicy


class RolloutWorker:
    """Collect trajectories from an environment family using a policy.

    :param env_family: The environment family to collect from.
    :param n_steps_per_env: Number of steps to collect per environment per rollout.
    """

    def __init__(
        self,
        env_family: EnvironmentFamily,
        n_steps_per_env: int = 200,
    ) -> None:
        self._env_family = env_family
        self._n_steps = n_steps_per_env

    def collect(
        self,
        policy: CausalPolicy,
        device: torch.device = torch.device("cpu"),
    ) -> MultiEnvTrajectoryBuffer:
        """Collect trajectories from all environments.

        :param policy: The current policy to use for action selection.
        :param device: Torch device for tensor operations.
        :returns: Buffer with one trajectory per environment.
        """
        buffer = MultiEnvTrajectoryBuffer()

        for env_idx in range(self._env_family.n_envs):
            trajectory = self._collect_single_env(policy, env_idx, device)
            buffer.add(trajectory)

        logger.debug(
            "Collected {} transitions from {} environments",
            buffer.total_transitions(),
            buffer.n_trajectories,
        )
        return buffer

    def _collect_single_env(
        self,
        policy: CausalPolicy,
        env_idx: int,
        device: torch.device,
    ) -> Trajectory:
        """Collect a trajectory from a single environment.

        :param policy: Current policy.
        :param env_idx: Environment index.
        :param device: Torch device.
        :returns: Trajectory from this environment.
        """
        env = self._env_family.make_env(env_idx)

        states_list: list[np.ndarray] = []
        actions_list: list[int] = []
        rewards_list: list[float] = []
        log_probs_list: list[float] = []
        values_list: list[float] = []
        next_states_list: list[np.ndarray] = []
        dones_list: list[bool] = []

        obs, _ = env.reset()

        for _ in range(self._n_steps):
            state_tensor = torch.from_numpy(
                np.asarray(obs, dtype=np.float32)
            ).unsqueeze(0).to(device)

            with torch.no_grad():
                output = policy(state_tensor)

            action = int(output.action.item())
            log_prob = float(output.log_prob.item())
            value = float(output.value.item())

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            states_list.append(np.asarray(obs, dtype=np.float32))
            actions_list.append(action)
            rewards_list.append(float(reward))
            log_probs_list.append(log_prob)
            values_list.append(value)
            next_states_list.append(np.asarray(next_obs, dtype=np.float32))
            dones_list.append(done)

            if done:
                obs, _ = env.reset()
            else:
                obs = next_obs

        env.close()

        return Trajectory(
            states=torch.from_numpy(np.stack(states_list)).to(device),
            actions=torch.tensor(actions_list, dtype=torch.long, device=device),
            rewards=torch.tensor(rewards_list, dtype=torch.float32, device=device),
            log_probs=torch.tensor(log_probs_list, dtype=torch.float32, device=device),
            values=torch.tensor(values_list, dtype=torch.float32, device=device),
            next_states=torch.from_numpy(np.stack(next_states_list)).to(device),
            dones=torch.tensor(dones_list, dtype=torch.float32, device=device),
            env_id=env_idx,
        )
