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
    :param gamma: Discount factor, used to bootstrap truncated episode rewards.
    :param env_param_names: When set, include these environment parameter values
        in each trajectory for context-conditional policies.
    """

    def __init__(
        self,
        env_family: EnvironmentFamily,
        n_steps_per_env: int = 200,
        gamma: float = 0.99,
        env_param_names: list[str] | None = None,
    ) -> None:
        self._env_family = env_family
        self._n_steps = n_steps_per_env
        self._gamma = gamma
        self._env_param_names = env_param_names

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

        is_continuous = policy.continuous

        states_list: list[np.ndarray] = []
        actions_list: list[int | np.ndarray] = []
        rewards_list: list[float] = []
        log_probs_list: list[float] = []
        values_list: list[float] = []
        next_states_list: list[np.ndarray] = []
        dones_list: list[bool] = []
        # Track original (unadjusted) episode returns for metrics
        episode_returns: list[float] = []
        current_episode_return = 0.0

        # Build context tensor once for this env (constant per environment)
        context_tensor: torch.Tensor | None = None
        if self._env_param_names:
            params = self._env_family.get_env_params(env_idx)
            context_tensor = torch.tensor(
                [params[name] for name in self._env_param_names],
                dtype=torch.float32,
            ).unsqueeze(0).to(device)  # (1, n_env_params)

        obs, _ = env.reset()

        for _ in range(self._n_steps):
            state_tensor = torch.from_numpy(
                np.asarray(obs, dtype=np.float32)
            ).unsqueeze(0).to(device)

            with torch.no_grad():
                output = policy(state_tensor, context=context_tensor)

            if is_continuous:
                action = output.action.squeeze(0).cpu().numpy()  # (action_dim,)
            else:
                action = int(output.action.item())
            log_prob = float(output.log_prob.item())
            value = float(output.value.item())

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            adjusted_reward = float(reward)
            current_episode_return += float(reward)

            # Bootstrap truncated episodes: add gamma * V(s') to reward
            # so that GAE correctly accounts for future returns beyond
            # the truncation boundary.
            if truncated and not terminated:
                next_state_tensor = torch.from_numpy(
                    np.asarray(next_obs, dtype=np.float32)
                ).unsqueeze(0).to(device)
                with torch.no_grad():
                    next_output = policy(next_state_tensor, context=context_tensor)
                bootstrap_v = float(next_output.value.item())
                adjusted_reward += self._gamma * bootstrap_v

            states_list.append(np.asarray(obs, dtype=np.float32))
            actions_list.append(action)
            rewards_list.append(adjusted_reward)
            log_probs_list.append(log_prob)
            values_list.append(value)
            next_states_list.append(np.asarray(next_obs, dtype=np.float32))
            dones_list.append(done)

            if done:
                episode_returns.append(current_episode_return)
                current_episode_return = 0.0
                obs, _ = env.reset()
            else:
                obs = next_obs

        env.close()

        # Compute bootstrap value for last step if trajectory ends mid-episode
        last_bootstrap_value = 0.0
        if not dones_list[-1]:
            last_state_tensor = torch.from_numpy(
                np.asarray(obs, dtype=np.float32)
            ).unsqueeze(0).to(device)
            with torch.no_grad():
                last_output = policy(last_state_tensor, context=context_tensor)
            last_bootstrap_value = float(last_output.value.item())

        if is_continuous:
            actions_tensor = torch.from_numpy(
                np.stack(actions_list)  # type: ignore[arg-type]
            ).to(dtype=torch.float32, device=device)
        else:
            actions_tensor = torch.tensor(
                actions_list, dtype=torch.long, device=device
            )

        # Build env_params tensor if needed
        env_params_tensor: torch.Tensor | None = None
        if self._env_param_names:
            params = self._env_family.get_env_params(env_idx)
            param_vec = torch.tensor(
                [params[name] for name in self._env_param_names],
                dtype=torch.float32,
            )  # (n_env_params,)
            env_params_tensor = param_vec.unsqueeze(0).expand(
                self._n_steps, -1
            ).to(device)  # (T, n_env_params)

        return Trajectory(
            states=torch.from_numpy(np.stack(states_list)).to(device),
            actions=actions_tensor,
            rewards=torch.tensor(rewards_list, dtype=torch.float32, device=device),
            log_probs=torch.tensor(log_probs_list, dtype=torch.float32, device=device),
            values=torch.tensor(values_list, dtype=torch.float32, device=device),
            next_states=torch.from_numpy(np.stack(next_states_list)).to(device),
            dones=torch.tensor(dones_list, dtype=torch.float32, device=device),
            env_id=env_idx,
            env_params=env_params_tensor,
            last_bootstrap_value=last_bootstrap_value,
            episode_returns=episode_returns,
        )
