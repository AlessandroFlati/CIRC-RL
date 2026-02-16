"""Rollout worker for collecting transitions under the composite policy.

Collects transitions using the analytic policy + residual correction,
storing both the composite actions and the analytic baseline needed
for residual training.

See ``CIRC-RL_Framework.md`` Section 3.7 (Phase 6: Bounded Residual
Learning).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from loguru import logger

from circ_rl.training.trajectory_buffer import (
    MultiEnvTrajectoryBuffer,
    Trajectory,
)

if TYPE_CHECKING:
    from circ_rl.analytic_policy.analytic_policy import AnalyticPolicy
    from circ_rl.environments.env_family import EnvironmentFamily
    from circ_rl.policy.residual_policy import ResidualPolicy


class ResidualRolloutWorker:
    """Collect trajectories from an environment family using composite policy.

    The composite policy is::

        action = analytic(state, env_idx) + residual(state, analytic_action)

    Unlike the standard :class:`RolloutWorker`, this worker:

    1. Computes analytic actions via the :class:`AnalyticPolicy` (per-env).
    2. Computes residual corrections via :class:`ResidualPolicy` (env-agnostic).
    3. Steps the environment with the composite action.
    4. Stores the raw (unbounded) residual output for PPO recomputation.

    :param env_family: The environment family to collect from.
    :param analytic_policy: The analytic policy (LQR/MPC).
    :param n_steps_per_env: Number of steps to collect per environment.
    :param gamma: Discount factor for bootstrap at truncation.
    """

    def __init__(
        self,
        env_family: EnvironmentFamily,
        analytic_policy: AnalyticPolicy,
        n_steps_per_env: int = 200,
        gamma: float = 0.99,
    ) -> None:
        self._env_family = env_family
        self._analytic = analytic_policy
        self._n_steps = n_steps_per_env
        self._gamma = gamma

    def collect(
        self,
        residual: ResidualPolicy,
        device: torch.device | None = None,
    ) -> MultiEnvTrajectoryBuffer:
        """Collect trajectories from all environments.

        :param residual: The residual correction policy.
        :param device: Torch device for tensor operations.
        :returns: Buffer with one trajectory per environment. The
            ``actions`` field contains the raw (unbounded) residual
            output (needed for PPO ``evaluate_actions``). The actual
            composite action applied to the env is not stored separately
            since it can be recomputed from analytic + bounded(raw).
        """
        if device is None:
            device = torch.device("cpu")

        buffer = MultiEnvTrajectoryBuffer()
        n_envs = self._env_family.n_envs

        for env_idx in range(n_envs):
            trajectory = self._collect_single_env(residual, env_idx, device)
            buffer.add(trajectory)

        logger.debug(
            "Residual rollout: {} transitions from {} environments",
            buffer.total_transitions(),
            buffer.n_trajectories,
        )
        return buffer

    def _collect_single_env(
        self,
        residual: ResidualPolicy,
        env_idx: int,
        device: torch.device,
    ) -> Trajectory:
        """Collect a trajectory from a single environment.

        :param residual: Residual correction policy.
        :param env_idx: Environment index.
        :param device: Torch device.
        :returns: Trajectory where ``actions`` stores the raw residual
            output (for PPO recomputation).
        """
        env = self._env_family.make_env(env_idx)

        states_list: list[np.ndarray] = []
        raw_actions_list: list[np.ndarray] = []
        rewards_list: list[float] = []
        log_probs_list: list[float] = []
        values_list: list[float] = []
        next_states_list: list[np.ndarray] = []
        dones_list: list[bool] = []
        episode_returns: list[float] = []
        current_episode_return = 0.0

        obs, _ = env.reset()

        for _ in range(self._n_steps):
            state_np = np.asarray(obs, dtype=np.float32)

            # Analytic action (no gradient)
            analytic_action = self._analytic.get_action(state_np, env_idx)

            # Residual correction
            state_t = torch.from_numpy(state_np).unsqueeze(0).to(device)
            analytic_t = (
                torch.from_numpy(analytic_action)
                .float().unsqueeze(0).to(device)
            )

            with torch.no_grad():
                res_out = residual(state_t, analytic_t)

            delta = res_out.delta_action.squeeze(0).cpu().numpy()  # (action_dim,)
            raw = res_out.raw_output.squeeze(0).cpu().numpy()  # (action_dim,)
            log_prob = float(res_out.log_prob.item())
            value = float(res_out.value.item())

            # Composite action
            composite_action = analytic_action + delta

            next_obs, reward, terminated, truncated, _ = env.step(composite_action)
            done = terminated or truncated

            adjusted_reward = float(reward)
            current_episode_return += float(reward)

            # Bootstrap truncated episodes
            if truncated and not terminated:
                next_state_np = np.asarray(next_obs, dtype=np.float32)
                next_analytic = self._analytic.get_action(next_state_np, env_idx)
                next_state_t = torch.from_numpy(next_state_np).unsqueeze(0).to(device)
                next_analytic_t = (
                    torch.from_numpy(next_analytic)
                    .float().unsqueeze(0).to(device)
                )
                with torch.no_grad():
                    next_res = residual(next_state_t, next_analytic_t)
                bootstrap_v = float(next_res.value.item())
                adjusted_reward += self._gamma * bootstrap_v

            states_list.append(state_np)
            raw_actions_list.append(raw)
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

        # Bootstrap value for last step if mid-episode
        last_bootstrap_value = 0.0
        if not dones_list[-1]:
            last_state_np = np.asarray(obs, dtype=np.float32)
            last_analytic = self._analytic.get_action(last_state_np, env_idx)
            last_state_t = torch.from_numpy(last_state_np).unsqueeze(0).to(device)
            last_analytic_t = (
                torch.from_numpy(last_analytic)
                .float().unsqueeze(0).to(device)
            )
            with torch.no_grad():
                last_res = residual(last_state_t, last_analytic_t)
            last_bootstrap_value = float(last_res.value.item())

        return Trajectory(
            states=torch.from_numpy(np.stack(states_list)).to(device),
            actions=torch.from_numpy(np.stack(raw_actions_list)).to(
                dtype=torch.float32, device=device,
            ),
            rewards=torch.tensor(rewards_list, dtype=torch.float32, device=device),
            log_probs=torch.tensor(log_probs_list, dtype=torch.float32, device=device),
            values=torch.tensor(values_list, dtype=torch.float32, device=device),
            next_states=torch.from_numpy(np.stack(next_states_list)).to(device),
            dones=torch.tensor(dones_list, dtype=torch.float32, device=device),
            env_id=env_idx,
            last_bootstrap_value=last_bootstrap_value,
            episode_returns=episode_returns,
        )
