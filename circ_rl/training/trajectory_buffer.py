"""Trajectory storage for multi-environment RL training.

Stores transitions grouped by environment for per-environment loss computation
(needed by IRM penalty and worst-case optimizer).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class Trajectory:
    """A single trajectory from one environment.

    :param states: States, shape ``(T, state_dim)``.
    :param actions: Actions, shape ``(T,)``.
    :param rewards: Rewards, shape ``(T,)``.
    :param log_probs: Log-probabilities under the policy, shape ``(T,)``.
    :param values: Value estimates, shape ``(T,)``.
    :param next_states: Next states, shape ``(T, state_dim)``.
    :param dones: Done flags, shape ``(T,)``.
    :param env_id: Which environment this trajectory came from.
    :param env_params: Environment parameters, shape ``(T, n_env_params)``.
        Each row is the same (constant per environment). None when
        env-param context is not used.
    :param last_bootstrap_value: V(s') for the last step when the trajectory
        ends mid-episode (not done). Used to bootstrap GAE correctly.
    :param episode_returns: Undiscounted sum of original rewards per completed
        episode in this trajectory. Used for metrics (not training).
    """

    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    log_probs: torch.Tensor
    values: torch.Tensor
    next_states: torch.Tensor
    dones: torch.Tensor
    env_id: int
    env_params: torch.Tensor | None = None
    last_bootstrap_value: float = 0.0
    episode_returns: list[float] = field(default_factory=list)

    @property
    def length(self) -> int:
        """Number of timesteps in this trajectory."""
        return int(self.states.shape[0])

    def compute_returns(self, gamma: float = 0.99) -> torch.Tensor:
        """Compute discounted returns.

        Bootstraps with ``last_bootstrap_value`` when the trajectory ends
        mid-episode (last step not done). Truncated episodes within the
        trajectory are handled by the rollout worker adjusting the reward
        at truncation points (adding ``gamma * V(s')``).

        :param gamma: Discount factor.
        :returns: Returns of shape ``(T,)``.
        """
        returns = torch.zeros_like(self.rewards)
        # Bootstrap with V(s') if the trajectory ends mid-episode
        running_return = self.last_bootstrap_value
        for t in reversed(range(self.length)):
            if self.dones[t]:
                running_return = 0.0
            running_return = float(self.rewards[t]) + gamma * running_return
            returns[t] = running_return
        return returns

    def compute_advantages(
        self, gamma: float = 0.99, gae_lambda: float = 0.95
    ) -> torch.Tensor:
        """Compute Generalized Advantage Estimation (GAE).

        Bootstraps with ``last_bootstrap_value`` when the trajectory ends
        mid-episode (last step not done). Truncated episodes within the
        trajectory are handled by the rollout worker adjusting the reward
        at truncation points (adding ``gamma * V(s')``).

        :param gamma: Discount factor.
        :param gae_lambda: GAE lambda for bias-variance trade-off.
        :returns: Advantages of shape ``(T,)``.
        """
        advantages = torch.zeros_like(self.rewards)
        gae = 0.0
        for t in reversed(range(self.length)):
            if t == self.length - 1:
                # Bootstrap with V(s') if trajectory ends mid-episode
                next_value = self.last_bootstrap_value
            else:
                next_value = float(self.values[t + 1])

            if self.dones[t]:
                next_value = 0.0
                gae = 0.0

            delta = float(self.rewards[t]) + gamma * next_value - float(self.values[t])
            gae = delta + gamma * gae_lambda * gae
            advantages[t] = gae

        return advantages


class MultiEnvTrajectoryBuffer:
    """Buffer storing trajectories grouped by environment.

    Supports per-environment retrieval for IRM and worst-case objectives.
    """

    def __init__(self) -> None:
        self._trajectories: list[Trajectory] = []

    def add(self, trajectory: Trajectory) -> None:
        """Add a trajectory to the buffer."""
        self._trajectories.append(trajectory)

    def clear(self) -> None:
        """Remove all stored trajectories."""
        self._trajectories.clear()

    @property
    def n_trajectories(self) -> int:
        """Total number of stored trajectories."""
        return len(self._trajectories)

    @property
    def env_ids(self) -> set[int]:
        """Set of unique environment IDs in the buffer."""
        return {t.env_id for t in self._trajectories}

    def get_env_trajectories(self, env_id: int) -> list[Trajectory]:
        """Get all trajectories from a specific environment.

        :param env_id: The environment ID.
        :returns: List of trajectories from that environment.
        """
        return [t for t in self._trajectories if t.env_id == env_id]

    def get_all_flat(self) -> Trajectory:
        """Flatten all trajectories into a single Trajectory.

        Concatenates all transitions (ignoring environment boundaries).

        :returns: A single Trajectory with all transitions.
        :raises ValueError: If the buffer is empty.
        """
        if not self._trajectories:
            raise ValueError("Buffer is empty")

        env_params: torch.Tensor | None = None
        if self._trajectories[0].env_params is not None:
            env_params = torch.cat([t.env_params for t in self._trajectories])  # type: ignore[misc]

        return Trajectory(
            states=torch.cat([t.states for t in self._trajectories]),
            actions=torch.cat([t.actions for t in self._trajectories]),
            rewards=torch.cat([t.rewards for t in self._trajectories]),
            log_probs=torch.cat([t.log_probs for t in self._trajectories]),
            values=torch.cat([t.values for t in self._trajectories]),
            next_states=torch.cat([t.next_states for t in self._trajectories]),
            dones=torch.cat([t.dones for t in self._trajectories]),
            env_id=-1,
            env_params=env_params,
        )

    def compute_all_advantages(
        self, gamma: float = 0.99, gae_lambda: float = 0.95
    ) -> torch.Tensor:
        """Compute GAE per-trajectory and concatenate.

        This avoids cross-trajectory GAE leakage that would occur if
        advantages were computed on the flattened buffer.

        :param gamma: Discount factor.
        :param gae_lambda: GAE lambda.
        :returns: Advantages of shape ``(total_transitions,)``.
        :raises ValueError: If the buffer is empty.
        """
        if not self._trajectories:
            raise ValueError("Buffer is empty")
        return torch.cat([
            t.compute_advantages(gamma, gae_lambda)
            for t in self._trajectories
        ])

    def compute_all_returns(self, gamma: float = 0.99) -> torch.Tensor:
        """Compute returns per-trajectory and concatenate.

        :param gamma: Discount factor.
        :returns: Returns of shape ``(total_transitions,)``.
        :raises ValueError: If the buffer is empty.
        """
        if not self._trajectories:
            raise ValueError("Buffer is empty")
        return torch.cat([
            t.compute_returns(gamma) for t in self._trajectories
        ])

    def total_transitions(self) -> int:
        """Total number of transitions across all trajectories."""
        return sum(t.length for t in self._trajectories)

    def __repr__(self) -> str:
        return (
            f"MultiEnvTrajectoryBuffer("
            f"n_trajectories={self.n_trajectories}, "
            f"n_envs={len(self.env_ids)}, "
            f"total_transitions={self.total_transitions()})"
        )
