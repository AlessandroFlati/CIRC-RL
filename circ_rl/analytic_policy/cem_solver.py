r"""Cross-Entropy Method (CEM) solver for discrete action spaces.

Sampling-based trajectory optimization over categorical action
distributions. Unlike MPPI (which treats actions as continuous
and requires post-hoc discretization), CEM works natively with
discrete action spaces by maintaining per-timestep categorical
probabilities.

Algorithm (Rubinstein & Kroese, 2004):

1. Initialize uniform categorical distribution over actions at
   each timestep.
2. For each iteration:
   a. Sample :math:`K` action sequences from the categorical
      distributions.
   b. Rollout trajectories through the dynamics model.
   c. Compute discounted rewards.
   d. Select elite trajectories (top fraction by reward).
   e. Update categorical probabilities from elite action
      frequencies with smoothing.
3. Return the best trajectory.

The key advantage over continuous MPPI + discretization: CEM uses
the exact same discrete action values that the dynamics model was
trained on (e.g., ``{0, 1, 2}`` for MountainCar), eliminating the
mismatch between planning and training action spaces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

from circ_rl.analytic_policy.ilqr_solver import ILQRSolution

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True)
class CEMConfig:
    r"""Configuration for the CEM solver.

    :param horizon: Planning horizon (number of timesteps).
    :param n_samples: Number of sampled action sequences :math:`K`.
    :param n_iterations: Number of CEM update iterations.
    :param n_actions: Number of discrete actions.
    :param elite_fraction: Fraction of top trajectories selected as
        elite for probability updates.
    :param gamma: Discount factor for the reward function.
    :param smoothing_alpha: Smoothing factor for probability updates.
        ``new_probs = alpha * elite_freq + (1 - alpha) * old_probs``.
        Lower values produce slower convergence with more exploration.
    :param replan_interval: Steps between replanning. ``None`` means
        plan once for the full horizon.
    :param adaptive_replan_threshold: State deviation threshold for
        triggering early replan. ``None`` disables.
    :param min_replan_interval: Minimum steps between replans.
    """

    horizon: int = 50
    n_samples: int = 256
    n_iterations: int = 5
    n_actions: int = 3
    elite_fraction: float = 0.2
    gamma: float = 0.99
    smoothing_alpha: float = 0.8
    replan_interval: int | None = None
    adaptive_replan_threshold: float | None = None
    min_replan_interval: int = 3

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {self.horizon}")
        if self.n_samples < 1:
            raise ValueError(f"n_samples must be >= 1, got {self.n_samples}")
        if self.n_iterations < 1:
            raise ValueError(
                f"n_iterations must be >= 1, got {self.n_iterations}",
            )
        if self.n_actions < 2:
            raise ValueError(
                f"n_actions must be >= 2, got {self.n_actions}",
            )
        if not (0.0 < self.elite_fraction <= 1.0):
            raise ValueError(
                f"elite_fraction must be in (0, 1], got {self.elite_fraction}",
            )
        if not (0.0 < self.smoothing_alpha <= 1.0):
            raise ValueError(
                f"smoothing_alpha must be in (0, 1], got {self.smoothing_alpha}",
            )
        if self.replan_interval is not None and self.replan_interval < 1:
            raise ValueError(
                f"replan_interval must be >= 1, got {self.replan_interval}",
            )
        if self.min_replan_interval < 1:
            raise ValueError(
                f"min_replan_interval must be >= 1, "
                f"got {self.min_replan_interval}",
            )
        if (
            self.adaptive_replan_threshold is not None
            and self.adaptive_replan_threshold <= 0
        ):
            raise ValueError(
                f"adaptive_replan_threshold must be > 0, "
                f"got {self.adaptive_replan_threshold}",
            )


class CEMSolver:
    r"""Cross-Entropy Method solver for discrete action optimization.

    Samples discrete action sequences from categorical distributions,
    evaluates them via dynamics rollout, and updates distributions
    toward high-reward trajectories.

    :param config: CEM configuration.
    :param dynamics_fn: Scalar dynamics ``(state, action) -> next_state``
        where ``state`` has shape ``(S,)`` and ``action`` shape ``(A,)``.
        Actions should be the discrete values (e.g., 0, 1, 2) that the
        dynamics model was trained on.
    :param reward_fn: Scalar reward ``(state, action) -> float``.
    :param action_values: The numeric action values fed to the dynamics
        model, indexed by discrete action index. For MountainCar:
        ``[0.0, 1.0, 2.0]``. For CartPole: ``[0.0, 1.0]``.
    :param discretization_values: The continuous values returned in
        ``nominal_actions`` for compatibility with the evaluation loop's
        ``_discretize_action()`` mapping. For MountainCar:
        ``[-1.0, 0.0, 1.0]``. For CartPole: ``[-1.0, 1.0]``.
    :param batched_dynamics_fn: Vectorized dynamics
        ``(states, actions) -> next_states`` where inputs have shape
        ``(K, dim)``. If ``None``, falls back to looping over scalar fn.
    :param batched_reward_fn: Vectorized reward
        ``(states, actions) -> rewards``. If ``None``, falls back to loop.
    """

    def __init__(
        self,
        config: CEMConfig,
        dynamics_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
        reward_fn: Callable[[np.ndarray, np.ndarray], float],
        action_values: list[float],
        discretization_values: list[float],
        batched_dynamics_fn: (
            Callable[[np.ndarray, np.ndarray], np.ndarray] | None
        ) = None,
        batched_reward_fn: (
            Callable[[np.ndarray, np.ndarray], np.ndarray] | None
        ) = None,
    ) -> None:
        if len(action_values) != config.n_actions:
            raise ValueError(
                f"action_values length ({len(action_values)}) must match "
                f"n_actions ({config.n_actions})",
            )
        if len(discretization_values) != config.n_actions:
            raise ValueError(
                f"discretization_values length ({len(discretization_values)}) "
                f"must match n_actions ({config.n_actions})",
            )
        self._config = config
        self._dynamics_fn = dynamics_fn
        self._reward_fn = reward_fn
        self._action_values = np.array(action_values, dtype=np.float64)
        self._discretization_values = np.array(
            discretization_values, dtype=np.float64,
        )
        self._batched_dynamics_fn = batched_dynamics_fn
        self._batched_reward_fn = batched_reward_fn
        self._rng = np.random.default_rng()

    @property
    def config(self) -> CEMConfig:
        """The CEM configuration."""
        return self._config

    def plan(
        self,
        initial_state: np.ndarray,
        action_dim: int,
        warm_start_actions: np.ndarray | None = None,
    ) -> ILQRSolution:
        r"""Plan using CEM from the given initial state.

        Returns an ``ILQRSolution`` for compatibility with
        ``_ILQRAnalyticPolicy``. ``nominal_actions`` contains
        ``discretization_values`` so the evaluation loop can
        round-trip through ``_discretize_action()``. Feedback
        gains are zero (open-loop; relies on replanning).

        :param initial_state: Shape ``(state_dim,)``.
        :param action_dim: Must be 1 for discrete environments.
        :param warm_start_actions: Ignored (CEM initializes from
            categorical distributions).
        :returns: Solution with nominal trajectory.
        """
        cfg = self._config
        H = cfg.horizon  # noqa: N806
        K = cfg.n_samples  # noqa: N806
        S = initial_state.shape[0]  # noqa: N806
        n_act = cfg.n_actions
        n_elite = max(1, int(K * cfg.elite_fraction))

        # Initialize uniform categorical probabilities: (H, n_actions)
        probs = np.ones((H, n_act)) / n_act

        # Pre-compute discount factors
        gammas = cfg.gamma ** np.arange(H)  # (H,)

        best_reward = -np.inf
        best_actions_idx = np.zeros(H, dtype=np.int64)

        for _iteration in range(cfg.n_iterations):
            # 1. Sample K action sequences from categorical distributions
            action_indices = np.zeros((K, H), dtype=np.int64)  # (K, H)
            for t in range(H):
                action_indices[:, t] = self._rng.choice(
                    n_act, size=K, p=probs[t],
                )

            # 2. Convert to dynamics values: (K, H, 1)
            dynamics_actions = self._action_values[
                action_indices
            ]  # (K, H)
            dynamics_actions_3d = dynamics_actions[
                :, :, np.newaxis
            ]  # (K, H, 1)

            # 3. Batched rollout
            states_buf = np.empty((K, H + 1, S))  # (K, H+1, S)
            self._batched_rollout(
                initial_state, dynamics_actions_3d, states_buf,
            )

            # 4. Compute discounted rewards
            rewards = self._batched_total_reward(
                states_buf, dynamics_actions_3d, gammas,
            )  # (K,)

            # 5. Select elite trajectories
            elite_idx = np.argpartition(rewards, -n_elite)[-n_elite:]
            elite_actions = action_indices[elite_idx]  # (n_elite, H)

            # Track best
            iter_best_k = int(rewards.argmax())
            if rewards[iter_best_k] > best_reward:
                best_reward = rewards[iter_best_k]
                best_actions_idx = action_indices[iter_best_k].copy()

            # 6. Update categorical probabilities from elite frequencies
            alpha = cfg.smoothing_alpha
            for t in range(H):
                counts = np.bincount(
                    elite_actions[:, t], minlength=n_act,
                ).astype(np.float64)  # (n_act,)
                elite_probs = counts / n_elite  # (n_act,)
                probs[t] = alpha * elite_probs + (1 - alpha) * probs[t]
                # Normalize
                probs[t] /= probs[t].sum()

        # Build nominal trajectory from best action sequence
        best_dynamics_actions = self._action_values[
            best_actions_idx
        ]  # (H,)
        best_dynamics_actions_2d = best_dynamics_actions[
            :, np.newaxis
        ]  # (H, 1)

        nominal_states = self._single_rollout(
            initial_state, best_dynamics_actions_2d,
        )  # (H+1, S)
        total_reward = self._single_total_reward(
            nominal_states, best_dynamics_actions_2d, gammas,
        )

        # Store discretization values for evaluation loop compatibility
        discretization_actions = self._discretization_values[
            best_actions_idx
        ]  # (H,)
        nominal_actions = discretization_actions[:, np.newaxis]  # (H, 1)

        # Zero feedback gains (open-loop; relies on replanning)
        feedback_gains = [np.zeros((1, S)) for _ in range(H)]
        feedforward_gains = [np.zeros(1) for _ in range(H)]

        logger.info(
            "CEM: {} iterations, K={}, reward={:.2f}",
            cfg.n_iterations, K, total_reward,
        )

        return ILQRSolution(
            nominal_states=nominal_states,
            nominal_actions=nominal_actions,
            feedback_gains=feedback_gains,
            feedforward_gains=feedforward_gains,
            total_reward=total_reward,
            converged=True,
            n_iterations=cfg.n_iterations,
        )

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _batched_rollout(
        self,
        initial_state: np.ndarray,
        actions: np.ndarray,
        out: np.ndarray,
    ) -> None:
        r"""Roll out K trajectories simultaneously (in-place).

        :param initial_state: Shape ``(S,)``.
        :param actions: Shape ``(K, H, A)``.
        :param out: Pre-allocated buffer ``(K, H+1, S)``, filled in-place.
        """
        K, H, _A = actions.shape  # noqa: N806
        out[:, 0] = initial_state[np.newaxis, :]  # broadcast (S,) -> (K, S)

        if self._batched_dynamics_fn is not None:
            for t in range(H):
                out[:, t + 1] = self._batched_dynamics_fn(
                    out[:, t], actions[:, t],
                )
        else:
            for t in range(H):
                for k in range(K):
                    out[k, t + 1] = self._dynamics_fn(
                        out[k, t], actions[k, t],
                    )

    def _batched_total_reward(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        gammas: np.ndarray,
    ) -> np.ndarray:
        r"""Compute discounted total reward for K trajectories.

        :param states: Shape ``(K, H+1, S)``.
        :param actions: Shape ``(K, H, A)``.
        :param gammas: Discount factors ``(H,)``.
        :returns: Total rewards ``(K,)``.
        """
        K, H, _A = actions.shape  # noqa: N806
        rewards = np.zeros(K)  # (K,)

        if self._batched_reward_fn is not None:
            for t in range(H):
                r_t = self._batched_reward_fn(
                    states[:, t], actions[:, t],
                )  # (K,)
                rewards += gammas[t] * r_t
        else:
            for t in range(H):
                for k in range(K):
                    rewards[k] += gammas[t] * self._reward_fn(
                        states[k, t], actions[k, t],
                    )
        return rewards

    def _single_rollout(
        self,
        initial_state: np.ndarray,
        actions: np.ndarray,
    ) -> np.ndarray:
        r"""Roll out a single trajectory.

        :param initial_state: Shape ``(S,)``.
        :param actions: Shape ``(H, A)``.
        :returns: States ``(H+1, S)``.
        """
        H, _A = actions.shape  # noqa: N806
        S = initial_state.shape[0]  # noqa: N806
        states = np.zeros((H + 1, S))  # (H+1, S)
        states[0] = initial_state
        for t in range(H):
            states[t + 1] = self._dynamics_fn(states[t], actions[t])
        return states

    def _single_total_reward(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        gammas: np.ndarray,
    ) -> float:
        r"""Compute discounted total reward for a single trajectory.

        :param states: Shape ``(H+1, S)``.
        :param actions: Shape ``(H, A)``.
        :param gammas: Discount factors ``(H,)``.
        :returns: Scalar total reward.
        """
        total = 0.0
        H = actions.shape[0]  # noqa: N806
        for t in range(H):
            total += gammas[t] * self._reward_fn(states[t], actions[t])
        return float(total)
