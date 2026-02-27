r"""Model Predictive Path Integral (MPPI) solver.

Sampling-based trajectory optimization that handles non-convex cost
landscapes without linearization. Particularly effective for swing-up
tasks and other problems where iLQR gets trapped in local minima.

Algorithm (Williams et al. 2017):

1. Sample :math:`K` action sequences around a mean trajectory.
2. Roll out all :math:`K` trajectories through the dynamics model.
3. Compute discounted cost for each trajectory.
4. Weight trajectories via softmax:
   :math:`w_k = \exp(-\frac{1}{\lambda} S_k) / \sum_j \exp(-\frac{1}{\lambda} S_j)`
5. Update the mean: :math:`\mu \leftarrow \sum_k w_k \cdot \epsilon_k`.
6. Repeat for ``n_iterations``.

Supports vectorized dynamics: when a batched dynamics function is
provided, all :math:`K` samples are evaluated simultaneously via numpy
broadcasting.

See ``docs/proposed_solutions.md`` Solution 1 for rationale.
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
class MPPIConfig:
    r"""Configuration for the MPPI solver.

    :param horizon: Planning horizon (number of timesteps).
    :param n_samples: Number of sampled action sequences :math:`K`.
    :param temperature: Softmax temperature :math:`\lambda`. Lower
        values concentrate weight on lower-cost trajectories; higher
        values explore more uniformly.
    :param noise_sigma: Standard deviation of action perturbation noise.
    :param n_iterations: Number of MPPI update iterations per plan call.
    :param gamma: Discount factor for the cost function.
    :param max_action: Maximum absolute action value (box constraint).
    :param colored_noise_beta: Exponent for colored noise generation.
        0.0 = white noise, 1.0 = pink noise, 2.0 = brown/red noise.
        Higher values produce smoother action sequences.
    :param replan_interval: Steps between replanning. ``None`` means
        plan once for the full horizon.
    :param adaptive_replan_threshold: State deviation threshold for
        triggering early replan. ``None`` disables.
    :param min_replan_interval: Minimum steps between replans.
    """

    horizon: int = 100
    n_samples: int = 256
    temperature: float = 1.0
    noise_sigma: float = 0.5
    n_iterations: int = 3
    gamma: float = 0.99
    max_action: float = 2.0
    colored_noise_beta: float = 1.0
    replan_interval: int | None = None
    adaptive_replan_threshold: float | None = None
    min_replan_interval: int = 3

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {self.horizon}")
        if self.n_samples < 1:
            raise ValueError(f"n_samples must be >= 1, got {self.n_samples}")
        if self.temperature <= 0:
            raise ValueError(
                f"temperature must be > 0, got {self.temperature}",
            )
        if self.noise_sigma <= 0:
            raise ValueError(
                f"noise_sigma must be > 0, got {self.noise_sigma}",
            )
        if self.n_iterations < 1:
            raise ValueError(
                f"n_iterations must be >= 1, got {self.n_iterations}",
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


class MPPISolver:
    r"""Model Predictive Path Integral solver.

    Samples :math:`K` noisy action sequences around a mean trajectory,
    rolls them out through the dynamics, computes trajectory costs,
    then updates the mean using softmax-weighted averaging.

    When a ``batched_dynamics_fn`` is provided, all :math:`K` rollouts
    are evaluated simultaneously via numpy broadcasting (no Python loop
    over samples).

    :param config: MPPI configuration.
    :param dynamics_fn: Scalar dynamics ``(state, action) -> next_state``
        where ``state`` has shape ``(S,)`` and ``action`` shape ``(A,)``.
    :param reward_fn: Scalar reward ``(state, action) -> float``.
    :param batched_dynamics_fn: Vectorized dynamics
        ``(states, actions) -> next_states`` where ``states`` has shape
        ``(K, S)`` and ``actions`` shape ``(K, A)``. If ``None``, falls
        back to looping over scalar ``dynamics_fn``.
    :param batched_reward_fn: Vectorized reward
        ``(states, actions) -> rewards`` where inputs have shape
        ``(K, dim)`` and output shape ``(K,)``. If ``None``, falls back
        to looping over scalar ``reward_fn``.
    """

    def __init__(
        self,
        config: MPPIConfig,
        dynamics_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
        reward_fn: Callable[[np.ndarray, np.ndarray], float],
        batched_dynamics_fn: (
            Callable[[np.ndarray, np.ndarray], np.ndarray] | None
        ) = None,
        batched_reward_fn: (
            Callable[[np.ndarray, np.ndarray], np.ndarray] | None
        ) = None,
    ) -> None:
        self._config = config
        self._dynamics_fn = dynamics_fn
        self._reward_fn = reward_fn
        self._batched_dynamics_fn = batched_dynamics_fn
        self._batched_reward_fn = batched_reward_fn
        self._rng = np.random.default_rng()

    @property
    def config(self) -> MPPIConfig:
        """The MPPI configuration."""
        return self._config

    def plan(
        self,
        initial_state: np.ndarray,
        action_dim: int,
        warm_start_actions: np.ndarray | None = None,
    ) -> ILQRSolution:
        r"""Plan using MPPI from the given initial state.

        Returns an ``ILQRSolution`` for compatibility with the existing
        ``_ILQRAnalyticPolicy`` wrapper. Feedback gains are zero matrices
        since MPPI is open-loop (relies on frequent replanning).

        :param initial_state: Shape ``(state_dim,)``.
        :param action_dim: Number of action dimensions.
        :param warm_start_actions: Optional shape ``(horizon, action_dim)``.
        :returns: Solution with nominal trajectory.
        """
        cfg = self._config
        H = cfg.horizon  # noqa: N806
        K = cfg.n_samples  # noqa: N806
        S = initial_state.shape[0]  # noqa: N806
        A = action_dim  # noqa: N806
        max_a = cfg.max_action

        # Initialize mean action sequence
        if warm_start_actions is not None:
            assert warm_start_actions.shape == (H, A), (
                f"Expected warm_start shape ({H}, {A}), "
                f"got {warm_start_actions.shape}"
            )
            mean_actions = warm_start_actions.copy()
        else:
            mean_actions = np.zeros((H, A))

        # Pre-allocate state buffer for batched rollout
        states_buf = np.empty((K, H + 1, S))

        # Pre-compute discount factors
        gammas = cfg.gamma ** np.arange(H)  # (H,)

        best_reward = -np.inf

        for iteration in range(cfg.n_iterations):
            # 1. Generate K colored-noise perturbations
            noise = self._generate_colored_noise(K, H, A)  # (K, H, A)

            # 2. Candidate action sequences: mean + noise
            actions = mean_actions[np.newaxis, :, :] + noise  # (K, H, A)
            np.clip(actions, -max_a, max_a, out=actions)

            # 3. Batched rollout
            self._batched_rollout(
                initial_state, actions, states_buf,
            )  # fills states_buf: (K, H+1, S)

            # 4. Compute discounted rewards for all K trajectories
            rewards = self._batched_total_reward(
                states_buf, actions, gammas,
            )  # (K,)

            # 5. MPPI weighting: softmax of reward / temperature
            #    We use reward (not cost) so higher is better
            beta = 1.0 / cfg.temperature
            shifted = rewards - rewards.max()  # numerical stability
            weights = np.exp(beta * shifted)  # (K,)
            weights_sum = weights.sum()
            if weights_sum < 1e-30:
                # All trajectories equally bad; uniform weights
                weights = np.ones(K) / K
            else:
                weights /= weights_sum

            # 6. Update mean: weighted average of action sequences
            mean_actions = np.einsum("k,kha->ha", weights, actions)
            np.clip(mean_actions, -max_a, max_a, out=mean_actions)

            iter_best = rewards.max()
            if iter_best > best_reward:
                best_reward = iter_best

        # Final rollout with optimized mean
        nominal_states = self._single_rollout(initial_state, mean_actions)
        total_reward = self._single_total_reward(
            nominal_states, mean_actions, gammas,
        )

        logger.info(
            "MPPI: {} iterations, K={}, reward={:.2f}",
            cfg.n_iterations,
            K,
            total_reward,
        )

        # Zero feedback gains (MPPI is open-loop; replanning handles
        # deviations)
        feedback_gains = [np.zeros((A, S)) for _ in range(H)]
        feedforward_gains = [np.zeros(A) for _ in range(H)]

        return ILQRSolution(
            nominal_states=nominal_states,
            nominal_actions=mean_actions,
            feedback_gains=feedback_gains,
            feedforward_gains=feedforward_gains,
            total_reward=total_reward,
            converged=True,
            n_iterations=cfg.n_iterations,
        )

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _generate_colored_noise(
        self,
        n_samples: int,
        horizon: int,
        action_dim: int,
    ) -> np.ndarray:
        r"""Generate temporally correlated noise for MPPI.

        Uses frequency-domain shaping: white noise spectrum is shaped
        as :math:`1/f^{\beta/2}` then inverse-FFT'd back to time domain.

        :returns: Noise of shape ``(K, H, A)``.
        """
        beta = self._config.colored_noise_beta
        sigma = self._config.noise_sigma
        rng = self._rng

        if beta == 0.0:
            return sigma * rng.standard_normal(
                (n_samples, horizon, action_dim),
            )

        # Frequency domain shaping
        n_freq = horizon // 2 + 1
        freqs = np.fft.rfftfreq(horizon, d=1.0)  # (n_freq,)
        freqs[0] = 1.0  # avoid division by zero at DC
        power = 1.0 / (freqs ** (beta / 2.0))  # (n_freq,)
        power[0] = 0.0  # zero DC component

        # Generate white noise in freq domain, shape, IFFT
        white_r = rng.standard_normal(
            (n_samples, action_dim, n_freq),
        )
        white_i = rng.standard_normal(
            (n_samples, action_dim, n_freq),
        )
        shaped = (white_r + 1j * white_i) * power[np.newaxis, np.newaxis, :]
        time_domain = np.fft.irfft(shaped, n=horizon, axis=2)  # (K, A, H)

        # Normalize to unit variance then scale by sigma
        std = time_domain.std(axis=2, keepdims=True)
        np.clip(std, 1e-8, None, out=std)
        time_domain /= std
        time_domain *= sigma

        # Transpose to (K, H, A)
        return np.transpose(time_domain, (0, 2, 1))

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
            # Fallback: loop over samples (slow)
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
        rewards = np.zeros(K)

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
        states = np.zeros((H + 1, S))
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
