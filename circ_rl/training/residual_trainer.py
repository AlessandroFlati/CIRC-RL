"""Residual policy trainer using PPO.

Trains the bounded residual correction network while keeping the
analytic policy frozen. Includes early-stop logic based on explained
variance.

See ``CIRC-RL_Framework.md`` Section 3.7 (Phase 6: Bounded Residual
Learning).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
from loguru import logger

from circ_rl.training.residual_rollout import ResidualRolloutWorker

if TYPE_CHECKING:
    from collections.abc import Callable

    from circ_rl.analytic_policy.analytic_policy import AnalyticPolicy
    from circ_rl.environments.env_family import EnvironmentFamily
    from circ_rl.policy.residual_policy import ResidualPolicy
    from circ_rl.training.trajectory_buffer import MultiEnvTrajectoryBuffer


@dataclass(frozen=True)
class ResidualTrainingConfig:
    r"""Configuration for residual policy training.

    :param n_iterations: Number of PPO training iterations.
    :param gamma: Discount factor.
    :param gae_lambda: GAE lambda.
    :param learning_rate: Learning rate for the residual network.
    :param clip_epsilon: PPO clipping parameter.
    :param n_steps_per_env: Rollout steps per environment per iteration.
    :param n_ppo_epochs: PPO epochs per iteration.
    :param mini_batch_size: Mini-batch size for PPO updates.
    :param entropy_coef: Entropy bonus coefficient.
    :param eta_max: Maximum correction fraction (passed to ResidualPolicy).
    :param explained_variance: :math:`\eta^2` from Phase 4 diagnostics.
    :param skip_if_eta2_above: Skip residual training entirely if
        explained variance exceeds this threshold. Default 0.98.
    :param abort_if_eta2_below: Abort residual training if explained
        variance is below this threshold (analytic policy too poor
        for residual to fix). Default 0.70.
    """

    n_iterations: int = 50
    gamma: float = 0.99
    gae_lambda: float = 0.95
    learning_rate: float = 3e-4
    clip_epsilon: float = 0.2
    n_steps_per_env: int = 200
    n_ppo_epochs: int = 4
    mini_batch_size: int = 64
    entropy_coef: float = 0.01
    eta_max: float = 0.1
    explained_variance: float = 0.90
    skip_if_eta2_above: float = 0.98
    abort_if_eta2_below: float = 0.70


@dataclass
class ResidualIterationMetrics:
    """Metrics from a single residual training iteration.

    :param iteration: Iteration number.
    :param policy_loss: PPO clipped policy loss.
    :param value_loss: Value function loss.
    :param entropy: Mean policy entropy.
    :param total_loss: Combined loss.
    :param mean_return: Mean return across environments.
    :param worst_env_return: Worst per-environment return.
    """

    iteration: int
    policy_loss: float
    value_loss: float
    entropy: float
    total_loss: float
    mean_return: float
    worst_env_return: float


class ResidualTrainer:
    """PPO trainer for the bounded residual correction.

    Keeps the analytic policy frozen and only trains the residual
    network. Supports early-skip (high :math:`\\eta^2`) and early-abort
    (low :math:`\\eta^2`) based on explained variance thresholds.

    :param analytic_policy: Frozen analytic policy.
    :param residual_policy: Trainable residual correction network.
    :param env_family: Environment family for rollouts.
    :param config: Training configuration.
    """

    def __init__(
        self,
        analytic_policy: AnalyticPolicy,
        residual_policy: ResidualPolicy,
        env_family: EnvironmentFamily,
        config: ResidualTrainingConfig,
    ) -> None:
        self._analytic = analytic_policy
        self._residual = residual_policy
        self._env_family = env_family
        self._config = config

        self._optimizer = torch.optim.Adam(
            residual_policy.parameters(), lr=config.learning_rate,
        )
        self._rollout = ResidualRolloutWorker(
            env_family=env_family,
            analytic_policy=analytic_policy,
            n_steps_per_env=config.n_steps_per_env,
            gamma=config.gamma,
        )
        self._device = torch.device("cpu")

    def to(self, device: torch.device) -> ResidualTrainer:
        """Move trainer to a device.

        :param device: Target device.
        :returns: Self for chaining.
        """
        self._device = device
        self._residual.to(device)
        return self

    def should_skip(self) -> bool:
        r"""Check if residual training should be skipped.

        :returns: True if :math:`\eta^2` > ``skip_if_eta2_above``.
        """
        return self._config.explained_variance > self._config.skip_if_eta2_above

    def should_abort(self) -> bool:
        r"""Check if residual training should be aborted.

        :returns: True if :math:`\eta^2` < ``abort_if_eta2_below``.
        """
        return self._config.explained_variance < self._config.abort_if_eta2_below

    def run(
        self,
        iteration_callback: (
            Callable[[int, ResidualIterationMetrics], None] | None
        ) = None,
    ) -> list[ResidualIterationMetrics]:
        """Run residual PPO training.

        :param iteration_callback: Optional callback per iteration.
        :returns: List of per-iteration metrics. Empty if skipped.
        :raises ValueError: If explained variance is below abort threshold.
        """
        if self.should_skip():
            logger.info(
                "Residual training SKIPPED: explained_variance={:.4f} > "
                "skip_threshold={:.4f}",
                self._config.explained_variance,
                self._config.skip_if_eta2_above,
            )
            return []

        if self.should_abort():
            raise ValueError(
                f"Residual training ABORTED: explained_variance="
                f"{self._config.explained_variance:.4f} < abort_threshold="
                f"{self._config.abort_if_eta2_below:.4f}. "
                f"Analytic policy is too poor for residual correction."
            )

        logger.info(
            "Starting residual training: {} iterations, {} environments, "
            "eta_max={}, explained_variance={:.4f}",
            self._config.n_iterations,
            self._env_family.n_envs,
            self._config.eta_max,
            self._config.explained_variance,
        )

        all_metrics: list[ResidualIterationMetrics] = []

        for iteration in range(self._config.n_iterations):
            metrics = self._train_iteration(iteration)
            all_metrics.append(metrics)

            logger.info(
                "Residual iter {}/{}: loss={:.4f}, return={:.2f}, "
                "worst={:.2f}",
                iteration + 1,
                self._config.n_iterations,
                metrics.total_loss,
                metrics.mean_return,
                metrics.worst_env_return,
            )

            if iteration_callback is not None:
                iteration_callback(iteration, metrics)

        return all_metrics

    def _train_iteration(self, iteration: int) -> ResidualIterationMetrics:
        """Execute a single PPO iteration on the residual network.

        :param iteration: Current iteration number.
        :returns: Iteration metrics.
        """
        # Collect rollouts under composite policy
        buffer = self._rollout.collect(self._residual, device=self._device)

        # Compute per-env returns for metrics
        env_returns = self._compute_env_returns(buffer)
        mean_return = float(np.mean(env_returns))
        worst_return = float(np.min(env_returns))

        # Per-trajectory returns and advantages
        returns = buffer.compute_all_returns(self._config.gamma)
        advantages = buffer.compute_all_advantages(
            self._config.gamma, self._config.gae_lambda,
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        flat = buffer.get_all_flat()

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_loss_val = 0.0
        n_updates = 0

        for _ in range(self._config.n_ppo_epochs):
            indices = torch.randperm(flat.length)
            for start in range(0, flat.length, self._config.mini_batch_size):
                end = min(start + self._config.mini_batch_size, flat.length)
                mb_idx = indices[start:end]

                mb_states = flat.states[mb_idx]  # (mb, state_dim)
                mb_raw_actions = flat.actions[mb_idx]  # (mb, action_dim) raw residual
                mb_old_log_probs = flat.log_probs[mb_idx]  # (mb,)
                mb_advantages = advantages[mb_idx]  # (mb,)
                mb_returns = returns[mb_idx]  # (mb,)

                # Recompute analytic actions for this batch
                mb_analytic = self._recompute_analytic_actions(
                    mb_states, flat, mb_idx,
                )  # (mb, action_dim)

                # Evaluate residual policy at taken raw actions
                output = self._residual.evaluate_actions(
                    mb_states, mb_analytic, mb_raw_actions,
                )

                # PPO clipped objective
                ratio = (output.log_prob - mb_old_log_probs).exp()
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self._config.clip_epsilon,
                    1.0 + self._config.clip_epsilon,
                ) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = 0.5 * nn.functional.mse_loss(output.value, mb_returns)

                # Entropy bonus
                entropy = output.entropy.mean()
                entropy_loss = -self._config.entropy_coef * entropy

                loss = policy_loss + value_loss + entropy_loss

                self._optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self._residual.parameters(), max_norm=0.5)
                self._optimizer.step()

                total_policy_loss += float(policy_loss.item())
                total_value_loss += float(value_loss.item())
                total_entropy += float(entropy.item())
                total_loss_val += float(loss.item())
                n_updates += 1

        n = max(n_updates, 1)
        return ResidualIterationMetrics(
            iteration=iteration,
            policy_loss=total_policy_loss / n,
            value_loss=total_value_loss / n,
            entropy=total_entropy / n,
            total_loss=total_loss_val / n,
            mean_return=mean_return,
            worst_env_return=worst_return,
        )

    def _recompute_analytic_actions(
        self,
        states: torch.Tensor,
        flat: object,
        mb_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Recompute analytic actions for a mini-batch.

        Since the analytic policy is deterministic, we can recompute
        from state + env_idx.

        :param states: Mini-batch states, shape ``(mb, state_dim)``.
        :param flat: Flattened trajectory (for env_id lookup).
        :param mb_idx: Mini-batch indices into the flat trajectory.
        :returns: Analytic actions, shape ``(mb, action_dim)``.
        """
        from circ_rl.training.trajectory_buffer import Trajectory

        assert isinstance(flat, Trajectory)
        assert flat.flat_env_ids is not None

        env_ids = flat.flat_env_ids[mb_idx]  # (mb,)
        analytic_list: list[np.ndarray] = []

        for i in range(states.shape[0]):
            state_np = states[i].cpu().numpy()
            env_idx = int(env_ids[i].item())
            a = self._analytic.get_action(state_np, env_idx)
            analytic_list.append(a)

        return torch.from_numpy(np.stack(analytic_list)).to(
            dtype=torch.float32, device=states.device,
        )

    def _compute_env_returns(
        self, buffer: MultiEnvTrajectoryBuffer,
    ) -> list[float]:
        """Compute mean episode return per environment.

        :param buffer: Trajectory buffer from rollout.
        :returns: List of per-environment mean returns.
        """
        env_returns: list[float] = []
        for env_id in sorted(buffer.env_ids):
            trajs = buffer.get_env_trajectories(env_id)
            all_ep_returns: list[float] = []
            for t in trajs:
                all_ep_returns.extend(t.episode_returns)
            if all_ep_returns:
                env_returns.append(float(np.mean(all_ep_returns)))
            else:
                total_reward = sum(float(t.rewards.sum()) for t in trajs)
                env_returns.append(total_reward / max(len(trajs), 1))
        return env_returns
