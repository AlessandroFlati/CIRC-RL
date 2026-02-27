"""Central CIRC-RL trainer combining all framework components.

Implements the composite objective from ``CIRC-RL_Framework.md`` Section 3.6,
Phase 3: policy optimization with causal features, IRM invariance,
complexity regularization, and Lagrangian constraint enforcement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
from loguru import logger

from circ_rl.constraints.const_lagrange import LagrangeMultiplierManager
from circ_rl.invariance.inv_irm_penalty import IRMPenalty
from circ_rl.invariance.inv_worst_case import WorstCaseOptimizer
from circ_rl.regularization.reg_composite import CompositeRegularizer
from circ_rl.training.rollout import RolloutWorker

if TYPE_CHECKING:
    from collections.abc import Callable

    from circ_rl.constraints.const_set import ConstraintSet
    from circ_rl.environments.env_family import EnvironmentFamily
    from circ_rl.policy.causal_policy import CausalPolicy
    from circ_rl.training.trajectory_buffer import MultiEnvTrajectoryBuffer
    from circ_rl.utils.logging import MetricsLogger


@dataclass
class TrainingConfig:
    """Configuration for the CIRC-RL trainer.

    :param n_iterations: Number of training iterations.
    :param gamma: Discount factor.
    :param gae_lambda: GAE lambda for advantage estimation.
    :param learning_rate: Learning rate for the policy optimizer.
    :param clip_epsilon: PPO-style clipping parameter.
    :param n_steps_per_env: Rollout steps per environment per iteration.
    :param n_ppo_epochs: Number of PPO update epochs per iteration.
    :param mini_batch_size: Mini-batch size for PPO updates.
    :param irm_weight: Weight for the IRM penalty.
    :param worst_case_temperature: Temperature for soft-min worst-case.
    :param worst_case_variance_weight: Weight for variance penalty.
    :param entropy_coef: Entropy bonus coefficient for exploration.
    :param lagrange_lr: Learning rate for Lagrange multiplier updates.
    :param dynamics_aux_weight: Weight for the auxiliary dynamics prediction
        loss. Only used when the policy has dynamics normalization enabled.
    """

    n_iterations: int = 100
    gamma: float = 0.99
    gae_lambda: float = 0.95
    learning_rate: float = 3e-4
    clip_epsilon: float = 0.2
    n_steps_per_env: int = 200
    n_ppo_epochs: int = 4
    mini_batch_size: int = 64
    irm_weight: float = 1.0
    worst_case_temperature: float = 1.0
    worst_case_variance_weight: float = 0.1
    entropy_coef: float = 0.0
    lagrange_lr: float = 0.01
    dynamics_aux_weight: float = 1.0


@dataclass
class IterationMetrics:
    """Metrics from a single training iteration.

    :param iteration: Iteration number.
    :param policy_loss: PPO policy loss.
    :param value_loss: Value function loss.
    :param irm_penalty: IRM invariance penalty.
    :param worst_case_loss: Worst-case return loss.
    :param regularization_total: Total regularization penalty.
    :param constraint_penalty: Lagrangian constraint penalty.
    :param total_loss: Sum of all loss components.
    :param mean_return: Mean return across environments.
    :param worst_env_return: Worst per-environment return.
    :param per_env_returns: Per-environment mean returns (one per env).
    """

    iteration: int
    policy_loss: float
    value_loss: float
    irm_penalty: float
    worst_case_loss: float
    regularization_total: float
    constraint_penalty: float
    total_loss: float
    mean_return: float
    worst_env_return: float
    per_env_returns: list[float] | None = None


class CIRCTrainer:
    """Central training loop for CIRC-RL.

    Combines:
    - PPO-style policy gradient for the base RL objective
    - IRM penalty for cross-environment invariance
    - Worst-case optimization for robustness
    - Composite regularization for simplicity
    - Lagrangian constraint enforcement for safety

    :param policy: The causal policy to train.
    :param env_family: The environment family.
    :param config: Training configuration.
    :param constraint_set: Optional constraint set.
    :param metrics_logger: Optional metrics logger.
    :param dynamics_scales: Per-environment dynamics scales from
        ``TransitionAnalyzer``. Maps ``env_idx -> D_e``. Required
        when the policy uses dynamics normalization.
    """

    def __init__(
        self,
        policy: CausalPolicy,
        env_family: EnvironmentFamily,
        config: TrainingConfig,
        constraint_set: ConstraintSet | None = None,
        metrics_logger: MetricsLogger | None = None,
        n_rollout_workers: int = 1,
        env_param_names: list[str] | None = None,
        dynamics_scales: dict[int, float] | None = None,
    ) -> None:
        self._policy = policy
        self._env_family = env_family
        self._dynamics_scales = dynamics_scales
        self._config = config
        self._logger = metrics_logger

        # Separate optimizers: the value head uses detached trunk features,
        # so its gradients must not be clipped together with the policy's
        # (otherwise the huge value loss gradient norm clips policy gradients
        # to essentially zero).
        self._policy_optimizer = torch.optim.Adam(
            policy.policy_parameters, lr=config.learning_rate
        )
        self._value_optimizer = torch.optim.Adam(
            policy.value_parameters, lr=config.learning_rate
        )
        self._irm = IRMPenalty(lambda_irm=config.irm_weight)
        self._worst_case = WorstCaseOptimizer(
            temperature=config.worst_case_temperature,
            variance_weight=config.worst_case_variance_weight,
        )
        self._regularizer = CompositeRegularizer()
        self._rollout_worker = RolloutWorker(
            env_family,
            n_steps_per_env=config.n_steps_per_env,
            gamma=config.gamma,
            n_workers=n_rollout_workers,
            env_param_names=env_param_names,
        )

        # Constraints
        if constraint_set is not None and constraint_set.n_constraints > 0:
            self._constraint_set = constraint_set
            self._lagrange = LagrangeMultiplierManager(
                constraint_set,
                learning_rate=config.lagrange_lr,
            )
        else:
            self._constraint_set = None
            self._lagrange = None

        self._device = torch.device("cpu")

    def to(self, device: torch.device) -> CIRCTrainer:
        """Move the trainer to a device.

        :param device: Target device.
        :returns: Self for chaining.
        """
        self._device = device
        self._policy.to(device)
        return self

    def run(
        self,
        iteration_callback: Callable[[int, IterationMetrics], None] | None = None,
    ) -> list[IterationMetrics]:
        """Run the full training loop.

        :param iteration_callback: Optional function called after each
            iteration with ``(iteration, metrics)``. Use for periodic
            checkpointing or custom logging.
        :returns: List of per-iteration metrics.
        """
        all_metrics: list[IterationMetrics] = []

        logger.info(
            "Starting CIRC-RL training: {} iterations, {} environments",
            self._config.n_iterations,
            self._env_family.n_envs,
        )

        for iteration in range(self._config.n_iterations):
            metrics = self._train_iteration(iteration)
            all_metrics.append(metrics)

            if self._logger is not None:
                self._log_metrics(metrics)

            logger.info(
                "Iteration {}/{}: loss={:.4f}, mean_return={:.2f}, "
                "worst_return={:.2f}",
                iteration + 1,
                self._config.n_iterations,
                metrics.total_loss,
                metrics.mean_return,
                metrics.worst_env_return,
            )

            if iteration_callback is not None:
                iteration_callback(iteration, metrics)

        logger.info(
            "Training complete. Final: loss={:.4f}, return={:.2f}",
            all_metrics[-1].total_loss,
            all_metrics[-1].mean_return,
        )
        return all_metrics

    def _train_iteration(self, iteration: int) -> IterationMetrics:
        """Execute a single training iteration.

        1. Collect rollouts from all environments.
        2. Compute per-environment returns for IRM and worst-case.
        3. Run PPO updates with composite loss.
        4. Update Lagrange multipliers.

        :param iteration: Current iteration number.
        :returns: Metrics for this iteration.
        """
        # Step 1: collect trajectories
        buffer = self._rollout_worker.collect(self._policy, device=self._device)

        # Step 2: compute per-environment returns
        env_returns = self._compute_env_returns(buffer)

        mean_return = float(np.mean(env_returns))
        worst_return = float(np.min(env_returns))

        # Step 3: PPO updates
        # Compute advantages per-trajectory to avoid cross-env GAE leakage
        returns = buffer.compute_all_returns(self._config.gamma)
        advantages = buffer.compute_all_advantages(
            self._config.gamma, self._config.gae_lambda
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        flat = buffer.get_all_flat()

        # Accumulate losses across PPO epochs
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_irm = 0.0
        total_wc = 0.0
        total_reg = 0.0
        total_constraint = 0.0
        total_loss_val = 0.0
        n_updates = 0

        for _ in range(self._config.n_ppo_epochs):
            indices = torch.randperm(flat.length)
            for start in range(0, flat.length, self._config.mini_batch_size):
                end = min(start + self._config.mini_batch_size, flat.length)
                mb_idx = indices[start:end]

                mb_states = flat.states[mb_idx]
                mb_actions = flat.actions[mb_idx]
                mb_old_log_probs = flat.log_probs[mb_idx]
                mb_advantages = advantages[mb_idx]
                mb_returns = returns[mb_idx]
                mb_context = (
                    flat.env_params[mb_idx]
                    if flat.env_params is not None
                    else None
                )

                # Forward pass
                output = self._policy.evaluate_actions(
                    mb_states, mb_actions, context=mb_context
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
                value_loss = nn.functional.mse_loss(output.value, mb_returns)

                # IRM penalty (skip expensive per-env evaluation when weight is 0)
                if self._config.irm_weight > 0:
                    env_loss_tensors = []
                    for env_id in buffer.env_ids:
                        env_trajs = buffer.get_env_trajectories(env_id)
                        if env_trajs:
                            env_states = torch.cat([t.states for t in env_trajs])
                            env_actions = torch.cat([t.actions for t in env_trajs])
                            env_context = (
                                torch.cat([t.env_params for t in env_trajs])
                                if env_trajs[0].env_params is not None
                                else None
                            )
                            env_out = self._policy.evaluate_actions(
                                env_states, env_actions, context=env_context
                            )
                            env_loss_tensors.append(-env_out.log_prob.mean())
                    irm_penalty = self._irm(env_loss_tensors)
                else:
                    irm_penalty = torch.tensor(0.0, device=self._device)

                # Worst-case loss
                env_returns_tensor = torch.tensor(
                    env_returns, dtype=torch.float32, device=self._device
                )
                wc_loss = self._worst_case(env_returns_tensor)

                # Regularization
                reg_loss, _ = self._regularizer(
                    self._policy,
                    output.entropy,
                    kl_divergence=output.kl_divergence,
                    log_prob=output.log_prob,
                    discrete_actions=not self._policy.continuous,
                )

                # Constraint penalty
                constraint_penalty = torch.tensor(0.0, device=self._device)
                if self._constraint_set is not None and self._lagrange is not None:
                    costs = self._constraint_set.evaluate_all(
                        mb_states,
                        mb_actions.unsqueeze(-1) if mb_actions.dim() == 1 else mb_actions,
                        mb_returns,
                        mb_states,  # Use states as proxy for next_states in mini-batch
                    )
                    constraint_penalty = self._lagrange.compute_lagrangian_penalty(costs)

                # Auxiliary dynamics prediction loss
                dynamics_aux_loss = torch.tensor(0.0, device=self._device)
                if (
                    self._dynamics_scales is not None
                    and self._policy._use_dynamics_norm
                    and mb_context is not None
                    and flat.flat_env_ids is not None
                ):
                    mb_env_ids = flat.flat_env_ids[mb_idx]  # (batch,) long
                    target_scales = torch.tensor(
                        [self._dynamics_scales[int(eid)] for eid in mb_env_ids],
                        dtype=torch.float32,
                        device=self._device,
                    )  # (batch,)
                    predicted_scale = self._policy._dynamics_predictor(
                        mb_context
                    ).squeeze(-1)  # (batch,)
                    dynamics_aux_loss = (
                        self._config.dynamics_aux_weight
                        * nn.functional.mse_loss(predicted_scale, target_scales)
                    )

                # Entropy bonus for exploration
                entropy_loss = -self._config.entropy_coef * output.entropy.mean()

                # Policy loss (everything except value)
                policy_total_loss = (
                    policy_loss
                    + entropy_loss
                    + irm_penalty
                    + 0.1 * wc_loss
                    + reg_loss
                    + constraint_penalty
                    + dynamics_aux_loss
                )

                # Value loss (separate to avoid gradient scale mismatch)
                value_total_loss = 0.5 * value_loss

                # Update policy (trunk + policy head)
                self._policy_optimizer.zero_grad()
                policy_total_loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(
                    self._policy.policy_parameters, max_norm=0.5
                )
                self._policy_optimizer.step()

                # Update value head
                self._value_optimizer.zero_grad()
                value_total_loss.backward()
                nn.utils.clip_grad_norm_(
                    self._policy.value_parameters, max_norm=0.5
                )
                self._value_optimizer.step()

                # Combined loss for metrics
                loss = policy_total_loss + value_total_loss

                total_policy_loss += float(policy_loss.item())
                total_value_loss += float(value_loss.item())
                total_irm += float(irm_penalty.item())
                total_wc += float(wc_loss.item())
                total_reg += float(reg_loss.item())
                total_constraint += float(constraint_penalty.item())
                total_loss_val += float(loss.item())
                n_updates += 1

        # Step 4: update Lagrange multipliers
        if self._lagrange is not None and self._constraint_set is not None:
            self._lagrange.dual_update(
                flat.states, flat.actions, flat.rewards, flat.next_states
            )

        n = max(n_updates, 1)
        return IterationMetrics(
            iteration=iteration,
            policy_loss=total_policy_loss / n,
            value_loss=total_value_loss / n,
            irm_penalty=total_irm / n,
            worst_case_loss=total_wc / n,
            regularization_total=total_reg / n,
            constraint_penalty=total_constraint / n,
            total_loss=total_loss_val / n,
            mean_return=mean_return,
            worst_env_return=worst_return,
            per_env_returns=env_returns,
        )

    def _compute_env_returns(
        self, buffer: MultiEnvTrajectoryBuffer
    ) -> list[float]:
        """Compute mean episode return per environment.

        Uses unadjusted episode returns (not the bootstrap-adjusted rewards
        used for GAE) so that logged metrics reflect actual performance.

        :param buffer: Trajectory buffer from rollout.
        :returns: List of per-environment mean returns.
        """
        env_returns = []
        for env_id in sorted(buffer.env_ids):
            trajs = buffer.get_env_trajectories(env_id)
            all_ep_returns: list[float] = []
            for t in trajs:
                all_ep_returns.extend(t.episode_returns)
            if all_ep_returns:
                env_returns.append(float(np.mean(all_ep_returns)))
            else:
                # Fallback: use raw reward sum if no completed episodes
                total_reward = sum(float(t.rewards.sum()) for t in trajs)
                env_returns.append(total_reward / max(len(trajs), 1))
        return env_returns

    def _log_metrics(self, metrics: IterationMetrics) -> None:
        """Log metrics to the MetricsLogger."""
        if self._logger is None:
            return

        step = metrics.iteration
        self._logger.log_scalar("loss/policy", metrics.policy_loss, step)
        self._logger.log_scalar("loss/value", metrics.value_loss, step)
        self._logger.log_scalar("loss/irm_penalty", metrics.irm_penalty, step)
        self._logger.log_scalar("loss/worst_case", metrics.worst_case_loss, step)
        self._logger.log_scalar("loss/regularization", metrics.regularization_total, step)
        self._logger.log_scalar("loss/constraint", metrics.constraint_penalty, step)
        self._logger.log_scalar("loss/total", metrics.total_loss, step)
        self._logger.log_scalar("return/mean", metrics.mean_return, step)
        self._logger.log_scalar("return/worst_env", metrics.worst_env_return, step)
