"""Diagnostic: test pure PPO on Pendulum-v1 (no IRM/worst-case/regularization).

Runs a minimal CIRC-RL training with auxiliary losses disabled to verify
the core PPO loop works correctly.
"""

from __future__ import annotations

import sys

import numpy as np
import torch
from loguru import logger

from circ_rl.environments.env_family import EnvironmentFamily
from circ_rl.policy.causal_policy import CausalPolicy
from circ_rl.training.circ_trainer import CIRCTrainer, TrainingConfig


def main() -> None:
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    device = torch.device("cpu")

    # Single environment (no multi-env complexity)
    env_family = EnvironmentFamily.from_gymnasium(
        base_env="Pendulum-v1",
        param_distributions={
            "g": (9.81, 9.81),  # Fixed gravity
            "m": (1.0, 1.0),
            "l": (1.0, 1.0),
        },
        n_envs=1,
        seed=42,
    )

    action_space = env_family.action_space
    action_dim = action_space.shape[0]  # type: ignore[union-attr]
    state_dim = env_family.observation_space.shape[0]  # type: ignore[union-attr]

    feature_mask = np.ones(state_dim, dtype=bool)

    policy = CausalPolicy(
        full_state_dim=state_dim,
        feature_mask=feature_mask,
        action_dim=action_dim,
        hidden_dims=(64, 64),
        continuous=True,
        action_low=action_space.low,  # type: ignore[union-attr]
        action_high=action_space.high,  # type: ignore[union-attr]
    )

    # Pure PPO with standard hyperparameters for continuous control
    config = TrainingConfig(
        n_iterations=300,
        gamma=0.99,
        gae_lambda=0.95,
        learning_rate=3e-4,
        clip_epsilon=0.2,
        n_steps_per_env=2048,  # Standard PPO uses 2048
        n_ppo_epochs=10,
        mini_batch_size=64,
        irm_weight=0.0,
        worst_case_temperature=1.0,
        worst_case_variance_weight=0.0,
        entropy_coef=0.0,
    )

    trainer = CIRCTrainer(
        policy=policy,
        env_family=env_family,
        config=config,
    )

    metrics = trainer.run()

    # Print summary every 20 iterations
    for m in metrics:
        if m.iteration % 20 == 0:
            print(
                f"  iter {m.iteration + 1:3d}: "
                f"loss={m.total_loss:10.2f}  "
                f"policy={m.policy_loss:8.4f}  "
                f"value={m.value_loss:10.2f}  "
                f"return={m.mean_return:8.1f}"
            )

    print(f"\nFirst return: {metrics[0].mean_return:.1f}")
    print(f"Last return:  {metrics[-1].mean_return:.1f}")


if __name__ == "__main__":
    main()
