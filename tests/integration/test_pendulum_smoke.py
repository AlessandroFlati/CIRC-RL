"""Smoke test for CIRCTrainer on Pendulum-v1 (continuous actions)."""

import numpy as np
import pytest

from circ_rl.environments.env_family import EnvironmentFamily
from circ_rl.policy.causal_policy import CausalPolicy
from circ_rl.training.circ_trainer import CIRCTrainer, TrainingConfig


@pytest.fixture
def pendulum_family() -> EnvironmentFamily:
    return EnvironmentFamily.from_gymnasium(
        base_env="Pendulum-v1",
        param_distributions={"g": (9.0, 11.0)},
        n_envs=2,
        seed=42,
    )


class TestPendulumSmoke:
    def test_runs_without_crashing(
        self, pendulum_family: EnvironmentFamily
    ) -> None:
        """Training should complete 3 iterations on Pendulum."""
        state_dim = 3
        action_dim = 1
        feature_mask = np.ones(state_dim, dtype=bool)

        action_space = pendulum_family.action_space
        policy = CausalPolicy(
            full_state_dim=state_dim,
            action_dim=action_dim,
            feature_mask=feature_mask,
            hidden_dims=(32, 32),
            continuous=True,
            action_low=action_space.low,
            action_high=action_space.high,
        )

        config = TrainingConfig(
            n_iterations=3,
            n_steps_per_env=50,
            n_ppo_epochs=2,
            mini_batch_size=32,
        )

        trainer = CIRCTrainer(
            policy=policy,
            env_family=pendulum_family,
            config=config,
        )

        metrics = trainer.run()
        assert len(metrics) == 3
        for m in metrics:
            assert isinstance(m.total_loss, float)
            assert isinstance(m.mean_return, float)
