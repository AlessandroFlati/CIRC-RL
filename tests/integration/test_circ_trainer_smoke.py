"""Smoke test for CIRCTrainer on CartPole with 2 environments.

Verifies that training runs for a few iterations without crashing
and produces logged metrics.
"""

import numpy as np
import pytest

from circ_rl.constraints.const_definition import StateBoundConstraint
from circ_rl.constraints.const_set import ConstraintSet
from circ_rl.environments.env_family import EnvironmentFamily
from circ_rl.policy.causal_policy import CausalPolicy
from circ_rl.training.circ_trainer import CIRCTrainer, TrainingConfig


@pytest.fixture
def cartpole_family() -> EnvironmentFamily:
    return EnvironmentFamily.from_gymnasium(
        base_env="CartPole-v1",
        param_distributions={"gravity": (9.0, 11.0)},
        n_envs=2,
        seed=42,
    )


class TestCIRCTrainerSmoke:
    def test_runs_without_crashing(self, cartpole_family: EnvironmentFamily) -> None:
        """Training should complete 3 iterations on CartPole."""
        # All features are causal (no feature selection for smoke test)
        feature_mask = np.ones(4, dtype=bool)

        policy = CausalPolicy(
            full_state_dim=4,
            action_dim=2,
            feature_mask=feature_mask,
            hidden_dims=(32, 32),
        )

        config = TrainingConfig(
            n_iterations=3,
            n_steps_per_env=50,
            n_ppo_epochs=2,
            mini_batch_size=32,
        )

        trainer = CIRCTrainer(
            policy=policy,
            env_family=cartpole_family,
            config=config,
        )

        metrics = trainer.run()
        assert len(metrics) == 3
        for m in metrics:
            assert isinstance(m.total_loss, float)
            assert isinstance(m.mean_return, float)

    def test_with_constraints(self, cartpole_family: EnvironmentFamily) -> None:
        """Training with constraints should complete."""
        feature_mask = np.ones(4, dtype=bool)

        policy = CausalPolicy(
            full_state_dim=4,
            action_dim=2,
            feature_mask=feature_mask,
            hidden_dims=(32, 32),
        )

        constraint = StateBoundConstraint(
            name="cart_position",
            threshold=0.1,
            state_dim_idx=0,
            lower=-2.4,
            upper=2.4,
        )
        constraint_set = ConstraintSet([constraint])

        config = TrainingConfig(
            n_iterations=2,
            n_steps_per_env=50,
            n_ppo_epochs=1,
            mini_batch_size=32,
        )

        trainer = CIRCTrainer(
            policy=policy,
            env_family=cartpole_family,
            config=config,
            constraint_set=constraint_set,
        )

        metrics = trainer.run()
        assert len(metrics) == 2

    def test_with_info_bottleneck(self, cartpole_family: EnvironmentFamily) -> None:
        """Training with IB encoder should complete."""
        feature_mask = np.ones(4, dtype=bool)

        policy = CausalPolicy(
            full_state_dim=4,
            action_dim=2,
            feature_mask=feature_mask,
            hidden_dims=(32, 32),
            use_info_bottleneck=True,
            latent_dim=8,
        )

        config = TrainingConfig(
            n_iterations=2,
            n_steps_per_env=50,
            n_ppo_epochs=1,
            mini_batch_size=32,
        )

        trainer = CIRCTrainer(
            policy=policy,
            env_family=cartpole_family,
            config=config,
        )

        metrics = trainer.run()
        assert len(metrics) == 2
