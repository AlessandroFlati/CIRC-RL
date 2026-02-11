"""Integration test: CIRC-RL on Pendulum-v1 continuous action benchmark."""

import numpy as np
import pytest
import torch

from circ_rl.environments.env_family import EnvironmentFamily
from circ_rl.evaluation.ensemble import EnsemblePolicy
from circ_rl.policy.causal_policy import CausalPolicy
from circ_rl.training.circ_trainer import CIRCTrainer, TrainingConfig
from circ_rl.training.rollout import RolloutWorker
from experiments.benchmarks.pendulum_family import make_pendulum_family_minimal


@pytest.fixture
def train_family() -> EnvironmentFamily:
    return make_pendulum_family_minimal(n_envs=2, seed=42)


class TestPendulumBenchmark:
    @pytest.mark.slow
    def test_pipeline_produces_ensemble(
        self, train_family: EnvironmentFamily
    ) -> None:
        """Full pipeline on Pendulum should produce a working ensemble."""
        state_dim = 3
        action_dim = 1
        feature_mask = np.ones(state_dim, dtype=bool)
        action_space = train_family.action_space

        policies: list[CausalPolicy] = []
        for _ in range(2):
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
                env_family=train_family,
                config=config,
            )
            trainer.run()
            policies.append(policy)

        rollout = RolloutWorker(train_family, n_steps_per_env=50)
        buffer = rollout.collect(policies[0])
        eval_traj = buffer.get_all_flat()

        ensemble = EnsemblePolicy.from_mdl_scores(policies, eval_traj)
        assert ensemble.n_policies == 2
        assert abs(ensemble.weights.sum() - 1.0) < 1e-6

    @pytest.mark.slow
    def test_ensemble_acts_on_held_out_env(
        self, train_family: EnvironmentFamily
    ) -> None:
        """Ensemble should produce valid actions on held-out Pendulum envs."""
        state_dim = 3
        action_dim = 1
        feature_mask = np.ones(state_dim, dtype=bool)
        action_space = train_family.action_space

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
            env_family=train_family,
            config=config,
        )
        trainer.run()

        rollout = RolloutWorker(train_family, n_steps_per_env=50)
        buffer = rollout.collect(policy)
        eval_traj = buffer.get_all_flat()
        ensemble = EnsemblePolicy.from_mdl_scores([policy], eval_traj)

        held_out = make_pendulum_family_minimal(n_envs=2, seed=999)
        held_out_env = held_out.make_env(0)
        obs, _ = held_out_env.reset()

        for _ in range(50):
            state = torch.from_numpy(np.asarray(obs, dtype=np.float32))
            action = ensemble.get_action(state, deterministic=True)
            assert isinstance(action, np.ndarray)
            obs, reward, terminated, truncated, _ = held_out_env.step(action)
            if terminated or truncated:
                break

        held_out_env.close()
