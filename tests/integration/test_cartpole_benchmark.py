"""Integration test: full CIRC-RL pipeline on CartPole benchmark.

Verifies that the pipeline runs end-to-end on a minimal CartPole family
and that the resulting ensemble can produce actions on a held-out environment.
"""

import numpy as np
import pytest
import torch

from circ_rl.environments.env_family import EnvironmentFamily
from circ_rl.evaluation.ensemble import EnsemblePolicy
from circ_rl.evaluation.mdl_scorer import MDLScorer
from circ_rl.policy.causal_policy import CausalPolicy
from circ_rl.training.circ_trainer import CIRCTrainer, TrainingConfig
from circ_rl.training.rollout import RolloutWorker
from experiments.benchmarks.cartpole_family import (
    make_cartpole_family,
    make_cartpole_family_minimal,
)


@pytest.fixture
def train_family() -> EnvironmentFamily:
    """Minimal CartPole family for training."""
    return make_cartpole_family_minimal(n_envs=2, seed=42)


@pytest.fixture
def held_out_family() -> EnvironmentFamily:
    """Held-out CartPole family for evaluation."""
    return make_cartpole_family_minimal(n_envs=2, seed=999)


class TestCartPoleBenchmark:
    @pytest.mark.slow
    def test_pipeline_produces_ensemble(
        self, train_family: EnvironmentFamily
    ) -> None:
        """Full pipeline on CartPole should produce a working ensemble."""
        state_dim = 4
        action_dim = 2
        feature_mask = np.ones(state_dim, dtype=bool)

        # Train 2 policies with tiny config
        policies: list[CausalPolicy] = []
        for _ in range(2):
            policy = CausalPolicy(
                full_state_dim=state_dim,
                action_dim=action_dim,
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
                env_family=train_family,
                config=config,
            )
            trainer.run()
            policies.append(policy)

        # Build ensemble
        rollout = RolloutWorker(train_family, n_steps_per_env=50)
        buffer = rollout.collect(policies[0])
        eval_traj = buffer.get_all_flat()

        ensemble = EnsemblePolicy.from_mdl_scores(policies, eval_traj)

        assert ensemble.n_policies == 2
        assert abs(ensemble.weights.sum() - 1.0) < 1e-6

    @pytest.mark.slow
    def test_ensemble_acts_on_held_out_env(
        self,
        train_family: EnvironmentFamily,
        held_out_family: EnvironmentFamily,
    ) -> None:
        """Ensemble trained on one family should produce actions on held-out envs."""
        state_dim = 4
        action_dim = 2
        feature_mask = np.ones(state_dim, dtype=bool)

        policy = CausalPolicy(
            full_state_dim=state_dim,
            action_dim=action_dim,
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
            env_family=train_family,
            config=config,
        )
        trainer.run()

        # Build single-policy ensemble
        rollout = RolloutWorker(train_family, n_steps_per_env=50)
        buffer = rollout.collect(policy)
        eval_traj = buffer.get_all_flat()

        ensemble = EnsemblePolicy.from_mdl_scores([policy], eval_traj)

        # Evaluate on held-out env
        held_out_env = held_out_family.make_env(0)
        obs, _ = held_out_env.reset()
        total_reward = 0.0

        for _ in range(100):
            state = torch.from_numpy(np.asarray(obs, dtype=np.float32))
            action = ensemble.get_action(state, deterministic=True)
            obs, reward, terminated, truncated, _ = held_out_env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                break

        held_out_env.close()

        # We don't assert reward > X since training is minimal,
        # just verify the ensemble produced valid actions without crashing
        assert isinstance(total_reward, float)

    def test_make_cartpole_family_creates_envs(self) -> None:
        """make_cartpole_family should create a valid family."""
        family = make_cartpole_family(n_envs=3, seed=42)
        assert family.n_envs == 3

        env = family.make_env(0)
        obs, _ = env.reset()
        assert obs.shape == (4,)
        env.close()

    def test_env_params_differ(self) -> None:
        """Environments in the family should have different parameters."""
        family = make_cartpole_family(n_envs=5, seed=42)
        params = [family.get_env_params(i) for i in range(5)]

        # At least gravity should differ across envs
        gravities = [p["gravity"] for p in params]
        assert len(set(gravities)) > 1
