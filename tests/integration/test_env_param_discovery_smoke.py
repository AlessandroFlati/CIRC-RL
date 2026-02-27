"""Integration smoke tests for env-param causal discovery."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from circ_rl.causal_discovery.builder import CausalGraphBuilder
from circ_rl.environments.data_collector import DataCollector
from circ_rl.environments.env_family import EnvironmentFamily
from circ_rl.feature_selection.inv_feature_selector import InvFeatureSelector
from circ_rl.policy.causal_policy import CausalPolicy
from circ_rl.training.circ_trainer import CIRCTrainer, TrainingConfig
from circ_rl.training.rollout import RolloutWorker


class TestEnvParamDiscoverySmoke:
    """Smoke tests for the env-param discovery pipeline."""

    @pytest.fixture()
    def pendulum_family(self) -> EnvironmentFamily:
        return EnvironmentFamily.from_gymnasium(
            base_env="Pendulum-v1",
            param_distributions={"g": (9.0, 11.0)},
            n_envs=2,
            seed=42,
        )

    def test_causal_discovery_with_env_params(
        self, pendulum_family: EnvironmentFamily
    ) -> None:
        """Verify causal discovery produces a graph containing ep_ nodes."""
        collector = DataCollector(pendulum_family, include_env_params=True)
        dataset = collector.collect(n_transitions_per_env=500, seed=42)

        assert dataset.env_params is not None
        assert dataset.n_env_params == 1

        state_dim = dataset.state_dim
        state_names = [f"s{i}" for i in range(state_dim)]
        action_names = ["action"]
        next_state_names = [f"s{i}_next" for i in range(state_dim)]
        ep_names = ["ep_g"]
        node_names = state_names + action_names + ["reward"] + next_state_names + ep_names

        graph = CausalGraphBuilder.discover(
            dataset,
            node_names,
            method="pc",
            alpha=0.05,
            env_param_names=ep_names,
        )

        # Graph should contain the ep_g node
        assert "ep_g" in graph.nodes
        # ep_g should NOT have any incoming edges from non-ep nodes
        for parent in graph.parents("ep_g"):
            assert parent.startswith("ep_")

    def test_feature_selection_with_conditional_invariance(
        self, pendulum_family: EnvironmentFamily
    ) -> None:
        """Verify feature selection with conditional invariance completes."""
        collector = DataCollector(pendulum_family, include_env_params=True)
        dataset = collector.collect(n_transitions_per_env=500, seed=42)

        state_dim = dataset.state_dim
        state_names = [f"s{i}" for i in range(state_dim)]
        action_names = ["action"]
        next_state_names = [f"s{i}_next" for i in range(state_dim)]
        ep_names = ["ep_g"]
        node_names = state_names + action_names + ["reward"] + next_state_names + ep_names

        graph = CausalGraphBuilder.discover(
            dataset,
            node_names,
            method="pc",
            alpha=0.05,
            env_param_names=ep_names,
        )

        selector = InvFeatureSelector(
            epsilon=0.5,
            min_ate=0.01,
            enable_conditional_invariance=True,
        )
        result = selector.select(
            dataset, graph, state_names,
            env_param_names=ep_names,
        )

        # Should complete without error and produce a valid result
        assert result.feature_mask.shape == (state_dim,)
        assert isinstance(result.context_dependent_features, dict)
        assert isinstance(result.context_param_names, list)

    def test_context_conditional_policy_training(
        self, pendulum_family: EnvironmentFamily
    ) -> None:
        """Verify a policy with context_dim > 0 trains without error."""
        state_dim = 3  # Pendulum state dim
        action_dim = 1
        context_dim = 1  # one env param (g)

        policy = CausalPolicy(
            full_state_dim=state_dim,
            action_dim=action_dim,
            feature_mask=np.ones(state_dim, dtype=bool),
            hidden_dims=(32, 32),
            continuous=True,
            action_low=np.array([-2.0], dtype=np.float32),
            action_high=np.array([2.0], dtype=np.float32),
            context_dim=context_dim,
        )

        rollout = RolloutWorker(
            pendulum_family,
            n_steps_per_env=50,
            env_param_names=["g"],
        )
        config = TrainingConfig(
            n_iterations=2,
            n_steps_per_env=50,
            learning_rate=3e-4,
        )

        trainer = CIRCTrainer(
            policy=policy,
            env_family=pendulum_family,
            config=config,
        )
        # Override rollout worker with one that includes env params
        trainer._rollout_worker = rollout

        metrics = trainer.run()
        assert len(metrics) == 2
        assert all(m.total_loss is not None for m in metrics)
