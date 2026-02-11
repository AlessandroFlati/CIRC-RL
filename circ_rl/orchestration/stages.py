"""Concrete pipeline stages for the CIRC-RL framework.

Implements the four phases of CIRC-RL as pipeline stages:
1. Causal Discovery
2. Feature Selection
3. Policy Optimization
4. Ensemble Construction
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from loguru import logger

from circ_rl.causal_discovery.builder import CausalGraphBuilder
from circ_rl.causal_discovery.mechanism_validator import MechanismValidator
from circ_rl.environments.data_collector import DataCollector
from circ_rl.environments.env_family import EnvironmentFamily
from circ_rl.evaluation.ensemble import EnsemblePolicy
from circ_rl.evaluation.mdl_scorer import MDLScorer
from circ_rl.feature_selection.inv_feature_selector import InvFeatureSelector
from circ_rl.orchestration.pipeline import PipelineStage, hash_config
from circ_rl.policy.causal_policy import CausalPolicy
from circ_rl.training.circ_trainer import CIRCTrainer, TrainingConfig
from circ_rl.training.rollout import RolloutWorker


class CausalDiscoveryStage(PipelineStage):
    """Phase 1: Discover causal structure from multi-environment data.

    :param env_family: The environment family.
    :param n_transitions_per_env: Transitions to collect per environment.
    :param discovery_method: Algorithm for causal discovery (pc, ges, fci).
    :param alpha: Significance level for CI tests.
    :param seed: Random seed for data collection.
    """

    def __init__(
        self,
        env_family: EnvironmentFamily,
        n_transitions_per_env: int = 5000,
        discovery_method: str = "pc",
        alpha: float = 0.05,
        seed: int = 42,
    ) -> None:
        super().__init__(name="causal_discovery")
        self._env_family = env_family
        self._n_transitions = n_transitions_per_env
        self._method = discovery_method
        self._alpha = alpha
        self._seed = seed

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Run causal discovery.

        :returns: Dict with keys: graph, dataset, node_names, validation_result.
        """
        collector = DataCollector(self._env_family)
        dataset = collector.collect(
            n_transitions_per_env=self._n_transitions, seed=self._seed
        )

        state_dim = dataset.state_dim
        action_dim = 1 if dataset.actions.ndim == 1 else dataset.actions.shape[1]

        state_names = [f"s{i}" for i in range(state_dim)]
        action_names = (
            ["action"]
            if action_dim == 1
            else [f"action_{i}" for i in range(action_dim)]
        )
        next_state_names = [f"s{i}_next" for i in range(state_dim)]
        node_names = state_names + action_names + ["reward"] + next_state_names

        builder = CausalGraphBuilder()
        graph = builder.discover(
            dataset, node_names, method=self._method, alpha=self._alpha
        )

        validator = MechanismValidator(alpha=self._alpha)
        validation = validator.validate_invariance(
            dataset, graph, node_names, target_node="reward"
        )

        logger.info(
            "Causal discovery complete: {} nodes, {} edges, invariant={}",
            len(graph.nodes),
            len(graph.edges),
            validation.is_invariant,
        )

        return {
            "graph": graph,
            "dataset": dataset,
            "node_names": node_names,
            "state_names": state_names,
            "validation_result": validation,
        }

    def config_hash(self) -> str:
        return hash_config({
            "n_transitions": self._n_transitions,
            "method": self._method,
            "alpha": self._alpha,
            "seed": self._seed,
            "n_envs": self._env_family.n_envs,
        })


class FeatureSelectionStage(PipelineStage):
    """Phase 2: Select invariant causal features.

    :param epsilon: Maximum cross-environment ATE variance.
    :param min_ate: Minimum absolute ATE.
    """

    def __init__(
        self,
        epsilon: float = 0.1,
        min_ate: float = 0.01,
    ) -> None:
        super().__init__(name="feature_selection", dependencies=["causal_discovery"])
        self._epsilon = epsilon
        self._min_ate = min_ate

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Run feature selection.

        :returns: Dict with keys: feature_mask, selected_features, result.
        """
        cd_output = inputs["causal_discovery"]
        graph = cd_output["graph"]
        dataset = cd_output["dataset"]
        state_names = cd_output["state_names"]

        selector = InvFeatureSelector(
            epsilon=self._epsilon, min_ate=self._min_ate
        )
        result = selector.select(dataset, graph, state_names)

        # If no features selected, fall back to all state features
        if len(result.selected_features) == 0:
            logger.warning(
                "No features selected by invariance filter; "
                "falling back to all state features"
            )
            feature_mask = np.ones(len(state_names), dtype=bool)
        else:
            feature_mask = result.feature_mask

        logger.info(
            "Feature selection: {}/{} features selected",
            int(feature_mask.sum()),
            len(state_names),
        )

        return {
            "feature_mask": feature_mask,
            "selected_features": result.selected_features,
            "result": result,
        }

    def config_hash(self) -> str:
        return hash_config({
            "epsilon": self._epsilon,
            "min_ate": self._min_ate,
        })


class PolicyOptimizationStage(PipelineStage):
    """Phase 3: Train causal policies.

    :param env_family: The environment family.
    :param training_config: Training configuration.
    :param n_policies: Number of policies to train (for ensemble).
    :param device: Torch device for training.
    """

    def __init__(
        self,
        env_family: EnvironmentFamily,
        training_config: TrainingConfig,
        n_policies: int = 3,
        device: str = "cpu",
    ) -> None:
        super().__init__(
            name="policy_optimization",
            dependencies=["feature_selection"],
        )
        self._env_family = env_family
        self._config = training_config
        self._n_policies = n_policies
        self._device = torch.device(device)

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Train policies.

        :returns: Dict with keys: policies, all_metrics.
        """
        fs_output = inputs["feature_selection"]
        feature_mask = fs_output["feature_mask"]

        state_dim = int(feature_mask.shape[0])
        action_dim = self._env_family.action_space.n  # type: ignore[union-attr]

        policies: list[CausalPolicy] = []
        all_metrics = []

        for i in range(self._n_policies):
            logger.info("Training policy {}/{}", i + 1, self._n_policies)

            policy = CausalPolicy(
                full_state_dim=state_dim,
                action_dim=action_dim,
                feature_mask=feature_mask,
                hidden_dims=(256, 256),
            )

            trainer = CIRCTrainer(
                policy=policy,
                env_family=self._env_family,
                config=self._config,
            )
            trainer.to(self._device)

            metrics = trainer.run()
            # Move policy back to CPU for serialization/ensemble
            policy.to(torch.device("cpu"))
            policies.append(policy)
            all_metrics.append(metrics)

        return {
            "policies": policies,
            "all_metrics": all_metrics,
            "feature_mask": feature_mask,
        }

    def config_hash(self) -> str:
        return hash_config({
            "n_iterations": self._config.n_iterations,
            "learning_rate": self._config.learning_rate,
            "n_policies": self._n_policies,
            "n_steps_per_env": self._config.n_steps_per_env,
        })


class EnsembleConstructionStage(PipelineStage):
    """Phase 4: Build MDL-weighted ensemble from trained policies.

    :param env_family: Environment family for evaluation.
    :param n_eval_steps: Steps for evaluation trajectory.
    :param complexity_weight: MDL complexity weight.
    """

    def __init__(
        self,
        env_family: EnvironmentFamily,
        n_eval_steps: int = 500,
        complexity_weight: float = 0.01,
    ) -> None:
        super().__init__(
            name="ensemble_construction",
            dependencies=["policy_optimization"],
        )
        self._env_family = env_family
        self._n_eval_steps = n_eval_steps
        self._complexity_weight = complexity_weight

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Build the ensemble.

        :returns: Dict with keys: ensemble, mdl_scores.
        """
        po_output = inputs["policy_optimization"]
        policies = po_output["policies"]
        feature_mask = po_output["feature_mask"]

        # Collect evaluation trajectory
        rollout = RolloutWorker(self._env_family, n_steps_per_env=self._n_eval_steps)
        if policies:
            buffer = rollout.collect(policies[0])
            eval_traj = buffer.get_all_flat()
        else:
            raise ValueError("No policies to evaluate")

        ensemble = EnsemblePolicy.from_mdl_scores(
            policies, eval_traj, complexity_weight=self._complexity_weight
        )

        logger.info(
            "Ensemble built: {} policies, weights={}",
            ensemble.n_policies,
            ensemble.weights.tolist(),
        )

        return {
            "ensemble": ensemble,
            "mdl_scores": ensemble.scores,
        }

    def config_hash(self) -> str:
        return hash_config({
            "n_eval_steps": self._n_eval_steps,
            "complexity_weight": self._complexity_weight,
        })
