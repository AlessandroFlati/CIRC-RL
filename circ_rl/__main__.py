"""Hydra entry point for CIRC-RL.

Usage::

    uv run python -m circ_rl

Or with config overrides::

    uv run python -m circ_rl training.n_iterations=100 environments.n_envs=5
"""

from __future__ import annotations

import hydra
from loguru import logger
from omegaconf import DictConfig

from circ_rl.environments.env_family import EnvironmentFamily
from circ_rl.orchestration.pipeline import CIRCPipeline
from circ_rl.orchestration.stages import (
    CausalDiscoveryStage,
    EnsembleConstructionStage,
    FeatureSelectionStage,
    PolicyOptimizationStage,
)
from circ_rl.training.circ_trainer import TrainingConfig
from circ_rl.utils.seeding import seed_everything


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run the CIRC-RL pipeline."""
    seed_everything(cfg.get("seed", 42))

    logger.info("Starting CIRC-RL pipeline")
    logger.info("Config: {}", dict(cfg))

    # Build environment family
    env_cfg = cfg.environments
    env_family = EnvironmentFamily.from_gymnasium(
        base_env=env_cfg.base_env,
        param_distributions=dict(env_cfg.param_distributions),
        n_envs=env_cfg.n_envs,
        seed=cfg.get("seed", 42),
    )

    # Build training config
    train_cfg = cfg.training
    training_config = TrainingConfig(
        n_iterations=train_cfg.n_iterations,
        gamma=train_cfg.gamma,
        learning_rate=train_cfg.learning_rate,
        n_steps_per_env=train_cfg.get("n_steps_per_env", 200),
        n_ppo_epochs=train_cfg.get("n_ppo_epochs", 4),
        mini_batch_size=train_cfg.get("mini_batch_size", 64),
        irm_weight=train_cfg.get("irm_weight", 1.0),
    )

    # Build pipeline stages
    cd_cfg = cfg.causal_discovery
    stages = [
        CausalDiscoveryStage(
            env_family=env_family,
            n_transitions_per_env=cd_cfg.n_transitions_per_env,
            discovery_method=cd_cfg.method,
            alpha=cd_cfg.alpha,
            seed=cfg.get("seed", 42),
        ),
        FeatureSelectionStage(
            epsilon=cfg.get("feature_selection", {}).get("epsilon", 0.1),
            min_ate=cfg.get("feature_selection", {}).get("min_ate", 0.01),
        ),
        PolicyOptimizationStage(
            env_family=env_family,
            training_config=training_config,
            n_policies=train_cfg.get("n_policies", 3),
        ),
        EnsembleConstructionStage(
            env_family=env_family,
        ),
    ]

    pipeline = CIRCPipeline(stages, cache_dir=cfg.get("cache_dir", ".circ_cache"))
    results = pipeline.run(force_from=cfg.get("force_from", None))

    ensemble = results["ensemble_construction"]["ensemble"]
    logger.info(
        "Pipeline complete. Ensemble has {} policies.",
        ensemble.n_policies,
    )


if __name__ == "__main__":
    main()
