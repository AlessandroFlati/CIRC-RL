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
    ValidationFeedbackStage,
)
from circ_rl.training.circ_trainer import TrainingConfig
from circ_rl.utils.seeding import seed_everything


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run the CIRC-RL pipeline."""
    seed_everything(cfg.get("seed", 42))

    device = cfg.get("device", "cpu")
    logger.info("Starting CIRC-RL pipeline (device={})", device)
    logger.info("Config: {}", dict(cfg))

    # Build environment family
    env_cfg = cfg.environments
    # Convert Hydra {low: X, high: Y} dicts to (low, high) tuples
    param_dists: dict[str, tuple[float, float]] = {}
    for name, dist in env_cfg.param_distributions.items():
        if isinstance(dist, (list, tuple)):
            param_dists[name] = (float(dist[0]), float(dist[1]))
        else:
            param_dists[name] = (float(dist.low), float(dist.high))

    env_family = EnvironmentFamily.from_gymnasium(
        base_env=env_cfg.base_env,
        param_distributions=param_dists,
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

    # Env-param discovery config
    ep_cfg = cfg.get("env_param_discovery", {})
    include_env_params = bool(ep_cfg.get("enabled", False))
    conditional_invariance = bool(ep_cfg.get("conditional_invariance", True))

    ep_correlation_threshold = float(ep_cfg.get("ep_correlation_threshold", 0.05))

    if include_env_params:
        logger.info(
            "Env-param causal discovery enabled (conditional_invariance={})",
            conditional_invariance,
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
            include_env_params=include_env_params,
            ep_correlation_threshold=ep_correlation_threshold,
        ),
        FeatureSelectionStage(
            epsilon=cfg.get("feature_selection", {}).get("epsilon", 0.1),
            min_ate=cfg.get("feature_selection", {}).get("min_ate", 0.01),
            enable_conditional_invariance=conditional_invariance and include_env_params,
        ),
        PolicyOptimizationStage(
            env_family=env_family,
            training_config=training_config,
            n_policies=train_cfg.get("n_policies", 3),
            device=device,
        ),
        EnsembleConstructionStage(
            env_family=env_family,
        ),
    ]

    # Add validation feedback stage when env-param discovery is enabled
    if include_env_params:
        validation_alpha = float(ep_cfg.get("validation_alpha", 0.05))
        stages.append(
            ValidationFeedbackStage(
                env_family=env_family,
                correlation_alpha=validation_alpha,
            )
        )

    pipeline = CIRCPipeline(stages, cache_dir=cfg.get("cache_dir", ".circ_cache"))
    results = pipeline.run(force_from=cfg.get("force_from", None))

    ensemble = results["ensemble_construction"]["ensemble"]
    logger.info(
        "Pipeline complete. Ensemble has {} policies.",
        ensemble.n_policies,
    )

    # Log validation feedback if available
    if "validation_feedback" in results:
        vf = results["validation_feedback"]
        suggested = vf.get("suggested_context_params", [])
        if suggested:
            logger.warning(
                "Validation feedback suggests additional context params: {}",
                suggested,
            )


if __name__ == "__main__":
    main()
