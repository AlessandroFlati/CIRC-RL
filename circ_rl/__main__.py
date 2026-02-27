"""Hydra entry point for CIRC-RL v2.

Usage::

    uv run python -m circ_rl

Or with config overrides::

    uv run python -m circ_rl training.n_iterations=100 environments.n_envs=5
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import hydra
from loguru import logger

from circ_rl.environments.env_family import EnvironmentFamily
from circ_rl.orchestration.pipeline import CIRCPipeline
from circ_rl.orchestration.stages import (
    AnalyticPolicyDerivationStage,
    CausalDiscoveryStage,
    DiagnosticValidationStage,
    FeatureSelectionStage,
    HypothesisFalsificationStage,
    HypothesisGenerationStage,
    ResidualLearningStage,
    TransitionAnalysisStage,
)
from circ_rl.utils.seeding import seed_everything

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from circ_rl.orchestration.pipeline import PipelineStage


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run the CIRC-RL v2 pipeline."""
    seed_everything(cfg.get("seed", 42))

    device = cfg.get("device", "cpu")
    logger.info("Starting CIRC-RL v2 pipeline (device={})", device)
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

    # Env-param discovery config
    ep_cfg = cfg.get("env_param_discovery", {})
    include_env_params = bool(ep_cfg.get("enabled", False))
    ep_correlation_threshold = float(ep_cfg.get("ep_correlation_threshold", 0.05))

    # Build v2 pipeline stages
    cd_cfg = cfg.causal_discovery
    stages: list[PipelineStage] = [
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
        ),
        TransitionAnalysisStage(),
        HypothesisGenerationStage(include_env_params=include_env_params),
        HypothesisFalsificationStage(),
        AnalyticPolicyDerivationStage(env_family=env_family),
        ResidualLearningStage(env_family=env_family),
        DiagnosticValidationStage(env_family=env_family),
    ]

    pipeline = CIRCPipeline(stages, cache_dir=cfg.get("cache_dir", ".circ_cache"))
    results = pipeline.run(force_from=cfg.get("force_from", None))

    # Report final result
    diag = results.get("diagnostic_validation", {})
    recommended = diag.get("recommended_action")
    if recommended is not None:
        logger.info(
            "Pipeline complete. Recommended action: {}",
            recommended.value,
        )
    else:
        logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
