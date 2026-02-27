# ruff: noqa: T201
"""End-to-end v2 pipeline on Pendulum-v1 across multiple physics variants.

Demonstrates the full scientific policy discovery cycle:
  Phase 1-2: Causal discovery + feature selection (reused from v1)
  Phase 3:   Symbolic regression -> dynamics/reward hypotheses
  Phase 4:   Falsification protocol
  Phase 5:   Analytic policy derivation (LQR/MPC)
  Phase 6:   Bounded residual learning
  Phase 7:   Diagnostic validation

Usage::

    uv run python experiments/pendulum_v2.py

Requires the ``symbolic`` extra::

    uv pip install -e ".[symbolic]"
"""

from __future__ import annotations

import sys
import time

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


def main() -> None:
    """Run the v2 scientific policy discovery pipeline on Pendulum."""
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    # -- Configuration --
    n_envs = 16
    n_transitions_discovery = 2000
    seed = 42

    print("=" * 60)
    print("PENDULUM v2 PIPELINE (scientific policy discovery)")
    print("=" * 60)

    # -- 1. Create environment family --
    env_family = EnvironmentFamily.from_gymnasium(
        base_env="Pendulum-v1",
        param_distributions={
            "g": (7.0, 13.0),
            "m": (0.5, 2.0),
            "l": (0.5, 1.5),
        },
        n_envs=n_envs,
        seed=seed,
        fixed_params={"max_torque": 100.0},
    )

    print(f"\n{n_envs} environments:")
    for i in range(n_envs):
        p = env_family.get_env_params(i)
        print(f"  env {i}: g={p['g']:.2f}, m={p['m']:.2f}, l={p['l']:.2f}")

    # -- 2. Build v2 pipeline stages --
    stages = [
        CausalDiscoveryStage(
            env_family=env_family,
            n_transitions_per_env=n_transitions_discovery,
            discovery_method="pc",
            alpha=0.05,
            seed=seed,
            include_env_params=True,
            ep_correlation_threshold=0.05,
        ),
        FeatureSelectionStage(
            epsilon=0.15,
            min_ate=0.01,
        ),
        TransitionAnalysisStage(),
        HypothesisGenerationStage(include_env_params=True),
        HypothesisFalsificationStage(),
        AnalyticPolicyDerivationStage(env_family=env_family),
        ResidualLearningStage(env_family=env_family),
        DiagnosticValidationStage(env_family=env_family),
    ]

    # -- 3. Run pipeline --
    pipeline = CIRCPipeline(stages, cache_dir=".circ_cache/pendulum_v2")

    print("\n--- Running v2 pipeline ---")
    t0 = time.time()
    results = pipeline.run()
    elapsed = time.time() - t0

    # -- 4. Report results --
    print(f"\nPipeline completed in {elapsed:.1f}s")

    # Hypothesis generation
    hyp_gen = results.get("hypothesis_generation", {})
    register = hyp_gen.get("register")
    if register is not None:
        print(f"\nHypotheses generated: {len(register.all_entries)}")

    # Falsification
    falsification = results.get("hypothesis_falsification", {})
    best_dynamics = falsification.get("best_dynamics", {})
    best_reward = falsification.get("best_reward")
    print(f"Best dynamics hypotheses: {len(best_dynamics)} dimensions")
    for dim_name, entry in best_dynamics.items():
        print(f"  {dim_name}: {entry.expression.expression_str}")
    if best_reward is not None:
        print(f"Best reward: {best_reward.expression.expression_str}")

    # Analytic policy
    analytic = results.get("analytic_policy_derivation", {})
    solver_type = analytic.get("solver_type", "unknown")
    eta2 = analytic.get("explained_variance", 0.0)
    print(f"\nSolver type: {solver_type}")
    print(f"Explained variance (eta^2): {eta2:.4f}")

    # Residual learning
    residual = results.get("residual_learning", {})
    skipped = residual.get("skipped", False)
    if skipped:
        print("Residual learning: SKIPPED (eta^2 high enough)")
    else:
        metrics = residual.get("residual_metrics", [])
        if metrics:
            print(f"Residual learning: {len(metrics)} iterations")

    # Diagnostics
    diagnostics = results.get("diagnostic_validation", {})
    recommended = diagnostics.get("recommended_action")
    if recommended is not None:
        print(f"\nRecommended action: {recommended.value}")
    else:
        print("\nNo diagnostic result available.")

    print("\nDone.")


if __name__ == "__main__":
    main()
