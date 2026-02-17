# ruff: noqa: T201
"""Full v2 pipeline on Pendulum-v1 with real PySR symbolic regression.

Runs all 8 v2 pipeline stages end-to-end:
  1. Causal Discovery (PC algorithm with env-param augmentation)
  2. Feature Selection (mechanism invariance)
  3. Transition Analysis (LOEO dynamics scales)
  4. Hypothesis Generation (PySR symbolic regression)
  5. Hypothesis Falsification (structural + OOD + trajectory tests)
  6. Analytic Policy Derivation (LQR/MPC from validated hypotheses)
  7. Residual Learning (bounded NN correction)
  8. Diagnostic Validation (premise/derivation/conclusion)

Requires PySR: ``pip install 'circ-rl[symbolic]'``

Usage::

    uv run python experiments/pendulum_v2_real.py

"""

from __future__ import annotations

import sys
import time

import numpy as np
from loguru import logger

from circ_rl.environments.env_family import EnvironmentFamily
from circ_rl.hypothesis.symbolic_regressor import SymbolicRegressionConfig
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
    """Run the full v2 pipeline with real PySR."""
    # Configure logging
    logger.remove()
    logger.add(
        sys.stderr,
        level="DEBUG",
        format=(
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
    )

    print("=" * 70)
    print("CIRC-RL v2 FULL PIPELINE -- REAL SYMBOLIC REGRESSION")
    print("  Environment: Pendulum-v1 with parametric variation (g, m, l)")
    print("  Methodology: PySR -> Falsification -> LQR/MPC -> Diagnostics")
    print("=" * 70)

    # -- Configuration --
    # 25 envs x 5000 transitions = 125k samples; max_samples=10k for SR
    # More envs give PySR stronger signal to discover parametric forms
    n_envs = 25
    n_transitions = 5000
    seed = 42

    # PySR configuration: validated sum-of-products config
    # Forces flat additive terms (no factored forms like (A+B)*C)
    # so the calibrated Chow test can assess structural consistency.
    sr_config = SymbolicRegressionConfig(
        max_complexity=30,
        n_iterations=80,
        populations=25,
        binary_operators=("+", "-", "*", "/"),
        unary_operators=("square",),
        parsimony=0.0005,
        timeout_seconds=600,
        deterministic=True,
        random_state=seed,
        nested_constraints={
            "*": {"+": 0, "-": 0},
            "/": {"+": 0, "-": 0},
        },
        complexity_of_operators={"square": 1},
        max_samples=10000,
    )

    t0 = time.time()

    # ====================================================================
    # STAGE 1: Create environment family
    # ====================================================================
    print("\n[1/8] Creating environment family...")
    env_family = EnvironmentFamily.from_gymnasium(
        base_env="Pendulum-v1",
        param_distributions={
            "g": (8.0, 12.0),
            "m": (0.8, 1.5),
            "l": (0.7, 1.3),
        },
        n_envs=n_envs,
        seed=seed,
    )

    for i in range(n_envs):
        p = env_family.get_env_params(i)
        logger.info(
            "Env {}: g={:.2f}, m={:.2f}, l={:.2f}",
            i,
            p["g"],
            p["m"],
            p["l"],
        )

    # ====================================================================
    # STAGE 2: Causal Discovery
    # ====================================================================
    print("\n[2/8] Running causal discovery...")
    cd_stage = CausalDiscoveryStage(
        env_family=env_family,
        n_transitions_per_env=n_transitions,
        discovery_method="pc",
        alpha=0.05,
        seed=seed,
        include_env_params=True,
        ep_correlation_threshold=0.05,
    )
    cd_output = cd_stage.run({})
    graph = cd_output["graph"]
    state_names = cd_output["state_names"]
    print(f"  Graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
    print(f"  State names: {state_names}")

    # ====================================================================
    # STAGE 3: Feature Selection
    # ====================================================================
    print("\n[3/8] Running feature selection...")
    fs_stage = FeatureSelectionStage(
        epsilon=0.15,
        min_ate=0.01,
        use_mechanism_invariance=True,
    )
    fs_output = fs_stage.run({"causal_discovery": cd_output})
    fs_result = fs_output["result"]
    print(f"  Selected features: {fs_result.selected_features}")
    print(f"  Feature weights: {fs_output['feature_weights']}")

    # ====================================================================
    # STAGE 4: Transition Analysis
    # ====================================================================
    print("\n[4/8] Running transition analysis...")
    ta_stage = TransitionAnalysisStage()
    ta_output = ta_stage.run(
        {
            "causal_discovery": cd_output,
            "feature_selection": fs_output,
        }
    )
    ta_result = ta_output["transition_result"]
    print(f"  Variant dims: {ta_result.variant_dims}")
    print(f"  Invariant dims: {ta_result.invariant_dims}")
    print(f"  Dynamics scales: {[f'{s:.4f}' for s in ta_result.dynamics_scales]}")

    # ====================================================================
    # STAGE 5: Hypothesis Generation (REAL PySR)
    # ====================================================================
    print("\n[5/8] Running symbolic regression (PySR)...")
    print(
        f"  Config: max_complexity={sr_config.max_complexity}, "
        f"n_iterations={sr_config.n_iterations}, "
        f"timeout={sr_config.timeout_seconds}s"
    )

    from circ_rl.hypothesis.derived_features import DerivedFeatureSpec

    reward_derived_specs = [
        DerivedFeatureSpec(
            name="theta",
            source_names=("s1", "s0"),
            compute_fn=np.arctan2,
        ),
    ]

    t_sr = time.time()
    hg_stage = HypothesisGenerationStage(
        include_env_params=True,
        sr_config=sr_config,
        reward_sr_config=sr_config,
        reward_derived_features=reward_derived_specs,
    )
    hg_output = hg_stage.run(
        {
            "causal_discovery": cd_output,
            "feature_selection": fs_output,
            "transition_analysis": ta_output,
        }
    )
    t_sr_total = time.time() - t_sr

    register = hg_output["register"]
    print(f"  {len(register.entries)} hypotheses in {t_sr_total:.1f}s")

    # Print all discovered expressions
    print("\n  --- Discovered Hypotheses ---")
    for entry in register.entries.values():
        print(
            f"  [{entry.target_variable}] {entry.expression.expression_str}"
            f"  (complexity={entry.complexity}, R2={entry.training_r2:.4f})"
        )

    # ====================================================================
    # STAGE 6: Hypothesis Falsification
    # ====================================================================
    print("\n[6/8] Running hypothesis falsification...")
    hf_stage = HypothesisFalsificationStage(
        structural_p_threshold=0.01,
        structural_min_relative_improvement=0.01,
        ood_confidence=0.99,
        held_out_fraction=0.2,
    )
    hf_output = hf_stage.run(
        {
            "hypothesis_generation": hg_output,
            "causal_discovery": cd_output,
        }
    )
    hf_result = hf_output["falsification_result"]
    best_dynamics = hf_output["best_dynamics"]
    best_reward = hf_output["best_reward"]
    print(
        f"  Tested: {hf_result.n_tested}, "
        f"Validated: {hf_result.n_validated}, "
        f"Falsified: {hf_result.n_falsified}"
    )
    print(f"  Best per target: {hf_result.best_per_target}")

    if best_dynamics:
        covered_targets = set(best_dynamics.keys())
        all_targets = {f"delta_{s}" for s in state_names}
        missing = all_targets - covered_targets
        n_cov = len(covered_targets)
        n_all = len(all_targets)
        print(f"\n  --- Best Validated Dynamics ({n_cov}/{n_all} dims) ---")
        for target, entry in best_dynamics.items():
            print(
                f"  [{target}] {entry.expression.expression_str}"
                f"  (R2={entry.training_r2:.4f}, MDL={entry.mdl_score:.2f})"
            )
        if missing:
            print(f"\n  WARNING: No validated hypothesis for: {sorted(missing)}")
            print("  These dimensions will use identity dynamics (delta=0).")
    else:
        print("\n  WARNING: No validated dynamics hypotheses!")
        print("  This likely means all expressions were falsified.")
        print("  The pipeline cannot proceed without dynamics models.")
        return

    if best_reward:
        print(
            f"  [reward] {best_reward.expression.expression_str}"
            f"  (R2={best_reward.training_r2:.4f}, "
            f"MDL={best_reward.mdl_score:.2f})"
        )

    # ====================================================================
    # STAGE 7: Analytic Policy Derivation
    # ====================================================================
    print("\n[7/8] Deriving analytic policy...")
    apd_stage = AnalyticPolicyDerivationStage(
        env_family=env_family,
        gamma=0.99,
    )
    apd_output = apd_stage.run(
        {
            "hypothesis_falsification": hf_output,
            "transition_analysis": ta_output,
        }
    )
    analytic_policy = apd_output["analytic_policy"]
    eta2 = apd_output["explained_variance"]
    solver_type = apd_output["solver_type"]
    print(f"  Solver: {solver_type}")
    print(f"  Explained variance (eta^2): {eta2:.4f}")

    # Test: get actions for sample states
    sample_states = [
        np.array([1.0, 0.0, 0.0], dtype=np.float32),  # upright
        np.array([0.0, 1.0, 0.0], dtype=np.float32),  # horizontal
        np.array([-1.0, 0.0, 0.0], dtype=np.float32),  # hanging down
    ]
    for state in sample_states:
        for env_idx in [0, n_envs - 1]:
            a = analytic_policy.get_action(state, env_idx)
            logger.info(
                "  Policy(env={}, state={}) -> action={:.4f}",
                env_idx,
                state.tolist(),
                a[0],
            )

    # ====================================================================
    # STAGE 8: Residual Learning
    # ====================================================================
    print("\n[8a/8] Running residual learning...")
    rl_stage = ResidualLearningStage(
        env_family=env_family,
        n_iterations=5,
        eta_max=0.1,
        skip_if_eta2_above=0.98,
        abort_if_eta2_below=0.50,
    )
    rl_output = rl_stage.run(
        {
            "analytic_policy_derivation": apd_output,
        }
    )
    skipped = rl_output["skipped"]
    residual_metrics = rl_output["residual_metrics"]
    if skipped:
        print("  Residual learning SKIPPED (eta^2 high enough)")
    elif residual_metrics:
        last = residual_metrics[-1]
        print(
            f"  Residual learning: {len(residual_metrics)} iterations, "
            f"final loss={last.total_loss:.4f}, "
            f"return={last.mean_return:.2f}"
        )
    else:
        print("  Residual learning completed (no metrics)")

    # ====================================================================
    # STAGE 8b: Diagnostic Validation
    # ====================================================================
    print("\n[8b/8] Running diagnostic validation...")
    diag_stage = DiagnosticValidationStage(
        env_family=env_family,
        premise_r2_threshold=0.3,
        derivation_divergence_threshold=2.0,
        conclusion_error_threshold=0.5,
    )
    diag_output = diag_stage.run(
        {
            "analytic_policy_derivation": apd_output,
            "residual_learning": rl_output,
            "causal_discovery": cd_output,
            "hypothesis_falsification": hf_output,
        }
    )
    diag_result = diag_output["diagnostic_result"]
    recommended = diag_output["recommended_action"]

    print(
        f"\n  Premise test:    passed={diag_result.premise_result.passed}"
        f" (R2={diag_result.premise_result.overall_r2:.4f})"
    )
    if diag_result.derivation_result:
        print(
            f"  Derivation test: passed="
            f"{diag_result.derivation_result.passed}, "
            f"mean_div={diag_result.derivation_result.mean_divergence:.4f}"
        )
    if diag_result.conclusion_result:
        print(
            f"  Conclusion test: passed="
            f"{diag_result.conclusion_result.passed}, "
            f"mean_err="
            f"{diag_result.conclusion_result.mean_relative_error:.4f}"
        )

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"RECOMMENDED ACTION: {recommended.value}")
    print(f"Total time: {elapsed:.1f}s")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
