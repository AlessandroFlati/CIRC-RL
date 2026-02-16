# ruff: noqa: T201
"""Minimal v2 pipeline test with hand-crafted hypotheses.

Runs all 8 v2 pipeline stages on Pendulum-v1 WITHOUT requiring PySR.
Stages 1-3 (causal discovery, feature selection, transition analysis)
run normally. Stages 4-5 (hypothesis generation, falsification) are
bypassed by injecting hand-crafted linear hypotheses for Pendulum
dynamics. Stages 6-8 (analytic policy, residual, diagnostics) run
normally on the injected hypotheses.

This demonstrates the full scientific policy discovery methodology
with proper logging at each stage.

Usage::

    uv run python experiments/pendulum_v2_minimal.py
"""

from __future__ import annotations

import sys
import time

import numpy as np
import sympy
from loguru import logger

from circ_rl.environments.env_family import EnvironmentFamily
from circ_rl.hypothesis.expression import SymbolicExpression
from circ_rl.hypothesis.hypothesis_register import (
    HypothesisEntry,
    HypothesisRegister,
    HypothesisStatus,
)
from circ_rl.orchestration.stages import (
    AnalyticPolicyDerivationStage,
    CausalDiscoveryStage,
    DiagnosticValidationStage,
    FeatureSelectionStage,
    ResidualLearningStage,
    TransitionAnalysisStage,
)


def _make_pendulum_hypotheses(
    state_names: list[str],
    action_names: list[str],
    env_param_names: list[str],
) -> tuple[HypothesisRegister, list[str]]:
    """Build hand-crafted linear dynamics hypotheses for Pendulum.

    Pendulum-v1 state: [cos(theta), sin(theta), theta_dot].
    We fit linear approximations of the discrete-time dynamics:
      delta_s0 = a00*s0 + a01*s1 + a02*s2 + b0*action
      delta_s1 = a10*s0 + a11*s1 + a12*s2 + b1*action
      delta_s2 = a20*s0 + a21*s1 + a22*s2 + b2*action

    For the linearized Pendulum near theta=0:
    - cos(theta) ~= 1, sin(theta) ~= theta, theta_dot = omega
    - d(cos)/dt = -sin*omega => delta_s0 ~ -s1*s2 (nonlinear, but
      linearized: small coefficient)
    - d(sin)/dt = cos*omega => delta_s1 ~ s0*s2 (linearized: small)
    - d(omega)/dt = -(g/l)*sin(theta) + torque/(m*l^2)
      => delta_s2 ~ -c*s1 + d*action

    We use simple linear forms that the LQR solver can handle.
    """
    register = HypothesisRegister()

    variable_names = state_names + action_names
    if env_param_names:
        variable_names = variable_names + [
            f"ep_{n}" for n in env_param_names
        ]

    # For each state dim, create a linear hypothesis
    hypotheses = {
        # cos(theta): nearly static at equilibrium, small linear drift
        "delta_s0": "0.0*s0 + 0.0*s1 - 0.05*s2",
        # sin(theta): driven by theta_dot
        "delta_s1": "0.0*s0 + 0.0*s1 + 0.05*s2",
        # theta_dot: driven by gravity (-sin) and torque
        "delta_s2": "0.0*s0 - 1.5*s1 + 0.0*s2 + 0.1*action",
    }

    for target, expr_str in hypotheses.items():
        sympy_expr = sympy.sympify(expr_str)
        expression = SymbolicExpression.from_sympy(
            sympy_expr, constant_symbols=None,
        )

        entry = HypothesisEntry(
            hypothesis_id=f"hand_{target}",
            target_variable=target,
            expression=expression,
            complexity=expression.complexity,
            training_r2=0.92,
            training_mse=0.01,
            status=HypothesisStatus.VALIDATED,
            falsification_reason=None,
            mdl_score=float(expression.complexity) * np.log(1000),
        )
        register.register(entry)

    # Reward hypothesis: R ~ -(sin^2 + 0.1*omega^2 + 0.001*torque^2)
    # Simplified quadratic: R = -s1^2 - 0.1*s2^2 - 0.001*action^2
    reward_str = "-s1**2 - 0.1*s2**2 - 0.001*action**2"
    reward_expr = SymbolicExpression.from_sympy(
        sympy.sympify(reward_str), constant_symbols=None,
    )
    reward_entry = HypothesisEntry(
        hypothesis_id="hand_reward",
        target_variable="reward",
        expression=reward_expr,
        complexity=reward_expr.complexity,
        training_r2=0.85,
        training_mse=0.05,
        status=HypothesisStatus.VALIDATED,
        falsification_reason=None,
        mdl_score=float(reward_expr.complexity) * np.log(1000),
    )
    register.register(reward_entry)

    return register, variable_names


def main() -> None:
    """Run the minimal v2 pipeline test."""
    # Configure logging: show timestamps and level
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
    print("CIRC-RL v2 MINIMAL PIPELINE TEST")
    print("  Stages 1-3: real causal discovery + feature selection")
    print("  Stages 4-5: hand-crafted Pendulum hypotheses (no PySR)")
    print("  Stages 6-8: real analytic policy + residual + diagnostics")
    print("=" * 70)

    # -- Configuration --
    n_envs = 5
    n_transitions = 1000
    seed = 42

    # -- 1. Create environment family --
    print("\n[1/8] Creating environment family...")
    t0 = time.time()

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
        logger.info("Env {}: g={:.2f}, m={:.2f}, l={:.2f}", i, p["g"], p["m"], p["l"])

    # ====================================================================
    # STAGE 1: Causal Discovery
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
    # STAGE 2: Feature Selection
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
    # STAGE 3: Transition Analysis
    # ====================================================================
    print("\n[4/8] Running transition analysis...")
    ta_stage = TransitionAnalysisStage()
    ta_output = ta_stage.run({
        "causal_discovery": cd_output,
        "feature_selection": fs_output,
    })
    ta_result = ta_output["transition_result"]
    print(f"  Variant dims: {ta_result.variant_dims}")
    print(f"  Invariant dims: {ta_result.invariant_dims}")
    print(f"  Dynamics scales: {[f'{s:.4f}' for s in ta_result.dynamics_scales]}")

    # ====================================================================
    # STAGE 4-5: Hand-crafted Hypotheses (bypass PySR + falsification)
    # ====================================================================
    print("\n[5/8] Injecting hand-crafted Pendulum hypotheses...")
    env_param_names = env_family.param_names or []
    register, variable_names = _make_pendulum_hypotheses(
        state_names,
        ["action"],
        env_param_names,
    )
    logger.info(
        "Injected {} hypotheses ({} dynamics + 1 reward)",
        len(register.entries.values()),
        len(state_names),
    )

    # Build the outputs that downstream stages expect
    best_dynamics: dict[str, HypothesisEntry] = {}
    best_reward: HypothesisEntry | None = None
    for entry in register.entries.values():
        if entry.target_variable == "reward":
            best_reward = entry
        else:
            best_dynamics[entry.target_variable] = entry
        logger.info(
            "  {} -> {} (complexity={}, R2={:.3f})",
            entry.target_variable,
            entry.expression.expression_str,
            entry.complexity,
            entry.training_r2,
        )

    hf_output = {
        "register": register,
        "best_dynamics": best_dynamics,
        "best_reward": best_reward,
        "variable_names": variable_names,
        "state_names": state_names,
    }

    # ====================================================================
    # STAGE 6: Analytic Policy Derivation
    # ====================================================================
    print("\n[6/8] Deriving analytic policy (LQR)...")
    apd_stage = AnalyticPolicyDerivationStage(
        env_family=env_family,
        gamma=0.99,
    )
    apd_output = apd_stage.run({
        "hypothesis_falsification": hf_output,
        "transition_analysis": ta_output,
    })
    analytic_policy = apd_output["analytic_policy"]
    eta2 = apd_output["explained_variance"]
    solver_type = apd_output["solver_type"]
    print(f"  Solver: {solver_type}")
    print(f"  Explained variance (eta^2): {eta2:.4f}")

    # Quick test: get actions for a sample state
    sample_state = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    for env_idx in range(n_envs):
        a = analytic_policy.get_action(sample_state, env_idx)
        logger.info("  Policy(env={}, state=[1,0,0]) -> action={:.4f}", env_idx, a[0])

    # ====================================================================
    # STAGE 7: Residual Learning
    # ====================================================================
    print("\n[7/8] Running residual learning...")
    rl_stage = ResidualLearningStage(
        env_family=env_family,
        n_iterations=5,
        eta_max=0.1,
        skip_if_eta2_above=0.98,
        abort_if_eta2_below=0.50,
    )
    rl_output = rl_stage.run({
        "analytic_policy_derivation": apd_output,
    })
    composite = rl_output["composite_policy"]
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

    # Test composite policy
    for env_idx in range(min(3, n_envs)):
        a = composite.get_action(sample_state, env_idx)
        logger.info(
            "  Composite(env={}, state=[1,0,0]) -> action={:.4f}",
            env_idx, a[0],
        )

    # ====================================================================
    # STAGE 8: Diagnostic Validation
    # ====================================================================
    print("\n[8/8] Running diagnostic validation...")
    diag_stage = DiagnosticValidationStage(
        env_family=env_family,
        premise_r2_threshold=0.3,
        derivation_divergence_threshold=2.0,
        conclusion_error_threshold=0.5,
    )
    diag_output = diag_stage.run({
        "analytic_policy_derivation": apd_output,
        "residual_learning": rl_output,
        "causal_discovery": cd_output,
        "hypothesis_falsification": hf_output,
    })
    diag_result = diag_output["diagnostic_result"]
    recommended = diag_output["recommended_action"]

    print(f"\n  Premise test:    passed={diag_result.premise_result.passed}")
    if diag_result.derivation_result:
        print(
            f"  Derivation test: passed={diag_result.derivation_result.passed}, "
            f"mean_div={diag_result.derivation_result.mean_divergence:.4f}"
        )
    if diag_result.conclusion_result:
        print(
            f"  Conclusion test: passed={diag_result.conclusion_result.passed}, "
            f"mean_err={diag_result.conclusion_result.mean_error:.4f}"
        )

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"RECOMMENDED ACTION: {recommended.value}")
    print(f"Total time: {elapsed:.1f}s")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
