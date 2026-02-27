"""Diagnostic suite orchestrating all validation tests.

Runs premise, derivation, and conclusion tests and produces
a recommended action based on the diagnostic table.

See ``CIRC-RL_Framework.md`` Section 3.9.4 (Diagnostic Summary).
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import TYPE_CHECKING

from loguru import logger

from circ_rl.diagnostics.conclusion_test import ConclusionTest, ConclusionTestResult
from circ_rl.diagnostics.derivation_test import DerivationTest, DerivationTestResult
from circ_rl.diagnostics.premise_test import PremiseTest, PremiseTestResult

if TYPE_CHECKING:
    from circ_rl.analytic_policy.analytic_policy import AnalyticPolicy
    from circ_rl.environments.data_collector import ExploratoryDataset
    from circ_rl.environments.env_family import EnvironmentFamily
    from circ_rl.hypothesis.expression import SymbolicExpression


class RecommendedAction(enum.Enum):
    """Recommended action based on diagnostic results.

    See ``CIRC-RL_Framework.md`` Section 3.9.4.
    """

    DEPLOY = "deploy_with_monitoring"
    RICHER_DYNAMICS = "return_to_phase_3_richer_dynamics"
    DEBUG_DERIVATION = "debug_derivation_computational_issue"
    RICHER_REWARD = "return_to_phase_3_richer_reward"


@dataclass(frozen=True)
class DiagnosticResult:
    """Result of the full diagnostic suite.

    :param premise_result: Result of the premise (dynamics) test.
    :param derivation_result: Result of the derivation (trajectory) test.
    :param conclusion_result: Result of the conclusion (return) test.
    :param recommended_action: What to do based on the results.
    """

    premise_result: PremiseTestResult
    derivation_result: DerivationTestResult | None
    conclusion_result: ConclusionTestResult | None
    recommended_action: RecommendedAction


class DiagnosticSuite:
    """Orchestrate the full diagnostic validation suite.

    Runs tests in order per the diagnostic table:

    +---------+------------+------------+-----------------------------------+
    | Premise | Derivation | Conclusion | Action                            |
    +=========+============+============+===================================+
    | FAIL    | --         | --         | Return to Phase 3: richer dynamics|
    +---------+------------+------------+-----------------------------------+
    | PASS    | FAIL       | --         | Debug derivation (computational)  |
    +---------+------------+------------+-----------------------------------+
    | PASS    | PASS       | FAIL       | Return to Phase 3: richer reward  |
    +---------+------------+------------+-----------------------------------+
    | PASS    | PASS       | PASS       | Deploy with monitoring            |
    +---------+------------+------------+-----------------------------------+

    See ``CIRC-RL_Framework.md`` Section 3.9.4.

    :param premise_r2_threshold: R^2 threshold for premise test.
    :param derivation_divergence_threshold: Divergence threshold for
        derivation test.
    :param conclusion_error_threshold: Relative error threshold for
        conclusion test.
    :param eval_episodes: Number of evaluation episodes per environment.
    :param max_steps: Maximum steps per evaluation episode.
    """

    def __init__(
        self,
        premise_r2_threshold: float = 0.5,
        derivation_divergence_threshold: float = 1.0,
        conclusion_error_threshold: float = 0.3,
        eval_episodes: int = 5,
        max_steps: int = 200,
    ) -> None:
        self._premise = PremiseTest(r2_threshold=premise_r2_threshold)
        self._derivation = DerivationTest(
            divergence_threshold=derivation_divergence_threshold,
        )
        self._conclusion = ConclusionTest(
            relative_error_threshold=conclusion_error_threshold,
            n_eval_episodes=eval_episodes,
            max_steps=max_steps,
        )

    def run(
        self,
        policy: AnalyticPolicy,
        dynamics_expressions: dict[int, SymbolicExpression],
        dataset: ExploratoryDataset,
        state_feature_names: list[str],
        variable_names: list[str],
        env_family: EnvironmentFamily,
        predicted_returns: dict[int, float],
        test_env_ids: list[int],
    ) -> DiagnosticResult:
        """Run the full diagnostic suite.

        Stops early on failure according to the diagnostic table.

        :param policy: The analytic policy.
        :param dynamics_expressions: Validated dynamics expressions.
        :param dataset: Test environment data.
        :param state_feature_names: State feature names.
        :param variable_names: Variable names for expression evaluation.
        :param env_family: Environment family for conclusion test.
        :param predicted_returns: Predicted returns per env.
        :param test_env_ids: Environment IDs to test.
        :returns: DiagnosticResult.
        """
        # Test 1: Premise
        logger.info("Running premise test...")
        premise_result = self._premise.test(
            dynamics_expressions, dataset, state_feature_names,
            variable_names, test_env_ids,
        )

        if not premise_result.passed:
            logger.warning(
                "Premise test FAILED: dynamics hypothesis invalid in test "
                "environments (R2={:.4f}). Recommended action: richer dynamics.",
                premise_result.overall_r2,
            )
            return DiagnosticResult(
                premise_result=premise_result,
                derivation_result=None,
                conclusion_result=None,
                recommended_action=RecommendedAction.RICHER_DYNAMICS,
            )

        # Test 2: Derivation
        logger.info("Running derivation test...")
        derivation_result = self._derivation.test(
            policy, dynamics_expressions, dataset, state_feature_names,
            variable_names, test_env_ids,
        )

        if not derivation_result.passed:
            logger.warning(
                "Derivation test FAILED: policy trajectories diverge from "
                "predictions (mean_div={:.4f}). Recommended action: debug "
                "derivation.",
                derivation_result.mean_divergence,
            )
            return DiagnosticResult(
                premise_result=premise_result,
                derivation_result=derivation_result,
                conclusion_result=None,
                recommended_action=RecommendedAction.DEBUG_DERIVATION,
            )

        # Test 3: Conclusion
        logger.info("Running conclusion test...")
        conclusion_result = self._conclusion.test(
            policy, env_family, predicted_returns, test_env_ids,
        )

        if not conclusion_result.passed:
            logger.warning(
                "Conclusion test FAILED: observed returns do not match "
                "predictions (mean_error={:.4f}). Recommended action: richer "
                "reward hypothesis.",
                conclusion_result.mean_relative_error,
            )
            return DiagnosticResult(
                premise_result=premise_result,
                derivation_result=derivation_result,
                conclusion_result=conclusion_result,
                recommended_action=RecommendedAction.RICHER_REWARD,
            )

        logger.info(
            "All diagnostic tests PASSED. Recommended action: deploy "
            "with monitoring.",
        )
        return DiagnosticResult(
            premise_result=premise_result,
            derivation_result=derivation_result,
            conclusion_result=conclusion_result,
            recommended_action=RecommendedAction.DEPLOY,
        )
