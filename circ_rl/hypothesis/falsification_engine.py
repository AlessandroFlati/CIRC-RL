"""Falsification engine orchestrating all hypothesis tests.

Runs structural consistency, OOD prediction, and trajectory prediction
tests on untested hypotheses, updating their status in the register.

See ``CIRC-RL_Framework.md`` Section 3.5 (Phase 4: Hypothesis
Falsification).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from loguru import logger

from circ_rl.hypothesis.expression import SymbolicExpression
from circ_rl.hypothesis.hypothesis_register import (
    HypothesisRegister,
    HypothesisStatus,
)
from circ_rl.hypothesis.mdl_symbolic import SymbolicMDLScorer
from circ_rl.hypothesis.ood_prediction import OODPredictionTest
from circ_rl.hypothesis.structural_consistency import StructuralConsistencyTest
from circ_rl.hypothesis.trajectory_prediction import TrajectoryPredictionTest

if TYPE_CHECKING:
    from circ_rl.environments.data_collector import ExploratoryDataset


@dataclass(frozen=True)
class FalsificationConfig:
    """Configuration for the falsification engine.

    :param structural_p_threshold: p-value threshold for structural
        consistency F-test. Default 0.01.
    :param ood_confidence: Confidence level for OOD prediction intervals.
        Default 0.99.
    :param ood_failure_fraction: Reject if more than this fraction of
        held-out environments fail. Default 0.2.
    :param trajectory_failure_fraction: Reject if more than this fraction
        of test trajectories diverge. Default 0.2.
    :param held_out_fraction: Fraction of environments to hold out for
        OOD testing. Default 0.2.
    :param trajectory_horizon: Maximum trajectory prediction horizon.
        Default 50.
    :param trajectory_divergence_factor: Base divergence threshold factor.
        Default 0.5.
    :param n_test_trajectories: Number of test trajectories per environment.
        Default 20.
    """

    structural_p_threshold: float = 0.01
    ood_confidence: float = 0.99
    ood_failure_fraction: float = 0.2
    trajectory_failure_fraction: float = 0.2
    held_out_fraction: float = 0.2
    trajectory_horizon: int = 50
    trajectory_divergence_factor: float = 0.5
    n_test_trajectories: int = 20


@dataclass(frozen=True)
class FalsificationResult:
    """Result of the full falsification pipeline.

    :param n_tested: Number of hypotheses tested.
    :param n_validated: Number that survived all tests.
    :param n_falsified: Number that failed at least one test.
    :param best_per_target: Best validated hypothesis ID per target
        variable, or None if no survivors.
    """

    n_tested: int
    n_validated: int
    n_falsified: int
    best_per_target: dict[str, str | None]


class FalsificationEngine:
    """Orchestrate falsification tests on all untested hypotheses.

    Runs tests in order (structural -> OOD -> trajectory), stopping
    early on first failure (fail-fast). Survivors are scored via
    symbolic MDL and the best per target variable is selected.

    See ``CIRC-RL_Framework.md`` Section 3.5.

    :param config: Falsification configuration.
    """

    def __init__(
        self,
        config: FalsificationConfig | None = None,
    ) -> None:
        self._config = config or FalsificationConfig()

    @property
    def config(self) -> FalsificationConfig:
        """The falsification configuration."""
        return self._config

    def run(
        self,
        register: HypothesisRegister,
        dataset: ExploratoryDataset,
        state_feature_names: list[str],
        variable_names: list[str],
    ) -> FalsificationResult:
        """Run falsification on all untested hypotheses.

        :param register: The hypothesis register (mutated in place).
        :param dataset: Multi-environment data.
        :param state_feature_names: Names of state features.
        :param variable_names: Variable names for expression evaluation.
        :returns: FalsificationResult summary.
        """
        cfg = self._config
        untested = register.get_by_status(HypothesisStatus.UNTESTED)

        if not untested:
            logger.info("No untested hypotheses to falsify")
            return FalsificationResult(
                n_tested=0, n_validated=0, n_falsified=0,
                best_per_target={},
            )

        logger.info("Falsifying {} untested hypotheses", len(untested))

        structural_test = StructuralConsistencyTest(
            p_threshold=cfg.structural_p_threshold,
        )
        ood_test = OODPredictionTest(
            confidence=cfg.ood_confidence,
            failure_fraction=cfg.ood_failure_fraction,
            held_out_fraction=cfg.held_out_fraction,
        )
        trajectory_test = TrajectoryPredictionTest(
            max_horizon=cfg.trajectory_horizon,
            divergence_threshold_factor=cfg.trajectory_divergence_factor,
            failure_fraction=cfg.trajectory_failure_fraction,
            n_trajectories=cfg.n_test_trajectories,
        )
        mdl_scorer = SymbolicMDLScorer()

        n_validated = 0
        n_falsified = 0

        for entry in untested:
            expr = entry.expression
            if not isinstance(expr, SymbolicExpression):
                logger.warning(
                    "Hypothesis '{}' has non-SymbolicExpression expression, "
                    "skipping falsification",
                    entry.hypothesis_id,
                )
                continue

            target_dim_idx = self._resolve_target_dim(
                entry.target_variable, state_feature_names,
            )

            # Test 1: Structural consistency
            struct_result = structural_test.test(
                expr, dataset, target_dim_idx, variable_names,
            )
            if not struct_result.passed:
                register.update_status(
                    entry.hypothesis_id,
                    HypothesisStatus.FALSIFIED,
                    reason=f"Structural consistency failed "
                    f"(F={struct_result.f_statistic:.2f}, "
                    f"p={struct_result.p_value:.6f})",
                )
                n_falsified += 1
                continue

            # Test 2: OOD prediction
            ood_result = ood_test.test(
                expr, dataset, target_dim_idx, variable_names,
            )
            if not ood_result.passed:
                register.update_status(
                    entry.hypothesis_id,
                    HypothesisStatus.FALSIFIED,
                    reason=f"OOD prediction failed "
                    f"(failure_fraction={ood_result.failure_fraction:.2f})",
                )
                n_falsified += 1
                continue

            # Test 3: Trajectory prediction (dynamics hypotheses only)
            if target_dim_idx >= 0:
                dyn_exprs = {target_dim_idx: expr}
                traj_result = trajectory_test.test(
                    dyn_exprs, dataset, state_feature_names, variable_names,
                )
                if not traj_result.passed:
                    register.update_status(
                        entry.hypothesis_id,
                        HypothesisStatus.FALSIFIED,
                        reason=f"Trajectory prediction failed "
                        f"(failure_fraction="
                        f"{traj_result.failure_fraction:.2f})",
                    )
                    n_falsified += 1
                    continue

            # All tests passed -> validate
            register.update_status(
                entry.hypothesis_id, HypothesisStatus.VALIDATED,
            )
            n_validated += 1

            # Score with MDL
            mdl_score = mdl_scorer.score(
                expr, dataset, target_dim_idx, variable_names,
            )
            register.set_mdl_score(entry.hypothesis_id, mdl_score.total)

        # Select best per target
        best_per_target: dict[str, str | None] = {}
        for target in register.target_variables:
            best = register.select_best(target)
            best_per_target[target] = (
                best.hypothesis_id if best is not None else None
            )

        logger.info(
            "Falsification complete: {}/{} validated, {}/{} falsified, "
            "best_per_target={}",
            n_validated, len(untested),
            n_falsified, len(untested),
            best_per_target,
        )

        return FalsificationResult(
            n_tested=len(untested),
            n_validated=n_validated,
            n_falsified=n_falsified,
            best_per_target=best_per_target,
        )

    @staticmethod
    def _resolve_target_dim(
        target_variable: str,
        state_feature_names: list[str],
    ) -> int:
        """Map target variable name to dimension index.

        :returns: Dimension index (>=0 for dynamics), or -1 for reward.
        """
        if target_variable == "reward":
            return -1

        # Parse "delta_sN" -> N
        if target_variable.startswith("delta_"):
            dim_name = target_variable[6:]  # strip "delta_"
            if dim_name in state_feature_names:
                return state_feature_names.index(dim_name)

        raise ValueError(
            f"Cannot resolve target variable '{target_variable}' "
            f"to a state dimension index. Expected 'reward' or "
            f"'delta_<dim_name>' where dim_name is in "
            f"{state_feature_names}"
        )
