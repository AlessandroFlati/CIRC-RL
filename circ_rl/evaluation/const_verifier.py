"""Constraint verification for filtering policies in the ensemble.

Verifies that policies satisfy hard constraints with a confidence margin
before including them in the final ensemble. Policies that violate
constraints are excluded.

See ``CIRC-RL_Framework.md`` Section 3.6, Phase 4.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from circ_rl.constraints.const_set import ConstraintSet
    from circ_rl.policy.causal_policy import CausalPolicy
    from circ_rl.training.trajectory_buffer import Trajectory


@dataclass(frozen=True)
class VerificationResult:
    """Result of constraint verification for a single policy.

    :param policy_idx: Index of the policy.
    :param feasible: Whether the policy satisfies all constraints.
    :param violations: Mapping of constraint name to violation amount.
    :param margin_violations: Constraints violated with k-sigma margin.
    """

    policy_idx: int
    feasible: bool
    violations: dict[str, float]
    margin_violations: list[str]


class ConstraintVerifier:
    """Verify policy feasibility with confidence margins.

    :param constraint_set: The set of constraints to verify.
    :param confidence_margin_k: Number of standard deviations for the
        safety margin (higher = more conservative).
    """

    def __init__(
        self,
        constraint_set: ConstraintSet,
        confidence_margin_k: float = 2.0,
    ) -> None:
        self._constraint_set = constraint_set
        self._margin_k = confidence_margin_k

    def verify(
        self,
        policy: CausalPolicy,
        evaluation_trajectory: Trajectory,
        policy_idx: int = 0,
    ) -> VerificationResult:
        """Verify a single policy against all constraints.

        A policy is feasible if for every constraint:
            E[C_i] + k * std(C_i) / sqrt(n) <= threshold

        :param policy: The policy to verify.
        :param evaluation_trajectory: Trajectory for evaluation.
        :param policy_idx: Index for identification.
        :returns: VerificationResult.
        """
        policy.eval()

        states = evaluation_trajectory.states
        actions = evaluation_trajectory.actions
        rewards = evaluation_trajectory.rewards
        next_states = evaluation_trajectory.next_states

        # For constraint evaluation, actions may need reshaping
        actions_for_eval = actions.unsqueeze(-1) if actions.dim() == 1 else actions

        violations: dict[str, float] = {}
        margin_violations: list[str] = []
        feasible = True

        for constraint in self._constraint_set.constraints:
            costs = constraint.evaluate(
                states, actions_for_eval, rewards, next_states
            )  # (batch,)

            mean_cost = float(costs.mean().item())
            std_cost = float(costs.std().item())
            n = costs.shape[0]

            # Violation with margin
            margin = self._margin_k * std_cost / (n ** 0.5)
            upper_bound = mean_cost + margin
            violation = upper_bound - constraint.threshold

            violations[constraint.name] = violation

            if violation > 0:
                feasible = False
                margin_violations.append(constraint.name)
                logger.debug(
                    "Policy {}: constraint '{}' violated "
                    "(mean={:.4f}, margin={:.4f}, threshold={:.4f})",
                    policy_idx,
                    constraint.name,
                    mean_cost,
                    margin,
                    constraint.threshold,
                )

        policy.train()

        return VerificationResult(
            policy_idx=policy_idx,
            feasible=feasible,
            violations=violations,
            margin_violations=margin_violations,
        )

    def filter_feasible(
        self,
        policies: list[CausalPolicy],
        evaluation_trajectory: Trajectory,
    ) -> list[tuple[int, CausalPolicy]]:
        """Filter policies to only those satisfying all constraints.

        :param policies: List of candidate policies.
        :param evaluation_trajectory: Trajectory for evaluation.
        :returns: List of (index, policy) for feasible policies.
        """
        feasible: list[tuple[int, CausalPolicy]] = []

        for i, policy in enumerate(policies):
            result = self.verify(policy, evaluation_trajectory, policy_idx=i)
            if result.feasible:
                feasible.append((i, policy))

        logger.info(
            "Constraint verification: {}/{} policies feasible",
            len(feasible),
            len(policies),
        )

        return feasible
