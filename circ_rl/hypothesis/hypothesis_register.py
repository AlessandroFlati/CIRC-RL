"""Hypothesis register for tracking candidate hypotheses.

Stores, queries, and manages candidate hypotheses through their lifecycle
(untested -> validated / falsified). Supports Pareto front extraction
and MDL-based selection.

See ``CIRC-RL_Framework.md`` Section 3.4.3 (The Hypothesis Register)
and Section 3.5.4 (Selection Among Surviving Hypotheses).
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

from loguru import logger


class HypothesisStatus(enum.Enum):
    """Lifecycle status of a hypothesis."""

    UNTESTED = "untested"
    VALIDATED = "validated"
    FALSIFIED = "falsified"


@dataclass
class HypothesisEntry:
    """A registered hypothesis with metadata and test results.

    See ``CIRC-RL_Framework.md`` Section 3.4.3.

    :param hypothesis_id: Unique identifier for this hypothesis.
    :param target_variable: The variable this hypothesis predicts
        (e.g., ``"delta_s0"``, ``"delta_s1"``, ``"reward"``).
    :param expression: The symbolic expression for this hypothesis.
    :param complexity: Symbolic complexity (:math:`C_{\\text{sym}}`).
    :param training_r2: :math:`R^2` on pooled training data.
    :param training_mse: Mean squared error on pooled training data.
    :param status: Current lifecycle status.
    :param falsification_reason: Human-readable reason for falsification,
        or None if not falsified.
    :param mdl_score: MDL score after falsification, or None if not scored.
    """

    hypothesis_id: str
    target_variable: str
    expression: object  # SymbolicExpression (avoid circular import at type level)
    complexity: int
    training_r2: float
    training_mse: float
    status: HypothesisStatus = HypothesisStatus.UNTESTED
    falsification_reason: str | None = None
    mdl_score: float | None = None


class HypothesisRegister:
    """Registry for managing candidate hypotheses.

    Tracks hypotheses through their lifecycle, extracts Pareto fronts
    (accuracy vs. complexity), and selects the best hypothesis per
    target variable using MDL.

    See ``CIRC-RL_Framework.md`` Section 3.4.3 and Section 3.5.4.
    """

    def __init__(self) -> None:
        self._entries: dict[str, HypothesisEntry] = {}
        self._counter = 0

    @property
    def entries(self) -> dict[str, HypothesisEntry]:
        """All registered entries keyed by hypothesis_id."""
        return dict(self._entries)

    def register(self, entry: HypothesisEntry) -> str:
        """Add a hypothesis to the register.

        :param entry: The hypothesis entry to register.
        :returns: The hypothesis_id assigned to this entry.
        :raises ValueError: If a hypothesis with the same ID already exists.
        """
        if entry.hypothesis_id in self._entries:
            raise ValueError(
                f"Hypothesis '{entry.hypothesis_id}' already registered"
            )
        self._entries[entry.hypothesis_id] = entry
        self._counter += 1
        logger.debug(
            "Registered hypothesis '{}' for target '{}' "
            "(complexity={}, R2={:.4f})",
            entry.hypothesis_id,
            entry.target_variable,
            entry.complexity,
            entry.training_r2,
        )
        return entry.hypothesis_id

    def get(self, hypothesis_id: str) -> HypothesisEntry:
        """Retrieve a hypothesis by ID.

        :param hypothesis_id: The hypothesis ID.
        :returns: The hypothesis entry.
        :raises KeyError: If not found.
        """
        return self._entries[hypothesis_id]

    def update_status(
        self,
        hypothesis_id: str,
        status: HypothesisStatus,
        reason: str | None = None,
    ) -> None:
        """Update the status of a hypothesis.

        :param hypothesis_id: The hypothesis ID.
        :param status: New status.
        :param reason: Reason for falsification (required when status is FALSIFIED).
        :raises KeyError: If hypothesis_id not found.
        :raises ValueError: If transition is invalid (e.g., FALSIFIED -> VALIDATED).
        """
        entry = self._entries[hypothesis_id]

        # Validate transitions
        if entry.status == HypothesisStatus.FALSIFIED:
            raise ValueError(
                f"Cannot change status of falsified hypothesis "
                f"'{hypothesis_id}'"
            )
        if (
            entry.status == HypothesisStatus.VALIDATED
            and status == HypothesisStatus.UNTESTED
        ):
            raise ValueError(
                f"Cannot revert validated hypothesis '{hypothesis_id}' "
                f"to untested"
            )

        entry.status = status
        if status == HypothesisStatus.FALSIFIED:
            entry.falsification_reason = reason

        logger.debug(
            "Hypothesis '{}' status -> {} {}",
            hypothesis_id,
            status.value,
            f"(reason: {reason})" if reason else "",
        )

    def set_mdl_score(self, hypothesis_id: str, score: float) -> None:
        """Set the MDL score for a hypothesis.

        :param hypothesis_id: The hypothesis ID.
        :param score: The MDL score (lower is better).
        :raises KeyError: If hypothesis_id not found.
        """
        self._entries[hypothesis_id].mdl_score = score

    def get_by_target(self, target_variable: str) -> list[HypothesisEntry]:
        """Get all hypotheses for a target variable.

        :param target_variable: The target variable name.
        :returns: List of hypothesis entries for this target.
        """
        return [
            e for e in self._entries.values()
            if e.target_variable == target_variable
        ]

    def get_by_status(self, status: HypothesisStatus) -> list[HypothesisEntry]:
        """Get all hypotheses with a given status.

        :param status: The status to filter by.
        :returns: List of hypothesis entries with this status.
        """
        return [
            e for e in self._entries.values()
            if e.status == status
        ]

    def pareto_front(
        self,
        target_variable: str,
        status_filter: HypothesisStatus | None = None,
    ) -> list[HypothesisEntry]:
        """Extract the Pareto front of accuracy vs. complexity.

        Returns hypotheses that are not dominated by any other hypothesis
        on both axes (lower complexity AND higher R2).

        :param target_variable: Filter to this target variable.
        :param status_filter: If set, only consider hypotheses with this status.
        :returns: Pareto-optimal hypotheses, sorted by complexity (ascending).
        """
        candidates = self.get_by_target(target_variable)
        if status_filter is not None:
            candidates = [c for c in candidates if c.status == status_filter]

        if not candidates:
            return []

        # Sort by complexity ascending, then R2 descending for tie-breaking
        candidates.sort(key=lambda e: (e.complexity, -e.training_r2))

        pareto: list[HypothesisEntry] = []
        best_r2 = -float("inf")

        for entry in candidates:
            if entry.training_r2 > best_r2:
                pareto.append(entry)
                best_r2 = entry.training_r2

        return pareto

    def select_best(
        self,
        target_variable: str,
    ) -> HypothesisEntry | None:
        r"""Select the best validated hypothesis for a target using MDL.

        Among validated hypotheses for the target variable, returns the
        one with the lowest MDL score. If no MDL scores are set, falls
        back to the simplest hypothesis with the highest :math:`R^2`.

        See ``CIRC-RL_Framework.md`` Section 3.5.4.

        :param target_variable: The target variable.
        :returns: The best hypothesis, or None if no validated hypotheses exist.
        """
        validated = [
            e for e in self.get_by_target(target_variable)
            if e.status == HypothesisStatus.VALIDATED
        ]
        if not validated:
            return None

        # Prefer MDL-scored hypotheses
        scored = [e for e in validated if e.mdl_score is not None]
        if scored:
            return min(scored, key=lambda e: e.mdl_score)  # type: ignore[arg-type,return-value]

        # Fallback: Pareto front, pick simplest with best R2
        pareto = self.pareto_front(
            target_variable, status_filter=HypothesisStatus.VALIDATED
        )
        if pareto:
            return pareto[0]

        return validated[0]

    @property
    def n_hypotheses(self) -> int:
        """Total number of registered hypotheses."""
        return len(self._entries)

    @property
    def target_variables(self) -> list[str]:
        """Unique target variables across all hypotheses."""
        return sorted({e.target_variable for e in self._entries.values()})

    def __repr__(self) -> str:
        status_counts = {}
        for e in self._entries.values():
            status_counts[e.status.value] = status_counts.get(e.status.value, 0) + 1
        return (
            f"HypothesisRegister(n={self.n_hypotheses}, "
            f"targets={self.target_variables}, "
            f"status={status_counts})"
        )
