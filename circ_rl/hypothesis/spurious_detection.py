"""Spurious term detection and pruning for symbolic expressions.

Identifies additive terms in SR-discovered expressions that contribute
negligibly to R2 and may represent discretization artifacts (e.g.,
O(dt^2) cross-terms from semi-implicit Euler integration).

Detection heuristic: for each additive term t_i in expression
h(x) = t_1 + t_2 + ... + t_k, compute the R2 ablation:

    delta_R2_i = R2(h) - R2(h - t_i)

Terms with delta_R2_i < threshold are flagged as spurious.

See ``CIRC-RL_Framework.md`` Section 7.2 (Spurious Term Detection).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import sympy
from loguru import logger

if TYPE_CHECKING:
    from circ_rl.environments.data_collector import ExploratoryDataset
    from circ_rl.hypothesis.expression import SymbolicExpression


@dataclass(frozen=True)
class TermAnalysis:
    """Analysis of a single additive term.

    :param term_expr: The sympy sub-expression for this term.
    :param term_str: String representation.
    :param r2_contribution: R2 drop when this term is removed.
    :param coefficient_magnitude: Absolute value of the leading numeric
        coefficient (0.0 if no numeric factor).
    :param is_spurious: Whether this term was flagged as spurious.
    """

    term_expr: sympy.Expr
    term_str: str
    r2_contribution: float
    coefficient_magnitude: float
    is_spurious: bool


@dataclass(frozen=True)
class SpuriousDetectionResult:
    """Result of spurious term detection.

    :param original_expr: The original sympy expression.
    :param pruned_expr: Expression with spurious terms removed.
    :param term_analyses: Per-term analysis results.
    :param n_spurious: Number of terms flagged as spurious.
    :param original_r2: R2 of the original expression.
    :param pruned_r2: R2 of the pruned expression.
    """

    original_expr: sympy.Expr
    pruned_expr: sympy.Expr
    term_analyses: list[TermAnalysis]
    n_spurious: int
    original_r2: float
    pruned_r2: float


class SpuriousTermDetector:
    """Detect and prune spurious additive terms from symbolic expressions.

    For each additive term, evaluates its R2 contribution via ablation
    (removing the term and measuring the R2 drop). Terms with negligible
    contribution are flagged as spurious and can be pruned.

    Uses calibrated R2 (``y = alpha * h(x) + beta``) to separate
    functional form quality from exact coefficient precision.

    See ``CIRC-RL_Framework.md`` Section 7.2.

    :param r2_contribution_threshold: Minimum R2 contribution for a
        term to be considered meaningful. Terms below this are flagged.
        Default 0.005.
    :param coefficient_ratio_threshold: If a term's leading coefficient
        magnitude is less than this fraction of the largest term's
        coefficient, flag it. Default 0.01. Set to 0.0 to disable.
    :param min_terms_to_keep: Never prune below this many terms.
        Default 1.
    """

    def __init__(
        self,
        r2_contribution_threshold: float = 0.005,
        coefficient_ratio_threshold: float = 0.01,
        min_terms_to_keep: int = 1,
    ) -> None:
        if r2_contribution_threshold < 0 or r2_contribution_threshold > 1:
            raise ValueError(
                "r2_contribution_threshold must be in [0, 1], "
                f"got {r2_contribution_threshold}"
            )
        if min_terms_to_keep < 1:
            raise ValueError(
                f"min_terms_to_keep must be >= 1, "
                f"got {min_terms_to_keep}"
            )
        self._r2_threshold = r2_contribution_threshold
        self._coeff_ratio_threshold = coefficient_ratio_threshold
        self._min_terms = min_terms_to_keep

    def detect(
        self,
        expression: SymbolicExpression,
        dataset: ExploratoryDataset,
        target_dim_idx: int,
        variable_names: list[str],
        derived_columns: dict[str, np.ndarray] | None = None,
        wrap_angular: bool = False,
    ) -> SpuriousDetectionResult:
        """Analyze and flag spurious terms in the expression.

        :param expression: The symbolic expression to analyze.
        :param dataset: Multi-environment data for R2 computation.
        :param target_dim_idx: Target state dimension (>=0) or -1
            for reward.
        :param variable_names: Variable names for expression evaluation.
        :param derived_columns: Pre-computed derived feature arrays.
        :param wrap_angular: Whether to wrap target deltas via atan2.
        :returns: SpuriousDetectionResult with per-term analysis.
        """
        from circ_rl.hypothesis.expression import SymbolicExpression as _SE
        from circ_rl.hypothesis.structural_consistency import (
            StructuralConsistencyTest,
        )

        # Build targets and features using existing infrastructure
        targets = _build_targets(dataset, target_dim_idx, wrap_angular)
        features = StructuralConsistencyTest._build_features(
            dataset, variable_names, derived_columns,
        )

        # Decompose expression into additive terms
        sympy_expr = expression.sympy_expr
        terms = list(sympy.Add.make_args(sympy_expr))

        if len(terms) <= 1:
            # Single term or constant: nothing to prune
            full_r2 = _compute_calibrated_r2(
                expression, features, targets, variable_names,
            )
            return SpuriousDetectionResult(
                original_expr=sympy_expr,
                pruned_expr=sympy_expr,
                term_analyses=[],
                n_spurious=0,
                original_r2=full_r2,
                pruned_r2=full_r2,
            )

        # Compute full R2
        full_r2 = _compute_calibrated_r2(
            expression, features, targets, variable_names,
        )

        # Extract leading coefficient magnitudes for each term
        coeff_magnitudes: list[float] = []
        for term in terms:
            coeff, _ = term.as_coeff_Mul()
            coeff_magnitudes.append(abs(float(coeff)))
        max_coeff = max(coeff_magnitudes) if coeff_magnitudes else 1.0

        # Ablation: compute R2 with each term removed
        term_analyses: list[TermAnalysis] = []
        for i, term in enumerate(terms):
            ablated_expr = sympy_expr - term
            if ablated_expr == sympy.S.Zero:
                r2_ablated = 0.0
            else:
                ablated_se = _SE.from_sympy(ablated_expr)
                r2_ablated = _compute_calibrated_r2(
                    ablated_se, features, targets, variable_names,
                )

            r2_contribution = full_r2 - r2_ablated
            coeff_ratio = (
                coeff_magnitudes[i] / max_coeff if max_coeff > 0 else 1.0
            )

            is_spurious = (
                r2_contribution < self._r2_threshold
                and (
                    self._coeff_ratio_threshold <= 0
                    or coeff_ratio < self._coeff_ratio_threshold
                )
            )

            term_analyses.append(TermAnalysis(
                term_expr=term,
                term_str=str(term),
                r2_contribution=r2_contribution,
                coefficient_magnitude=coeff_magnitudes[i],
                is_spurious=is_spurious,
            ))

        # Ensure we keep at least min_terms
        n_non_spurious = sum(
            1 for ta in term_analyses if not ta.is_spurious
        )
        if n_non_spurious < self._min_terms:
            # Sort by R2 contribution descending and un-flag top terms
            sorted_by_r2 = sorted(
                range(len(term_analyses)),
                key=lambda j: term_analyses[j].r2_contribution,
                reverse=True,
            )
            final_analyses: list[TermAnalysis] = list(term_analyses)
            kept = 0
            for idx in sorted_by_r2:
                if kept >= self._min_terms:
                    break
                if final_analyses[idx].is_spurious:
                    ta = final_analyses[idx]
                    final_analyses[idx] = TermAnalysis(
                        term_expr=ta.term_expr,
                        term_str=ta.term_str,
                        r2_contribution=ta.r2_contribution,
                        coefficient_magnitude=ta.coefficient_magnitude,
                        is_spurious=False,
                    )
                kept += 1 if not final_analyses[idx].is_spurious else 0
            # Recount
            for idx in sorted_by_r2:
                if not final_analyses[idx].is_spurious:
                    kept += 1
                if kept >= self._min_terms:
                    break
            term_analyses = final_analyses

        # Build pruned expression
        kept_terms = [
            ta.term_expr for ta in term_analyses if not ta.is_spurious
        ]
        if not kept_terms:
            pruned_expr = sympy_expr
        else:
            pruned_expr = sympy.Add(*kept_terms)

        pruned_se = _SE.from_sympy(pruned_expr)
        pruned_r2 = _compute_calibrated_r2(
            pruned_se, features, targets, variable_names,
        )

        n_spurious = sum(1 for ta in term_analyses if ta.is_spurious)

        if n_spurious > 0:
            logger.info(
                "Spurious detection: {}/{} terms flagged "
                "(R2: {:.6f} -> {:.6f})",
                n_spurious, len(terms),
                full_r2, pruned_r2,
            )

        return SpuriousDetectionResult(
            original_expr=sympy_expr,
            pruned_expr=pruned_expr,
            term_analyses=term_analyses,
            n_spurious=n_spurious,
            original_r2=full_r2,
            pruned_r2=pruned_r2,
        )


def _build_targets(
    dataset: ExploratoryDataset,
    target_dim_idx: int,
    wrap_angular: bool,
) -> np.ndarray:
    """Build target array from dataset.

    :returns: Array of shape ``(N,)``.
    """
    if target_dim_idx >= 0:
        targets = (
            dataset.next_states[:, target_dim_idx]
            - dataset.states[:, target_dim_idx]
        )  # (N,)
        if wrap_angular:
            targets = np.arctan2(np.sin(targets), np.cos(targets))
    else:
        targets = dataset.rewards  # (N,)
    return targets


def _compute_calibrated_r2(
    expression: SymbolicExpression,
    features: np.ndarray,
    targets: np.ndarray,
    variable_names: list[str],
) -> float:
    """Compute calibrated R2 of expression via ``y = alpha * h(x) + beta``.

    This tests the functional form quality rather than exact coefficient
    precision, matching the approach used in the structural consistency test.

    :returns: Calibrated R2. Returns ``-inf`` on evaluation failure.
    """
    try:
        fn = expression.to_callable(variable_names)
        h_x = np.asarray(fn(features), dtype=np.float64).ravel()
    except Exception:
        return -float("inf")

    n = len(targets)
    if h_x.shape[0] == 1:
        h_x = np.broadcast_to(h_x, (n,)).copy()

    if not np.all(np.isfinite(h_x)):
        return -float("inf")

    design = np.column_stack([np.ones(n), h_x])  # (N, 2)
    try:
        coeffs = np.linalg.lstsq(design, targets, rcond=None)[0]
    except np.linalg.LinAlgError:
        return -float("inf")

    y_pred = design @ coeffs
    ss_res = float(np.sum((targets - y_pred) ** 2))
    ss_tot = float(np.sum((targets - np.mean(targets)) ** 2))

    if ss_tot < 1e-12:
        return 1.0 if ss_res < 1e-12 else -float("inf")

    return 1.0 - ss_res / ss_tot
