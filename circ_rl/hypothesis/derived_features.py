"""Derived feature specifications for hypothesis generation.

Supports pre-computing features from existing state variables (e.g.,
``theta = atan2(sin_theta, cos_theta)``) for use in symbolic regression,
falsification, and analytic policy derivation.

See ``CIRC-RL_Framework.md`` Section 3.4 (Structural Hypothesis
Generation).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True)
class DerivedFeatureSpec:
    """Specification for a feature derived from existing state variables.

    :param name: Variable name for the derived feature (e.g., ``"theta"``).
    :param source_names: Names of source variables the feature depends on.
        These must be present in the state feature names or variable names.
    :param compute_fn: Callable that computes the derived feature from
        source arrays. Takes one 1D array per source name and returns
        a 1D array of the same length. For example,
        ``np.arctan2`` for ``theta = atan2(sin_theta, cos_theta)``.
    :param sympy_fn: Optional sympy equivalent of ``compute_fn``,
        used for building analytic reward derivatives. For example,
        ``sympy.atan2`` for ``np.arctan2``. When provided, enables
        symbolic composition of the reward expression through the
        derived feature for exact differentiation.
    """

    name: str
    source_names: tuple[str, ...]
    compute_fn: Callable[..., Any]
    sympy_fn: Callable[..., Any] | None = None


def compute_derived_columns(
    specs: list[DerivedFeatureSpec],
    states: np.ndarray,
    state_names: list[str],
) -> dict[str, np.ndarray]:
    """Pre-compute derived feature columns from state data.

    :param specs: List of derived feature specifications.
    :param states: State array of shape ``(N, state_dim)``.
    :param state_names: Names of state features (columns of ``states``).
    :returns: Dict mapping derived feature name to ``(N,)`` array.
    :raises ValueError: If a source variable is not found in state_names.
    """
    result: dict[str, np.ndarray] = {}

    for spec in specs:
        source_arrays: list[np.ndarray] = []
        for src_name in spec.source_names:
            if src_name not in state_names:
                raise ValueError(
                    f"Derived feature '{spec.name}' requires source "
                    f"'{src_name}' which is not in state_names "
                    f"{state_names}"
                )
            idx = state_names.index(src_name)
            source_arrays.append(states[:, idx])

        computed = spec.compute_fn(*source_arrays)
        result[spec.name] = np.asarray(computed, dtype=np.float64).ravel()

    return result


def compute_derived_single(
    specs: list[DerivedFeatureSpec],
    state: np.ndarray,
    state_names: list[str],
) -> dict[str, float]:
    """Compute derived features for a single state vector.

    Used at runtime by the MPC solver to augment the state with
    derived features before evaluating reward expressions.

    :param specs: List of derived feature specifications.
    :param state: State vector of shape ``(state_dim,)``.
    :param state_names: Names of state features.
    :returns: Dict mapping derived feature name to scalar value.
    :raises ValueError: If a source variable is not found in state_names.
    """
    result: dict[str, float] = {}

    for spec in specs:
        source_values: list[float] = []
        for src_name in spec.source_names:
            if src_name not in state_names:
                raise ValueError(
                    f"Derived feature '{spec.name}' requires source "
                    f"'{src_name}' which is not in state_names "
                    f"{state_names}"
                )
            idx = state_names.index(src_name)
            source_values.append(float(state[idx]))

        computed = spec.compute_fn(*source_values)
        result[spec.name] = float(computed)

    return result
