"""Observation space analysis: constraint detection and canonical reparametrization.

Detects algebraic constraints among observation dimensions (e.g.,
``s0^2 + s1^2 = 1`` for cos/sin encodings) using polynomial PCA, classifies
the underlying manifold (circle, linear, etc.), and builds canonical
coordinate mappings (e.g., ``theta = atan2(s1, s0)``).

Dynamics discovered in canonical coordinates are simpler and more exact
than in the original (often redundant) observation space.

See ``CIRC-RL_Framework.md`` Section 3.3.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from circ_rl.environments.data_collector import ExploratoryDataset


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ObservationAnalysisConfig:
    """Configuration for observation space analysis.

    :param singular_value_threshold: Ratio ``sigma_i / sigma_max`` below
        which a singular value is considered zero (indicating a constraint).
        Default 1e-3.
    :param circle_tolerance: Maximum relative difference between the two
        diagonal coefficients to classify a quadratic constraint as a circle.
        Default 0.1 (10%).
    :param cross_term_tolerance: Maximum ratio of cross-term magnitude to
        diagonal magnitude for circle classification. Default 0.1.
    :param min_samples: Minimum samples required for reliable SVD.
        Default 500.
    :param max_polynomial_degree: Maximum degree of polynomial features
        for constraint detection. Currently only degree 2 is supported.
        Default 2.
    """

    singular_value_threshold: float = 1e-3
    circle_tolerance: float = 0.1
    cross_term_tolerance: float = 0.1
    min_samples: int = 500
    max_polynomial_degree: int = 2


@dataclass(frozen=True)
class AlgebraicConstraint:
    """A detected algebraic constraint among observation dimensions.

    Example: ``s0^2 + s1^2 = 1`` is represented as
    ``involved_dims=(0, 1)``, ``constraint_type="circle"``,
    ``coefficients={('s0','s0'): 1.0, ('s1','s1'): 1.0}``,
    ``rhs=1.0``.

    :param involved_dims: Observation dimension indices participating
        in this constraint.
    :param constraint_type: Classification of the constraint manifold.
        One of ``"circle"``, ``"linear"``, ``"quadratic"``.
    :param coefficients: Mapping from monomial tuple to coefficient.
        For a quadratic constraint like ``a*x^2 + b*y^2 = c``,
        keys are ``('x','x')`` for ``x^2``, ``('x','y')`` for ``x*y``, etc.
        Linear terms use single-element tuples: ``('x',)``.
    :param rhs: Right-hand side constant of the constraint equation.
    :param residual_std: Standard deviation of constraint residuals across
        the dataset, measuring how well the data satisfies the constraint.
    """

    involved_dims: tuple[int, ...]
    constraint_type: str
    coefficients: dict[tuple[str, ...], float]
    rhs: float
    residual_std: float


@dataclass(frozen=True)
class CanonicalMapping:
    """Maps constrained observation dimensions to canonical coordinates.

    Example: ``[cos(theta), sin(theta)]`` -> ``[theta]`` via
    ``atan2(sin, cos)``.

    :param source_dims: Observation dimension indices consumed by this
        mapping (e.g., ``(0, 1)`` for dims 0 and 1).
    :param canonical_names: Names of the canonical coordinates produced
        (e.g., ``("phi_0",)``).
    :param obs_to_canonical: Function mapping observation subset
        ``(n_samples, len(source_dims))`` to canonical coordinates
        ``(n_samples, len(canonical_names))``.
    :param canonical_to_obs: Inverse function mapping canonical coordinates
        back to observation subset.
    :param is_angular: Per canonical coordinate, whether it represents an
        angle requiring wrapping for delta computation.
    """

    source_dims: tuple[int, ...]
    canonical_names: tuple[str, ...]
    obs_to_canonical: Callable[..., Any]
    canonical_to_obs: Callable[..., Any]
    is_angular: tuple[bool, ...]


@dataclass(frozen=True)
class ObservationAnalysisResult:
    """Result of observation space analysis.

    :param constraints: Detected algebraic constraints.
    :param mappings: Canonical coordinate mappings for classified constraints.
    :param canonical_state_names: Names of canonical state dimensions,
        combining mapped coordinates and unconstrained originals.
        E.g., ``["phi_0", "s2"]`` for Pendulum-v1.
    :param canonical_states: Canonical state representations for the
        dataset, shape ``(N, canonical_dim)``.
    :param canonical_next_states: Canonical next-state representations,
        shape ``(N, canonical_dim)``.
    :param unconstrained_dims: Observation dimensions not involved in
        any detected constraint, passed through unchanged.
    :param obs_to_canonical_fn: Function converting a full observation
        vector ``(obs_dim,)`` to canonical coordinates ``(canonical_dim,)``.
    :param canonical_to_obs_fn: Function converting canonical coordinates
        back to observation space.
    :param angular_dims: Indices into the canonical state vector that
        represent angular coordinates (need wrapping for deltas).
    """

    constraints: list[AlgebraicConstraint]
    mappings: list[CanonicalMapping]
    canonical_state_names: list[str]
    canonical_states: np.ndarray
    canonical_next_states: np.ndarray
    unconstrained_dims: tuple[int, ...]
    obs_to_canonical_fn: Callable[..., Any]
    canonical_to_obs_fn: Callable[..., Any]
    angular_dims: tuple[int, ...] = field(default=())


# ---------------------------------------------------------------------------
# Main analyzer
# ---------------------------------------------------------------------------


class ObservationAnalyzer:
    """Detect algebraic constraints and build canonical reparametrizations.

    Uses polynomial PCA (degree-2 polynomial features + SVD) to detect
    near-zero-dimensional structure in the observation space, then
    classifies the geometry (circle, linear, etc.) and builds
    invertible mappings to canonical coordinates.

    :param config: Analysis configuration.
    """

    def __init__(
        self,
        config: ObservationAnalysisConfig | None = None,
    ) -> None:
        self._config = config or ObservationAnalysisConfig()

    def analyze(
        self,
        dataset: ExploratoryDataset,
        state_names: list[str],
    ) -> ObservationAnalysisResult:
        """Run observation space analysis on collected data.

        :param dataset: Multi-environment exploratory dataset.
        :param state_names: Names of state dimensions.
        :returns: Analysis result with constraints, mappings, and
            canonical dataset.
        :raises ValueError: If dataset has fewer samples than
            ``config.min_samples``.
        """
        cfg = self._config
        states = dataset.states  # (N, obs_dim)
        n_samples, obs_dim = states.shape

        if n_samples < cfg.min_samples:
            raise ValueError(
                f"Need at least {cfg.min_samples} samples for observation "
                f"analysis, got {n_samples}"
            )

        if obs_dim < 2:
            logger.info(
                "Observation dim={}, too small for constraint detection",
                obs_dim,
            )
            return self._no_constraints_result(dataset, state_names)

        # Step 1: Build degree-2 polynomial features and detect constraints
        constraints = self._detect_constraints(states, state_names)

        if not constraints:
            logger.info("No algebraic constraints detected in observation space")
            return self._no_constraints_result(dataset, state_names)

        logger.info(
            "Detected {} algebraic constraint(s): {}",
            len(constraints),
            [(c.constraint_type, c.involved_dims) for c in constraints],
        )

        # Step 2: Build canonical mappings for classified constraints
        mappings = self._build_mappings(constraints, state_names)

        if not mappings:
            logger.info("No constraints could be mapped to canonical coordinates")
            result = self._no_constraints_result(dataset, state_names)
            # Still report detected constraints even without mappings
            return ObservationAnalysisResult(
                constraints=constraints,
                mappings=[],
                canonical_state_names=result.canonical_state_names,
                canonical_states=result.canonical_states,
                canonical_next_states=result.canonical_next_states,
                unconstrained_dims=result.unconstrained_dims,
                obs_to_canonical_fn=result.obs_to_canonical_fn,
                canonical_to_obs_fn=result.canonical_to_obs_fn,
                angular_dims=(),
            )

        # Step 3: Build canonical state vectors and names
        return self._build_canonical_result(
            dataset, state_names, constraints, mappings,
        )

    # ------------------------------------------------------------------
    # Constraint detection
    # ------------------------------------------------------------------

    def _detect_constraints(
        self,
        states: np.ndarray,
        state_names: list[str],
    ) -> list[AlgebraicConstraint]:
        """Detect algebraic constraints by direct geometric testing.

        Tests all pairs of dimensions for circle constraints
        (``si^2 + sj^2 = const``) and uses PCA on the raw data to
        detect linear constraints.

        :param states: Observation data, shape ``(N, obs_dim)``.
        :param state_names: Names of observation dimensions.
        :returns: List of detected constraints.
        """
        constraints: list[AlgebraicConstraint] = []

        # Circle detection: test all pairs
        circle_constraints = self._detect_circle_constraints(
            states, state_names,
        )
        constraints.extend(circle_constraints)

        # Linear detection: PCA on raw data
        circle_dims: set[int] = set()
        for c in circle_constraints:
            circle_dims.update(c.involved_dims)

        linear_constraints = self._detect_linear_constraints(
            states, state_names, exclude_dims=circle_dims,
        )
        constraints.extend(linear_constraints)

        return constraints

    def _detect_circle_constraints(
        self,
        states: np.ndarray,
        state_names: list[str],
    ) -> list[AlgebraicConstraint]:
        """Detect circle constraints by testing all pairs of dims.

        For each pair ``(i, j)``, checks whether
        ``std(si^2 + sj^2) / mean(si^2 + sj^2)`` is below threshold.

        :param states: Shape ``(N, obs_dim)``.
        :param state_names: Dimension names.
        :returns: Circle constraints found.
        """
        cfg = self._config
        obs_dim = states.shape[1]
        constraints: list[AlgebraicConstraint] = []
        used_dims: set[int] = set()

        for i in range(obs_dim):
            if i in used_dims:
                continue
            for j in range(i + 1, obs_dim):
                if j in used_dims:
                    continue

                r2 = states[:, i] ** 2 + states[:, j] ** 2  # (N,)
                mean_r2 = float(np.mean(r2))

                if mean_r2 < 1e-10:
                    continue

                relative_std = float(np.std(r2)) / mean_r2

                if relative_std > cfg.circle_tolerance:
                    continue

                residual_std = float(np.std(r2 - mean_r2))

                logger.info(
                    "Detected circle constraint: {}^2 + {}^2 = {:.4f} "
                    "(relative_std={:.6f}, residual_std={:.6f})",
                    state_names[i], state_names[j],
                    mean_r2, relative_std, residual_std,
                )

                constraints.append(AlgebraicConstraint(
                    involved_dims=(i, j),
                    constraint_type="circle",
                    coefficients={
                        (state_names[i], state_names[i]): 1.0,
                        (state_names[j], state_names[j]): 1.0,
                    },
                    rhs=float(mean_r2),
                    residual_std=residual_std,
                ))
                used_dims.add(i)
                used_dims.add(j)
                break  # dim i is consumed, move to next

        return constraints

    def _detect_linear_constraints(
        self,
        states: np.ndarray,
        state_names: list[str],
        exclude_dims: set[int] | None = None,
    ) -> list[AlgebraicConstraint]:
        """Detect linear constraints via PCA on raw observation data.

        Performs SVD on centered observations and checks for near-zero
        singular values, which indicate linear dependencies among dims.

        :param states: Shape ``(N, obs_dim)``.
        :param state_names: Dimension names.
        :param exclude_dims: Dims already consumed by other constraints.
        :returns: Linear constraints found.
        """
        cfg = self._config
        exclude = exclude_dims or set()
        active_dims = [d for d in range(states.shape[1]) if d not in exclude]

        if len(active_dims) < 2:
            return []

        active_states = states[:, active_dims]  # (N, n_active)
        active_names = [state_names[d] for d in active_dims]

        # Center the data
        means = active_states.mean(axis=0)  # (n_active,)
        centered = active_states - means  # (N, n_active)

        # SVD
        _u, sigmas, vt = np.linalg.svd(centered, full_matrices=False)

        if sigmas[0] < 1e-12:
            return []

        threshold = cfg.singular_value_threshold * sigmas[0]
        constraints: list[AlgebraicConstraint] = []

        for idx in range(len(sigmas)):
            if sigmas[idx] >= threshold:
                continue

            # This singular vector defines a linear constraint
            v = vt[idx]  # (n_active,) -- normal to the constraint hyperplane
            # Constraint: v @ (x - means) = 0 => v @ x = v @ means

            # Identify involved dims (nonzero coefficients)
            involved_active = [
                k for k in range(len(active_dims))
                if abs(v[k]) > 1e-8
            ]
            if len(involved_active) < 2:
                continue

            involved_dims = tuple(active_dims[k] for k in involved_active)

            # Compute residual
            residual = centered @ v  # (N,)
            residual_std = float(np.std(residual))

            # Normalize by data scale
            data_scale = float(np.std(active_states[:, involved_active[0]]))
            if data_scale > 1e-10:
                relative_residual = residual_std / data_scale
            else:
                relative_residual = residual_std

            if relative_residual > 0.01:
                continue

            # Normalize so largest coefficient is 1
            max_coeff_idx = int(np.argmax(np.abs(v[involved_active])))
            norm_factor = v[involved_active[max_coeff_idx]]
            normalized_v = v / norm_factor

            coefficients: dict[tuple[str, ...], float] = {}
            for k in involved_active:
                coefficients[(active_names[k],)] = float(normalized_v[k])

            rhs = float(normalized_v @ means)

            logger.info(
                "Detected linear constraint involving dims {} "
                "(residual_std={:.6f})",
                involved_dims, residual_std,
            )

            constraints.append(AlgebraicConstraint(
                involved_dims=involved_dims,
                constraint_type="linear",
                coefficients=coefficients,
                rhs=rhs,
                residual_std=residual_std,
            ))

        return constraints

    # ------------------------------------------------------------------
    # Canonical mapping construction
    # ------------------------------------------------------------------

    def _build_mappings(
        self,
        constraints: list[AlgebraicConstraint],
        state_names: list[str],
    ) -> list[CanonicalMapping]:
        """Build canonical coordinate mappings for classified constraints.

        :param constraints: Detected constraints.
        :param state_names: Observation dimension names.
        :returns: List of canonical mappings (one per actionable constraint).
        """
        mappings: list[CanonicalMapping] = []
        mapped_dims: set[int] = set()
        canonical_counter = 0

        for constraint in constraints:
            # Skip if any dim already mapped
            if any(d in mapped_dims for d in constraint.involved_dims):
                continue

            if constraint.constraint_type == "circle":
                mapping = self._build_circle_mapping(
                    constraint, state_names, canonical_counter,
                )
                if mapping is not None:
                    mappings.append(mapping)
                    mapped_dims.update(constraint.involved_dims)
                    canonical_counter += 1

            # Linear constraints: could eliminate one dim, but this is
            # more complex and less common. Skip for now.

        return mappings

    def _build_circle_mapping(
        self,
        constraint: AlgebraicConstraint,
        state_names: list[str],
        canonical_idx: int,
    ) -> CanonicalMapping | None:
        """Build atan2-based canonical mapping for a circle constraint.

        For ``x^2 + y^2 = R^2``, maps ``(x, y)`` to ``phi = atan2(y, x)``.

        Convention: ``x = R*cos(phi)``, ``y = R*sin(phi)``,
        so ``phi = atan2(y, x) = atan2(sin_component, cos_component)``.
        The first dim in ``involved_dims`` is treated as cos (x),
        the second as sin (y).

        :param constraint: Circle constraint.
        :param state_names: Observation dimension names.
        :param canonical_idx: Index for naming the canonical coordinate.
        :returns: CanonicalMapping, or None if construction fails.
        """
        dim_cos, dim_sin = constraint.involved_dims
        r = float(np.sqrt(constraint.rhs))

        if r < 1e-10:
            return None

        canonical_name = f"phi_{canonical_idx}"

        def obs_to_canonical(obs_subset: np.ndarray) -> np.ndarray:
            """Map ``(cos_vals, sin_vals)`` to ``(angle,)``.

            :param obs_subset: Shape ``(N, 2)`` or ``(2,)``.
            """
            if obs_subset.ndim == 1:
                return np.array([np.arctan2(obs_subset[1], obs_subset[0])])
            return np.arctan2(
                obs_subset[:, 1], obs_subset[:, 0],
            ).reshape(-1, 1)  # (N, 1)

        def canonical_to_obs(canonical: np.ndarray) -> np.ndarray:
            """Map ``(angle,)`` to ``(cos_val, sin_val) * R``.

            :param canonical: Shape ``(N, 1)`` or ``(1,)``.
            """
            if canonical.ndim == 1:
                return np.array([
                    r * np.cos(canonical[0]),
                    r * np.sin(canonical[0]),
                ])
            phi = canonical[:, 0]  # (N,)
            return np.column_stack([
                r * np.cos(phi),
                r * np.sin(phi),
            ])  # (N, 2)

        return CanonicalMapping(
            source_dims=(dim_cos, dim_sin),
            canonical_names=(canonical_name,),
            obs_to_canonical=obs_to_canonical,
            canonical_to_obs=canonical_to_obs,
            is_angular=(True,),
        )

    # ------------------------------------------------------------------
    # Canonical dataset construction
    # ------------------------------------------------------------------

    def _build_canonical_result(
        self,
        dataset: ExploratoryDataset,
        state_names: list[str],
        constraints: list[AlgebraicConstraint],
        mappings: list[CanonicalMapping],
    ) -> ObservationAnalysisResult:
        """Construct the full canonical result with datasets and functions.

        Canonical state = [mapped_coords_0, mapped_coords_1, ..., unconstrained_dims]

        Angular deltas (for dynamics targets) use ``atan2(sin(d), cos(d))``
        wrapping to avoid discontinuities at +/-pi.

        :param dataset: Original exploratory dataset.
        :param state_names: Original state names.
        :param constraints: All detected constraints.
        :param mappings: Canonical mappings to apply.
        :returns: Complete analysis result.
        """
        obs_dim = dataset.states.shape[1]

        # Determine which dims are mapped vs unconstrained
        mapped_dims: set[int] = set()
        for m in mappings:
            mapped_dims.update(m.source_dims)
        unconstrained_dims = tuple(
            d for d in range(obs_dim) if d not in mapped_dims
        )

        # Build canonical state names
        canonical_names: list[str] = []
        angular_dim_indices: list[int] = []
        current_dim = 0
        for m in mappings:
            for i, name in enumerate(m.canonical_names):
                canonical_names.append(name)
                if m.is_angular[i]:
                    angular_dim_indices.append(current_dim)
                current_dim += 1
        for d in unconstrained_dims:
            canonical_names.append(state_names[d])
            current_dim += 1

        # Convert states to canonical
        canonical_states = self._obs_to_canonical_batch(
            dataset.states, mappings, unconstrained_dims,
        )  # (N, canonical_dim)
        canonical_next_states = self._obs_to_canonical_batch(
            dataset.next_states, mappings, unconstrained_dims,
        )  # (N, canonical_dim)

        # Build single-vector conversion functions (for runtime use)
        def obs_to_canonical_fn(obs: np.ndarray) -> np.ndarray:
            """Convert observation vector to canonical coordinates.

            :param obs: Shape ``(obs_dim,)``.
            :returns: Shape ``(canonical_dim,)``.
            """
            parts: list[np.ndarray] = []
            for m in mappings:
                obs_subset = obs[list(m.source_dims)]
                parts.append(m.obs_to_canonical(obs_subset))
            for d in unconstrained_dims:
                parts.append(np.array([obs[d]]))
            return np.concatenate(parts)

        def canonical_to_obs_fn(canonical: np.ndarray) -> np.ndarray:
            """Convert canonical coordinates back to observation space.

            :param canonical: Shape ``(canonical_dim,)``.
            :returns: Shape ``(obs_dim,)``.
            """
            obs = np.zeros(obs_dim)
            idx = 0
            for m in mappings:
                n_canonical = len(m.canonical_names)
                obs_subset = m.canonical_to_obs(canonical[idx:idx + n_canonical])
                for k, d in enumerate(m.source_dims):
                    obs[d] = obs_subset[k]
                idx += n_canonical
            for d in unconstrained_dims:
                obs[d] = canonical[idx]
                idx += 1
            return obs

        logger.info(
            "Built canonical representation: {} -> {} dims, "
            "names={}, angular_dims={}",
            obs_dim, len(canonical_names),
            canonical_names, angular_dim_indices,
        )

        return ObservationAnalysisResult(
            constraints=constraints,
            mappings=mappings,
            canonical_state_names=canonical_names,
            canonical_states=canonical_states,
            canonical_next_states=canonical_next_states,
            unconstrained_dims=unconstrained_dims,
            obs_to_canonical_fn=obs_to_canonical_fn,
            canonical_to_obs_fn=canonical_to_obs_fn,
            angular_dims=tuple(angular_dim_indices),
        )

    def _obs_to_canonical_batch(
        self,
        states: np.ndarray,
        mappings: list[CanonicalMapping],
        unconstrained_dims: tuple[int, ...],
    ) -> np.ndarray:
        """Convert batch of observations to canonical coordinates.

        :param states: Shape ``(N, obs_dim)``.
        :param mappings: Canonical mappings.
        :param unconstrained_dims: Dims to pass through.
        :returns: Shape ``(N, canonical_dim)``.
        """
        parts: list[np.ndarray] = []
        for m in mappings:
            obs_subset = states[:, list(m.source_dims)]  # (N, n_source)
            canonical = m.obs_to_canonical(obs_subset)  # (N, n_canonical)
            parts.append(canonical)

        if unconstrained_dims:
            parts.append(states[:, list(unconstrained_dims)])

        return np.column_stack(parts) if parts else np.empty(
            (states.shape[0], 0),
        )

    @staticmethod
    def _no_constraints_result(
        dataset: ExploratoryDataset,
        state_names: list[str],
    ) -> ObservationAnalysisResult:
        """Build a pass-through result when no constraints are found."""
        obs_dim = dataset.states.shape[1]
        return ObservationAnalysisResult(
            constraints=[],
            mappings=[],
            canonical_state_names=list(state_names),
            canonical_states=dataset.states,
            canonical_next_states=dataset.next_states,
            unconstrained_dims=tuple(range(obs_dim)),
            obs_to_canonical_fn=lambda obs: obs.copy(),
            canonical_to_obs_fn=lambda can: can.copy(),
            angular_dims=(),
        )


# ---------------------------------------------------------------------------
# Utility: angular delta wrapping
# ---------------------------------------------------------------------------


def wrap_angular_deltas(
    canonical_states: np.ndarray,
    canonical_next_states: np.ndarray,
    angular_dims: tuple[int, ...],
) -> np.ndarray:
    """Compute wrapped deltas for canonical coordinates.

    For angular dimensions, uses ``atan2(sin(d), cos(d))`` to handle
    discontinuities at +/-pi. For non-angular dimensions, computes
    simple subtraction.

    :param canonical_states: Shape ``(N, canonical_dim)``.
    :param canonical_next_states: Shape ``(N, canonical_dim)``.
    :param angular_dims: Indices of angular dimensions.
    :returns: Wrapped deltas, shape ``(N, canonical_dim)``.
    """
    deltas = canonical_next_states - canonical_states  # (N, canonical_dim)

    for d in angular_dims:
        raw_delta = deltas[:, d]  # (N,)
        deltas[:, d] = np.arctan2(np.sin(raw_delta), np.cos(raw_delta))

    return deltas
