"""Transition dynamics analysis for dynamics-normalized policy optimization.

Implements Phase 2.5 of the CIRC-RL pipeline: analyze transition dynamics
across environments to estimate per-environment dynamics scales and identify
state dimensions with variant/invariant transition mechanisms.

See ``CIRC-RL_Framework.md`` Section 3.7 (Transition Dynamics Normalization).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from circ_rl.environments.data_collector import ExploratoryDataset


@dataclass(frozen=True)
class TransitionAnalysisResult:
    """Result of transition dynamics analysis.

    :param dynamics_scales: Per-environment action effectiveness,
        shape ``(n_envs,)``. Higher values mean actions have a larger
        effect on state transitions.
    :param reference_scale: Mean dynamics scale across training environments.
        Used as the reference for normalization.
    :param transition_loeo_r2: Per-environment minimum LOEO R^2 across
        state dimensions. Maps ``env_id -> min_r2``.
    :param per_dim_loeo_r2: Per-state-dimension minimum LOEO R^2.
        Maps ``state_dim_name -> min_r2``.
    :param variant_dims: State dimensions with variant transitions
        (LOEO R^2 below threshold).
    :param invariant_dims: State dimensions with invariant transitions
        (LOEO R^2 above threshold).
    :param action_coefficients: Per-environment action effect matrices,
        shape ``(n_envs, state_dim, action_dim)``. ``B_e[i, j]`` is the
        partial derivative of ``delta_s_i`` w.r.t. ``action_j`` in env ``e``.
    """

    dynamics_scales: np.ndarray
    reference_scale: float
    transition_loeo_r2: dict[int, float]
    per_dim_loeo_r2: dict[str, float]
    variant_dims: list[str]
    invariant_dims: list[str]
    action_coefficients: np.ndarray


class TransitionAnalyzer:
    r"""Analyze transition dynamics across environments.

    Estimates per-environment dynamics scales and tests whether transition
    mechanisms are invariant using Leave-One-Environment-Out (LOEO) R^2.

    The **dynamics scale** of environment :math:`e` is:

    .. math::

        D_e = \|B_e\|_F

    where :math:`B_e` is the matrix of action coefficients in the linear
    approximation :math:`\Delta s_i \approx B_e[i, :] \cdot a + \text{state terms}`.

    :param loeo_r2_threshold: Minimum LOEO R^2 for a state dimension to be
        considered invariant. Default 0.9.
    :param max_loeo_samples: Maximum number of training samples per LOEO
        fold. When the training partition exceeds this, a random subset
        is drawn (seeded for reproducibility). Default 10000. Set to 0
        to disable subsampling.
    """

    def __init__(
        self,
        loeo_r2_threshold: float = 0.9,
        max_loeo_samples: int = 10000,
    ) -> None:
        if loeo_r2_threshold <= 0 or loeo_r2_threshold >= 1:
            raise ValueError(
                f"loeo_r2_threshold must be in (0, 1), got {loeo_r2_threshold}"
            )
        self._loeo_r2_threshold = loeo_r2_threshold
        self._max_loeo_samples = max_loeo_samples

    def analyze(
        self,
        dataset: ExploratoryDataset,
        state_feature_names: list[str],
        action_dim: int,
    ) -> TransitionAnalysisResult:
        """Run transition dynamics analysis.

        :param dataset: Multi-environment exploratory data with next_states.
        :param state_feature_names: Names of state features.
        :param action_dim: Number of action dimensions.
        :returns: TransitionAnalysisResult with dynamics scales and LOEO results.
        """
        unique_envs = sorted(set(dataset.env_ids.tolist()))
        state_dim = len(state_feature_names)

        logger.info(
            "Transition analysis: {} state dims, {} action dims, {} envs",
            state_dim, action_dim, len(unique_envs),
        )

        # Step 1: LOEO transition test
        per_dim_loeo_r2, transition_loeo_r2 = self._loeo_transition_test(
            dataset, unique_envs, state_feature_names,
            self._max_loeo_samples,
        )

        variant_dims = [
            name for name, r2 in per_dim_loeo_r2.items()
            if r2 < self._loeo_r2_threshold
        ]
        invariant_dims = [
            name for name, r2 in per_dim_loeo_r2.items()
            if r2 >= self._loeo_r2_threshold
        ]

        # Step 2: Estimate dynamics scales
        action_coefficients, dynamics_scales = self._estimate_dynamics_scales(
            dataset, unique_envs, state_dim, action_dim,
        )
        reference_scale = float(np.mean(dynamics_scales))

        logger.info(
            "Dynamics scales: {} (ref={:.4f}), variant_dims={}, invariant_dims={}",
            [f"{s:.4f}" for s in dynamics_scales],
            reference_scale,
            variant_dims,
            invariant_dims,
        )

        return TransitionAnalysisResult(
            dynamics_scales=dynamics_scales,
            reference_scale=reference_scale,
            transition_loeo_r2=transition_loeo_r2,
            per_dim_loeo_r2=per_dim_loeo_r2,
            variant_dims=variant_dims,
            invariant_dims=invariant_dims,
            action_coefficients=action_coefficients,
        )

    def _loeo_transition_test(
        self,
        dataset: ExploratoryDataset,
        unique_envs: list[int],
        state_feature_names: list[str],
        max_loeo_samples: int = 10000,
    ) -> tuple[dict[str, float], dict[int, float]]:
        r"""Leave-One-Environment-Out R^2 test for transition invariance.

        For each state dimension :math:`s_i`, trains a
        ``HistGradientBoostingRegressor`` on all environments except one,
        then evaluates R^2 on the held-out environment.

        :param dataset: Multi-environment data.
        :param unique_envs: Sorted list of environment IDs.
        :param state_feature_names: Names of state features.
        :param max_loeo_samples: Maximum training samples per fold.
            When the training partition exceeds this, a random subset
            is drawn. Set to 0 to disable subsampling.
        :returns: Tuple of (per_dim_min_r2, per_env_min_r2).
        """
        from sklearn.ensemble import HistGradientBoostingRegressor

        states = dataset.states  # (N, state_dim)
        actions = dataset.actions  # (N,) or (N, action_dim)
        next_states = dataset.next_states  # (N, state_dim)
        env_ids = dataset.env_ids  # (N,)

        # Build input features: [states, actions]
        if actions.ndim == 1:
            features = np.column_stack([states, actions])  # (N, state_dim + 1)
        else:
            features = np.column_stack([states, actions])  # (N, state_dim + action_dim)

        state_dim = states.shape[1]
        per_dim_min_r2: dict[str, float] = {}
        per_env_r2s: dict[int, list[float]] = {e: [] for e in unique_envs}

        for dim_idx in range(state_dim):
            dim_name = state_feature_names[dim_idx]
            targets = next_states[:, dim_idx]  # (N,)
            dim_min_r2 = 1.0

            for held_out in unique_envs:
                train_mask = env_ids != held_out
                test_mask = env_ids == held_out

                x_train = features[train_mask]
                y_train = targets[train_mask]

                # Subsample training data if too large
                if max_loeo_samples > 0 and x_train.shape[0] > max_loeo_samples:
                    rng = np.random.default_rng(42 + held_out + dim_idx * 1000)
                    idx = rng.choice(
                        x_train.shape[0], max_loeo_samples, replace=False,
                    )
                    x_train = x_train[idx]
                    y_train = y_train[idx]

                model = HistGradientBoostingRegressor(
                    max_iter=100, max_depth=5, random_state=42,
                )
                model.fit(x_train, y_train)

                y_pred = model.predict(features[test_mask])
                y_true = targets[test_mask]
                ss_res = float(np.sum((y_true - y_pred) ** 2))
                ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))

                r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
                dim_min_r2 = min(dim_min_r2, r2)
                per_env_r2s[held_out].append(r2)

                logger.debug(
                    "LOEO transition: dim={}, held_out={}, R^2={:.4f}",
                    dim_name, held_out, r2,
                )

            per_dim_min_r2[dim_name] = dim_min_r2

        # Per-env minimum R^2 across all state dimensions
        per_env_min_r2 = {
            env_id: min(r2s) if r2s else 0.0
            for env_id, r2s in per_env_r2s.items()
        }

        return per_dim_min_r2, per_env_min_r2

    @staticmethod
    def _estimate_dynamics_scales(
        dataset: ExploratoryDataset,
        unique_envs: list[int],
        state_dim: int,
        action_dim: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""Estimate per-environment dynamics scales from transition data.

        For each environment, fits a linear regression:

        .. math::

            \Delta s_i = s'_i - s_i \approx b_0 + W_s \cdot s + B_e[i,:] \cdot a

        and extracts the action coefficients :math:`B_e[i, j]`.

        The dynamics scale is the Frobenius norm of :math:`B_e`:

        .. math::

            D_e = \|B_e\|_F

        :param dataset: Multi-environment data.
        :param unique_envs: Sorted list of environment IDs.
        :param state_dim: Number of state dimensions.
        :param action_dim: Number of action dimensions.
        :returns: Tuple of (action_coefficients, dynamics_scales).
            action_coefficients shape: ``(n_envs, state_dim, action_dim)``.
            dynamics_scales shape: ``(n_envs,)``.
        """
        n_envs = len(unique_envs)
        action_coefficients = np.zeros(
            (n_envs, state_dim, action_dim), dtype=np.float64,
        )  # (n_envs, state_dim, action_dim)
        dynamics_scales = np.zeros(n_envs, dtype=np.float64)  # (n_envs,)

        for env_i, env_id in enumerate(unique_envs):
            mask = dataset.env_ids == env_id
            states_e = dataset.states[mask]  # (n_e, state_dim)
            actions_e = dataset.actions[mask]  # (n_e,) or (n_e, action_dim)
            next_states_e = dataset.next_states[mask]  # (n_e, state_dim)
            n_e = states_e.shape[0]

            if actions_e.ndim == 1:
                actions_2d = actions_e.reshape(-1, 1)  # (n_e, 1)
            else:
                actions_2d = actions_e  # (n_e, action_dim)

            # Design matrix: [1, states, actions]
            design = np.column_stack([
                np.ones(n_e),
                states_e,
                actions_2d,
            ])  # (n_e, 1 + state_dim + action_dim)

            for dim_idx in range(state_dim):
                delta_s = next_states_e[:, dim_idx] - states_e[:, dim_idx]  # (n_e,)
                beta = np.linalg.lstsq(design, delta_s, rcond=None)[0]
                # Action coefficients are the last action_dim entries
                action_coefficients[env_i, dim_idx, :] = beta[
                    1 + state_dim : 1 + state_dim + action_dim
                ]

            # Dynamics scale = Frobenius norm of B_e
            dynamics_scales[env_i] = float(
                np.linalg.norm(action_coefficients[env_i], ord="fro")
            )

        return action_coefficients, dynamics_scales
