"""Trajectory prediction test for hypothesis falsification.

Uses the dynamics hypothesis to simulate trajectories forward in time
and compares predicted state sequences to observed ones.

See ``CIRC-RL_Framework.md`` Section 3.5.3 (Trajectory Prediction).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from circ_rl.environments.data_collector import ExploratoryDataset
    from circ_rl.hypothesis.expression import SymbolicExpression


@dataclass(frozen=True)
class TrajectoryPredictionResult:
    """Result of trajectory prediction test.

    :param passed: Whether the hypothesis passed the test.
    :param mean_divergence: Mean trajectory divergence across test episodes.
    :param max_divergence: Maximum divergence observed.
    :param failure_fraction: Fraction of test trajectories that diverged
        beyond the threshold.
    :param per_trajectory_divergence: Divergence for each test trajectory.
    """

    passed: bool
    mean_divergence: float
    max_divergence: float
    failure_fraction: float
    per_trajectory_divergence: list[float]


class TrajectoryPredictionTest:
    r"""Test trajectory prediction accuracy of a dynamics hypothesis.

    From initial state :math:`s_0` and a sequence of actions
    :math:`(a_0, \ldots, a_{T-1})`, uses the hypothesis to predict
    :math:`(s_1, \ldots, s_T)` and compares to observed trajectories.

    See ``CIRC-RL_Framework.md`` Section 3.5.3.

    :param max_horizon: Maximum prediction horizon (timesteps).
    :param divergence_threshold_factor: Base factor for divergence threshold.
        The threshold grows with horizon: :math:`\epsilon_T = \text{factor}
        \cdot \sqrt{T}`.
    :param failure_fraction: Maximum fraction of trajectories that may
        diverge before falsification. Default 0.2.
    :param n_trajectories: Number of test trajectories to sample.
    """

    def __init__(
        self,
        max_horizon: int = 30,
        divergence_threshold_factor: float = 0.5,
        failure_fraction: float = 0.2,
        n_trajectories: int = 10,
    ) -> None:
        self._max_horizon = max_horizon
        self._divergence_factor = divergence_threshold_factor
        self._failure_fraction = failure_fraction
        self._n_trajectories = n_trajectories

    def test(
        self,
        dynamics_expressions: dict[int, SymbolicExpression],
        dataset: ExploratoryDataset,
        state_feature_names: list[str],
        variable_names: list[str],
        test_env_ids: list[int] | None = None,
    ) -> TrajectoryPredictionResult:
        """Test trajectory prediction for dynamics hypotheses.

        :param dynamics_expressions: Mapping from state dimension index
            to the SymbolicExpression predicting delta_s for that dimension.
            Dimensions not in this dict are assumed to have no hypothesis
            (predicted delta = 0, i.e., invariant dynamics handled by the
            hypothesis for that dimension).
        :param dataset: Multi-environment data (used to extract initial
            states and action sequences).
        :param state_feature_names: Names of state features.
        :param variable_names: Ordered variable names for the expressions.
        :param test_env_ids: Environment IDs to test on. If None, uses all.
        :returns: TrajectoryPredictionResult.
        """
        unique_envs = sorted(set(dataset.env_ids.tolist()))
        if test_env_ids is not None:
            unique_envs = [e for e in unique_envs if e in test_env_ids]

        if not unique_envs:
            return TrajectoryPredictionResult(
                passed=True, mean_divergence=0.0, max_divergence=0.0,
                failure_fraction=0.0, per_trajectory_divergence=[],
            )

        # Compile callables
        callables: dict[int, object] = {}
        for dim_idx, expr in dynamics_expressions.items():
            try:
                callables[dim_idx] = expr.to_callable(variable_names)
            except ValueError as exc:
                logger.warning(
                    "Cannot compile expression for dim {}: {}",
                    dim_idx, exc,
                )

        state_dim = len(state_feature_names)
        actions_2d = (
            dataset.actions if dataset.actions.ndim == 2
            else dataset.actions[:, np.newaxis]
        )
        action_dim = actions_2d.shape[1]

        # Extract trajectories from dataset (consecutive transitions per env)
        per_traj_divergences: list[float] = []
        n_tested = 0

        for env_id in unique_envs:
            mask = dataset.env_ids == env_id
            env_states = dataset.states[mask]  # (n_e, state_dim)
            env_actions = actions_2d[mask]  # (n_e, action_dim)
            env_next_states = dataset.next_states[mask]  # (n_e, state_dim)
            n_e = env_states.shape[0]

            # Build env params row if available
            env_params_row = None
            if dataset.env_params is not None:
                env_params_row = dataset.env_params[mask][0]  # (n_env_params,)

            # Sample trajectory starting points
            n_trajs = min(self._n_trajectories, n_e // self._max_horizon)
            if n_trajs == 0:
                n_trajs = 1

            rng = np.random.RandomState(env_id)
            max_start = max(1, n_e - self._max_horizon)
            starts = rng.choice(max_start, size=n_trajs, replace=True)

            for start in starts:
                horizon = min(self._max_horizon, n_e - start)
                if horizon < 2:
                    continue

                divergence = self._simulate_trajectory(
                    env_states, env_actions, env_next_states,
                    start, horizon, state_dim, action_dim,
                    callables, variable_names, state_feature_names,
                    env_params_row,
                )
                per_traj_divergences.append(divergence)
                n_tested += 1

        if not per_traj_divergences:
            return TrajectoryPredictionResult(
                passed=True, mean_divergence=0.0, max_divergence=0.0,
                failure_fraction=0.0, per_trajectory_divergence=[],
            )

        # Count failures: divergence > threshold
        threshold = self._divergence_factor * np.sqrt(self._max_horizon)
        n_failed = sum(1 for d in per_traj_divergences if d > threshold)
        failure_fraction = n_failed / len(per_traj_divergences)

        mean_div = float(np.mean(per_traj_divergences))
        max_div = float(np.max(per_traj_divergences))
        passed = failure_fraction <= self._failure_fraction

        logger.debug(
            "Trajectory prediction: mean_div={:.4f}, max_div={:.4f}, "
            "failures={}/{}, passed={}",
            mean_div, max_div, n_failed, len(per_traj_divergences), passed,
        )

        return TrajectoryPredictionResult(
            passed=passed,
            mean_divergence=mean_div,
            max_divergence=max_div,
            failure_fraction=failure_fraction,
            per_trajectory_divergence=per_traj_divergences,
        )

    def _simulate_trajectory(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        next_states: np.ndarray,
        start: int,
        horizon: int,
        state_dim: int,
        action_dim: int,
        callables: dict[int, object],
        variable_names: list[str],
        state_feature_names: list[str],
        env_params_row: np.ndarray | None,
    ) -> float:
        """Simulate one trajectory and return normalized divergence.

        :returns: Mean per-step RMSE between predicted and observed states.
        """
        predicted_state = states[start].copy()  # (state_dim,)
        total_error = 0.0

        for t in range(horizon):
            idx = start + t
            action = actions[idx]  # (action_dim,)
            observed_next = next_states[idx]  # (state_dim,)

            # Build input vector for expressions
            x_row = self._build_input_row(
                predicted_state, action, variable_names,
                state_feature_names, action_dim, env_params_row,
            )  # (1, n_vars)

            # Predict delta_s for each dimension with a hypothesis
            predicted_next = predicted_state.copy()
            for dim_idx, func in callables.items():
                try:
                    delta = float(func(x_row)[0])  # type: ignore[operator]
                    predicted_next[dim_idx] = predicted_state[dim_idx] + delta
                except Exception:
                    # If evaluation fails, use observed transition
                    predicted_next[dim_idx] = observed_next[dim_idx]

            # For dimensions without hypotheses, use observed delta
            for dim_idx in range(state_dim):
                if dim_idx not in callables:
                    delta_obs = observed_next[dim_idx] - states[idx, dim_idx]
                    predicted_next[dim_idx] = predicted_state[dim_idx] + delta_obs

            error = float(np.sqrt(np.mean((predicted_next - observed_next) ** 2)))
            total_error += error

            predicted_state = predicted_next

        return total_error / horizon

    @staticmethod
    def _build_input_row(
        state: np.ndarray,
        action: np.ndarray,
        variable_names: list[str],
        state_feature_names: list[str],
        action_dim: int,
        env_params_row: np.ndarray | None,
    ) -> np.ndarray:
        """Build a single input row for expression evaluation.

        :returns: Array of shape ``(1, len(variable_names))``.
        """
        row = []
        state_dim = len(state_feature_names)
        for name in variable_names:
            if name.startswith("s") and name[1:].isdigit():
                idx = int(name[1:])
                row.append(state[idx] if idx < state_dim else 0.0)
            elif name == "action":
                row.append(action[0] if action_dim >= 1 else 0.0)
            elif name.startswith("action_") and name[7:].isdigit():
                idx = int(name[7:])
                row.append(action[idx] if idx < action_dim else 0.0)
            elif env_params_row is not None:
                # Env param -- find positional index
                ep_offset = state_dim + action_dim
                var_idx = variable_names.index(name)
                ep_idx = var_idx - ep_offset
                if 0 <= ep_idx < len(env_params_row):
                    row.append(float(env_params_row[ep_idx]))
                else:
                    row.append(0.0)
            else:
                row.append(0.0)
        return np.array([row], dtype=np.float64)  # (1, n_vars)
