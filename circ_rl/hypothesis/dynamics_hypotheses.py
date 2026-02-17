"""Dynamics hypothesis generation via symbolic regression.

For each state dimension with variant dynamics, runs symbolic regression
to discover analytic functional forms: delta_s_i = h_i(s, a; theta_e).

See ``CIRC-RL_Framework.md`` Section 3.4.1 (Dynamics Hypotheses).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

from circ_rl.hypothesis.hypothesis_register import (
    HypothesisEntry,
    HypothesisRegister,
    HypothesisStatus,
)
from circ_rl.hypothesis.symbolic_regressor import (
    SymbolicRegressionConfig,
    SymbolicRegressor,
)

if TYPE_CHECKING:
    from circ_rl.environments.data_collector import ExploratoryDataset
    from circ_rl.feature_selection.transition_analyzer import TransitionAnalysisResult
    from circ_rl.hypothesis.expression import SymbolicExpression


class DynamicsHypothesisGenerator:
    r"""Generate dynamics hypotheses for variant state dimensions.

    For each variant state dimension :math:`s_i`, runs symbolic regression
    on the pooled transition data to discover candidate expressions:

    .. math::

        \Delta s_i = h_i(s, a; \theta_e)

    The generator produces a Pareto front of expressions for each dimension
    and registers them in the hypothesis register.

    See ``CIRC-RL_Framework.md`` Section 3.4.1.

    :param sr_config: Configuration for symbolic regression.
    :param include_env_params: If True, include environment parameters as
        input features to the symbolic regression (enables discovery of
        parametric relationships like :math:`\beta_e \propto 1/(m l^2)`).
    """

    def __init__(
        self,
        sr_config: SymbolicRegressionConfig | None = None,
        include_env_params: bool = True,
    ) -> None:
        self._sr_config = sr_config or SymbolicRegressionConfig()
        self._include_env_params = include_env_params

    def generate(
        self,
        dataset: ExploratoryDataset,
        transition_result: TransitionAnalysisResult,
        state_feature_names: list[str],
        register: HypothesisRegister,
        env_param_names: list[str] | None = None,
        angular_dims: tuple[int, ...] = (),
    ) -> list[str]:
        """Generate dynamics hypotheses for all state dimensions.

        Runs symbolic regression on every state dimension. For variant
        dimensions (identified by the transition analysis), environment
        parameters are included as input features so SR can discover
        parametric relationships. For invariant dimensions, only state
        and action features are used.

        :param dataset: Multi-environment exploratory data with next_states.
        :param transition_result: Result from TransitionAnalyzer identifying
            variant/invariant dimensions.
        :param state_feature_names: Names of state features.
        :param register: The hypothesis register to populate.
        :param env_param_names: Names of environment parameters (raw, not
            ``ep_``-prefixed). Required if include_env_params=True and
            dataset has env_params.
        :param angular_dims: Indices of canonical dimensions that represent
            angular coordinates. Deltas for these dimensions are wrapped
            via ``atan2(sin(d), cos(d))`` to avoid discontinuities.
        :returns: List of hypothesis_ids that were registered.
        """
        variant_dims = set(transition_result.variant_dims)
        all_dims = list(state_feature_names)

        logger.info(
            "Generating dynamics hypotheses for {} dimensions "
            "({} variant, {} invariant): {}",
            len(all_dims),
            len(variant_dims),
            len(all_dims) - len(variant_dims),
            all_dims,
        )

        regressor = SymbolicRegressor(self._sr_config)
        all_ids: list[str] = []

        for dim_name in all_dims:
            dim_idx = state_feature_names.index(dim_name)
            target_var = f"delta_{dim_name}"

            # Always include env params when available so SR can
            # discover parametric relationships. The LOEO invariance
            # test is too lenient (high R^2 even when coefficients
            # genuinely differ across envs); the falsification test
            # (structural consistency) is the proper filter.
            dim_env_params = env_param_names

            # Build input features and target
            is_angular = dim_idx in angular_dims
            x, y, var_names = self._build_regression_data(
                dataset, dim_idx, state_feature_names, dim_env_params,
                wrap_angular=is_angular,
            )

            logger.info(
                "Running SR for {}: {} samples, {} features",
                target_var, x.shape[0], x.shape[1],
            )

            expressions = regressor.fit(x, y, var_names)

            # Register each expression
            for i, expr in enumerate(expressions):
                entry = self._make_entry(
                    expr, target_var, x, y, var_names, idx=i,
                )
                register.register(entry)
                all_ids.append(entry.hypothesis_id)

            logger.info(
                "Registered {} hypotheses for {}",
                len(expressions), target_var,
            )

        return all_ids

    def _build_regression_data(
        self,
        dataset: ExploratoryDataset,
        dim_idx: int,
        state_feature_names: list[str],
        env_param_names: list[str] | None,
        wrap_angular: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Build input/target arrays for symbolic regression.

        :param wrap_angular: If True, wrap the delta via atan2(sin, cos)
            to handle angular discontinuities at +/-pi.
        :returns: Tuple of (x, y, variable_names).
        """
        states = dataset.states  # (N, state_dim)
        actions = dataset.actions  # (N,) or (N, action_dim)
        next_states = dataset.next_states  # (N, state_dim)

        # Target: delta_s_i = next_states[:, i] - states[:, i]
        y = next_states[:, dim_idx] - states[:, dim_idx]  # (N,)
        if wrap_angular:
            y = np.arctan2(np.sin(y), np.cos(y))  # (N,)

        # Input features: [states, actions, env_params (optional)]
        actions_2d = actions if actions.ndim == 2 else actions[:, np.newaxis]
        action_dim = actions_2d.shape[1]
        action_names = (
            ["action"] if action_dim == 1
            else [f"action_{i}" for i in range(action_dim)]
        )

        var_names = list(state_feature_names) + action_names
        feature_arrays = [states, actions_2d]

        if (
            self._include_env_params
            and env_param_names
            and dataset.env_params is not None
        ):
            feature_arrays.append(dataset.env_params)
            var_names.extend(env_param_names)

        x = np.column_stack(feature_arrays)  # (N, n_features)
        return x, y, var_names

    @staticmethod
    def _make_entry(
        expr: SymbolicExpression,
        target_var: str,
        x: np.ndarray,
        y: np.ndarray,
        var_names: list[str],
        idx: int,
    ) -> HypothesisEntry:
        """Create a HypothesisEntry from a symbolic expression.

        :returns: A new HypothesisEntry with training fit metrics computed.
        """
        hypothesis_id = f"dyn_{target_var}_{idx}"

        # Compute training fit
        try:
            func = expr.to_callable(var_names)
            y_pred = func(x)
            ss_res = float(np.sum((y - y_pred) ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            mse = float(np.mean((y - y_pred) ** 2))
        except Exception:
            logger.warning(
                "Failed to evaluate expression '{}' for training metrics",
                expr.expression_str,
            )
            r2 = 0.0
            mse = float("inf")

        return HypothesisEntry(
            hypothesis_id=hypothesis_id,
            target_variable=target_var,
            expression=expr,
            complexity=expr.complexity,
            training_r2=r2,
            training_mse=mse,
            status=HypothesisStatus.UNTESTED,
        )
