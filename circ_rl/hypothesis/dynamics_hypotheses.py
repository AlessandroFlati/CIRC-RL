"""Dynamics hypothesis generation via symbolic regression.

For each state dimension with variant dynamics, runs symbolic regression
to discover analytic functional forms: delta_s_i = h_i(s, a; theta_e).

See ``CIRC-RL_Framework.md`` Section 3.4.1 (Dynamics Hypotheses).
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
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


def _dynamics_sr_worker(
    sr_config: SymbolicRegressionConfig,
    target_var: str,
    x: np.ndarray,
    y: np.ndarray,
    var_names: list[str],
) -> list[SymbolicExpression]:
    """Top-level worker for parallel per-dimension SR.

    Must be top-level for pickling by ProcessPoolExecutor.
    Sets ``_CIRC_SR_SUBPROCESS=1`` to prevent nested seed parallelism.
    """
    os.environ["_CIRC_SR_SUBPROCESS"] = "1"
    logger.info(
        "SR worker for {}: {} samples, {} features",
        target_var, x.shape[0], x.shape[1],
    )
    regressor = SymbolicRegressor(sr_config)
    return regressor.fit(x, y, var_names)


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
    :param use_templates: If True, try physics template matching before
        running PySR. When a template achieves R2 >= ``template_min_r2``,
        PySR is skipped for that dimension. This is orders of magnitude
        faster than symbolic regression.
    :param template_min_r2: Minimum R2 for a physics template to be
        accepted (skipping PySR for that dimension).
    """

    def __init__(
        self,
        sr_config: SymbolicRegressionConfig | None = None,
        include_env_params: bool = True,
        use_templates: bool = True,
        template_min_r2: float = 0.99,
    ) -> None:
        self._sr_config = sr_config or SymbolicRegressionConfig()
        self._include_env_params = include_env_params
        self._use_templates = use_templates
        self._template_min_r2 = template_min_r2

    def generate(
        self,
        dataset: ExploratoryDataset,
        transition_result: TransitionAnalysisResult,
        state_feature_names: list[str],
        register: HypothesisRegister,
        env_param_names: list[str] | None = None,
        angular_dims: tuple[int, ...] = (),
        parallel: bool = False,
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
        :param parallel: If True, run SR for each dimension in parallel
            using ``ProcessPoolExecutor``. Each dimension gets its own
            process with a separate Julia runtime.
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

        # Prepare per-dimension work items
        dim_tasks: list[tuple[str, int, np.ndarray, np.ndarray, list[str]]] = []
        for dim_name in all_dims:
            dim_idx = state_feature_names.index(dim_name)
            target_var = f"delta_{dim_name}"
            dim_env_params = env_param_names
            is_angular = dim_idx in angular_dims
            x, y, var_names = self._build_regression_data(
                dataset, dim_idx, state_feature_names, dim_env_params,
                wrap_angular=is_angular,
            )
            dim_tasks.append((target_var, dim_idx, x, y, var_names))

        if parallel and len(dim_tasks) > 1:
            return self._generate_parallel(
                dim_tasks, register,
            )
        return self._generate_sequential(
            dim_tasks, register,
        )

    def _try_templates(
        self,
        target_var: str,
        x: np.ndarray,
        y: np.ndarray,
        var_names: list[str],
    ) -> list[SymbolicExpression] | None:
        """Try physics template matching before PySR.

        :returns: List of expressions if a template matched above
            threshold, or None to fall through to PySR.
        """
        if not self._use_templates:
            return None

        from circ_rl.hypothesis.physics_templates import TemplateBasedIdentifier

        identifier = TemplateBasedIdentifier(min_r2=self._template_min_r2)
        results = identifier.identify(x, y, var_names)

        if results:
            logger.info(
                "Template matched for {}: {} results (best R2={:.6f}), "
                "skipping PySR",
                target_var, len(results), results[0][1],
            )
            return [expr for expr, _r2 in results]

        logger.debug(
            "No template matched for {} (threshold R2={}), "
            "falling through to PySR",
            target_var, self._template_min_r2,
        )
        return None

    def _generate_sequential(
        self,
        dim_tasks: list[tuple[str, int, np.ndarray, np.ndarray, list[str]]],
        register: HypothesisRegister,
    ) -> list[str]:
        """Run SR for each dimension sequentially.

        Tries physics template matching first; falls through to PySR
        if no template matches above the R2 threshold.
        """
        regressor = SymbolicRegressor(self._sr_config)
        all_ids: list[str] = []

        for target_var, _dim_idx, x, y, var_names in dim_tasks:
            # Try template matching first
            template_exprs = self._try_templates(target_var, x, y, var_names)

            if template_exprs is not None:
                expressions = template_exprs
            else:
                logger.info(
                    "Running SR for {}: {} samples, {} features",
                    target_var, x.shape[0], x.shape[1],
                )
                expressions = regressor.fit(x, y, var_names)

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

    def _generate_parallel(
        self,
        dim_tasks: list[tuple[str, int, np.ndarray, np.ndarray, list[str]]],
        register: HypothesisRegister,
    ) -> list[str]:
        """Run SR for each dimension in parallel processes."""
        logger.info(
            "Running SR for {} dimensions in parallel",
            len(dim_tasks),
        )

        all_ids: list[str] = []

        with ProcessPoolExecutor(max_workers=len(dim_tasks)) as pool:
            futures = {
                pool.submit(
                    _dynamics_sr_worker,
                    self._sr_config,
                    target_var,
                    x,
                    y,
                    var_names,
                ): target_var
                for target_var, _dim_idx, x, y, var_names in dim_tasks
            }

            for future in as_completed(futures):
                target_var = futures[future]
                expressions = future.result()
                # Find the matching task to get x, y, var_names for _make_entry
                task = next(
                    t for t in dim_tasks if t[0] == target_var
                )
                _, _, x, y, var_names = task

                for i, expr in enumerate(expressions):
                    entry = self._make_entry(
                        expr, target_var, x, y, var_names, idx=i,
                    )
                    register.register(entry)
                    all_ids.append(entry.hypothesis_id)

                logger.info(
                    "Registered {} hypotheses for {} (parallel)",
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
