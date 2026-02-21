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
    use_tiered: bool = True,
    tiered_r2_threshold: float = 0.95,
) -> list[SymbolicExpression]:
    """Top-level worker for parallel per-dimension SR.

    Must be top-level for pickling by ProcessPoolExecutor.
    Sets ``_CIRC_SR_SUBPROCESS=1`` to prevent nested seed parallelism.

    When ``use_tiered`` is True, runs a quick low-complexity pass first
    and only runs the full config if R2 < ``tiered_r2_threshold``.
    """
    import dataclasses as dc

    os.environ["_CIRC_SR_SUBPROCESS"] = "1"
    logger.info(
        "SR worker for {}: {} samples, {} features",
        target_var, x.shape[0], x.shape[1],
    )

    if use_tiered:
        # Quick pass
        quick_config = dc.replace(
            sr_config,
            max_complexity=min(15, sr_config.max_complexity),
            n_iterations=min(20, sr_config.n_iterations),
            populations=min(15, sr_config.populations),
            n_sr_runs=1,
            timeout_seconds=min(60, sr_config.timeout_seconds),
        )
        quick_regressor = SymbolicRegressor(quick_config)
        quick_exprs = quick_regressor.fit(x, y, var_names)

        if quick_exprs:
            best_r2 = SymbolicRegressor._best_r2(
                quick_exprs, x, y, var_names,
            )
            if best_r2 >= tiered_r2_threshold:
                logger.info(
                    "Tiered SR worker for {}: quick pass sufficient "
                    "(R2={:.6f}), skipping full",
                    target_var, best_r2,
                )
                return quick_exprs

        # Full pass
        full_regressor = SymbolicRegressor(sr_config)
        full_exprs = full_regressor.fit(x, y, var_names)

        # Merge and deduplicate
        all_exprs = quick_exprs + full_exprs
        seen: set[str] = set()
        unique: list[SymbolicExpression] = []
        for expr in sorted(all_exprs, key=lambda e: e.complexity):
            if expr.expression_str not in seen:
                seen.add(expr.expression_str)
                unique.append(expr)
        return unique

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

    Uses a tiered discovery strategy (fastest to slowest):

    1. **Physics templates** (~100ms): try known parametric forms.
    2. **Quick SR** (~30s): low-complexity, few iterations. If best
       R2 >= ``tiered_r2_threshold``, skip the full run.
    3. **Full SR** (minutes): full complexity and iteration budget.

    Templates above ``template_min_r2`` are registered as hypothesis
    candidates, but PySR is only skipped when the best template achieves
    R2 >= ``template_skip_pysr_r2``. This ensures that multiple candidate
    expressions (both template-based and SR-based) compete during
    falsification.

    See ``CIRC-RL_Framework.md`` Section 3.4.1.

    :param sr_config: Configuration for symbolic regression (used as
        the "full" tier).
    :param include_env_params: If True, include environment parameters as
        input features to the symbolic regression (enables discovery of
        parametric relationships like :math:`\beta_e \propto 1/(m l^2)`).
    :param use_templates: If True, try physics template matching before
        running PySR. Matching templates are registered as hypothesis
        candidates alongside PySR results.
    :param template_min_r2: Minimum R2 for a physics template to be
        registered as a hypothesis candidate. Default 0.90.
    :param template_skip_pysr_r2: If the best template achieves R2 >=
        this threshold, PySR is skipped for that dimension entirely.
        Default 0.999.
    :param use_tiered_sr: If True, run a quick low-complexity SR pass
        before the full config. When the quick pass finds R2 >=
        ``tiered_r2_threshold``, the full pass is skipped.
    :param tiered_r2_threshold: Minimum R2 from the quick SR pass to
        skip the full run. Default 0.95.
    """

    def __init__(
        self,
        sr_config: SymbolicRegressionConfig | None = None,
        include_env_params: bool = True,
        use_templates: bool = True,
        template_min_r2: float = 0.90,
        template_skip_pysr_r2: float = 0.999,
        use_tiered_sr: bool = True,
        tiered_r2_threshold: float = 0.95,
    ) -> None:
        self._sr_config = sr_config or SymbolicRegressionConfig()
        self._include_env_params = include_env_params
        self._use_templates = use_templates
        self._template_min_r2 = template_min_r2
        self._template_skip_pysr_r2 = template_skip_pysr_r2
        self._use_tiered_sr = use_tiered_sr
        self._tiered_r2_threshold = tiered_r2_threshold

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
    ) -> tuple[list[SymbolicExpression], bool]:
        """Try physics template matching before PySR.

        Returns ALL templates above ``template_min_r2`` as candidates,
        and signals whether to skip PySR (only when best R2 >=
        ``template_skip_pysr_r2``).

        :returns: Tuple of (template_expressions, skip_pysr). The list
            may be empty if no template matches above threshold.
        """
        if not self._use_templates:
            return [], False

        from circ_rl.hypothesis.physics_templates import TemplateBasedIdentifier

        identifier = TemplateBasedIdentifier(min_r2=self._template_min_r2)
        results = identifier.identify(x, y, var_names)

        if not results:
            logger.debug(
                "No template matched for {} (threshold R2={}), "
                "falling through to PySR",
                target_var, self._template_min_r2,
            )
            return [], False

        best_r2 = results[0][1]
        skip_pysr = best_r2 >= self._template_skip_pysr_r2

        if skip_pysr:
            logger.info(
                "Template matched for {}: {} results (best R2={:.6f} "
                ">= {:.4f}), skipping PySR",
                target_var, len(results), best_r2,
                self._template_skip_pysr_r2,
            )
        else:
            logger.info(
                "Template matched for {}: {} results (best R2={:.6f} "
                "< {:.4f}), will also run PySR",
                target_var, len(results), best_r2,
                self._template_skip_pysr_r2,
            )

        return [expr for expr, _r2 in results], skip_pysr

    def _make_quick_sr_config(self) -> SymbolicRegressionConfig:
        """Build a reduced-budget SR config for the quick tier.

        Uses half the complexity, quarter the iterations, and single
        seed -- typically runs in ~30s vs minutes for the full config.
        """
        import dataclasses as dc

        full = self._sr_config
        return dc.replace(
            full,
            max_complexity=min(15, full.max_complexity),
            n_iterations=min(20, full.n_iterations),
            populations=min(15, full.populations),
            n_sr_runs=1,
            timeout_seconds=min(60, full.timeout_seconds),
        )

    def _run_tiered_sr(
        self,
        target_var: str,
        x: np.ndarray,
        y: np.ndarray,
        var_names: list[str],
    ) -> list[SymbolicExpression]:
        """Run tiered SR: quick pass first, full pass only if needed.

        :returns: List of Pareto-front expressions (merged from both tiers
            if the full tier runs).
        """
        # Tier 1: quick low-complexity pass
        quick_config = self._make_quick_sr_config()
        quick_regressor = SymbolicRegressor(quick_config)

        logger.info(
            "Tiered SR for {} -- quick pass (complexity<={}, iters={})",
            target_var, quick_config.max_complexity, quick_config.n_iterations,
        )
        quick_exprs = quick_regressor.fit(x, y, var_names)

        # Check if quick pass is sufficient
        if quick_exprs:
            best_r2 = SymbolicRegressor._best_r2(quick_exprs, x, y, var_names)
            if best_r2 >= self._tiered_r2_threshold:
                logger.info(
                    "Tiered SR for {}: quick pass sufficient "
                    "(best R2={:.6f} >= {:.4f}), skipping full pass",
                    target_var, best_r2, self._tiered_r2_threshold,
                )
                return quick_exprs

            logger.info(
                "Tiered SR for {}: quick pass insufficient "
                "(best R2={:.6f} < {:.4f}), running full pass",
                target_var, best_r2, self._tiered_r2_threshold,
            )

        # Tier 2: full SR
        full_regressor = SymbolicRegressor(self._sr_config)
        full_exprs = full_regressor.fit(x, y, var_names)

        # Merge and deduplicate
        all_exprs = quick_exprs + full_exprs
        seen: set[str] = set()
        unique: list[SymbolicExpression] = []
        for expr in sorted(all_exprs, key=lambda e: e.complexity):
            if expr.expression_str not in seen:
                seen.add(expr.expression_str)
                unique.append(expr)
        return unique

    def _generate_sequential(
        self,
        dim_tasks: list[tuple[str, int, np.ndarray, np.ndarray, list[str]]],
        register: HypothesisRegister,
    ) -> list[str]:
        """Run SR for each dimension sequentially.

        Discovery strategy (fastest to slowest):

        1. Physics template matching (~100ms) -- always registered
        2. Quick SR pass (low complexity, ~30s) -- skipped if templates
           achieve R2 >= ``template_skip_pysr_r2``
        3. Full SR pass (only if quick pass R2 < threshold)
        """
        all_ids: list[str] = []

        for target_var, _dim_idx, x, y, var_names in dim_tasks:
            expressions: list[SymbolicExpression] = []

            # Try template matching first
            template_exprs, skip_pysr = self._try_templates(
                target_var, x, y, var_names,
            )
            expressions.extend(template_exprs)

            # Run PySR unless templates are near-perfect
            if not skip_pysr:
                if self._use_tiered_sr:
                    sr_exprs = self._run_tiered_sr(
                        target_var, x, y, var_names,
                    )
                else:
                    logger.info(
                        "Running SR for {}: {} samples, {} features",
                        target_var, x.shape[0], x.shape[1],
                    )
                    regressor = SymbolicRegressor(self._sr_config)
                    sr_exprs = regressor.fit(x, y, var_names)
                expressions.extend(sr_exprs)

            for i, expr in enumerate(expressions):
                entry = self._make_entry(
                    expr, target_var, x, y, var_names, idx=i,
                )
                register.register(entry)
                all_ids.append(entry.hypothesis_id)

            logger.info(
                "Registered {} hypotheses for {} ({} template, {} SR)",
                len(expressions), target_var,
                len(template_exprs),
                len(expressions) - len(template_exprs),
            )

        return all_ids

    def _generate_parallel(
        self,
        dim_tasks: list[tuple[str, int, np.ndarray, np.ndarray, list[str]]],
        register: HypothesisRegister,
    ) -> list[str]:
        """Run SR for each dimension in parallel processes.

        Tries physics template matching first for each dimension.
        Only dimensions where templates don't achieve R2 >=
        ``template_skip_pysr_r2`` are submitted to PySR in parallel.
        Template expressions are always registered as candidates.
        """
        all_ids: list[str] = []

        # Try templates first (fast, in main process)
        sr_tasks: list[tuple[str, int, np.ndarray, np.ndarray, list[str]]] = []
        template_count_per_dim: dict[str, int] = {}

        for target_var, dim_idx, x, y, var_names in dim_tasks:
            template_exprs, skip_pysr = self._try_templates(
                target_var, x, y, var_names,
            )

            # Register template expressions
            n_existing = len(all_ids)
            for i, expr in enumerate(template_exprs):
                entry = self._make_entry(
                    expr, target_var, x, y, var_names, idx=i,
                )
                register.register(entry)
                all_ids.append(entry.hypothesis_id)
            template_count_per_dim[target_var] = len(all_ids) - n_existing

            if template_count_per_dim[target_var] > 0:
                logger.info(
                    "Registered {} template hypotheses for {}",
                    template_count_per_dim[target_var], target_var,
                )

            if not skip_pysr:
                sr_tasks.append((target_var, dim_idx, x, y, var_names))

        if not sr_tasks:
            logger.info(
                "All dimensions matched templates (R2 >= {}), skipping PySR",
                self._template_skip_pysr_r2,
            )
            return all_ids

        logger.info(
            "Running SR for {} dimensions in parallel",
            len(sr_tasks),
        )

        with ProcessPoolExecutor(max_workers=len(sr_tasks)) as pool:
            futures = {
                pool.submit(
                    _dynamics_sr_worker,
                    self._sr_config,
                    target_var,
                    x,
                    y,
                    var_names,
                    self._use_tiered_sr,
                    self._tiered_r2_threshold,
                ): target_var
                for target_var, _dim_idx, x, y, var_names in sr_tasks
            }

            for future in as_completed(futures):
                target_var = futures[future]
                expressions = future.result()
                task = next(
                    t for t in sr_tasks if t[0] == target_var
                )
                _, _, x, y, var_names = task

                # Offset index by number of templates already registered
                idx_offset = template_count_per_dim.get(target_var, 0)
                for i, expr in enumerate(expressions):
                    entry = self._make_entry(
                        expr, target_var, x, y, var_names,
                        idx=idx_offset + i,
                    )
                    register.register(entry)
                    all_ids.append(entry.hypothesis_id)

                logger.info(
                    "Registered {} SR hypotheses for {} (parallel)",
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
