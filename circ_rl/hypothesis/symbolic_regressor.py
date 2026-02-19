"""PySR-based symbolic regression wrapper for hypothesis generation.

Wraps PySR to discover analytic expressions from numerical data,
producing Pareto fronts of candidate hypotheses ordered by
complexity and accuracy.

See ``CIRC-RL_Framework.md`` Section 3.4 (Phase 3: Structural
Hypothesis Generation).
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from loguru import logger

from circ_rl.hypothesis.expression import SymbolicExpression

if TYPE_CHECKING:
    import numpy as np


def _sr_worker(
    config: SymbolicRegressionConfig,
    x: np.ndarray,
    y: np.ndarray,
    variable_names: list[str],
    seed: int,
) -> list[SymbolicExpression]:
    """Top-level worker function for parallel SR seeds.

    Must be top-level (not a method) for pickling by ProcessPoolExecutor.
    Sets ``_CIRC_SR_SUBPROCESS=1`` to prevent nested parallelism.
    """
    os.environ["_CIRC_SR_SUBPROCESS"] = "1"
    regressor = SymbolicRegressor(config)
    return regressor._single_run(x, y, variable_names, seed)


@dataclass(frozen=True)
class SymbolicRegressionConfig:
    """Configuration for symbolic regression.

    :param max_complexity: Maximum expression tree complexity allowed.
    :param n_iterations: Number of PySR evolutionary iterations.
    :param populations: Number of evolutionary populations.
    :param binary_operators: Allowed binary operators.
    :param unary_operators: Allowed unary operators.
    :param parsimony: Parsimony coefficient (penalty per complexity unit).
    :param timeout_seconds: Maximum wall-clock time for the search.
    :param deterministic: If True, use fixed random state for reproducibility.
    :param random_state: Random seed (used only when deterministic=True).
    :param nested_constraints: Controls operator nesting depth.
        E.g., ``{"/": {"+": 0, "-": 0}}`` prevents addition/subtraction
        inside division, forcing separate additive terms instead of
        factored forms.
    :param complexity_of_operators: Custom complexity per operator.
        E.g., ``{"square": 1, "sin": 2}`` makes ``square`` cheap.
    :param constraints: Max subtree size per operator argument.
        E.g., ``{"sin": 10}`` limits what appears inside ``sin``.
    :param max_samples: Maximum number of samples to use for regression.
        If the input data has more rows than this, a random subset is
        selected (seeded by ``random_state``). None means no subsampling.
    :param n_sr_runs: Number of SR runs with different seeds.
        When > 1, runs PySR multiple times with seeds
        ``random_state, random_state + 1, ...`` and merges the
        Pareto fronts by deduplicating expression strings. This
        combats PySR's stochasticity at the cost of wall-clock time.
        Default 1 (single run).
    :param parallel_seeds: If True and ``n_sr_runs > 1``, run seeds
        in parallel using ``ProcessPoolExecutor``. Each PySR process
        gets its own Julia runtime. Disabled automatically when
        running inside a subprocess (to avoid nested parallelism).
    :param early_stop_r2: If set, stop multi-seed runs early when
        any expression achieves training R2 >= this threshold.
        Only used when ``n_sr_runs > 1``. None means no early
        stopping.
    """

    max_complexity: int = 30
    n_iterations: int = 40
    populations: int = 15
    binary_operators: tuple[str, ...] = ("+", "-", "*", "/")
    unary_operators: tuple[str, ...] = ("sin", "cos", "sqrt", "abs")
    parsimony: float = 0.0032
    timeout_seconds: int = 300
    deterministic: bool = True
    random_state: int = 42
    nested_constraints: dict[str, dict[str, int]] | None = None
    complexity_of_operators: dict[str, int | float] | None = None
    constraints: dict[str, int | tuple[int, int]] | None = None
    max_samples: int | None = None
    n_sr_runs: int = 1
    parallel_seeds: bool = True
    early_stop_r2: float | None = None


class SymbolicRegressor:
    """PySR wrapper for discovering analytic expressions.

    Fits symbolic regression models using PySR and returns a list
    of candidate expressions forming a Pareto front of accuracy
    vs. complexity.

    See ``CIRC-RL_Framework.md`` Section 3.4.

    :param config: Symbolic regression configuration.
    """

    def __init__(self, config: SymbolicRegressionConfig | None = None) -> None:
        self._config = config or SymbolicRegressionConfig()

    @property
    def config(self) -> SymbolicRegressionConfig:
        """The regression configuration."""
        return self._config

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        variable_names: list[str],
    ) -> list[SymbolicExpression]:
        """Run symbolic regression to discover analytic expressions.

        When ``n_sr_runs > 1``, runs PySR multiple times with different
        seeds and merges the Pareto fronts, deduplicating by expression
        string. This combats PySR's stochasticity.

        :param x: Input data of shape ``(n_samples, n_features)``.
        :param y: Target values of shape ``(n_samples,)``.
        :param variable_names: Names of input features (columns of x).
        :returns: List of SymbolicExpressions forming the Pareto front,
            sorted by complexity (ascending).
        :raises ImportError: If PySR is not installed.
        :raises ValueError: If input dimensions are inconsistent.
        """
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"x has {x.shape[0]} samples but y has {y.shape[0]}")
        if x.shape[1] != len(variable_names):
            raise ValueError(
                f"x has {x.shape[1]} features but {len(variable_names)} "
                f"variable names provided"
            )

        import numpy as np_local

        cfg = self._config

        # Subsample if requested and data is large
        if cfg.max_samples is not None and x.shape[0] > cfg.max_samples:
            n_original = x.shape[0]
            rng = np_local.random.default_rng(cfg.random_state)
            idx = rng.choice(n_original, cfg.max_samples, replace=False)
            x = x[idx]
            y = y[idx]
            logger.info(
                "Subsampled from {} to {} samples",
                n_original,
                cfg.max_samples,
            )

        n_runs = max(1, cfg.n_sr_runs)

        if n_runs == 1:
            return self._single_run(x, y, variable_names, cfg.random_state)

        seeds = [cfg.random_state + i for i in range(n_runs)]

        # Decide whether to parallelize: only if requested AND not
        # already inside a subprocess (prevents nested parallelism)
        use_parallel = (
            cfg.parallel_seeds
            and n_runs > 1
            and os.environ.get("_CIRC_SR_SUBPROCESS") != "1"
        )

        all_expressions: list[SymbolicExpression] = []

        if use_parallel:
            all_expressions = self._run_seeds_parallel(
                x, y, variable_names, seeds, cfg.early_stop_r2,
            )
        else:
            all_expressions = self._run_seeds_sequential(
                x, y, variable_names, seeds, cfg.early_stop_r2,
            )

        # Sort by complexity and deduplicate
        all_expressions.sort(key=lambda e: e.complexity)
        seen: set[str] = set()
        unique: list[SymbolicExpression] = []
        for expr in all_expressions:
            if expr.expression_str not in seen:
                seen.add(expr.expression_str)
                unique.append(expr)

        logger.info(
            "Multi-seed SR complete: {} unique expressions from {} runs",
            len(unique),
            n_runs,
        )

        return unique

    def _run_seeds_sequential(
        self,
        x: np.ndarray,
        y: np.ndarray,
        variable_names: list[str],
        seeds: list[int],
        early_stop_r2: float | None,
    ) -> list[SymbolicExpression]:
        """Run multi-seed SR sequentially with optional early stopping."""
        import numpy as np_local

        all_expressions: list[SymbolicExpression] = []
        for run_idx, seed in enumerate(seeds):
            logger.info(
                "Multi-seed SR run {}/{} (seed={})",
                run_idx + 1,
                len(seeds),
                seed,
            )
            run_expressions = self._single_run(x, y, variable_names, seed)
            all_expressions.extend(run_expressions)

            if early_stop_r2 is not None and run_expressions:
                best_r2 = self._best_r2(run_expressions, x, y, variable_names)
                if best_r2 >= early_stop_r2:
                    logger.info(
                        "Early stop: R2={:.6f} >= {:.6f} after seed {}",
                        best_r2,
                        early_stop_r2,
                        seed,
                    )
                    break

        return all_expressions

    def _run_seeds_parallel(
        self,
        x: np.ndarray,
        y: np.ndarray,
        variable_names: list[str],
        seeds: list[int],
        early_stop_r2: float | None,
    ) -> list[SymbolicExpression]:
        """Run multi-seed SR in parallel using ProcessPoolExecutor."""
        import numpy as np_local

        all_expressions: list[SymbolicExpression] = []

        logger.info(
            "Running {} SR seeds in parallel",
            len(seeds),
        )

        with ProcessPoolExecutor(max_workers=len(seeds)) as pool:
            futures = {
                pool.submit(
                    _sr_worker,
                    self._config,
                    x,
                    y,
                    variable_names,
                    seed,
                ): seed
                for seed in seeds
            }

            for future in as_completed(futures):
                seed = futures[future]
                run_expressions = future.result()
                all_expressions.extend(run_expressions)
                logger.info(
                    "Parallel SR seed {} complete: {} expressions",
                    seed,
                    len(run_expressions),
                )

                if early_stop_r2 is not None and run_expressions:
                    best_r2 = self._best_r2(
                        run_expressions, x, y, variable_names,
                    )
                    if best_r2 >= early_stop_r2:
                        logger.info(
                            "Early stop: R2={:.6f} >= {:.6f}, "
                            "cancelling remaining seeds",
                            best_r2,
                            early_stop_r2,
                        )
                        for f in futures:
                            f.cancel()
                        break

        return all_expressions

    @staticmethod
    def _best_r2(
        expressions: list[SymbolicExpression],
        x: np.ndarray,
        y: np.ndarray,
        variable_names: list[str],
    ) -> float:
        """Compute the best R2 among a list of expressions."""
        import numpy as np_local

        best = -float("inf")
        for expr in expressions:
            try:
                fn = expr.to_callable(variable_names)
                y_pred = fn(x)
                ss_res = float(np_local.sum((y - y_pred) ** 2))
                ss_tot = float(np_local.sum((y - y.mean()) ** 2))
                r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
                if r2 > best:
                    best = r2
            except Exception:
                continue
        return best

    def _single_run(
        self,
        x: np.ndarray,
        y: np.ndarray,
        variable_names: list[str],
        seed: int,
    ) -> list[SymbolicExpression]:
        """Run a single PySR regression pass.

        :param x: Input data of shape ``(n_samples, n_features)``.
        :param y: Target values of shape ``(n_samples,)``.
        :param variable_names: Names of input features.
        :param seed: Random seed for this run.
        :returns: Deduplicated Pareto front expressions, sorted by
            complexity.
        """
        try:
            from pysr import PySRRegressor
        except ImportError as exc:
            raise ImportError(
                "PySR is required for symbolic regression. "
                "Install with: pip install 'circ-rl[symbolic]'"
            ) from exc

        import sympy

        cfg = self._config

        logger.info(
            "Starting symbolic regression: {} samples, {} features, "
            "max_complexity={}, n_iterations={}, seed={}",
            x.shape[0],
            x.shape[1],
            cfg.max_complexity,
            cfg.n_iterations,
            seed,
        )

        extra_kwargs: dict[str, Any] = {}
        if cfg.nested_constraints is not None:
            extra_kwargs["nested_constraints"] = cfg.nested_constraints
        if cfg.complexity_of_operators is not None:
            extra_kwargs["complexity_of_operators"] = cfg.complexity_of_operators
        if cfg.constraints is not None:
            extra_kwargs["constraints"] = cfg.constraints

        model = PySRRegressor(
            niterations=cfg.n_iterations,
            maxsize=cfg.max_complexity,
            populations=cfg.populations,
            binary_operators=list(cfg.binary_operators),
            unary_operators=list(cfg.unary_operators),
            parsimony=cfg.parsimony,
            timeout_in_seconds=cfg.timeout_seconds,
            deterministic=cfg.deterministic,
            random_state=seed if cfg.deterministic else None,
            parallelism="serial" if cfg.deterministic else "multithreading",
            verbosity=0,
            **extra_kwargs,
        )

        model.fit(x, y, variable_names=variable_names)

        # Extract Pareto front from PySR results
        expressions: list[SymbolicExpression] = []
        equations = model.equations_

        if equations is None or len(equations) == 0:
            logger.warning("Symbolic regression produced no equations")
            return expressions

        for _, row in equations.iterrows():
            sympy_expr = row["sympy_format"]
            if not isinstance(sympy_expr, sympy.Basic):
                continue

            expr = SymbolicExpression.from_sympy(sympy.sympify(sympy_expr))
            expressions.append(expr)

        # Sort by complexity ascending
        expressions.sort(key=lambda e: e.complexity)

        # Deduplicate by expression string
        seen: set[str] = set()
        unique: list[SymbolicExpression] = []
        for expr in expressions:
            if expr.expression_str not in seen:
                seen.add(expr.expression_str)
                unique.append(expr)

        logger.info(
            "Symbolic regression complete: {} unique expressions on Pareto front",
            len(unique),
        )

        return unique
