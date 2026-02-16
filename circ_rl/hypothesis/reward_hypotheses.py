"""Reward hypothesis generation via symbolic regression.

Discovers analytic functional forms for the reward mechanism:
R = g(s, a) if invariant, or R = g(s, a; theta_e) if variant.

See ``CIRC-RL_Framework.md`` Section 3.4.2 (Reward Hypotheses).
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
    from circ_rl.feature_selection.inv_feature_selector import FeatureSelectionResult
    from circ_rl.hypothesis.expression import SymbolicExpression


class RewardHypothesisGenerator:
    r"""Generate reward hypotheses via symbolic regression.

    If the reward mechanism is invariant (verified in Phase 2), seeks:

    .. math::

        R = g(s, a)

    If not invariant, seeks:

    .. math::

        R = g(s, a; \theta_e)

    See ``CIRC-RL_Framework.md`` Section 3.4.2.

    :param sr_config: Configuration for symbolic regression.
    """

    def __init__(
        self,
        sr_config: SymbolicRegressionConfig | None = None,
    ) -> None:
        self._sr_config = sr_config or SymbolicRegressionConfig()

    def generate(
        self,
        dataset: ExploratoryDataset,
        feature_selection_result: FeatureSelectionResult,
        state_feature_names: list[str],
        register: HypothesisRegister,
        reward_is_invariant: bool,
        env_param_names: list[str] | None = None,
        derived_columns: dict[str, np.ndarray] | None = None,
    ) -> list[str]:
        """Generate reward hypotheses.

        :param dataset: Multi-environment exploratory data.
        :param feature_selection_result: Phase 2 feature selection results.
        :param state_feature_names: Names of state features.
        :param register: The hypothesis register to populate.
        :param reward_is_invariant: Whether the reward mechanism was found
            to be invariant in Phase 2.
        :param env_param_names: Names of environment parameters (raw, not
            ``ep_``-prefixed). Used only when reward is not invariant.
        :param derived_columns: Pre-computed derived feature arrays
            (name -> (N,) array) to include as additional SR features.
        :returns: List of hypothesis_ids that were registered.
        """
        logger.info(
            "Generating reward hypotheses: invariant={}, {} features",
            reward_is_invariant,
            len(state_feature_names),
        )

        regressor = SymbolicRegressor(self._sr_config)

        # Build input features
        x, y, var_names = self._build_regression_data(
            dataset,
            state_feature_names,
            reward_is_invariant,
            env_param_names,
            derived_columns,
        )

        logger.info(
            "Running SR for reward: {} samples, {} features (invariant={})",
            x.shape[0], x.shape[1], reward_is_invariant,
        )

        expressions = regressor.fit(x, y, var_names)

        all_ids: list[str] = []
        for i, expr in enumerate(expressions):
            entry = self._make_entry(expr, x, y, var_names, idx=i)
            register.register(entry)
            all_ids.append(entry.hypothesis_id)

        logger.info("Registered {} reward hypotheses", len(expressions))
        return all_ids

    def _build_regression_data(
        self,
        dataset: ExploratoryDataset,
        state_feature_names: list[str],
        reward_is_invariant: bool,
        env_param_names: list[str] | None,
        derived_columns: dict[str, np.ndarray] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Build input/target arrays for reward symbolic regression.

        :returns: Tuple of (x, y, variable_names).
        """
        states = dataset.states  # (N, state_dim)
        actions = dataset.actions  # (N,) or (N, action_dim)
        y = dataset.rewards  # (N,)

        actions_2d = actions if actions.ndim == 2 else actions[:, np.newaxis]
        action_dim = actions_2d.shape[1]
        action_names = (
            ["action"] if action_dim == 1
            else [f"action_{i}" for i in range(action_dim)]
        )

        var_names = list(state_feature_names) + action_names
        feature_arrays = [states, actions_2d]

        # Include env params only if reward is NOT invariant
        # (must come BEFORE derived features to match _build_features
        # positional env param matching)
        if (
            not reward_is_invariant
            and env_param_names
            and dataset.env_params is not None
        ):
            feature_arrays.append(dataset.env_params)
            var_names.extend(env_param_names)

        # Include derived features LAST (e.g., theta = atan2(s1, s0))
        if derived_columns:
            for name, col in derived_columns.items():
                var_names.append(name)
                feature_arrays.append(col[:, np.newaxis])

        x = np.column_stack(feature_arrays)  # (N, n_features)
        return x, y, var_names

    @staticmethod
    def _make_entry(
        expr: SymbolicExpression,
        x: np.ndarray,
        y: np.ndarray,
        var_names: list[str],
        idx: int,
    ) -> HypothesisEntry:
        """Create a HypothesisEntry from a symbolic expression.

        :returns: A new HypothesisEntry with training fit metrics computed.
        """
        hypothesis_id = f"reward_{idx}"

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
            target_variable="reward",
            expression=expr,
            complexity=expr.complexity,
            training_r2=r2,
            training_mse=mse,
            status=HypothesisStatus.UNTESTED,
        )
