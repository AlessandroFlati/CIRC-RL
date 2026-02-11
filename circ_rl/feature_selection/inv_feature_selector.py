"""Invariance-based feature selection for robust RL.

Implements Phase 2 of the CIRC-RL pipeline: select features that are
(1) ancestors of reward in the causal graph and (2) have stable causal
effects across environments.

See ``CIRC-RL_Framework.md`` Section 3.6, Phase 2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from loguru import logger

from circ_rl.causal_discovery.causal_graph import CausalGraph
from circ_rl.environments.data_collector import ExploratoryDataset
from circ_rl.feature_selection.causal_effect import CausalEffectEstimator


@dataclass(frozen=True)
class FeatureSelectionResult:
    """Result of invariance-based feature selection.

    :param selected_features: Names of features retained for policy learning.
    :param feature_mask: Boolean array of shape ``(n_features,)`` indicating
        which state dimensions are selected.
    :param ate_variance: Mapping of feature name to cross-environment
        variance of ATE estimates.
    :param rejected_features: Mapping of feature name to rejection reason.
    """

    selected_features: list[str]
    feature_mask: np.ndarray
    ate_variance: dict[str, float]
    rejected_features: dict[str, str]


class InvFeatureSelector:
    r"""Select features with invariant causal effects on reward.

    For each candidate feature :math:`f`:

    1. Check :math:`f \in \text{Anc}_{\mathcal{G}}(R)` (ancestor of reward).
    2. Estimate :math:`\text{ATE}_e(f \to R)` in each environment :math:`e`.
    3. Compute cross-environment variance:
       :math:`\text{Var}_e[\text{ATE}_e(f \to R)]`.
    4. Retain :math:`f` if :math:`\text{Var}_e < \epsilon`.

    :param epsilon: Maximum allowed cross-environment ATE variance.
    :param min_ate: Minimum absolute ATE to consider a feature relevant.

    See ``CIRC-RL_Framework.md`` Section 3.6, Phase 2:
    :math:`\mathcal{F}_{\text{robust}} = \{f : \text{Var}_e[P_e(R|do(f))] < \epsilon\}`.
    """

    def __init__(
        self,
        epsilon: float = 0.1,
        min_ate: float = 0.01,
    ) -> None:
        if epsilon <= 0:
            raise ValueError(f"epsilon must be > 0, got {epsilon}")
        self._epsilon = epsilon
        self._min_ate = min_ate
        self._estimator = CausalEffectEstimator()

    def select(
        self,
        dataset: ExploratoryDataset,
        graph: CausalGraph,
        state_feature_names: list[str],
    ) -> FeatureSelectionResult:
        """Run invariance-based feature selection.

        :param dataset: Multi-environment exploratory data.
        :param graph: The causal graph (must contain state features and reward).
        :param state_feature_names: Names of state features (matching columns
            of ``dataset.states``).
        :returns: FeatureSelectionResult with selected features and diagnostics.
        :raises ValueError: If state_feature_names length doesn't match state_dim.
        """
        if len(state_feature_names) != dataset.state_dim:
            raise ValueError(
                f"state_feature_names has {len(state_feature_names)} entries "
                f"but dataset has state_dim={dataset.state_dim}"
            )

        reward_ancestors = graph.ancestors_of_reward()
        unique_envs = sorted(set(dataset.env_ids.tolist()))

        logger.info(
            "Feature selection: {} candidate features, {} environments, "
            "epsilon={}, min_ate={}",
            len(state_feature_names),
            len(unique_envs),
            self._epsilon,
            self._min_ate,
        )

        selected: list[str] = []
        ate_variance: dict[str, float] = {}
        rejected: dict[str, str] = {}

        for feat_idx, feat_name in enumerate(state_feature_names):
            # Step 1: check if feature is an ancestor of reward
            if feat_name not in reward_ancestors:
                rejected[feat_name] = "not an ancestor of reward"
                logger.debug("{}: rejected (not ancestor of reward)", feat_name)
                continue

            # Step 2: estimate ATE per environment
            per_env_ates: list[float] = []
            for env_id in unique_envs:
                env_data = dataset.get_env_data(env_id)
                ate = self._estimate_feature_ate(
                    env_data, feat_idx, feat_name, graph, state_feature_names
                )
                per_env_ates.append(ate)

            # Step 3: compute cross-environment variance
            variance = float(np.var(per_env_ates))
            ate_variance[feat_name] = variance
            mean_ate = float(np.mean(np.abs(per_env_ates)))

            # Step 4: filter by variance and minimum effect size
            if variance >= self._epsilon:
                rejected[feat_name] = (
                    f"ATE variance too high: {variance:.4f} >= {self._epsilon}"
                )
                logger.debug(
                    "{}: rejected (variance={:.4f} >= epsilon={})",
                    feat_name,
                    variance,
                    self._epsilon,
                )
            elif mean_ate < self._min_ate:
                rejected[feat_name] = (
                    f"ATE too small: mean |ATE|={mean_ate:.4f} < {self._min_ate}"
                )
                logger.debug(
                    "{}: rejected (mean |ATE|={:.4f} < min_ate={})",
                    feat_name,
                    mean_ate,
                    self._min_ate,
                )
            else:
                selected.append(feat_name)
                logger.debug(
                    "{}: selected (variance={:.4f}, mean |ATE|={:.4f})",
                    feat_name,
                    variance,
                    mean_ate,
                )

        # Build mask over state features
        feature_mask = np.array(
            [name in selected for name in state_feature_names],
            dtype=np.bool_,
        )

        logger.info(
            "Selected {}/{} features: {}",
            len(selected),
            len(state_feature_names),
            selected,
        )

        return FeatureSelectionResult(
            selected_features=selected,
            feature_mask=feature_mask,
            ate_variance=ate_variance,
            rejected_features=rejected,
        )

    def _estimate_feature_ate(
        self,
        env_data: ExploratoryDataset,
        feat_idx: int,
        feat_name: str,
        graph: CausalGraph,
        state_feature_names: list[str],
    ) -> float:
        """Estimate the ATE of a single feature on reward in one environment.

        Constructs a flat data matrix from the environment data and uses
        CausalEffectEstimator with the back-door adjustment set.

        :returns: The ATE estimate (float).
        """
        flat_data = env_data.to_flat_array()  # (n, state_dim + action_dim + 1 + state_dim)

        # Build node names matching the flat array columns:
        # [state_features..., action, reward, next_state_features...]
        action_dim = 1 if env_data.actions.ndim == 1 else env_data.actions.shape[1]
        action_names = (
            ["action"]
            if action_dim == 1
            else [f"action_{i}" for i in range(action_dim)]
        )
        next_state_names = [f"{name}_next" for name in state_feature_names]
        node_names = state_feature_names + action_names + ["reward"] + next_state_names

        adjustment = CausalEffectEstimator.find_adjustment_set(
            graph, feat_name, graph.reward_node
        )

        # Only keep adjustment variables that actually exist in node_names
        valid_adjustment = frozenset(a for a in adjustment if a in node_names)

        result = CausalEffectEstimator.estimate_ate(
            flat_data, node_names, feat_name, graph.reward_node, valid_adjustment
        )
        return result.ate
