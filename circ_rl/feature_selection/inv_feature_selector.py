"""Invariance-based feature selection for robust RL.

Implements Phase 2 of the CIRC-RL pipeline: select features that are
(1) ancestors of reward in the causal graph and (2) have stable causal
mechanisms across environments.

See ``CIRC-RL_Framework.md`` Section 3.6, Phase 2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

from circ_rl.feature_selection.causal_effect import CausalEffectEstimator

if TYPE_CHECKING:
    from circ_rl.causal_discovery.causal_graph import CausalGraph
    from circ_rl.environments.data_collector import ExploratoryDataset


@dataclass(frozen=True)
class FeatureSelectionResult:
    """Result of invariance-based feature selection.

    :param selected_features: Names of features retained for policy learning
        (those with ``feature_weights > 0``).
    :param feature_mask: Boolean array of shape ``(n_features,)`` indicating
        which state dimensions are selected (derived from ``feature_weights > 0``).
    :param feature_weights: Float array of shape ``(n_features,)`` with values
        in ``[0.0, 1.0]``. Mechanism-invariant features get weight 1.0;
        mechanism-non-invariant features get a soft penalty; non-ancestors and
        irrelevant features get weight 0.0.
    :param ate_variance: Mapping of feature name to cross-environment
        variance of ATE estimates. Only populated in ATE-variance mode
        (``use_mechanism_invariance=False``).
    :param mechanism_p_values: Mapping of feature name to minimum Chow-test
        p-value across environment pairs. Only populated in mechanism-invariance
        mode (``use_mechanism_invariance=True``).
    :param rejected_features: Mapping of feature name to rejection reason.
    :param context_dependent_features: Mapping of feature name to the list of
        environment-parameter node names (``ep_``-prefixed) that explain its
        non-invariance. Empty when conditional invariance is not enabled.
    :param context_param_names: Sorted list of all env-param names needed as
        policy context (union of all ``context_dependent_features`` values).
    """

    selected_features: list[str]
    feature_mask: np.ndarray
    feature_weights: np.ndarray
    ate_variance: dict[str, float]
    mechanism_p_values: dict[str, float]
    rejected_features: dict[str, str]
    context_dependent_features: dict[str, list[str]] = field(default_factory=dict)
    context_param_names: list[str] = field(default_factory=list)


class InvFeatureSelector:
    r"""Select features with invariant causal mechanisms on reward.

    Two modes of operation:

    **Mechanism invariance mode** (default, ``use_mechanism_invariance=True``):

    Uses a two-stage approach:

    *Stage 1 -- Global LOEO pre-check:*
    Train a gradient-boosted model on all environments except one, predict
    on the held-out environment. If the minimum leave-one-environment-out
    :math:`R^2` exceeds ``loeo_r2_threshold``, the reward mechanism is
    globally invariant and all features that pass ancestor and relevance
    checks receive weight 1.0.  This non-parametric pre-check avoids
    false positives from polynomial model misspecification (e.g., when the
    reward function involves transcendental functions like ``arctan2``).

    *Stage 2 -- Per-feature Chow test (fallback):*
    If the LOEO pre-check fails, run a per-feature polynomial Chow test
    using Frisch--Waugh--Lovell partial regression to identify which
    specific features have variant mechanisms.

    **ATE variance mode** (legacy, ``use_mechanism_invariance=False``):
    Uses cross-environment ATE variance with a hard threshold (epsilon).

    :param epsilon: Maximum allowed cross-environment ATE variance (ATE mode).
    :param min_ate: Minimum absolute ATE to consider a feature relevant.
    :param use_mechanism_invariance: When True (default), use mechanism
        invariance testing instead of ATE variance.
    :param mechanism_alpha: Significance level for the per-feature Chow test.
    :param min_weight: Floor for soft weights of non-invariant features.
        Features that fail mechanism invariance get weight
        ``max(min_weight, p_value * n_pairs / alpha)``.
    :param poly_degree: Polynomial degree for basis expansion in the Chow test.
    :param loeo_r2_threshold: Minimum leave-one-environment-out R^2 for the
        global pre-check to pass. Default 0.9.
    :param skip_ancestor_check: When True, skip the graph-based ancestor
        check and evaluate all features.
    :param enable_conditional_invariance: Legacy flag for ATE variance mode.

    See ``CIRC-RL_Framework.md`` Section 3.6, Phase 2.
    """

    def __init__(
        self,
        epsilon: float = 0.1,
        min_ate: float = 0.01,
        use_mechanism_invariance: bool = False,
        mechanism_alpha: float = 0.05,
        min_weight: float = 0.1,
        poly_degree: int = 2,
        loeo_r2_threshold: float = 0.9,
        enable_conditional_invariance: bool = False,
        skip_ancestor_check: bool = False,
    ) -> None:
        if epsilon <= 0:
            raise ValueError(f"epsilon must be > 0, got {epsilon}")
        if mechanism_alpha <= 0 or mechanism_alpha >= 1:
            raise ValueError(
                f"mechanism_alpha must be in (0, 1), got {mechanism_alpha}"
            )
        if min_weight < 0 or min_weight > 1:
            raise ValueError(f"min_weight must be in [0, 1], got {min_weight}")
        self._epsilon = epsilon
        self._min_ate = min_ate
        self._use_mechanism_invariance = use_mechanism_invariance
        self._mechanism_alpha = mechanism_alpha
        self._min_weight = min_weight
        self._poly_degree = poly_degree
        self._loeo_r2_threshold = loeo_r2_threshold
        self._enable_conditional_invariance = enable_conditional_invariance
        self._skip_ancestor_check = skip_ancestor_check
        self._estimator = CausalEffectEstimator()

    def select(
        self,
        dataset: ExploratoryDataset,
        graph: CausalGraph,
        state_feature_names: list[str],
        env_param_names: list[str] | None = None,
    ) -> FeatureSelectionResult:
        """Run invariance-based feature selection.

        :param dataset: Multi-environment exploratory data.
        :param graph: The causal graph (must contain state features and reward).
        :param state_feature_names: Names of state features (matching columns
            of ``dataset.states``).
        :param env_param_names: Names of environment-parameter nodes
            (``ep_``-prefixed) in the graph.
        :returns: FeatureSelectionResult with selected features and diagnostics.
        :raises ValueError: If state_feature_names length doesn't match state_dim.
        """
        if len(state_feature_names) != dataset.state_dim:
            raise ValueError(
                f"state_feature_names has {len(state_feature_names)} entries "
                f"but dataset has state_dim={dataset.state_dim}"
            )

        if self._use_mechanism_invariance:
            return self._select_mechanism_invariance(
                dataset, graph, state_feature_names,
            )
        return self._select_ate_variance(
            dataset, graph, state_feature_names, env_param_names,
        )

    # ------------------------------------------------------------------
    # Mechanism invariance mode (new)
    # ------------------------------------------------------------------

    def _select_mechanism_invariance(
        self,
        dataset: ExploratoryDataset,
        graph: CausalGraph,
        state_feature_names: list[str],
    ) -> FeatureSelectionResult:
        """Select features using mechanism invariance.

        Two-stage approach:

        1. **Global LOEO pre-check**: train a gradient-boosted model with
           leave-one-environment-out and check R^2. If the held-out R^2
           is consistently high, the reward mechanism is globally invariant
           and all eligible features get weight 1.0.
        2. **Per-feature Chow test (fallback)**: if the LOEO pre-check
           fails, use FWL partial regression + polynomial Chow test to
           identify which specific features have variant mechanisms.
        """
        reward_ancestors = graph.ancestors_of_reward()
        unique_envs = sorted(set(dataset.env_ids.tolist()))
        n_pairs = max(1, len(list(combinations(unique_envs, 2))))

        logger.info(
            "Feature selection (mechanism invariance): {} candidates, "
            "{} envs, alpha={}, poly_degree={}, min_weight={}",
            len(state_feature_names),
            len(unique_envs),
            self._mechanism_alpha,
            self._poly_degree,
            self._min_weight,
        )

        # ----------------------------------------------------------
        # Stage 0: filter by ancestor check and ATE relevance
        # ----------------------------------------------------------
        eligible: list[tuple[int, str]] = []  # (feat_idx, feat_name)
        rejected: dict[str, str] = {}

        for feat_idx, feat_name in enumerate(state_feature_names):
            if not self._skip_ancestor_check and feat_name not in reward_ancestors:
                rejected[feat_name] = "not an ancestor of reward"
                logger.debug("{}: rejected (not ancestor of reward)", feat_name)
                continue

            per_env_ates: list[float] = []
            for env_id in unique_envs:
                env_data = dataset.get_env_data(env_id)
                ate = self._estimate_feature_ate(
                    env_data, feat_idx, feat_name, graph, state_feature_names,
                )
                per_env_ates.append(ate)

            mean_ate = float(np.mean(np.abs(per_env_ates)))
            if mean_ate < self._min_ate:
                rejected[feat_name] = (
                    f"ATE too small: mean |ATE|={mean_ate:.4f} < {self._min_ate}"
                )
                logger.debug(
                    "{}: rejected (mean |ATE|={:.4f} < min_ate={})",
                    feat_name, mean_ate, self._min_ate,
                )
                continue

            eligible.append((feat_idx, feat_name))

        # ----------------------------------------------------------
        # Stage 1: Global LOEO pre-check (non-parametric)
        # ----------------------------------------------------------
        globally_invariant = False
        if len(unique_envs) >= 2:
            min_loeo_r2 = self._loeo_mechanism_precheck(
                dataset, unique_envs,
            )
            globally_invariant = min_loeo_r2 >= self._loeo_r2_threshold
            if globally_invariant:
                logger.info(
                    "LOEO pre-check PASSED (min R^2={:.4f} >= {:.4f}): "
                    "mechanism globally invariant, all eligible features "
                    "get weight 1.0",
                    min_loeo_r2, self._loeo_r2_threshold,
                )
            else:
                logger.info(
                    "LOEO pre-check FAILED (min R^2={:.4f} < {:.4f}): "
                    "falling back to per-feature Chow test",
                    min_loeo_r2, self._loeo_r2_threshold,
                )

        # ----------------------------------------------------------
        # Stage 2: assign weights
        # ----------------------------------------------------------
        selected: list[str] = []
        weights: dict[str, float] = {}
        mechanism_p_values: dict[str, float] = {}

        for feat_idx, feat_name in eligible:
            if globally_invariant or len(unique_envs) < 2:
                # LOEO passed: invariant mechanism for all features
                mechanism_p_values[feat_name] = 1.0
                weights[feat_name] = 1.0
                selected.append(feat_name)
                logger.debug(
                    "{}: weight=1.0 (LOEO globally invariant)", feat_name,
                )
                continue

            # Per-feature FWL Chow test (fallback)
            min_p = self._test_feature_mechanism_invariance(
                dataset, graph, feat_idx, feat_name,
                state_feature_names, unique_envs,
            )
            mechanism_p_values[feat_name] = min_p

            bonferroni_alpha = self._mechanism_alpha / n_pairs
            if min_p >= bonferroni_alpha:
                w = 1.0
                logger.debug(
                    "{}: mechanism invariant (min_p={:.4f} >= {:.4f}), "
                    "weight=1.0",
                    feat_name, min_p, bonferroni_alpha,
                )
            else:
                w = max(self._min_weight, min_p / bonferroni_alpha)
                logger.info(
                    "{}: mechanism non-invariant (min_p={:.4f} < {:.4f}), "
                    "weight={:.3f}",
                    feat_name, min_p, bonferroni_alpha, w,
                )
            weights[feat_name] = w
            selected.append(feat_name)

        # Build weight and mask arrays
        feature_weights = np.array(
            [weights.get(name, 0.0) for name in state_feature_names],
            dtype=np.float32,
        )
        feature_mask = feature_weights > 0

        logger.info(
            "Selected {}/{} features: {} (weights: {})",
            len(selected),
            len(state_feature_names),
            selected,
            {n: f"{weights[n]:.3f}" for n in selected},
        )

        return FeatureSelectionResult(
            selected_features=selected,
            feature_mask=feature_mask,
            feature_weights=feature_weights,
            ate_variance={},
            mechanism_p_values=mechanism_p_values,
            rejected_features=rejected,
        )

    def _test_feature_mechanism_invariance(
        self,
        dataset: ExploratoryDataset,
        graph: CausalGraph,
        feat_idx: int,
        feat_name: str,
        state_feature_names: list[str],
        unique_envs: list[int],
    ) -> float:
        r"""Test mechanism invariance for a single feature across all env pairs.

        Uses the Frisch--Waugh--Lovell (FWL) partial regression approach
        to obtain a **per-feature** invariance test while controlling for
        omitted-variable bias:

        1. In each environment, linearly regress both the reward and
           feature *f* on all **other** state features.
        2. Collect the residuals :math:`\tilde{R}` and :math:`\tilde{f}`.
        3. Run a polynomial Chow test on
           :math:`\tilde{R} \sim \text{poly}(\tilde{f})` across environment
           pairs.

        By FWL, the partial regression coefficient of *f* in
        :math:`\tilde{R} \sim \tilde{f}` equals its coefficient in the
        full multiple regression :math:`R \sim \text{all features}`.
        The Chow test on the partialed relationship therefore tests
        *only* whether *f*'s contribution to reward changes -- not
        whether any other feature's contribution changes.

        :returns: Minimum p-value across environment pairs (high = invariant).
        """
        n_features = len(state_feature_names)
        other_idxs = [i for i in range(n_features) if i != feat_idx]

        env_ids = dataset.env_ids
        min_p = 1.0

        for env_a, env_b in combinations(unique_envs, 2):
            mask_a = env_ids == env_a
            mask_b = env_ids == env_b
            states_a = dataset.states[mask_a]  # (n_a, n_features)
            states_b = dataset.states[mask_b]  # (n_b, n_features)
            rewards_a = dataset.rewards[mask_a]  # (n_a,)
            rewards_b = dataset.rewards[mask_b]  # (n_b,)

            if other_idxs:
                # FWL step: partial out other features per env
                resid_a = self._fwl_partial_out(
                    states_a, rewards_a, feat_idx, other_idxs,
                    poly_degree=self._poly_degree,
                )
                resid_b = self._fwl_partial_out(
                    states_b, rewards_b, feat_idx, other_idxs,
                    poly_degree=self._poly_degree,
                )
            else:
                # No other features to partial out; use raw data
                resid_a = np.column_stack(
                    [states_a[:, feat_idx], rewards_a],
                )
                resid_b = np.column_stack(
                    [states_b[:, feat_idx], rewards_b],
                )

            # Chow test on the partial regression:
            # resid_reward ~ poly(resid_feature)
            p_val = CausalEffectEstimator.test_mechanism_invariance(
                resid_a, resid_b,
                target_idx=1,
                predictor_idxs=[0],
                poly_degree=self._poly_degree,
            )
            min_p = min(min_p, p_val)

        return min_p

    @staticmethod
    def _fwl_partial_out(
        states: np.ndarray,
        rewards: np.ndarray,
        feat_idx: int,
        other_idxs: list[int],
        poly_degree: int = 2,
    ) -> np.ndarray:
        r"""Frisch--Waugh--Lovell residualization with polynomial controls.

        Regresses both reward and ``states[:, feat_idx]`` on
        **polynomial features** of the control variables
        ``states[:, other_idxs]`` and returns the residuals stacked as
        ``(n, 2)`` -- column 0 is the partialed feature, column 1 is
        the partialed reward.

        Using polynomial (rather than linear) controls is essential when
        the reward function has non-linear interactions between features
        (e.g., Pendulum's ``arctan2(sin, cos)``).  Linear partialing
        would leave non-linear residual confounding, causing the
        downstream Chow test to falsely reject invariant mechanisms.

        :param states: State matrix, shape ``(n, n_features)``.
        :param rewards: Reward vector, shape ``(n,)``.
        :param feat_idx: Index of the feature being tested.
        :param other_idxs: Indices of control features to partial out.
        :param poly_degree: Polynomial degree for control expansion.
        :returns: Array of shape ``(n, 2)`` with columns
            ``[resid_feature, resid_reward]``.
        """
        from sklearn.preprocessing import PolynomialFeatures

        n = states.shape[0]
        controls = states[:, other_idxs]  # (n, n_other)

        # Polynomial expansion of controls
        degree = poly_degree
        pf = PolynomialFeatures(degree=degree, include_bias=False)
        poly_controls = pf.fit_transform(controls)  # (n, n_poly)

        # Safeguard: reduce degree if too many features for sample size
        while degree > 1 and (poly_controls.shape[1] + 1) > n / 5:
            degree -= 1
            pf = PolynomialFeatures(degree=degree, include_bias=False)
            poly_controls = pf.fit_transform(controls)

        design = np.column_stack(
            [np.ones(n), poly_controls],
        )  # (n, n_poly + 1)

        # Partial out controls from reward
        beta_r = np.linalg.lstsq(design, rewards, rcond=None)[0]
        resid_reward = rewards - design @ beta_r  # (n,)

        # Partial out controls from feature f
        feature = states[:, feat_idx]  # (n,)
        beta_f = np.linalg.lstsq(design, feature, rcond=None)[0]
        resid_feature = feature - design @ beta_f  # (n,)

        return np.column_stack([resid_feature, resid_reward])  # (n, 2)

    @staticmethod
    def _loeo_mechanism_precheck(
        dataset: ExploratoryDataset,
        unique_envs: list[int],
    ) -> float:
        r"""Leave-One-Environment-Out R^2 pre-check for global mechanism invariance.

        Trains a ``HistGradientBoostingRegressor`` on all environments
        except one, then evaluates on the held-out environment. Returns
        the **minimum** R^2 across all held-out environments.

        A high minimum R^2 (e.g., > 0.9) indicates that a model trained
        on other environments predicts the held-out environment's rewards
        almost perfectly, which is only possible if the reward mechanism
        is the same across environments.

        This non-parametric pre-check is robust to model misspecification:
        gradient-boosted trees can capture arbitrary non-linear reward
        functions (including transcendental functions like ``arctan2``)
        that polynomial models cannot.

        :param dataset: Multi-environment exploratory data.
        :param unique_envs: Sorted list of environment IDs.
        :returns: Minimum leave-one-environment-out R^2.
        """
        from sklearn.ensemble import HistGradientBoostingRegressor

        states = dataset.states
        rewards = dataset.rewards
        env_ids = dataset.env_ids
        min_r2 = 1.0

        for held_out in unique_envs:
            train_mask = env_ids != held_out
            test_mask = env_ids == held_out

            model = HistGradientBoostingRegressor(
                max_iter=200, max_depth=5, random_state=42,
            )
            model.fit(states[train_mask], rewards[train_mask])

            y_pred = model.predict(states[test_mask])
            y_true = rewards[test_mask]
            ss_res = float(np.sum((y_true - y_pred) ** 2))
            ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))

            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            logger.debug(
                "LOEO pre-check: held-out env {} -> R^2={:.4f}", held_out, r2,
            )
            min_r2 = min(min_r2, r2)

        return min_r2

    # ------------------------------------------------------------------
    # ATE variance mode (legacy, backward compat)
    # ------------------------------------------------------------------

    def _select_ate_variance(
        self,
        dataset: ExploratoryDataset,
        graph: CausalGraph,
        state_feature_names: list[str],
        env_param_names: list[str] | None,
    ) -> FeatureSelectionResult:
        """Select features using cross-environment ATE variance (legacy)."""
        reward_ancestors = graph.ancestors_of_reward()
        unique_envs = sorted(set(dataset.env_ids.tolist()))

        logger.info(
            "Feature selection (ATE variance): {} candidates, {} envs, "
            "epsilon={}, min_ate={}, conditional_invariance={}",
            len(state_feature_names),
            len(unique_envs),
            self._epsilon,
            self._min_ate,
            self._enable_conditional_invariance,
        )

        selected: list[str] = []
        ate_variance: dict[str, float] = {}
        rejected: dict[str, str] = {}
        context_dependent: dict[str, list[str]] = {}

        for feat_idx, feat_name in enumerate(state_feature_names):
            # Step 1: check if feature is an ancestor of reward
            if not self._skip_ancestor_check and feat_name not in reward_ancestors:
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
                # Step 4a: conditional invariance check
                if self._try_conditional_invariance(
                    dataset, graph, feat_idx, feat_name,
                    state_feature_names, env_param_names,
                    unique_envs, variance, mean_ate,
                    context_dependent, selected,
                    per_env_ates=per_env_ates,
                ):
                    continue

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

        # Build union of all env params needed as context
        all_context_params: set[str] = set()
        for ep_list in context_dependent.values():
            all_context_params.update(ep_list)
        context_param_names_out = sorted(all_context_params)

        # Build mask and weights over state features
        feature_mask = np.array(
            [name in selected for name in state_feature_names],
            dtype=np.bool_,
        )
        feature_weights = feature_mask.astype(np.float32)

        logger.info(
            "Selected {}/{} features: {} (context_dependent: {})",
            len(selected),
            len(state_feature_names),
            selected,
            list(context_dependent.keys()),
        )

        return FeatureSelectionResult(
            selected_features=selected,
            feature_mask=feature_mask,
            feature_weights=feature_weights,
            ate_variance=ate_variance,
            mechanism_p_values={},
            rejected_features=rejected,
            context_dependent_features=context_dependent,
            context_param_names=context_param_names_out,
        )

    def _try_conditional_invariance(
        self,
        dataset: ExploratoryDataset,
        graph: CausalGraph,
        feat_idx: int,
        feat_name: str,
        state_feature_names: list[str],
        env_param_names: list[str] | None,
        unique_envs: list[int],
        variance: float,
        mean_ate: float,
        context_dependent: dict[str, list[str]],
        selected: list[str],
        per_env_ates: list[float] | None = None,
    ) -> bool:
        """Attempt to rescue a high-variance feature via conditional invariance.

        Checks whether the cross-environment ATE variation is *explained by*
        env-param variation. Since env params are constant within each env,
        we cannot condition on them inside per-env regressions. Instead we
        regress the per-env ATEs on the per-env ep param values and check
        whether the *residual* variance (unexplained by ep params) falls
        below epsilon.

        :param per_env_ates: Already-computed per-env ATE values (one per env
            in ``unique_envs`` order). Required for residual-variance check.
        :returns: True if the feature was rescued (added to selected), False otherwise.
        """
        if not self._enable_conditional_invariance:
            return False
        if not env_param_names:
            return False
        if per_env_ates is None:
            return False

        # Find env-param ancestors of this feature AND of reward.
        ep_ancestors_feat = graph.env_param_ancestors_of(feat_name)
        ep_ancestors_reward = graph.env_param_ancestors_of(graph.reward_node)
        relevant_eps = ep_ancestors_feat | ep_ancestors_reward

        if not relevant_eps:
            return False

        if dataset.env_params is None:
            return False

        # Build per-env ep param matrix: (n_envs, n_relevant_eps)
        graph_ep_sorted = sorted(graph.env_param_nodes)
        relevant_ep_indices = [
            graph_ep_sorted.index(ep)
            for ep in sorted(relevant_eps)
            if ep in graph_ep_sorted
        ]
        if not relevant_ep_indices:
            return False

        # Extract the ep param value for each env (constant within env)
        ep_values_per_env = []  # (n_envs, n_relevant_eps)
        for env_id in unique_envs:
            env_data = dataset.get_env_data(env_id)
            if env_data.env_params is None:
                return False
            ep_row = env_data.env_params[0, relevant_ep_indices]  # (n_relevant_eps,)
            ep_values_per_env.append(ep_row)

        ep_matrix = np.array(ep_values_per_env)  # (n_envs, n_relevant_eps)
        ate_vector = np.array(per_env_ates)  # (n_envs,)

        residual_variance = self._residual_ate_variance(ate_vector, ep_matrix)

        if residual_variance < self._epsilon and mean_ate >= self._min_ate:
            ep_list = sorted(relevant_eps)
            context_dependent[feat_name] = ep_list
            selected.append(feat_name)
            logger.info(
                "{}: context-dependent (variance {:.4f} -> residual {:.4f} after "
                "regressing on {})",
                feat_name,
                variance,
                residual_variance,
                ep_list,
            )
            return True

        logger.debug(
            "{}: conditional invariance failed (residual_variance={:.4f})",
            feat_name,
            residual_variance,
        )
        return False

    @staticmethod
    def _residual_ate_variance(
        ate_vector: np.ndarray,
        ep_matrix: np.ndarray,
    ) -> float:
        """Compute the residual variance of ATEs after regressing on ep params.

        Uses Ridge regression and Kernel Ridge Regression with polynomial
        (degree 2, 3) and RBF kernels, selected via leave-one-out
        cross-validation.

        :param ate_vector: Per-environment ATE values, shape ``(n_envs,)``.
        :param ep_matrix: Per-environment ep param values, shape ``(n_envs, n_ep_params)``.
        :returns: Variance of LOO-CV residuals.
        """
        from sklearn.kernel_ridge import KernelRidge
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        n_envs = ate_vector.shape[0]

        if n_envs <= 3:
            return float(np.var(ate_vector))

        scaler = StandardScaler()
        ep_scaled = scaler.fit_transform(ep_matrix)  # (n_envs, n_ep)

        loo_residuals = np.zeros(n_envs)
        alphas = [0.1, 1.0, 10.0]

        def _make_models() -> list[tuple[str, object]]:
            models: list[tuple[str, object]] = []
            for a in alphas:
                models.append((f"ridge_{a}", Ridge(alpha=a)))
                models.append((f"krr_poly2_{a}", KernelRidge(alpha=a, kernel="poly", degree=2)))
                models.append((f"krr_poly3_{a}", KernelRidge(alpha=a, kernel="poly", degree=3)))
                models.append((f"krr_rbf_{a}", KernelRidge(alpha=a, kernel="rbf")))
            return models

        for i in range(n_envs):
            train_mask = np.ones(n_envs, dtype=bool)
            train_mask[i] = False

            x_train = ep_scaled[train_mask]
            y_train = ate_vector[train_mask]
            x_test = ep_scaled[i : i + 1]

            best_pred = float(np.mean(y_train))
            best_inner_mse = float("inf")

            for _name, model_template in _make_models():
                try:
                    import copy

                    model = copy.deepcopy(model_template)
                    model.fit(x_train, y_train)  # type: ignore[union-attr]
                    pred = float(model.predict(x_test)[0])  # type: ignore[union-attr]

                    inner_preds = np.zeros(len(y_train))
                    for j in range(len(y_train)):
                        inner_mask = np.ones(len(y_train), dtype=bool)
                        inner_mask[j] = False
                        if inner_mask.sum() < 2:
                            continue
                        inner_model = copy.deepcopy(model_template)
                        inner_model.fit(  # type: ignore[union-attr]
                            x_train[inner_mask], y_train[inner_mask]
                        )
                        inner_preds[j] = inner_model.predict(  # type: ignore[union-attr]
                            x_train[j : j + 1]
                        )[0]

                    inner_mse = float(np.mean((y_train - inner_preds) ** 2))
                    if inner_mse < best_inner_mse:
                        best_inner_mse = inner_mse
                        best_pred = pred
                except np.linalg.LinAlgError:
                    continue

            loo_residuals[i] = ate_vector[i] - best_pred

        return float(np.var(loo_residuals))

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

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
