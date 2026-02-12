"""Invariance-based feature selection for robust RL.

Implements Phase 2 of the CIRC-RL pipeline: select features that are
(1) ancestors of reward in the causal graph and (2) have stable causal
effects across environments.

See ``CIRC-RL_Framework.md`` Section 3.6, Phase 2.
"""

from __future__ import annotations

from dataclasses import dataclass, field

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
    :param context_dependent_features: Mapping of feature name to the list of
        environment-parameter node names (``ep_``-prefixed) that explain its
        non-invariance. Empty when conditional invariance is not enabled.
    :param context_param_names: Sorted list of all env-param names needed as
        policy context (union of all ``context_dependent_features`` values).
    """

    selected_features: list[str]
    feature_mask: np.ndarray
    ate_variance: dict[str, float]
    rejected_features: dict[str, str]
    context_dependent_features: dict[str, list[str]] = field(default_factory=dict)
    context_param_names: list[str] = field(default_factory=list)


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
        enable_conditional_invariance: bool = False,
    ) -> None:
        if epsilon <= 0:
            raise ValueError(f"epsilon must be > 0, got {epsilon}")
        self._epsilon = epsilon
        self._min_ate = min_ate
        self._enable_conditional_invariance = enable_conditional_invariance
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
            (``ep_``-prefixed) in the graph. Required when
            ``enable_conditional_invariance`` is True.
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

        # Build mask over state features
        feature_mask = np.array(
            [name in selected for name in state_feature_names],
            dtype=np.bool_,
        )

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
            ate_variance=ate_variance,
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
        # Using ancestors (not just direct parents) catches indirect paths
        # like ep_g -> s_next -> reward.
        ep_ancestors_feat = graph.env_param_ancestors_of(feat_name)
        ep_ancestors_reward = graph.env_param_ancestors_of(graph.reward_node)
        relevant_eps = ep_ancestors_feat | ep_ancestors_reward

        if not relevant_eps:
            return False

        if dataset.env_params is None:
            return False

        # Build per-env ep param matrix: (n_envs, n_relevant_eps)
        # We need to map relevant ep names to dataset env_params column indices.
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
            # All rows have the same ep values within an env; take row 0
            ep_row = env_data.env_params[0, relevant_ep_indices]  # (n_relevant_eps,)
            ep_values_per_env.append(ep_row)

        ep_matrix = np.array(ep_values_per_env)  # (n_envs, n_relevant_eps)
        ate_vector = np.array(per_env_ates)  # (n_envs,)

        # Regress ATE ~ ep_params via OLS to partial out the ep effect
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
        cross-validation. Ridge handles linear relationships; polynomial
        kernels capture quadratic/cubic ATE-vs-ep patterns with proper
        extrapolation at LOO boundary points; the RBF kernel captures
        arbitrary non-linear patterns (interpolation only).
        Inner LOO on the training fold selects the best model.

        When ``n_envs <= 3``, falls back to raw ATE variance (too few
        points for any meaningful regression).

        LOO-CV residuals provide an unbiased estimate of generalization
        variance, avoiding the optimistic bias of in-sample residuals.

        :param ate_vector: Per-environment ATE values, shape ``(n_envs,)``.
        :param ep_matrix: Per-environment ep param values, shape ``(n_envs, n_ep_params)``.
        :returns: Variance of LOO-CV residuals.
        """
        from sklearn.kernel_ridge import KernelRidge
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        n_envs = ate_vector.shape[0]

        # Need at least 4 environments for meaningful LOO-CV
        if n_envs <= 3:
            return float(np.var(ate_vector))

        # Standardize ep params for stable kernel computation
        scaler = StandardScaler()
        ep_scaled = scaler.fit_transform(ep_matrix)  # (n_envs, n_ep)

        # LOO-CV: predict each point using model trained on the rest
        loo_residuals = np.zeros(n_envs)
        alphas = [0.1, 1.0, 10.0]

        # Model candidates:
        # - Ridge: linear with intercept, extrapolates well for linear relationships
        # - KRR+poly(2): captures quadratic ATE-vs-ep and extrapolates properly
        # - KRR+poly(3): captures cubic relationships
        # - KRR+RBF: captures arbitrary non-linear patterns (interpolation only)
        # KernelRidge with linear kernel omits the intercept, so Ridge is used.
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

            x_train = ep_scaled[train_mask]  # (n_envs-1, n_ep)
            y_train = ate_vector[train_mask]  # (n_envs-1,)
            x_test = ep_scaled[i : i + 1]  # (1, n_ep)

            best_pred = float(np.mean(y_train))
            best_inner_mse = float("inf")

            for _name, model_template in _make_models():
                try:
                    import copy

                    model = copy.deepcopy(model_template)
                    model.fit(x_train, y_train)  # type: ignore[union-attr]
                    pred = float(model.predict(x_test)[0])  # type: ignore[union-attr]

                    # Inner LOO on training fold for model selection
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
