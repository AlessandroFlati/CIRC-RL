"""Experiment: validate KRR non-linear capture on SyntheticNonlinearEnv.

Runs the full causal discovery + feature selection pipeline on a synthetic
environment where ATE(s0) = k^2 (quadratic in env param k). Demonstrates
that KRR+RBF captures the non-linear relationship while Ridge alone cannot.

Usage::

    uv run python experiments/test_nonlinear_env.py
"""

from __future__ import annotations

import sys

import numpy as np
from loguru import logger

# Register the synthetic env with gymnasium
import circ_rl.environments.synthetic_nonlinear  # noqa: F401
from circ_rl.causal_discovery.builder import CausalGraphBuilder
from circ_rl.environments.data_collector import DataCollector
from circ_rl.environments.env_family import EnvironmentFamily
from circ_rl.feature_selection.inv_feature_selector import InvFeatureSelector


def _ridge_only_residual_variance(
    ate_vector: np.ndarray,
    ep_matrix: np.ndarray,
) -> float:
    """Compute residual variance using Ridge (linear) only -- no RBF.

    For comparison against the full KRR+RBF implementation.
    """
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    n_envs = ate_vector.shape[0]
    if n_envs <= 3:
        return float(np.var(ate_vector))

    scaler = StandardScaler()
    ep_scaled = scaler.fit_transform(ep_matrix)

    loo_residuals = np.zeros(n_envs)

    for i in range(n_envs):
        train_mask = np.ones(n_envs, dtype=bool)
        train_mask[i] = False
        x_train = ep_scaled[train_mask]
        y_train = ate_vector[train_mask]
        x_test = ep_scaled[i : i + 1]

        best_pred = float(np.mean(y_train))
        best_inner_mse = float("inf")

        for alpha in [0.1, 1.0, 10.0]:
            model = Ridge(alpha=alpha)
            model.fit(x_train, y_train)
            pred = float(model.predict(x_test)[0])

            # Inner LOO for model selection
            inner_preds = np.zeros(len(y_train))
            for j in range(len(y_train)):
                inner_mask = np.ones(len(y_train), dtype=bool)
                inner_mask[j] = False
                if inner_mask.sum() < 2:
                    continue
                inner_model = Ridge(alpha=alpha)
                inner_model.fit(x_train[inner_mask], y_train[inner_mask])
                inner_preds[j] = inner_model.predict(x_train[j : j + 1])[0]

            inner_mse = float(np.mean((y_train - inner_preds) ** 2))
            if inner_mse < best_inner_mse:
                best_inner_mse = inner_mse
                best_pred = pred

        loo_residuals[i] = ate_vector[i] - best_pred

    return float(np.var(loo_residuals))


def main() -> None:
    """Run the non-linear env experiment."""
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    n_envs = 8
    n_transitions = 2000

    print("=" * 70)
    print("SYNTHETIC NONLINEAR ENV (k: 1-5, 8 envs)")
    print("reward = k^2 + k^2 * s0 + 2.0 * s1 + 0.5 * action + 0.1 * noise")
    print("=" * 70)

    # 1. Create environment family
    ef = EnvironmentFamily.from_gymnasium(
        base_env="SyntheticNonlinear-v0",
        param_distributions={"k": (1.0, 5.0)},
        n_envs=n_envs,
        seed=42,
    )

    print(f"\nEnvironment params per env:")
    for i in range(n_envs):
        params = ef.get_env_params(i)
        print(f"  env {i}: k={params['k']:.3f}, k^2={params['k']**2:.3f}")

    # 2. Collect data
    collector = DataCollector(ef, include_env_params=True)
    ds = collector.collect(n_transitions_per_env=n_transitions, seed=42)

    # 3. Causal discovery
    state_names = ["s0", "s1", "s2"]
    action_names = ["action_0"]
    next_state_names = ["s0_next", "s1_next", "s2_next"]
    ep_names = ["ep_k"]
    node_names = state_names + action_names + ["reward"] + next_state_names + ep_names

    graph = CausalGraphBuilder.discover(
        ds,
        node_names,
        method="pc",
        alpha=0.01,
        env_param_names=ep_names,
        ep_correlation_threshold=0.05,
    )

    print(f"\nCausal graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
    print(f"Edges: {sorted(graph.edges)}")
    print(f"ep_k -> reward edge: {('ep_k', 'reward') in graph.edges}")
    print(f"ep_k ancestors of reward: {graph.env_param_ancestors_of('reward')}")

    # 4. Feature selection (full KRR+RBF)
    selector = InvFeatureSelector(
        epsilon=0.1, min_ate=0.01, enable_conditional_invariance=True
    )
    result = selector.select(ds, graph, state_names, env_param_names=ep_names)

    print(f"\n--- Feature Selection Results ---")
    print(f"Selected features: {result.selected_features}")
    print(f"Context-dependent: {result.context_dependent_features}")
    print(f"Context params: {result.context_param_names}")
    print(f"Rejected: {result.rejected_features}")

    # 5. Compute per-env ATEs manually for comparison
    print(f"\n--- Per-Env ATE Analysis ---")
    unique_envs = sorted(set(ds.env_ids.tolist()))
    per_env_ates_s0: list[float] = []
    per_env_ates_s1: list[float] = []
    per_env_k: list[float] = []

    for env_id in unique_envs:
        env_data = ds.get_env_data(env_id)
        flat = env_data.to_flat_array()
        # Columns: s0, s1, s2, action_0, reward, s0_next, s1_next, s2_next
        s0_col = flat[:, 0]
        s1_col = flat[:, 1]
        reward_col = flat[:, 4]

        # Simple regression coefficient as ATE proxy
        ate_s0 = float(np.corrcoef(s0_col, reward_col)[0, 1] * np.std(reward_col) / np.std(s0_col))
        ate_s1 = float(np.corrcoef(s1_col, reward_col)[0, 1] * np.std(reward_col) / np.std(s1_col))

        k_val = env_data.env_params[0, 0] if env_data.env_params is not None else 0.0
        per_env_ates_s0.append(ate_s0)
        per_env_ates_s1.append(ate_s1)
        per_env_k.append(float(k_val))

    print(f"{'Env':>3} | {'k':>6} | {'k^2':>6} | {'ATE(s0)':>8} | {'ATE(s1)':>8}")
    print("-" * 50)
    for i, env_id in enumerate(unique_envs):
        print(
            f"{env_id:3d} | {per_env_k[i]:6.3f} | {per_env_k[i]**2:6.3f} | "
            f"{per_env_ates_s0[i]:8.3f} | {per_env_ates_s1[i]:8.3f}"
        )

    ate_arr = np.array(per_env_ates_s0)
    ep_arr = np.array(per_env_k).reshape(-1, 1)
    raw_var = float(np.var(ate_arr))

    # 6. Compare Ridge-only vs full KRR+RBF
    ridge_residual = _ridge_only_residual_variance(ate_arr, ep_arr)
    krr_residual = InvFeatureSelector._residual_ate_variance(ate_arr, ep_arr)

    print(f"\n--- Residual Variance Comparison (s0) ---")
    print(f"Raw ATE variance:           {raw_var:.6f}")
    print(f"Ridge-only residual var:    {ridge_residual:.6f}")
    print(f"KRR+RBF residual var:       {krr_residual:.6f}")
    print(f"Epsilon threshold:          0.100000")
    print(f"Ridge rescues s0?           {'YES' if ridge_residual < 0.1 else 'NO'}")
    print(f"KRR+RBF rescues s0?         {'YES' if krr_residual < 0.1 else 'NO'}")

    reduction = (1 - krr_residual / raw_var) * 100
    print(f"\nKRR+RBF variance reduction: {reduction:.1f}%")

    if krr_residual < 0.1 and ridge_residual >= 0.1:
        print("\n>>> KRR+RBF succeeds where Ridge alone fails! <<<")
    elif krr_residual < 0.1 and ridge_residual < 0.1:
        print("\n>>> Both succeed, but KRR+RBF has lower residual. <<<")
    elif krr_residual >= 0.1:
        print("\n>>> WARNING: KRR+RBF did not rescue s0. Check parameters. <<<")

    # 7. s1 variance
    ate_s1_var = float(np.var(per_env_ates_s1))
    print(f"\nATE variance(s1): {ate_s1_var:.6f} (should be ~0)")


if __name__ == "__main__":
    main()
