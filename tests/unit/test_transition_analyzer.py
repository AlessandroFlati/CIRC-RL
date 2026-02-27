"""Tests for circ_rl.feature_selection.transition_analyzer."""

import numpy as np
import pytest

from circ_rl.environments.data_collector import ExploratoryDataset
from circ_rl.feature_selection.transition_analyzer import TransitionAnalyzer


def _make_known_scaling_dataset(
    n_per_env: int = 1000,
    n_envs: int = 4,
    seed: int = 42,
) -> tuple[ExploratoryDataset, list[float]]:
    """Synthetic dataset where delta_s = alpha_e * action + noise.

    Each environment has a known action scaling factor alpha_e.
    State is 2D, action is 1D.
    """
    rng = np.random.RandomState(seed)
    # Known per-env scales: action effectiveness varies
    alphas = [0.5, 1.0, 2.0, 4.0][:n_envs]

    all_states = []
    all_actions = []
    all_next_states = []
    all_rewards = []
    all_env_ids = []

    for env_id in range(n_envs):
        alpha = alphas[env_id]
        states = rng.randn(n_per_env, 2).astype(np.float32)
        actions = rng.randn(n_per_env, 1).astype(np.float32)

        # delta_s = alpha * action + small noise (linear in action only)
        delta_s = alpha * actions + 0.01 * rng.randn(n_per_env, 1).astype(np.float32)
        # Both state dims affected equally for simplicity
        delta = np.column_stack([delta_s[:, 0], delta_s[:, 0]])  # (n, 2)
        next_states = states + delta

        rewards = rng.randn(n_per_env).astype(np.float32)
        env_ids = np.full(n_per_env, env_id, dtype=np.int32)

        all_states.append(states)
        all_actions.append(actions)
        all_next_states.append(next_states)
        all_rewards.append(rewards)
        all_env_ids.append(env_ids)

    dataset = ExploratoryDataset(
        states=np.concatenate(all_states),
        actions=np.concatenate(all_actions),
        next_states=np.concatenate(all_next_states),
        rewards=np.concatenate(all_rewards),
        env_ids=np.concatenate(all_env_ids),
    )
    return dataset, alphas


def _make_invariant_transition_dataset(
    n_per_env: int = 1000,
    n_envs: int = 3,
    seed: int = 42,
) -> ExploratoryDataset:
    """Dataset with SAME transition function across all environments."""
    rng = np.random.RandomState(seed)
    all_states = []
    all_actions = []
    all_next_states = []
    all_rewards = []
    all_env_ids = []

    for env_id in range(n_envs):
        states = rng.randn(n_per_env, 2).astype(np.float32)
        actions = rng.randn(n_per_env, 1).astype(np.float32)
        # SAME transition function across all envs
        next_states = states + 0.5 * actions + 0.01 * rng.randn(n_per_env, 2).astype(np.float32)
        rewards = rng.randn(n_per_env).astype(np.float32)
        env_ids = np.full(n_per_env, env_id, dtype=np.int32)

        all_states.append(states)
        all_actions.append(actions)
        all_next_states.append(next_states)
        all_rewards.append(rewards)
        all_env_ids.append(env_ids)

    return ExploratoryDataset(
        states=np.concatenate(all_states),
        actions=np.concatenate(all_actions),
        next_states=np.concatenate(all_next_states),
        rewards=np.concatenate(all_rewards),
        env_ids=np.concatenate(all_env_ids),
    )


def _make_variant_transition_dataset(
    n_per_env: int = 1000,
    n_envs: int = 3,
    seed: int = 42,
) -> ExploratoryDataset:
    """Dataset with DIFFERENT transition functions across environments."""
    rng = np.random.RandomState(seed)
    all_states = []
    all_actions = []
    all_next_states = []
    all_rewards = []
    all_env_ids = []

    for env_id in range(n_envs):
        states = rng.randn(n_per_env, 2).astype(np.float32)
        actions = rng.randn(n_per_env, 1).astype(np.float32)
        # Different scaling per env: alpha varies significantly
        alpha = 0.2 + env_id * 2.0  # 0.2, 2.2, 4.2
        next_states = states + alpha * actions + 0.01 * rng.randn(n_per_env, 2).astype(np.float32)
        rewards = rng.randn(n_per_env).astype(np.float32)
        env_ids = np.full(n_per_env, env_id, dtype=np.int32)

        all_states.append(states)
        all_actions.append(actions)
        all_next_states.append(next_states)
        all_rewards.append(rewards)
        all_env_ids.append(env_ids)

    return ExploratoryDataset(
        states=np.concatenate(all_states),
        actions=np.concatenate(all_actions),
        next_states=np.concatenate(all_next_states),
        rewards=np.concatenate(all_rewards),
        env_ids=np.concatenate(all_env_ids),
    )


class TestTransitionAnalyzer:

    def test_dynamics_scale_detects_known_scaling(self) -> None:
        """Estimated dynamics scales should match known alpha_e values."""
        dataset, alphas = _make_known_scaling_dataset(n_per_env=2000, n_envs=4)
        analyzer = TransitionAnalyzer()
        result = analyzer.analyze(dataset, ["s0", "s1"], action_dim=1)

        assert result.dynamics_scales.shape == (4,)
        # Each env's B_e matrix is [[alpha], [alpha]] so
        # Frobenius norm = sqrt(alpha^2 + alpha^2) = alpha * sqrt(2)
        for i, alpha in enumerate(alphas):
            expected_scale = alpha * np.sqrt(2)
            assert result.dynamics_scales[i] == pytest.approx(
                expected_scale, rel=0.1
            ), f"env {i}: expected ~{expected_scale:.3f}, got {result.dynamics_scales[i]:.3f}"

    def test_reference_scale_is_mean(self) -> None:
        """Reference scale should be the mean of per-env scales."""
        dataset, _ = _make_known_scaling_dataset(n_per_env=1000, n_envs=4)
        analyzer = TransitionAnalyzer()
        result = analyzer.analyze(dataset, ["s0", "s1"], action_dim=1)

        assert result.reference_scale == pytest.approx(
            float(np.mean(result.dynamics_scales)), rel=1e-6,
        )

    def test_action_coefficients_shape(self) -> None:
        """Action coefficients should have shape (n_envs, state_dim, action_dim)."""
        dataset, _ = _make_known_scaling_dataset(n_per_env=500, n_envs=3)
        analyzer = TransitionAnalyzer()
        result = analyzer.analyze(dataset, ["s0", "s1"], action_dim=1)

        assert result.action_coefficients.shape == (3, 2, 1)

    def test_transition_loeo_passes_invariant_dynamics(self) -> None:
        """Invariant transitions should get high LOEO R^2."""
        dataset = _make_invariant_transition_dataset(n_per_env=2000)
        analyzer = TransitionAnalyzer(loeo_r2_threshold=0.9)
        result = analyzer.analyze(dataset, ["s0", "s1"], action_dim=1)

        # All dims should be invariant (high R^2)
        for name, r2 in result.per_dim_loeo_r2.items():
            assert r2 >= 0.9, f"{name}: LOEO R^2={r2:.4f} < 0.9 (expected invariant)"

        assert len(result.invariant_dims) == 2
        assert len(result.variant_dims) == 0

    def test_transition_loeo_detects_variant_dynamics(self) -> None:
        """Variant transitions should get low LOEO R^2."""
        dataset = _make_variant_transition_dataset(n_per_env=2000)
        analyzer = TransitionAnalyzer(loeo_r2_threshold=0.9)
        result = analyzer.analyze(dataset, ["s0", "s1"], action_dim=1)

        # With very different transition functions, at least one dim should be variant
        assert len(result.variant_dims) >= 1, (
            f"Expected variant dims but got invariant_dims={result.invariant_dims}, "
            f"R^2={result.per_dim_loeo_r2}"
        )

    def test_loeo_r2_threshold_validation(self) -> None:
        """Threshold must be in (0, 1)."""
        with pytest.raises(ValueError, match="loeo_r2_threshold"):
            TransitionAnalyzer(loeo_r2_threshold=0.0)
        with pytest.raises(ValueError, match="loeo_r2_threshold"):
            TransitionAnalyzer(loeo_r2_threshold=1.0)

    def test_dynamics_scales_ordering(self) -> None:
        """Envs with larger action effects should have higher dynamics scales."""
        dataset, alphas = _make_known_scaling_dataset(n_per_env=2000, n_envs=4)
        analyzer = TransitionAnalyzer()
        result = analyzer.analyze(dataset, ["s0", "s1"], action_dim=1)

        # Scales should be monotonically increasing with alpha
        for i in range(len(alphas) - 1):
            assert result.dynamics_scales[i] < result.dynamics_scales[i + 1], (
                f"Expected scale[{i}] < scale[{i+1}], "
                f"got {result.dynamics_scales[i]:.4f} >= {result.dynamics_scales[i+1]:.4f}"
            )
