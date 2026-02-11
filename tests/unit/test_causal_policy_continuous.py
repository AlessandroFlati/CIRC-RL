"""Tests for CausalPolicy continuous action space support."""

import numpy as np
import pytest
import torch

from circ_rl.policy.causal_policy import CausalPolicy


def _make_continuous_policy(
    state_dim: int = 3, action_dim: int = 1, **kwargs: object
) -> CausalPolicy:
    mask = np.ones(state_dim, dtype=bool)
    return CausalPolicy(
        full_state_dim=state_dim,
        action_dim=action_dim,
        feature_mask=mask,
        hidden_dims=(32, 32),
        continuous=True,
        action_low=np.full(action_dim, -2.0),
        action_high=np.full(action_dim, 2.0),
        **kwargs,  # type: ignore[arg-type]
    )


class TestCausalPolicyContinuous:
    def test_forward_output_shapes(self) -> None:
        policy = _make_continuous_policy(state_dim=3, action_dim=1)
        state = torch.randn(8, 3)
        output = policy(state)
        assert output.action.shape == (8, 1)
        assert output.log_prob.shape == (8,)
        assert output.entropy.shape == (8,)
        assert output.value.shape == (8,)
        assert output.kl_divergence.shape == (8,)

    def test_multidim_action(self) -> None:
        policy = _make_continuous_policy(state_dim=4, action_dim=3)
        state = torch.randn(8, 4)
        output = policy(state)
        assert output.action.shape == (8, 3)
        assert output.log_prob.shape == (8,)

    def test_action_within_bounds(self) -> None:
        policy = _make_continuous_policy()
        state = torch.randn(200, 3)
        output = policy(state)
        assert (output.action >= -2.0).all()
        assert (output.action <= 2.0).all()

    def test_evaluate_actions_log_prob_shape(self) -> None:
        policy = _make_continuous_policy()
        state = torch.randn(8, 3)
        output = policy(state)
        eval_output = policy.evaluate_actions(state, output.action.detach())
        assert eval_output.log_prob.shape == (8,)
        assert eval_output.value.shape == (8,)

    def test_get_action_returns_ndarray(self) -> None:
        policy = _make_continuous_policy()
        state = torch.randn(3)
        action = policy.get_action(state)
        assert isinstance(action, np.ndarray)
        assert action.shape == (1,)
        assert -2.0 <= action[0] <= 2.0

    def test_get_action_deterministic_reproducible(self) -> None:
        policy = _make_continuous_policy()
        state = torch.randn(3)
        a1 = policy.get_action(state, deterministic=True)
        a2 = policy.get_action(state, deterministic=True)
        np.testing.assert_array_equal(a1, a2)

    def test_gradient_flows(self) -> None:
        policy = _make_continuous_policy()
        state = torch.randn(4, 3)
        output = policy(state)
        loss = output.log_prob.sum() + output.value.sum()
        loss.backward()
        for name, param in policy.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_continuous_property_true(self) -> None:
        policy = _make_continuous_policy()
        assert policy.continuous is True

    def test_discrete_property_false(self) -> None:
        mask = np.ones(4, dtype=bool)
        policy = CausalPolicy(full_state_dim=4, action_dim=2, feature_mask=mask)
        assert policy.continuous is False

    def test_raises_without_bounds(self) -> None:
        mask = np.ones(3, dtype=bool)
        with pytest.raises(ValueError, match="action_low and action_high"):
            CausalPolicy(
                full_state_dim=3,
                action_dim=1,
                feature_mask=mask,
                continuous=True,
            )

    def test_raises_wrong_bounds_shape(self) -> None:
        mask = np.ones(3, dtype=bool)
        with pytest.raises(ValueError, match="action bounds must have shape"):
            CausalPolicy(
                full_state_dim=3,
                action_dim=2,
                feature_mask=mask,
                continuous=True,
                action_low=np.array([-1.0]),  # wrong shape
                action_high=np.array([1.0]),
            )

    def test_with_info_bottleneck(self) -> None:
        policy = _make_continuous_policy(
            use_info_bottleneck=True, latent_dim=8
        )
        state = torch.randn(4, 3)
        output = policy(state)
        assert output.action.shape == (4, 1)
        assert output.kl_divergence.shape == (4,)
        # KL should be non-zero with IB encoder
        assert output.kl_divergence.sum().item() != 0.0
