"""Tests for circ_rl.policy encoder and causal_policy."""

import numpy as np
import pytest
import torch

from circ_rl.policy.encoder import InformationBottleneckEncoder
from circ_rl.policy.causal_policy import CausalPolicy


class TestInformationBottleneckEncoder:
    def test_output_shapes(self) -> None:
        enc = InformationBottleneckEncoder(input_dim=4, latent_dim=8)
        x = torch.randn(16, 4)
        z, mu, logvar = enc(x)
        assert z.shape == (16, 8)
        assert mu.shape == (16, 8)
        assert logvar.shape == (16, 8)

    def test_deterministic_encoding(self) -> None:
        enc = InformationBottleneckEncoder(input_dim=4, latent_dim=8)
        x = torch.randn(16, 4)
        z_det = enc.encode_deterministic(x)
        assert z_det.shape == (16, 8)

    def test_kl_divergence_nonnegative(self) -> None:
        mu = torch.randn(32, 8)
        logvar = torch.randn(32, 8)
        kl = InformationBottleneckEncoder.kl_divergence(mu, logvar)
        assert kl.shape == (32,)
        # KL is nonneg for each sample (up to numerical error)
        assert (kl > -0.01).all()

    def test_kl_zero_at_prior(self) -> None:
        mu = torch.zeros(16, 8)
        logvar = torch.zeros(16, 8)
        kl = InformationBottleneckEncoder.kl_divergence(mu, logvar)
        assert torch.allclose(kl, torch.zeros(16), atol=1e-5)

    def test_latent_dim_property(self) -> None:
        enc = InformationBottleneckEncoder(input_dim=4, latent_dim=16)
        assert enc.latent_dim == 16


class TestCausalPolicy:
    def test_forward_output_shapes(self) -> None:
        mask = np.array([True, True, False, True])
        policy = CausalPolicy(full_state_dim=4, action_dim=2, feature_mask=mask)
        state = torch.randn(8, 4)
        output = policy(state)
        assert output.action.shape == (8,)
        assert output.log_prob.shape == (8,)
        assert output.entropy.shape == (8,)
        assert output.value.shape == (8,)
        assert output.kl_divergence.shape == (8,)

    def test_evaluate_actions(self) -> None:
        mask = np.array([True, True])
        policy = CausalPolicy(full_state_dim=2, action_dim=3, feature_mask=mask)
        state = torch.randn(8, 2)
        action = torch.randint(0, 3, (8,))
        output = policy.evaluate_actions(state, action)
        assert output.log_prob.shape == (8,)
        assert (output.action == action).all()

    def test_get_action_returns_int(self) -> None:
        mask = np.array([True, True])
        policy = CausalPolicy(full_state_dim=2, action_dim=3, feature_mask=mask)
        state = torch.randn(2)
        action = policy.get_action(state)
        assert isinstance(action, int)
        assert 0 <= action < 3

    def test_get_action_deterministic(self) -> None:
        mask = np.array([True, True])
        policy = CausalPolicy(full_state_dim=2, action_dim=3, feature_mask=mask)
        state = torch.randn(2)
        a1 = policy.get_action(state, deterministic=True)
        a2 = policy.get_action(state, deterministic=True)
        assert a1 == a2

    def test_with_info_bottleneck(self) -> None:
        mask = np.array([True, True, True])
        policy = CausalPolicy(
            full_state_dim=3,
            action_dim=2,
            feature_mask=mask,
            use_info_bottleneck=True,
            latent_dim=8,
        )
        state = torch.randn(4, 3)
        output = policy(state)
        assert output.action.shape == (4,)
        # KL should be > 0 (since encoder produces non-trivial posterior)
        assert output.kl_divergence.shape == (4,)

    def test_gradient_flows_through_policy(self) -> None:
        mask = np.array([True, True])
        policy = CausalPolicy(full_state_dim=2, action_dim=2, feature_mask=mask)
        state = torch.randn(4, 2)
        output = policy(state)
        loss = output.log_prob.sum() + output.value.sum()
        loss.backward()
        for name, param in policy.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"


class TestDynamicsNormalization:
    """Tests for dynamics-normalized policy (continuous + context)."""

    def _make_policy(self) -> CausalPolicy:
        """Create a continuous policy with dynamics normalization."""
        return CausalPolicy(
            full_state_dim=3,
            action_dim=1,
            feature_mask=np.ones(3, dtype=bool),
            hidden_dims=(32, 32),
            continuous=True,
            action_low=np.array([-2.0]),
            action_high=np.array([2.0]),
            context_dim=2,
            use_dynamics_normalization=True,
            dynamics_reference_scale=1.0,
        )

    def test_dynamics_norm_requires_context(self) -> None:
        """Dynamics normalization requires context_dim > 0."""
        with pytest.raises(ValueError, match="context_dim"):
            CausalPolicy(
                full_state_dim=2,
                action_dim=1,
                feature_mask=np.ones(2, dtype=bool),
                continuous=True,
                action_low=np.array([-1.0]),
                action_high=np.array([1.0]),
                context_dim=0,
                use_dynamics_normalization=True,
            )

    def test_dynamics_norm_trunk_includes_context(self) -> None:
        """When dynamics norm is active, trunk should still include context."""
        policy = self._make_policy()
        # Trunk input should be state_dim + context_dim = 3 + 2 = 5
        first_layer = policy._trunk[0]
        assert isinstance(first_layer, torch.nn.Linear)
        assert first_layer.in_features == 5, (
            f"Trunk input dim should be 5 (state + context), got {first_layer.in_features}"
        )

    def test_dynamics_norm_disabled_includes_context(self) -> None:
        """Without dynamics norm, trunk should include context."""
        policy = CausalPolicy(
            full_state_dim=3,
            action_dim=1,
            feature_mask=np.ones(3, dtype=bool),
            hidden_dims=(32, 32),
            continuous=True,
            action_low=np.array([-2.0]),
            action_high=np.array([2.0]),
            context_dim=2,
            use_dynamics_normalization=False,
        )
        first_layer = policy._trunk[0]
        assert isinstance(first_layer, torch.nn.Linear)
        assert first_layer.in_features == 5, (
            f"Trunk input dim should be 5 (state + context), got {first_layer.in_features}"
        )

    def test_dynamics_norm_forward_shapes(self) -> None:
        """Forward pass should produce correct output shapes."""
        policy = self._make_policy()
        state = torch.randn(4, 3)
        context = torch.randn(4, 2)
        output = policy(state, context=context)
        assert output.action.shape == (4, 1)
        assert output.log_prob.shape == (4,)
        assert output.entropy.shape == (4,)
        assert output.value.shape == (4,)

    def test_dynamics_predictor_produces_different_scales(self) -> None:
        """Dynamics predictor should produce different scales for different contexts."""
        policy = self._make_policy()
        policy.eval()

        # Two very different contexts
        ctx_low = torch.tensor([[0.1, 0.1]])
        ctx_high = torch.tensor([[10.0, 10.0]])

        with torch.no_grad():
            scale_low = policy._dynamics_predictor(ctx_low)
            scale_high = policy._dynamics_predictor(ctx_high)

        # Scales should differ for different contexts
        assert not torch.allclose(scale_low, scale_high, atol=1e-3), (
            f"Dynamics predictor scales should differ: "
            f"low={scale_low.item():.4f}, high={scale_high.item():.4f}"
        )

    def test_dynamics_predictor_in_policy_params(self) -> None:
        """Dynamics predictor params should be in policy_parameters."""
        policy = self._make_policy()
        policy_params = policy.policy_parameters
        predictor_params = list(policy._dynamics_predictor.parameters())
        for p in predictor_params:
            assert any(
                pp is p for pp in policy_params
            ), "Dynamics predictor param not in policy_parameters"

    def test_dynamics_norm_evaluate_actions_consistent(self) -> None:
        """evaluate_actions should produce valid log_probs."""
        policy = self._make_policy()
        state = torch.randn(8, 3)
        context = torch.randn(8, 2)

        # Forward to get actions
        output1 = policy(state, context=context)
        action = output1.action

        # Evaluate the same actions
        output2 = policy.evaluate_actions(state, action, context=context)
        assert output2.log_prob.shape == (8,)
        assert torch.isfinite(output2.log_prob).all()

    def test_dynamics_norm_get_action_inference(self) -> None:
        """get_action should work with dynamics normalization."""
        policy = self._make_policy()
        policy.eval()
        state = torch.randn(3)
        context = torch.tensor([1.0, 1.0])
        action = policy.get_action(state, deterministic=True, context=context)
        assert isinstance(action, np.ndarray)
        assert action.shape == (1,)

    def test_dynamics_predictor_auxiliary_gradient_flows(self) -> None:
        """Gradients should flow through dynamics predictor via auxiliary loss."""
        policy = self._make_policy()
        context = torch.randn(4, 2)
        target_scales = torch.tensor([0.5, 1.0, 1.5, 2.0])

        # Simulate the auxiliary dynamics loss from the trainer
        predicted_scale = policy._dynamics_predictor(context).squeeze(-1)
        aux_loss = torch.nn.functional.mse_loss(predicted_scale, target_scales)
        aux_loss.backward()

        # Check dynamics predictor has gradients
        for name, param in policy._dynamics_predictor.named_parameters():
            assert param.grad is not None, (
                f"No gradient for dynamics_predictor.{name}"
            )

    def test_dynamics_norm_backward_compat(self) -> None:
        """Without dynamics norm, behavior should be unchanged."""
        mask = np.ones(3, dtype=bool)
        policy = CausalPolicy(
            full_state_dim=3,
            action_dim=1,
            feature_mask=mask,
            hidden_dims=(32, 32),
            continuous=True,
            action_low=np.array([-2.0]),
            action_high=np.array([2.0]),
            context_dim=2,
            use_dynamics_normalization=False,
        )
        state = torch.randn(4, 3)
        context = torch.randn(4, 2)
        output = policy(state, context=context)
        assert output.action.shape == (4, 1)
        assert torch.isfinite(output.log_prob).all()

    def test_dynamics_norm_discrete_ignored(self) -> None:
        """Dynamics normalization should be silently disabled for discrete actions."""
        policy = CausalPolicy(
            full_state_dim=3,
            action_dim=4,
            feature_mask=np.ones(3, dtype=bool),
            hidden_dims=(32, 32),
            continuous=False,
            context_dim=2,
            use_dynamics_normalization=True,
        )
        # Should silently disable (continuous=False)
        assert not policy._use_dynamics_norm
        state = torch.randn(4, 3)
        context = torch.randn(4, 2)
        output = policy(state, context=context)
        assert output.action.shape == (4,)
