"""Tests for circ_rl.policy.causal_q_network, encoder, and causal_policy."""

import numpy as np
import pytest
import torch

from circ_rl.policy.causal_q_network import CausalQNetwork
from circ_rl.policy.encoder import InformationBottleneckEncoder
from circ_rl.policy.causal_policy import CausalPolicy


class TestCausalQNetwork:
    def test_output_shape(self) -> None:
        mask = np.array([True, False, True, True])
        net = CausalQNetwork(full_state_dim=4, action_dim=2, feature_mask=mask)
        state = torch.randn(8, 4)
        q_values = net(state)
        assert q_values.shape == (8, 2)

    def test_causal_dim(self) -> None:
        mask = np.array([True, False, True, True])
        net = CausalQNetwork(full_state_dim=4, action_dim=2, feature_mask=mask)
        assert net.causal_dim == 3

    def test_q_value_for_action(self) -> None:
        mask = np.array([True, True])
        net = CausalQNetwork(full_state_dim=2, action_dim=3, feature_mask=mask)
        state = torch.randn(4, 2)
        action = torch.tensor([0, 1, 2, 0])
        q = net.q_value(state, action)
        assert q.shape == (4,)

    def test_rejects_empty_mask(self) -> None:
        mask = np.array([False, False, False])
        with pytest.raises(ValueError, match="zero features"):
            CausalQNetwork(full_state_dim=3, action_dim=2, feature_mask=mask)

    def test_rejects_wrong_mask_shape(self) -> None:
        mask = np.array([True, True])
        with pytest.raises(ValueError, match="does not match"):
            CausalQNetwork(full_state_dim=4, action_dim=2, feature_mask=mask)

    def test_gradient_flows(self) -> None:
        mask = np.array([True, True])
        net = CausalQNetwork(full_state_dim=2, action_dim=2, feature_mask=mask)
        state = torch.randn(4, 2)
        q = net(state)
        loss = q.sum()
        loss.backward()
        for param in net.parameters():
            assert param.grad is not None


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
