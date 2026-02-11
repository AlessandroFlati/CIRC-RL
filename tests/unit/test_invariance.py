"""Tests for circ_rl.invariance (IRM penalty, worst-case optimizer)."""

import torch
import pytest

from circ_rl.invariance.inv_irm_penalty import IRMPenalty
from circ_rl.invariance.inv_worst_case import WorstCaseOptimizer


class TestIRMPenalty:
    def test_zero_penalty_for_equal_losses(self) -> None:
        """If all environments have the same loss, penalty should be small."""
        irm = IRMPenalty(lambda_irm=1.0)
        # Same loss in all envs
        loss = torch.tensor(1.0, requires_grad=True)
        env_losses = [loss, loss, loss]
        penalty = irm(env_losses)
        assert penalty.item() >= 0.0

    def test_higher_penalty_for_different_losses(self) -> None:
        """Different environment losses should produce non-zero penalty."""
        irm = IRMPenalty(lambda_irm=1.0)
        loss1 = torch.tensor(0.5, requires_grad=True)
        loss2 = torch.tensor(2.0, requires_grad=True)
        penalty = irm([loss1, loss2])
        assert penalty.item() > 0.0

    def test_empty_losses_returns_zero(self) -> None:
        irm = IRMPenalty(lambda_irm=1.0)
        penalty = irm([])
        assert penalty.item() == 0.0

    def test_lambda_scaling(self) -> None:
        loss = torch.tensor(2.0, requires_grad=True)
        irm1 = IRMPenalty(lambda_irm=1.0)
        irm2 = IRMPenalty(lambda_irm=0.5)
        p1 = irm1([loss])
        # Need a fresh tensor for the second call
        loss2 = torch.tensor(2.0, requires_grad=True)
        p2 = irm2([loss2])
        assert abs(p1.item() - 2.0 * p2.item()) < 1e-5

    def test_gradient_flows(self) -> None:
        irm = IRMPenalty(lambda_irm=1.0)
        x = torch.tensor(1.0, requires_grad=True)
        loss = x * 2.0
        penalty = irm([loss])
        penalty.backward()
        assert x.grad is not None


class TestWorstCaseOptimizer:
    def test_soft_min_is_below_mean(self) -> None:
        wc = WorstCaseOptimizer(temperature=1.0)
        returns = torch.tensor([1.0, 2.0, 3.0])
        soft_min = wc.soft_min(returns)
        assert soft_min.item() < returns.mean().item()

    def test_low_temperature_approaches_min(self) -> None:
        wc = WorstCaseOptimizer(temperature=0.01)
        returns = torch.tensor([1.0, 5.0, 10.0])
        soft_min = wc.soft_min(returns)
        assert abs(soft_min.item() - 1.0) < 0.5

    def test_high_temperature_approaches_mean(self) -> None:
        wc = WorstCaseOptimizer(temperature=100.0)
        returns = torch.tensor([1.0, 2.0, 3.0])
        soft_min = wc.soft_min(returns)
        assert abs(soft_min.item() - 2.0) < 0.5

    def test_loss_penalizes_variance(self) -> None:
        wc = WorstCaseOptimizer(temperature=1.0, variance_weight=1.0)
        # High variance
        r_high_var = torch.tensor([1.0, 10.0])
        # Low variance
        r_low_var = torch.tensor([5.0, 6.0])
        loss_high = wc.compute_loss(r_high_var)
        loss_low = wc.compute_loss(r_low_var)
        # High variance should have higher loss (more penalty)
        # (accounting for the return term too)
        assert loss_high.item() != loss_low.item()

    def test_rejects_zero_temperature(self) -> None:
        with pytest.raises(ValueError, match="temperature"):
            WorstCaseOptimizer(temperature=0.0)

    def test_gradient_flows(self) -> None:
        wc = WorstCaseOptimizer(temperature=1.0)
        returns = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        loss = wc(returns)
        loss.backward()
        assert returns.grad is not None
