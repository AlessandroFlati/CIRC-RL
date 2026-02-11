"""Tests for circ_rl.regularization components."""

import torch
import torch.nn as nn
import pytest

from circ_rl.regularization.reg_parametric import ParametricComplexity
from circ_rl.regularization.reg_functional import FunctionalComplexity
from circ_rl.regularization.reg_path import PathComplexity
from circ_rl.regularization.reg_info_bottleneck import InformationBottleneckLoss
from circ_rl.regularization.reg_composite import CompositeRegularizer


class TestParametricComplexity:
    def test_positive_for_nonzero_params(self) -> None:
        model = nn.Linear(4, 2)
        pc = ParametricComplexity(weight=1.0)
        loss = pc(model)
        assert loss.item() > 0.0

    def test_scales_with_weight(self) -> None:
        model = nn.Linear(4, 2)
        pc1 = ParametricComplexity(weight=1.0)
        pc2 = ParametricComplexity(weight=2.0)
        assert abs(pc2(model).item() - 2.0 * pc1(model).item()) < 1e-5

    def test_larger_model_higher_complexity(self) -> None:
        small = nn.Linear(2, 1)
        large = nn.Sequential(nn.Linear(100, 100), nn.Linear(100, 100))
        pc = ParametricComplexity(weight=1e-4)
        assert pc(large).item() > pc(small).item()


class TestFunctionalComplexity:
    def test_high_entropy_low_complexity(self) -> None:
        fc = FunctionalComplexity(weight=1.0)
        high_entropy = torch.tensor([2.0, 2.0, 2.0])
        low_entropy = torch.tensor([0.1, 0.1, 0.1])
        # High entropy should give lower (more negative) complexity
        assert fc(high_entropy).item() < fc(low_entropy).item()

    def test_returns_negative_mean(self) -> None:
        fc = FunctionalComplexity(weight=1.0)
        entropy = torch.tensor([1.0, 2.0, 3.0])
        expected = -2.0
        assert abs(fc(entropy).item() - expected) < 1e-5


class TestPathComplexity:
    def test_constant_actions_zero_complexity(self) -> None:
        pc = PathComplexity(weight=1.0)
        actions = torch.tensor([[0, 0, 0, 0], [1, 1, 1, 1]])
        loss = pc(actions, discrete=True)
        assert loss.item() == 0.0

    def test_alternating_actions_high_complexity(self) -> None:
        pc = PathComplexity(weight=1.0)
        actions = torch.tensor([[0, 1, 0, 1]])
        loss = pc(actions, discrete=True)
        assert loss.item() > 0.0

    def test_single_step_returns_zero(self) -> None:
        pc = PathComplexity(weight=1.0)
        actions = torch.tensor([[0]])
        loss = pc(actions, discrete=True)
        assert loss.item() == 0.0

    def test_continuous_path_complexity(self) -> None:
        pc = PathComplexity(weight=1.0)
        actions = torch.tensor([[[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]]])
        loss = pc(actions, discrete=False)
        assert loss.item() > 0.0


class TestInformationBottleneckLoss:
    def test_zero_kl_positive_ib(self) -> None:
        ib = InformationBottleneckLoss(beta=1.0)
        kl = torch.zeros(8)
        log_prob = -torch.ones(8)
        loss = ib(kl, log_prob)
        # KL=0, reconstruction = -(-1) = 1
        assert loss.item() > 0.0

    def test_beta_scaling(self) -> None:
        kl = torch.ones(8) * 0.5
        log_prob = -torch.ones(8) * 2.0

        ib1 = InformationBottleneckLoss(beta=0.0)
        ib2 = InformationBottleneckLoss(beta=1.0)
        # With beta=0, only KL term
        assert ib1(kl, log_prob).item() < ib2(kl, log_prob).item()

    def test_rejects_negative_beta(self) -> None:
        with pytest.raises(ValueError, match="beta"):
            InformationBottleneckLoss(beta=-1.0)


class TestCompositeRegularizer:
    def test_returns_breakdown(self) -> None:
        model = nn.Linear(4, 2)
        reg = CompositeRegularizer()
        entropy = torch.ones(8)
        total, breakdown = reg(model, entropy)
        assert total.item() != 0.0
        assert breakdown.parametric > 0.0

    def test_with_all_components(self) -> None:
        model = nn.Linear(4, 2)
        reg = CompositeRegularizer(ib_beta=0.1)
        entropy = torch.ones(8)
        kl = torch.ones(8) * 0.5
        log_prob = -torch.ones(8)
        actions = torch.tensor([[0, 1, 0, 1]] * 8)

        total, breakdown = reg(
            model, entropy, actions=actions,
            kl_divergence=kl, log_prob=log_prob
        )
        assert breakdown.parametric > 0.0
        assert breakdown.path > 0.0
        assert breakdown.info_bottleneck > 0.0
