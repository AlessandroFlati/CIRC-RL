"""Tests for circ_rl.utils.seeding."""

import random

import numpy as np
import pytest
import torch

from circ_rl.utils.seeding import seed_everything


def test_seed_everything_produces_deterministic_python_random() -> None:
    seed_everything(123)
    values_a = [random.random() for _ in range(10)]

    seed_everything(123)
    values_b = [random.random() for _ in range(10)]

    assert values_a == values_b


def test_seed_everything_produces_deterministic_numpy() -> None:
    seed_everything(456)
    values_a = np.random.rand(10).tolist()

    seed_everything(456)
    values_b = np.random.rand(10).tolist()

    assert values_a == values_b


def test_seed_everything_produces_deterministic_torch() -> None:
    seed_everything(789)
    values_a = torch.randn(10).tolist()

    seed_everything(789)
    values_b = torch.randn(10).tolist()

    assert values_a == values_b


def test_seed_everything_different_seeds_produce_different_values() -> None:
    seed_everything(1)
    values_a = torch.randn(10).tolist()

    seed_everything(2)
    values_b = torch.randn(10).tolist()

    assert values_a != values_b


def test_seed_everything_rejects_negative_seed() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        seed_everything(-1)
