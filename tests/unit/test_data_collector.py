"""Tests for circ_rl.environments.data_collector."""

import numpy as np
import pytest

from circ_rl.environments.data_collector import DataCollector, ExploratoryDataset
from circ_rl.environments.env_family import EnvironmentFamily


@pytest.fixture
def cartpole_family() -> EnvironmentFamily:
    return EnvironmentFamily.from_gymnasium(
        base_env="CartPole-v1",
        param_distributions={"gravity": (9.0, 11.0)},
        n_envs=2,
        seed=42,
    )


class TestDataCollector:
    def test_collect_returns_correct_shapes(
        self, cartpole_family: EnvironmentFamily
    ) -> None:
        collector = DataCollector(cartpole_family)
        dataset = collector.collect(n_transitions_per_env=50, seed=42)

        assert dataset.states.shape[0] == 100  # 2 envs * 50 transitions
        assert dataset.actions.shape[0] == 100
        assert dataset.next_states.shape[0] == 100
        assert dataset.rewards.shape[0] == 100
        assert dataset.env_ids.shape[0] == 100

    def test_env_ids_are_correct(
        self, cartpole_family: EnvironmentFamily
    ) -> None:
        collector = DataCollector(cartpole_family)
        dataset = collector.collect(n_transitions_per_env=30, seed=42)

        unique_ids = set(dataset.env_ids.tolist())
        assert unique_ids == {0, 1}
        # Each env should have 30 transitions
        assert np.sum(dataset.env_ids == 0) == 30
        assert np.sum(dataset.env_ids == 1) == 30

    def test_n_transitions_property(
        self, cartpole_family: EnvironmentFamily
    ) -> None:
        collector = DataCollector(cartpole_family)
        dataset = collector.collect(n_transitions_per_env=20, seed=42)
        assert dataset.n_transitions == 40
        assert dataset.n_environments == 2

    def test_get_env_data(
        self, cartpole_family: EnvironmentFamily
    ) -> None:
        collector = DataCollector(cartpole_family)
        dataset = collector.collect(n_transitions_per_env=25, seed=42)

        env0_data = dataset.get_env_data(0)
        assert env0_data.n_transitions == 25
        assert all(env0_data.env_ids == 0)

    def test_to_flat_array(
        self, cartpole_family: EnvironmentFamily
    ) -> None:
        collector = DataCollector(cartpole_family)
        dataset = collector.collect(n_transitions_per_env=10, seed=42)
        flat = dataset.to_flat_array()

        # CartPole: 4 state dims + 1 action + 1 reward + 4 next_state = 10
        assert flat.shape[0] == 20
        assert flat.shape[1] == 10

    def test_rejects_zero_transitions(
        self, cartpole_family: EnvironmentFamily
    ) -> None:
        collector = DataCollector(cartpole_family)
        with pytest.raises(ValueError, match="n_transitions_per_env must be >= 1"):
            collector.collect(n_transitions_per_env=0)
