"""Tests for circ_rl.environments.env_family."""

import gymnasium as gym
import pytest

from circ_rl.environments.env_family import EnvironmentFamily


class TestEnvironmentFamily:
    def test_from_gymnasium_creates_family(self) -> None:
        family = EnvironmentFamily.from_gymnasium(
            base_env="CartPole-v1",
            param_distributions={"gravity": (9.0, 11.0)},
            n_envs=3,
            seed=42,
        )
        assert family.n_envs == 3
        assert "gravity" in family.param_names

    def test_make_env_returns_valid_env(self) -> None:
        family = EnvironmentFamily.from_gymnasium(
            base_env="CartPole-v1",
            param_distributions={"gravity": (9.0, 11.0)},
            n_envs=2,
            seed=42,
        )
        env = family.make_env(0)
        obs, _ = env.reset()
        assert obs is not None
        env.close()

    def test_env_params_are_varied(self) -> None:
        family = EnvironmentFamily.from_gymnasium(
            base_env="CartPole-v1",
            param_distributions={"gravity": (5.0, 15.0)},
            n_envs=5,
            seed=42,
        )
        gravities = [family.get_env_params(i)["gravity"] for i in range(5)]
        # Parameters should be different (with high probability)
        assert len(set(gravities)) > 1
        # All within range
        for g in gravities:
            assert 5.0 <= g <= 15.0

    def test_observation_space_consistent(self) -> None:
        family = EnvironmentFamily.from_gymnasium(
            base_env="CartPole-v1",
            param_distributions={"gravity": (9.0, 11.0)},
            n_envs=2,
        )
        assert family.observation_space is not None
        assert family.action_space is not None

    def test_out_of_range_env_idx_raises(self) -> None:
        family = EnvironmentFamily.from_gymnasium(
            base_env="CartPole-v1",
            param_distributions={"gravity": (9.0, 11.0)},
            n_envs=2,
        )
        with pytest.raises(IndexError):
            family.make_env(5)

    def test_sample_envs(self) -> None:
        family = EnvironmentFamily.from_gymnasium(
            base_env="CartPole-v1",
            param_distributions={"gravity": (9.0, 11.0)},
            n_envs=3,
        )
        samples = family.sample_envs(2)
        assert len(samples) == 2
        for idx, env in samples:
            assert 0 <= idx < 3
            env.close()

    def test_n_envs_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="n_envs must be >= 1"):
            EnvironmentFamily.from_gymnasium(
                base_env="CartPole-v1",
                param_distributions={"gravity": (9.0, 11.0)},
                n_envs=0,
            )

    def test_cartpole_derived_attrs_recomputed(self) -> None:
        """Verify total_mass and polemass_length are recomputed after setattr."""
        family = EnvironmentFamily.from_gymnasium(
            base_env="CartPole-v1",
            param_distributions={
                "masscart": (0.5, 2.0),
                "masspole": (0.05, 0.2),
                "length": (0.3, 0.8),
            },
            n_envs=3,
            seed=42,
        )
        for i in range(3):
            env = family.make_env(i)
            u = env.unwrapped
            params = family.get_env_params(i)
            expected_total = params["masscart"] + params["masspole"]
            expected_pml = params["masspole"] * params["length"]
            assert abs(u.total_mass - expected_total) < 1e-10, (
                f"Env {i}: total_mass={u.total_mass}, expected {expected_total}"
            )
            assert abs(u.polemass_length - expected_pml) < 1e-10, (
                f"Env {i}: polemass_length={u.polemass_length}, expected {expected_pml}"
            )
            env.close()

    def test_repr(self) -> None:
        family = EnvironmentFamily.from_gymnasium(
            base_env="CartPole-v1",
            param_distributions={"gravity": (9.0, 11.0)},
            n_envs=3,
        )
        r = repr(family)
        assert "n_envs=3" in r
        assert "gravity" in r
