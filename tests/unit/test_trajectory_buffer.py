"""Tests for circ_rl.training.trajectory_buffer."""

import torch
import pytest

from circ_rl.training.trajectory_buffer import Trajectory, MultiEnvTrajectoryBuffer


def _make_trajectory(env_id: int = 0, length: int = 10) -> Trajectory:
    return Trajectory(
        states=torch.randn(length, 4),
        actions=torch.randint(0, 2, (length,)),
        rewards=torch.ones(length),
        log_probs=torch.randn(length),
        values=torch.randn(length),
        next_states=torch.randn(length, 4),
        dones=torch.zeros(length),
        env_id=env_id,
    )


class TestTrajectory:
    def test_length(self) -> None:
        t = _make_trajectory(length=15)
        assert t.length == 15

    def test_compute_returns(self) -> None:
        t = _make_trajectory(length=5)
        returns = t.compute_returns(gamma=0.99)
        assert returns.shape == (5,)
        # With all rewards=1 and no dones, returns should be monotonically increasing
        # from end to start
        for i in range(4):
            assert returns[i].item() >= returns[i + 1].item()

    def test_compute_returns_with_done(self) -> None:
        t = Trajectory(
            states=torch.randn(5, 2),
            actions=torch.zeros(5, dtype=torch.long),
            rewards=torch.ones(5),
            log_probs=torch.zeros(5),
            values=torch.zeros(5),
            next_states=torch.randn(5, 2),
            dones=torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0]),
            env_id=0,
        )
        returns = t.compute_returns(gamma=0.99)
        # Before done: t=0 accumulates 3 steps (t=0,1,2) -> higher return
        # After done: t=3 accumulates 2 steps (t=3,4) -> lower return
        assert returns[3].item() < returns[0].item()

    def test_compute_advantages(self) -> None:
        t = _make_trajectory(length=10)
        adv = t.compute_advantages(gamma=0.99, gae_lambda=0.95)
        assert adv.shape == (10,)


class TestMultiEnvTrajectoryBuffer:
    def test_add_and_count(self) -> None:
        buf = MultiEnvTrajectoryBuffer()
        buf.add(_make_trajectory(env_id=0))
        buf.add(_make_trajectory(env_id=1))
        assert buf.n_trajectories == 2
        assert buf.env_ids == {0, 1}

    def test_get_env_trajectories(self) -> None:
        buf = MultiEnvTrajectoryBuffer()
        buf.add(_make_trajectory(env_id=0, length=10))
        buf.add(_make_trajectory(env_id=1, length=15))
        buf.add(_make_trajectory(env_id=0, length=20))

        env0 = buf.get_env_trajectories(0)
        assert len(env0) == 2
        assert env0[0].length == 10
        assert env0[1].length == 20

    def test_total_transitions(self) -> None:
        buf = MultiEnvTrajectoryBuffer()
        buf.add(_make_trajectory(env_id=0, length=10))
        buf.add(_make_trajectory(env_id=1, length=15))
        assert buf.total_transitions() == 25

    def test_get_all_flat(self) -> None:
        buf = MultiEnvTrajectoryBuffer()
        buf.add(_make_trajectory(env_id=0, length=10))
        buf.add(_make_trajectory(env_id=1, length=15))
        flat = buf.get_all_flat()
        assert flat.length == 25
        assert flat.states.shape == (25, 4)

    def test_get_all_flat_empty_raises(self) -> None:
        buf = MultiEnvTrajectoryBuffer()
        with pytest.raises(ValueError, match="empty"):
            buf.get_all_flat()

    def test_clear(self) -> None:
        buf = MultiEnvTrajectoryBuffer()
        buf.add(_make_trajectory(env_id=0))
        buf.clear()
        assert buf.n_trajectories == 0
