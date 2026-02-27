"""Tests for circ_rl.utils.checkpointing."""

import pytest
import torch

from circ_rl.utils.checkpointing import CheckpointManager


@pytest.fixture
def ckpt_dir(tmp_path):
    return str(tmp_path / "checkpoints")


class TestCheckpointManager:
    def test_save_and_load_latest(self, ckpt_dir: str) -> None:
        mgr = CheckpointManager(ckpt_dir, max_to_keep=3)
        state = {"model": torch.tensor([1.0, 2.0, 3.0]), "lr": 0.001}
        mgr.save(state, step=10)

        loaded, step = mgr.load_latest()
        assert step == 10
        assert torch.equal(loaded["model"], state["model"])
        assert loaded["lr"] == 0.001

    def test_load_latest_returns_most_recent(self, ckpt_dir: str) -> None:
        mgr = CheckpointManager(ckpt_dir, max_to_keep=5)
        for i in range(3):
            mgr.save({"value": i}, step=i * 10)

        _, step = mgr.load_latest()
        assert step == 20

    def test_save_best_and_load_best(self, ckpt_dir: str) -> None:
        mgr = CheckpointManager(ckpt_dir, max_to_keep=3)
        mgr.save({"value": "not_best"}, step=10)
        mgr.save({"value": "the_best"}, step=20, is_best=True)
        mgr.save({"value": "after_best"}, step=30)

        loaded, step = mgr.load_best()
        assert step == 20
        assert loaded["value"] == "the_best"

    def test_max_to_keep_prunes_old_checkpoints(self, ckpt_dir: str) -> None:
        mgr = CheckpointManager(ckpt_dir, max_to_keep=2)
        paths = []
        for i in range(4):
            p = mgr.save({"value": i}, step=i)
            paths.append(p)

        assert not paths[0].exists()
        assert not paths[1].exists()
        assert paths[2].exists()
        assert paths[3].exists()

    def test_load_latest_raises_when_empty(self, ckpt_dir: str) -> None:
        mgr = CheckpointManager(ckpt_dir, max_to_keep=3)
        with pytest.raises(FileNotFoundError, match="No checkpoints found"):
            mgr.load_latest()

    def test_load_best_raises_when_no_best(self, ckpt_dir: str) -> None:
        mgr = CheckpointManager(ckpt_dir, max_to_keep=3)
        mgr.save({"value": 1}, step=1)
        with pytest.raises(FileNotFoundError, match="No best checkpoint"):
            mgr.load_best()

    def test_emergency_save(self, ckpt_dir: str) -> None:
        mgr = CheckpointManager(ckpt_dir, max_to_keep=3)
        state = {"model": torch.tensor([42.0])}
        path = mgr.emergency_save(state, step=99)

        assert path.exists()
        assert "emergency" in path.name

        loaded = torch.load(path, weights_only=False)
        assert loaded["step"] == 99
        assert torch.equal(loaded["model"], state["model"])

    def test_max_to_keep_rejects_zero(self) -> None:
        with pytest.raises(ValueError, match="max_to_keep must be >= 1"):
            CheckpointManager("/tmp/test", max_to_keep=0)
