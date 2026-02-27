"""Checkpoint management: save, restore, and emergency-save training state."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import torch
from loguru import logger


class CheckpointManager:
    """Manage training checkpoints with automatic pruning.

    Saves and restores full training state (model weights, optimizer state,
    Lagrange multipliers, training step, etc.). Supports emergency saves
    triggered by signal handlers before a crash.

    :param checkpoint_dir: Directory to store checkpoint files.
    :param max_to_keep: Maximum number of periodic checkpoints to retain.
        Older checkpoints are pruned automatically. The "best" checkpoint
        is never pruned.
    """

    def __init__(self, checkpoint_dir: str, max_to_keep: int = 5) -> None:
        if max_to_keep < 1:
            raise ValueError(f"max_to_keep must be >= 1, got {max_to_keep}")

        self._dir = Path(checkpoint_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._max_to_keep = max_to_keep
        self._periodic_checkpoints: list[Path] = []

    def save(
        self,
        state: dict[str, Any],
        step: int,
        *,
        is_best: bool = False,
    ) -> Path:
        """Save a checkpoint.

        :param state: Dictionary containing all state to persist (model,
            optimizer, multipliers, step, etc.).
        :param step: Current training step number.
        :param is_best: If True, also save as ``best.pt``.
        :returns: Path to the saved checkpoint file.
        """
        state["step"] = step
        path = self._dir / f"checkpoint_{step:08d}.pt"
        torch.save(state, path)
        logger.info("Checkpoint saved: {} (step {})", path.name, step)

        self._periodic_checkpoints.append(path)
        self._prune()

        if is_best:
            best_path = self._dir / "best.pt"
            shutil.copy2(path, best_path)
            logger.info("Best checkpoint updated at step {}", step)

        return path

    def load_latest(self) -> tuple[dict[str, Any], int]:
        """Load the most recent periodic checkpoint.

        :returns: Tuple of (state_dict, step).
        :raises FileNotFoundError: If no checkpoints exist.
        """
        checkpoints = sorted(self._dir.glob("checkpoint_*.pt"))
        if not checkpoints:
            raise FileNotFoundError(
                f"No checkpoints found in {self._dir}"
            )

        path = checkpoints[-1]
        state = torch.load(path, weights_only=False)
        step: int = state.pop("step")
        logger.info("Loaded latest checkpoint: {} (step {})", path.name, step)
        return state, step

    def load_best(self) -> tuple[dict[str, Any], int]:
        """Load the best checkpoint.

        :returns: Tuple of (state_dict, step).
        :raises FileNotFoundError: If no best checkpoint exists.
        """
        path = self._dir / "best.pt"
        if not path.exists():
            raise FileNotFoundError(f"No best checkpoint found at {path}")

        state = torch.load(path, weights_only=False)
        step: int = state.pop("step")
        logger.info("Loaded best checkpoint (step {})", step)
        return state, step

    def emergency_save(self, state: dict[str, Any], step: int) -> Path:
        """Save an emergency checkpoint synchronously before a crash.

        This method writes synchronously and does not prune old checkpoints.

        :param state: Dictionary containing all state to persist.
        :param step: Current training step number.
        :returns: Path to the emergency checkpoint file.
        """
        state["step"] = step
        path = self._dir / f"emergency_{step:08d}.pt"
        torch.save(state, path)
        logger.warning("Emergency checkpoint saved: {} (step {})", path.name, step)
        return path

    def _prune(self) -> None:
        """Remove oldest periodic checkpoints beyond max_to_keep."""
        while len(self._periodic_checkpoints) > self._max_to_keep:
            old_path = self._periodic_checkpoints.pop(0)
            if old_path.exists():
                old_path.unlink()
                logger.debug("Pruned old checkpoint: {}", old_path.name)
