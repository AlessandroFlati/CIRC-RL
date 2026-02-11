"""Reproducibility utilities: seeding and git hash tracking."""

import random
import subprocess

import numpy as np
import torch
from loguru import logger


def seed_everything(seed: int) -> None:
    """Set deterministic seeds for Python, NumPy, PyTorch, and CUDA.

    :param seed: The random seed to use across all libraries.
    :raises ValueError: If seed is negative.
    """
    if seed < 0:
        raise ValueError(f"Seed must be non-negative, got {seed}")

    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    logger.info("All random seeds set to {}", seed)


def get_git_hash() -> str:
    """Return the current short git commit hash.

    :returns: The 7-character short hash of the current HEAD commit.
    :raises RuntimeError: If git is not available or the directory is not
        a git repository.
    """
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],  # noqa: S603, S607
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()
