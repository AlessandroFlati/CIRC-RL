"""Unified logging: Loguru for application logs, TensorBoard for metrics."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger
from torch.utils.tensorboard import SummaryWriter

from circ_rl.utils.seeding import get_git_hash

if TYPE_CHECKING:
    import torch
    from omegaconf import DictConfig


class MetricsLogger:
    """Unified logging facade combining Loguru and TensorBoard.

    :param log_dir: Directory for TensorBoard event files.
    :param experiment_name: Human-readable experiment identifier.
    """

    def __init__(self, log_dir: str, experiment_name: str) -> None:
        self._log_dir = Path(log_dir)
        self._experiment_name = experiment_name
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._writer = SummaryWriter(log_dir=str(self._log_dir))
        logger.info(
            "MetricsLogger initialized: experiment='{}', log_dir='{}'",
            experiment_name,
            log_dir,
        )

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log a scalar value to TensorBoard.

        :param tag: Metric name (e.g., ``"train/loss"``).
        :param value: Scalar value.
        :param step: Global step number.
        """
        self._writer.add_scalar(tag, value, step)

    def log_scalars(
        self, main_tag: str, tag_scalar_dict: dict[str, float], step: int
    ) -> None:
        """Log multiple scalars under a common group.

        :param main_tag: Group name (e.g., ``"losses"``).
        :param tag_scalar_dict: Mapping of sub-tag to value.
        :param step: Global step number.
        """
        self._writer.add_scalars(main_tag, tag_scalar_dict, step)

    def log_histogram(self, tag: str, values: torch.Tensor, step: int) -> None:
        """Log a tensor as a histogram to TensorBoard.

        :param tag: Histogram name.
        :param values: Tensor of values.
        :param step: Global step number.
        """
        self._writer.add_histogram(tag, values, step)

    def log_config(self, config: DictConfig) -> None:
        """Log the full resolved Hydra configuration as text.

        :param config: The resolved OmegaConf DictConfig.
        """
        from omegaconf import OmegaConf

        config_str = OmegaConf.to_yaml(config, resolve=True)
        self._writer.add_text("config", f"```yaml\n{config_str}\n```")
        config_path = self._log_dir / "config.yaml"
        config_path.write_text(config_str)
        logger.info("Configuration logged to {}", config_path)

    def log_git_hash(self) -> None:
        """Log the current git commit hash."""
        git_hash = get_git_hash()
        self._writer.add_text("git_hash", git_hash)
        logger.info("Git hash: {}", git_hash)

    def flush(self) -> None:
        """Flush TensorBoard writer."""
        self._writer.flush()

    def close(self) -> None:
        """Close TensorBoard writer."""
        self._writer.close()
