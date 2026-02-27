"""Tests for circ_rl.utils.logging."""

from pathlib import Path

import torch

from circ_rl.utils.logging import MetricsLogger


class TestMetricsLogger:
    def test_init_creates_log_dir(self, tmp_path: Path) -> None:
        log_dir = str(tmp_path / "logs" / "nested")
        logger = MetricsLogger(log_dir, "test_experiment")
        assert Path(log_dir).exists()
        logger.close()

    def test_log_scalar_writes_event_file(self, tmp_path: Path) -> None:
        log_dir = str(tmp_path / "logs")
        logger = MetricsLogger(log_dir, "test_experiment")
        logger.log_scalar("test/loss", 0.5, step=1)
        logger.flush()
        logger.close()

        event_files = list(Path(log_dir).glob("events.out.tfevents.*"))
        assert len(event_files) >= 1

    def test_log_histogram(self, tmp_path: Path) -> None:
        log_dir = str(tmp_path / "logs")
        logger = MetricsLogger(log_dir, "test_experiment")
        values = torch.randn(100)
        logger.log_histogram("test/weights", values, step=1)
        logger.flush()
        logger.close()

        event_files = list(Path(log_dir).glob("events.out.tfevents.*"))
        assert len(event_files) >= 1

    def test_log_scalars_group(self, tmp_path: Path) -> None:
        log_dir = str(tmp_path / "logs")
        logger = MetricsLogger(log_dir, "test_experiment")
        logger.log_scalars(
            "losses",
            {"irm": 0.1, "complexity": 0.2, "constraint": 0.3},
            step=1,
        )
        logger.flush()
        logger.close()
