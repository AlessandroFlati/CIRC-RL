"""Integration test: ablation study produces comparison metrics.

Verifies that the ablation script runs to completion on a minimal
configuration and produces structured results for all variants.
"""

import numpy as np
import pytest

from circ_rl.environments.env_family import EnvironmentFamily
from experiments.ablation import AblationResult, run_ablation, summarize_ablation


@pytest.fixture
def tiny_family() -> EnvironmentFamily:
    """Minimal environment family for ablation testing."""
    return EnvironmentFamily.from_gymnasium(
        base_env="CartPole-v1",
        param_distributions={"gravity": (9.0, 11.0)},
        n_envs=2,
        seed=42,
    )


class TestAblation:
    @pytest.mark.slow
    def test_ablation_produces_results(
        self, tiny_family: EnvironmentFamily
    ) -> None:
        """Ablation study should produce results for all variants."""
        results = run_ablation(
            env_family=tiny_family,
            n_iterations=3,
            n_transitions_per_env=500,
            seed=42,
        )

        # Should have all expected variants
        assert "full_circ_rl" in results
        assert "no_feature_selection" in results
        assert "no_irm" in results
        assert "no_worst_case" in results

        for name, result in results.items():
            assert isinstance(result, AblationResult)
            assert isinstance(result.mean_return, float)
            assert isinstance(result.worst_env_return, float)
            assert isinstance(result.final_loss, float)
            assert result.n_selected_features > 0
            assert len(result.all_metrics) == 3

    @pytest.mark.slow
    def test_summarize_ablation(self, tiny_family: EnvironmentFamily) -> None:
        """Summarize function should produce a formatted table."""
        results = run_ablation(
            env_family=tiny_family,
            n_iterations=2,
            n_transitions_per_env=500,
            seed=42,
        )

        summary = summarize_ablation(results)
        assert isinstance(summary, str)
        assert "full_circ_rl" in summary
        assert "no_irm" in summary
        # Should have header and separator lines
        lines = summary.strip().split("\n")
        assert len(lines) >= 3  # header + separator + at least 1 variant
