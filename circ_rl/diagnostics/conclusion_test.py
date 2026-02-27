"""Conclusion test: verify observed returns match predictions.

Tests whether the return obtained in test environments is compatible
with the return predicted by the theory.

See ``CIRC-RL_Framework.md`` Section 3.9.3.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from circ_rl.analytic_policy.analytic_policy import AnalyticPolicy
    from circ_rl.environments.env_family import EnvironmentFamily


@dataclass(frozen=True)
class ConclusionTestResult:
    """Result of the conclusion test.

    :param passed: Whether observed returns match predicted returns.
    :param per_env_observed: Observed mean return per test environment.
    :param per_env_predicted: Predicted mean return per test environment.
    :param per_env_relative_error: Relative error per test environment.
    :param mean_relative_error: Mean relative error across test envs.
    """

    passed: bool
    per_env_observed: dict[int, float]
    per_env_predicted: dict[int, float]
    per_env_relative_error: dict[int, float]
    mean_relative_error: float


class ConclusionTest:
    r"""Test whether observed returns match predicted returns.

    Quantifies:

    .. math::

        |R_{\text{observed}} - R_{\text{predicted}}| / |R_{\text{predicted}}|

    If this test fails but 3.9.1 and 3.9.2 pass, the reward hypothesis
    is wrong.

    See ``CIRC-RL_Framework.md`` Section 3.9.3.

    :param relative_error_threshold: Maximum allowed mean relative error.
        Default 0.3 (30%).
    :param n_eval_episodes: Number of evaluation episodes per environment.
    :param max_steps: Maximum steps per episode.
    """

    def __init__(
        self,
        relative_error_threshold: float = 0.3,
        n_eval_episodes: int = 5,
        max_steps: int = 200,
    ) -> None:
        self._threshold = relative_error_threshold
        self._n_episodes = n_eval_episodes
        self._max_steps = max_steps

    def test(
        self,
        policy: AnalyticPolicy,
        env_family: EnvironmentFamily,
        predicted_returns: dict[int, float],
        test_env_ids: list[int],
    ) -> ConclusionTestResult:
        """Run the conclusion test.

        :param policy: The analytic policy.
        :param env_family: Environment family for evaluation.
        :param predicted_returns: Predicted mean return per env from theory.
        :param test_env_ids: Environment IDs to test.
        :returns: ConclusionTestResult.
        """
        per_env_observed: dict[int, float] = {}
        per_env_predicted: dict[int, float] = {}
        per_env_error: dict[int, float] = {}

        for env_id in test_env_ids:
            # Run episodes
            env = env_family.make_env(env_id)
            episode_returns: list[float] = []

            for ep in range(self._n_episodes):
                obs, _ = env.reset(seed=env_id * 100 + ep)
                total_reward = 0.0

                for _ in range(self._max_steps):
                    action = policy.get_action(np.asarray(obs), env_id)
                    obs, reward, terminated, truncated, _ = env.step(action)
                    total_reward += float(reward)
                    if terminated or truncated:
                        break

                episode_returns.append(total_reward)

            env.close()

            observed_mean = float(np.mean(episode_returns))
            per_env_observed[env_id] = observed_mean

            if env_id in predicted_returns:
                predicted = predicted_returns[env_id]
                per_env_predicted[env_id] = predicted

                if abs(predicted) > 1e-10:
                    rel_error = abs(observed_mean - predicted) / abs(predicted)
                else:
                    rel_error = abs(observed_mean)
                per_env_error[env_id] = rel_error
            else:
                per_env_predicted[env_id] = 0.0
                per_env_error[env_id] = 1.0

        if per_env_error:
            mean_error = float(np.mean(list(per_env_error.values())))
        else:
            mean_error = 0.0
        passed = mean_error <= self._threshold

        logger.info(
            "Conclusion test: mean_relative_error={:.4f}, "
            "threshold={:.4f}, passed={}",
            mean_error, self._threshold, passed,
        )

        return ConclusionTestResult(
            passed=passed,
            per_env_observed=per_env_observed,
            per_env_predicted=per_env_predicted,
            per_env_relative_error=per_env_error,
            mean_relative_error=mean_error,
        )
