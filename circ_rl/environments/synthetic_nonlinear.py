"""Synthetic environment with non-linear env-param effect on reward.

Designed for validating KRR-based conditional invariance: the ATE of
feature s0 on reward varies quadratically with env param ``k``, which
a linear model cannot capture but KRR with RBF kernel can.

Reward function::

    reward = k^2 + k^2 * s0 + 2.0 * s1 + 0.5 * action + 0.1 * noise

The ``k^2`` constant term ensures that the mean reward shifts with ``k``,
making the ``ep_k -> reward`` edge detectable by both PC (Fisher Z) and
the Pearson correlation pre-screen. The ATE of s0 on reward is still
``k^2`` (the constant is absorbed by the regression intercept).

- **s0**: context-dependent (ATE = k^2, quadratic in k)
- **s1**: invariant (ATE = 2.0, constant across envs)
- **s2**: irrelevant (not in reward function)
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class SyntheticNonlinearEnv(gym.Env[np.ndarray, np.ndarray]):
    """Gymnasium environment with non-linear env-param effect on reward.

    :param k: Environment parameter that scales s0's effect on reward
        quadratically (ATE of s0 = k^2). Modifiable via ``setattr``
        for use with :class:`~circ_rl.environments.env_family.EnvironmentFamily`.
    :param render_mode: Unused, present for Gymnasium API compatibility.
    """

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(
        self,
        k: float = 3.0,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.k = k
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(3,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )
        self._state: np.ndarray = np.zeros(3, dtype=np.float32)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset to a random state.

        :param seed: Random seed for reproducibility.
        :param options: Unused.
        :returns: Tuple of (observation, info).
        """
        super().reset(seed=seed)
        self._state = self.np_random.standard_normal(3).astype(np.float32)
        return self._state.copy(), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Take a step: compute reward and sample new random state.

        Reward = k^2 + k^2 * s0 + 2.0 * s1 + 0.5 * action + 0.1 * noise.

        The constant ``k^2`` term shifts the mean reward with k, making the
        ep_k -> reward relationship detectable. The ATE of s0 is still k^2.

        :param action: Action in [-1, 1].
        :returns: Tuple of (next_obs, reward, terminated, truncated, info).
        """
        s0, s1, _s2 = self._state
        a = float(np.clip(action, -1.0, 1.0)[0])

        reward = (
            (self.k ** 2)
            + (self.k ** 2) * s0
            + 2.0 * s1
            + 0.5 * a
            + 0.1 * float(self.np_random.standard_normal())
        )

        # Transition to new random state (no dynamics)
        self._state = self.np_random.standard_normal(3).astype(np.float32)

        return self._state.copy(), reward, False, False, {}


# Register with gymnasium so EnvironmentFamily.from_gymnasium() can use it
gym.register(
    id="SyntheticNonlinear-v0",
    entry_point="circ_rl.environments.synthetic_nonlinear:SyntheticNonlinearEnv",
)
