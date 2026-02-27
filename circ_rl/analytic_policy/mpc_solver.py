"""Model Predictive Control solver for nonlinear dynamics.

Solves a receding-horizon optimal control problem using
scipy.optimize.minimize.

See ``CIRC-RL_Framework.md`` Section 3.6.2 (Nonlinear Known Dynamics
+ Known Reward) and Section 3.6.4 (Constraint Integration).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger
from scipy import optimize

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True)
class MPCConfig:
    """Configuration for the MPC solver.

    :param horizon: Prediction/control horizon (timesteps).
    :param gamma: Discount factor for the cost function.
    :param max_action: Maximum absolute action value (box constraint).
    :param method: Scipy optimization method. Default ``"SLSQP"``.
    :param max_iter: Maximum optimizer iterations. Default 100.
    :param tol: Optimizer tolerance. Default 1e-6.
    """

    horizon: int = 10
    gamma: float = 0.99
    max_action: float = 2.0
    method: str = "SLSQP"
    max_iter: int = 100
    tol: float = 1e-6


class MPCSolver:
    r"""Receding-horizon MPC for nonlinear dynamics.

    Solves online:

    .. math::

        a^*_t = \arg\max_{a_t, \ldots, a_{t+H}} \sum_{k=0}^{H}
        \gamma^k R(s_{t+k}, a_{t+k})

    subject to:

    .. math::

        s_{t+k+1} = h(s_{t+k}, a_{t+k}; \theta_e)

    The validated dynamics hypothesis serves as the model.

    See ``CIRC-RL_Framework.md`` Section 3.6.2.

    :param config: MPC configuration.
    :param dynamics_fn: Callable ``(state, action) -> next_state``.
        The validated dynamics model.
    :param reward_fn: Callable ``(state, action) -> reward``.
        If None, uses negative squared state norm as default cost.
    """

    def __init__(
        self,
        config: MPCConfig,
        dynamics_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
        reward_fn: Callable[[np.ndarray, np.ndarray], float] | None = None,
    ) -> None:
        self._config = config
        self._dynamics_fn = dynamics_fn
        self._reward_fn = reward_fn or self._default_reward

    @property
    def config(self) -> MPCConfig:
        """The MPC configuration."""
        return self._config

    def solve(
        self,
        state: np.ndarray,
        action_dim: int,
    ) -> np.ndarray:
        """Solve MPC for the current state and return the first action.

        :param state: Current state, shape ``(state_dim,)``.
        :param action_dim: Number of action dimensions.
        :returns: Optimal first action, shape ``(action_dim,)``.
        """
        cfg = self._config
        horizon = cfg.horizon

        # Decision variable: all actions over the horizon
        # Shape: (horizon * action_dim,)
        x0 = np.zeros(horizon * action_dim)

        # Bounds: [-max_action, max_action] for each action component
        bounds = [(-cfg.max_action, cfg.max_action)] * (horizon * action_dim)

        def neg_total_reward(x: np.ndarray) -> float:
            """Negative total discounted reward (to minimize)."""
            actions = x.reshape(horizon, action_dim)
            s = state.copy()
            total = 0.0

            for k in range(horizon):
                a = actions[k]
                r = self._reward_fn(s, a)
                total += (cfg.gamma ** k) * r
                s = self._dynamics_fn(s, a)

            return -total  # Minimize negative reward = maximize reward

        result = optimize.minimize(
            neg_total_reward,
            x0,
            method=cfg.method,
            bounds=bounds,
            options={"maxiter": cfg.max_iter, "ftol": cfg.tol},
        )

        if not result.success:
            logger.debug(
                "MPC optimizer did not converge: {}", result.message,
            )

        # Extract first action
        actions = result.x.reshape(horizon, action_dim)
        return actions[0]  # (action_dim,)

    @staticmethod
    def _default_reward(state: np.ndarray, action: np.ndarray) -> float:
        """Default cost: negative squared state + action norm."""
        return -float(np.sum(state ** 2) + 0.01 * np.sum(action ** 2))
