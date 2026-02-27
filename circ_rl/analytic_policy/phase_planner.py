"""Phase-based planner with automatic solver switching.

Switches between global (MPPI) and local (iLQR) solvers based on
state-dependent phase detection. Typically: MPPI for energy pumping
and non-convex exploration, iLQR for stabilization near the goal.

See ``docs/proposed_solutions.md`` Solution 5 for rationale.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

import numpy as np
from loguru import logger

from circ_rl.analytic_policy.ilqr_solver import ILQRSolution

if TYPE_CHECKING:
    from collections.abc import Callable

    from circ_rl.analytic_policy.ilqr_solver import ILQRConfig, ILQRSolver
    from circ_rl.analytic_policy.mppi_solver import MPPIConfig, MPPISolver


class PhasePlanner:
    """Phase-based planner switching between global and local solvers.

    Uses a state-dependent phase detector to decide which solver to
    invoke at each replanning step:

    - **Global solver** (typically MPPI): for energy pumping, non-convex
      exploration, and approach phases where the cost landscape has many
      local minima.
    - **Local solver** (typically iLQR): for stabilization near the goal
      where the cost landscape is locally quadratic and feedback gains
      improve closed-loop performance.

    The phase detector is a callable ``(state) -> bool`` returning True
    when the local solver should be used (i.e., the system is near the
    goal and ready for stabilization).

    Handles horizon mismatches between solvers by truncating or padding
    warm-start actions as needed.

    :param global_solver: Solver for global/non-convex phases.
    :param local_solver: Solver for local stabilization.
    :param use_local_fn: Phase detector ``(state) -> bool``. Returns
        ``True`` when the system should switch to the local solver.
    """

    def __init__(
        self,
        global_solver: Union[MPPISolver, ILQRSolver],
        local_solver: ILQRSolver,
        use_local_fn: Callable[[np.ndarray], bool],
    ) -> None:
        self._global = global_solver
        self._local = local_solver
        self._use_local_fn = use_local_fn
        self._last_phase: str = "global"

    @property
    def config(self) -> Union[MPPIConfig, ILQRConfig]:
        """Configuration for replanning logic.

        Returns the global solver's config since it controls the
        overall planning cadence (replan_interval, etc.).
        """
        return self._global.config

    @property
    def last_phase(self) -> str:
        """Name of the last phase used: ``'global'`` or ``'local'``."""
        return self._last_phase

    def _resize_warm_start(
        self,
        warm_start: np.ndarray | None,
        target_horizon: int,
        action_dim: int,
    ) -> np.ndarray | None:
        """Resize warm-start actions to match the target solver's horizon.

        :param warm_start: Previous plan's actions ``(H_prev, A)`` or None.
        :param target_horizon: Target solver's horizon.
        :param action_dim: Number of action dimensions.
        :returns: Resized actions ``(target_horizon, A)`` or None.
        """
        if warm_start is None:
            return None

        h_prev = warm_start.shape[0]
        if h_prev == target_horizon:
            return warm_start
        elif h_prev > target_horizon:
            # Truncate to target horizon
            return warm_start[:target_horizon]
        else:
            # Pad with zeros
            pad = np.zeros((target_horizon - h_prev, action_dim))
            return np.vstack([warm_start, pad])

    def plan(
        self,
        initial_state: np.ndarray,
        action_dim: int,
        warm_start_actions: np.ndarray | None = None,
    ) -> ILQRSolution:
        """Plan from the given state using the appropriate phase solver.

        :param initial_state: Current state, shape ``(S,)``.
        :param action_dim: Number of action dimensions.
        :param warm_start_actions: Optional warm-start ``(H, A)``.
        :returns: Trajectory solution (``ILQRSolution``).
        """
        if self._use_local_fn(initial_state):
            self._last_phase = "local"
            logger.debug("PhasePlanner: using local solver (stabilization)")
            target_h = self._local.config.horizon
            ws = self._resize_warm_start(
                warm_start_actions, target_h, action_dim,
            )
            return self._local.plan(initial_state, action_dim, ws)
        else:
            self._last_phase = "global"
            logger.debug("PhasePlanner: using global solver (exploration)")
            target_h = self._global.config.horizon
            ws = self._resize_warm_start(
                warm_start_actions, target_h, action_dim,
            )
            return self._global.plan(initial_state, action_dim, ws)
