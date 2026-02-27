r"""Iterative Linear Quadratic Regulator (iLQR) solver.

Solves nonlinear trajectory optimization by iteratively linearizing
dynamics and quadraticizing cost along a nominal trajectory, then
solving the resulting LQR sub-problem via dynamic programming.

The algorithm:

1. **Forward pass**: Roll out trajectory under current control sequence.
2. **Backward pass**: Linearize dynamics (Jacobians :math:`A_t, B_t`)
   and quadraticize cost at each timestep, then solve the Bellman
   recursion to get feedforward :math:`k_t` and feedback :math:`K_t`
   gains.
3. **Line search**: Apply control update
   :math:`u_t = \bar{u}_t + \alpha k_t + K_t (x_t - \bar{x}_t)`
   with backtracking on :math:`\alpha`.
4. **Regularization**: Levenberg-Marquardt damping on
   :math:`Q_{uu}` to ensure positive-definiteness.
5. **Convergence**: Stop when relative cost change falls below
   threshold.

Internally uses cost-minimization convention (cost = -reward) with
standard iLQR formulas (Tassa et al. 2012). The public interface
accepts reward functions and reports total reward.

See ``CIRC-RL_Framework.md`` Section 3.6.2 (Nonlinear Known Dynamics).
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True)
class ILQRConfig:
    r"""Configuration for the iLQR solver.

    :param horizon: Planning horizon (number of timesteps).
    :param max_iterations: Maximum iLQR iterations.
    :param convergence_tol: Relative cost change threshold for
        convergence: :math:`|J_{new} - J_{old}| / |J_{old}|`.
    :param mu_init: Initial Levenberg-Marquardt regularization.
    :param mu_min: Minimum regularization.
    :param mu_max: Maximum regularization (triggers early stop).
    :param mu_factor: Multiplicative factor for regularization
        adjustment.
    :param alpha_min: Minimum line search step size.
    :param alpha_decay: Line search backtracking factor.
    :param gamma: Discount factor for the cost function.
    :param max_action: Maximum absolute action value (box constraint).
    :param n_random_restarts: Number of random action initializations
        to try in addition to the default (zero or warm-start). The best
        solution across all runs is returned. 0 disables multi-start.
    :param restart_scale: Scale factor for random restart action
        initialization. Actions are sampled from
        :math:`\mathcal{N}(0, \sigma)` where
        :math:`\sigma = \text{restart\_scale} \times \text{max\_action}`,
        then clipped to ``[-max_action, max_action]``.
    :param replan_interval: How often (in timesteps) to replan from
        the current observed state. Must be ``<= horizon``. When set
        ``< horizon``, the controller operates in MPC mode: it plans
        over the full horizon but only executes ``replan_interval``
        steps before replanning. ``None`` means same as ``horizon``
        (single-plan mode).
    :param use_tanh_squash: If ``True``, use smooth tanh squashing
        instead of hard clipping for action bounds in the line search.
        Provides gradient information near boundaries, improving
        optimization when actions are near ``max_action``.
    :param parallel_restarts: If ``True``, run multi-start restarts
        in parallel using ``ThreadPoolExecutor``. NumPy releases the
        GIL during computation, so thread-based parallelism is
        effective. Set to ``False`` for deterministic single-threaded
        execution.
    :param adaptive_replan_threshold: State deviation norm threshold
        for adaptive replanning. When the observed state deviates from
        the predicted nominal trajectory by more than this threshold,
        an immediate replan is triggered. ``None`` disables adaptive
        replanning (uses fixed ``replan_interval`` only).
        See ``CIRC-RL_Framework.md`` Section 7.2.
    :param min_replan_interval: Minimum steps between replans to
        prevent thrashing when the model is noisy. Adaptive replanning
        cannot trigger before this many steps since the last replan.
        Default 3.
    """

    horizon: int = 200
    max_iterations: int = 50
    convergence_tol: float = 1e-4
    mu_init: float = 1.0
    mu_min: float = 1e-6
    mu_max: float = 1e6
    mu_factor: float = 10.0
    alpha_min: float = 1e-4
    alpha_decay: float = 0.5
    gamma: float = 0.99
    max_action: float = 2.0
    n_random_restarts: int = 0
    restart_scale: float = 0.3
    replan_interval: int | None = None
    use_tanh_squash: bool = True
    parallel_restarts: bool = True
    adaptive_replan_threshold: float | None = None
    min_replan_interval: int = 3

    def __post_init__(self) -> None:
        if self.replan_interval is not None:
            if self.replan_interval < 1:
                raise ValueError(
                    f"replan_interval must be >= 1, "
                    f"got {self.replan_interval}"
                )
            if self.replan_interval > self.horizon:
                raise ValueError(
                    f"replan_interval ({self.replan_interval}) must be "
                    f"<= horizon ({self.horizon})"
                )
        if self.adaptive_replan_threshold is not None:
            if self.adaptive_replan_threshold <= 0:
                raise ValueError(
                    f"adaptive_replan_threshold must be > 0, "
                    f"got {self.adaptive_replan_threshold}"
                )
        if self.min_replan_interval < 1:
            raise ValueError(
                f"min_replan_interval must be >= 1, "
                f"got {self.min_replan_interval}"
            )


@dataclass
class ILQRSolution:
    r"""Solution from the iLQR solver.

    The optimal closed-loop policy is:

    .. math::

        u_t = \bar{u}_t + k_t + K_t (x_t - \bar{x}_t)

    where :math:`\bar{x}, \bar{u}` are the nominal trajectory,
    :math:`k_t` are feedforward gains, and :math:`K_t` are feedback
    gains.

    :param nominal_states: Nominal state trajectory,
        shape ``(horizon+1, state_dim)``.
    :param nominal_actions: Nominal action sequence,
        shape ``(horizon, action_dim)``.
    :param feedback_gains: Time-varying feedback gain matrices,
        ``horizon`` elements each of shape ``(action_dim, state_dim)``.
    :param feedforward_gains: Time-varying feedforward gain vectors,
        ``horizon`` elements each of shape ``(action_dim,)``.
    :param total_reward: Total discounted reward of the nominal
        trajectory.
    :param converged: Whether the solver converged.
    :param n_iterations: Number of iLQR iterations performed.
    """

    nominal_states: np.ndarray
    nominal_actions: np.ndarray
    feedback_gains: list[np.ndarray] = field(default_factory=list)
    feedforward_gains: list[np.ndarray] = field(default_factory=list)
    total_reward: float = 0.0
    converged: bool = False
    n_iterations: int = 0


class ILQRSolver:
    r"""Iterative LQR solver for nonlinear trajectory optimization.

    Given nonlinear dynamics :math:`x_{t+1} = f(x_t, u_t)` and
    stage reward :math:`r(x_t, u_t)`, finds the control sequence
    that maximizes the total discounted reward:

    .. math::

        \max_{u_0, \ldots, u_{T-1}} \sum_{t=0}^{T-1} \gamma^t r(x_t, u_t)

    Internally converts to cost minimization (:math:`c = -r`) and
    applies the standard iLQR algorithm (Tassa et al. 2012).

    Optionally accepts analytic Jacobian functions for the dynamics,
    computed from symbolic expressions via ``sympy.diff()``. Falls
    back to finite differences if not provided.

    See ``CIRC-RL_Framework.md`` Section 3.6.2.

    :param config: iLQR configuration.
    :param dynamics_fn: Callable ``(state, action) -> next_state``.
    :param reward_fn: Callable ``(state, action) -> float``.
    :param dynamics_jac_state_fn: Optional callable
        ``(state, action) -> A``, where ``A`` is the Jacobian
        :math:`\partial f / \partial x` of shape
        ``(state_dim, state_dim)``.
    :param dynamics_jac_action_fn: Optional callable
        ``(state, action) -> B``, where ``B`` is the Jacobian
        :math:`\partial f / \partial u` of shape
        ``(state_dim, action_dim)``.
    :param terminal_cost_fn: Optional callable
        ``(state) -> (cost, gradient, hessian)`` providing the terminal
        value function. ``cost`` is a scalar, ``gradient`` has shape
        ``(state_dim,)``, ``hessian`` has shape
        ``(state_dim, state_dim)``. Used to initialize the backward
        pass instead of zeros.
    :param reward_derivatives_fn: Optional callable
        ``(state, action) -> (r_x, r_u, r_xx, r_uu, r_ux)`` providing
        analytic reward derivatives. When provided, replaces finite-
        difference computation in the backward pass. Derivatives use
        the **reward** sign convention (positive); the solver negates
        them to obtain cost derivatives internally.
    """

    def __init__(
        self,
        config: ILQRConfig,
        dynamics_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
        reward_fn: Callable[[np.ndarray, np.ndarray], float],
        dynamics_jac_state_fn: (
            Callable[[np.ndarray, np.ndarray], np.ndarray] | None
        ) = None,
        dynamics_jac_action_fn: (
            Callable[[np.ndarray, np.ndarray], np.ndarray] | None
        ) = None,
        terminal_cost_fn: (
            Callable[
                [np.ndarray],
                tuple[float, np.ndarray, np.ndarray],
            ]
            | None
        ) = None,
        reward_derivatives_fn: (
            Callable[
                [np.ndarray, np.ndarray],
                tuple[
                    np.ndarray,
                    np.ndarray,
                    np.ndarray,
                    np.ndarray,
                    np.ndarray,
                ],
            ]
            | None
        ) = None,
    ) -> None:
        self._config = config
        self._dynamics_fn = dynamics_fn
        self._reward_fn = reward_fn
        self._jac_state_fn = dynamics_jac_state_fn
        self._jac_action_fn = dynamics_jac_action_fn
        self._terminal_cost_fn = terminal_cost_fn
        self._reward_derivatives_fn = reward_derivatives_fn

    @property
    def config(self) -> ILQRConfig:
        """The iLQR configuration."""
        return self._config

    def plan(
        self,
        initial_state: np.ndarray,
        action_dim: int,
        warm_start_actions: np.ndarray | None = None,
    ) -> ILQRSolution:
        """Optimize a full trajectory from the given initial state.

        When ``n_random_restarts > 0``, runs the optimization multiple
        times with random action initializations and returns the best
        solution. The first run uses the provided ``warm_start_actions``
        (or zeros); subsequent runs use uniform random actions.

        :param initial_state: Starting state, shape ``(state_dim,)``.
        :param action_dim: Number of action dimensions.
        :param warm_start_actions: Optional initial action sequence,
            shape ``(horizon, action_dim)``. Defaults to zeros.
        :returns: The optimized trajectory with feedback gains.
        """
        cfg = self._config

        if cfg.n_random_restarts <= 0:
            return self._plan_single(
                initial_state, action_dim, warm_start_actions,
            )

        # Pre-generate all random action sequences (RNG not thread-safe)
        rng = np.random.default_rng()
        all_inits: list[np.ndarray | None] = [warm_start_actions]
        for _ in range(cfg.n_random_restarts):
            all_inits.append(
                np.clip(
                    rng.normal(
                        0,
                        cfg.restart_scale * cfg.max_action,
                        size=(cfg.horizon, action_dim),
                    ),
                    -cfg.max_action,
                    cfg.max_action,
                )
            )

        n_runs = len(all_inits)

        if cfg.parallel_restarts and n_runs > 1:
            max_workers = min(n_runs, os.cpu_count() or 4)
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = [
                    pool.submit(
                        self._plan_single,
                        initial_state,
                        action_dim,
                        init_actions,
                    )
                    for init_actions in all_inits
                ]
                solutions = [f.result() for f in futures]
        else:
            solutions = [
                self._plan_single(initial_state, action_dim, init_actions)
                for init_actions in all_inits
            ]

        best_sol = max(solutions, key=lambda s: s.total_reward)

        logger.info(
            "iLQR multi-start: best of {} runs, reward={:.2f}",
            n_runs,
            best_sol.total_reward,
        )
        return best_sol

    def _plan_single(
        self,
        initial_state: np.ndarray,
        action_dim: int,
        warm_start_actions: np.ndarray | None = None,
    ) -> ILQRSolution:
        """Run a single iLQR optimization from the given initialization.

        :param initial_state: Starting state, shape ``(state_dim,)``.
        :param action_dim: Number of action dimensions.
        :param warm_start_actions: Optional initial action sequence,
            shape ``(horizon, action_dim)``. Defaults to zeros.
        :returns: The optimized trajectory with feedback gains.
        """
        cfg = self._config
        state_dim = initial_state.shape[0]
        horizon = cfg.horizon
        multi_start = cfg.n_random_restarts > 0

        # Initialize action sequence
        if warm_start_actions is not None:
            assert warm_start_actions.shape == (horizon, action_dim), (
                f"Expected warm_start shape ({horizon}, {action_dim}), "
                f"got {warm_start_actions.shape}"
            )
            actions = warm_start_actions.copy()
        else:
            actions = np.zeros((horizon, action_dim))

        # Initial forward pass
        states = self._rollout(initial_state, actions)
        current_cost = self._total_cost(states, actions)

        mu = cfg.mu_init
        converged = False
        n_iter = 0

        # Storage for gains (from last successful backward pass)
        k_gains: list[np.ndarray] = []
        big_k_gains: list[np.ndarray] = []

        for iteration in range(cfg.max_iterations):
            n_iter = iteration + 1

            # Backward pass (cost minimization convention)
            backward_ok, k_gains, big_k_gains = self._backward_pass(
                states, actions, state_dim, action_dim, mu,
            )

            if not backward_ok:
                # Increase regularization and retry
                mu = min(mu * cfg.mu_factor, cfg.mu_max)
                if mu >= cfg.mu_max:
                    logger.debug(
                        "iLQR: regularization exceeded mu_max={}, "
                        "stopping at iteration {}",
                        cfg.mu_max,
                        n_iter,
                    )
                    break
                continue

            # Forward pass with line search
            new_states, new_actions, new_cost = self._line_search(
                states, actions, k_gains, big_k_gains,
                current_cost, action_dim,
            )

            if new_cost < current_cost:
                # Cost decreased (= reward increased): accept
                rel_improvement = abs(current_cost - new_cost) / max(
                    abs(current_cost), 1e-10,
                )

                states = new_states
                actions = new_actions
                current_cost = new_cost

                # Decrease regularization on success
                mu = max(mu / cfg.mu_factor, cfg.mu_min)

                logger.debug(
                    "iLQR iter {}: reward={:.2f}, "
                    "rel_improvement={:.6f}, mu={:.2e}",
                    n_iter,
                    -current_cost,
                    rel_improvement,
                    mu,
                )

                if rel_improvement < cfg.convergence_tol:
                    converged = True
                    break
            else:
                # No improvement: increase regularization
                mu = min(mu * cfg.mu_factor, cfg.mu_max)
                if mu >= cfg.mu_max:
                    logger.debug(
                        "iLQR: regularization exceeded mu_max at iter {}",
                        n_iter,
                    )
                    break

        total_reward = -current_cost

        # Use debug logging for individual runs during multi-start
        log_fn = logger.debug if multi_start else logger.info
        log_fn(
            "iLQR: {} in {} iterations, reward={:.2f}, mu={:.2e}",
            "converged" if converged else "stopped",
            n_iter,
            total_reward,
            mu,
        )

        return ILQRSolution(
            nominal_states=states,
            nominal_actions=actions,
            feedback_gains=big_k_gains,
            feedforward_gains=k_gains,
            total_reward=total_reward,
            converged=converged,
            n_iterations=n_iter,
        )

    def _rollout(
        self,
        initial_state: np.ndarray,
        actions: np.ndarray,
    ) -> np.ndarray:
        """Forward-simulate the dynamics under the given actions.

        :param initial_state: Shape ``(state_dim,)``.
        :param actions: Shape ``(horizon, action_dim)``.
        :returns: States, shape ``(horizon+1, state_dim)``.
        """
        horizon = actions.shape[0]
        state_dim = initial_state.shape[0]
        states = np.zeros((horizon + 1, state_dim))
        states[0] = initial_state

        for t in range(horizon):
            states[t + 1] = self._dynamics_fn(states[t], actions[t])

        return states

    def _total_cost(
        self,
        states: np.ndarray,
        actions: np.ndarray,
    ) -> float:
        """Compute total discounted cost (= -reward) along a trajectory.

        :param states: Shape ``(horizon+1, state_dim)``.
        :param actions: Shape ``(horizon, action_dim)``.
        :returns: Scalar total discounted cost.
        """
        gamma = self._config.gamma
        horizon = actions.shape[0]
        total = 0.0

        for t in range(horizon):
            total -= (gamma ** t) * self._reward_fn(states[t], actions[t])

        return total

    def _backward_pass(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        state_dim: int,
        action_dim: int,
        mu: float,
    ) -> tuple[bool, list[np.ndarray], list[np.ndarray]]:
        r"""Backward pass: compute feedforward and feedback gains.

        Uses cost-minimization convention (:math:`c = -r`).
        The Q-function is quadratically approximated at each timestep:

        .. math::

            Q_{uu} = c_{uu} + \gamma B^T V'_{xx} B + \mu I

        where :math:`c_{uu} = -r_{uu}` (positive for concave rewards).

        :param states: Nominal states, shape ``(T+1, state_dim)``.
        :param actions: Nominal actions, shape ``(T, action_dim)``.
        :param state_dim: State dimensionality.
        :param action_dim: Action dimensionality.
        :param mu: Levenberg-Marquardt regularization parameter.
        :returns: Tuple of (success, k_gains, K_gains).
        """
        gamma = self._config.gamma
        horizon = actions.shape[0]

        # Terminal value function
        if self._terminal_cost_fn is not None:
            try:
                _tc_val, v_x, v_xx = self._terminal_cost_fn(states[-1])
                v_x = np.asarray(v_x, dtype=np.float64)  # (state_dim,)
                v_xx = np.asarray(v_xx, dtype=np.float64)  # (state_dim, state_dim)
            except np.linalg.LinAlgError:
                v_x = np.zeros(state_dim)  # (state_dim,)
                v_xx = np.zeros((state_dim, state_dim))  # (state_dim, state_dim)
        else:
            v_x = np.zeros(state_dim)  # (state_dim,)
            v_xx = np.zeros((state_dim, state_dim))  # (state_dim, state_dim)

        k_gains: list[np.ndarray] = [
            np.zeros(action_dim) for _ in range(horizon)
        ]
        big_k_gains: list[np.ndarray] = [
            np.zeros((action_dim, state_dim)) for _ in range(horizon)
        ]

        for t in range(horizon - 1, -1, -1):
            x_t = states[t]
            u_t = actions[t]

            # Dynamics Jacobians
            a_mat = self._get_dynamics_jac_state(x_t, u_t, state_dim)
            b_mat = self._get_dynamics_jac_action(
                x_t, u_t, state_dim, action_dim,
            )

            # Cost derivatives (cost = -reward)
            # c_x = -r_x, c_u = -r_u, c_xx = -r_xx, etc.
            c_x, c_u, c_xx, c_uu, c_ux = self._cost_derivatives(
                x_t, u_t, state_dim, action_dim,
            )

            # Q-function terms (standard cost-minimization iLQR)
            q_x = c_x + gamma * a_mat.T @ v_x  # (state_dim,)
            q_u = c_u + gamma * b_mat.T @ v_x  # (action_dim,)

            q_xx = c_xx + gamma * a_mat.T @ v_xx @ a_mat  # (n, n)
            q_ux = c_ux + gamma * b_mat.T @ v_xx @ a_mat  # (m, n)
            q_uu = c_uu + gamma * b_mat.T @ v_xx @ b_mat  # (m, m)

            # Regularize Q_uu (must be positive definite for minimum)
            q_uu_reg = q_uu + mu * np.eye(action_dim)  # (m, m)

            # Check positive-definiteness via Cholesky
            try:
                cho = np.linalg.cholesky(q_uu_reg)
            except np.linalg.LinAlgError:
                return False, k_gains, big_k_gains

            # Compute gains: k = -Q_uu^{-1} Q_u, K = -Q_uu^{-1} Q_ux
            # Using Cholesky: solve Q_uu_reg @ k = -Q_u
            k_t = -np.linalg.solve(
                cho @ cho.T, q_u,
            )  # (action_dim,)
            big_k_t = -np.linalg.solve(
                cho @ cho.T, q_ux,
            )  # (action_dim, state_dim)

            k_gains[t] = k_t
            big_k_gains[t] = big_k_t

            # Update value function (simplified formulas for symmetric
            # Q_uu):
            #   V_x = Q_x + K^T Q_u   (= Q_x - Q_ux^T Q_uu^{-1} Q_u)
            #   V_xx = Q_xx + K^T Q_ux (= Q_xx - Q_ux^T Q_uu^{-1} Q_ux)
            v_x = q_x + big_k_t.T @ q_u
            v_xx = q_xx + big_k_t.T @ q_ux

            # Symmetrize V_xx for numerical stability
            v_xx = 0.5 * (v_xx + v_xx.T)

        return True, k_gains, big_k_gains

    def _line_search(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        k_gains: list[np.ndarray],
        big_k_gains: list[np.ndarray],
        current_cost: float,
        action_dim: int,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Forward pass with backtracking line search.

        Tries decreasing step sizes until cost improves.

        :param states: Nominal states, shape ``(T+1, n)``.
        :param actions: Nominal actions, shape ``(T, m)``.
        :param k_gains: Feedforward gains, T x (m,).
        :param big_k_gains: Feedback gains, T x (m, n).
        :param current_cost: Current total cost.
        :param action_dim: Action dimensionality.
        :returns: Tuple of (new_states, new_actions, new_cost).
        """
        cfg = self._config
        horizon = actions.shape[0]

        best_states = states
        best_actions = actions
        best_cost = current_cost

        alpha = 1.0
        while alpha >= cfg.alpha_min:
            new_states = np.zeros_like(states)
            new_actions = np.zeros_like(actions)
            new_states[0] = states[0]

            for t in range(horizon):
                dx = new_states[t] - states[t]  # (state_dim,)
                new_actions[t] = (
                    actions[t]
                    + alpha * k_gains[t]
                    + big_k_gains[t] @ dx
                )

                # Bound actions: smooth tanh or hard clip
                if cfg.use_tanh_squash:
                    new_actions[t] = cfg.max_action * np.tanh(
                        new_actions[t] / cfg.max_action,
                    )
                else:
                    np.clip(
                        new_actions[t],
                        -cfg.max_action,
                        cfg.max_action,
                        out=new_actions[t],
                    )

                new_states[t + 1] = self._dynamics_fn(
                    new_states[t], new_actions[t],
                )

            new_cost = self._total_cost(new_states, new_actions)

            if new_cost < best_cost:
                best_states = new_states
                best_actions = new_actions
                best_cost = new_cost
                break

            alpha *= cfg.alpha_decay

        return best_states, best_actions, best_cost

    def _get_dynamics_jac_state(
        self,
        state: np.ndarray,
        action: np.ndarray,
        state_dim: int,
    ) -> np.ndarray:
        r"""Get the dynamics Jacobian :math:`\partial f / \partial x`.

        Uses analytic Jacobian if available, otherwise finite differences.

        :returns: Shape ``(state_dim, state_dim)``.
        """
        if self._jac_state_fn is not None:
            return self._jac_state_fn(state, action)

        return _finite_diff_jac_state(
            self._dynamics_fn, state, action, state_dim,
        )

    def _get_dynamics_jac_action(
        self,
        state: np.ndarray,
        action: np.ndarray,
        state_dim: int,
        action_dim: int,
    ) -> np.ndarray:
        r"""Get the dynamics Jacobian :math:`\partial f / \partial u`.

        Uses analytic Jacobian if available, otherwise finite differences.

        :returns: Shape ``(state_dim, action_dim)``.
        """
        if self._jac_action_fn is not None:
            return self._jac_action_fn(state, action)

        return _finite_diff_jac_action(
            self._dynamics_fn, state, action, state_dim, action_dim,
        )

    def _cost_derivatives(
        self,
        state: np.ndarray,
        action: np.ndarray,
        state_dim: int,
        action_dim: int,
        eps: float = 1e-5,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        r"""Compute cost derivatives, analytically or via finite differences.

        Cost is defined as :math:`c = -r` (negated reward). When
        ``reward_derivatives_fn`` is available, uses exact analytic
        derivatives; otherwise falls back to finite differences.

        :returns: Tuple ``(c_x, c_u, c_xx, c_uu, c_ux)`` where
            ``c_x`` has shape ``(state_dim,)``,
            ``c_u`` has shape ``(action_dim,)``,
            ``c_xx`` has shape ``(state_dim, state_dim)``,
            ``c_uu`` has shape ``(action_dim, action_dim)``,
            ``c_ux`` has shape ``(action_dim, state_dim)``.
        """
        if self._reward_derivatives_fn is not None:
            r_x, r_u, r_xx, r_uu, r_ux = self._reward_derivatives_fn(
                state, action,
            )
            return -r_x, -r_u, -r_xx, -r_uu, -r_ux

        r0 = self._reward_fn(state, action)

        # Gradient w.r.t. state: c_x = -r_x
        c_x = np.zeros(state_dim)
        for i in range(state_dim):
            s_plus = state.copy()
            s_minus = state.copy()
            s_plus[i] += eps
            s_minus[i] -= eps
            c_x[i] = -(
                self._reward_fn(s_plus, action)
                - self._reward_fn(s_minus, action)
            ) / (2 * eps)

        # Gradient w.r.t. action: c_u = -r_u
        c_u = np.zeros(action_dim)
        for i in range(action_dim):
            a_plus = action.copy()
            a_minus = action.copy()
            a_plus[i] += eps
            a_minus[i] -= eps
            c_u[i] = -(
                self._reward_fn(state, a_plus)
                - self._reward_fn(state, a_minus)
            ) / (2 * eps)

        # Hessian w.r.t. state: c_xx = -r_xx
        c_xx = np.zeros((state_dim, state_dim))
        for i in range(state_dim):
            for j in range(i, state_dim):
                if i == j:
                    s_plus = state.copy()
                    s_minus = state.copy()
                    s_plus[i] += eps
                    s_minus[i] -= eps
                    c_xx[i, i] = -(
                        self._reward_fn(s_plus, action)
                        - 2 * r0
                        + self._reward_fn(s_minus, action)
                    ) / (eps ** 2)
                else:
                    s_pp = state.copy()
                    s_pm = state.copy()
                    s_mp = state.copy()
                    s_mm = state.copy()
                    s_pp[i] += eps
                    s_pp[j] += eps
                    s_pm[i] += eps
                    s_pm[j] -= eps
                    s_mp[i] -= eps
                    s_mp[j] += eps
                    s_mm[i] -= eps
                    s_mm[j] -= eps
                    c_xx[i, j] = -(
                        self._reward_fn(s_pp, action)
                        - self._reward_fn(s_pm, action)
                        - self._reward_fn(s_mp, action)
                        + self._reward_fn(s_mm, action)
                    ) / (4 * eps ** 2)
                    c_xx[j, i] = c_xx[i, j]

        # Hessian w.r.t. action: c_uu = -r_uu
        c_uu = np.zeros((action_dim, action_dim))
        for i in range(action_dim):
            for j in range(i, action_dim):
                if i == j:
                    a_plus = action.copy()
                    a_minus = action.copy()
                    a_plus[i] += eps
                    a_minus[i] -= eps
                    c_uu[i, i] = -(
                        self._reward_fn(state, a_plus)
                        - 2 * r0
                        + self._reward_fn(state, a_minus)
                    ) / (eps ** 2)
                else:
                    a_pp = action.copy()
                    a_pm = action.copy()
                    a_mp = action.copy()
                    a_mm = action.copy()
                    a_pp[i] += eps
                    a_pp[j] += eps
                    a_pm[i] += eps
                    a_pm[j] -= eps
                    a_mp[i] -= eps
                    a_mp[j] += eps
                    a_mm[i] -= eps
                    a_mm[j] -= eps
                    c_uu[i, j] = -(
                        self._reward_fn(state, a_pp)
                        - self._reward_fn(state, a_pm)
                        - self._reward_fn(state, a_mp)
                        + self._reward_fn(state, a_mm)
                    ) / (4 * eps ** 2)
                    c_uu[j, i] = c_uu[i, j]

        # Cross-Hessian: c_ux = -r_ux
        c_ux = np.zeros((action_dim, state_dim))
        for i in range(action_dim):
            for j in range(state_dim):
                s_plus = state.copy()
                s_minus = state.copy()
                a_plus = action.copy()
                a_minus = action.copy()
                s_plus[j] += eps
                s_minus[j] -= eps
                a_plus[i] += eps
                a_minus[i] -= eps

                c_ux[i, j] = -(
                    self._reward_fn(s_plus, a_plus)
                    - self._reward_fn(s_plus, a_minus)
                    - self._reward_fn(s_minus, a_plus)
                    + self._reward_fn(s_minus, a_minus)
                ) / (4 * eps ** 2)

        return c_x, c_u, c_xx, c_uu, c_ux


def _finite_diff_jac_state(
    dynamics_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    state: np.ndarray,
    action: np.ndarray,
    state_dim: int,
    eps: float = 1e-5,
) -> np.ndarray:
    r"""Compute :math:`\partial f / \partial x` via central differences.

    :returns: Jacobian, shape ``(state_dim, state_dim)``.
    """
    jac = np.zeros((state_dim, state_dim))
    for i in range(state_dim):
        s_plus = state.copy()
        s_minus = state.copy()
        s_plus[i] += eps
        s_minus[i] -= eps
        jac[:, i] = (
            dynamics_fn(s_plus, action)
            - dynamics_fn(s_minus, action)
        ) / (2 * eps)
    return jac


def _finite_diff_jac_action(
    dynamics_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    state: np.ndarray,
    action: np.ndarray,
    state_dim: int,
    action_dim: int,
    eps: float = 1e-5,
) -> np.ndarray:
    r"""Compute :math:`\partial f / \partial u` via central differences.

    :returns: Jacobian, shape ``(state_dim, action_dim)``.
    """
    jac = np.zeros((state_dim, action_dim))
    for i in range(action_dim):
        a_plus = action.copy()
        a_minus = action.copy()
        a_plus[i] += eps
        a_minus[i] -= eps
        jac[:, i] = (
            dynamics_fn(state, a_plus)
            - dynamics_fn(state, a_minus)
        ) / (2 * eps)
    return jac


def make_quadratic_terminal_cost(
    reward_fn: Callable[[np.ndarray, np.ndarray], float],
    action_dim: int,
    gamma: float,
    state_dim: int,
    eps: float = 1e-5,
    scale_override: float | None = None,
    max_hessian_eigval: float = 1e4,
) -> Callable[[np.ndarray], tuple[float, np.ndarray, np.ndarray]]:
    r"""Build a terminal cost function from the running reward.

    Approximates the infinite-horizon cost-to-go by assuming the
    system stays at the terminal state under zero action:

    .. math::

        V_T(x) \approx s \cdot c(x, 0)

    where :math:`c = -r` and :math:`s` is either ``scale_override``
    or :math:`\gamma / (1 - \gamma)`. The gradient and Hessian are
    computed via finite differences at the terminal state. Hessian
    eigenvalues are clamped to ``[-max_hessian_eigval, max_hessian_eigval]``
    to prevent ill-conditioning.

    :param reward_fn: Stage reward ``(state, action) -> float``.
    :param action_dim: Action dimensionality.
    :param gamma: Discount factor.
    :param state_dim: State dimensionality.
    :param eps: Finite difference step size.
    :param scale_override: If provided, use this as the terminal cost
        scale instead of ``gamma / (1 - gamma)``. Useful to reduce
        terminal cost aggressiveness when gamma is close to 1.
    :param max_hessian_eigval: Maximum absolute eigenvalue for the
        scaled terminal Hessian. Eigenvalues exceeding this are
        clamped to prevent ill-conditioning.
    :returns: Callable ``(state) -> (cost, gradient, hessian)``.
    """
    if scale_override is not None:
        scale = scale_override
    else:
        scale = gamma / (1.0 - gamma)

    def terminal_cost_fn(
        state: np.ndarray,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        zero_action = np.zeros(action_dim)
        r0 = reward_fn(state, zero_action)
        c0 = -r0

        # Gradient: c_x = -r_x via central differences
        c_x = np.zeros(state_dim)  # (state_dim,)
        for i in range(state_dim):
            s_plus = state.copy()
            s_minus = state.copy()
            s_plus[i] += eps
            s_minus[i] -= eps
            c_x[i] = -(
                reward_fn(s_plus, zero_action)
                - reward_fn(s_minus, zero_action)
            ) / (2 * eps)

        # Hessian: c_xx = -r_xx via central differences
        c_xx = np.zeros((state_dim, state_dim))  # (state_dim, state_dim)
        for i in range(state_dim):
            for j in range(i, state_dim):
                if i == j:
                    s_plus = state.copy()
                    s_minus = state.copy()
                    s_plus[i] += eps
                    s_minus[i] -= eps
                    c_xx[i, i] = -(
                        reward_fn(s_plus, zero_action)
                        - 2 * r0
                        + reward_fn(s_minus, zero_action)
                    ) / (eps ** 2)
                else:
                    s_pp = state.copy()
                    s_pm = state.copy()
                    s_mp = state.copy()
                    s_mm = state.copy()
                    s_pp[i] += eps
                    s_pp[j] += eps
                    s_pm[i] += eps
                    s_pm[j] -= eps
                    s_mp[i] -= eps
                    s_mp[j] += eps
                    s_mm[i] -= eps
                    s_mm[j] -= eps
                    c_xx[i, j] = -(
                        reward_fn(s_pp, zero_action)
                        - reward_fn(s_pm, zero_action)
                        - reward_fn(s_mp, zero_action)
                        + reward_fn(s_mm, zero_action)
                    ) / (4 * eps ** 2)
                    c_xx[j, i] = c_xx[i, j]

        scaled_hessian = scale * c_xx  # (state_dim, state_dim)

        # Guard against NaN/Inf from numerical Hessian
        if not np.all(np.isfinite(scaled_hessian)):
            scaled_hessian = np.eye(state_dim) * max_hessian_eigval

        # Clamp eigenvalues to prevent ill-conditioning
        try:
            eigvals, eigvecs = np.linalg.eigh(scaled_hessian)
        except np.linalg.LinAlgError:
            # Fallback: use identity-scaled Hessian
            scaled_hessian = np.eye(state_dim) * max_hessian_eigval
            eigvals = np.full(state_dim, max_hessian_eigval)
            eigvecs = np.eye(state_dim)
        eigvals = np.clip(
            eigvals, -max_hessian_eigval, max_hessian_eigval,
        )
        scaled_hessian = eigvecs @ np.diag(eigvals) @ eigvecs.T

        return (
            scale * c0,
            scale * c_x,
            scaled_hessian,
        )

    return terminal_cost_fn
