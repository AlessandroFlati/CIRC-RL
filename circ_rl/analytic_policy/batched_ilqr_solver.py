r"""Batched numpy iLQR solver for multi-restart optimization.

Processes all random restarts as a single ``(B, ...)`` numpy batch,
replacing the per-restart ``ThreadPoolExecutor`` in ``ILQRSolver``.
For small state/action dimensions (e.g. Pendulum: S=2, A=1) where
individual matrix operations are tiny and Python dispatch overhead
dominates, batching eliminates the per-restart GIL contention and
achieves near-linear speedup in the number of restarts.

Key differences from ``ILQRSolver``:

- All B restarts share a single iteration loop (no per-restart
  early stopping -- converged restarts are simply not updated).
- Dynamics/Jacobian/cost computations are batched: one Python dispatch
  per timestep for all B restarts instead of B dispatches.
- Uses ``np.linalg.cholesky`` and ``np.linalg.solve`` with batch
  dimensions for the Riccati recursion.

See ``CIRC-RL_Framework.md`` Section 3.6.2.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from collections.abc import Callable

    from circ_rl.analytic_policy.ilqr_solver import ILQRConfig, ILQRSolution


def _batched_cholesky_safe(
    matrices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Batched Cholesky with per-element fallback on failure.

    :param matrices: Positive-definite matrices, shape ``(B, A, A)``.
    :returns: Tuple of (cholesky_factors, success_mask).
        cholesky_factors has shape ``(B, A, A)``.
        success_mask has shape ``(B,)`` -- True where Cholesky succeeded.
    """
    B, A, _ = matrices.shape
    cho = np.zeros_like(matrices)  # (B, A, A)
    ok = np.ones(B, dtype=bool)  # (B,)

    if A == 1:
        # Scalar case: Cholesky = sqrt(value)
        vals = matrices[:, 0, 0]  # (B,)
        ok = vals > 0
        cho[ok, 0, 0] = np.sqrt(vals[ok])
        return cho, ok

    try:
        cho = np.linalg.cholesky(matrices)
        return cho, ok
    except np.linalg.LinAlgError:
        # Some elements failed; fall back to per-element
        for b in range(B):
            try:
                cho[b] = np.linalg.cholesky(matrices[b])
            except np.linalg.LinAlgError:
                ok[b] = False
        return cho, ok


class BatchedNumpyILQRSolver:
    r"""Batched iLQR that processes all restarts in numpy arrays.

    For systems where ``state_dim + action_dim`` is small (< ~6),
    individual numpy operations on tiny matrices are dominated by
    Python dispatch overhead. This solver batches all B restarts
    into ``(B, ...)`` arrays so that each timestep's matrix algebra
    is a single batched numpy call.

    Speedup over ``ILQRSolver`` with ``ThreadPoolExecutor``:

    - Eliminates B separate Python-level for-loops.
    - Eliminates GIL contention between restart threads.
    - Leverages vectorized BLAS for batched matmul/solve.

    :param config: iLQR configuration.
    :param batched_dynamics_fn: ``(B, S), (B, A) -> (B, S)``
        vectorized dynamics.
    :param reward_fn: ``(S,), (A,) -> float`` scalar reward.
    :param batched_jac_state_fn: ``(B, S), (B, A) -> (B, S, S)``
        batched state Jacobian.
    :param batched_jac_action_fn: ``(B, S), (B, A) -> (B, S, A)``
        batched action Jacobian.
    :param reward_derivatives_fn: Optional
        ``(S,), (A,) -> (r_x, r_u, r_xx, r_uu, r_ux)``
        analytic reward derivatives.
    :param batched_reward_fn: Optional ``(B, S), (B, A) -> (B,)``
        vectorized reward. When provided, cost evaluation is batched
        over restarts (H loop over vectorized B calls instead of
        B*H scalar calls). Falls back to ``reward_fn`` if not given.
    :param terminal_cost_fn: Optional
        ``(S,) -> (cost, gradient, hessian)`` for terminal value.
    """

    def __init__(
        self,
        config: ILQRConfig,
        batched_dynamics_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
        reward_fn: Callable[[np.ndarray, np.ndarray], float],
        batched_jac_state_fn: Callable[
            [np.ndarray, np.ndarray], np.ndarray
        ],
        batched_jac_action_fn: Callable[
            [np.ndarray, np.ndarray], np.ndarray
        ],
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
        batched_reward_fn: (
            Callable[[np.ndarray, np.ndarray], np.ndarray] | None
        ) = None,
        terminal_cost_fn: (
            Callable[
                [np.ndarray],
                tuple[float, np.ndarray, np.ndarray],
            ]
            | None
        ) = None,
    ) -> None:
        self._config = config
        self._batched_dynamics_fn = batched_dynamics_fn
        self._reward_fn = reward_fn
        self._batched_reward_fn = batched_reward_fn
        self._batched_jac_state_fn = batched_jac_state_fn
        self._batched_jac_action_fn = batched_jac_action_fn
        self._reward_derivatives_fn = reward_derivatives_fn
        self._terminal_cost_fn = terminal_cost_fn

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
        """Optimize a trajectory using batched multi-restart iLQR.

        All restarts are processed as a single ``(B, ...)`` batch.

        :param initial_state: Starting state, shape ``(state_dim,)``.
        :param action_dim: Number of action dimensions.
        :param warm_start_actions: Optional initial action sequence,
            shape ``(horizon, action_dim)``.
        :returns: Best solution across all restarts.
        """
        from circ_rl.analytic_policy.ilqr_solver import ILQRSolution

        cfg = self._config
        B = cfg.n_random_restarts + 1
        H = cfg.horizon
        S = initial_state.shape[0]
        A = action_dim

        # Initialize all action sequences: (B, H, A)
        actions = np.zeros((B, H, A))
        if warm_start_actions is not None:
            actions[0] = warm_start_actions
        rng = np.random.default_rng()
        for i in range(1, B):
            actions[i] = np.clip(
                rng.normal(
                    0,
                    cfg.restart_scale * cfg.max_action,
                    size=(H, A),
                ),
                -cfg.max_action,
                cfg.max_action,
            )

        # Initial forward rollout: (B, H+1, S)
        states = self._batched_rollout(initial_state, actions, S)
        costs = self._batched_total_cost(states, actions)  # (B,)

        # Per-restart state
        mu = np.full(B, cfg.mu_init)
        converged = np.zeros(B, dtype=bool)
        max_n_iter = 0

        # Pre-allocate gains: (B, H, A) and (B, H, A, S)
        k_gains = np.zeros((B, H, A))
        big_k_gains = np.zeros((B, H, A, S))

        for iteration in range(cfg.max_iterations):
            max_n_iter = iteration + 1

            # Skip iteration for converged restarts (still included in
            # batch arrays but costs/states/actions are not updated)
            active = ~converged

            if not active.any():
                break

            # Batched backward pass
            bw_ok, new_k, new_K = self._batched_backward(
                states, actions, S, A, mu,
            )

            # Failed restarts: increase mu
            failed = active & ~bw_ok
            mu[failed] = np.minimum(
                mu[failed] * cfg.mu_factor, cfg.mu_max,
            )

            # Check for mu overflow
            mu_overflow = active & (mu >= cfg.mu_max)
            converged[mu_overflow] = True  # Treat as "stopped"

            # Update gains for successful restarts
            succeeded = active & bw_ok
            if succeeded.any():
                k_gains[succeeded] = new_k[succeeded]
                big_k_gains[succeeded] = new_K[succeeded]

            # Batched line search for successful restarts
            new_states, new_actions, new_costs = self._batched_line_search(
                states, actions, k_gains, big_k_gains, costs, S, A,
            )

            # Evaluate improvement per restart
            improved = succeeded & (new_costs < costs)
            rel_improvement = np.abs(costs - new_costs) / np.maximum(
                np.abs(costs), 1e-10,
            )

            # Accept improvements
            states[improved] = new_states[improved]
            actions[improved] = new_actions[improved]
            costs[improved] = new_costs[improved]
            mu[improved] = np.maximum(
                mu[improved] / cfg.mu_factor, cfg.mu_min,
            )

            # Check convergence
            newly_converged = improved & (
                rel_improvement < cfg.convergence_tol
            )
            converged[newly_converged] = True

            # Not improved: increase mu
            not_improved = succeeded & ~improved
            mu[not_improved] = np.minimum(
                mu[not_improved] * cfg.mu_factor, cfg.mu_max,
            )

        # Select best restart
        best_idx = int(np.argmin(costs))
        total_reward = -float(costs[best_idx])

        # Build ILQRSolution with lists of gains (matching ILQRSolver)
        k_list = [k_gains[best_idx, t].copy() for t in range(H)]
        K_list = [big_k_gains[best_idx, t].copy() for t in range(H)]

        logger.info(
            "Batched iLQR: best of {} restarts, reward={:.2f}, "
            "converged={}",
            B,
            total_reward,
            bool(converged[best_idx]),
        )

        return ILQRSolution(
            nominal_states=states[best_idx].copy(),
            nominal_actions=actions[best_idx].copy(),
            feedback_gains=K_list,
            feedforward_gains=k_list,
            total_reward=total_reward,
            converged=bool(converged[best_idx]),
            n_iterations=max_n_iter,
        )

    def _batched_rollout(
        self,
        initial_state: np.ndarray,
        actions: np.ndarray,
        state_dim: int,
    ) -> np.ndarray:
        """Forward-simulate dynamics for all restarts.

        :param initial_state: Shape ``(S,)``.
        :param actions: Shape ``(B, H, A)``.
        :param state_dim: State dimensionality.
        :returns: States, shape ``(B, H+1, S)``.
        """
        B, H, _A = actions.shape
        states = np.zeros((B, H + 1, state_dim))  # (B, H+1, S)
        states[:, 0] = initial_state[None, :]  # Broadcast: (1, S) -> (B, S)

        for t in range(H):
            states[:, t + 1] = self._batched_dynamics_fn(
                states[:, t], actions[:, t],
            )  # (B, S)

        return states

    def _batched_total_cost(
        self,
        states: np.ndarray,
        actions: np.ndarray,
    ) -> np.ndarray:
        """Compute total discounted cost for all restarts.

        When ``batched_reward_fn`` is available, uses vectorized
        evaluation over restarts (H calls of (B,) batches). Otherwise
        falls back to B*H scalar calls.

        :param states: Shape ``(B, H+1, S)``.
        :param actions: Shape ``(B, H, A)``.
        :returns: Shape ``(B,)``.
        """
        gamma = self._config.gamma
        B, H, _A = actions.shape
        costs = np.zeros(B)  # (B,)
        gamma_powers = gamma ** np.arange(H)  # (H,)

        if self._batched_reward_fn is not None:
            for t in range(H):
                rewards = self._batched_reward_fn(
                    states[:, t], actions[:, t],
                )  # (B,)
                costs -= gamma_powers[t] * rewards
        else:
            for t in range(H):
                for b in range(B):
                    costs[b] -= gamma_powers[t] * self._reward_fn(
                        states[b, t], actions[b, t],
                    )

        return costs

    def _batched_backward(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        state_dim: int,
        action_dim: int,
        mu: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""Batched backward pass: Riccati recursion for all restarts.

        :param states: Shape ``(B, H+1, S)``.
        :param actions: Shape ``(B, H, A)``.
        :param state_dim: State dimensionality S.
        :param action_dim: Action dimensionality A.
        :param mu: Per-restart regularization, shape ``(B,)``.
        :returns: Tuple of (ok, k_gains, K_gains) where
            ok has shape ``(B,)``,
            k_gains has shape ``(B, H, A)``,
            K_gains has shape ``(B, H, A, S)``.
        """
        gamma = self._config.gamma
        B = states.shape[0]
        H = actions.shape[1]

        # Terminal value function
        v_x = np.zeros((B, state_dim))  # (B, S)
        v_xx = np.zeros((B, state_dim, state_dim))  # (B, S, S)

        if self._terminal_cost_fn is not None:
            # Terminal cost is state-dependent; evaluate per restart
            for b in range(B):
                try:
                    _tc_val, vx_b, vxx_b = self._terminal_cost_fn(
                        states[b, -1],
                    )
                    v_x[b] = vx_b
                    v_xx[b] = vxx_b
                except np.linalg.LinAlgError:
                    pass  # Leave as zeros

        k_gains = np.zeros((B, H, action_dim))  # (B, H, A)
        big_k_gains = np.zeros(
            (B, H, action_dim, state_dim),
        )  # (B, H, A, S)
        ok = np.ones(B, dtype=bool)  # (B,)

        # Pre-compute cost derivatives (batched where possible)
        # For analytic derivatives, compute per-sample and stack
        cd_x = np.zeros((B, H, state_dim))  # (B, H, S)
        cd_u = np.zeros((B, H, action_dim))  # (B, H, A)
        cd_xx = np.zeros((B, H, state_dim, state_dim))  # (B, H, S, S)
        cd_uu = np.zeros((B, H, action_dim, action_dim))  # (B, H, A, A)
        cd_ux = np.zeros((B, H, action_dim, state_dim))  # (B, H, A, S)

        self._fill_cost_derivatives(
            states, actions, state_dim, action_dim,
            cd_x, cd_u, cd_xx, cd_uu, cd_ux,
        )

        # Regularization identity
        mu_eye = mu[:, None, None] * np.eye(action_dim)[None]  # (B, A, A)

        for t in range(H - 1, -1, -1):
            x_t = states[:, t]  # (B, S)
            u_t = actions[:, t]  # (B, A)

            # Batched Jacobians
            a_mat = self._batched_jac_state_fn(x_t, u_t)  # (B, S, S)
            b_mat = self._batched_jac_action_fn(x_t, u_t)  # (B, S, A)

            # Cost derivatives at this timestep
            c_x = cd_x[:, t]  # (B, S)
            c_u = cd_u[:, t]  # (B, A)
            c_xx = cd_xx[:, t]  # (B, S, S)
            c_uu = cd_uu[:, t]  # (B, A, A)
            c_ux = cd_ux[:, t]  # (B, A, S)

            # Q-function terms: batched matrix operations
            # a_mat.T: (B, S, S) -> transpose last two dims
            a_t = np.swapaxes(a_mat, -2, -1)  # (B, S, S)
            b_t = np.swapaxes(b_mat, -2, -1)  # (B, A, S)

            # q_x = c_x + gamma * A^T @ v_x
            q_x = c_x + gamma * np.einsum(
                "bij,bj->bi", a_t, v_x,
            )  # (B, S)

            # q_u = c_u + gamma * B^T @ v_x
            q_u = c_u + gamma * np.einsum(
                "bij,bj->bi", b_t, v_x,
            )  # (B, A)

            # q_xx = c_xx + gamma * A^T @ V_xx @ A
            q_xx = c_xx + gamma * np.einsum(
                "bji,bjk,bkl->bil", a_mat, v_xx, a_mat,
            )  # (B, S, S)

            # q_ux = c_ux + gamma * B^T @ V_xx @ A
            q_ux = c_ux + gamma * np.einsum(
                "bji,bjk,bkl->bil", b_mat, v_xx, a_mat,
            )  # (B, A, S)

            # q_uu = c_uu + gamma * B^T @ V_xx @ B
            q_uu = c_uu + gamma * np.einsum(
                "bji,bjk,bkl->bil", b_mat, v_xx, b_mat,
            )  # (B, A, A)

            # Regularize
            q_uu_reg = q_uu + mu_eye  # (B, A, A)

            # Batched Cholesky (safe: handles per-element failures)
            cho, cho_ok = _batched_cholesky_safe(q_uu_reg)

            # Mark failed restarts
            ok &= cho_ok

            # For failed restarts, set cho to identity to avoid
            # downstream NaN (gains will be ignored anyway)
            failed_mask = ~cho_ok
            if failed_mask.any():
                cho[failed_mask] = np.eye(action_dim)

            # cho @ cho.T
            cho_full = np.einsum(
                "bij,bkj->bik", cho, cho,
            )  # (B, A, A)

            # k = -solve(cho@cho.T, q_u)
            k_t = -np.linalg.solve(
                cho_full, q_u[..., None],
            ).squeeze(-1)  # (B, A)

            # K = -solve(cho@cho.T, q_ux)
            big_k_t = -np.linalg.solve(
                cho_full, q_ux,
            )  # (B, A, S)

            k_gains[:, t] = k_t
            big_k_gains[:, t] = big_k_t

            # Update value function
            # v_x = q_x + K^T @ q_u
            v_x = q_x + np.einsum(
                "bji,bj->bi", big_k_t, q_u,
            )  # (B, S)

            # v_xx = q_xx + K^T @ q_ux
            v_xx = q_xx + np.einsum(
                "bji,bjk->bik", big_k_t, q_ux,
            )  # (B, S, S)

            # Symmetrize
            v_xx = 0.5 * (v_xx + np.swapaxes(v_xx, -2, -1))

        return ok, k_gains, big_k_gains

    def _fill_cost_derivatives(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        state_dim: int,
        action_dim: int,
        cd_x: np.ndarray,
        cd_u: np.ndarray,
        cd_xx: np.ndarray,
        cd_uu: np.ndarray,
        cd_ux: np.ndarray,
    ) -> None:
        """Fill pre-allocated cost derivative arrays.

        Uses analytic reward derivatives if available, then tries
        batched finite differences (if ``batched_reward_fn`` is set),
        then falls back to per-sample scalar finite differences.

        :param states: Shape ``(B, H+1, S)``.
        :param actions: Shape ``(B, H, A)``.
        :param cd_x: Output, shape ``(B, H, S)``.
        :param cd_u: Output, shape ``(B, H, A)``.
        :param cd_xx: Output, shape ``(B, H, S, S)``.
        :param cd_uu: Output, shape ``(B, H, A, A)``.
        :param cd_ux: Output, shape ``(B, H, A, S)``.
        """
        B, H, _A = actions.shape

        if self._reward_derivatives_fn is not None:
            for b in range(B):
                for t in range(H):
                    r_x, r_u, r_xx, r_uu, r_ux = (
                        self._reward_derivatives_fn(
                            states[b, t], actions[b, t],
                        )
                    )
                    cd_x[b, t] = -r_x
                    cd_u[b, t] = -r_u
                    cd_xx[b, t] = -r_xx
                    cd_uu[b, t] = -r_uu
                    cd_ux[b, t] = -r_ux
        elif self._batched_reward_fn is not None:
            self._batched_fd_cost_derivatives(
                states, actions, state_dim, action_dim,
                cd_x, cd_u, cd_xx, cd_uu, cd_ux,
            )
        else:
            eps = 1e-5
            for b in range(B):
                for t in range(H):
                    self._fd_cost_derivatives(
                        states[b, t], actions[b, t],
                        state_dim, action_dim, eps,
                        cd_x[b, t], cd_u[b, t],
                        cd_xx[b, t], cd_uu[b, t], cd_ux[b, t],
                    )

    def _fd_cost_derivatives(
        self,
        state: np.ndarray,
        action: np.ndarray,
        state_dim: int,
        action_dim: int,
        eps: float,
        c_x: np.ndarray,
        c_u: np.ndarray,
        c_xx: np.ndarray,
        c_uu: np.ndarray,
        c_ux: np.ndarray,
    ) -> None:
        """Fill cost derivatives via finite differences.

        Writes directly into pre-allocated output arrays.
        """
        r0 = self._reward_fn(state, action)

        for i in range(state_dim):
            s_plus = state.copy()
            s_minus = state.copy()
            s_plus[i] += eps
            s_minus[i] -= eps
            r_plus = self._reward_fn(s_plus, action)
            r_minus = self._reward_fn(s_minus, action)
            c_x[i] = -(r_plus - r_minus) / (2 * eps)
            c_xx[i, i] = -(r_plus - 2 * r0 + r_minus) / (eps ** 2)

        for i in range(action_dim):
            a_plus = action.copy()
            a_minus = action.copy()
            a_plus[i] += eps
            a_minus[i] -= eps
            r_plus = self._reward_fn(state, a_plus)
            r_minus = self._reward_fn(state, a_minus)
            c_u[i] = -(r_plus - r_minus) / (2 * eps)
            c_uu[i, i] = -(r_plus - 2 * r0 + r_minus) / (eps ** 2)

        for i in range(action_dim):
            for j in range(state_dim):
                s_plus = state.copy()
                s_minus = state.copy()
                s_plus[j] += eps
                s_minus[j] -= eps
                a_cur = action.copy()
                a_cur[i] += eps
                r_pp = self._reward_fn(s_plus, a_cur)
                r_pm = self._reward_fn(s_minus, a_cur)
                a_cur[i] -= 2 * eps
                r_mp = self._reward_fn(s_plus, a_cur)
                r_mm = self._reward_fn(s_minus, a_cur)
                c_ux[i, j] = -(r_pp - r_pm - r_mp + r_mm) / (
                    4 * eps ** 2
                )

    def _batched_fd_cost_derivatives(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        state_dim: int,
        action_dim: int,
        cd_x: np.ndarray,
        cd_u: np.ndarray,
        cd_xx: np.ndarray,
        cd_uu: np.ndarray,
        cd_ux: np.ndarray,
    ) -> None:
        """Vectorized finite-difference cost derivatives over batch.

        For each timestep t, evaluates the batched reward function
        with perturbed states/actions for all B restarts at once.
        This replaces B*H*(2S+2A+A*S*2+1) scalar reward calls with
        H*(2S+2A+A*S*2+1) batched calls, each processing B elements.

        :param states: Shape ``(B, H+1, S)``.
        :param actions: Shape ``(B, H, A)``.
        :param cd_x: Output, shape ``(B, H, S)``.
        :param cd_u: Output, shape ``(B, H, A)``.
        :param cd_xx: Output, shape ``(B, H, S, S)``.
        :param cd_uu: Output, shape ``(B, H, A, A)``.
        :param cd_ux: Output, shape ``(B, H, A, S)``.
        """
        assert self._batched_reward_fn is not None
        eps = 1e-5
        inv_2eps = 1.0 / (2.0 * eps)
        inv_eps2 = 1.0 / (eps * eps)
        inv_4eps2 = 1.0 / (4.0 * eps * eps)
        B, H, _A = actions.shape
        br = self._batched_reward_fn

        for t in range(H):
            s_t = states[:, t]  # (B, S)
            a_t = actions[:, t]  # (B, A)
            r0 = br(s_t, a_t)  # (B,)

            # State derivatives
            for i in range(state_dim):
                s_plus = s_t.copy()
                s_minus = s_t.copy()
                s_plus[:, i] += eps
                s_minus[:, i] -= eps
                r_plus = br(s_plus, a_t)  # (B,)
                r_minus = br(s_minus, a_t)  # (B,)
                cd_x[:, t, i] = -(r_plus - r_minus) * inv_2eps
                cd_xx[:, t, i, i] = -(
                    r_plus - 2.0 * r0 + r_minus
                ) * inv_eps2

            # Action derivatives
            for i in range(action_dim):
                a_plus = a_t.copy()
                a_minus = a_t.copy()
                a_plus[:, i] += eps
                a_minus[:, i] -= eps
                r_plus = br(s_t, a_plus)  # (B,)
                r_minus = br(s_t, a_minus)  # (B,)
                cd_u[:, t, i] = -(r_plus - r_minus) * inv_2eps
                cd_uu[:, t, i, i] = -(
                    r_plus - 2.0 * r0 + r_minus
                ) * inv_eps2

            # Cross derivatives d^2c / du_i ds_j
            for i in range(action_dim):
                for j in range(state_dim):
                    s_plus = s_t.copy()
                    s_minus = s_t.copy()
                    s_plus[:, j] += eps
                    s_minus[:, j] -= eps

                    a_up = a_t.copy()
                    a_up[:, i] += eps
                    r_pp = br(s_plus, a_up)  # (B,)
                    r_pm = br(s_minus, a_up)  # (B,)

                    a_down = a_t.copy()
                    a_down[:, i] -= eps
                    r_mp = br(s_plus, a_down)  # (B,)
                    r_mm = br(s_minus, a_down)  # (B,)

                    cd_ux[:, t, i, j] = -(
                        r_pp - r_pm - r_mp + r_mm
                    ) * inv_4eps2

    def _batched_line_search(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        k_gains: np.ndarray,
        big_k_gains: np.ndarray,
        current_costs: np.ndarray,
        state_dim: int,
        action_dim: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Batched forward pass with backtracking line search.

        :param states: Shape ``(B, H+1, S)``.
        :param actions: Shape ``(B, H, A)``.
        :param k_gains: Shape ``(B, H, A)``.
        :param big_k_gains: Shape ``(B, H, A, S)``.
        :param current_costs: Shape ``(B,)``.
        :param state_dim: State dimensionality.
        :param action_dim: Action dimensionality.
        :returns: Tuple (new_states, new_actions, new_costs).
        """
        cfg = self._config
        B, H, A = actions.shape

        best_states = states.copy()
        best_actions = actions.copy()
        best_costs = current_costs.copy()

        # Track which restarts still need improvement
        needs_search = np.ones(B, dtype=bool)

        alpha = 1.0
        while alpha >= cfg.alpha_min:
            new_states = np.zeros_like(states)  # (B, H+1, S)
            new_actions = np.zeros_like(actions)  # (B, H, A)
            new_states[:, 0] = states[:, 0]

            for t in range(H):
                # dx = new_states[:, t] - states[:, t]  # (B, S)
                dx = new_states[:, t] - states[:, t]

                # new_action = action + alpha * k + K @ dx
                new_actions[:, t] = (
                    actions[:, t]
                    + alpha * k_gains[:, t]
                    + np.einsum(
                        "bas,bs->ba", big_k_gains[:, t], dx,
                    )
                )

                # Bound actions
                if cfg.use_tanh_squash:
                    new_actions[:, t] = cfg.max_action * np.tanh(
                        new_actions[:, t] / cfg.max_action,
                    )
                else:
                    np.clip(
                        new_actions[:, t],
                        -cfg.max_action,
                        cfg.max_action,
                        out=new_actions[:, t],
                    )

                # Batched dynamics
                new_states[:, t + 1] = self._batched_dynamics_fn(
                    new_states[:, t], new_actions[:, t],
                )

            # Batched cost evaluation
            new_costs = self._batched_total_cost(
                new_states, new_actions,
            )  # (B,)

            # Accept improvements
            improved = needs_search & (new_costs < best_costs)
            if improved.any():
                best_states[improved] = new_states[improved]
                best_actions[improved] = new_actions[improved]
                best_costs[improved] = new_costs[improved]
                needs_search[improved] = False

            if not needs_search.any():
                break

            alpha *= cfg.alpha_decay

        return best_states, best_actions, best_costs


def build_batched_jacobian_fns(
    dynamics_expressions: dict[int, object],
    state_names: list[str],
    action_names: list[str],
    state_dim: int,
    env_params: dict[str, float] | None,
    calibration_coefficients: dict[int, tuple[float, float]] | None = None,
) -> tuple[
    Callable[[np.ndarray, np.ndarray], np.ndarray],
    Callable[[np.ndarray, np.ndarray], np.ndarray],
] | None:
    r"""Build batched Jacobian functions for ``BatchedNumpyILQRSolver``.

    Each Jacobian element is a sympy-lambdified function with numpy
    backend, so it naturally broadcasts over the batch dimension.

    :param dynamics_expressions: Per-dimension symbolic dynamics.
        Values must have a ``.sympy_expr`` attribute.
    :param state_names: Canonical state variable names.
    :param action_names: Action variable names.
    :param state_dim: Number of state dimensions.
    :param env_params: Environment parameter values to substitute.
    :param calibration_coefficients: Per-dimension ``(alpha, beta)``
        calibration coefficients.
    :returns: Tuple of (jac_state_fn, jac_action_fn) where
        ``jac_state_fn(states, actions)`` returns ``(B, S, S)`` and
        ``jac_action_fn(states, actions)`` returns ``(B, S, A)``.
        Returns ``None`` if construction fails.
    """
    try:
        import sympy
    except ImportError:
        return None

    var_names = list(state_names) + list(action_names)
    symbols = [sympy.Symbol(n) for n in var_names]
    action_dim = len(action_names)

    # Build per-element lambdified Jacobian functions
    jac_state_elements: dict[
        tuple[int, int], Callable[..., object]
    ] = {}
    jac_action_elements: dict[
        tuple[int, int], Callable[..., object]
    ] = {}

    for dim_idx, expr_obj in dynamics_expressions.items():
        sympy_expr = expr_obj.sympy_expr  # type: ignore[union-attr]
        if env_params:
            subs = {sympy.Symbol(k): v for k, v in env_params.items()}
            sympy_expr = sympy_expr.subs(subs)

        # State Jacobian elements: d(delta_dim) / d(state_j)
        for j in range(state_dim):
            deriv = sympy.diff(sympy_expr, symbols[j])
            fn = sympy.lambdify(symbols, deriv, modules=["numpy"])
            jac_state_elements[(dim_idx, j)] = fn

        # Action Jacobian elements: d(delta_dim) / d(action_j)
        for j in range(action_dim):
            deriv = sympy.diff(sympy_expr, symbols[state_dim + j])
            fn = sympy.lambdify(symbols, deriv, modules=["numpy"])
            jac_action_elements[(dim_idx, j)] = fn

    _cal = calibration_coefficients

    def batched_jac_state_fn(
        states: np.ndarray,
        actions: np.ndarray,
    ) -> np.ndarray:
        """Compute batched state Jacobian: (B, S), (B, A) -> (B, S, S)."""
        B = states.shape[0]
        jac = np.zeros((B, state_dim, state_dim))  # (B, S, S)
        # Identity for next_state = state + delta
        for i in range(state_dim):
            jac[:, i, i] = 1.0

        n_state = states.shape[1]
        cols: list[np.ndarray] = [states[:, i] for i in range(n_state)]
        cols += [actions[:, i] for i in range(actions.shape[1])]

        for (di, j), fn in jac_state_elements.items():
            val = fn(*cols)  # (B,) or scalar
            if isinstance(val, (int, float)):
                val = np.full(B, float(val))
            if _cal is not None and di in _cal:
                val = val * _cal[di][0]
            jac[:, di, j] += val

        return jac

    def batched_jac_action_fn(
        states: np.ndarray,
        actions: np.ndarray,
    ) -> np.ndarray:
        """Compute batched action Jacobian: (B, S), (B, A) -> (B, S, A)."""
        B = states.shape[0]
        jac = np.zeros((B, state_dim, action_dim))  # (B, S, A)

        n_state = states.shape[1]
        cols: list[np.ndarray] = [states[:, i] for i in range(n_state)]
        cols += [actions[:, i] for i in range(actions.shape[1])]

        for (di, j), fn in jac_action_elements.items():
            val = fn(*cols)  # (B,) or scalar
            if isinstance(val, (int, float)):
                val = np.full(B, float(val))
            if _cal is not None and di in _cal:
                val = val * _cal[di][0]
            jac[:, di, j] += val

        return jac

    return batched_jac_state_fn, batched_jac_action_fn
