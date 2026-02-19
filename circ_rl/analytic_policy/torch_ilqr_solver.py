r"""GPU-accelerated batched iLQR solver using PyTorch.

Solves the same trajectory optimization problem as the numpy-based
``ILQRSolver``, but batches across multiple restarts (and optionally
environments) using PyTorch tensor operations. This enables GPU
acceleration for the matrix operations in the backward and forward passes.

The batch dimension ``B`` represents independent iLQR instances
(typically one per random restart). All forward/backward/line-search
operations are fully vectorized over ``B``.

Key differences from numpy solver:

- Dynamics and reward are PyTorch callables (from ``fast_dynamics.py``).
- All intermediate tensors are ``(B, ...)`` shaped.
- Cholesky + solve use ``torch.linalg`` batched operations.
- Line search evaluates all alpha values simultaneously.

Falls back gracefully: if torch dynamics compilation fails, returns
``None`` so the caller uses the numpy solver.

See ``CIRC-RL_Framework.md`` Section 3.6.2 (Nonlinear Known Dynamics).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from loguru import logger

if TYPE_CHECKING:
    from collections.abc import Callable

    from circ_rl.analytic_policy.ilqr_solver import ILQRConfig, ILQRSolution


@dataclass(frozen=True)
class TorchILQRConfig:
    """Configuration for the batched PyTorch iLQR solver.

    Mirrors ``ILQRConfig`` but adds batch-specific settings.

    :param n_batch: Number of parallel restart trajectories.
    :param device: PyTorch device (``"cuda"``, ``"cpu"``, or ``"auto"``).
    :param dtype: Tensor dtype (default ``torch.float64`` for numerical
        stability in Riccati recursion).
    :param n_alpha: Number of line search step sizes to try
        simultaneously.
    """

    n_batch: int = 9
    device: str = "auto"
    dtype: torch.dtype = torch.float64
    n_alpha: int = 8


def _resolve_device(device: str) -> torch.device:
    """Resolve ``"auto"`` to cuda/cpu."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


class TorchILQRSolver:
    r"""Batched iLQR solver using PyTorch.

    Plans a trajectory by running ``B`` restarts in parallel as a single
    batched computation. Each restart uses a different random action
    initialization; the best solution (highest reward) is returned.

    Dynamics and reward must be provided as PyTorch-compatible callables
    (typically compiled from sympy via ``fast_dynamics.compile_torch_fn``).

    :param numpy_config: The standard ``ILQRConfig`` (horizon, gamma,
        max_action, etc.).
    :param torch_config: Batch and device settings.
    :param dynamics_fns: Dict mapping ``dim_idx -> callable``. Each
        callable takes ``(s0, s1, ..., a0, ...)`` as individual tensors
        and returns a ``(B,)`` delta tensor.
    :param reward_fn: Callable ``(state_tensor, action_tensor) -> (B,)``
        reward values. ``state_tensor`` has shape ``(B, state_dim)``,
        ``action_tensor`` has shape ``(B, action_dim)``.
    :param state_dim: State dimensionality.
    :param action_dim: Action dimensionality.
    :param angular_dims: Dimensions that are angular coordinates (wrapped
        to ``[-pi, pi]`` after dynamics update).
    """

    def __init__(
        self,
        numpy_config: ILQRConfig,
        torch_config: TorchILQRConfig,
        dynamics_fns: dict[int, Callable[..., Any]],
        reward_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        state_dim: int,
        action_dim: int,
        angular_dims: tuple[int, ...] = (),
    ) -> None:
        self._cfg = numpy_config
        self._tcfg = torch_config
        self._dynamics_fns = dynamics_fns
        self._reward_fn = reward_fn
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._angular_dims = angular_dims

        self._device = _resolve_device(torch_config.device)
        self._dtype = torch_config.dtype

    def plan(
        self,
        initial_state: np.ndarray,
        warm_start_actions: np.ndarray | None = None,
    ) -> ILQRSolution:
        """Optimize a trajectory using batched iLQR.

        Runs ``n_batch`` restarts simultaneously and returns the best.

        :param initial_state: Shape ``(state_dim,)``.
        :param warm_start_actions: Optional shape ``(horizon, action_dim)``.
            Used as the first restart; remaining restarts are random.
        :returns: Best ``ILQRSolution`` across all restarts.
        """
        from circ_rl.analytic_policy.ilqr_solver import ILQRSolution

        cfg = self._cfg
        tcfg = self._tcfg
        H = cfg.horizon
        B = tcfg.n_batch
        S = self._state_dim
        A = self._action_dim
        dev = self._device
        dt = self._dtype

        # Initialize batch of action sequences: (B, H, A)
        actions = torch.zeros(B, H, A, device=dev, dtype=dt)

        if warm_start_actions is not None:
            actions[0] = torch.tensor(
                warm_start_actions, device=dev, dtype=dt,
            )

        # Random restarts for remaining batch elements
        if B > 1:
            noise = torch.randn(
                B - 1, H, A, device=dev, dtype=dt,
            ) * (cfg.restart_scale * cfg.max_action)
            actions[1:] = noise.clamp(-cfg.max_action, cfg.max_action)

        # Initial state: (B, S) - same for all restarts
        x0 = torch.tensor(
            initial_state, device=dev, dtype=dt,
        ).unsqueeze(0).expand(B, -1)  # (B, S)

        # Initial forward pass
        states = self._rollout(x0, actions)  # (B, H+1, S)
        costs = self._total_cost(states, actions)  # (B,)

        mu = cfg.mu_init * torch.ones(B, device=dev, dtype=dt)  # (B,)
        converged = torch.zeros(B, device=dev, dtype=torch.bool)

        for iteration in range(cfg.max_iterations):
            # Backward pass for all batches
            ok, k_gains, big_k_gains = self._backward_pass(
                states, actions, mu,
            )
            # ok: (B,), k: (B, H, A), K: (B, H, A, S)

            # Increase mu where backward pass failed
            failed = ~ok
            mu = torch.where(
                failed,
                (mu * cfg.mu_factor).clamp(max=cfg.mu_max),
                mu,
            )

            # Skip line search for failed batches (use existing trajectory)
            if ok.any():
                new_states, new_actions, new_costs = self._line_search(
                    states, actions, k_gains, big_k_gains, costs,
                )  # each (B, ...)

                improved = new_costs < costs  # (B,)
                improved = improved & ok  # only update successful batches

                # Compute relative improvement
                rel_imp = (
                    (costs - new_costs).abs()
                    / costs.abs().clamp(min=1e-10)
                )  # (B,)

                # Accept improvements
                states = torch.where(
                    improved.unsqueeze(1).unsqueeze(2),
                    new_states,
                    states,
                )
                actions = torch.where(
                    improved.unsqueeze(1).unsqueeze(2),
                    new_actions,
                    actions,
                )
                costs = torch.where(improved, new_costs, costs)

                # Adjust regularization
                mu = torch.where(
                    improved,
                    (mu / cfg.mu_factor).clamp(min=cfg.mu_min),
                    (mu * cfg.mu_factor).clamp(max=cfg.mu_max),
                )

                # Check convergence
                converged = converged | (improved & (rel_imp < cfg.convergence_tol))

            if converged.all():
                break

        # Select best restart
        rewards = -costs  # (B,)
        best_idx = rewards.argmax().item()

        best_states_np = states[best_idx].cpu().numpy()  # (H+1, S)
        best_actions_np = actions[best_idx].cpu().numpy()  # (H, A)

        # Extract feedback gains for the best restart
        feedback_list: list[np.ndarray] = []
        feedforward_list: list[np.ndarray] = []
        for t in range(H):
            feedback_list.append(
                big_k_gains[best_idx, t].cpu().numpy(),
            )
            feedforward_list.append(
                k_gains[best_idx, t].cpu().numpy(),
            )

        total_reward = float(rewards[best_idx].item())

        logger.info(
            "Torch iLQR: best of {} restarts, reward={:.2f}, "
            "device={}, converged={}",
            B,
            total_reward,
            self._device,
            bool(converged[best_idx].item()),
        )

        return ILQRSolution(
            nominal_states=best_states_np,
            nominal_actions=best_actions_np,
            feedback_gains=feedback_list,
            feedforward_gains=feedforward_list,
            total_reward=total_reward,
            converged=bool(converged[best_idx].item()),
            n_iterations=iteration + 1,
        )

    def _dynamics_step(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Apply one dynamics step for all batches.

        :param state: Shape ``(B, S)``.
        :param action: Shape ``(B, A)``.
        :returns: Next state, shape ``(B, S)``.
        """
        S = self._state_dim
        next_state = state.clone()  # (B, S)

        # Build variable list: [s0, s1, ..., a0, a1, ...]
        # Each is shape (B,)
        var_list = []
        for i in range(S):
            var_list.append(state[:, i])
        for i in range(self._action_dim):
            var_list.append(action[:, i])

        for dim_idx, fn in self._dynamics_fns.items():
            delta = fn(*var_list)  # (B,)
            next_state[:, dim_idx] = next_state[:, dim_idx] + delta

        # Wrap angular dims
        for d in self._angular_dims:
            next_state[:, d] = torch.atan2(
                torch.sin(next_state[:, d]),
                torch.cos(next_state[:, d]),
            )

        return next_state

    def _rollout(
        self,
        x0: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Forward-simulate all batch trajectories.

        :param x0: Shape ``(B, S)``.
        :param actions: Shape ``(B, H, A)``.
        :returns: States, shape ``(B, H+1, S)``.
        """
        B = x0.shape[0]
        H = actions.shape[1]
        S = self._state_dim

        states = torch.zeros(
            B, H + 1, S, device=x0.device, dtype=x0.dtype,
        )
        states[:, 0] = x0

        for t in range(H):
            states[:, t + 1] = self._dynamics_step(
                states[:, t], actions[:, t],
            )

        return states

    def _total_cost(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Compute total discounted cost for each batch.

        :param states: Shape ``(B, H+1, S)``.
        :param actions: Shape ``(B, H, A)``.
        :returns: Shape ``(B,)``.
        """
        H = actions.shape[1]
        gamma = self._cfg.gamma

        # Discount factors: (H,)
        gammas = torch.tensor(
            [gamma ** t for t in range(H)],
            device=states.device, dtype=states.dtype,
        )

        total = torch.zeros(
            states.shape[0], device=states.device, dtype=states.dtype,
        )  # (B,)

        for t in range(H):
            r_t = self._reward_fn(
                states[:, t], actions[:, t],
            )  # (B,)
            total = total - gammas[t] * r_t

        return total

    def _dynamics_jacobians(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Compute dynamics Jacobians via autograd.

        Uses ``torch.autograd.functional.jacobian`` for automatic
        differentiation of the dynamics function.

        :param state: Shape ``(B, S)``.
        :param action: Shape ``(B, A)``.
        :returns: Tuple ``(A_mat, B_mat)`` where
            ``A_mat`` has shape ``(B, S, S)`` and
            ``B_mat`` has shape ``(B, S, A)``.
        """
        S = self._state_dim
        A = self._action_dim
        B = state.shape[0]
        dev = state.device
        dt = state.dtype

        # Finite differences (more reliable than autograd for lambdified fns)
        eps = 1e-5

        # A_mat: df/dx, shape (B, S, S)
        a_mat = torch.zeros(B, S, S, device=dev, dtype=dt)
        base_next = self._dynamics_step(state, action)  # (B, S)

        for j in range(S):
            s_plus = state.clone()
            s_plus[:, j] += eps
            next_plus = self._dynamics_step(s_plus, action)
            a_mat[:, :, j] = (next_plus - base_next) / eps

        # B_mat: df/du, shape (B, S, A)
        b_mat = torch.zeros(B, S, A, device=dev, dtype=dt)
        for j in range(A):
            a_plus = action.clone()
            a_plus[:, j] += eps
            next_plus = self._dynamics_step(state, a_plus)
            b_mat[:, :, j] = (next_plus - base_next) / eps

        return a_mat, b_mat

    def _reward_derivatives(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[
        torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor, torch.Tensor,
    ]:
        r"""Compute reward derivatives via finite differences.

        :param state: Shape ``(B, S)``.
        :param action: Shape ``(B, A)``.
        :returns: ``(r_x, r_u, r_xx, r_uu, r_ux)`` with shapes
            ``(B, S)``, ``(B, A)``, ``(B, S, S)``,
            ``(B, A, A)``, ``(B, A, S)``.
        """
        S = self._state_dim
        A = self._action_dim
        B = state.shape[0]
        dev = state.device
        dt = state.dtype
        eps = 1e-5

        r0 = self._reward_fn(state, action)  # (B,)

        # r_x: (B, S)
        r_x = torch.zeros(B, S, device=dev, dtype=dt)
        for i in range(S):
            s_plus = state.clone()
            s_minus = state.clone()
            s_plus[:, i] += eps
            s_minus[:, i] -= eps
            r_x[:, i] = (
                self._reward_fn(s_plus, action)
                - self._reward_fn(s_minus, action)
            ) / (2 * eps)

        # r_u: (B, A)
        r_u = torch.zeros(B, A, device=dev, dtype=dt)
        for i in range(A):
            a_plus = action.clone()
            a_minus = action.clone()
            a_plus[:, i] += eps
            a_minus[:, i] -= eps
            r_u[:, i] = (
                self._reward_fn(state, a_plus)
                - self._reward_fn(state, a_minus)
            ) / (2 * eps)

        # r_xx: (B, S, S) - diagonal approximation for speed
        r_xx = torch.zeros(B, S, S, device=dev, dtype=dt)
        for i in range(S):
            s_plus = state.clone()
            s_minus = state.clone()
            s_plus[:, i] += eps
            s_minus[:, i] -= eps
            r_xx[:, i, i] = (
                self._reward_fn(s_plus, action)
                - 2 * r0
                + self._reward_fn(s_minus, action)
            ) / (eps * eps)

        # r_uu: (B, A, A) - diagonal approximation
        r_uu = torch.zeros(B, A, A, device=dev, dtype=dt)
        for i in range(A):
            a_plus = action.clone()
            a_minus = action.clone()
            a_plus[:, i] += eps
            a_minus[:, i] -= eps
            r_uu[:, i, i] = (
                self._reward_fn(state, a_plus)
                - 2 * r0
                + self._reward_fn(state, a_minus)
            ) / (eps * eps)

        # r_ux: (B, A, S) - cross terms via finite differences
        r_ux = torch.zeros(B, A, S, device=dev, dtype=dt)
        for i in range(A):
            for j in range(S):
                s_plus = state.clone()
                a_plus = action.clone()
                s_plus[:, j] += eps
                a_plus[:, i] += eps
                r_pp = self._reward_fn(s_plus, a_plus)

                s_minus = state.clone()
                a_minus = action.clone()
                s_minus[:, j] -= eps
                a_minus[:, i] -= eps
                r_mm = self._reward_fn(s_minus, a_minus)

                s_pm = state.clone()
                a_pm = action.clone()
                s_pm[:, j] += eps
                a_pm[:, i] -= eps
                r_pm = self._reward_fn(s_pm, a_pm)

                s_mp = state.clone()
                a_mp = action.clone()
                s_mp[:, j] -= eps
                a_mp[:, i] += eps
                r_mp = self._reward_fn(s_mp, a_mp)

                r_ux[:, i, j] = (r_pp - r_pm - r_mp + r_mm) / (4 * eps * eps)

        return r_x, r_u, r_xx, r_uu, r_ux

    def _backward_pass(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        mu: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Batched backward pass: compute gains for all restarts.

        :param states: Shape ``(B, H+1, S)``.
        :param actions: Shape ``(B, H, A)``.
        :param mu: Regularization, shape ``(B,)``.
        :returns: Tuple ``(ok, k_gains, K_gains)`` where
            ``ok`` has shape ``(B,)``,
            ``k_gains`` has shape ``(B, H, A)``,
            ``K_gains`` has shape ``(B, H, A, S)``.
        """
        cfg = self._cfg
        B = states.shape[0]
        H = actions.shape[1]
        S = self._state_dim
        A = self._action_dim
        dev = states.device
        dt = states.dtype
        gamma = cfg.gamma

        k_gains = torch.zeros(B, H, A, device=dev, dtype=dt)
        big_k_gains = torch.zeros(B, H, A, S, device=dev, dtype=dt)
        ok = torch.ones(B, device=dev, dtype=torch.bool)

        # Terminal value function: zeros (no terminal cost in batched version)
        v_x = torch.zeros(B, S, device=dev, dtype=dt)  # (B, S)
        v_xx = torch.zeros(B, S, S, device=dev, dtype=dt)  # (B, S, S)

        for t in range(H - 1, -1, -1):
            x_t = states[:, t]  # (B, S)
            u_t = actions[:, t]  # (B, A)

            # Dynamics Jacobians
            a_mat, b_mat = self._dynamics_jacobians(x_t, u_t)
            # a_mat: (B, S, S), b_mat: (B, S, A)

            # Reward derivatives -> cost derivatives (negate)
            r_x, r_u, r_xx, r_uu, r_ux = self._reward_derivatives(x_t, u_t)
            c_x = -r_x  # (B, S)
            c_u = -r_u  # (B, A)
            c_xx = -r_xx  # (B, S, S)
            c_uu = -r_uu  # (B, A, A)
            c_ux = -r_ux  # (B, A, S)

            # Q-function terms (batched matmul)
            # q_x = c_x + gamma * A^T @ v_x
            q_x = c_x + gamma * torch.bmm(
                a_mat.transpose(1, 2), v_x.unsqueeze(2),
            ).squeeze(2)  # (B, S)

            # q_u = c_u + gamma * B^T @ v_x
            q_u = c_u + gamma * torch.bmm(
                b_mat.transpose(1, 2), v_x.unsqueeze(2),
            ).squeeze(2)  # (B, A)

            # q_xx = c_xx + gamma * A^T @ V_xx @ A
            q_xx = c_xx + gamma * torch.bmm(
                a_mat.transpose(1, 2),
                torch.bmm(v_xx, a_mat),
            )  # (B, S, S)

            # q_ux = c_ux + gamma * B^T @ V_xx @ A
            q_ux = c_ux + gamma * torch.bmm(
                b_mat.transpose(1, 2),
                torch.bmm(v_xx, a_mat),
            )  # (B, A, S)

            # q_uu = c_uu + gamma * B^T @ V_xx @ B
            q_uu = c_uu + gamma * torch.bmm(
                b_mat.transpose(1, 2),
                torch.bmm(v_xx, b_mat),
            )  # (B, A, A)

            # Regularize: Q_uu + mu * I
            mu_diag = mu.unsqueeze(1).unsqueeze(2) * torch.eye(
                A, device=dev, dtype=dt,
            ).unsqueeze(0)  # (B, A, A)
            q_uu_reg = q_uu + mu_diag  # (B, A, A)

            # Cholesky factorization (batched)
            try:
                cho = torch.linalg.cholesky(q_uu_reg)  # (B, A, A)
            except torch.linalg.LinAlgError:
                # Some batches may fail; mark them
                ok[:] = False
                break

            # Solve for gains: k = -Q_uu^{-1} @ q_u, K = -Q_uu^{-1} @ Q_ux
            k_t = -torch.cholesky_solve(
                q_u.unsqueeze(2), cho,
            ).squeeze(2)  # (B, A)
            big_k_t = -torch.cholesky_solve(
                q_ux, cho,
            )  # (B, A, S)

            k_gains[:, t] = k_t
            big_k_gains[:, t] = big_k_t

            # Update value function
            # V_x = Q_x + K^T @ Q_u
            v_x = q_x + torch.bmm(
                big_k_t.transpose(1, 2), q_u.unsqueeze(2),
            ).squeeze(2)  # (B, S)

            # V_xx = Q_xx + K^T @ Q_ux
            v_xx = q_xx + torch.bmm(
                big_k_t.transpose(1, 2), q_ux,
            )  # (B, S, S)

            # Symmetrize
            v_xx = 0.5 * (v_xx + v_xx.transpose(1, 2))

        return ok, k_gains, big_k_gains

    def _line_search(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        k_gains: torch.Tensor,
        big_k_gains: torch.Tensor,
        current_costs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Batched line search with multiple alpha values.

        :param states: Shape ``(B, H+1, S)``.
        :param actions: Shape ``(B, H, A)``.
        :param k_gains: Shape ``(B, H, A)``.
        :param big_k_gains: Shape ``(B, H, A, S)``.
        :param current_costs: Shape ``(B,)``.
        :returns: Tuple ``(best_states, best_actions, best_costs)``.
        """
        cfg = self._cfg
        H = actions.shape[1]
        B = states.shape[0]
        dev = states.device
        dt = states.dtype

        best_states = states.clone()
        best_actions = actions.clone()
        best_costs = current_costs.clone()

        # Generate alpha schedule
        n_alpha = self._tcfg.n_alpha
        alphas = torch.tensor(
            [cfg.alpha_decay ** i for i in range(n_alpha)],
            device=dev, dtype=dt,
        )  # (n_alpha,)

        for alpha_idx in range(n_alpha):
            alpha = alphas[alpha_idx]

            new_states = torch.zeros_like(states)
            new_actions = torch.zeros_like(actions)
            new_states[:, 0] = states[:, 0]

            for t in range(H):
                dx = new_states[:, t] - states[:, t]  # (B, S)
                new_actions[:, t] = (
                    actions[:, t]
                    + alpha * k_gains[:, t]
                    + torch.bmm(
                        big_k_gains[:, t],
                        dx.unsqueeze(2),
                    ).squeeze(2)
                )

                # Bound actions (tanh squash)
                if cfg.use_tanh_squash:
                    new_actions[:, t] = cfg.max_action * torch.tanh(
                        new_actions[:, t] / cfg.max_action,
                    )
                else:
                    new_actions[:, t] = new_actions[:, t].clamp(
                        -cfg.max_action, cfg.max_action,
                    )

                new_states[:, t + 1] = self._dynamics_step(
                    new_states[:, t], new_actions[:, t],
                )

            new_costs = self._total_cost(new_states, new_actions)

            # Accept where improved
            improved = new_costs < best_costs  # (B,)
            if improved.any():
                mask3 = improved.unsqueeze(1).unsqueeze(2)
                best_states = torch.where(mask3, new_states, best_states)
                best_actions = torch.where(mask3, new_actions, best_actions)
                best_costs = torch.where(improved, new_costs, best_costs)

            # If all batches improved at this alpha, skip smaller alphas
            if improved.all():
                break

        return best_states, best_actions, best_costs


def build_torch_ilqr_solver(
    numpy_config: ILQRConfig,
    dynamics_expressions: dict[int, Any],
    state_names: list[str],
    action_names: list[str],
    state_dim: int,
    env_params: dict[str, float] | None,
    reward_fn_numpy: Any,
    angular_dims: tuple[int, ...] = (),
    device: str = "auto",
) -> TorchILQRSolver | None:
    """Build a TorchILQRSolver from symbolic expressions.

    Compiles dynamics expressions to PyTorch callables using
    ``fast_dynamics.build_torch_dynamics_fns``. The reward function
    is wrapped to operate on torch tensors.

    :returns: Solver instance, or None if compilation fails.
    """
    try:
        from circ_rl.analytic_policy.fast_dynamics import (
            build_torch_dynamics_fns,
        )

        torch_dyn_fns = build_torch_dynamics_fns(
            dynamics_expressions,
            state_names,
            action_names,
            state_dim,
            env_params,
        )
        if torch_dyn_fns is None:
            logger.debug("Torch dynamics compilation failed, skipping GPU iLQR")
            return None

    except Exception as exc:
        logger.debug("Failed to build torch dynamics: {}", exc)
        return None

    # Wrap numpy reward as torch-compatible
    np_reward = reward_fn_numpy

    def torch_reward_fn(
        state: torch.Tensor, action: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate reward for a batch of states/actions."""
        B = state.shape[0]
        rewards = torch.zeros(B, device=state.device, dtype=state.dtype)
        for i in range(B):
            s_np = state[i].detach().cpu().numpy()
            a_np = action[i].detach().cpu().numpy()
            rewards[i] = np_reward(s_np, a_np)
        return rewards

    tcfg = TorchILQRConfig(
        n_batch=max(1, numpy_config.n_random_restarts + 1),
        device=device,
    )

    return TorchILQRSolver(
        numpy_config=numpy_config,
        torch_config=tcfg,
        dynamics_fns=torch_dyn_fns,
        reward_fn=torch_reward_fn,
        state_dim=state_dim,
        action_dim=len(action_names),
        angular_dims=angular_dims,
    )
