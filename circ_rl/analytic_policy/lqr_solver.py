"""Linear-Quadratic Regulator solver via DARE.

Solves the Discrete Algebraic Riccati Equation to derive optimal
linear feedback gains for systems with linear dynamics and quadratic cost.

See ``CIRC-RL_Framework.md`` Section 3.6.1 (Linear Dynamics + Quadratic
Reward).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from loguru import logger
from scipy import linalg


@dataclass(frozen=True)
class LinearDynamics:
    r"""Linear dynamics model: :math:`s_{t+1} = A s_t + B a_t + c`.

    :param a_matrix: State transition matrix, shape ``(state_dim, state_dim)``.
    :param b_matrix: Control input matrix, shape ``(state_dim, action_dim)``.
    :param c_vector: Constant offset vector, shape ``(state_dim,)``.
        Defaults to zeros if not provided.
    """

    a_matrix: np.ndarray
    b_matrix: np.ndarray
    c_vector: np.ndarray | None = None

    @property
    def state_dim(self) -> int:
        """State dimensionality."""
        return int(self.a_matrix.shape[0])

    @property
    def action_dim(self) -> int:
        """Action dimensionality."""
        return int(self.b_matrix.shape[1])

    def __post_init__(self) -> None:
        assert self.b_matrix.shape[0] == self.a_matrix.shape[0], (
            f"B row count {self.b_matrix.shape[0]} != A row count "
            f"{self.a_matrix.shape[0]}"
        )


@dataclass(frozen=True)
class QuadraticCost:
    r"""Quadratic cost: :math:`c_t = s_t^T Q s_t + a_t^T R a_t`.

    :param q_matrix: State cost matrix, shape ``(state_dim, state_dim)``.
        Must be positive semi-definite.
    :param r_matrix: Action cost matrix, shape ``(action_dim, action_dim)``.
        Must be positive definite.
    """

    q_matrix: np.ndarray
    r_matrix: np.ndarray

    def __post_init__(self) -> None:
        assert self.q_matrix.shape[0] == self.q_matrix.shape[1], (
            f"Q must be square, got shape {self.q_matrix.shape}"
        )
        assert self.r_matrix.shape[0] == self.r_matrix.shape[1], (
            f"R must be square, got shape {self.r_matrix.shape}"
        )


@dataclass(frozen=True)
class LQRSolution:
    r"""Solution of the LQR problem.

    The optimal feedback law is: :math:`a^*_t = -K s_t`.

    :param k_gain: Feedback gain matrix, shape ``(action_dim, state_dim)``.
    :param p_matrix: Solution to the DARE, shape ``(state_dim, state_dim)``.
    :param is_stable: Whether the closed-loop system is stable
        (all eigenvalues of :math:`A - BK` inside the unit circle).
    """

    k_gain: np.ndarray
    p_matrix: np.ndarray
    is_stable: bool

    @property
    def state_dim(self) -> int:
        """State dimensionality."""
        return int(self.k_gain.shape[1])

    @property
    def action_dim(self) -> int:
        """Action dimensionality."""
        return int(self.k_gain.shape[0])


class LQRSolver:
    r"""Solve the discrete-time LQR problem via DARE.

    Given linear dynamics :math:`s_{t+1} = A s_t + B a_t` and quadratic
    cost :math:`c_t = s_t^T Q s_t + a_t^T R a_t`, the optimal policy is:

    .. math::

        a^*_t = -K \cdot s_t

    where :math:`K = (R + B^T P B)^{-1} B^T P A` and :math:`P` solves
    the DARE:

    .. math::

        P = Q + \gamma A^T P A - \gamma^2 A^T P B (R + \gamma B^T P B)^{-1} B^T P A

    See ``CIRC-RL_Framework.md`` Section 3.6.1.
    """

    def solve(
        self,
        dynamics: LinearDynamics,
        cost: QuadraticCost,
        gamma: float = 0.99,
    ) -> LQRSolution:
        """Solve the LQR problem for a single system.

        :param dynamics: Linear dynamics (A, B).
        :param cost: Quadratic cost (Q, R).
        :param gamma: Discount factor.
        :returns: LQRSolution with optimal gain and Riccati solution.
        :raises ValueError: If DARE has no solution (unstabilizable system).
        """
        a_mat = dynamics.a_matrix  # (n, n)
        b_mat = dynamics.b_matrix  # (n, m)
        q_mat = cost.q_matrix  # (n, n)
        r_mat = cost.r_matrix  # (m, m)

        if a_mat.shape[0] != a_mat.shape[1]:
            raise ValueError(
                f"A must be square for LQR, got shape {a_mat.shape}"
            )

        # Scale A by sqrt(gamma) and Q by 1 to incorporate discount
        # DARE: P = Q + gamma*A^T P A
        #        - gamma^2*A^T P B (R+gamma B^T P B)^-1 B^T P A
        # scipy.linalg.solve_discrete_are solves:
        #   A^T P A - P - A^T P B (R + B^T P B)^-1 B^T P A + Q = 0
        # So we pass sqrt(gamma)*A, sqrt(gamma)*B
        a_scaled = np.sqrt(gamma) * a_mat
        b_scaled = np.sqrt(gamma) * b_mat

        try:
            p_mat = linalg.solve_discrete_are(
                a_scaled, b_scaled, q_mat, r_mat,
            )
        except (linalg.LinAlgError, np.linalg.LinAlgError) as exc:
            raise ValueError(
                f"DARE has no solution (system may be unstabilizable): {exc}"
            ) from exc

        # Optimal gain: K = (R + gamma * B^T P B)^-1 * gamma * B^T P A
        btp = gamma * b_mat.T @ p_mat  # (m, n)
        k_gain = np.linalg.solve(
            r_mat + gamma * b_mat.T @ p_mat @ b_mat,
            btp @ a_mat,
        )  # (m, n)

        # Check stability: eigenvalues of (A - B*K)
        closed_loop = a_mat - b_mat @ k_gain
        eigenvalues = np.linalg.eigvals(closed_loop)
        is_stable = bool(np.all(np.abs(eigenvalues) < 1.0))

        logger.debug(
            "LQR solved: state_dim={}, action_dim={}, stable={}, "
            "max_eigval={:.4f}",
            dynamics.state_dim, dynamics.action_dim, is_stable,
            float(np.max(np.abs(eigenvalues))),
        )

        return LQRSolution(
            k_gain=k_gain,
            p_matrix=p_mat,
            is_stable=is_stable,
        )

    def solve_per_env(
        self,
        dynamics_per_env: dict[int, LinearDynamics],
        cost: QuadraticCost,
        gamma: float = 0.99,
    ) -> dict[int, LQRSolution]:
        """Solve LQR for multiple environments with different dynamics.

        See ``CIRC-RL_Framework.md`` Section 3.6.1 (Cross-environment
        normalization): :math:`K_e = (R + B_e^T P_e B_e)^{-1} B_e^T P_e A`.

        :param dynamics_per_env: Mapping from env_idx to LinearDynamics.
        :param cost: Shared quadratic cost.
        :param gamma: Discount factor.
        :returns: Mapping from env_idx to LQRSolution.
        """
        solutions: dict[int, LQRSolution] = {}
        for env_idx, dynamics in dynamics_per_env.items():
            try:
                solutions[env_idx] = self.solve(dynamics, cost, gamma)
            except ValueError:
                logger.warning(
                    "LQR failed for env {}: system may be unstabilizable",
                    env_idx,
                )
        return solutions
