"""Unified analytic policy interface.

Combines LQR or MPC with optional action normalization to provide
a single get_action() interface.

See ``CIRC-RL_Framework.md`` Section 3.6 (Phase 5: Analytic Policy
Derivation).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import sympy

from circ_rl.analytic_policy.lqr_solver import (
    LinearDynamics,
    LQRSolution,
    QuadraticCost,
)
from circ_rl.analytic_policy.mpc_solver import MPCConfig, MPCSolver
from circ_rl.hypothesis.expression import SymbolicExpression

if TYPE_CHECKING:
    from circ_rl.analytic_policy.action_normalizer import ActionNormalizer
    from circ_rl.hypothesis.hypothesis_register import HypothesisEntry


class AnalyticPolicy:
    """Unified analytic policy: get_action(state, env_idx) -> action.

    Wraps either an LQR gain or an MPC solver, with optional action
    normalization across environments.

    See ``CIRC-RL_Framework.md`` Section 3.6.

    :param dynamics_hypothesis: Validated dynamics hypothesis.
    :param reward_hypothesis: Validated reward hypothesis (optional;
        needed for MPC, not for LQR with known cost).
    :param solver_type: ``"lqr"`` or ``"mpc"``.
    :param state_dim: State dimensionality.
    :param action_dim: Action dimensionality.
    :param n_envs: Number of environments.
    :param action_normalizer: Optional action normalizer for cross-env
        dynamics scale adaptation.
    :param lqr_solutions: Pre-computed LQR solutions per environment.
        Required when solver_type is ``"lqr"``.
    :param mpc_config: MPC configuration. Required when solver_type
        is ``"mpc"``.
    :param dynamics_fn: Dynamics callable ``(state, action) -> next_state``.
        Required for MPC.
    :param reward_fn: Reward callable ``(state, action) -> float``.
        Required for MPC.
    :param action_low: Lower action bounds, shape ``(action_dim,)``.
    :param action_high: Upper action bounds, shape ``(action_dim,)``.
    """

    def __init__(
        self,
        dynamics_hypothesis: HypothesisEntry,
        reward_hypothesis: HypothesisEntry | None,
        solver_type: Literal["lqr", "mpc"],
        state_dim: int,
        action_dim: int,
        n_envs: int,
        action_normalizer: ActionNormalizer | None = None,
        lqr_solutions: dict[int, LQRSolution] | None = None,
        mpc_config: MPCConfig | None = None,
        dynamics_fn: object | None = None,
        reward_fn: object | None = None,
        action_low: np.ndarray | None = None,
        action_high: np.ndarray | None = None,
    ) -> None:
        self._dynamics_hypothesis = dynamics_hypothesis
        self._reward_hypothesis = reward_hypothesis
        self._solver_type = solver_type
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._n_envs = n_envs
        self._normalizer = action_normalizer
        self._lqr_solutions = lqr_solutions or {}
        self._mpc_config = mpc_config
        self._dynamics_fn = dynamics_fn
        self._reward_fn = reward_fn
        self._action_low = action_low
        self._action_high = action_high

        if solver_type == "lqr" and not self._lqr_solutions:
            raise ValueError(
                "lqr_solutions required when solver_type is 'lqr'"
            )
        if solver_type == "mpc" and dynamics_fn is None:
            raise ValueError(
                "dynamics_fn required when solver_type is 'mpc'"
            )

    @property
    def solver_type(self) -> str:
        """The solver type (lqr or mpc)."""
        return self._solver_type

    @property
    def n_free_parameters(self) -> int:
        """Number of free parameters in the analytic policy.

        For LQR this is 0 (the gain K is fully determined by A, B, Q, R).
        For MPC this is 0 (the policy is computed online from the model).
        """
        return 0

    @property
    def complexity(self) -> int:
        """Symbolic complexity of the underlying hypothesis."""
        expr = self._dynamics_hypothesis.expression
        if isinstance(expr, SymbolicExpression):
            return expr.complexity
        return self._dynamics_hypothesis.complexity

    def get_action(
        self,
        state: np.ndarray,
        env_idx: int,
    ) -> np.ndarray:
        """Compute the optimal action for a given state and environment.

        :param state: Current state, shape ``(state_dim,)``.
        :param env_idx: Environment index.
        :returns: Optimal action, shape ``(action_dim,)``.
        """
        if self._solver_type == "lqr":
            action = self._lqr_action(state, env_idx)
        else:
            action = self._mpc_action(state, env_idx)

        # Clip to action bounds
        if self._action_low is not None and self._action_high is not None:
            action = np.clip(action, self._action_low, self._action_high)

        return action

    def _lqr_action(self, state: np.ndarray, env_idx: int) -> np.ndarray:
        """Compute action using LQR gain.

        :returns: Action from :math:`a = -K s`.
        """
        if env_idx in self._lqr_solutions:
            sol = self._lqr_solutions[env_idx]
        elif self._lqr_solutions:
            # Use the first available solution as fallback
            sol = next(iter(self._lqr_solutions.values()))
        else:
            raise ValueError(
                f"No LQR solution available for env_idx={env_idx}"
            )

        action = -sol.k_gain @ state  # (action_dim,)

        # Apply action normalization if available
        if self._normalizer is not None:
            action = self._normalizer.normalize_action(action, env_idx)

        return action

    def _mpc_action(self, state: np.ndarray, env_idx: int) -> np.ndarray:
        """Compute action using MPC."""
        if self._mpc_config is None:
            raise ValueError("MPC config not set")


        solver = MPCSolver(
            config=self._mpc_config,
            dynamics_fn=self._dynamics_fn,  # type: ignore[arg-type]
            reward_fn=self._reward_fn,  # type: ignore[arg-type]
        )

        action = solver.solve(state, self._action_dim)

        # Apply action normalization if available
        if self._normalizer is not None:
            action = self._normalizer.normalize_action(action, env_idx)

        return action

    def __repr__(self) -> str:
        return (
            f"AnalyticPolicy(solver={self._solver_type}, "
            f"state_dim={self._state_dim}, action_dim={self._action_dim}, "
            f"complexity={self.complexity})"
        )


def extract_linear_dynamics(
    expression: SymbolicExpression,
    state_feature_names: list[str],
    action_names: list[str],
    env_params: dict[str, float] | None = None,
) -> LinearDynamics:
    """Extract A, B, c matrices from a linear symbolic expression.

    Given an expression for ``delta_s_i``, extracts the coefficients
    to build matrices A and B where ``delta_s = A @ s + B @ a + c``.

    Note: This builds one row of A and B for a single state dimension.
    To build the full system, call for each dimension and stack.

    :param expression: Symbolic expression for delta_s_i.
    :param state_feature_names: Names of state variables.
    :param action_names: Names of action variables.
    :param env_params: Environment parameter values (substituted into
        the expression before extracting coefficients).
    :returns: LinearDynamics (A row, B row, c scalar packed as matrices).
    """
    expr = expression.sympy_expr

    # Substitute env params if provided
    if env_params:
        subs = {sympy.Symbol(k): v for k, v in env_params.items()}
        expr = expr.subs(subs)

    state_syms = [sympy.Symbol(n) for n in state_feature_names]
    action_syms = [sympy.Symbol(n) for n in action_names]

    state_dim = len(state_feature_names)
    action_dim = len(action_names)

    # Extract coefficients (linear expression: c + sum_i a_i*x_i)
    # For delta_s_i, the effective dynamics are:
    # s_i(t+1) = s_i(t) + delta_s_i = s_i(t) + (A_row @ s + B_row @ a + c)
    # So the full A matrix row should be I[i,:] + A_row
    a_row = np.zeros(state_dim, dtype=np.float64)
    b_row = np.zeros(action_dim, dtype=np.float64)

    expanded = sympy.expand(expr)

    for j, s_sym in enumerate(state_syms):
        coeff = expanded.coeff(s_sym)
        a_row[j] = float(coeff)

    for j, a_sym in enumerate(action_syms):
        coeff = expanded.coeff(a_sym)
        b_row[j] = float(coeff)

    # Constant term
    c_val = float(expanded.subs(
        dict.fromkeys(state_syms + action_syms, 0)
    ))

    # Build as 1x matrices (single row for one delta_s dimension)
    return LinearDynamics(
        a_matrix=a_row.reshape(1, state_dim),
        b_matrix=b_row.reshape(1, action_dim),
        c_vector=np.array([c_val]),
    )


def extract_quadratic_cost(
    reward_expr: sympy.Expr,
    state_vars: list[str],
    action_vars: list[str],
) -> QuadraticCost | None:
    r"""Extract Q and R matrices from a quadratic reward expression.

    Attempts to decompose :math:`R = -s^T Q s - a^T R a` from the
    reward expression. Returns None if the expression is not quadratic.

    :param reward_expr: Sympy expression for the reward.
    :param state_vars: Names of state variables.
    :param action_vars: Names of action variables.
    :returns: QuadraticCost if extractable, None otherwise.
    """
    state_syms = [sympy.Symbol(v) for v in state_vars]
    action_syms = [sympy.Symbol(v) for v in action_vars]
    all_syms = state_syms + action_syms

    # Check that it's a polynomial of degree <= 2
    try:
        poly = sympy.Poly(reward_expr, *all_syms)
    except (sympy.PolynomialError, sympy.GeneratorsNeeded):
        return None

    if poly.total_degree() > 2:
        return None

    state_dim = len(state_vars)
    action_dim = len(action_vars)

    # Extract Q matrix (from -s^T Q s terms)
    q_matrix = np.zeros((state_dim, state_dim), dtype=np.float64)
    for i, si in enumerate(state_syms):
        for j, sj in enumerate(state_syms):
            if i == j:
                coeff = float(reward_expr.coeff(si, 2))
                q_matrix[i, j] = -coeff  # Negate: reward = -s^T Q s
            else:
                coeff = float(reward_expr.coeff(si * sj))
                q_matrix[i, j] = -coeff / 2  # Off-diagonal split

    # Extract R matrix (from -a^T R a terms)
    r_matrix = np.zeros((action_dim, action_dim), dtype=np.float64)
    for i, ai in enumerate(action_syms):
        for j, aj in enumerate(action_syms):
            if i == j:
                coeff = float(reward_expr.coeff(ai, 2))
                r_matrix[i, j] = -coeff
            else:
                coeff = float(reward_expr.coeff(ai * aj))
                r_matrix[i, j] = -coeff / 2

    # Validate PSD for Q and PD for R
    q_eigvals = np.linalg.eigvalsh(q_matrix)
    r_eigvals = np.linalg.eigvalsh(r_matrix)

    if np.any(q_eigvals < -1e-10):
        return None  # Q not PSD
    if np.any(r_eigvals <= 0):
        return None  # R not PD

    return QuadraticCost(q_matrix=q_matrix, r_matrix=r_matrix)
