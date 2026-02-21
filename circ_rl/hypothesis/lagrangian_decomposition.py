"""Lagrangian decomposition for multi-DOF dynamics discovery.

For systems governed by Euler-Lagrange equations (e.g. Acrobot, double
pendulum), PySR cannot discover the coupled dynamics because they involve
mass matrix inversion M^(-1). This module decomposes the problem:

1. Detect the multi-DOF structure from state layout
2. Fit the 8 EL coefficients per environment via nonlinear least squares
   (NLS) on the RK4-integrated forward model
3. Fit parametric templates across environments
4. Compose forward dynamics symbolically via M^(-1)

The Euler-Lagrange equations are:

.. math::

    M(q) \\ddot{q} = \\tau - C(q, \\dot{q}) \\dot{q} - G(q)

Each component (M_ij, C_i, G_i) is a simple algebraic function of the
environment parameters that can be identified from data.

See ``CIRC-RL_Framework.md`` Section 3.4 (Hypothesis Generation).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import sympy
from loguru import logger
from scipy.optimize import least_squares

from circ_rl.hypothesis.expression import SymbolicExpression

if TYPE_CHECKING:
    from collections.abc import Callable

    from circ_rl.environments.data_collector import ExploratoryDataset


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MultiDOFStructure:
    """Detected multi-DOF Lagrangian structure.

    :param n_dof: Number of degrees of freedom.
    :param angle_dims: State indices for generalized coordinates (angles).
    :param velocity_dims: State indices for generalized velocities.
    :param actuated_dofs: Which DOFs receive direct torque input.
    """

    n_dof: int
    angle_dims: tuple[int, ...]
    velocity_dims: tuple[int, ...]
    actuated_dofs: tuple[int, ...]


@dataclass
class LagrangianDecompositionResult:
    """Result of the full Lagrangian decomposition pipeline.

    :param dynamics_expressions: Per-dim SymbolicExpression for delta_s.
    :param per_env_r2: R2 of the NLS forward model per environment.
    :param composed_r2: R2 of the composed forward dynamics on pooled data.
    :param per_env_coefficients: Fitted coefficient vectors per env.
    :param coefficient_names: Names for the coefficients.
    :param parametric_templates: Symbolic templates for each coefficient.
    """

    dynamics_expressions: dict[int, SymbolicExpression]
    per_env_r2: dict[int, float]
    composed_r2: dict[int, float]
    per_env_coefficients: dict[int, np.ndarray]
    coefficient_names: list[str]
    parametric_templates: dict[str, sympy.Expr] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Coefficient names
# ---------------------------------------------------------------------------

COEFF_NAMES = [
    "d1_const", "d1_cos", "d2_const", "d2_cos",
    "d3", "h_coriolis", "g_sin1", "g_sin12",
]


# ---------------------------------------------------------------------------
# RK4 forward model for 2-DOF EL dynamics
# ---------------------------------------------------------------------------


def _el_derivs(
    state: np.ndarray,
    torque: float,
    coeffs: np.ndarray,
) -> np.ndarray:
    """Compute derivatives from 2-DOF Euler-Lagrange dynamics.

    Matches Gymnasium Acrobot ``_dsdt`` structure exactly (book variant).

    :param state: ``[theta1, theta2, dtheta1, dtheta2]``.
    :param torque: Applied torque on actuated joint.
    :param coeffs: 8 EL coefficients ``[d1c, d1cos, d2c, d2cos, d3,
        h, gs1, gs12]``.
    :returns: ``[dtheta1, dtheta2, ddtheta1, ddtheta2]``.
    """
    t1, t2, w1, w2 = state[0], state[1], state[2], state[3]
    d1c, d1cos, d2c, d2cos, d3v, h, gs1, gs12 = (
        coeffs[0], coeffs[1], coeffs[2], coeffs[3],
        coeffs[4], coeffs[5], coeffs[6], coeffs[7],
    )

    cos_t2 = math.cos(t2)
    sin_t2 = math.sin(t2)

    d1 = d1c + d1cos * cos_t2
    d2 = d2c + d2cos * cos_t2

    # phi2 and phi1 match Gymnasium _dsdt (book variant)
    # phi2 = gs12 * cos(t1 + t2 - pi/2) = gs12 * sin(t1 + t2)
    phi2 = gs12 * math.sin(t1 + t2)
    phi1 = (
        -h * w2**2 * sin_t2
        - 2 * h * w2 * w1 * sin_t2
        + gs1 * math.sin(t1)
        + phi2
    )

    # ddtheta2 = (tau + d2/d1*phi1 - h*w1^2*sin(t2) - phi2) / (d3 - d2^2/d1)
    ddtheta2 = (
        torque + d2 / d1 * phi1 - h * w1**2 * sin_t2 - phi2
    ) / (d3v - d2**2 / d1)

    ddtheta1 = -(d2 * ddtheta2 + phi1) / d1

    return np.array([w1, w2, ddtheta1, ddtheta2])


def _rk4_forward(
    state: np.ndarray,
    torque: float,
    coeffs: np.ndarray,
    dt: float,
    max_vel_1: float,
    max_vel_2: float,
) -> np.ndarray:
    """One RK4 step for 2-DOF EL dynamics.

    :param state: ``[theta1, theta2, dtheta1, dtheta2]``.
    :param torque: Applied torque.
    :param coeffs: 8 EL coefficients.
    :param dt: Integration timestep.
    :param max_vel_1: Velocity clipping bound for DOF 1.
    :param max_vel_2: Velocity clipping bound for DOF 2.
    :returns: Next state ``[theta1, theta2, dtheta1, dtheta2]``.
    """
    k1 = _el_derivs(state, torque, coeffs)
    k2 = _el_derivs(state + dt / 2 * k1, torque, coeffs)
    k3 = _el_derivs(state + dt / 2 * k2, torque, coeffs)
    k4 = _el_derivs(state + dt * k3, torque, coeffs)

    ns = state + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

    # Wrap angles to [-pi, pi]
    ns[0] = math.atan2(math.sin(ns[0]), math.cos(ns[0]))
    ns[1] = math.atan2(math.sin(ns[1]), math.cos(ns[1]))
    # Clip velocities
    ns[2] = max(-max_vel_1, min(max_vel_1, ns[2]))
    ns[3] = max(-max_vel_2, min(max_vel_2, ns[3]))
    return ns


def _compute_initial_guess(env_params: dict[str, float]) -> np.ndarray:
    """Compute physics-informed initial guess for EL coefficients.

    Uses known Acrobot formulas with lc = l/2, I = 1, g = 9.8.

    :param env_params: ``{LINK_MASS_1, LINK_MASS_2, LINK_LENGTH_1, LINK_LENGTH_2}``.
    :returns: Initial coefficient vector of shape ``(8,)``.
    """
    m1 = env_params["LINK_MASS_1"]
    m2 = env_params["LINK_MASS_2"]
    l1 = env_params["LINK_LENGTH_1"]
    l2 = env_params["LINK_LENGTH_2"]
    lc1 = l1 / 2.0
    lc2 = l2 / 2.0
    g = 9.8

    d1_const = m1 * lc1**2 + m2 * (l1**2 + lc2**2) + 2.0  # I1+I2=2
    d1_cos = 2 * m2 * l1 * lc2
    d2_const = m2 * lc2**2 + 1.0  # I2 = 1
    d2_cos = m2 * l1 * lc2
    d3 = m2 * lc2**2 + 1.0  # I2 = 1
    h_coriolis = m2 * l1 * lc2
    g_sin1 = (m1 * lc1 + m2 * l1) * g
    g_sin12 = m2 * lc2 * g

    return np.array([
        d1_const, d1_cos, d2_const, d2_cos,
        d3, h_coriolis, g_sin1, g_sin12,
    ])


# ---------------------------------------------------------------------------
# LagrangianDecomposer
# ---------------------------------------------------------------------------


class LagrangianDecomposer:
    """Decompose multi-DOF Lagrangian dynamics via NLS regression.

    For a 2-DOF system (e.g. Acrobot), the Euler-Lagrange equations define
    8 unknown coefficient functions of environment parameters. This class:

    1. Detects the multi-DOF structure from state layout
    2. Fits coefficients per environment via NLS on the RK4 forward model
    3. Fits parametric templates across environments
    4. Composes forward dynamics symbolically via M^(-1)

    NLS (nonlinear least squares) is used instead of OLS because the data
    is generated with RK4 integration (dt=0.2), making the Euler
    acceleration approximation too inaccurate for linear regression.

    :param dt: Integration timestep.
    :param coefficient_r2_threshold: Minimum R2 for per-env regression.
    :param max_vel_1: Velocity clipping bound for DOF 1.
    :param max_vel_2: Velocity clipping bound for DOF 2.
    :param max_samples_per_env: Maximum samples used per env for NLS
        (subsampled for speed).
    """

    def __init__(
        self,
        dt: float = 0.2,
        coefficient_r2_threshold: float = 0.95,
        max_vel_1: float = 4.0 * np.pi,
        max_vel_2: float = 9.0 * np.pi,
        max_samples_per_env: int = 2000,
    ) -> None:
        self._dt = dt
        self._r2_threshold = coefficient_r2_threshold
        self._max_vel_1 = max_vel_1
        self._max_vel_2 = max_vel_2
        self._max_samples = max_samples_per_env

    def detect_structure(
        self,
        state_names: list[str],
        angular_dims: tuple[int, ...],
        action_dim: int,
    ) -> MultiDOFStructure | None:
        """Detect multi-DOF Lagrangian structure from state layout.

        Requires at least 2 angular dimensions with corresponding velocity
        dimensions (angle_dims and velocity_dims must pair up).

        :param state_names: Canonical state variable names.
        :param angular_dims: Indices of angular state dimensions.
        :param action_dim: Number of action dimensions.
        :returns: Detected structure, or None if not a multi-DOF system.
        """
        n_state = len(state_names)
        if len(angular_dims) < 2:
            return None

        n_dof = len(angular_dims)
        if n_state < 2 * n_dof:
            return None

        velocity_dims = tuple(
            i for i in range(n_state) if i not in angular_dims
        )
        if len(velocity_dims) != n_dof:
            return None

        actuated_dofs = tuple(range(n_dof - action_dim, n_dof))

        logger.info(
            "Detected {}-DOF Lagrangian structure: "
            "angles={}, velocities={}, actuated={}",
            n_dof, angular_dims, velocity_dims, actuated_dofs,
        )

        return MultiDOFStructure(
            n_dof=n_dof,
            angle_dims=angular_dims,
            velocity_dims=velocity_dims,
            actuated_dofs=actuated_dofs,
        )

    def fit_per_env_coefficients_nls(
        self,
        dataset: ExploratoryDataset,
        structure: MultiDOFStructure,
        env_params_per_env: dict[int, dict[str, float]],
    ) -> tuple[dict[int, np.ndarray], dict[int, float]]:
        """Fit the 8 EL coefficients per environment via NLS.

        Uses scipy.optimize.least_squares with the RK4 forward model
        to match observed transitions. The initial guess comes from
        the known physics formulas using the environment parameters.

        :param dataset: Canonical dataset.
        :param structure: Detected multi-DOF structure.
        :param env_params_per_env: Per-env parameter dicts.
        :returns: ``(per_env_coeffs, per_env_r2)``.
        """
        ai0, ai1 = structure.angle_dims
        vi0, vi1 = structure.velocity_dims

        unique_envs = np.unique(dataset.env_ids)
        per_env_coeffs: dict[int, np.ndarray] = {}
        per_env_r2: dict[int, float] = {}

        for env_id in unique_envs:
            eid = int(env_id)
            mask = dataset.env_ids == env_id

            states = dataset.states[mask]  # (n_e, state_dim)
            next_states = dataset.next_states[mask]
            actions = dataset.actions[mask]
            if actions.ndim == 2:
                actions = actions[:, 0]  # (n_e,)

            # Filter velocity-clipped samples
            vel_ok = (
                (np.abs(states[:, vi0]) < self._max_vel_1 * 0.99)
                & (np.abs(states[:, vi1]) < self._max_vel_2 * 0.99)
                & (np.abs(next_states[:, vi0]) < self._max_vel_1 * 0.99)
                & (np.abs(next_states[:, vi1]) < self._max_vel_2 * 0.99)
            )
            states = states[vel_ok]
            next_states = next_states[vel_ok]
            actions = actions[vel_ok]

            # Subsample for speed
            n_e = states.shape[0]
            if n_e > self._max_samples:
                rng = np.random.default_rng(eid)
                idx = rng.choice(n_e, self._max_samples, replace=False)
                states = states[idx]
                next_states = next_states[idx]
                actions = actions[idx]
                n_e = self._max_samples

            # Observed velocity deltas (targets for NLS)
            obs_dw1 = next_states[:, vi0] - states[:, vi0]  # (n_e,)
            obs_dw2 = next_states[:, vi1] - states[:, vi1]  # (n_e,)

            # Initial guess from known physics
            x0 = _compute_initial_guess(env_params_per_env[eid])

            # NLS residual function
            dt = self._dt
            mv1 = self._max_vel_1
            mv2 = self._max_vel_2

            def _residuals(
                coeffs: np.ndarray,
                _states: np.ndarray = states,
                _actions: np.ndarray = actions,
                _obs_dw1: np.ndarray = obs_dw1,
                _obs_dw2: np.ndarray = obs_dw2,
            ) -> np.ndarray:
                n = _states.shape[0]
                res = np.empty(2 * n, dtype=np.float64)
                for i in range(n):
                    pred = _rk4_forward(
                        _states[i], float(_actions[i]),
                        coeffs, dt, mv1, mv2,
                    )
                    res[2 * i] = (
                        pred[vi0] - _states[i, vi0] - _obs_dw1[i]
                    )
                    res[2 * i + 1] = (
                        pred[vi1] - _states[i, vi1] - _obs_dw2[i]
                    )
                return res

            result = least_squares(
                _residuals, x0, method="lm", max_nfev=200,
            )
            coeffs = result.x

            # Compute R2 on velocity deltas
            final_res = _residuals(coeffs)
            pred_dw1 = obs_dw1 + final_res[0::2]
            pred_dw2 = obs_dw2 + final_res[1::2]
            # Actually: res[2i] = pred_dw1[i] - obs_dw1[i]
            # so pred_dw1 = obs_dw1 + res[0::2]
            all_obs = np.concatenate([obs_dw1, obs_dw2])
            all_pred = np.concatenate([
                obs_dw1 + final_res[0::2],
                obs_dw2 + final_res[1::2],
            ])
            ss_res = np.sum((all_obs - all_pred) ** 2)
            ss_tot = np.sum((all_obs - all_obs.mean()) ** 2)
            r2 = 1.0 - ss_res / max(ss_tot, 1e-15)

            per_env_coeffs[eid] = coeffs
            per_env_r2[eid] = float(r2)

            logger.debug(
                "Env {}: NLS R2={:.6f}, nfev={}, coeffs={}",
                eid, r2, result.nfev,
                {n: f"{c:.4f}" for n, c in zip(COEFF_NAMES, coeffs)},
            )

        r2_values = list(per_env_r2.values())
        logger.info(
            "Per-env NLS regression: mean R2={:.4f}, min R2={:.4f}, "
            "max R2={:.4f} ({} envs)",
            np.mean(r2_values), np.min(r2_values),
            np.max(r2_values), len(unique_envs),
        )

        return per_env_coeffs, per_env_r2

    def fit_parametric_templates(
        self,
        per_env_coeffs: dict[int, np.ndarray],
        env_params_per_env: dict[int, dict[str, float]],
        param_names: list[str],
        coefficient_names: list[str],
    ) -> dict[str, sympy.Expr]:
        """Fit parametric templates for each EL coefficient.

        Each coefficient b_j is a known function of the environment
        parameters. We fit via least squares on the NLS-recovered
        coefficient values across environments.

        :param per_env_coeffs: NLS-fitted coefficient vectors per env.
        :param env_params_per_env: Env params dict per env_id.
        :param param_names: Names of environment parameters.
        :param coefficient_names: Names of the 8 coefficients.
        :returns: Dict mapping coefficient name to sympy expression
            in terms of environment parameter symbols.
        """
        m1, m2 = sympy.symbols("LINK_MASS_1 LINK_MASS_2")
        l1, l2 = sympy.symbols("LINK_LENGTH_1 LINK_LENGTH_2")

        # Template basis functions for each coefficient
        templates: dict[str, list[tuple[sympy.Expr, Any]]] = {
            "d1_const": [
                (m1 * l1**2, lambda p: p["LINK_MASS_1"] * p["LINK_LENGTH_1"]**2),
                (m2 * l1**2, lambda p: p["LINK_MASS_2"] * p["LINK_LENGTH_1"]**2),
                (m2 * l2**2, lambda p: p["LINK_MASS_2"] * p["LINK_LENGTH_2"]**2),
                (sympy.Integer(1), lambda p: 1.0),
            ],
            "d1_cos": [
                (m2 * l1 * l2, lambda p: p["LINK_MASS_2"] * p["LINK_LENGTH_1"] * p["LINK_LENGTH_2"]),
            ],
            "d2_const": [
                (m2 * l2**2, lambda p: p["LINK_MASS_2"] * p["LINK_LENGTH_2"]**2),
                (sympy.Integer(1), lambda p: 1.0),
            ],
            "d2_cos": [
                (m2 * l1 * l2, lambda p: p["LINK_MASS_2"] * p["LINK_LENGTH_1"] * p["LINK_LENGTH_2"]),
            ],
            "d3": [
                (m2 * l2**2, lambda p: p["LINK_MASS_2"] * p["LINK_LENGTH_2"]**2),
                (sympy.Integer(1), lambda p: 1.0),
            ],
            "h_coriolis": [
                (m2 * l1 * l2, lambda p: p["LINK_MASS_2"] * p["LINK_LENGTH_1"] * p["LINK_LENGTH_2"]),
            ],
            "g_sin1": [
                (m1 * l1, lambda p: p["LINK_MASS_1"] * p["LINK_LENGTH_1"]),
                (m2 * l1, lambda p: p["LINK_MASS_2"] * p["LINK_LENGTH_1"]),
            ],
            "g_sin12": [
                (m2 * l2, lambda p: p["LINK_MASS_2"] * p["LINK_LENGTH_2"]),
            ],
        }

        env_ids_sorted = sorted(per_env_coeffs.keys())
        n_envs = len(env_ids_sorted)
        fitted_templates: dict[str, sympy.Expr] = {}

        for j, cname in enumerate(coefficient_names):
            if cname not in templates:
                raise ValueError(
                    f"No template defined for coefficient '{cname}'"
                )

            basis_list = templates[cname]
            n_basis = len(basis_list)

            X = np.zeros((n_envs, n_basis), dtype=np.float64)
            y_obs = np.zeros(n_envs, dtype=np.float64)

            for i, eid in enumerate(env_ids_sorted):
                params = env_params_per_env[eid]
                y_obs[i] = per_env_coeffs[eid][j]
                for k, (_, fn) in enumerate(basis_list):
                    X[i, k] = fn(params)

            result = np.linalg.lstsq(X, y_obs, rcond=None)
            alphas = result[0]

            # Compose symbolic expression
            expr: sympy.Expr = sympy.Integer(0)
            for k, (sym_basis, _) in enumerate(basis_list):
                coeff = float(alphas[k])
                expr = expr + sympy.nsimplify(
                    coeff, rational=True, tolerance=0.01,
                ) * sym_basis

            fitted_templates[cname] = sympy.expand(expr)

            # Report fit quality
            y_pred = X @ alphas
            ss_res = np.sum((y_obs - y_pred) ** 2)
            ss_tot = np.sum((y_obs - y_obs.mean()) ** 2)
            r2_t = 1.0 - ss_res / max(ss_tot, 1e-15) if n_envs > 1 else 1.0

            logger.info(
                "Template {}: {} (R2={:.6f}, alphas={})",
                cname, fitted_templates[cname], r2_t,
                [f"{a:.4f}" for a in alphas],
            )

        return fitted_templates

    def compose_dynamics(
        self,
        templates: dict[str, sympy.Expr],
        structure: MultiDOFStructure,
        state_names: list[str],
        action_names: list[str],
    ) -> dict[int, SymbolicExpression]:
        """Compose forward dynamics from fitted EL coefficient templates.

        Inverts the 2x2 mass matrix symbolically:

        .. math::

            \\begin{pmatrix} \\ddot{q}_1 \\\\ \\ddot{q}_2 \\end{pmatrix}
            = M^{-1} (\\tau - C - G)

        Returns delta expressions (dt * acceleration) as SymbolicExpressions.

        :param templates: Symbolic templates for each coefficient.
        :param structure: The multi-DOF structure.
        :param state_names: Canonical state variable names.
        :param action_names: Action variable names.
        :returns: Dict mapping velocity dim index to SymbolicExpression.
        """
        if structure.n_dof != 2:
            raise ValueError("Only 2-DOF composition supported")

        s = {name: sympy.Symbol(name) for name in state_names}
        a = {name: sympy.Symbol(name) for name in action_names}

        t1_sym = s[state_names[structure.angle_dims[0]]]
        t2_sym = s[state_names[structure.angle_dims[1]]]
        w1_sym = s[state_names[structure.velocity_dims[0]]]
        w2_sym = s[state_names[structure.velocity_dims[1]]]
        torque_sym = a[action_names[0]]

        d1 = templates["d1_const"] + templates["d1_cos"] * sympy.cos(t2_sym)
        d2 = templates["d2_const"] + templates["d2_cos"] * sympy.cos(t2_sym)
        d3 = templates["d3"]

        h = templates["h_coriolis"]
        coriolis_1 = h * sympy.sin(t2_sym) * (
            w2_sym**2 + 2 * w1_sym * w2_sym
        )
        coriolis_2 = -h * sympy.sin(t2_sym) * w1_sym**2

        gravity_1 = (
            templates["g_sin1"] * sympy.sin(t1_sym)
            + templates["g_sin12"] * sympy.sin(t1_sym + t2_sym)
        )
        gravity_2 = templates["g_sin12"] * sympy.sin(t1_sym + t2_sym)

        rhs1 = -coriolis_1 - gravity_1
        rhs2 = torque_sym - coriolis_2 - gravity_2

        det_m = d1 * d3 - d2**2
        ddtheta1 = (d3 * rhs1 - d2 * rhs2) / det_m
        ddtheta2 = (-d2 * rhs1 + d1 * rhs2) / det_m

        dt_sym = sympy.Rational(self._dt).limit_denominator(1000)
        delta_w1 = dt_sym * ddtheta1
        delta_w2 = dt_sym * ddtheta2

        vi0 = structure.velocity_dims[0]
        vi1 = structure.velocity_dims[1]

        result: dict[int, SymbolicExpression] = {
            vi0: SymbolicExpression.from_sympy(delta_w1),
            vi1: SymbolicExpression.from_sympy(delta_w2),
        }

        logger.info(
            "Composed dynamics: delta_s{} complexity={}, delta_s{} complexity={}",
            vi0, result[vi0].complexity,
            vi1, result[vi1].complexity,
        )

        return result

    def decompose(
        self,
        dataset: ExploratoryDataset,
        state_names: list[str],
        action_names: list[str],
        env_param_names: list[str],
        angular_dims: tuple[int, ...],
    ) -> LagrangianDecompositionResult | None:
        """Run the full Lagrangian decomposition pipeline.

        :param dataset: Canonical dataset with multi-env transitions.
        :param state_names: Canonical state names.
        :param action_names: Action variable names.
        :param env_param_names: Environment parameter names.
        :param angular_dims: Indices of angular state dimensions.
        :returns: Decomposition result, or None if structure not detected.
        """
        action_dim = (
            1 if dataset.actions.ndim == 1 else dataset.actions.shape[1]
        )

        # Step 1: Detect structure
        structure = self.detect_structure(
            state_names, angular_dims, action_dim,
        )
        if structure is None:
            logger.info("No multi-DOF structure detected")
            return None

        # Step 2: Extract per-env params
        env_params_per_env = self._extract_env_params(
            dataset, env_param_names,
        )

        # Step 3: NLS per-env regression
        per_env_coeffs, per_env_r2 = self.fit_per_env_coefficients_nls(
            dataset, structure, env_params_per_env,
        )

        bad_envs = [
            eid for eid, r2 in per_env_r2.items()
            if r2 < self._r2_threshold
        ]
        if bad_envs:
            logger.warning(
                "{} environments below R2 threshold {}: {}",
                len(bad_envs), self._r2_threshold,
                {eid: f"{per_env_r2[eid]:.4f}" for eid in bad_envs},
            )

        # Step 4: Fit parametric templates
        templates = self.fit_parametric_templates(
            per_env_coeffs,
            env_params_per_env,
            env_param_names,
            COEFF_NAMES,
        )

        # Step 5: Compose forward dynamics
        dynamics_expressions = self.compose_dynamics(
            templates, structure, state_names, action_names,
        )

        # Step 6: Evaluate composed R2 (via RK4 forward model)
        composed_r2 = self._evaluate_composed_r2(
            dynamics_expressions, dataset, structure,
            state_names, action_names, env_param_names,
            templates=templates,
        )

        logger.info(
            "Composed forward dynamics R2: {}",
            {f"dim_{k}": f"{v:.6f}" for k, v in composed_r2.items()},
        )

        return LagrangianDecompositionResult(
            dynamics_expressions=dynamics_expressions,
            per_env_r2=per_env_r2,
            composed_r2=composed_r2,
            per_env_coefficients=per_env_coeffs,
            coefficient_names=COEFF_NAMES,
            parametric_templates=templates,
        )

    def _extract_env_params(
        self,
        dataset: ExploratoryDataset,
        env_param_names: list[str],
    ) -> dict[int, dict[str, float]]:
        """Extract per-env parameter dicts from dataset."""
        if dataset.env_params is None:
            raise ValueError("Dataset has no env_params")

        env_params_per_env: dict[int, dict[str, float]] = {}
        unique_envs = np.unique(dataset.env_ids)

        for env_id in unique_envs:
            mask = dataset.env_ids == env_id
            row = dataset.env_params[mask][0]
            env_params_per_env[int(env_id)] = {
                name: float(row[i])
                for i, name in enumerate(env_param_names)
            }

        return env_params_per_env

    def _evaluate_composed_r2(
        self,
        dynamics_expressions: dict[int, SymbolicExpression],
        dataset: ExploratoryDataset,
        structure: MultiDOFStructure,
        state_names: list[str],
        action_names: list[str],
        env_param_names: list[str],
        templates: dict[str, sympy.Expr] | None = None,
    ) -> dict[int, float]:
        """Evaluate R2 of composed dynamics on pooled data.

        Uses the RK4 forward model with template-substituted coefficients
        per environment for accurate evaluation. The symbolic expressions
        use Euler integration (dt * ddtheta) which is inaccurate for
        dt=0.2, but the underlying EL coefficients are correct -- so we
        evaluate via RK4 to measure the true quality.

        Falls back to direct expression evaluation if templates are not
        provided.
        """
        if templates is None or structure.n_dof != 2:
            return self._evaluate_composed_r2_direct(
                dynamics_expressions, dataset, structure,
                state_names, action_names, env_param_names,
            )

        vi0 = structure.velocity_dims[0]
        vi1 = structure.velocity_dims[1]

        # Evaluate template coefficients per env
        env_params_per_env = self._extract_env_params(
            dataset, env_param_names,
        )
        env_coeffs: dict[int, np.ndarray] = {}
        for eid, params in env_params_per_env.items():
            subs = {
                sympy.Symbol(k): v for k, v in params.items()
            }
            coeffs = np.array([
                float(templates[cn].subs(subs))
                for cn in COEFF_NAMES
            ])
            env_coeffs[eid] = coeffs

        # RK4 forward prediction per sample
        actions = dataset.actions
        if actions.ndim == 2:
            actions = actions[:, 0]

        n = dataset.states.shape[0]
        pred_dw1 = np.empty(n, dtype=np.float64)
        pred_dw2 = np.empty(n, dtype=np.float64)

        for i in range(n):
            eid = int(dataset.env_ids[i])
            ns = _rk4_forward(
                dataset.states[i], float(actions[i]),
                env_coeffs[eid], self._dt,
                self._max_vel_1, self._max_vel_2,
            )
            pred_dw1[i] = ns[vi0] - dataset.states[i, vi0]
            pred_dw2[i] = ns[vi1] - dataset.states[i, vi1]

        obs_dw1 = dataset.next_states[:, vi0] - dataset.states[:, vi0]
        obs_dw2 = dataset.next_states[:, vi1] - dataset.states[:, vi1]

        composed_r2: dict[int, float] = {}
        for dim_idx, obs, pred in [
            (vi0, obs_dw1, pred_dw1),
            (vi1, obs_dw2, pred_dw2),
        ]:
            ss_res = np.sum((obs - pred) ** 2)
            ss_tot = np.sum((obs - obs.mean()) ** 2)
            r2 = 1.0 - ss_res / max(ss_tot, 1e-15)
            composed_r2[dim_idx] = float(r2)

        return composed_r2

    def _evaluate_composed_r2_direct(
        self,
        dynamics_expressions: dict[int, SymbolicExpression],
        dataset: ExploratoryDataset,
        structure: MultiDOFStructure,
        state_names: list[str],
        action_names: list[str],
        env_param_names: list[str],
    ) -> dict[int, float]:
        """Fallback: evaluate composed R2 by direct expression evaluation."""
        var_names = (
            list(state_names) + list(action_names) + list(env_param_names)
        )

        actions_2d = dataset.actions
        if actions_2d.ndim == 1:
            actions_2d = actions_2d.reshape(-1, 1)
        parts = [dataset.states, actions_2d]
        if dataset.env_params is not None:
            parts.append(dataset.env_params)
        X = np.hstack(parts)  # (N, n_vars)

        composed_r2: dict[int, float] = {}

        for dim_idx, expr in dynamics_expressions.items():
            observed = (
                dataset.next_states[:, dim_idx] - dataset.states[:, dim_idx]
            )
            if dim_idx in structure.angle_dims:
                observed = np.arctan2(np.sin(observed), np.cos(observed))

            fn = expr.to_callable(var_names)
            predicted = fn(X)

            ss_res = np.sum((observed - predicted) ** 2)
            ss_tot = np.sum((observed - observed.mean()) ** 2)
            r2 = 1.0 - ss_res / max(ss_tot, 1e-15)
            composed_r2[dim_idx] = float(r2)

        return composed_r2


# ---------------------------------------------------------------------------
# Vectorized RK4 dynamics for MPPI / iLQR
# ---------------------------------------------------------------------------


def _el_derivs_batched(
    states: np.ndarray,
    torques: np.ndarray,
    coeffs: np.ndarray,
) -> np.ndarray:
    """Vectorized EL derivatives for K parallel states.

    :param states: ``(K, 4)`` -- ``[theta1, theta2, dtheta1, dtheta2]``.
    :param torques: ``(K,)`` -- applied torques.
    :param coeffs: ``(8,)`` -- EL coefficients.
    :returns: ``(K, 4)`` derivatives.
    """
    t1 = states[:, 0]  # (K,)
    t2 = states[:, 1]  # (K,)
    w1 = states[:, 2]  # (K,)
    w2 = states[:, 3]  # (K,)

    d1c, d1cos, d2c, d2cos = coeffs[0], coeffs[1], coeffs[2], coeffs[3]
    d3v, h, gs1, gs12 = coeffs[4], coeffs[5], coeffs[6], coeffs[7]

    cos_t2 = np.cos(t2)  # (K,)
    sin_t2 = np.sin(t2)  # (K,)

    d1 = d1c + d1cos * cos_t2  # (K,)
    d2 = d2c + d2cos * cos_t2  # (K,)

    phi2 = gs12 * np.sin(t1 + t2)  # (K,)
    phi1 = (
        -h * w2**2 * sin_t2
        - 2 * h * w2 * w1 * sin_t2
        + gs1 * np.sin(t1)
        + phi2
    )  # (K,)

    ddtheta2 = (
        torques + d2 / d1 * phi1 - h * w1**2 * sin_t2 - phi2
    ) / (d3v - d2**2 / d1)  # (K,)

    ddtheta1 = -(d2 * ddtheta2 + phi1) / d1  # (K,)

    out = np.empty_like(states)  # (K, 4)
    out[:, 0] = w1
    out[:, 1] = w2
    out[:, 2] = ddtheta1
    out[:, 3] = ddtheta2
    return out


def build_lagrangian_batched_dynamics_fn(
    templates: dict[str, sympy.Expr],
    env_params: dict[str, float],
    dt: float = 0.2,
    max_vel_1: float = 4.0 * np.pi,
    max_vel_2: float = 9.0 * np.pi,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Build a vectorized RK4 dynamics function from Lagrangian templates.

    Unlike ``build_batched_dynamics_fn`` which uses Euler integration
    (state + dt * expression), this function integrates using RK4 which
    is stable for the coupled Lagrangian dynamics at dt=0.2.

    :param templates: Symbolic templates for each EL coefficient.
    :param env_params: Environment parameter values to substitute.
    :param dt: Integration timestep.
    :param max_vel_1: Velocity clipping bound for DOF 1.
    :param max_vel_2: Velocity clipping bound for DOF 2.
    :returns: Callable ``(states, actions) -> next_states`` where
        ``states`` has shape ``(K, 4)`` and ``actions`` has shape ``(K, 1)``.
    """
    subs = {sympy.Symbol(k): v for k, v in env_params.items()}
    coeffs = np.array([
        float(templates[cn].subs(subs)) for cn in COEFF_NAMES
    ])

    _dt = dt
    _mv1 = max_vel_1
    _mv2 = max_vel_2
    _coeffs = coeffs

    def batched_dynamics_fn(
        states: np.ndarray,
        actions: np.ndarray,
    ) -> np.ndarray:
        """Vectorized RK4 dynamics: (K, 4), (K, 1) -> (K, 4)."""
        torques = actions[:, 0]  # (K,)

        k1 = _el_derivs_batched(states, torques, _coeffs)  # (K, 4)
        k2 = _el_derivs_batched(
            states + _dt / 2 * k1, torques, _coeffs,
        )
        k3 = _el_derivs_batched(
            states + _dt / 2 * k2, torques, _coeffs,
        )
        k4 = _el_derivs_batched(
            states + _dt * k3, torques, _coeffs,
        )

        ns = states + _dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)  # (K, 4)

        # Wrap angles to [-pi, pi]
        ns[:, 0] = np.arctan2(np.sin(ns[:, 0]), np.cos(ns[:, 0]))
        ns[:, 1] = np.arctan2(np.sin(ns[:, 1]), np.cos(ns[:, 1]))
        # Clip velocities
        ns[:, 2] = np.clip(ns[:, 2], -_mv1, _mv1)
        ns[:, 3] = np.clip(ns[:, 3], -_mv2, _mv2)
        return ns

    return batched_dynamics_fn


def build_lagrangian_scalar_dynamics_fn(
    templates: dict[str, sympy.Expr],
    env_params: dict[str, float],
    dt: float = 0.2,
    max_vel_1: float = 4.0 * np.pi,
    max_vel_2: float = 9.0 * np.pi,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Build a scalar RK4 dynamics function from Lagrangian templates.

    Same as ``build_lagrangian_batched_dynamics_fn`` but takes single
    state/action vectors (not batched). For use as the MPPI scalar
    dynamics function.

    :param templates: Symbolic templates for each EL coefficient.
    :param env_params: Environment parameter values to substitute.
    :param dt: Integration timestep.
    :param max_vel_1: Velocity clipping bound for DOF 1.
    :param max_vel_2: Velocity clipping bound for DOF 2.
    :returns: Callable ``(state, action) -> next_state`` where
        ``state`` has shape ``(4,)`` and ``action`` has shape ``(1,)``.
    """
    subs = {sympy.Symbol(k): v for k, v in env_params.items()}
    coeffs = np.array([
        float(templates[cn].subs(subs)) for cn in COEFF_NAMES
    ])

    _dt = dt
    _mv1 = max_vel_1
    _mv2 = max_vel_2

    def scalar_dynamics_fn(
        state: np.ndarray,
        action: np.ndarray,
    ) -> np.ndarray:
        """Scalar RK4 dynamics: (4,), (1,) -> (4,)."""
        return _rk4_forward(
            state, float(action[0]),
            coeffs, _dt, _mv1, _mv2,
        )

    return scalar_dynamics_fn


# ---------------------------------------------------------------------------
# Energy computation for cost shaping
# ---------------------------------------------------------------------------


def evaluate_coefficients(
    templates: dict[str, sympy.Expr],
    env_params: dict[str, float],
) -> dict[str, float]:
    """Evaluate parametric templates for a specific environment.

    :param templates: Symbolic templates mapping coefficient names to
        sympy expressions in terms of environment parameters.
    :param env_params: Environment parameter values.
    :returns: Dict mapping coefficient names to float values.
    """
    subs = {sympy.Symbol(k): v for k, v in env_params.items()}
    return {cn: float(templates[cn].subs(subs)) for cn in COEFF_NAMES}


def compute_mechanical_energy(
    state: np.ndarray,
    coeffs: dict[str, float],
) -> float:
    """Compute total mechanical energy E = T + V for a single state.

    :param state: Canonical state ``[phi_0, phi_1, w1, w2]``.
    :param coeffs: EL coefficient values (from ``evaluate_coefficients``).
    :returns: Total mechanical energy.
    """
    phi_0, phi_1 = state[0], state[1]
    w1, w2 = state[2], state[3]

    d1 = coeffs["d1_const"] + coeffs["d1_cos"] * math.cos(phi_1)
    d2 = coeffs["d2_const"] + coeffs["d2_cos"] * math.cos(phi_1)
    d3 = coeffs["d3"]

    # Kinetic energy: T = 0.5 * qdot^T M qdot
    kinetic = 0.5 * (d1 * w1**2 + 2.0 * d2 * w1 * w2 + d3 * w2**2)

    # Potential energy: V = -g_sin1*cos(phi_0) - g_sin12*cos(phi_0+phi_1)
    potential = (
        -coeffs["g_sin1"] * math.cos(phi_0)
        - coeffs["g_sin12"] * math.cos(phi_0 + phi_1)
    )

    return kinetic + potential


def compute_mechanical_energy_batched(
    states: np.ndarray,
    coeffs: dict[str, float],
) -> np.ndarray:
    """Compute total mechanical energy for K parallel states.

    :param states: Shape ``(K, 4)`` canonical states.
    :param coeffs: EL coefficient values.
    :returns: Shape ``(K,)`` total mechanical energies.
    """
    phi_0 = states[:, 0]  # (K,)
    phi_1 = states[:, 1]  # (K,)
    w1 = states[:, 2]  # (K,)
    w2 = states[:, 3]  # (K,)

    d1 = coeffs["d1_const"] + coeffs["d1_cos"] * np.cos(phi_1)  # (K,)
    d2 = coeffs["d2_const"] + coeffs["d2_cos"] * np.cos(phi_1)  # (K,)
    d3 = coeffs["d3"]

    # Kinetic energy: T = 0.5 * qdot^T M qdot
    kinetic = 0.5 * (d1 * w1**2 + 2.0 * d2 * w1 * w2 + d3 * w2**2)  # (K,)

    # Potential energy
    potential = (
        -coeffs["g_sin1"] * np.cos(phi_0)
        - coeffs["g_sin12"] * np.cos(phi_0 + phi_1)
    )  # (K,)

    return kinetic + potential  # (K,)


def compute_goal_energy(coeffs: dict[str, float]) -> float:
    """Compute mechanical energy at the upright goal state.

    Goal: phi_0=pi, phi_1=0, w1=0, w2=0 (upright, stationary).

    :param coeffs: EL coefficient values.
    :returns: Goal energy E*.
    """
    # T = 0 (zero velocity), V = -g_sin1*cos(pi) - g_sin12*cos(pi)
    return coeffs["g_sin1"] + coeffs["g_sin12"]
