# ruff: noqa: ANN001 ANN201

"""Unit tests for Lagrangian decomposition of multi-DOF dynamics."""

from __future__ import annotations

from math import cos, pi, sin

import numpy as np
import pytest

from circ_rl.environments.data_collector import ExploratoryDataset
from circ_rl.hypothesis.lagrangian_decomposition import (
    COEFF_NAMES,
    LagrangianDecomposer,
    _compute_initial_guess,
    _rk4_forward,
)


# ---------------------------------------------------------------------------
# Acrobot ground-truth dynamics (from Gymnasium _dsdt, "book" variant)
# ---------------------------------------------------------------------------

GRAVITY = 9.8
LINK_MOI = 1.0
MAX_VEL_1 = 4.0 * pi
MAX_VEL_2 = 9.0 * pi


def _acrobot_dsdt(
    s: np.ndarray,
    m1: float, m2: float, l1: float, l2: float,
) -> tuple[float, float, float, float]:
    """Compute Acrobot derivatives (book variant)."""
    lc1 = l1 / 2.0
    lc2 = l2 / 2.0
    g = GRAVITY
    t1, t2, w1, w2, torque = s

    d1 = (
        m1 * lc1**2
        + m2 * (l1**2 + lc2**2 + 2 * l1 * lc2 * cos(t2))
        + 2.0  # I1 + I2
    )
    d2 = m2 * (lc2**2 + l1 * lc2 * cos(t2)) + 1.0  # I2
    phi2 = m2 * lc2 * g * cos(t1 + t2 - pi / 2.0)
    phi1 = (
        -m2 * l1 * lc2 * w2**2 * sin(t2)
        - 2 * m2 * l1 * lc2 * w2 * w1 * sin(t2)
        + (m1 * lc1 + m2 * l1) * g * cos(t1 - pi / 2.0)
        + phi2
    )

    ddtheta2 = (
        torque + d2 / d1 * phi1
        - m2 * l1 * lc2 * w1**2 * sin(t2)
        - phi2
    ) / (m2 * lc2**2 + 1.0 - d2**2 / d1)

    ddtheta1 = -(d2 * ddtheta2 + phi1) / d1
    return (w1, w2, ddtheta1, ddtheta2)


def _rk4_step_true(
    m1: float, m2: float, l1: float, l2: float,
    state: np.ndarray, torque: float, dt: float = 0.2,
) -> np.ndarray:
    """One RK4 step using the true Acrobot dynamics."""
    def derivs(s: np.ndarray) -> np.ndarray:
        return np.array(_acrobot_dsdt(s, m1, m2, l1, l2))

    s_aug = np.append(state, torque)
    k1 = derivs(s_aug)
    k2 = derivs(np.append(state + dt / 2 * k1, torque))
    k3 = derivs(np.append(state + dt / 2 * k2, torque))
    k4 = derivs(np.append(state + dt * k3, torque))
    ns = state + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

    ns[0] = np.arctan2(np.sin(ns[0]), np.cos(ns[0]))
    ns[1] = np.arctan2(np.sin(ns[1]), np.cos(ns[1]))
    ns[2] = np.clip(ns[2], -MAX_VEL_1, MAX_VEL_1)
    ns[3] = np.clip(ns[3], -MAX_VEL_2, MAX_VEL_2)
    return ns


def _generate_acrobot_dataset(
    env_configs: list[dict[str, float]],
    n_per_env: int = 500,
    seed: int = 42,
) -> ExploratoryDataset:
    """Generate synthetic Acrobot transitions."""
    rng = np.random.default_rng(seed)
    param_names = [
        "LINK_MASS_1", "LINK_MASS_2",
        "LINK_LENGTH_1", "LINK_LENGTH_2",
    ]

    all_states: list[np.ndarray] = []
    all_next_states: list[np.ndarray] = []
    all_actions: list[float] = []
    all_rewards: list[float] = []
    all_env_ids: list[int] = []
    all_env_params: list[list[float]] = []

    for env_id, cfg in enumerate(env_configs):
        m1, m2 = cfg["LINK_MASS_1"], cfg["LINK_MASS_2"]
        l1, l2 = cfg["LINK_LENGTH_1"], cfg["LINK_LENGTH_2"]

        for _ in range(n_per_env):
            state = np.array([
                rng.uniform(-pi, pi),
                rng.uniform(-pi, pi),
                rng.uniform(-4, 4),
                rng.uniform(-6, 6),
            ])
            torque = rng.uniform(-1, 1)
            ns = _rk4_step_true(m1, m2, l1, l2, state, torque)

            all_states.append(state)
            all_next_states.append(ns)
            all_actions.append(torque)
            all_rewards.append(0.0)
            all_env_ids.append(env_id)
            all_env_params.append([cfg[p] for p in param_names])

    return ExploratoryDataset(
        states=np.array(all_states),
        actions=np.array(all_actions),
        next_states=np.array(all_next_states),
        rewards=np.array(all_rewards),
        env_ids=np.array(all_env_ids, dtype=np.int32),
        env_params=np.array(all_env_params),
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TRAIN_CONFIGS = [
    {"LINK_MASS_1": 1.0, "LINK_MASS_2": 1.0,
     "LINK_LENGTH_1": 1.0, "LINK_LENGTH_2": 1.0},
    {"LINK_MASS_1": 0.8, "LINK_MASS_2": 1.3,
     "LINK_LENGTH_1": 0.9, "LINK_LENGTH_2": 1.2},
    {"LINK_MASS_1": 1.2, "LINK_MASS_2": 0.7,
     "LINK_LENGTH_1": 1.3, "LINK_LENGTH_2": 0.8},
    {"LINK_MASS_1": 1.5, "LINK_MASS_2": 1.5,
     "LINK_LENGTH_1": 0.7, "LINK_LENGTH_2": 1.4},
    {"LINK_MASS_1": 0.7, "LINK_MASS_2": 0.9,
     "LINK_LENGTH_1": 1.1, "LINK_LENGTH_2": 1.1},
]

STATE_NAMES = ["s0", "s1", "s2", "s3"]
ACTION_NAMES = ["action"]
PARAM_NAMES = [
    "LINK_MASS_1", "LINK_MASS_2",
    "LINK_LENGTH_1", "LINK_LENGTH_2",
]
ANGULAR_DIMS = (0, 1)


# ---------------------------------------------------------------------------
# Tests: Structure detection
# ---------------------------------------------------------------------------


class TestDetectStructure:
    """Test multi-DOF structure detection."""

    def test_2dof_detected(self):
        decomposer = LagrangianDecomposer()
        structure = decomposer.detect_structure(
            STATE_NAMES, ANGULAR_DIMS, action_dim=1,
        )
        assert structure is not None
        assert structure.n_dof == 2
        assert structure.angle_dims == (0, 1)
        assert structure.velocity_dims == (2, 3)
        assert structure.actuated_dofs == (1,)

    def test_1dof_returns_none(self):
        decomposer = LagrangianDecomposer()
        result = decomposer.detect_structure(["s0", "s1"], (0,), 1)
        assert result is None

    def test_no_angles_returns_none(self):
        decomposer = LagrangianDecomposer()
        result = decomposer.detect_structure(
            ["s0", "s1", "s2", "s3"], (), 1,
        )
        assert result is None


# ---------------------------------------------------------------------------
# Tests: RK4 forward model
# ---------------------------------------------------------------------------


class TestRK4Forward:
    """Test that _rk4_forward matches true Acrobot dynamics."""

    def test_matches_true_dynamics(self):
        """RK4 with true coefficients should match Gymnasium dynamics."""
        cfg = TRAIN_CONFIGS[0]
        coeffs = _compute_initial_guess(cfg)

        rng = np.random.default_rng(123)
        for _ in range(20):
            state = np.array([
                rng.uniform(-pi, pi),
                rng.uniform(-pi, pi),
                rng.uniform(-4, 4),
                rng.uniform(-6, 6),
            ])
            torque = rng.uniform(-1, 1)

            ns_true = _rk4_step_true(
                cfg["LINK_MASS_1"], cfg["LINK_MASS_2"],
                cfg["LINK_LENGTH_1"], cfg["LINK_LENGTH_2"],
                state, torque,
            )
            ns_model = _rk4_forward(
                state, torque, coeffs, 0.2, MAX_VEL_1, MAX_VEL_2,
            )
            np.testing.assert_allclose(ns_model, ns_true, atol=1e-10)

    def test_matches_different_env(self):
        """RK4 with env-specific coefficients should match."""
        cfg = TRAIN_CONFIGS[1]  # m1=0.8, m2=1.3, l1=0.9, l2=1.2
        coeffs = _compute_initial_guess(cfg)

        rng = np.random.default_rng(456)
        for _ in range(20):
            state = np.array([
                rng.uniform(-pi, pi),
                rng.uniform(-pi, pi),
                rng.uniform(-3, 3),
                rng.uniform(-5, 5),
            ])
            torque = rng.uniform(-1, 1)

            ns_true = _rk4_step_true(
                cfg["LINK_MASS_1"], cfg["LINK_MASS_2"],
                cfg["LINK_LENGTH_1"], cfg["LINK_LENGTH_2"],
                state, torque,
            )
            ns_model = _rk4_forward(
                state, torque, coeffs, 0.2, MAX_VEL_1, MAX_VEL_2,
            )
            np.testing.assert_allclose(ns_model, ns_true, atol=1e-10)


# ---------------------------------------------------------------------------
# Tests: Initial guess
# ---------------------------------------------------------------------------


class TestInitialGuess:
    """Test physics-informed initial guess."""

    def test_default_env(self):
        """Default env (m=l=1) should give known coefficient values."""
        cfg = TRAIN_CONFIGS[0]
        coeffs = _compute_initial_guess(cfg)
        assert len(coeffs) == 8
        # d1_const = 0.25 + 1 + 0.25 + 2 = 3.5
        assert abs(coeffs[0] - 3.5) < 1e-10
        # d1_cos = 2*1*1*0.5 = 1.0
        assert abs(coeffs[1] - 1.0) < 1e-10
        # d2_const = 0.25 + 1 = 1.25
        assert abs(coeffs[2] - 1.25) < 1e-10
        # d3 = 1.25
        assert abs(coeffs[4] - 1.25) < 1e-10
        # h = 0.5
        assert abs(coeffs[5] - 0.5) < 1e-10
        # g_sin1 = 1.5*9.8 = 14.7
        assert abs(coeffs[6] - 14.7) < 1e-10
        # g_sin12 = 0.5*9.8 = 4.9
        assert abs(coeffs[7] - 4.9) < 1e-10


# ---------------------------------------------------------------------------
# Tests: NLS per-env regression
# ---------------------------------------------------------------------------


class TestNLSRegression:
    """Test NLS coefficient fitting."""

    def test_recovers_true_coefficients(self):
        """NLS should recover the true EL coefficients from data."""
        cfg = TRAIN_CONFIGS[0]
        dataset = _generate_acrobot_dataset([cfg], n_per_env=500)
        decomposer = LagrangianDecomposer()
        structure = decomposer.detect_structure(
            STATE_NAMES, ANGULAR_DIMS, action_dim=1,
        )
        assert structure is not None

        env_params = {0: cfg}
        per_env_coeffs, per_env_r2 = (
            decomposer.fit_per_env_coefficients_nls(
                dataset, structure, env_params,
            )
        )

        assert 0 in per_env_coeffs
        assert per_env_r2[0] > 0.999, f"R2={per_env_r2[0]:.6f}"

        # Check recovered coefficients match true values
        true_coeffs = _compute_initial_guess(cfg)
        np.testing.assert_allclose(
            per_env_coeffs[0], true_coeffs, rtol=0.01,
        )

    def test_multi_env_r2(self):
        """All environments should have high R2."""
        dataset = _generate_acrobot_dataset(
            TRAIN_CONFIGS[:3], n_per_env=500,
        )
        decomposer = LagrangianDecomposer()
        structure = decomposer.detect_structure(
            STATE_NAMES, ANGULAR_DIMS, action_dim=1,
        )
        assert structure is not None

        env_params = {i: c for i, c in enumerate(TRAIN_CONFIGS[:3])}
        _, per_env_r2 = decomposer.fit_per_env_coefficients_nls(
            dataset, structure, env_params,
        )

        for eid, r2 in per_env_r2.items():
            assert r2 > 0.99, f"Env {eid} R2={r2:.6f}"


# ---------------------------------------------------------------------------
# Tests: Parametric template fitting
# ---------------------------------------------------------------------------


class TestParametricTemplates:
    """Test template fitting across environments."""

    def test_recovers_true_formulas(self):
        """Templates should recover known coefficient formulas."""
        import sympy

        dataset = _generate_acrobot_dataset(
            TRAIN_CONFIGS, n_per_env=500,
        )
        decomposer = LagrangianDecomposer()
        structure = decomposer.detect_structure(
            STATE_NAMES, ANGULAR_DIMS, action_dim=1,
        )
        assert structure is not None

        env_params = {i: c for i, c in enumerate(TRAIN_CONFIGS)}
        per_env_coeffs, _ = decomposer.fit_per_env_coefficients_nls(
            dataset, structure, env_params,
        )
        templates = decomposer.fit_parametric_templates(
            per_env_coeffs, env_params, PARAM_NAMES, COEFF_NAMES,
        )

        # Verify at default env
        m1s, m2s = sympy.symbols("LINK_MASS_1 LINK_MASS_2")
        l1s, l2s = sympy.symbols("LINK_LENGTH_1 LINK_LENGTH_2")
        subs = {m1s: 1.0, m2s: 1.0, l1s: 1.0, l2s: 1.0}

        d1_val = float(templates["d1_const"].subs(subs))
        assert abs(d1_val - 3.5) < 0.05, f"d1_const={d1_val}"

        d3_val = float(templates["d3"].subs(subs))
        assert abs(d3_val - 1.25) < 0.05, f"d3={d3_val}"

        gs1_val = float(templates["g_sin1"].subs(subs))
        assert abs(gs1_val - 14.7) < 0.2, f"g_sin1={gs1_val}"

        gs12_val = float(templates["g_sin12"].subs(subs))
        assert abs(gs12_val - 4.9) < 0.1, f"g_sin12={gs12_val}"


# ---------------------------------------------------------------------------
# Tests: Composed forward dynamics
# ---------------------------------------------------------------------------


class TestComposedDynamics:
    """Test that composed dynamics match true Acrobot dynamics."""

    def test_composed_r2_high(self):
        """Composed dynamics should achieve high R2 on velocity dims."""
        dataset = _generate_acrobot_dataset(
            TRAIN_CONFIGS, n_per_env=500,
        )
        decomposer = LagrangianDecomposer()
        result = decomposer.decompose(
            dataset, STATE_NAMES, ACTION_NAMES,
            PARAM_NAMES, ANGULAR_DIMS,
        )
        assert result is not None

        # The composed expression uses Euler (dt*ddtheta) but data is
        # RK4-integrated, so R2 won't be perfect. But should be > 0.90.
        for dim_idx, r2 in result.composed_r2.items():
            assert r2 > 0.90, (
                f"Composed R2 for dim {dim_idx} = {r2:.4f}"
            )


# ---------------------------------------------------------------------------
# Tests: Full decompose pipeline
# ---------------------------------------------------------------------------


class TestFullDecompose:
    """Test the full decompose() method."""

    def test_returns_result(self):
        dataset = _generate_acrobot_dataset(
            TRAIN_CONFIGS[:3], n_per_env=500,
        )
        decomposer = LagrangianDecomposer()
        result = decomposer.decompose(
            dataset, STATE_NAMES, ACTION_NAMES,
            PARAM_NAMES, ANGULAR_DIMS,
        )
        assert result is not None
        assert 2 in result.dynamics_expressions
        assert 3 in result.dynamics_expressions
        assert len(result.per_env_r2) == 3
        assert len(result.coefficient_names) == 8

    def test_returns_none_for_1dof(self):
        dataset = _generate_acrobot_dataset(
            TRAIN_CONFIGS[:1], n_per_env=100,
        )
        decomposer = LagrangianDecomposer()
        result = decomposer.decompose(
            dataset, ["s0", "s1"], ["action"],
            PARAM_NAMES, (0,),
        )
        assert result is None

    def test_per_env_r2(self):
        dataset = _generate_acrobot_dataset(
            TRAIN_CONFIGS[:3], n_per_env=500,
        )
        decomposer = LagrangianDecomposer()
        result = decomposer.decompose(
            dataset, STATE_NAMES, ACTION_NAMES,
            PARAM_NAMES, ANGULAR_DIMS,
        )
        assert result is not None
        for eid in range(3):
            assert eid in result.per_env_r2
            assert result.per_env_r2[eid] > 0.99
