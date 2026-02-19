# ruff: noqa: T201
"""Diagnostic: why does PySR struggle with delta_s2?

Checks:
1. Ground truth model R2 for delta_s2 (with/without velocity clipping)
2. Fraction of data affected by velocity clipping
3. Data scale and distribution analysis
4. Whether the nested_constraints block the true expression
5. What expressions PySR finds with different configurations
"""
from __future__ import annotations

import numpy as np
import sympy

from circ_rl.environments.data_collector import DataCollector, ExploratoryDataset
from circ_rl.environments.env_family import EnvironmentFamily
from circ_rl.observation_analysis.observation_analyzer import ObservationAnalyzer
from circ_rl.utils.seeding import seed_everything


def main() -> None:
    seed_everything(42)

    # Create same training family as pendulum_compare.py
    train_family = EnvironmentFamily.from_gymnasium(
        base_env="Pendulum-v1",
        param_distributions={
            "g": (8.0, 12.0),
            "m": (0.8, 1.5),
            "l": (0.7, 1.3),
        },
        n_envs=25,
        seed=42,
    )

    # Collect data
    collector = DataCollector(train_family, include_env_params=True)
    dataset = collector.collect(n_transitions_per_env=5000, seed=42)
    env_param_names = ["g", "m", "l"]

    # Observation analysis -> canonical coords
    analyzer = ObservationAnalyzer()
    oa_result = analyzer.analyze(dataset, ["s0", "s1", "s2"])
    print(f"Canonical coords: {oa_result.canonical_state_names}")

    canonical_dataset = ExploratoryDataset(
        states=oa_result.canonical_states,
        actions=dataset.actions,
        next_states=oa_result.canonical_next_states,
        rewards=dataset.rewards,
        env_ids=dataset.env_ids,
        env_params=dataset.env_params,
    )

    # Extract arrays
    phi_0 = canonical_dataset.states[:, 0]  # theta (angle)
    s2 = canonical_dataset.states[:, 1]  # thdot (angular velocity)
    phi_0_next = canonical_dataset.next_states[:, 0]
    s2_next = canonical_dataset.next_states[:, 1]
    action = dataset.actions.ravel()

    assert dataset.env_params is not None
    g_arr = dataset.env_params[:, 0]
    m_arr = dataset.env_params[:, 1]
    l_arr = dataset.env_params[:, 2]

    delta_s2 = s2_next - s2

    # ======================================================================
    # 1. Basic statistics
    # ======================================================================
    print("\n=== 1. Data statistics ===")
    print(f"  N samples: {len(delta_s2)}")
    print(f"  delta_s2: mean={delta_s2.mean():.6f}, std={delta_s2.std():.6f}, "
          f"min={delta_s2.min():.4f}, max={delta_s2.max():.4f}")
    print(f"  s2 (thdot): mean={s2.mean():.4f}, std={s2.std():.4f}, "
          f"min={s2.min():.4f}, max={s2.max():.4f}")
    print(f"  action: mean={action.mean():.4f}, std={action.std():.4f}, "
          f"min={action.min():.4f}, max={action.max():.4f}")
    print(f"  phi_0: mean={phi_0.mean():.4f}, std={phi_0.std():.4f}")
    print(f"  g: [{g_arr.min():.2f}, {g_arr.max():.2f}]")
    print(f"  m: [{m_arr.min():.2f}, {m_arr.max():.2f}]")
    print(f"  l: [{l_arr.min():.2f}, {l_arr.max():.2f}]")

    # ======================================================================
    # 2. Ground truth models
    # ======================================================================
    print("\n=== 2. Ground truth models ===")

    dt = 0.05

    # Pendulum-v1 dynamics (semi-implicit Euler):
    #   newthdot = thdot + (-3g/(2l)*sin(th+pi) + 3/(ml^2)*u) * dt
    #   newthdot_clipped = clip(newthdot, -8, 8)
    #   delta_s2 = newthdot_clipped - thdot

    # Model A: no-clip acceleration
    accel = -3.0 * g_arr / (2.0 * l_arr) * np.sin(phi_0 + np.pi) + \
            3.0 / (m_arr * l_arr**2) * action
    pred_a = accel * dt
    r2_a = 1.0 - np.var(delta_s2 - pred_a) / np.var(delta_s2)
    print(f"  Model A (accel*dt, no clip):  R2={r2_a:.6f}")

    # Model B: with velocity clipping
    new_s2 = s2 + accel * dt
    new_s2_clipped = np.clip(new_s2, -8.0, 8.0)
    pred_b = new_s2_clipped - s2
    r2_b = 1.0 - np.var(delta_s2 - pred_b) / np.var(delta_s2)
    print(f"  Model B (with clipping):      R2={r2_b:.6f}")

    # ======================================================================
    # 3. Velocity clipping analysis
    # ======================================================================
    print("\n=== 3. Velocity clipping analysis ===")
    would_clip = np.abs(new_s2) > 8.0
    n_clip = would_clip.sum()
    print(f"  Transitions hitting clip: {n_clip}/{len(delta_s2)} "
          f"({100.0 * n_clip / len(delta_s2):.2f}%)")

    # Of those that clip, how big is the error?
    if n_clip > 0:
        clip_error = np.abs(pred_a[would_clip] - delta_s2[would_clip])
        print(f"  Clipped transitions: mean_error={clip_error.mean():.6f}, "
              f"max_error={clip_error.max():.6f}")

    # R2 on non-clipped data only
    mask_noclip = ~would_clip
    r2_a_noclip = 1.0 - np.var(delta_s2[mask_noclip] - pred_a[mask_noclip]) / \
                  np.var(delta_s2[mask_noclip])
    print(f"  Model A R2 on non-clipped data: {r2_a_noclip:.6f}")
    print(f"  Total delta_s2 variance: {np.var(delta_s2):.6f}")
    print(f"  Non-clipped delta_s2 variance: {np.var(delta_s2[mask_noclip]):.6f}")

    # Distribution of s2 values
    pct_high = np.mean(np.abs(s2) > 6.0) * 100
    pct_very_high = np.mean(np.abs(s2) > 7.0) * 100
    print(f"  |s2| > 6: {pct_high:.1f}%, |s2| > 7: {pct_very_high:.1f}%")

    # ======================================================================
    # 4. Term-by-term analysis for PySR search
    # ======================================================================
    print("\n=== 4. Term-by-term analysis ===")

    # The true expression: 0.075*g*sin(phi_0)/l + 0.15*action/(m*l^2)
    # (using sin(phi_0+pi) = -sin(phi_0))
    gravity_term = 0.075 * g_arr * np.sin(phi_0) / l_arr
    torque_term = 0.15 * action / (m_arr * l_arr**2)

    print(f"  gravity term (0.075*g*sin(phi_0)/l): "
          f"mean={gravity_term.mean():.6f}, std={gravity_term.std():.6f}")
    print(f"  torque term (0.15*action/(m*l^2)):   "
          f"mean={torque_term.mean():.6f}, std={torque_term.std():.6f}")
    print(f"  Ratio of std(gravity)/std(torque): "
          f"{gravity_term.std()/torque_term.std():.4f}")

    # How much variance does each term explain?
    r2_grav = 1.0 - np.var(delta_s2 - gravity_term) / np.var(delta_s2)
    r2_torq = 1.0 - np.var(delta_s2 - torque_term) / np.var(delta_s2)
    print(f"  R2 from gravity term alone: {r2_grav:.6f}")
    print(f"  R2 from torque term alone:  {r2_torq:.6f}")

    # ======================================================================
    # 5. Complexity analysis of true expression
    # ======================================================================
    print("\n=== 5. Expression complexity ===")
    phi_0_sym, s2_sym = sympy.symbols("phi_0 s2")
    action_sym = sympy.Symbol("action")
    g_sym, m_sym, l_sym = sympy.symbols("g m l")

    # True delta_s2 expression
    true_expr = (
        sympy.Rational(3, 40) * g_sym * sympy.sin(phi_0_sym) / l_sym
        + sympy.Rational(3, 20) * action_sym / (m_sym * l_sym**2)
    )
    print(f"  True expression: {true_expr}")

    from circ_rl.hypothesis.expression import SymbolicExpression

    se = SymbolicExpression.from_sympy(true_expr)
    print(f"  Tree node count: {se.complexity}")

    # With float constants (like PySR would find)
    approx_expr = (
        0.075 * g_sym * sympy.sin(phi_0_sym) / l_sym
        + 0.15 * action_sym / (m_sym * l_sym**2)
    )
    se2 = SymbolicExpression.from_sympy(approx_expr)
    print(f"  Approx expression: {approx_expr}")
    print(f"  Approx tree nodes: {se2.complexity}")

    # ======================================================================
    # 6. What PySR sees: feature columns
    # ======================================================================
    print("\n=== 6. Feature analysis (what PySR gets) ===")

    # PySR gets [phi_0, s2, action, g, m, l] as columns
    # and target delta_s2
    feature_names = ["phi_0", "s2", "action", "g", "m", "l"]
    features = np.column_stack([
        phi_0, s2, action, g_arr, m_arr, l_arr,
    ])

    # Correlation of features with delta_s2
    print(f"  Correlations with delta_s2:")
    for i, name in enumerate(feature_names):
        corr = np.corrcoef(features[:, i], delta_s2)[0, 1]
        print(f"    {name:>8s}: r={corr:+.4f}")

    # The key issue: action/(m*l^2) is hard for PySR because
    # it needs to discover the interaction g*sin(phi_0)/l AND action/(m*l^2)
    # simultaneously. Let's see what simpler models PySR might find.

    # ======================================================================
    # 7. Per-env R2 of the true model
    # ======================================================================
    print("\n=== 7. Per-env R2 of true model ===")
    unique_envs = sorted(set(dataset.env_ids.tolist()))
    print(f"  {'Env':>4s}  {'g':>6s}  {'m':>6s}  {'l':>6s}  "
          f"{'R2_nocl':>10s}  {'R2_clip':>10s}  {'%clip':>7s}")
    for env_id in unique_envs[:10]:  # First 10
        mask = dataset.env_ids == env_id
        ep = dataset.env_params[mask][0]
        y_env = delta_s2[mask]
        p_a = pred_a[mask]
        p_b = pred_b[mask]
        r2_env_a = 1.0 - np.var(y_env - p_a) / np.var(y_env)
        r2_env_b = 1.0 - np.var(y_env - p_b) / np.var(y_env)
        pct_clip_env = 100.0 * would_clip[mask].sum() / mask.sum()
        print(f"  {env_id:4d}  {ep[0]:6.2f}  {ep[1]:6.2f}  {ep[2]:6.2f}  "
              f"{r2_env_a:10.6f}  {r2_env_b:10.6f}  {pct_clip_env:6.1f}%")
    print(f"  ... (showing 10/{len(unique_envs)} envs)")

    # ======================================================================
    # 8. Subsampled data analysis (PySR uses max_samples=10000)
    # ======================================================================
    print("\n=== 8. Subsampled data (10000 samples, like PySR) ===")
    rng = np.random.default_rng(42)
    idx = rng.choice(len(delta_s2), 10000, replace=False)
    sub_delta = delta_s2[idx]
    sub_pred_a = pred_a[idx]
    sub_pred_b = pred_b[idx]
    sub_clip = would_clip[idx]

    r2_sub_a = 1.0 - np.var(sub_delta - sub_pred_a) / np.var(sub_delta)
    r2_sub_b = 1.0 - np.var(sub_delta - sub_pred_b) / np.var(sub_delta)
    pct_sub_clip = 100.0 * sub_clip.sum() / len(sub_clip)
    print(f"  Subsampled R2 (no clip): {r2_sub_a:.6f}")
    print(f"  Subsampled R2 (clipped): {r2_sub_b:.6f}")
    print(f"  Subsampled clip %: {pct_sub_clip:.1f}%")

    # How noisy is the target for PySR?
    residual_a = sub_delta - sub_pred_a
    print(f"  Residual after true model (no clip): "
          f"std={residual_a.std():.6f}, "
          f"as % of target std={100 * residual_a.std() / sub_delta.std():.1f}%")

    # ======================================================================
    # 9. Nested constraints analysis
    # ======================================================================
    print("\n=== 9. Nested constraints impact ===")
    print("  Current constraints: * and / cannot contain + or -")
    print("  True expression: 0.075*g*sin(phi_0)/l + 0.15*action/(m*l^2)")
    print("  Breaking down sub-expressions:")
    print("    Term 1: Mul(0.075, Div(Mul(g, sin(phi_0)), l))")
    print("      -> No + or - inside * or /, OK")
    print("    Term 2: Mul(0.15, Div(action, Mul(m, square(l))))")
    print("      -> No + or - inside * or /, OK")
    print("    Top level: Add(Term1, Term2)")
    print("      -> Addition is at the top level, OK")
    print("  Conclusion: nested constraints should NOT block the true expression")

    # ======================================================================
    # 10. Could s2 be confusing things?
    # ======================================================================
    print("\n=== 10. Does s2 correlate with delta_s2 spuriously? ===")
    corr_s2 = np.corrcoef(s2, delta_s2)[0, 1]
    print(f"  Correlation(s2, delta_s2) = {corr_s2:.4f}")

    # Simple linear regression of delta_s2 on s2
    slope = np.cov(s2, delta_s2)[0, 1] / np.var(s2)
    intercept = delta_s2.mean() - slope * s2.mean()
    pred_linear = slope * s2 + intercept
    r2_linear = 1.0 - np.var(delta_s2 - pred_linear) / np.var(delta_s2)
    print(f"  Linear(s2): slope={slope:.6f}, R2={r2_linear:.6f}")
    print("  (PySR might latch onto s2 as a predictor due to clipping correlation)")

    # The clipping creates a correlation: when s2 is high and positive,
    # delta_s2 tends to be negative (clipping pulls velocity back),
    # and vice versa.
    high_s2_mask = np.abs(s2) > 6.0
    if high_s2_mask.sum() > 0:
        corr_high = np.corrcoef(s2[high_s2_mask], delta_s2[high_s2_mask])[0, 1]
        print(f"  Correlation for |s2|>6: {corr_high:.4f}")
        corr_low = np.corrcoef(s2[~high_s2_mask], delta_s2[~high_s2_mask])[0, 1]
        print(f"  Correlation for |s2|<6: {corr_low:.4f}")


if __name__ == "__main__":
    main()
