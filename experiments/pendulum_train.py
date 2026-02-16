"""Train CIRC-RL on Pendulum-v1 across multiple physics variants.

Training-only script with periodic checkpoint saving.  Video recording
is handled separately by ``pendulum_record.py``.

Usage::

    uv run python experiments/pendulum_train.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch
from loguru import logger

from circ_rl.causal_discovery.builder import CausalGraphBuilder
from circ_rl.environments.data_collector import DataCollector
from circ_rl.environments.env_family import EnvironmentFamily
from circ_rl.feature_selection.inv_feature_selector import InvFeatureSelector
from circ_rl.policy.causal_policy import CausalPolicy
from circ_rl.training.circ_trainer import CIRCTrainer, IterationMetrics, TrainingConfig


def main() -> None:
    """Train on Pendulum family and save periodic checkpoints."""
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: {}", device)

    # -- Configuration --
    n_envs = 16
    n_transitions_discovery = 2000
    n_train_iterations = 400
    n_steps_per_env = 1024
    n_workers = 4
    checkpoint_every = 25
    env_param_names = ["g", "m", "l"]
    output_dir = "experiments/outputs"
    os.makedirs(output_dir, exist_ok=True)

    # -- 1. Create environment family --
    print("=" * 60)
    print("PENDULUM CIRC-RL TRAINING (full causal pipeline)")
    print("=" * 60)

    max_torque = 100.0  # effectively unlimited torque
    env_family = EnvironmentFamily.from_gymnasium(
        base_env="Pendulum-v1",
        param_distributions={
            "g": (7.0, 13.0),
            "m": (0.5, 2.0),
            "l": (0.5, 1.5),
        },
        n_envs=n_envs,
        seed=42,
        fixed_params={"max_torque": max_torque},
    )

    print(f"\nDevice: {device}")
    print(f"Workers: {n_workers}")
    print(f"\nEnvironment parameters:")
    for i in range(n_envs):
        p = env_family.get_env_params(i)
        print(f"  env {i}: g={p['g']:.2f}, m={p['m']:.2f}, l={p['l']:.2f}")

    # -- 2. Causal discovery (with env params in the graph) --
    print("\n--- Phase 1: Causal Discovery ---")
    collector = DataCollector(env_family, include_env_params=True)
    dataset = collector.collect(
        n_transitions_per_env=n_transitions_discovery, seed=42
    )

    state_dim = dataset.state_dim
    action_space = env_family.action_space
    action_dim = action_space.shape[0]  # type: ignore[union-attr]
    state_names = [f"s{i}" for i in range(state_dim)]
    action_names = [f"action_{i}" for i in range(action_dim)]
    next_state_names = [f"s{i}_next" for i in range(state_dim)]
    ep_names = [f"ep_{name}" for name in env_param_names]
    node_names = state_names + action_names + ["reward"] + next_state_names + ep_names

    graph = CausalGraphBuilder.discover(
        dataset, node_names, method="pc", alpha=0.05,
        env_param_names=ep_names, ep_correlation_threshold=0.05,
        state_feature_names=state_names, nonlinear_state_reward_screen=True,
    )
    print(f"Graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
    print(f"Edges: {sorted(graph.edges)}")
    ep_ancestors = graph.env_param_ancestors_of("reward")
    print(f"Env-param ancestors of reward: {ep_ancestors}")

    # -- 3. Feature selection (mechanism invariance + soft weights) --
    print("\n--- Phase 2: Feature Selection ---")
    selector = InvFeatureSelector(
        epsilon=0.15, min_ate=0.01,
        use_mechanism_invariance=True,
        poly_degree=2,
        min_weight=0.1,
        skip_ancestor_check=True,
    )
    fs_result = selector.select(dataset, graph, state_names)
    print(f"Selected: {fs_result.selected_features}")
    print(f"Weights: { {n: f'{w:.3f}' for n, w in zip(state_names, fs_result.feature_weights)} }")
    print(f"Mechanism p-values: {fs_result.mechanism_p_values}")
    print(f"Rejected: {fs_result.rejected_features}")

    feature_weights = fs_result.feature_weights
    if not any(feature_weights > 0):
        print("No features selected -- using all features")
        feature_weights = np.ones(state_dim, dtype=np.float32)

    # All env params passed as context (not driven by conditional invariance)
    rollout_param_names = env_param_names
    context_dim = len(rollout_param_names)
    print(f"Context params for policy: {rollout_param_names} (dim={context_dim})")

    # -- 3.5. Transition dynamics analysis --
    print("\n--- Phase 2.5: Transition Dynamics Analysis ---")
    from circ_rl.feature_selection.transition_analyzer import TransitionAnalyzer

    analyzer = TransitionAnalyzer()
    transition_result = analyzer.analyze(dataset, state_names, action_dim)
    print(f"Dynamics scales per env: {[f'{s:.4f}' for s in transition_result.dynamics_scales]}")
    print(f"Reference scale: {transition_result.reference_scale:.4f}")
    for name, r2 in transition_result.per_dim_loeo_r2.items():
        status = "invariant" if r2 >= 0.9 else "VARIANT"
        print(f"  {name}: LOEO R^2 = {r2:.4f} ({status})")

    # -- 4. Train policy --
    print("\n--- Phase 3: Policy Optimization ---")
    action_low = np.array([-max_torque], dtype=np.float32)
    action_high = np.array([max_torque], dtype=np.float32)

    policy = CausalPolicy(
        full_state_dim=state_dim,
        feature_mask=feature_weights,
        action_dim=action_dim,
        hidden_dims=(64, 64),
        continuous=True,
        action_low=action_low,
        action_high=action_high,
        context_dim=context_dim,
        use_dynamics_normalization=True,
        dynamics_reference_scale=transition_result.reference_scale,
    )
    policy.to(device)

    # Resume from last checkpoint if available
    resume_ckpt = os.path.join(output_dir, "pendulum_policy.pt")
    resume_iter = 0
    if os.path.exists(resume_ckpt):
        print(f"  Resuming from checkpoint: {resume_ckpt}")
        state_dict = torch.load(resume_ckpt, map_location=device, weights_only=True)
        policy.load_state_dict(state_dict)
        # Infer resume iteration from numbered checkpoints
        for i in range(n_train_iterations, 0, -1):
            tag = f"{i:03d}"
            if os.path.exists(os.path.join(output_dir, f"pendulum_policy_iter{tag}.pt")):
                resume_iter = i
                break
        print(f"  Resuming from iteration {resume_iter}")

    dynamics_scale_map = {
        i: float(transition_result.dynamics_scales[i])
        for i in range(n_envs)
    }

    remaining_iterations = n_train_iterations - resume_iter
    config = TrainingConfig(
        n_iterations=remaining_iterations,
        gamma=0.99,
        gae_lambda=0.95,
        learning_rate=1e-4,
        clip_epsilon=0.2,
        n_steps_per_env=n_steps_per_env,
        n_ppo_epochs=10,
        mini_batch_size=64,
        irm_weight=0.5,
        worst_case_temperature=1.0,
        worst_case_variance_weight=0.5,
        entropy_coef=0.01,
        dynamics_aux_weight=1.0,
    )

    trainer = CIRCTrainer(
        policy=policy,
        env_family=env_family,
        config=config,
        n_rollout_workers=n_workers,
        env_param_names=rollout_param_names,
        dynamics_scales=dynamics_scale_map,
    )
    trainer.to(device)

    ckpt_path = os.path.join(output_dir, "pendulum_policy.pt")
    best_ckpt_path = os.path.join(output_dir, "pendulum_policy_best.pt")
    t0 = time.time()
    best_worst = float("-inf")

    def _checkpoint_callback(iteration: int, metrics: IterationMetrics) -> None:
        nonlocal best_worst
        abs_iter = resume_iter + iteration + 1  # absolute iteration number
        # Periodic numbered checkpoint
        if abs_iter % checkpoint_every == 0:
            iter_tag = f"{abs_iter:03d}"
            numbered_path = os.path.join(
                output_dir, f"pendulum_policy_iter{iter_tag}.pt",
            )
            torch.save(policy.state_dict(), numbered_path)
            torch.save(policy.state_dict(), ckpt_path)
            elapsed = time.time() - t0
            eta_total = elapsed / (iteration + 1) * remaining_iterations
            eta_remaining = eta_total - elapsed
            print(
                f"  [iter {abs_iter}] Checkpoint saved: {numbered_path}. "
                f"Elapsed: {elapsed / 60:.0f}min, "
                f"ETA: {eta_remaining / 60:.0f}min"
            )
        # Best worst-case checkpoint (saved every time a new best is found)
        if metrics.worst_env_return > best_worst:
            best_worst = metrics.worst_env_return
            torch.save(policy.state_dict(), best_ckpt_path)

    metrics_history = trainer.run(iteration_callback=_checkpoint_callback)

    # Print training summary
    if metrics_history:
        first = metrics_history[0]
        last = metrics_history[-1]
        elapsed = time.time() - t0
        print(f"\nTraining complete ({len(metrics_history)} iterations "
              f"in {elapsed / 60:.0f} minutes):")
        print(f"  First: mean_return={first.mean_return:.1f}, "
              f"worst={first.worst_env_return:.1f}")
        print(f"  Last:  mean_return={last.mean_return:.1f}, "
              f"worst={last.worst_env_return:.1f}")
        print(f"  Best worst-case return: {best_worst:.1f}")

    # Final checkpoint
    torch.save(policy.state_dict(), ckpt_path)
    print(f"  Final checkpoint saved to {ckpt_path}")
    print(f"  Best checkpoint saved to {best_ckpt_path}")


if __name__ == "__main__":
    main()
