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
from circ_rl.training.circ_trainer import CIRCTrainer, TrainingConfig


def main() -> None:
    """Train on Pendulum family and save periodic checkpoints."""
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: {}", device)

    # -- Configuration --
    n_envs = 6
    n_transitions_discovery = 2000
    n_train_iterations = 1000
    n_steps_per_env = 1024
    n_workers = 4
    checkpoint_every = 50
    env_param_names = ["g", "m", "l"]
    output_dir = "experiments/outputs"
    os.makedirs(output_dir, exist_ok=True)

    # -- 1. Create environment family --
    print("=" * 60)
    print("PENDULUM CIRC-RL TRAINING (1000 iterations)")
    print("=" * 60)

    env_family = EnvironmentFamily.from_gymnasium(
        base_env="Pendulum-v1",
        param_distributions={
            "g": (7.0, 13.0),
            "m": (0.5, 2.0),
            "l": (0.5, 1.5),
        },
        n_envs=n_envs,
        seed=42,
    )

    print(f"\nDevice: {device}")
    print(f"Workers: {n_workers}")
    print(f"\nEnvironment parameters:")
    for i in range(n_envs):
        p = env_family.get_env_params(i)
        print(f"  env {i}: g={p['g']:.2f}, m={p['m']:.2f}, l={p['l']:.2f}")

    # -- 2. Causal discovery --
    print("\n--- Phase 1: Causal Discovery ---")
    collector = DataCollector(env_family)
    dataset = collector.collect(
        n_transitions_per_env=n_transitions_discovery, seed=42
    )

    state_dim = dataset.state_dim
    action_space = env_family.action_space
    action_dim = action_space.shape[0]  # type: ignore[union-attr]
    state_names = [f"s{i}" for i in range(state_dim)]
    action_names = [f"action_{i}" for i in range(action_dim)]
    next_state_names = [f"s{i}_next" for i in range(state_dim)]
    node_names = state_names + action_names + ["reward"] + next_state_names

    graph = CausalGraphBuilder.discover(
        dataset, node_names, method="pc", alpha=0.05
    )
    print(f"Graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges")

    # -- 3. Feature selection --
    print("\n--- Phase 2: Feature Selection ---")
    selector = InvFeatureSelector(epsilon=0.15, min_ate=0.01)
    fs_result = selector.select(dataset, graph, state_names)
    print(f"Selected: {fs_result.selected_features}")

    feature_mask = fs_result.feature_mask
    if not any(feature_mask):
        print("No features selected -- using all features")
        feature_mask = np.ones(state_dim, dtype=bool)

    # -- 4. Train policy --
    print("\n--- Phase 3: Policy Optimization ---")
    action_low = action_space.low   # type: ignore[union-attr]
    action_high = action_space.high  # type: ignore[union-attr]

    policy = CausalPolicy(
        full_state_dim=state_dim,
        feature_mask=feature_mask,
        action_dim=action_dim,
        hidden_dims=(128, 128),
        continuous=True,
        action_low=action_low,
        action_high=action_high,
        context_dim=len(env_param_names),
    )
    policy.to(device)

    config = TrainingConfig(
        n_iterations=n_train_iterations,
        gamma=0.99,
        gae_lambda=0.95,
        learning_rate=3e-4,
        clip_epsilon=0.2,
        n_steps_per_env=n_steps_per_env,
        n_ppo_epochs=10,
        mini_batch_size=64,
        irm_weight=0.1,
        worst_case_temperature=1.0,
        worst_case_variance_weight=0.1,
        entropy_coef=0.01,
    )

    trainer = CIRCTrainer(
        policy=policy,
        env_family=env_family,
        config=config,
        n_rollout_workers=n_workers,
        env_param_names=env_param_names,
    )
    trainer.to(device)

    ckpt_path = os.path.join(output_dir, "pendulum_policy.pt")
    t0 = time.time()

    def _checkpoint_callback(iteration: int, metrics: object) -> None:
        if (iteration + 1) % checkpoint_every == 0:
            torch.save(policy.state_dict(), ckpt_path)
            elapsed = time.time() - t0
            eta_total = elapsed / (iteration + 1) * n_train_iterations
            eta_remaining = eta_total - elapsed
            print(
                f"  [iter {iteration + 1}] Checkpoint saved. "
                f"Elapsed: {elapsed / 60:.0f}min, "
                f"ETA: {eta_remaining / 60:.0f}min"
            )

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

    # Final checkpoint
    torch.save(policy.state_dict(), ckpt_path)
    print(f"  Final checkpoint saved to {ckpt_path}")


if __name__ == "__main__":
    main()
