"""Train CIRC-RL on Pendulum-v1 and record videos in several environments.

Trains a CausalPolicy across a Pendulum family with varied gravity,
mass, and length, then records side-by-side videos of the final model
acting in environments with different physics parameters.

Usage::

    uv run python experiments/pendulum_video.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch
from loguru import logger

from circ_rl.causal_discovery.builder import CausalGraphBuilder
from circ_rl.environments.data_collector import DataCollector
from circ_rl.environments.env_family import EnvironmentFamily
from circ_rl.feature_selection.inv_feature_selector import InvFeatureSelector
from circ_rl.policy.causal_policy import CausalPolicy
from circ_rl.training.circ_trainer import CIRCTrainer, TrainingConfig
from circ_rl.training.rollout import RolloutWorker


def _record_episode(
    env_family: EnvironmentFamily,
    env_idx: int,
    policy: CausalPolicy,
    max_steps: int = 200,
    device: torch.device | None = None,
) -> tuple[list[np.ndarray], float]:
    """Record one episode with rgb_array frames.

    :param env_family: The environment family.
    :param env_idx: Which environment to record.
    :param policy: Trained policy.
    :param max_steps: Maximum episode length.
    :param device: Torch device for inference.
    :returns: (frames, total_reward) tuple.
    """
    import gymnasium as gym

    if device is None:
        device = torch.device("cpu")

    params = env_family.get_env_params(env_idx)
    env = gym.make("Pendulum-v1", render_mode="rgb_array")
    unwrapped = env.unwrapped
    for attr, value in params.items():
        setattr(unwrapped, attr, value)

    obs, _ = env.reset(seed=42)
    frames: list[np.ndarray] = []
    total_reward = 0.0

    for _ in range(max_steps):
        frame = env.render()
        assert isinstance(frame, np.ndarray)
        frames.append(frame)

        state_tensor = torch.tensor(
            obs, dtype=torch.float32
        ).unsqueeze(0).to(device)

        action_np = policy.get_action(state_tensor, deterministic=True)
        assert isinstance(action_np, np.ndarray)

        obs, reward, terminated, truncated, _ = env.step(action_np)
        total_reward += float(reward)

        if terminated or truncated:
            break

    env.close()
    return frames, total_reward


def _make_grid_video(
    all_frames: list[list[np.ndarray]],
    labels: list[str],
    output_path: str,
    fps: int = 30,
) -> None:
    """Combine multiple episode frame lists into a grid video.

    :param all_frames: List of frame lists (one per environment).
    :param labels: Label for each environment.
    :param output_path: Path to write the output mp4 file.
    :param fps: Frames per second.
    """
    import imageio.v3 as iio

    # Pad shorter episodes with their last frame
    max_len = max(len(f) for f in all_frames)
    for frames in all_frames:
        while len(frames) < max_len:
            frames.append(frames[-1].copy())

    # Determine grid layout: 2 columns
    n = len(all_frames)
    n_cols = min(n, 2)
    n_rows = (n + n_cols - 1) // n_cols

    # Add labels to frames
    grid_frames = []
    for t in range(max_len):
        row_images = []
        for r in range(n_rows):
            col_images = []
            for c in range(n_cols):
                idx = r * n_cols + c
                if idx < n:
                    frame = all_frames[idx][t].copy()
                    # Burn label into top-left corner
                    _add_text_to_frame(frame, labels[idx])
                    col_images.append(frame)
                else:
                    # Black padding
                    col_images.append(
                        np.zeros_like(all_frames[0][0])
                    )
            row_images.append(np.hstack(col_images))
        grid_frame = np.vstack(row_images)
        grid_frames.append(grid_frame)

    iio.imwrite(
        output_path,
        np.stack(grid_frames),
        fps=fps,
        codec="libx264",
    )
    logger.info("Video saved to {}", output_path)


def _add_text_to_frame(frame: np.ndarray, text: str) -> None:
    """Burn text label into top-left corner of an RGB frame.

    Uses a simple pixel-based approach (no PIL/cv2 dependency).
    Draws a dark background rectangle then white text characters
    from a minimal 5x7 bitmap font.
    """
    # Simple approach: draw a dark bar at the top with the text
    h, w = frame.shape[:2]
    bar_height = 30
    frame[:bar_height, :] = (frame[:bar_height, :].astype(np.float32) * 0.3).astype(
        np.uint8
    )

    # Minimal bitmap font for basic ASCII (uppercase, digits, common punctuation)
    _FONT = _get_bitmap_font()
    x_offset = 8
    y_offset = 5
    scale = 2

    for ch in text:
        glyph = _FONT.get(ch, _FONT.get("?", []))
        if not glyph:
            x_offset += 6 * scale
            continue
        for row_idx, row_bits in enumerate(glyph):
            for col_idx in range(5):
                if row_bits & (1 << (4 - col_idx)):
                    y = y_offset + row_idx * scale
                    x = x_offset + col_idx * scale
                    frame[y : y + scale, x : x + scale] = 255
        x_offset += 6 * scale


def _get_bitmap_font() -> dict[str, list[int]]:
    """Return a minimal 5x7 bitmap font for common characters.

    Each character is a list of 7 ints; each int encodes 5 pixels
    (bit 4 = leftmost, bit 0 = rightmost).
    """
    return {
        "0": [0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E],
        "1": [0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E],
        "2": [0x0E, 0x11, 0x01, 0x06, 0x08, 0x10, 0x1F],
        "3": [0x0E, 0x11, 0x01, 0x06, 0x01, 0x11, 0x0E],
        "4": [0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02],
        "5": [0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E],
        "6": [0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E],
        "7": [0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08],
        "8": [0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E],
        "9": [0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C],
        ".": [0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C],
        "-": [0x00, 0x00, 0x00, 0x1F, 0x00, 0x00, 0x00],
        "=": [0x00, 0x00, 0x1F, 0x00, 0x1F, 0x00, 0x00],
        " ": [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
        ",": [0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x08],
        "g": [0x00, 0x00, 0x0F, 0x11, 0x0F, 0x01, 0x0E],
        "m": [0x00, 0x00, 0x1A, 0x15, 0x15, 0x11, 0x11],
        "l": [0x0C, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E],
        "R": [0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11],
        "G": [0x0E, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0F],
        "M": [0x11, 0x1B, 0x15, 0x15, 0x11, 0x11, 0x11],
        "L": [0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F],
        ":": [0x00, 0x0C, 0x0C, 0x00, 0x0C, 0x0C, 0x00],
        "/": [0x01, 0x02, 0x02, 0x04, 0x08, 0x08, 0x10],
    }


def main() -> None:
    """Train on Pendulum family and record videos."""
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    os.environ["SDL_VIDEODRIVER"] = "dummy"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -- Configuration --
    # Use 2048 steps per env (spanning ~10 episodes) for enough PPO data.
    # Truncation bootstrapping in the rollout worker handles episode
    # boundaries correctly.
    n_envs = 6
    n_transitions_discovery = 2000
    n_train_iterations = 200
    n_steps_per_env = 1024
    video_envs = [0, 1, 2, 3, 4, 5]  # Record all 6
    output_dir = "experiments/outputs"
    os.makedirs(output_dir, exist_ok=True)

    # -- 1. Create environment family --
    print("=" * 60)
    print("PENDULUM CIRC-RL TRAINING + VIDEO")
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

    print("\nEnvironment parameters:")
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
    print(f"Edges: {sorted(graph.edges)}")

    # -- 3. Feature selection --
    print("\n--- Phase 2: Feature Selection ---")
    selector = InvFeatureSelector(epsilon=0.15, min_ate=0.01)
    fs_result = selector.select(dataset, graph, state_names)
    print(f"Selected: {fs_result.selected_features}")
    print(f"Rejected: {fs_result.rejected_features}")

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
        irm_weight=0.0,
        worst_case_temperature=1.0,
        worst_case_variance_weight=0.0,
        entropy_coef=0.01,
    )

    trainer = CIRCTrainer(
        policy=policy,
        env_family=env_family,
        config=config,
    )
    trainer.to(device)

    metrics_history = trainer.run()

    # Print training summary
    if metrics_history:
        first = metrics_history[0]
        last = metrics_history[-1]
        print(f"\nTraining complete ({len(metrics_history)} iterations):")
        print(f"  First: mean_return={first.mean_return:.1f}, "
              f"worst={first.worst_env_return:.1f}")
        print(f"  Last:  mean_return={last.mean_return:.1f}, "
              f"worst={last.worst_env_return:.1f}")

    # Save checkpoint for re-recording without retraining
    ckpt_path = os.path.join(output_dir, "pendulum_policy.pt")
    torch.save(policy.state_dict(), ckpt_path)
    print(f"  Checkpoint saved to {ckpt_path}")

    # -- 5. Record videos --
    print("\n--- Phase 4: Video Recording ---")
    policy.eval()
    policy.to(torch.device("cpu"))

    all_frames: list[list[np.ndarray]] = []
    labels: list[str] = []

    for env_idx in video_envs:
        params = env_family.get_env_params(env_idx)
        label = (
            f"g={params['g']:.1f} m={params['m']:.1f} "
            f"l={params['l']:.1f}"
        )
        print(f"  Recording env {env_idx}: {label}...", end=" ")

        frames, total_reward = _record_episode(
            env_family, env_idx, policy,
            max_steps=500, device=torch.device("cpu"),
        )
        print(f"R={total_reward:.1f} ({len(frames)} frames)")

        # Add reward to label
        labels.append(f"{label} R={total_reward:.0f}")
        all_frames.append(frames)

    # Save grid video (20fps for comfortable viewing -- 500 frames = 25s)
    video_path = os.path.join(output_dir, "pendulum_circ_rl.mp4")
    _make_grid_video(all_frames, labels, video_path, fps=20)

    # Also save individual videos
    for i, env_idx in enumerate(video_envs):
        import imageio.v3 as iio

        individual_path = os.path.join(
            output_dir, f"pendulum_env{env_idx}.mp4"
        )
        iio.imwrite(
            individual_path,
            np.stack(all_frames[i]),
            fps=20,
            codec="libx264",
        )

    print(f"\nVideos saved to {output_dir}/")
    print(f"  Grid: pendulum_circ_rl.mp4")
    for env_idx in video_envs:
        print(f"  Individual: pendulum_env{env_idx}.mp4")


if __name__ == "__main__":
    main()
