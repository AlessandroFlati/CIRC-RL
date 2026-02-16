"""Record Pendulum videos from a saved checkpoint.

Loads a trained CausalPolicy checkpoint and records a grid video
showing the policy in all 6 physics variants.

Usage::

    uv run python experiments/pendulum_record.py [checkpoint_path] [--tag TAG]
    uv run python experiments/pendulum_record.py --best --tag best

If checkpoint_path is omitted, defaults to experiments/outputs/pendulum_policy.pt.
Use --best to load experiments/outputs/pendulum_policy_best.pt instead.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from loguru import logger

from circ_rl.environments.data_collector import DataCollector
from circ_rl.environments.env_family import EnvironmentFamily
from circ_rl.feature_selection.transition_analyzer import TransitionAnalyzer
from circ_rl.policy.causal_policy import CausalPolicy


def _record_episode(
    env_family: EnvironmentFamily,
    env_idx: int,
    policy: CausalPolicy,
    max_steps: int = 600,
    env_param_names: list[str] | None = None,
) -> tuple[list[np.ndarray], float]:
    """Record one episode with rgb_array frames.

    :param env_family: The environment family.
    :param env_idx: Which environment to record.
    :param policy: Trained policy.
    :param max_steps: Maximum episode length.
    :param env_param_names: When set, pass these env params as context.
    :returns: (frames, total_reward) tuple.
    """
    import gymnasium as gym

    params = env_family.get_env_params(env_idx)
    env = gym.make(
        "Pendulum-v1",
        render_mode="rgb_array",
        max_episode_steps=max_steps,
    )
    unwrapped = env.unwrapped
    for attr, value in params.items():
        setattr(unwrapped, attr, value)

    # Build context tensor if policy uses env params
    context_tensor: torch.Tensor | None = None
    if env_param_names and policy.context_dim > 0:
        context_tensor = torch.tensor(
            [params[name] for name in env_param_names],
            dtype=torch.float32,
        )

    obs, _ = env.reset(seed=42)
    frames: list[np.ndarray] = []
    total_reward = 0.0

    for _ in range(max_steps):
        frame = env.render()
        assert isinstance(frame, np.ndarray)
        frames.append(frame)

        state_tensor = torch.tensor(
            obs, dtype=torch.float32
        ).unsqueeze(0)

        action_np = policy.get_action(
            state_tensor, deterministic=True, context=context_tensor,
        )
        assert isinstance(action_np, np.ndarray)

        obs, reward, terminated, truncated, _ = env.step(action_np)
        total_reward += float(reward)

        if terminated:
            break

    env.close()
    return frames, total_reward


def _make_grid_video(
    all_frames: list[list[np.ndarray]],
    labels: list[str],
    output_path: str,
    fps: int = 20,
) -> None:
    """Combine multiple episode frame lists into a 3x2 grid video.

    :param all_frames: List of frame lists (one per environment).
    :param labels: Label for each environment.
    :param output_path: Path to write the output mp4 file.
    :param fps: Frames per second.
    """
    import imageio.v3 as iio

    max_len = max(len(f) for f in all_frames)
    for frames in all_frames:
        while len(frames) < max_len:
            frames.append(frames[-1].copy())

    n = len(all_frames)
    n_cols = min(n, 3)
    n_rows = (n + n_cols - 1) // n_cols

    grid_frames = []
    for t in range(max_len):
        row_images = []
        for r in range(n_rows):
            col_images = []
            for c in range(n_cols):
                idx = r * n_cols + c
                if idx < n:
                    frame = all_frames[idx][t].copy()
                    _add_text_to_frame(frame, labels[idx])
                    col_images.append(frame)
                else:
                    col_images.append(np.zeros_like(all_frames[0][0]))
            row_images.append(np.hstack(col_images))
        grid_frames.append(np.vstack(row_images))

    iio.imwrite(
        output_path,
        np.stack(grid_frames),
        fps=fps,
        codec="libx264",
    )


def _add_text_to_frame(frame: np.ndarray, text: str) -> None:
    """Burn text label into top-left corner of an RGB frame."""
    bar_height = 30
    frame[:bar_height, :] = (
        frame[:bar_height, :].astype(np.float32) * 0.3
    ).astype(np.uint8)

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
    """Return a minimal 5x7 bitmap font for common characters."""
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
        "i": [0x04, 0x00, 0x0C, 0x04, 0x04, 0x04, 0x0E],
        "t": [0x04, 0x04, 0x0E, 0x04, 0x04, 0x04, 0x03],
        "e": [0x00, 0x00, 0x0E, 0x11, 0x1F, 0x10, 0x0E],
        "r": [0x00, 0x00, 0x16, 0x19, 0x10, 0x10, 0x10],
    }


def main() -> None:
    """Load checkpoint and record grid video."""
    parser = argparse.ArgumentParser(
        description="Record Pendulum videos from checkpoint"
    )
    parser.add_argument(
        "checkpoint",
        nargs="?",
        default=None,
        help="Path to policy checkpoint (.pt file)",
    )
    parser.add_argument(
        "--best",
        action="store_true",
        help="Load best worst-case checkpoint (pendulum_policy_best.pt)",
    )
    parser.add_argument(
        "--tag",
        default="",
        help="Tag to append to output filename (e.g. 'iter100')",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help="Video frames per second (default: 20)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=600,
        help="Simulation steps per environment (default: 600, = 30s at 20fps)",
    )
    args = parser.parse_args()

    # Resolve checkpoint path
    if args.checkpoint is not None:
        checkpoint_path = args.checkpoint
    elif args.best:
        checkpoint_path = "experiments/outputs/pendulum_policy_best.pt"
    else:
        checkpoint_path = "experiments/outputs/pendulum_policy.pt"

    logger.remove()
    logger.add(sys.stderr, level="INFO")
    os.environ["SDL_VIDEODRIVER"] = "dummy"

    output_dir = "experiments/outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Recreate the same env family (same seed = same params)
    env_family = EnvironmentFamily.from_gymnasium(
        base_env="Pendulum-v1",
        param_distributions={
            "g": (7.0, 13.0),
            "m": (0.5, 2.0),
            "l": (0.5, 1.5),
        },
        n_envs=6,
        seed=42,
    )

    state_dim = 3   # Pendulum obs: [cos(theta), sin(theta), theta_dot]
    action_dim = 1   # Pendulum action: torque
    env_param_names = ["g", "m", "l"]

    import gymnasium as gym
    action_space = gym.make("Pendulum-v1").action_space
    action_low = action_space.low   # type: ignore[union-attr]
    action_high = action_space.high  # type: ignore[union-attr]

    # Compute dynamics reference scale (same env family as training)
    collector = DataCollector(env_family, include_env_params=True)
    discovery_data = collector.collect(n_transitions_per_env=2000, seed=42)
    state_names = [f"s{i}" for i in range(state_dim)]
    ta_result = TransitionAnalyzer().analyze(discovery_data, state_names, action_dim)

    policy = CausalPolicy(
        full_state_dim=state_dim,
        feature_mask=np.ones(state_dim, dtype=bool),
        action_dim=action_dim,
        hidden_dims=(128, 128),
        continuous=True,
        action_low=action_low,
        action_high=action_high,
        context_dim=len(env_param_names),
        use_dynamics_normalization=True,
        dynamics_reference_scale=ta_result.reference_scale,
    )

    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    policy.load_state_dict(state_dict)
    policy.eval()

    # Record all 6 environments
    all_frames: list[list[np.ndarray]] = []
    labels: list[str] = []

    for env_idx in range(6):
        params = env_family.get_env_params(env_idx)
        label = (
            f"g={params['g']:.1f} m={params['m']:.1f} "
            f"l={params['l']:.1f}"
        )
        print(f"  Recording env {env_idx}: {label}...", end=" ", flush=True)

        frames, total_reward = _record_episode(
            env_family, env_idx, policy,
            max_steps=args.steps,
            env_param_names=env_param_names,
        )
        print(f"R={total_reward:.1f} ({len(frames)} frames)")

        labels.append(f"{label} R={total_reward:.0f}")
        all_frames.append(frames)

    tag = f"_{args.tag}" if args.tag else ""
    video_path = os.path.join(output_dir, f"pendulum_circ_rl{tag}.mp4")
    _make_grid_video(all_frames, labels, video_path, fps=args.fps)
    print(f"\nGrid video saved: {video_path}")
    duration = len(all_frames[0]) / args.fps
    print(f"  {len(all_frames[0])} frames at {args.fps}fps = {duration:.0f}s")


if __name__ == "__main__":
    main()
