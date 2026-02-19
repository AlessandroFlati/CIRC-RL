"""Record best-checkpoint Pendulum video on 50 diverse environments.

Tests generalization of the trained policy across a wide range of
physics parameters (g, m, l).

Usage::

    uv run python experiments/pendulum_record_50.py
"""

from __future__ import annotations

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
    max_steps: int = 400,
    env_param_names: list[str] | None = None,
    fixed_params: dict[str, float] | None = None,
) -> tuple[list[np.ndarray], float]:
    """Record one episode with rgb_array frames."""
    import gymnasium as gym

    params = env_family.get_env_params(env_idx)
    env = gym.make(
        "Pendulum-v1",
        render_mode="rgb_array",
        max_episode_steps=max_steps,
    )
    unwrapped = env.unwrapped
    if fixed_params:
        for attr, value in fixed_params.items():
            setattr(unwrapped, attr, value)
    for attr, value in params.items():
        setattr(unwrapped, attr, value)

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
        # Downscale frames for the large grid (500x500 -> 200x200)
        frame = frame[::2, ::2, :].copy()  # Simple 2x downsample
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


def _add_text_to_frame(frame: np.ndarray, text: str) -> None:
    """Burn text label into top-left corner of an RGB frame."""
    bar_height = 16
    frame[:bar_height, :] = (
        frame[:bar_height, :].astype(np.float32) * 0.3
    ).astype(np.uint8)

    _FONT = _get_bitmap_font()
    x_offset = 4
    y_offset = 2
    scale = 1

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
                    if y < frame.shape[0] and x < frame.shape[1]:
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
    }


def _make_grid_video(
    all_frames: list[list[np.ndarray]],
    labels: list[str],
    output_path: str,
    fps: int = 20,
    n_cols: int = 10,
) -> None:
    """Combine multiple episode frame lists into a grid video."""
    import imageio.v3 as iio

    max_len = max(len(f) for f in all_frames)
    for frames in all_frames:
        while len(frames) < max_len:
            frames.append(frames[-1].copy())

    n = len(all_frames)
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


def main() -> None:
    """Load checkpoint and record 50-env grid video."""
    import argparse

    parser = argparse.ArgumentParser(description="Record 50-env Pendulum video")
    parser.add_argument(
        "checkpoint", nargs="?", default=None,
        help="Path to checkpoint (.pt). Default: best checkpoint.",
    )
    parser.add_argument(
        "--tag", default="",
        help="Tag appended to output filename (e.g. 'iter025').",
    )
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="INFO")
    os.environ["SDL_VIDEODRIVER"] = "dummy"

    output_dir = "experiments/outputs"
    checkpoint_path = args.checkpoint or os.path.join(
        output_dir, "pendulum_policy_best.pt",
    )

    n_envs = 50
    env_param_names = ["g", "m", "l"]
    max_torque = 100.0  # effectively unlimited torque

    # Create 50-env family with a different seed for novel params
    env_family = EnvironmentFamily.from_gymnasium(
        base_env="Pendulum-v1",
        param_distributions={
            "g": (5.0, 15.0),      # wider range than training (7-13)
            "m": (0.3, 3.0),       # wider range than training (0.5-2.0)
            "l": (0.3, 2.0),       # wider range than training (0.5-1.5)
        },
        n_envs=n_envs,
        seed=123,  # different seed = different params
        fixed_params={"max_torque": max_torque},
    )

    action_low = np.array([-max_torque], dtype=np.float32)
    action_high = np.array([max_torque], dtype=np.float32)

    # Compute dynamics reference scale from the TRAINING env family
    # (same params as training, not the 50-env test family)
    train_env_family = EnvironmentFamily.from_gymnasium(
        base_env="Pendulum-v1",
        param_distributions={
            "g": (7.0, 13.0),
            "m": (0.5, 2.0),
            "l": (0.5, 1.5),
        },
        n_envs=16,
        seed=42,
        fixed_params={"max_torque": max_torque},
    )
    collector = DataCollector(train_env_family, include_env_params=True)
    discovery_data = collector.collect(n_transitions_per_env=2000, seed=42)
    state_names = [f"s{i}" for i in range(3)]
    ta_result = TransitionAnalyzer().analyze(discovery_data, state_names, 1)
    print(f"Reference dynamics scale: {ta_result.reference_scale:.4f}")

    policy = CausalPolicy(
        full_state_dim=3,
        feature_mask=np.ones(3, dtype=bool),
        action_dim=1,
        hidden_dims=(64, 64),
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

    # Record all 50 environments
    all_frames: list[list[np.ndarray]] = []
    labels: list[str] = []
    returns: list[float] = []

    for env_idx in range(n_envs):
        params = env_family.get_env_params(env_idx)
        label = (
            f"g={params['g']:.1f} m={params['m']:.1f} "
            f"l={params['l']:.1f}"
        )
        print(f"  [{env_idx+1:2d}/50] {label}...", end=" ", flush=True)

        frames, total_reward = _record_episode(
            env_family, env_idx, policy,
            max_steps=400,
            env_param_names=env_param_names,
            fixed_params={"max_torque": max_torque},
        )
        print(f"R={total_reward:.0f}")

        labels.append(f"R={total_reward:.0f}")
        all_frames.append(frames)
        returns.append(total_reward)

    # Summary stats
    returns_arr = np.array(returns)
    print(f"\n--- Summary across 50 environments ---")
    print(f"  Mean return:  {returns_arr.mean():.1f}")
    print(f"  Worst return: {returns_arr.min():.1f}")
    print(f"  Best return:  {returns_arr.max():.1f}")
    print(f"  Std return:   {returns_arr.std():.1f}")
    print(f"  Median:       {np.median(returns_arr):.1f}")

    # Save grid video (10 columns x 5 rows)
    tag_suffix = f"_{args.tag}" if args.tag else "_best"
    video_path = os.path.join(output_dir, f"pendulum_50env{tag_suffix}.mp4")
    _make_grid_video(all_frames, labels, video_path, fps=20, n_cols=10)
    print(f"\nGrid video saved: {video_path}")
    print(f"  {len(all_frames[0])} frames at 20fps = {len(all_frames[0])/20:.0f}s")


if __name__ == "__main__":
    main()
