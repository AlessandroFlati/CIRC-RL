#!/bin/bash
# Record 50-env videos for all numbered checkpoints.
# Usage: bash experiments/record_all_checkpoints.sh

set -e
export PATH="$HOME/.local/bin:/usr/bin:/bin:$PATH"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

source .venv/bin/activate

OUTPUT_DIR="experiments/outputs"

for ckpt in "$OUTPUT_DIR"/pendulum_policy_iter*.pt; do
    if [ ! -f "$ckpt" ]; then
        echo "No checkpoints found in $OUTPUT_DIR"
        exit 1
    fi

    # Extract tag from filename: pendulum_policy_iter025.pt -> iter025
    tag=$(basename "$ckpt" .pt | sed 's/pendulum_policy_//')
    video="$OUTPUT_DIR/pendulum_50env_${tag}.mp4"

    if [ -f "$video" ]; then
        echo "SKIP $tag (video already exists)"
        continue
    fi

    echo "=== Recording $tag from $ckpt ==="
    python experiments/pendulum_record_50.py "$ckpt" --tag "$tag"
    echo ""
done

# Also record best checkpoint
best_ckpt="$OUTPUT_DIR/pendulum_policy_best.pt"
best_video="$OUTPUT_DIR/pendulum_50env_best.mp4"
if [ -f "$best_ckpt" ] && [ ! -f "$best_video" ]; then
    echo "=== Recording best checkpoint ==="
    python experiments/pendulum_record_50.py "$best_ckpt" --tag best
fi

echo "Done. All videos saved to $OUTPUT_DIR/"
