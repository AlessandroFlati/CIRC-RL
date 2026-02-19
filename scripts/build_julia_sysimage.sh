#!/usr/bin/env bash
#
# Build a Julia sysimage with SymbolicRegression.jl precompiled.
# This eliminates the ~4-minute Julia precompilation on first PySR import.
#
# Usage:
#   ./scripts/build_julia_sysimage.sh
#
# After building, set the environment variable before running PySR:
#   export JULIA_SYSIMAGE_PATH="$(pwd)/.julia_sysimage.so"
#
# Or add to your shell profile for persistence.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "=== Julia Sysimage Builder for CIRC-RL ==="
echo ""

# Check Julia is available
if ! command -v julia &> /dev/null; then
    echo "ERROR: Julia not found. Install Julia >= 1.9 first."
    echo "  https://julialang.org/downloads/"
    exit 1
fi

JULIA_VERSION=$(julia --version 2>&1 | grep -oP '\d+\.\d+')
echo "Julia version: $(julia --version)"

# Check minimum version (1.9+)
MAJOR=$(echo "$JULIA_VERSION" | cut -d. -f1)
MINOR=$(echo "$JULIA_VERSION" | cut -d. -f2)
if [ "$MAJOR" -lt 1 ] || ([ "$MAJOR" -eq 1 ] && [ "$MINOR" -lt 9 ]); then
    echo "ERROR: Julia >= 1.9 required for sysimage building (found $JULIA_VERSION)"
    exit 1
fi

# Install PackageCompiler if needed
echo ""
echo "Installing/updating Julia packages..."
julia -e 'using Pkg; Pkg.add(["PackageCompiler", "SymbolicRegression"])'

# Build the sysimage
echo ""
echo "Building sysimage (this takes 5-15 minutes)..."
julia "${SCRIPT_DIR}/build_julia_sysimage.jl"

SYSIMAGE_PATH="${PROJECT_DIR}/.julia_sysimage.so"

if [ -f "$SYSIMAGE_PATH" ]; then
    echo ""
    echo "=== Build successful ==="
    echo ""
    echo "Sysimage: ${SYSIMAGE_PATH}"
    SIZE=$(du -h "$SYSIMAGE_PATH" | cut -f1)
    echo "Size: ${SIZE}"
    echo ""
    echo "To use, set before running experiments:"
    echo "  export JULIA_SYSIMAGE_PATH=\"${SYSIMAGE_PATH}\""
    echo ""
    echo "Or add to .bashrc / .zshrc for persistence."
else
    echo ""
    echo "ERROR: Sysimage was not created. Check output above for errors."
    exit 1
fi
