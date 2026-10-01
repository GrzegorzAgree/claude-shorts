#!/usr/bin/env bash
# Setup dependencies for claude-shorts
# Usage: bash setup.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== claude-shorts dependency setup ==="
echo ""

# --- Python Virtual Environment ---

# Reuse the shared venv from claude-video if it exists
if [ -d "$HOME/.video-skill" ]; then
    VENV="$HOME/.video-skill"
    echo "[Python] Found shared venv at ~/.video-skill"
else
    VENV="$HOME/.shorts-skill"
    if [ -d "$VENV" ]; then
        echo "[Python] Found venv at ~/.shorts-skill"
    else
        echo "[Python] Creating venv at ~/.shorts-skill..."
        python3 -m venv "$VENV"
    fi
    "$VENV/bin/pip" install --quiet --upgrade pip
fi

# PyTorch is only used to detect CUDA (transcribe.py falls back to CPU without it)
if command -v nvidia-smi &>/dev/null; then
    echo "[Python] NVIDIA GPU detected — installing PyTorch with CUDA..."
    "$VENV/bin/pip" install --quiet torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu128
else
    echo "[Python] No NVIDIA GPU — skipping PyTorch (CPU transcription)"
fi

echo "[Python] Installing dependencies from requirements.txt..."
"$VENV/bin/pip" install --quiet -r "$SCRIPT_DIR/requirements.txt"

echo "[Python] Venv ready: $VENV"
echo ""

# --- Node.js / Remotion ---

echo "[Node] Setting up Remotion project..."

if [ ! -f "$SCRIPT_DIR/remotion/package.json" ]; then
    echo "[Node] ERROR: remotion/package.json not found"
    echo "       Run this script from the claude-shorts project root"
    exit 1
fi

cd "$SCRIPT_DIR/remotion"

if [ -d "node_modules" ]; then
    echo "[Node] node_modules exists — running npm install to check for updates..."
else
    echo "[Node] Installing Remotion dependencies..."
fi

npm install --silent 2>&1 | tail -1 || npm install

echo "[Node] Remotion ready"
echo ""

# --- System Dependencies ---

MISSING=()

if ! command -v ffmpeg &>/dev/null; then
    MISSING+=("ffmpeg")
fi

if ! command -v jq &>/dev/null; then
    MISSING+=("jq")
fi

if [ ${#MISSING[@]} -gt 0 ]; then
    echo "[System] Missing packages: ${MISSING[*]}"
    echo "[System] Install with: sudo apt install ${MISSING[*]}"
else
    echo "[System] All system dependencies present (ffmpeg, jq)"
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "Python venv: $VENV"
echo "Remotion:    $SCRIPT_DIR/remotion/"
echo ""
echo "To use: invoke /shorts in Claude Code"
