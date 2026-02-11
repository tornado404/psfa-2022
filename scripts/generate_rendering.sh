#!/bin/bash
set -e

# Arguments
SPEAKER_ID=$1
AUDIO_PATH=$2
TASK_ID=$3

echo "==============================================="
echo "[WSL] generate_rendering.sh started"
echo "[WSL] Received Arguments:"
echo "  - Speaker ID: $SPEAKER_ID"
echo "  - Audio Path: $AUDIO_PATH"
echo "  - Task ID:    $TASK_ID"
echo "==============================================="

# 1. Define Project Root
PROJECT_ROOT="/mnt/d/Downloads/psfa-2022-main"

# 2. Check Project Root
if [ ! -d "$PROJECT_ROOT" ]; then
    echo "[WSL] Error: Project root directory not found at: $PROJECT_ROOT"
    exit 10
fi

cd "$PROJECT_ROOT" || { echo "[WSL] Error: Failed to cd into $PROJECT_ROOT"; exit 11; }
echo "[WSL] Changed working directory to: $(pwd)"

# 3. Activate Conda Environment
CONDA_BASE=""
if [ -d "$HOME/miniconda3" ]; then
    CONDA_BASE="$HOME/miniconda3"
elif [ -d "$HOME/anaconda3" ]; then
    CONDA_BASE="$HOME/anaconda3"
elif [ -d "/home/sdk/miniconda3" ]; then
    CONDA_BASE="/home/sdk/miniconda3"
elif [ -d "/home/sdk/anaconda3" ]; then
    CONDA_BASE="/home/sdk/anaconda3"
else
    CONDA_EXE=$(which conda 2>/dev/null)
    if [ -n "$CONDA_EXE" ]; then
        CONDA_BIN_DIR=$(dirname "$CONDA_EXE")
        CONDA_BASE=$(dirname "$CONDA_BIN_DIR")
    fi
fi

if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    echo "[WSL] Found conda at $CONDA_BASE"
    source "$CONDA_BASE/etc/profile.d/conda.sh"
else
    echo "[WSL] Warning: Could not find conda.sh. Hoping conda is in PATH and initialized."
fi

echo "[WSL] Activating conda environment 'psfa'..."
conda activate psfa || { echo "[WSL] Error: Failed to activate environment 'psfa'"; exit 13; }

source scripts/functions.sh

# Disable wgpu in docker WGPU FILAMENT VULKAN SKIP
export MIKU_RENDER_BACKEND=WGPU

# Speaker & exp.
speaker=$SPEAKER_ID
exp=decmp
exp_name=animnet-decmp

# Locate the generated output directory
# We expect it to be under runs/anime/$speaker/$exp_name/generated/
# It should contain $TASK_ID/input/dump-offsets-final.npy
GENERATED_BASE="runs/anime/$speaker/$exp_name/generated"

echo "[WSL] Searching for generated files in $GENERATED_BASE..."
# Find the directory that ends with the Task ID
# We use -name instead of -path to avoid issues with brackets [] in the path
PARENT_DIR=$(find "$GENERATED_BASE" -type d -name "*$TASK_ID" -print -quit)

if [ -z "$PARENT_DIR" ]; then
    echo "[WSL] Error: Could not find generated files for Task ID: $TASK_ID"
    echo "[WSL] Search path: $GENERATED_BASE"
    exit 14
fi

TARGET_DIR="$PARENT_DIR/input"
echo "[WSL] Found target directory: $TARGET_DIR"
# Make it absolute path just in case
TARGET_DIR_ABS="$PROJECT_ROOT/$TARGET_DIR"

OFFSETS_FILE="$TARGET_DIR_ABS/dump-offsets-final.npy"
AUDIO_FILE="$TARGET_DIR_ABS/audio.wav"

if [ ! -f "$OFFSETS_FILE" ]; then
    echo "[WSL] Error: Offsets file not found: $OFFSETS_FILE"
    exit 15
fi

# Neural Rendering.
# Please change the arguments.
echo "[WSL] Running Neural Renderer..."
python -m scripts.neural_render \
  --nr_ckpt="$PROJECT_ROOT/runs/neural_renderer/$speaker/checkpoints/epoch_60.pth" \
  --out_path="$TARGET_DIR_ABS/output.mp4" \
  --audio_path="$AUDIO_FILE" \
  --offsets_npy="$OFFSETS_FILE" \
  --iden_path="$PROJECT_ROOT/assets/datasets/talk_video/celebtalk/data/$speaker/fitted/identity/identity.obj" \
  --reenact_video="$PROJECT_ROOT/assets/datasets/talk_video/celebtalk/data/$speaker/avoffset_corrected/vld-000-fps25.mp4" \
  --reenact_coeff="$PROJECT_ROOT/assets/datasets/talk_video/celebtalk/data/$speaker/fitted/vld-000" \
  --reenact_static_frame=0 \
;

echo "\n"
VIDEO_PATH="$TARGET_DIR_ABS/output.mp4"
echo "[WSL] Done! Video generated at: $VIDEO_PATH"

VIDEO_PATH_OUT=""
if [ -f "$VIDEO_PATH" ]; then
  VIDEO_PATH_OUT="$VIDEO_PATH"
fi

OFFSETS_PATH_OUT=""
if [ -f "$OFFSETS_FILE" ]; then
  OFFSETS_PATH_OUT="$OFFSETS_FILE"
fi

AUDIO_PATH_OUT=""
if [ -f "$AUDIO_FILE" ]; then
  AUDIO_PATH_OUT="$AUDIO_FILE"
fi

export RES_OUTPUT_DIR="$TARGET_DIR_ABS"
export RES_VIDEO_PATH="$VIDEO_PATH_OUT"
export RES_OFFSETS_PATH="$OFFSETS_PATH_OUT"
export RES_AUDIO_PATH="$AUDIO_PATH_OUT"

python -c "import json, os; print('__RESULT__ ' + json.dumps({
  'output_dir': os.environ.get('RES_OUTPUT_DIR'),
  'key_files': {
    'video': os.environ.get('RES_VIDEO_PATH'),
    'offsets': os.environ.get('RES_OFFSETS_PATH'),
  }
}))"

