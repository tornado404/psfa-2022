#!/bin/bash
set -e

# Arguments received from the caller (e.g., Windows Python script)
SPEAKER_ID=$1
AUDIO_PATH=$2
TASK_ID=$3

echo "==============================================="
echo "[WSL] test_gpu.sh started"
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
# Try to find conda base path
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
    # Try to find via which conda
    CONDA_EXE=$(which conda 2>/dev/null)
    if [ -n "$CONDA_EXE" ]; then
        # /path/to/condabin/conda -> /path/to
        # usually .../miniconda3/condabin/conda or .../miniconda3/bin/conda
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

# 4. Run Python Task/Check
echo "[WSL] Running Python GPU/Environment Check..."

# Using python to verify environment and GPU
# We can pass the arguments to the python script if needed
python -c "
import sys
import torch
print(f'Python Version: {sys.version}')
print(f'Torch Version: {torch.__version__}')
cuda_available = torch.cuda.is_available()
print(f'CUDA Available: {cuda_available}')

if cuda_available:
    print(f'CUDA Device: {torch.cuda.get_device_name(0)}')
else:
    print('Warning: CUDA is NOT available.')

# Simulate processing based on arguments
print(f'Processing Task: {sys.argv[1]}')
" "$TASK_ID"

RET_CODE=$?

if [ $RET_CODE -eq 0 ]; then
    echo "[WSL] Python script executed successfully."
    exit 0
else
    echo "[WSL] Python script failed with return code $RET_CODE"
    exit $RET_CODE
fi
